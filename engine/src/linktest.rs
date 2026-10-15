//! `trashcan link-test`: links between two machines under the traffic the models make, every byte checked.
//!
//! A server (`link-test serve`) answers sessions, one thread each; a client (`link-test run --connect ADDR,...`)
//! opens lanes to it (one address per cable, as tensor parallelism does), sends the test, and both run it:
//! - `pingpong`: SIZE bytes there and back, COUNT times (latency);
//! - `exchange`: both send SIZE bytes at once over every lane, COUNT times (tensor-parallel all-reduce steps);
//! - `bulk`: SIZE bytes one way and a short answer, COUNT times, then the other way (throughput);
//! - `chain`: the Flash-Next chain's pattern per decode token, 40 KiB on, 10 KiB and a result back, with a 5 MiB
//!   prompt chunk every 16 tokens (latency);
//! - `soak`: all of them in turn for SECONDS, reporting every 10 s.
//!
//! Payloads are a seeded pattern XORed with a key per message, so a byte from the wrong place or message shows.
//! The client logs when its first lane changes path (stream/tcp) and every round trip that took over 20 ms.
//!
//! Fault injection (client side, for tests only): `--drop N` / `--corrupt N` lose or damage every Nth stream frame
//! the client sends; `--overdrive N` lets it send N frames past its credits, overrunning the other end's ring.
//!
//! `link-test churn` runs sessions back to back, each a client process of its own, with normal exits, killed
//! clients and server crashes mixed (`--server-exits`; run the server in a loop that restarts it): every session's
//! stream must come up within `--within-ms` of its link's start, with no failed epoch. A server started with
//! `--load-ms` (or `--freeze-ms`) is busy (or stopped) from each session's start, as a node loading its stage is.
use crate::link::{self, Accepted, Found, Frames, Lanes, Link, Options, Streams, Sysfs, Tuning, Wait};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    io::{self, IoSlice},
    net::{SocketAddr, TcpListener},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant},
};

#[derive(clap::Subcommand)]
pub enum Cmd {
    /// Answer link tests, a thread per session.
    Serve {
        #[arg(long, default_value = "0.0.0.0:7075")]
        listen: String,
        #[command(flatten)]
        load: Load,
    },
    /// Run a test against `link-test serve` and print one JSON line (soak: one every 10 s and a summary), after a
    /// line with `up_ms` as soon as the stream is up.
    Run(Run),
    /// Sessions back to back against `link-test serve`, one process each, with kills and server exits mixed; one
    /// JSON line per session and a summary. Fails unless every stream came up in time with no failed epoch.
    Churn(Churn),
    /// CRC32C throughput on this CPU.
    Crc {
        #[arg(long, default_value_t = 40960)]
        size: usize,
        #[arg(long, default_value_t = 20000)]
        iters: usize,
    },
}

/// A busy server (`link-test serve`), from the start of each session, while its links come up: as a node loading its
/// stage (GPU uploads of several GiB) while the driver's link starts.
#[derive(clap::Args, Clone, Copy)]
pub struct Load {
    /// Load this machine for this many milliseconds: threads that write `--load-mb` MiB each over and over (CPU and
    /// memory).
    #[arg(long, default_value_t = 0)]
    load_ms: u64,
    /// Load threads (0: one per CPU).
    #[arg(long, default_value_t = 0)]
    load_threads: usize,
    #[arg(long, default_value_t = 256)]
    load_mb: usize,
    /// Stop this whole process (its links' threads too) for this many milliseconds, from a random 0-20 ms after the
    /// session's start: the worst case of a busy machine, saying nothing at all.
    #[arg(long, default_value_t = 0)]
    freeze_ms: u64,
}

impl Load {
    fn start(&self, seed: u64) {
        if self.load_ms > 0 {
            let until = Instant::now() + Duration::from_millis(self.load_ms);
            let n = if self.load_threads > 0 { self.load_threads } else { std::thread::available_parallelism().map_or(4, |n| n.get()) };
            for t in 0..n {
                let mb = self.load_mb;
                std::thread::spawn(move || {
                    let mut buf = vec![0u8; mb << 20];
                    let mut k = t as u8;
                    while Instant::now() < until {
                        buf.fill(k);
                        k = k.wrapping_add(1);
                        std::hint::black_box(&buf);
                    }
                });
            }
        }
        if self.freeze_ms > 0 {
            let pid = std::process::id();
            let delay = (splitmix(seed) % 21) as f64 / 1e3;
            let secs = self.freeze_ms as f64 / 1e3;
            let _ = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!("sleep {delay:.3}; kill -STOP {pid}; sleep {secs:.3}; kill -CONT {pid}"))
                .spawn();
        }
    }
}

#[derive(clap::Args)]
pub struct Run {
    /// One address per cable to the other machine, e.g. indite.tb0:7075.
    #[arg(long, value_delimiter = ',', required = true)]
    connect: Vec<String>,
    #[command(flatten)]
    test: Test,
    /// A name for the result line.
    #[arg(long, default_value = "")]
    label: String,
    /// Lose every Nth stream frame this side sends (0: none).
    #[arg(long, default_value_t = 0)]
    drop: u64,
    /// Damage every Nth stream frame this side sends (0: none).
    #[arg(long, default_value_t = 0)]
    corrupt: u64,
    /// Send this many frames past the credits.
    #[arg(long, default_value_t = 0)]
    overdrive: u32,
}

#[derive(clap::Args)]
pub struct Churn {
    /// The server's address on the cable, e.g. enlight.tb0:7075.
    #[arg(long, required = true)]
    connect: String,
    #[arg(long, default_value_t = 200)]
    sessions: usize,
    /// A session passes when its stream is up within this many milliseconds of its link's start.
    #[arg(long, default_value_t = 100.)]
    within_ms: f64,
    /// Session kinds in turn: n (normal exit), k (this side killed while on the stream), s (the server exits).
    #[arg(long, default_value = "nkns")]
    pattern: String,
    #[arg(long, default_value = "churn")]
    label: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Pingpong,
    Exchange,
    Bulk,
    Chain,
    Soak,
}

#[derive(Clone, Debug, Serialize, Deserialize, clap::Args)]
pub struct Test {
    #[arg(long, value_enum, default_value = "pingpong")]
    mode: Mode,
    /// Bytes per message (pingpong, exchange: multiple of 4; bulk).
    #[arg(long, default_value_t = 64)]
    size: usize,
    #[arg(long, default_value_t = 10000)]
    count: usize,
    /// Soak duration.
    #[arg(long, default_value_t = 60.)]
    seconds: f64,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// The server gives its stream up this many seconds into the test, as a failing device would (0: never).
    #[arg(long, default_value_t = 0.)]
    server_drops_stream: f64,
    /// The server process exits this many seconds into the test, as a crash would (0: never).
    #[arg(long, default_value_t = 0.)]
    #[serde(default)]
    server_exits: f64,
}

fn unix_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

const LARGEST: usize = 8 << 20;
const CHAIN_FWD: usize = 40 << 10;
const CHAIN_HIDDEN: usize = 10 << 10;
const CHAIN_RESULT: usize = 256;
const CHAIN_PROMPT: usize = 5 << 20;
const SLOW: Duration = Duration::from_millis(20);

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The seeded pattern every message is made from.
struct Pattern(Vec<u64>);

impl Pattern {
    fn new(seed: u64) -> Self {
        Self((0..LARGEST / 8 + 1).map(|i| splitmix(seed ^ (i as u64).wrapping_mul(0x2545_f491_4f6c_dd1d))).collect())
    }

    fn key(dir: u64, what: u64, i: u64) -> u64 {
        splitmix(dir << 60 ^ what << 48 ^ i) | 1
    }

    /// Message bytes for `key`.
    fn fill(&self, out: &mut [u8], key: u64) {
        let (words, tail) = out.split_at_mut(out.len() / 8 * 8);
        for (w, p) in words.chunks_exact_mut(8).zip(&self.0) {
            w.copy_from_slice(&(p ^ key).to_le_bytes());
        }
        let last = (self.0[words.len() / 8] ^ key).to_le_bytes();
        tail.copy_from_slice(&last[..tail.len()]);
    }

    /// The first byte of `got` that is not message `key`'s.
    fn check(&self, got: &[u8], key: u64) -> Option<usize> {
        let (words, tail) = got.split_at(got.len() / 8 * 8);
        for (i, (w, p)) in words.chunks_exact(8).zip(&self.0).enumerate() {
            if u64::from_le_bytes(w.try_into().unwrap()) != p ^ key {
                return Some(i * 8);
            }
        }
        let last = (self.0[words.len() / 8] ^ key).to_le_bytes();
        (tail != &last[..tail.len()]).then_some(words.len())
    }
}

/// Frames this side sends, lost or damaged on purpose.
struct Chaos {
    dev: Box<dyn Frames>,
    frames: u64,
    drop: u64,
    corrupt: u64,
}

impl Frames for Chaos {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.dev.read(buf)
    }

    fn writev(&mut self, bufs: &[IoSlice]) -> io::Result<usize> {
        let mut all = Vec::new();
        for b in bufs {
            all.extend_from_slice(b);
        }
        // The driver cuts a write into 4 KiB frames; fault them one by one, writing the others through.
        let mut done = 0;
        for frame in all.chunks(4096) {
            self.frames += 1;
            let n = self.frames;
            if self.drop > 0 && n.is_multiple_of(self.drop) {
                done += frame.len();
                continue;
            }
            let mut f = frame.to_vec();
            if self.corrupt > 0 && n.is_multiple_of(self.corrupt) {
                let at = (n as usize).wrapping_mul(2654435761) % f.len();
                f[at] ^= 0x40;
            }
            match self.dev.writev(&[IoSlice::new(&f)]) {
                Ok(_) => done += frame.len(),
                Err(e) if done == 0 => return Err(e),
                Err(_) => break,
            }
        }
        Ok(done)
    }

    fn fd(&self) -> Option<i32> {
        self.dev.fd()
    }
}

struct ChaosStreams {
    drop: u64,
    corrupt: u64,
}

impl Streams for ChaosStreams {
    fn find(&self, local: &SocketAddr) -> Result<Found, String> {
        Sysfs.find(local)
    }

    fn open(&self, found: &Found) -> io::Result<Box<dyn Frames>> {
        let dev = Sysfs.open(found)?;
        Ok(Box::new(Chaos { dev, frames: 0, drop: self.drop, corrupt: self.corrupt }))
    }
}

/// Latencies in microseconds.
#[derive(Default)]
struct Lat(Vec<f64>);

impl Lat {
    fn add(&mut self, d: Duration) {
        self.0.push(d.as_secs_f64() * 1e6);
    }

    fn summary(&mut self) -> Value {
        if self.0.is_empty() {
            return Value::Null;
        }
        self.0.sort_by(f64::total_cmp);
        let q = |p: f64| self.0[((self.0.len() - 1) as f64 * p).round() as usize];
        let mean = self.0.iter().sum::<f64>() / self.0.len() as f64;
        json!({"n": self.0.len(), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": q(1.0), "mean": mean})
    }
}

/// Path changes of a link and round trips over `SLOW`, in seconds since the start.
struct Events {
    start: Instant,
    log: Mutex<Vec<Value>>,
}

impl Events {
    fn note(&self, what: Value) {
        let t = self.start.elapsed().as_secs_f64();
        let mut v = what;
        v["t"] = json!((t * 1e3).round() / 1e3);
        v["unix"] = json!(unix_now());
        eprintln!("link-test: {v}");
        self.log.lock().unwrap().push(v);
    }
}

/// Watch the first lane's path every millisecond until `stop`; print `{"label", "up_ms"}` once the stream is up.
fn watch(link: Link, events: Arc<Events>, stop: Arc<AtomicBool>, label: String) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut path = link.path();
        let mut told = false;
        loop {
            if !told {
                if let Some(ms) = link.stats().first_up_ms {
                    println!("{}", json!({"label": label, "up_ms": ms}));
                    told = true;
                }
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
            let now = link.path();
            if now != path {
                events.note(json!({"path": now, "was": path}));
                path = now;
            }
        }
    })
}

/// One side of a test. `dir` is 0 on the client, 1 on the server; messages a side sends are keyed with its `dir`.
struct Side<'a> {
    lanes: &'a mut Lanes,
    pat: &'a Pattern,
    dir: u64,
    out: Vec<u8>,
    inp: Vec<u8>,
    checked: u64,
    slow: Vec<Value>,
    start: Instant,
}

impl Side<'_> {
    fn link(&self) -> &Link {
        &self.lanes.links()[0]
    }

    fn send(&mut self, what: u64, i: u64, len: usize) -> Result<()> {
        self.pat.fill(&mut self.out[..len], Pattern::key(self.dir, what, i));
        self.lanes.links()[0].send(&[&self.out[..len]])?;
        Ok(())
    }

    fn recv(&mut self, what: u64, i: u64, len: usize, wait: Wait) -> Result<()> {
        self.lanes.links()[0].read_exact(&mut self.inp[..len], wait)?;
        self.verify(what, i, len)
    }

    fn verify(&mut self, what: u64, i: u64, len: usize) -> Result<()> {
        if let Some(at) = self.pat.check(&self.inp[..len], Pattern::key(1 - self.dir, what, i)) {
            bail!("message {what}/{i}: byte {at} of {len} differs");
        }
        self.checked += len as u64;
        Ok(())
    }

    fn slow(&mut self, what: &str, i: u64, d: Duration) {
        if d > SLOW {
            let t = self.start.elapsed().as_secs_f64();
            let v = json!({"slow": what, "i": i, "ms": d.as_secs_f64() * 1e3, "t": (t * 1e3).round() / 1e3, "unix": unix_now(),
                "path": self.link().path()});
            eprintln!("link-test: {v}");
            self.slow.push(v);
        }
    }

    fn pingpong(&mut self, size: usize, count: usize, base: u64, lat: &mut Lat) -> Result<()> {
        for k in 0..count as u64 {
            let i = base + k;
            let t0 = Instant::now();
            if self.dir == 0 {
                self.send(1, i, size)?;
                self.recv(1, i, size, Wait::Stall(link::STALL))?;
            } else {
                self.recv(1, i, size, Wait::Stall(link::STALL))?;
                self.send(1, i, size)?;
            }
            let d = t0.elapsed();
            lat.add(d);
            self.slow("pingpong", i, d);
        }
        Ok(())
    }

    fn exchange(&mut self, size: usize, count: usize, base: u64, lat: &mut Lat) -> Result<()> {
        ensure!(size.is_multiple_of(4), "exchange sizes are multiples of 4");
        for k in 0..count as u64 {
            let i = base + k;
            self.pat.fill(&mut self.out[..size], Pattern::key(self.dir, 2, i));
            let t0 = Instant::now();
            let (out, inp) = (&self.out[..size], &mut self.inp[..size]);
            self.lanes.exchange_within(out, inp, Duration::from_secs(10))?;
            let d = t0.elapsed();
            lat.add(d);
            self.verify(2, i, size)?;
            self.slow("exchange", i, d);
        }
        Ok(())
    }

    /// SIZE bytes from the client, then from the server; the receiver answers each with 8 bytes.
    fn bulk(&mut self, size: usize, count: usize, base: u64) -> Result<(f64, f64)> {
        let mut rates = [0f64; 2];
        for sender in 0..2u64 {
            let t0 = Instant::now();
            for k in 0..count as u64 {
                let i = base + k;
                if self.dir == sender {
                    self.send(3 + sender, i, size)?;
                    self.recv(5, i, 8, Wait::Stall(link::STALL))?;
                } else {
                    self.recv(3 + sender, i, size, Wait::Stall(link::STALL))?;
                    self.send(5, i, 8)?;
                }
            }
            rates[sender as usize] = (size * count) as f64 / t0.elapsed().as_secs_f64() / 1e6;
        }
        Ok((rates[0], rates[1]))
    }

    /// Decode tokens of the chain: the residual on, the hidden state and a result back; a prompt chunk every 16.
    fn chain(&mut self, count: usize, base: u64, lat: &mut Lat) -> Result<()> {
        for k in 0..count as u64 {
            let i = base + k;
            if self.dir == 0 {
                if i.is_multiple_of(16) {
                    self.send(6, i, CHAIN_PROMPT)?;
                }
                let t0 = Instant::now();
                self.send(7, i, CHAIN_FWD)?;
                self.recv(8, i, CHAIN_HIDDEN, Wait::Idle { spin: Duration::from_micros(100) })?;
                self.recv(9, i, CHAIN_RESULT, Wait::Stall(link::STALL))?;
                let d = t0.elapsed();
                lat.add(d);
                self.slow("chain", i, d);
            } else {
                if i.is_multiple_of(16) {
                    self.recv(6, i, CHAIN_PROMPT, Wait::Idle { spin: Duration::from_micros(100) })?;
                }
                self.recv(7, i, CHAIN_FWD, Wait::Idle { spin: Duration::from_micros(100) })?;
                self.send(8, i, CHAIN_HIDDEN)?;
                self.send(9, i, CHAIN_RESULT)?;
            }
        }
        Ok(())
    }
}

fn stats_json(lanes: &Lanes) -> Value {
    json!(lanes.links().iter().map(|l| json!({"path": l.path(), "stats": l.stats()})).collect::<Vec<_>>())
}

/// Run `t` on this side; returns its results.
fn run_test(lanes: &mut Lanes, t: &Test, dir: u64, events: Option<&Arc<Events>>, report: &mut dyn FnMut(Value)) -> Result<Value> {
    let pat = Pattern::new(t.seed);
    let start = Instant::now();
    let mut side = Side { lanes, pat: &pat, dir, out: vec![0; LARGEST], inp: vec![0; LARGEST], checked: 0, slow: Vec::new(), start };
    let mut lat = Lat::default();
    let mut out = json!({});
    match t.mode {
        Mode::Pingpong => side.pingpong(t.size, t.count, 0, &mut lat)?,
        Mode::Exchange => side.exchange(t.size, t.count, 0, &mut lat)?,
        Mode::Chain => side.chain(t.count, 0, &mut lat)?,
        Mode::Bulk => {
            let (a, b) = side.bulk(t.size, t.count, 0)?;
            out["mb_s"] = json!({"client_to_server": a, "server_to_client": b});
        }
        Mode::Soak => {
            // Rounds of every pattern until the time is up; both sides count rounds, the client decides when to
            // stop (a round number, sent before each round: 0 ends).
            let mut lats: [Lat; 3] = Default::default();
            let mut rates = Vec::new();
            let mut round = 0u64;
            let mut last_report = Instant::now();
            loop {
                if dir == 0 {
                    let go = start.elapsed().as_secs_f64() < t.seconds;
                    side.lanes.send_message(&(if go { round + 1 } else { 0 }).to_le_bytes())?;
                    if !go {
                        break;
                    }
                } else {
                    let m = side.lanes.recv_message_idle()?;
                    if m == 0u64.to_le_bytes() {
                        break;
                    }
                }
                let base = round * 1000;
                side.chain(100, base, &mut lats[0])?;
                side.exchange(40 << 10, 100, base, &mut lats[1])?;
                side.pingpong(64, 100, base, &mut lats[2])?;
                rates.push(side.bulk(5 << 20, 1, base)?);
                round += 1;
                if dir == 0 && last_report.elapsed() > Duration::from_secs(10) {
                    last_report = Instant::now();
                    report(json!({"progress": round, "elapsed_s": start.elapsed().as_secs_f64(), "checked_bytes": side.checked,
                        "path": side.link().path(), "stats": side.link().stats()}));
                }
            }
            out["rounds"] = json!(round);
            out["chain_us"] = lats[0].summary();
            out["exchange_40k_us"] = lats[1].summary();
            out["pingpong_64_us"] = lats[2].summary();
            let n = rates.len().max(1) as f64;
            out["bulk_5m_mb_s"] = json!({"client_to_server": rates.iter().map(|r| r.0).sum::<f64>() / n,
                "server_to_client": rates.iter().map(|r| r.1).sum::<f64>() / n});
        }
    }
    if t.mode != Mode::Soak && t.mode != Mode::Bulk {
        out["lat_us"] = lat.summary();
    }
    out["elapsed_s"] = json!(start.elapsed().as_secs_f64());
    out["checked_bytes"] = json!(side.checked);
    out["slow"] = json!(side.slow);
    out["lanes"] = stats_json(side.lanes);
    if let Some(e) = events {
        out["events"] = json!(e.log.lock().unwrap().clone());
    }
    Ok(out)
}

fn serve(listen: &str, load: Load) -> Result<()> {
    let listener = TcpListener::bind(listen).with_context(|| format!("listening on {listen}"))?;
    eprintln!("link-test: serving on {listen}");
    let mut seed = unix_now().to_bits();
    loop {
        let (sock, peer) = listener.accept()?;
        // Lanes of one session come in a burst; take them here, run the session on a thread of its own.
        let first = match Link::accept_with(sock, &Options::default()) {
            Ok(Accepted::Link(l, tag)) => {
                seed = splitmix(seed);
                load.start(seed);
                (l, tag)
            }
            Ok(Accepted::Joined(_)) => continue,
            Err(e) => {
                eprintln!("link-test: {peer}: {e:#}");
                continue;
            }
        };
        let lanes = match collect(&listener, first) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("link-test: {peer}: {e:#}");
                continue;
            }
        };
        std::thread::spawn(move || {
            let mut lanes = lanes;
            let r = (|| -> Result<()> {
                loop {
                    let m = match lanes.recv_message_idle() {
                        Ok(m) => m,
                        Err(_) => return Ok(()),
                    };
                    let t: Test = serde_json::from_slice(&m)?;
                    eprintln!("link-test: {peer}: {t:?} over {}", lanes.kind());
                    if t.server_drops_stream > 0. {
                        let link = lanes.links()[0].clone();
                        let after = Duration::from_secs_f64(t.server_drops_stream);
                        std::thread::spawn(move || {
                            std::thread::sleep(after);
                            eprintln!("link-test: giving the stream up at {:.3}", unix_now());
                            link.give_up_stream("link-test fault injection");
                        });
                    }
                    if t.server_exits > 0. {
                        let after = Duration::from_secs_f64(t.server_exits);
                        std::thread::spawn(move || {
                            std::thread::sleep(after);
                            eprintln!("link-test: exiting at {:.3} (fault injection)", unix_now());
                            // SAFETY: ends the process at once, as a crash would; the kernel closes its files.
                            unsafe { libc::_exit(9) };
                        });
                    }
                    let res = run_test(&mut lanes, &t, 1, None, &mut |_| {})?;
                    lanes.send_message(&serde_json::to_vec(&res)?)?;
                }
            })();
            if let Err(e) = r {
                eprintln!("link-test: {peer}: {e:#}");
            }
        });
    }
}

/// The rest of a session's lanes after its first (tag `linktest lane i/n`).
fn collect(listener: &TcpListener, first: (Link, String)) -> Result<Lanes> {
    let (_, n) = first.1.rsplit_once('/').context("not a link-test session")?;
    let n: usize = n.parse()?;
    let mut links = vec![None; n];
    let i: usize = first.1.rsplit_once(" lane ").and_then(|(_, r)| r.split_once('/')).and_then(|(i, _)| i.parse().ok()).context("lane")?;
    links[i] = Some(first.0);
    while links.iter().any(Option::is_none) {
        let (sock, _) = listener.accept()?;
        match Link::accept_with(sock, &Options::default())? {
            Accepted::Link(l, tag) => {
                let i: usize = tag.rsplit_once(" lane ").and_then(|(_, r)| r.split_once('/')).and_then(|(i, _)| i.parse().ok()).context("lane")?;
                links[i] = Some(l);
            }
            Accepted::Joined(_) => {}
        }
    }
    Ok(Lanes::from_links(links.into_iter().map(Option::unwrap).collect()))
}

fn run(r: &Run) -> Result<()> {
    let mut o = Options::default();
    if r.drop > 0 || r.corrupt > 0 {
        o.streams = Arc::new(ChaosStreams { drop: r.drop, corrupt: r.corrupt });
    }
    o.tune = Tuning { overdrive: r.overdrive, ..Tuning::default() };
    let t0 = Instant::now();
    let mut lanes = Lanes::connect_with(&r.connect, "linktest", &o)?;
    let setup_ms = t0.elapsed().as_secs_f64() * 1e3;
    let events = Arc::new(Events { start: Instant::now(), log: Mutex::new(Vec::new()) });
    let stop = Arc::new(AtomicBool::new(false));
    let w = watch(lanes.links()[0].clone(), events.clone(), stop.clone(), r.label.clone());
    lanes.send_message(&serde_json::to_vec(&r.test)?)?;
    let label = r.label.clone();
    let mut report = |v: Value| println!("{}", json!({"label": label, "report": v}));
    let start_unix = unix_now();
    let res = run_test(&mut lanes, &r.test, 0, Some(&events), &mut report);
    stop.store(true, Ordering::Relaxed);
    let _ = w.join();
    let mut res = match res {
        Ok(r) => r,
        Err(e) => {
            let ev = events.log.lock().unwrap().clone();
            println!("{}", json!({"label": r.label, "error": format!("{e:#}"), "unix": unix_now(), "start_unix": start_unix,
                "elapsed_s": events.start.elapsed().as_secs_f64(), "events": ev, "lanes": stats_json(&lanes)}));
            return Err(e);
        }
    };
    res["start_unix"] = json!(start_unix);
    let server: Value = serde_json::from_slice(&lanes.recv_message()?)?;
    res["server"] = server;
    res["setup_ms"] = json!(setup_ms);
    res["up_ms"] = json!(lanes.links()[0].stats().first_up_ms);
    println!("{}", json!({"label": r.label, "connect": r.connect, "test": r.test, "result": res}));
    Ok(())
}

/// What one churn session's client printed: the `up_ms` line and its result (or error) line.
fn churn_lines(lines: &mpsc::Receiver<String>, wait: Option<Duration>, up_ms: &mut Option<f64>, last: &mut Option<Value>) {
    let deadline = wait.map(|w| Instant::now() + w);
    loop {
        let line = match deadline {
            Some(d) => match lines.recv_timeout(d.saturating_duration_since(Instant::now())) {
                Ok(l) => l,
                Err(_) => return,
            },
            None => match lines.recv() {
                Ok(l) => l,
                Err(_) => return,
            },
        };
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        if let Some(ms) = v.get("up_ms").and_then(Value::as_f64) {
            if v.get("result").is_none() {
                *up_ms = Some(ms);
                if deadline.is_some() {
                    return;
                }
                continue;
            }
        }
        *last = Some(v);
    }
}

fn churn(c: &Churn) -> Result<()> {
    let kinds: Vec<char> = c.pattern.chars().collect();
    ensure!(!kinds.is_empty() && kinds.iter().all(|k| "nks".contains(*k)), "--pattern takes n, k and s");
    let exe = std::env::current_exe()?;
    let mut seed = unix_now().to_bits();
    let (mut ups, mut failed, mut passed) = (Vec::new(), 0usize, 0usize);
    for i in 0..c.sessions {
        let kind = kinds[i % kinds.len()];
        let label = format!("{}-{i}", c.label);
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["link-test", "run", "--connect", &c.connect, "--label", &label, "--mode", "pingpong", "--size", "64"]);
        match kind {
            'n' => cmd.args(["--count", "2000"]),
            'k' => cmd.args(["--count", "100000000"]),
            _ => cmd.args(["--count", "100000000", "--server-exits", "0.2"]),
        };
        cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped());
        let t0 = Instant::now();
        let mut child = cmd.spawn().context("starting a session")?;
        let (tx, rx) = mpsc::channel();
        let out = child.stdout.take().context("stdout")?;
        let reader = std::thread::spawn(move || {
            for line in io::BufRead::lines(io::BufReader::new(out)).map_while(Result::ok) {
                let _ = tx.send(line);
            }
        });
        let mut err = child.stderr.take().context("stderr")?;
        let errors = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = io::Read::read_to_string(&mut err, &mut s);
            s
        });
        let (mut up_ms, mut last) = (None, None);
        churn_lines(&rx, Some(Duration::from_secs(5)), &mut up_ms, &mut last);
        if kind == 'k' {
            // Killed 50-250 ms after its stream came up (or after 5 s without it).
            seed = splitmix(seed);
            std::thread::sleep(Duration::from_millis(50 + seed % 200));
            child.kill()?;
        }
        let status = child.wait()?;
        churn_lines(&rx, None, &mut up_ms, &mut last);
        let _ = reader.join();
        let stderr = errors.join().unwrap_or_default();
        let not_opened = stderr.matches(" not opened (").count();
        let lane = last.as_ref().and_then(|v| v["result"]["lanes"].get(0).or_else(|| v["lanes"].get(0))).cloned();
        let ok = up_ms.is_some_and(|ms| ms <= c.within_ms) && not_opened == 0;
        failed += not_opened;
        passed += ok as usize;
        if let Some(ms) = up_ms {
            ups.push(ms);
        }
        println!(
            "{}",
            json!({"session": i, "kind": kind.to_string(), "up_ms": up_ms, "pass": ok, "failed_epochs": not_opened,
                "rc": status.code(), "ms": t0.elapsed().as_secs_f64() * 1e3, "lane": lane,
                "error": last.as_ref().and_then(|v| v.get("error")).cloned(),
                "stream_lines": stderr.lines().filter(|l| l.contains(" stream ")).collect::<Vec<_>>()})
        );
    }
    ups.sort_by(f64::total_cmp);
    let pct = |p: f64| ups.get(((ups.len() as f64 - 1.) * p).round() as usize).copied();
    let pass = passed == c.sessions;
    println!(
        "{}",
        json!({"churn": c.label, "sessions": c.sessions, "pattern": c.pattern, "within_ms": c.within_ms,
            "passed": passed, "on_stream": ups.len(), "failed_epochs": failed,
            "up_ms": {"p50": pct(0.5), "p99": pct(0.99), "max": ups.last()}, "pass": pass})
    );
    ensure!(pass, "{} of {} sessions reached the stream within {} ms with no failed epoch", passed, c.sessions, c.within_ms);
    Ok(())
}

/// GB/s of the link's CRC32C over SIZE bytes: one buffer (`one`), cut in stream records (`records`, as a write or a
/// read checks them), and the crc32c crate alone (`crate`).
fn crc_speed(size: usize, iters: usize) {
    let data: Vec<u8> = (0..size).map(|i| (i * 31 % 251) as u8).collect();
    let rate = |f: &mut dyn FnMut() -> u32| {
        let t0 = Instant::now();
        let mut x = 0u32;
        for _ in 0..iters {
            x ^= f();
        }
        std::hint::black_box(x);
        (size * iters) as f64 / t0.elapsed().as_secs_f64() / 1e9
    };
    let parts: Vec<&[u8]> = data.chunks(4048).collect();
    let mut out = Vec::new();
    let one = rate(&mut || link::crc(&data));
    let records = rate(&mut || {
        link::crcs(&parts, &mut out);
        out[0]
    });
    let krate = rate(&mut || crc32c::crc32c(&data));
    #[cfg(target_arch = "x86_64")]
    let hw = std::arch::is_x86_feature_detected!("sse4.2");
    #[cfg(not(target_arch = "x86_64"))]
    let hw = true;
    println!("{}", json!({"size": size, "iters": iters, "gb_s": records, "one_gb_s": one, "crate_gb_s": krate, "sse4_2": hw}));
}

pub fn main(cmd: &Cmd) -> Result<()> {
    match cmd {
        Cmd::Serve { listen, load } => serve(listen, *load),
        Cmd::Run(r) => run(r),
        Cmd::Churn(c) => churn(c),
        Cmd::Crc { size, iters } => {
            crc_speed(*size, *iters);
            Ok(())
        }
    }
}
