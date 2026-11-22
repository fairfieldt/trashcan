//! Links between machines: the one transport every model uses across Thunderbolt cables (and any other network).
//!
//! A `Link` is an ordered, reliable byte stream each way between two processes, opened over TCP (`connect` and
//! `accept`; the TCP connection names the session and carries control records). When the TCP connection runs over a
//! Thunderbolt link and both ends have that cable's USB4STREAM device (created by `tblink`), the data goes on the
//! stream, which has about half TCP's latency: `connect` returns once the stream is up, and a stream that fails, or
//! does not come up within `Tuning::stream_wait`, ends the link with an error. Such a link never carries on over
//! TCP. Without a stream at both ends (another network, tests) the link is TCP only. See `proto.rs` for the protocol
//! and `device.rs` for what the driver does.
//!
//! Above the byte stream:
//! - `Lanes`: the links to one machine, one per cable, for tensor-parallel exchanges (each lane carries an equal
//!   slice) and small control messages (`[u32 length][bytes]` on the first lane). With several cables every lane
//!   also gets a TCP path over the next cable, so a TCP-only lane whose cable goes away carries on over another one.
//! - `Conn`: the pipeline chains' frames, `[u32 header length][u64 blob length][JSON header][blob]`.
//!
//! The application drives a link while it sends, receives or exchanges: that thread polls the paths without sleeping
//! (or with `poll()` on idle waits), returns credits and acknowledgements inline, and never opens or closes a
//! device. `service.rs` keeps every link going in between.
mod device;
mod proto;
mod record;
mod service;
#[cfg(test)]
mod sim;
#[cfg(test)]
mod tests;

pub use device::{Found, Frames, Streams, Sysfs};
pub use record::{crc, crcs};
pub use proto::{Stats, Tuning};

use anyhow::{bail, ensure, Context, Result};
use proto::{Phase, Sink, State};
use record::{kind, Header, HEADER};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io::{self, ErrorKind, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex, MutexGuard, Weak},
    time::{Duration, Instant},
};

/// A wait that sees no progress for this long has failed (the other machine died or hangs).
pub const STALL: Duration = Duration::from_secs(30);
/// An idle wait for the next message spins this long after the last progress before it sleeps in `poll()`.
const BUSY: Duration = Duration::from_millis(50);
/// A pipeline frame's first bytes: spin this long, then sleep in `poll()` (woken by the data).
const CONN_SPIN: Duration = Duration::from_micros(100);
/// `write` returns once at most this much of what it queued is still to be handed to a path.
const SEND_AHEAD: u64 = 1 << 20;
const VERSION: u64 = 1;
/// A closing link waits at most this long for its last bytes to arrive.
const FINISH: Duration = Duration::from_secs(2);
/// `connect` waits at most this long, once the stream is up, for both sides to have measured the hold of their
/// directions (`proto.rs`); a link that has not by then goes on with the frames an unmeasured direction gets.
const HOLD_WAIT: Duration = Duration::from_millis(100);
/// An idle wait on a stream device that cannot wake a `poll()` (busy-polled rings, see `device.rs`) sleeps this
/// long between looks, and `IDLE_NAP_LONG` once nothing has arrived for `IDLE_LONG`.
const IDLE_NAP: Duration = Duration::from_micros(50);
const IDLE_NAP_LONG: Duration = Duration::from_millis(1);
const IDLE_LONG: Duration = Duration::from_millis(200);

/// How a link finds its stream and its timing.
#[derive(Clone)]
pub struct Options {
    pub streams: Arc<dyn Streams>,
    pub tune: Tuning,
    /// Keep trying to connect this long (the other machine may still be starting).
    pub connect_for: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self { streams: Arc::new(Sysfs), tune: Tuning::default(), connect_for: Duration::from_secs(60) }
    }
}

/// How long a read may wait.
#[derive(Clone, Copy, Debug)]
pub enum Wait {
    /// Fail when no byte arrives for this long; spin meanwhile.
    Stall(Duration),
    /// No deadline for the first byte (the next request to a server); spin `spin` after the last progress, then
    /// sleep in `poll()`. After the first byte, `STALL`.
    Idle { spin: Duration },
    /// The first byte within this long (sleeping in `poll()`), then `STALL`.
    First(Duration),
}

pub(crate) struct Inner {
    st: Mutex<State>,
    streams: Arc<dyn Streams>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Normally closed already (`Owner`); this may run on the maintenance thread, which must not wait.
        self.st.get_mut().unwrap_or_else(|e| e.into_inner()).shutdown();
    }
}

/// The application's hold on a link: when the last `Link` clone goes, the link finishes sending and closes, on the
/// application's thread.
struct Owner {
    inner: Arc<Inner>,
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.inner.finish();
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Keep the link going until everything queued has arrived (at most `FINISH`), then say goodbye: a goodbye over
    /// TCP must not overtake data still on the stream. The lock is held only while pumping.
    fn finish(&self) {
        let deadline = Instant::now() + FINISH;
        loop {
            {
                let mut st = self.lock();
                let now = Instant::now();
                if st.delivered() || st.is_closed() || now >= deadline || st.pump(now, &mut Sink::none()).is_err() {
                    st.shutdown();
                    return;
                }
            }
            std::thread::sleep(Duration::from_micros(50));
        }
    }

    /// The stream epoch that wants a device, if any (never waits for the lock).
    fn take_job(&self) -> Option<proto::Job> {
        self.st.try_lock().ok()?.stream.job.take()
    }

    /// Open a stream device for `job` (outside the lock: it can take a second).
    fn run_job(&self, job: proto::Job) {
        let found = self.streams.find(&job.local);
        let r = match found {
            Err(why) => {
                self.lock().no_stream(job.epoch, &why, Instant::now());
                return;
            }
            Ok(found) => match &job.expect {
                Some((local, peer)) if found.local != *peer || found.peer != *local => {
                    let why = format!("the two ends found different cables ({} here)", found.label);
                    self.lock().no_stream(job.epoch, &why, Instant::now());
                    return;
                }
                _ => {
                    let t = Instant::now();
                    let r = self.streams.open(&found).map(|d| (d, found));
                    let waited = t.elapsed();
                    let asked = job.at.elapsed();
                    if waited > Duration::from_millis(100) || asked > Duration::from_millis(100) {
                        // The driver holds an open back while an earlier close of the device runs (0.5 s or more);
                        // a busy machine can be slow to start the open.
                        let label = self.lock().label.clone();
                        proto::say(format!(
                            "link {label}: opening the stream device took {waited:.0?}, {asked:.0?} after the epoch asked for it [{}]",
                            proto::clock()
                        ));
                    }
                    r
                }
            },
        };
        self.lock().install(job.epoch, r, Instant::now());
    }
}

/// One link to another machine (see the module documentation). Clones share the link: one thread may read while
/// another writes.
#[derive(Clone)]
pub struct Link {
    inner: Arc<Inner>,
    _owner: Arc<Owner>,
}

/// This machine's host name.
pub fn hostname() -> String {
    let mut b = [0u8; 256];
    // SAFETY: gethostname writes at most b.len() bytes.
    if unsafe { libc::gethostname(b.as_mut_ptr() as *mut libc::c_char, b.len()) } != 0 {
        return "?".into();
    }
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..n]).into_owned()
}

fn random_u64() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());
    h.write_u32(std::process::id());
    h.finish()
}

#[cfg(target_os = "linux")]
fn keepalive(s: &TcpStream) {
    use std::os::unix::io::AsRawFd;
    let fd = s.as_raw_fd();
    let set = |level: libc::c_int, name: libc::c_int, v: libc::c_int| {
        // SAFETY: setsockopt with an int option on a socket this process owns.
        unsafe {
            libc::setsockopt(fd, level, name, &v as *const _ as *const libc::c_void, std::mem::size_of::<libc::c_int>() as libc::socklen_t)
        };
    };
    set(libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, 2);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, 1);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPCNT, 3);
}

#[cfg(not(target_os = "linux"))]
fn keepalive(_: &TcpStream) {}

/// Handshake records go out whole, in blocking mode with a timeout.
fn hs_send(sock: &mut TcpStream, k: u8, session: u64, payload: &Value) -> io::Result<()> {
    let p = serde_json::to_vec(payload)?;
    let h = Header { kind: k, len: p.len() as u16, session: session as u32, pcrc: crc(&p), ..Default::default() };
    sock.write_all(&h.encode())?;
    sock.write_all(&p)
}

fn hs_recv(sock: &mut TcpStream) -> io::Result<(Header, Value)> {
    let mut b = [0u8; HEADER];
    sock.read_exact(&mut b)?;
    let h = Header::decode(&b).ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "not a trashcan link (another program or version)"))?;
    let mut p = vec![0u8; h.len as usize];
    sock.read_exact(&mut p)?;
    if crc(&p) != h.pcrc {
        return Err(io::Error::new(ErrorKind::InvalidData, "damaged handshake"));
    }
    let v = serde_json::from_slice(&p).map_err(|e| io::Error::new(ErrorKind::InvalidData, e))?;
    Ok((h, v))
}

/// Connect, trying again every 100 ms for `limit` while the other machine is still starting (refused) or its link is
/// still coming up (unreachable, timed out); a name that does not resolve fails at once.
fn connect_retry(addr: &str, limit: Duration) -> Result<TcpStream> {
    let start = Instant::now();
    let mut told = false;
    loop {
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e)
                if start.elapsed() < limit
                    && matches!(
                        e.kind(),
                        ErrorKind::ConnectionRefused
                            | ErrorKind::HostUnreachable
                            | ErrorKind::NetworkUnreachable
                            | ErrorKind::TimedOut
                            | ErrorKind::ConnectionReset
                            | ErrorKind::AddrNotAvailable
                    ) =>
            {
                if !told && start.elapsed() > Duration::from_secs(2) {
                    eprintln!("link: waiting for {addr} ({e})");
                    told = true;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                let waited = start.elapsed();
                let after = if waited > Duration::from_secs(1) { format!(" (tried for {waited:.0?})") } else { String::new() };
                return Err(e).with_context(|| format!("connecting to {addr}{after}"));
            }
        }
    }
}

/// This end's stream for the cable under `sock`, as told to the other end.
fn stream_info(o: &Options, sock: &TcpStream) -> Value {
    match sock.local_addr().map_err(|e| e.to_string()).and_then(|a| o.streams.find(&a)) {
        Ok(f) => json!({"local": f.local, "peer": f.peer, "ring": f.ring}),
        Err(why) => {
            if !why.is_empty() {
                eprintln!("link: no stream: {why}");
            }
            Value::Null
        }
    }
}

/// Both ends have a stream and it is the same cable.
fn same_cable(a: &Value, b: &Value) -> bool {
    !a.is_null() && !b.is_null() && a["local"] == b["peer"] && a["peer"] == b["local"]
}

/// Links accepted on this machine, by session, for JOIN.
static ACCEPTED: Mutex<BTreeMap<u64, Weak<Inner>>> = Mutex::new(BTreeMap::new());

/// What `Link::accept_with` took in.
pub enum Accepted {
    Link(Link, String),
    /// A further TCP path of a link accepted earlier (its session).
    Joined(u64),
}

impl Link {
    fn start(st: State, streams: Arc<dyn Streams>) -> Link {
        let inner = Arc::new(Inner { st: Mutex::new(st), streams });
        service::register(&inner);
        Link { _owner: Arc::new(Owner { inner: inner.clone() }), inner }
    }

    pub fn connect(addr: &str, tag: &str) -> Result<Link> {
        Self::connect_with(addr, tag, &Options::default())
    }

    /// Connect to `addr` (keeps trying for `o.connect_for`), introduce this side with `tag` (what the other side
    /// should do with the link), and, when both ends have a stream, return once it is up.
    pub fn connect_with(addr: &str, tag: &str, o: &Options) -> Result<Link> {
        let (link, same) = Self::handshake(addr, tag, o)?;
        link.settle(same)?;
        Ok(link)
    }

    fn handshake(addr: &str, tag: &str, o: &Options) -> Result<(Link, bool)> {
        let mut sock = connect_retry(addr, o.connect_for)?;
        sock.set_nodelay(true)?;
        keepalive(&sock);
        sock.set_read_timeout(Some(Duration::from_secs(30)))?;
        sock.set_write_timeout(Some(Duration::from_secs(30)))?;
        let session = random_u64();
        let mine = stream_info(o, &sock);
        hs_send(&mut sock, kind::HELLO, session, &json!({"v": VERSION, "session": session, "tag": tag, "host": hostname(), "stream": mine}))?;
        let (h, w) = hs_recv(&mut sock).with_context(|| format!("waiting for {addr} to answer"))?;
        ensure!(h.kind == kind::WELCOME && h.session == session as u32, "{addr} did not accept the link: {w}");
        if let Some(e) = w.get("error") {
            bail!("{addr}: {}", e.as_str().unwrap_or("refused"));
        }
        sock.set_read_timeout(None)?;
        sock.set_write_timeout(None)?;
        sock.set_nonblocking(true)?;
        let host = w["host"].as_str().unwrap_or("?");
        let mut st = State::new(session, true, format!("{host} ({addr})"), o.tune.clone(), sock, random_u64() as u32)?;
        let same = same_cable(&mine, &w["stream"]);
        st.peer = host.to_owned();
        st.expects_stream = same;
        Ok((Link::start(st, o.streams.clone()), same))
    }

    /// The connecting side, right after the handshake: with a stream at both ends, open epochs until one is up (the
    /// link ends after `stream_wait` without), so that the session's data goes on the stream from its start, and
    /// then give both sides up to `HOLD_WAIT` to measure the hold of their directions while nothing else is on the
    /// stream (a few milliseconds on a cable that holds nothing).
    fn settle(&self, same: bool) -> Result<()> {
        if !same {
            return Ok(());
        }
        self.inner.lock().first_epoch(Instant::now());
        let mut up: Option<Instant> = None;
        loop {
            if let Some(job) = self.inner.take_job() {
                self.inner.run_job(job);
            }
            let mut st = self.inner.lock();
            st.pump(Instant::now(), &mut Sink::none())?;
            if st.stream.phase == Phase::Up {
                let since = *up.get_or_insert_with(Instant::now);
                if st.hold_measured() == (true, true) || since.elapsed() > HOLD_WAIT {
                    return Ok(());
                }
            } else {
                up = None;
            }
            drop(st);
            std::thread::sleep(Duration::from_micros(200));
        }
    }

    /// Take a further TCP path to the same machine (over another cable), used when the link's own path fails.
    pub fn join(&self, addr: &str) -> Result<()> {
        let mut sock = TcpStream::connect(addr).with_context(|| format!("connecting to {addr}"))?;
        sock.set_nodelay(true)?;
        keepalive(&sock);
        sock.set_read_timeout(Some(Duration::from_secs(30)))?;
        sock.set_write_timeout(Some(Duration::from_secs(30)))?;
        let session = self.inner.lock().session;
        hs_send(&mut sock, kind::JOIN, session, &json!({"v": VERSION, "session": session}))?;
        let (h, w) = hs_recv(&mut sock)?;
        ensure!(h.kind == kind::WELCOME && w.get("error").is_none(), "{addr} refused the path: {w}");
        sock.set_read_timeout(None)?;
        sock.set_write_timeout(None)?;
        sock.set_nonblocking(true)?;
        self.inner.lock().tcp.push(proto::TcpPath::new(sock));
        Ok(())
    }

    /// Answer a connection from `connect_with` (a new link and its tag) or from `join`.
    pub fn accept_with(mut sock: TcpStream, o: &Options) -> Result<Accepted> {
        let peer = sock.peer_addr()?;
        sock.set_nodelay(true)?;
        keepalive(&sock);
        sock.set_read_timeout(Some(Duration::from_secs(30)))?;
        sock.set_write_timeout(Some(Duration::from_secs(30)))?;
        let (h, hello) = hs_recv(&mut sock).with_context(|| format!("handshake from {peer}"))?;
        let session = hello["session"].as_u64().context("handshake without a session")?;
        ensure!(h.session == session as u32, "handshake from {peer}: session mismatch");
        ensure!(hello["v"].as_u64() == Some(VERSION), "{peer} speaks link version {}, this binary {VERSION}", hello["v"]);
        let me = hostname();
        if h.kind == kind::JOIN {
            let link = ACCEPTED.lock().unwrap_or_else(|e| e.into_inner()).get(&session).and_then(Weak::upgrade);
            let Some(inner) = link else {
                hs_send(&mut sock, kind::WELCOME, session, &json!({"error": "no such link"}))?;
                bail!("{peer} joined an unknown link");
            };
            hs_send(&mut sock, kind::WELCOME, session, &json!({"v": VERSION, "host": me}))?;
            sock.set_read_timeout(None)?;
            sock.set_write_timeout(None)?;
            sock.set_nonblocking(true)?;
            inner.lock().tcp.push(proto::TcpPath::new(sock));
            return Ok(Accepted::Joined(session));
        }
        ensure!(h.kind == kind::HELLO, "{peer}: unexpected handshake record {}", h.kind);
        let mine = stream_info(o, &sock);
        let same = same_cable(&mine, &hello["stream"]);
        hs_send(&mut sock, kind::WELCOME, session, &json!({"v": VERSION, "host": me, "stream": mine}))?;
        sock.set_read_timeout(None)?;
        sock.set_write_timeout(None)?;
        sock.set_nonblocking(true)?;
        let host = hello["host"].as_str().unwrap_or("?");
        let mut st = State::new(session, false, format!("{host} ({peer})"), o.tune.clone(), sock, 0)?;
        st.peer = host.to_owned();
        st.expects_stream = same;
        let link = Link::start(st, o.streams.clone());
        {
            let mut a = ACCEPTED.lock().unwrap_or_else(|e| e.into_inner());
            a.retain(|_, w| w.strong_count() > 0);
            a.insert(session, Arc::downgrade(&link.inner));
        }
        // The connecting side opens the stream epoch; this side's maintenance thread answers it.
        Ok(Accepted::Link(link, hello["tag"].as_str().unwrap_or("").to_owned()))
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        let mut st = self.inner.lock();
        st.last_app = Instant::now();
        st
    }

    /// The path carrying data now: "stream" or "tcp".
    pub fn path(&self) -> &'static str {
        self.inner.lock().path()
    }

    /// The link has ended (closed, or failed: see `report` and the link's messages).
    pub fn is_closed(&self) -> bool {
        self.inner.lock().is_closed()
    }

    pub fn stats(&self) -> Stats {
        self.inner.lock().stats
    }

    /// How this link carries its data, for run reports. `transport`: "stream" or "tcp" (`path`); a link whose stream
    /// failed has ended. `failed_epochs` counts the epochs that did not come up before the first that did.
    pub fn report(&self) -> Value {
        let st = self.inner.lock();
        let s = st.stats;
        let transport = st.path();
        json!({"to": st.peer, "transport": transport, "stream_at_both_ends": st.expects_stream,
            "first_up_ms": s.first_up_ms, "epochs": s.epochs, "failed_epochs": s.failed_epochs,
            "fallbacks": s.fallbacks, "sent_stream": s.sent_stream, "sent_tcp": s.sent_tcp,
            "hold": s.hold, "hold_raises": s.hold_raises, "pushes": s.pushes, "nudges": s.nudges})
    }

    /// The protocol state, for messages.
    #[cfg(test)]
    pub fn describe(&self) -> String {
        self.inner.lock().describe()
    }

    /// Say goodbye and close. Readers on other threads get an error.
    pub fn shutdown(&self) {
        self.inner.finish();
    }

    /// Give the stream up as a failing device would (fault injection in `trashcan link-test`): the link ends with an
    /// error on both sides.
    pub fn give_up_stream(&self, why: &str) {
        self.inner.lock().stream_down(Instant::now(), why, true);
    }

    /// Cut TCP path `i` here, as if its cable had gone (tests).
    #[cfg(test)]
    pub fn kill_path(&self, i: usize) {
        let _ = self.inner.lock().tcp[i].sock.shutdown(std::net::Shutdown::Both);
    }

    /// Queue `parts` and send; returns once at most `SEND_AHEAD` bytes of them still wait for a path.
    pub fn send(&self, parts: &[&[u8]]) -> io::Result<()> {
        {
            let mut st = self.lock();
            for p in parts {
                st.queue(p)?;
            }
            // The frames go before this turn reads anything: the other side may be waiting for them.
            st.send_now(Instant::now())?;
        }
        let mut fds = Vec::new();
        let mut last = Instant::now();
        loop {
            let mut polled = false;
            let (left, progress) = {
                let mut st = self.lock();
                let p = st.pump(Instant::now(), &mut Sink::none())?;
                let left = st.unsent();
                if left > SEND_AHEAD && !p {
                    polled = st.poll_fds(&mut fds);
                }
                (left, p)
            };
            if left <= SEND_AHEAD {
                return Ok(());
            }
            let now = Instant::now();
            if progress {
                last = now;
            } else if now - last > STALL {
                return Err(io::Error::new(ErrorKind::TimedOut, "sending stalled"));
            } else if now - last > Duration::from_micros(200) {
                idle_wait(&mut fds, 1, polled, now - last);
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// Fill `buf` in order.
    pub fn read_exact(&self, buf: &mut [u8], wait: Wait) -> io::Result<()> {
        self.read_some(buf, wait, true).map(|_| ())
    }

    /// Spin, moving the link along, until bytes wait to be read or `limit` has passed: for a frame expected shortly,
    /// before a read that would sleep (`qwen4::net`). The lock is only tried, so a thread sending on the link meanwhile
    /// (a node's replies while its request reader waits for the next step) is not held up.
    pub fn spin_readable(&self, limit: Duration) {
        let start = Instant::now();
        while start.elapsed() < limit {
            if let Ok(mut st) = self.inner.st.try_lock() {
                let now = Instant::now();
                st.last_app = now;
                if !st.inq.is_empty() || st.pump(now, &mut Sink::none()).is_err() || !st.inq.is_empty() {
                    return;
                }
            }
            for _ in 0..32 {
                std::hint::spin_loop();
            }
        }
    }

    /// Read into `buf`: all of it (`exact`), or what has arrived once something has. Without `exact`, the end of the
    /// link with nothing left to read is Ok(0).
    fn read_some(&self, buf: &mut [u8], wait: Wait, exact: bool) -> io::Result<usize> {
        let start = Instant::now();
        let mut last = start;
        let mut filled = 0;
        let mut fds = Vec::new();
        loop {
            let progress = {
                let mut st = self.lock();
                filled += st.inq.take(&mut buf[filled..]);
                if filled == buf.len() || (!exact && filled > 0) {
                    return Ok(filled);
                }
                let mut sink = Sink { buf: &mut buf[filled..], filled: 0 };
                let r = st.pump(Instant::now(), &mut sink);
                let got = sink.filled;
                filled += got;
                if filled == buf.len() || (!exact && filled > 0) {
                    return Ok(filled);
                }
                match r {
                    Err(e) if !exact && e.kind() == ErrorKind::UnexpectedEof => return Ok(0),
                    r => r?,
                };
                got > 0
            };
            let now = Instant::now();
            if progress {
                last = now;
                continue;
            }
            let first = filled == 0;
            let quiet = now - last;
            let (limit, spin) = match wait {
                Wait::Stall(d) => (Some(d), true),
                Wait::Idle { spin } if first => (None, quiet < spin),
                Wait::First(d) if first => (Some(d), false),
                _ => (Some(STALL), true),
            };
            if limit.is_some_and(|l| quiet > l) {
                let what = if first { "waiting for the other machine" } else { "receiving" };
                return Err(io::Error::new(ErrorKind::TimedOut, format!("{what}: nothing for {quiet:.0?} ({})", self.inner.lock().describe())));
            }
            if spin {
                std::hint::spin_loop();
            } else {
                let polled = self.inner.lock().poll_fds(&mut fds);
                idle_wait(&mut fds, 2, polled, quiet);
            }
        }
    }
}

/// A link as a byte stream for code written against `Read` (a `BufReader`, `read_exact`): a read waits like a TCP
/// socket's, without a deadline, spinning briefly and then sleeping in `poll()`; the end of the link reads as 0.
impl Read for Link {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        self.read_some(buf, Wait::Idle { spin: CONN_SPIN }, false)
    }
}

/// `Write` for a link: the bytes are queued and sent as `Link::send` does; `flush` has nothing to do.
impl Write for Link {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.send(&[buf])?;
        Ok(buf.len())
    }

    fn write_vectored(&mut self, bufs: &[io::IoSlice]) -> io::Result<usize> {
        let parts: Vec<&[u8]> = bufs.iter().map(|b| &**b).collect();
        self.send(&parts)?;
        Ok(parts.iter().map(|p| p.len()).sum())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn poll(fds: &mut [libc::pollfd], ms: i32) {
    if fds.is_empty() {
        std::thread::sleep(Duration::from_millis(ms as u64));
        return;
    }
    // SAFETY: fds is a valid array of pollfd for its length.
    unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, ms) };
}

/// Sleep until one of `fds` is ready or `ms` have passed. With `polled`, the link's stream device is not among them
/// (nothing wakes a sleeper when a frame arrives on busy-polled rings), so the sleep is a short one: `IDLE_NAP`
/// while the link has been `quiet` for less than `IDLE_LONG`, then `IDLE_NAP_LONG`, so that a link nobody uses
/// leaves its core alone.
fn idle_wait(fds: &mut [libc::pollfd], ms: i32, polled: bool, quiet: Duration) {
    if !polled {
        return poll(fds, ms);
    }
    let nap = if quiet < IDLE_LONG { IDLE_NAP } else { IDLE_NAP_LONG };
    #[cfg(target_os = "linux")]
    {
        // A thread's timers fire up to 50 us late by default (timer slack): too coarse for a 50 us nap.
        thread_local!(static SLACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });
        if !SLACK.replace(true) {
            // SAFETY: sets this thread's timer slack, in nanoseconds.
            unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, 1000 as libc::c_ulong) };
        }
        let ts = libc::timespec { tv_sec: 0, tv_nsec: nap.as_nanos() as libc::c_long };
        // SAFETY: fds is a valid array of pollfd for its length, ts a valid timeout, and no signal mask is given.
        unsafe { libc::ppoll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, &ts, std::ptr::null()) };
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = fds;
        std::thread::sleep(nap);
    }
}

/// When an exchange had issued all its sends and when its first incoming byte arrived, as time since its start:
/// the one-way trip to the other machine plus its lateness, and the transfer that follows (`Mesh` tallies them).
#[derive(Clone, Copy, Debug, Default)]
pub struct Trip {
    pub sent: Duration,
    pub first: Duration,
    pub total: Duration,
}

/// The links to one neighbouring machine, one per cable, in an order both machines agree on.
pub struct Lanes {
    lanes: Vec<Link>,
}

impl Lanes {
    /// Every lane's `Link::report`, for run reports and diagnostics.
    pub fn reports(&self) -> Vec<Value> {
        self.lanes.iter().map(|l| l.report()).collect()
    }
}

/// Tag of lane `i` of `n` of group `group` (what `Lanes::accept_groups` sorts by).
fn lane_tag(group: &str, i: usize, n: usize) -> String {
    format!("{group} lane {i}/{n}")
}

fn parse_lane_tag(tag: &str) -> Option<(String, usize, usize)> {
    let (group, rest) = tag.rsplit_once(" lane ")?;
    let (i, n) = rest.split_once('/')?;
    Some((group.to_owned(), i.parse().ok()?, n.parse().ok()?))
}

impl Lanes {
    /// One link per address, in order, introduced as lanes of `group`. With several cables each lane also gets a
    /// TCP path over the next cable.
    pub fn connect(addrs: &[String], group: &str) -> Result<Self> {
        Self::connect_with(addrs, group, &Options::default())
    }

    pub fn connect_with(addrs: &[String], group: &str, o: &Options) -> Result<Self> {
        ensure!(!addrs.is_empty(), "no lanes");
        let n = addrs.len();
        let mut lanes = Vec::new();
        for (i, a) in addrs.iter().enumerate() {
            lanes.push(Link::handshake(a, &lane_tag(group, i, n), o).with_context(|| format!("lane {i} to {a}"))?);
        }
        if n > 1 {
            for (i, (l, _)) in lanes.iter().enumerate() {
                let backup = &addrs[(i + 1) % n];
                if let Err(e) = l.join(backup) {
                    eprintln!("link: lane {i} has no second path over {backup}: {e:#}");
                }
            }
        }
        for (i, (l, same)) in lanes.iter().enumerate() {
            l.settle(*same).with_context(|| format!("lane {i} to {}", addrs[i]))?;
        }
        Ok(Self { lanes: lanes.into_iter().map(|l| l.0).collect() })
    }

    /// Accept the lanes of one machine's group.
    pub fn accept(listener: &TcpListener) -> Result<(Self, String)> {
        Self::accept_groups(listener, 1).map(|mut g| g.pop().map(|(group, l)| (l, group)).unwrap())
    }

    /// Accept connections until `count` groups have all their lanes, each in lane order, and (with several lanes)
    /// the further path each lane takes over the next cable, or 2 s have passed waiting for those.
    pub fn accept_groups(listener: &TcpListener, count: usize) -> Result<Vec<(String, Self)>> {
        Self::accept_groups_with(listener, count, &Options::default())
    }

    pub fn accept_groups_with(listener: &TcpListener, count: usize, o: &Options) -> Result<Vec<(String, Self)>> {
        let mut groups: BTreeMap<String, Vec<Option<Link>>> = BTreeMap::new();
        let mut done: Vec<(String, Self)> = Vec::new();
        let mut joined: Vec<u64> = Vec::new();
        let mut complete_at: Option<Instant> = None;
        loop {
            if done.len() == count {
                let sessions: Vec<u64> = done.iter().filter(|g| g.1.lanes.len() > 1).flat_map(|g| g.1.lanes.iter().map(|l| l.inner.lock().session)).collect();
                let all = sessions.iter().all(|s| joined.contains(s));
                let at = *complete_at.get_or_insert_with(Instant::now);
                if all || at.elapsed() > Duration::from_secs(2) {
                    listener.set_nonblocking(false)?;
                    return Ok(done);
                }
                listener.set_nonblocking(true)?;
            }
            let (sock, peer) = match listener.accept() {
                Ok(c) => c,
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            sock.set_nonblocking(false)?;
            let (link, tag) = match Link::accept_with(sock, o) {
                Ok(Accepted::Link(l, t)) => (l, t),
                Ok(Accepted::Joined(s)) => {
                    joined.push(s);
                    continue;
                }
                Err(e) => {
                    eprintln!("link: connection from {peer} refused: {e:#}");
                    continue;
                }
            };
            let Some((group, i, n)) = parse_lane_tag(&tag) else {
                eprintln!("link: {peer} is not a lane ({tag:?})");
                continue;
            };
            let slots = groups.entry(group.clone()).or_insert_with(|| (0..n).map(|_| None).collect());
            ensure!(slots.len() == n && i < n && slots[i].is_none(), "unexpected lane {i} of {n} from {peer} ({group})");
            slots[i] = Some(link);
            if slots.iter().all(Option::is_some) {
                let lanes = groups.remove(&group).unwrap().into_iter().map(Option::unwrap).collect();
                done.push((group, Self { lanes }));
            }
        }
    }

    pub fn from_links(lanes: Vec<Link>) -> Self {
        Self { lanes }
    }

    pub fn count(&self) -> usize {
        self.lanes.len()
    }

    /// The paths carrying the lanes: "stream", "tcp" or "stream+tcp".
    pub fn kind(&self) -> String {
        let paths: Vec<&str> = self.lanes.iter().map(Link::path).collect();
        if paths.iter().all(|p| *p == paths[0]) {
            paths[0].to_owned()
        } else {
            "stream+tcp".into()
        }
    }

    pub fn links(&self) -> &[Link] {
        &self.lanes
    }

    /// Send a small control message on the first lane: a length and the bytes.
    pub fn send_message(&mut self, bytes: &[u8]) -> Result<()> {
        self.lanes[0].send(&[&(bytes.len() as u32).to_le_bytes(), bytes])?;
        Ok(())
    }

    pub fn recv_message(&mut self) -> Result<Vec<u8>> {
        self.recv_waiting(Wait::Stall(STALL))
    }

    /// `recv_message` for a message that may come hours later (the next request to a server): no deadline for its
    /// first bytes, polled in `poll()` once the lane has been quiet for `BUSY`, so an idle machine leaves its cores
    /// alone; the rest must then arrive within `STALL`. The end of the link is an error.
    pub fn recv_message_idle(&mut self) -> Result<Vec<u8>> {
        self.recv_waiting(Wait::Idle { spin: BUSY })
    }

    /// Wait at most `first_byte` for a message to start. Used for model readiness, whose cold disk reads may exceed
    /// `STALL`; once the first byte arrives, the rest keeps the ordinary progress deadline.
    pub fn recv_message_first_within(&mut self, first_byte: Duration) -> Result<Vec<u8>> {
        self.recv_waiting(Wait::First(first_byte))
    }

    fn recv_waiting(&mut self, wait: Wait) -> Result<Vec<u8>> {
        let l = &self.lanes[0];
        let mut len = [0u8; 4];
        l.read_exact(&mut len, wait).context("the other machine closed the lanes")?;
        let mut bytes = vec![0u8; u32::from_le_bytes(len) as usize];
        l.read_exact(&mut bytes, Wait::Stall(STALL))?;
        Ok(bytes)
    }

    /// Send `out` to the peer and receive as many bytes into `inp`; the peer makes the matching call. Lane `i`
    /// carries the `i`th of `count` equal, 4-byte aligned slices, and every lane moves both ways at once.
    pub fn exchange(&mut self, out: &[u8], inp: &mut [u8]) -> Result<()> {
        self.exchange_within(out, inp, STALL).map(|_| ())
    }

    /// `exchange`, failing when the lanes make no progress for `stall`.
    pub fn exchange_within(&mut self, out: &[u8], inp: &mut [u8], stall: Duration) -> Result<Trip> {
        exchange_all(&mut [(self, out, inp)], stall)
    }
}

/// Several `exchange`s at once, each with the peer of its lanes, polled together from the calling thread so that the
/// cables to different machines carry their slices at the same time. Every send is queued first; the links' credits
/// keep each stream within the other side's receive ring.
pub fn exchange_all(jobs: &mut [(&mut Lanes, &[u8], &mut [u8])], stall: Duration) -> Result<Trip> {
    for (_, out, inp) in jobs.iter() {
        ensure!(out.len() == inp.len() && out.len().is_multiple_of(4), "exchange sizes must match and align to 4 bytes");
    }
    struct Part<'a> {
        link: &'a Link,
        inp: &'a mut [u8],
        got: usize,
        sent: bool,
    }
    let start = Instant::now();
    let mut parts = Vec::new();
    for (lanes, out, inp) in jobs.iter_mut() {
        let n = lanes.lanes.len();
        let cuts: Vec<usize> = (0..=n).map(|i| out.len() / 4 * i / n * 4).collect();
        let mut rest: &mut [u8] = inp;
        for (i, link) in lanes.lanes.iter().enumerate() {
            let (chunk, tail) = std::mem::take(&mut rest).split_at_mut(cuts[i + 1] - cuts[i]);
            rest = tail;
            let mut st = link.lock();
            st.queue(&out[cuts[i]..cuts[i + 1]])?;
            // The frames go before anything is read: the other side's exchange waits for them, and the turn that
            // follows in the loop below reads.
            st.send_now(Instant::now()).with_context(|| format!("lane {i} to {}", st.label))?;
            parts.push(Part { link, inp: chunk, got: 0, sent: false });
        }
    }
    let mut trip = Trip::default();
    let mut last = start;
    let (mut all_sent, mut any_read) = (false, false);
    loop {
        let (mut busy, mut done, mut sent) = (false, true, true);
        for p in parts.iter_mut() {
            if p.got == p.inp.len() && p.sent {
                continue;
            }
            let mut st = p.link.lock();
            let before = p.got;
            p.got += st.inq.take(&mut p.inp[p.got..]);
            let moved = if p.got == p.inp.len() && st.unsent() == 0 {
                // Complete: the other side may already have finished and closed the link.
                p.sent = true;
                false
            } else {
                let mut sink = Sink { buf: &mut p.inp[p.got..], filled: 0 };
                let r = st.pump(Instant::now(), &mut sink);
                p.got += sink.filled;
                p.sent = st.unsent() == 0;
                match r {
                    Ok(m) => m,
                    Err(_) if p.got == p.inp.len() && p.sent => true,
                    Err(e) => return Err(e).with_context(|| format!("exchange with {}", st.label)),
                }
            };
            busy |= moved || p.got > before;
            done &= p.got == p.inp.len() && p.sent;
            sent &= p.sent;
        }
        let now = Instant::now();
        if !all_sent && sent {
            all_sent = true;
            trip.sent = now - start;
        }
        if !any_read && parts.iter().any(|p| p.got > 0) {
            any_read = true;
            trip.first = now - start;
        }
        if done {
            trip.total = now - start;
            return Ok(trip);
        }
        if busy {
            last = now;
        } else if now - last > stall {
            let states: Vec<String> = parts.iter().map(|p| p.link.inner.lock().describe()).collect();
            bail!("lanes stalled for {stall:?}: {}", states.join("; "));
        } else {
            std::hint::spin_loop();
        }
    }
}

/// The pipeline chains' connection to the next or previous machine: frames `[u32 header length][u64 blob length]
/// [JSON header][blob]` over a link. A header with "error" is the other machine's failure.
#[derive(Clone)]
pub struct Conn {
    pub link: Link,
}

impl Conn {
    pub fn connect(addr: &str) -> Result<Self> {
        Ok(Self { link: Link::connect(addr, "chain")? })
    }

    pub fn send(&self, header: &Value, blob: &[u8]) -> Result<()> {
        let h = serde_json::to_vec(header)?;
        let mut lens = [0u8; 12];
        lens[..4].copy_from_slice(&(h.len() as u32).to_le_bytes());
        lens[4..].copy_from_slice(&(blob.len() as u64).to_le_bytes());
        self.link.send(&[&lens, &h, blob])?;
        Ok(())
    }

    pub fn recv(&self) -> Result<(Value, Vec<u8>)> {
        let mut len = [0u8; 12];
        self.link.read_exact(&mut len, Wait::Idle { spin: CONN_SPIN })?;
        let h = u32::from_le_bytes(len[..4].try_into()?) as usize;
        let b = u64::from_le_bytes(len[4..].try_into()?) as usize;
        let mut header = vec![0u8; h];
        self.link.read_exact(&mut header, Wait::Stall(STALL))?;
        let mut blob = vec![0u8; b];
        self.link.read_exact(&mut blob, Wait::Stall(STALL))?;
        let header: Value = serde_json::from_slice(&header)?;
        if let Some(e) = header.get("error") {
            bail!("remote: {}", e.as_str().unwrap_or("error"));
        }
        Ok((header, blob))
    }

    pub fn shutdown(&self) {
        self.link.shutdown();
    }
}
