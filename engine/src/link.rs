//! Byte-stream lanes to a neighbouring machine: one TCP connection per Thunderbolt link (or Ethernet), or one
//! USB4STREAM device (`/dev/tbstreamN`) per cable. An exchange cuts a message into one contiguous slice per
//! lane and moves the slices in parallel, so one, two or three cables carry it.
//!
//! Every lane is non-blocking: an exchange polls all lanes in both directions from the calling thread, so the
//! latency-critical path never sleeps and wakes. USB4STREAM has no Tx flow control on these controllers (see
//! `greet` and `exchange`), which the lanes make up for.
use anyhow::{bail, ensure, Context, Result};
use std::{
    fs::{File, OpenOptions},
    io::{self, ErrorKind, Read, Write},
    net::{TcpListener, TcpStream},
    os::unix::fs::OpenOptionsExt,
    time::{Duration, Instant},
};

/// O_NONBLOCK on x86-64 Linux.
const O_NONBLOCK: i32 = 0o4000;
/// A lane that makes no progress for this long has failed (the peer died or the link dropped).
const STALL: Duration = Duration::from_secs(30);
const MAGIC: u32 = 0x454e_414c;
const HELLO: u32 = 1;
const READY: u32 = 2;

pub enum Lane {
    Tcp(TcpStream),
    Stream(File),
}

impl Read for Lane {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Lane::Tcp(s) => s.read(buf),
            Lane::Stream(f) => f.read(buf),
        }
    }
}

impl Write for Lane {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Lane::Tcp(s) => s.write(buf),
            Lane::Stream(f) => f.write(buf),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Bytes moved by one non-blocking call; 0 when the lane would block.
fn step(r: io::Result<usize>) -> io::Result<usize> {
    match r {
        Ok(0) => Err(ErrorKind::UnexpectedEof.into()),
        Ok(n) => Ok(n),
        Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => Ok(0),
        Err(e) => Err(e),
    }
}

/// Write all of `buf`, spinning while the lane is full.
fn send(lane: &mut Lane, mut buf: &[u8]) -> io::Result<()> {
    let mut last = Instant::now();
    while !buf.is_empty() {
        match step(lane.write(buf))? {
            0 if last.elapsed() > STALL => return Err(ErrorKind::TimedOut.into()),
            0 => std::hint::spin_loop(),
            n => {
                buf = &buf[n..];
                last = Instant::now();
            }
        }
    }
    Ok(())
}

/// Fill `buf`, spinning while the lane is empty.
fn recv(lane: &mut Lane, mut buf: &mut [u8]) -> io::Result<()> {
    let mut last = Instant::now();
    while !buf.is_empty() {
        match step(lane.read(buf))? {
            0 if last.elapsed() > STALL => return Err(ErrorKind::TimedOut.into()),
            0 => std::hint::spin_loop(),
            n => {
                buf = &mut buf[n..];
                last = Instant::now();
            }
        }
    }
    Ok(())
}

fn record(kind: u32, index: usize, count: usize) -> [u8; 16] {
    let mut r = [0u8; 16];
    for (i, word) in [MAGIC, kind, index as u32, count as u32].into_iter().enumerate() {
        r[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
    }
    r
}

/// (kind, index, count) of a lane record.
fn parse(r: &[u8; 16]) -> Result<(u32, usize, usize)> {
    let word = |i: usize| u32::from_le_bytes(r[4 * i..4 * i + 4].try_into().unwrap());
    ensure!(word(0) == MAGIC, "lane handshake: unexpected bytes");
    Ok((word(1), word(2) as usize, word(3) as usize))
}

/// Open a stream lane. A USB4STREAM Tx ring runs without flow control (the Falcon Ridge NHI wedges with it,
/// kernel/patches-7.2/0005), so frames sent before the other end opens pile up and later burst past its receive
/// ring, dropping frames. Only the starting side (`wait` false) therefore repeats its hello, every 10 ms; the
/// waiting side answers the first hello with one `ready` and skips repeats until the starting side's `ready`.
/// Each side sends nothing else before its data. Records are written whole, so a lost frame loses whole records.
fn greet(lane: &mut Lane, index: usize, count: usize, wait: bool) -> Result<()> {
    let start = Instant::now();
    let mut hello: Option<Instant> = None;
    let mut answered = false;
    let mut r = [0u8; 16];
    let mut got = 0;
    loop {
        if !wait {
            ensure!(start.elapsed() < STALL, "no answer from the other machine");
            if hello.is_none_or(|t| t.elapsed() > Duration::from_millis(10)) {
                send(lane, &record(HELLO, index, count))?;
                hello = Some(Instant::now());
            }
        }
        match step(lane.read(&mut r[got..]))? {
            0 => std::thread::sleep(Duration::from_micros(100)),
            n => got += n,
        }
        if got < r.len() {
            continue;
        }
        got = 0;
        let (kind, i, n) = parse(&r)?;
        ensure!((i, n) == (index, count), "reached the other machine's lane {i} of {n}");
        match (wait, kind) {
            (true, HELLO) if !answered => {
                send(lane, &record(READY, index, count))?;
                answered = true;
            }
            (true, HELLO) => {}
            (true, READY) if answered => return Ok(()),
            (false, READY) => {
                send(lane, &record(READY, index, count))?;
                return Ok(());
            }
            _ => bail!("unexpected handshake record {kind}"),
        }
    }
}

fn tcp(s: TcpStream) -> Result<Lane> {
    s.set_nodelay(true)?;
    s.set_nonblocking(true)?;
    Ok(Lane::Tcp(s))
}

/// The lanes to one neighbour, in an order both machines agree on.
pub struct Lanes {
    lanes: Vec<Lane>,
    pub kind: &'static str,
}

impl Lanes {
    /// One TCP connection per address, in order. Each announces its index and the count, so the listener
    /// orders them the same way.
    pub fn connect(addrs: &[String]) -> Result<Self> {
        ensure!(!addrs.is_empty(), "no lanes");
        let mut lanes = Vec::new();
        for (i, a) in addrs.iter().enumerate() {
            let s = TcpStream::connect(a).with_context(|| format!("connecting lane {i} to {a}"))?;
            let mut lane = tcp(s)?;
            send(&mut lane, &record(HELLO, i, addrs.len()))?;
            lanes.push(lane);
        }
        Ok(Self { lanes, kind: "tcp" })
    }

    /// Accept the TCP lanes of one peer session.
    pub fn accept(listener: &TcpListener) -> Result<Self> {
        let mut slots: Vec<Option<Lane>> = Vec::new();
        loop {
            let (s, peer) = listener.accept()?;
            let mut lane = tcp(s)?;
            let mut r = [0u8; 16];
            recv(&mut lane, &mut r).with_context(|| format!("lane from {peer}"))?;
            let (_, i, n) = parse(&r)?;
            if slots.is_empty() {
                slots.resize_with(n, || None);
            }
            ensure!(n == slots.len() && i < n && slots[i].is_none(), "unexpected lane {i} of {n} from {peer}");
            slots[i] = Some(lane);
            if slots.iter().all(Option::is_some) {
                return Ok(Self { lanes: slots.into_iter().map(Option::unwrap).collect(), kind: "tcp" });
            }
        }
    }

    /// USB4STREAM devices, one per cable, listed in the same cable order on both machines. A stream only
    /// carries frames while both ends have it open, so the lanes start with a handshake (see `greet`), which
    /// also catches a lane order that differs between the machines. With `wait`, this side waits for the
    /// other machine indefinitely; the other side must start the session.
    pub fn streams(paths: &[String], wait: bool) -> Result<Self> {
        ensure!(!paths.is_empty(), "no lanes");
        let mut lanes = Vec::new();
        for p in paths {
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(O_NONBLOCK)
                .open(p)
                .with_context(|| format!("opening {p}"))?;
            lanes.push(Lane::Stream(f));
        }
        let n = lanes.len();
        for (i, lane) in lanes.iter_mut().enumerate() {
            greet(lane, i, n, wait).with_context(|| format!("stream lane {i} ({})", paths[i]))?;
        }
        Ok(Self { lanes, kind: "stream" })
    }

    pub fn count(&self) -> usize {
        self.lanes.len()
    }

    /// Send a small control message on the first lane: a length and the bytes.
    pub fn send_message(&mut self, bytes: &[u8]) -> Result<()> {
        send(&mut self.lanes[0], &(bytes.len() as u32).to_le_bytes())?;
        send(&mut self.lanes[0], bytes)?;
        Ok(())
    }

    pub fn recv_message(&mut self) -> Result<Vec<u8>> {
        let mut len = [0u8; 4];
        recv(&mut self.lanes[0], &mut len)?;
        let mut bytes = vec![0u8; u32::from_le_bytes(len) as usize];
        recv(&mut self.lanes[0], &mut bytes)?;
        Ok(bytes)
    }

    /// Send `out` to the peer and receive as many bytes into `inp`; the peer makes the matching call. Lane `i`
    /// carries the `i`th of `count` equal, 4-byte aligned slices, and every lane moves both ways at once.
    /// Without Tx flow control a stream must never have more in flight than the other end's receive ring
    /// holds (256 frames of 4 KiB), so stream lanes move at most ROUND bytes each way per round, and a round
    /// ends only when its bytes have arrived from the other machine.
    pub fn exchange(&mut self, out: &[u8], inp: &mut [u8]) -> Result<()> {
        ensure!(out.len() == inp.len() && out.len().is_multiple_of(4), "exchange sizes must match and align to 4 bytes");
        const ROUND: usize = 256 << 10;
        let n = self.lanes.len();
        let cuts: Vec<usize> = (0..=n).map(|i| out.len() / 4 * i / n * 4).collect();
        let round = if self.kind == "stream" { ROUND } else { usize::MAX };
        let mut begin: Vec<usize> = cuts[..n].to_vec();
        while begin != cuts[1..] {
            let ends: Vec<usize> = (0..n).map(|i| cuts[i + 1].min(begin[i].saturating_add(round))).collect();
            self.exchange_round(out, inp, &begin, &ends)?;
            begin = ends;
        }
        Ok(())
    }

    /// Move `out[begin[i]..end[i]]` out and into `inp` at the same offsets on every lane `i`.
    fn exchange_round(&mut self, out: &[u8], inp: &mut [u8], begin: &[usize], end: &[usize]) -> Result<()> {
        let mut sent = begin.to_vec();
        let mut read = begin.to_vec();
        let mut last = Instant::now();
        loop {
            let mut busy = false;
            for (i, lane) in self.lanes.iter_mut().enumerate() {
                if sent[i] < end[i] {
                    let k = step(lane.write(&out[sent[i]..end[i]]))?;
                    sent[i] += k;
                    busy |= k > 0;
                }
                if read[i] < end[i] {
                    let k = step(lane.read(&mut inp[read[i]..end[i]]))?;
                    read[i] += k;
                    busy |= k > 0;
                }
            }
            if sent == end && read == end {
                return Ok(());
            }
            if busy {
                last = Instant::now();
            } else if last.elapsed() > STALL {
                bail!("lanes stalled for {STALL:?}");
            } else {
                std::hint::spin_loop();
            }
        }
    }
}
