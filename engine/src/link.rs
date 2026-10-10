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
pub const STALL: Duration = Duration::from_secs(30);
/// An idle wait for the next message (`Lanes::recv_message_idle`) polls without pausing this long, since the
/// messages of a busy peer follow each other closely, then pauses this long between polls.
const BUSY: Duration = Duration::from_millis(50);
const NAP: Duration = Duration::from_millis(1);
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

/// When an exchange had issued all its sends and when its first incoming byte arrived, as time since its start:
/// the one-way trip to the other machine plus its lateness, and the transfer that follows (`Mesh` tallies them).
#[derive(Clone, Copy, Debug, Default)]
pub struct Trip {
    pub sent: Duration,
    pub first: Duration,
    pub total: Duration,
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

    /// `recv_message` for a message that may come hours later (the next request to a server): no deadline for
    /// its first bytes, polled in short sleeps once the lane has been quiet for `BUSY`, so an idle machine leaves
    /// its cores alone; the rest must then arrive within `STALL`. The end of the lane (the other side closed it)
    /// or a reset is an error, as is a lane handshake (the other side started a new session).
    pub fn recv_message_idle(&mut self) -> Result<Vec<u8>> {
        self.recv_message_waiting(None, BUSY)
    }

    /// Wait at most `first_byte` for a message to start, sleeping between polls. Used for model readiness,
    /// whose cold disk reads may exceed the active protocol's `STALL`; once the first byte arrives, the
    /// remaining header and body keep the ordinary progress deadline. EOF/reset still fails immediately.
    pub fn recv_message_first_within(&mut self, first_byte: Duration) -> Result<Vec<u8>> {
        self.recv_message_waiting(Some(first_byte), Duration::ZERO)
    }

    fn recv_message_waiting(&mut self, first_byte: Option<Duration>, busy: Duration) -> Result<Vec<u8>> {
        let lane = &mut self.lanes[0];
        let mut len = [0u8; 4];
        let start = Instant::now();
        let got = loop {
            match step(lane.read(&mut len)).context("the other machine closed the lanes")? {
                0 if first_byte.is_some_and(|limit| start.elapsed() >= limit) => {
                    return Err(io::Error::from(ErrorKind::TimedOut)).context("waiting for the other machine's first message byte");
                }
                0 if start.elapsed() < busy => std::hint::spin_loop(),
                0 => std::thread::sleep(NAP),
                n => break n,
            }
        };
        recv(lane, &mut len[got..])?;
        if u32::from_le_bytes(len) == MAGIC {
            // Take the rest of the hello, so the next session's handshake reads whole records.
            recv(lane, &mut [0u8; 12])?;
            bail!("the other machine started a new session");
        }
        let mut bytes = vec![0u8; u32::from_le_bytes(len) as usize];
        recv(lane, &mut bytes)?;
        Ok(bytes)
    }

    /// Send `out` to the peer and receive as many bytes into `inp`; the peer makes the matching call. Lane `i`
    /// carries the `i`th of `count` equal, 4-byte aligned slices, and every lane moves both ways at once.
    /// Without Tx flow control a stream must never have more in flight than the other end's receive ring
    /// holds (256 frames of 4 KiB), so stream lanes move at most ROUND bytes each way per round, and a round
    /// ends only when its bytes have arrived from the other machine.
    pub fn exchange(&mut self, out: &[u8], inp: &mut [u8]) -> Result<()> {
        self.exchange_within(out, inp, STALL).map(|_| ())
    }

    /// `exchange`, failing when the lanes make no progress for `stall`.
    pub fn exchange_within(&mut self, out: &[u8], inp: &mut [u8], stall: Duration) -> Result<Trip> {
        ensure!(out.len() == inp.len() && out.len().is_multiple_of(4), "exchange sizes must match and align to 4 bytes");
        const ROUND: usize = 256 << 10;
        let n = self.lanes.len();
        let cuts: Vec<usize> = (0..=n).map(|i| out.len() / 4 * i / n * 4).collect();
        let round = if self.kind == "stream" { ROUND } else { usize::MAX };
        let mut begin: Vec<usize> = cuts[..n].to_vec();
        let start = Instant::now();
        let mut trip = Trip::default();
        while begin != cuts[1..] {
            let ends: Vec<usize> = (0..n).map(|i| cuts[i + 1].min(begin[i].saturating_add(round))).collect();
            let t = self.exchange_round(out, inp, &begin, &ends, stall)?;
            if begin == cuts[..n] {
                trip = t;
            }
            begin = ends;
        }
        trip.total = start.elapsed();
        Ok(trip)
    }

    /// Move `out[begin[i]..end[i]]` out and into `inp` at the same offsets on every lane `i`.
    fn exchange_round(&mut self, out: &[u8], inp: &mut [u8], begin: &[usize], end: &[usize], stall: Duration) -> Result<Trip> {
        let mut sent = begin.to_vec();
        let mut read = begin.to_vec();
        let start = Instant::now();
        let mut last = start;
        let mut trip = Trip::default();
        let (mut all_sent, mut any_read) = (false, false);
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
            if !all_sent && sent == end {
                all_sent = true;
                trip.sent = start.elapsed();
            }
            if !any_read && read != begin {
                any_read = true;
                trip.first = start.elapsed();
            }
            if sent == end && read == end {
                trip.total = start.elapsed();
                return Ok(trip);
            }
            if busy {
                last = Instant::now();
            } else if last.elapsed() > stall {
                bail!("lanes stalled for {stall:?}");
            } else {
                std::hint::spin_loop();
            }
        }
    }
}

/// Several `exchange`s at once, each with the peer of its lanes, polled together from the calling thread so that the
/// cables to different machines carry their slices at the same time. Each lane moves its slice in rounds of at most
/// ROUND bytes each way on streams (see `exchange`); a lane's round ends only when its bytes have arrived. One
/// exchange alone is `Lanes::exchange_within`, which keeps the lanes of its rounds in step (measured 0.6% faster for
/// a TP4 decode than this loop, results/qwen/tp8-20261007).
pub fn exchange_all(jobs: &mut [(&mut Lanes, &[u8], &mut [u8])], stall: Duration) -> Result<Trip> {
    for (_, out, inp) in jobs.iter() {
        ensure!(out.len() == inp.len() && out.len().is_multiple_of(4), "exchange sizes must match and align to 4 bytes");
    }
    const ROUND: usize = 256 << 10;
    /// One lane's slice: bytes `..end` of it may move in the current round.
    struct Part<'a> {
        lane: &'a mut Lane,
        out: &'a [u8],
        inp: &'a mut [u8],
        round: usize,
        end: usize,
        sent: usize,
        read: usize,
    }
    let mut parts = Vec::new();
    for (lanes, out, inp) in jobs.iter_mut() {
        let n = lanes.lanes.len();
        let round = if lanes.kind == "stream" { ROUND } else { usize::MAX };
        let cuts: Vec<usize> = (0..=n).map(|i| out.len() / 4 * i / n * 4).collect();
        let mut rest: &mut [u8] = inp;
        for (i, lane) in lanes.lanes.iter_mut().enumerate() {
            let (chunk, tail) = std::mem::take(&mut rest).split_at_mut(cuts[i + 1] - cuts[i]);
            rest = tail;
            let len = chunk.len();
            parts.push(Part { lane, out: &out[cuts[i]..cuts[i + 1]], inp: chunk, round, end: len.min(round), sent: 0, read: 0 });
        }
    }
    let start = Instant::now();
    let mut last = start;
    let mut trip = Trip::default();
    let (mut all_sent, mut any_read) = (false, false);
    loop {
        let (mut busy, mut done, mut sent, mut read) = (false, true, true, false);
        // Every lane's send first, then the reads: the sends go out back to back.
        for p in parts.iter_mut() {
            if p.sent < p.end {
                let k = step(p.lane.write(&p.out[p.sent..p.end]))?;
                p.sent += k;
                busy |= k > 0;
            }
        }
        for p in parts.iter_mut() {
            if p.read < p.end {
                let k = step(p.lane.read(&mut p.inp[p.read..p.end]))?;
                p.read += k;
                busy |= k > 0;
            }
            if p.sent == p.end && p.read == p.end && p.end < p.out.len() {
                p.end = p.out.len().min(p.end.saturating_add(p.round));
                busy = true;
            }
            done &= p.read == p.out.len() && p.sent == p.out.len();
            // The first round's sends issued, and the first byte of any lane.
            sent &= p.sent >= p.out.len().min(p.round);
            read |= p.read > 0;
        }
        if !all_sent && sent {
            all_sent = true;
            trip.sent = start.elapsed();
        }
        if !any_read && read {
            any_read = true;
            trip.first = start.elapsed();
        }
        if done {
            trip.total = start.elapsed();
            return Ok(trip);
        }
        if busy {
            last = Instant::now();
        } else if last.elapsed() > stall {
            bail!("lanes stalled for {stall:?}");
        } else {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn pair(count: usize, stream_rounds: bool) -> (Lanes, Lanes) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let accepted = thread::spawn(move || Lanes::accept(&listener).unwrap());
        let mut a = Lanes::connect(&vec![address; count]).unwrap();
        let mut b = accepted.join().unwrap();
        if stream_rounds {
            // Exercise the USB4STREAM round boundaries over local sockets. This does not simulate the
            // device's receive ring or measure USB4STREAM throughput.
            a.kind = "stream";
            b.kind = "stream";
        }
        (a, b)
    }

    fn payload(len: usize, side: usize, iteration: usize) -> Vec<u8> {
        (0..len).map(|i| (i.wrapping_mul(13) + side * 97 + iteration * 31) as u8).collect()
    }

    #[test]
    fn message_first_within_accepts_delayed_ready_and_preserves_framing() {
        let (mut a, mut b) = pair(1, false);
        let peer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(75));
            b.send_message(br#"{"ok":true}"#).unwrap();
            b.send_message(b"next").unwrap();
        });
        assert_eq!(a.recv_message_first_within(Duration::from_secs(2)).unwrap(), br#"{"ok":true}"#);
        assert_eq!(a.recv_message().unwrap(), b"next");
        peer.join().unwrap();
    }

    #[test]
    fn message_first_within_times_out_without_consuming_a_later_message() {
        let (mut a, mut b) = pair(1, false);
        let limit = Duration::from_millis(20);
        let start = Instant::now();
        let error = a.recv_message_first_within(limit).unwrap_err();
        assert_eq!(error.downcast_ref::<io::Error>().unwrap().kind(), ErrorKind::TimedOut);
        assert!(start.elapsed() >= limit);
        assert!(start.elapsed() < Duration::from_secs(2));
        b.send_message(b"ready").unwrap();
        assert_eq!(a.recv_message_first_within(Duration::from_secs(2)).unwrap(), b"ready");
    }

    #[test]
    fn message_first_within_does_not_apply_start_deadline_to_header_or_body() {
        let (mut a, mut b) = pair(1, false);
        let header = 5u32.to_le_bytes();
        send(&mut b.lanes[0], &header[..1]).unwrap();
        let peer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            send(&mut b.lanes[0], &header[1..]).unwrap();
            send(&mut b.lanes[0], b"re").unwrap();
            thread::sleep(Duration::from_millis(50));
            send(&mut b.lanes[0], b"ady").unwrap();
        });
        // The first byte is already buffered; both later pauses exceed this start deadline.
        assert_eq!(a.recv_message_first_within(Duration::from_millis(10)).unwrap(), b"ready");
        peer.join().unwrap();
    }

    #[test]
    fn message_first_within_preserves_eof_at_each_framing_stage() {
        let header = 5u32.to_le_bytes();
        for prefix in [vec![], header[..1].to_vec(), [header.as_slice(), b"re"].concat()] {
            let (mut a, mut b) = pair(1, false);
            send(&mut b.lanes[0], &prefix).unwrap();
            drop(b);
            let start = Instant::now();
            let error = a.recv_message_first_within(Duration::from_secs(300)).unwrap_err();
            assert_eq!(error.downcast_ref::<io::Error>().unwrap().kind(), ErrorKind::UnexpectedEof);
            assert!(start.elapsed() < Duration::from_secs(2));
        }
    }

    #[test]
    fn message_first_within_rejects_a_peer_restart() {
        let (mut a, mut b) = pair(1, false);
        send(&mut b.lanes[0], &record(HELLO, 0, 1)).unwrap();
        assert!(a.recv_message_first_within(Duration::from_secs(2)).unwrap_err().to_string().contains("new session"));
    }

    #[test]
    fn message_idle_still_accepts_delayed_requests() {
        let (mut a, mut b) = pair(1, false);
        let peer = thread::spawn(move || {
            thread::sleep(BUSY + Duration::from_millis(25));
            b.send_message(b"request").unwrap();
        });
        assert_eq!(a.recv_message_idle().unwrap(), b"request");
        peer.join().unwrap();
    }

    #[test]
    fn exchanges_preserve_round_boundaries_and_control_messages() {
        // Empty lane slices, unequal slices, decode/verification shapes, exact round boundaries, and
        // multi-round prefill; sizes change within a session and control messages follow each exchange.
        let lengths = [0, 4, 8, 20, 5120 * 4, 3 * 5120 * 4, 3 * (256 << 10), 3 * (256 << 10) + 4, 256 * 5120 * 4, 4];
        for count in [1, 3, 5] {
            for stream_rounds in [false, true] {
                let (mut a, mut b) = pair(count, stream_rounds);
                thread::scope(|s| {
                    let peer = s.spawn(|| {
                        for (iteration, len) in lengths.into_iter().enumerate() {
                            let out = payload(len, 1, iteration);
                            let mut inp = vec![0; len];
                            b.exchange(&out, &mut inp).unwrap();
                            assert_eq!(inp, payload(len, 0, iteration));
                            assert_eq!(b.recv_message().unwrap(), iteration.to_le_bytes());
                            b.send_message(&iteration.to_le_bytes()).unwrap();
                        }
                    });
                    for (iteration, len) in lengths.into_iter().enumerate() {
                        let out = payload(len, 0, iteration);
                        let mut inp = vec![0; len];
                        a.exchange(&out, &mut inp).unwrap();
                        assert_eq!(inp, payload(len, 1, iteration));
                        a.send_message(&iteration.to_le_bytes()).unwrap();
                        assert_eq!(a.recv_message().unwrap(), iteration.to_le_bytes());
                    }
                    peer.join().unwrap();
                });
            }
        }
    }

    #[test]
    fn exchange_errors_and_retry() {
        let (mut a, mut b) = pair(3, false);
        assert!(a.exchange(&[0; 3], &mut [0; 3]).is_err());
        assert!(a.exchange(&[0; 4], &mut [0; 8]).is_err());
        // Failed validation must not write anything, so a valid exchange on the same lanes still works.
        thread::scope(|s| {
            let peer = s.spawn(|| b.exchange(&[9; 4], &mut [0; 4]).unwrap());
            let mut inp = [0; 4];
            a.exchange(&[7; 4], &mut inp).unwrap();
            assert_eq!(inp, [9; 4]);
            peer.join().unwrap();
        });
        let error = a.exchange_within(&[0; 4], &mut [0; 4], Duration::from_millis(10)).unwrap_err();
        assert!(error.to_string().contains("stalled"));
        drop(b);
        assert!(a.exchange(&[0; 4], &mut [0; 4]).is_err());
    }
}
