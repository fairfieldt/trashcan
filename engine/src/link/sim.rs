//! A simulated USB4STREAM cable for tests, with the driver's behaviour (device.rs) and injected faults:
//! - a write is cut into frames of up to 4 KiB; a frame that finds the other end's ring (`ring` frames) full is
//!   lost; `read` returns frames back to back and keeps the rest of a frame it could not fit;
//! - frames written while the other end is closed wait (up to 255) and arrive when it opens, even after the writer
//!   closed (the worst case for stale data);
//! - closing an end sends a CLOSE, which the other end reads once as end of file;
//! - faults: drop, damage or cut one frame in N (at random: a fixed period can line up with the sender's
//!   retransmissions and hit the same record every time), lose everything (`dead`), refuse to open, and `hold`: the last
//!   `hold` frames to arrive are not read until more follow them (a direction of a cable between Mac Pros holds up
//!   to three, docs/kernel.md); `hold_at` gives each receiving end its own;
//! - an end opened while the same end is still open is turned away at once (`AddrInUse`), as the kernel device's
//!   `flock` and tbd do.
use super::device::{Found, Frames, Streams};
use std::{
    collections::VecDeque,
    io::{self, ErrorKind, IoSlice},
    net::SocketAddr,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Faults {
    pub drop_every: u64,
    pub corrupt_every: u64,
    pub truncate_every: u64,
    pub dead: bool,
    pub refuse_open: bool,
    pub hold: usize,
    /// Frames held at each receiving end (side 0, side 1), where that is more than `hold`.
    pub hold_at: [usize; 2],
    /// Opening this side's end waits this long (as an open waits for an earlier close of the device).
    pub slow_open: Option<(usize, std::time::Duration)>,
    /// Closing this side's end takes this long to begin: the end stays open meanwhile (a link closes its device on a
    /// thread of its own, which a busy machine can run late).
    pub slow_close: Option<(usize, std::time::Duration)>,
    /// This side's first open reads nothing of what the other side writes (one direction of an epoch dead, as seen
    /// on the Mac Pros); its next open works.
    pub deaf_once: Option<usize>,
    /// (side, n): in this side's first open, its frames after the first n are lost (its direction of the epoch goes
    /// dead after a few frames, as the main suite saw); its next open works.
    pub mute_once: Option<(usize, u64)>,
}

enum Frame {
    Data(Vec<u8>),
    Close,
}

#[derive(Default)]
struct Dir {
    ring: VecDeque<Frame>,
    offset: usize,
    pile: VecDeque<Frame>,
    count: u64,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Counts {
    pub frames: u64,
    pub dropped: u64,
    pub damaged: u64,
    pub overrun: u64,
}

pub struct Wire {
    dirs: [Dir; 2],
    open: [bool; 2],
    ring: usize,
    pub faults: Faults,
    pub counts: Counts,
    /// Opens of the `deaf_once` side so far.
    deaf_opens: u32,
    /// Opens of the `mute_once` side so far, and frames each side has written since it opened.
    mute_opens: u32,
    wrote: [u64; 2],
}

/// The cable; `end(side)` opens one end.
pub struct Pair(Mutex<Wire>);

impl Pair {
    pub fn new(ring: usize) -> Arc<Pair> {
        Arc::new(Pair(Mutex::new(Wire {
            dirs: Default::default(),
            open: [false; 2],
            ring,
            faults: Faults::default(),
            counts: Counts::default(),
            deaf_opens: 0,
            mute_opens: 0,
            wrote: [0; 2],
        })))
    }

    pub fn set_faults(&self, f: Faults) {
        self.0.lock().unwrap().faults = f;
    }

    pub fn counts(&self) -> Counts {
        self.0.lock().unwrap().counts
    }

    /// Open end `side`; what waited for it arrives (and is lost beyond the ring).
    pub fn end(self: &Arc<Self>, side: usize) -> io::Result<End> {
        let slow = self.0.lock().unwrap().faults.slow_open.filter(|(s, _)| *s == side);
        if let Some((_, wait)) = slow {
            std::thread::sleep(wait);
        }
        let mut w = self.0.lock().unwrap();
        if w.faults.refuse_open {
            return Err(io::Error::other("simulated open failure"));
        }
        if w.open[side] {
            return Err(io::Error::new(ErrorKind::AddrInUse, "simulated device in use"));
        }
        if w.faults.deaf_once == Some(side) {
            w.deaf_opens += 1;
            if w.deaf_opens > 1 {
                w.faults.deaf_once = None;
            }
        }
        if w.faults.mute_once.is_some_and(|(s, _)| s == side) {
            w.mute_opens += 1;
            if w.mute_opens > 1 {
                w.faults.mute_once = None;
            }
        }
        w.wrote[side] = 0;
        w.open[side] = true;
        let ring = w.ring;
        let d = &mut w.dirs[side];
        d.ring.clear();
        d.offset = 0;
        while let Some(f) = d.pile.pop_front() {
            if d.ring.len() < ring {
                d.ring.push_back(f);
            }
        }
        Ok(End { pair: self.clone(), side })
    }

    /// Frames written toward `side` while it is closed, as if an earlier session had sent them.
    pub fn stale(&self, side: usize, bytes: &[u8]) {
        let mut w = self.0.lock().unwrap();
        for c in bytes.chunks(4096) {
            w.dirs[side].pile.push_back(Frame::Data(c.to_vec()));
        }
    }

    /// End of file at `side`, as if the other end had closed and reopened.
    pub fn close_from(&self, side: usize) {
        let mut w = self.0.lock().unwrap();
        w.dirs[side].ring.push_back(Frame::Close);
    }
}

/// Frame `k` is hit by a fault of rate one in `n` (0: never), pseudo-randomly (`salt` tells the faults apart).
fn one_in(n: u64, k: u64, salt: u64) -> bool {
    let mut x = k.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 31;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 29;
    n > 0 && x.is_multiple_of(n)
}

pub struct End {
    pair: Arc<Pair>,
    side: usize,
}

impl Frames for End {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut w = self.pair.0.lock().unwrap();
        let hold = w.faults.hold.max(w.faults.hold_at[self.side]);
        let d = &mut w.dirs[self.side];
        let mut n = 0;
        while n < buf.len() {
            if d.ring.len() <= hold {
                break;
            }
            match d.ring.front() {
                None => break,
                Some(Frame::Close) => {
                    if n == 0 {
                        d.ring.pop_front();
                        return Ok(0);
                    }
                    break;
                }
                Some(Frame::Data(f)) => {
                    let k = (f.len() - d.offset).min(buf.len() - n);
                    buf[n..n + k].copy_from_slice(&f[d.offset..d.offset + k]);
                    n += k;
                    d.offset += k;
                    if d.offset == f.len() {
                        d.ring.pop_front();
                        d.offset = 0;
                    }
                }
            }
        }
        if n == 0 {
            return Err(ErrorKind::WouldBlock.into());
        }
        Ok(n)
    }

    fn writev(&mut self, bufs: &[IoSlice]) -> io::Result<usize> {
        let mut all = Vec::new();
        for b in bufs {
            all.extend_from_slice(b);
        }
        let mut w = self.pair.0.lock().unwrap();
        let w = &mut *w;
        let peer = 1 - self.side;
        let mut taken = 0;
        for chunk in all.chunks(4096) {
            let d = &mut w.dirs[peer];
            d.count += 1;
            w.counts.frames += 1;
            let k = d.count;
            let f = w.faults;
            w.wrote[self.side] += 1;
            let muted = f.mute_once.is_some_and(|(s, n)| s == self.side && w.wrote[self.side] > n);
            if f.dead || muted || f.deaf_once == Some(peer) || one_in(f.drop_every, k, 1) {
                w.counts.dropped += 1;
                taken += chunk.len();
                continue;
            }
            let mut frame = chunk.to_vec();
            if one_in(f.corrupt_every, k, 2) {
                let i = (k as usize * 7919) % frame.len();
                frame[i] ^= 0x5a;
                w.counts.damaged += 1;
            }
            if one_in(f.truncate_every, k, 3) && frame.len() > 1 {
                frame.truncate(frame.len() / 2);
                w.counts.damaged += 1;
            }
            if !w.open[peer] {
                if d.pile.len() >= 255 {
                    break;
                }
                d.pile.push_back(Frame::Data(frame));
            } else if d.ring.len() < w.ring {
                d.ring.push_back(Frame::Data(frame));
            } else {
                w.counts.overrun += 1;
            }
            taken += chunk.len();
        }
        if taken == 0 && !all.is_empty() {
            return Err(ErrorKind::WouldBlock.into());
        }
        Ok(taken)
    }
}

impl Drop for End {
    fn drop(&mut self) {
        let slow = self.pair.0.lock().unwrap().faults.slow_close.filter(|(s, _)| *s == self.side);
        if let Some((_, wait)) = slow {
            std::thread::sleep(wait);
        }
        let mut w = self.pair.0.lock().unwrap();
        w.open[self.side] = false;
        let peer = 1 - self.side;
        let d = &mut w.dirs[self.side];
        d.ring.clear();
        d.offset = 0;
        if w.open[peer] {
            w.dirs[peer].ring.push_back(Frame::Close);
        } else {
            w.dirs[peer].pile.push_back(Frame::Close);
        }
    }
}

/// The streams of one end of a simulated cable.
pub struct SimStreams {
    pub pair: Arc<Pair>,
    pub side: usize,
    pub ring: u32,
}

impl Streams for SimStreams {
    fn find(&self, _: &SocketAddr) -> Result<Found, String> {
        Ok(Found {
            path: format!("sim{}", self.side),
            ring: self.ring,
            local: format!("controller-{}", self.side),
            peer: format!("controller-{}", 1 - self.side),
            label: format!("sim end {}", self.side),
            polled: true,
        })
    }

    fn open(&self, _: &Found) -> io::Result<Box<dyn Frames>> {
        Ok(Box::new(self.pair.end(self.side)?))
    }
}
