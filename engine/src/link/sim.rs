//! A simulated USB4STREAM cable for tests, with the driver's behaviour (device.rs) and injected faults:
//! - a write is cut into frames of up to 4 KiB; a frame that finds the other end's ring (`ring` frames) full is
//!   lost; `read` returns frames back to back and keeps the rest of a frame it could not fit;
//! - frames written while the other end is closed wait (up to 255) and arrive when it opens, even after the writer
//!   closed (the worst case for stale data);
//! - closing an end sends a CLOSE, which the other end reads once as end of file;
//! - faults: drop, damage or cut every Nth frame, lose everything (`dead`), refuse to open, and `hold`: the last
//!   `hold` frames to arrive are not read until more follow them (the Mac Pros' NHIs hold up to three, results/kernel/
//!   usb4stream-reliable-20261013).
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
}

/// The cable; `end(side)` opens one end.
pub struct Pair(Mutex<Wire>);

impl Pair {
    pub fn new(ring: usize) -> Arc<Pair> {
        Arc::new(Pair(Mutex::new(Wire { dirs: Default::default(), open: [false; 2], ring, faults: Faults::default(), counts: Counts::default() })))
    }

    pub fn set_faults(&self, f: Faults) {
        self.0.lock().unwrap().faults = f;
    }

    pub fn counts(&self) -> Counts {
        self.0.lock().unwrap().counts
    }

    /// Open end `side`; what waited for it arrives (and is lost beyond the ring).
    pub fn end(self: &Arc<Self>, side: usize) -> io::Result<End> {
        let mut w = self.0.lock().unwrap();
        if w.faults.refuse_open {
            return Err(io::Error::other("simulated open failure"));
        }
        if w.open[side] {
            return Err(io::Error::new(ErrorKind::AddrInUse, "simulated device in use"));
        }
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

pub struct End {
    pair: Arc<Pair>,
    side: usize,
}

impl Frames for End {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut w = self.pair.0.lock().unwrap();
        let hold = w.faults.hold;
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
            if f.dead || (f.drop_every > 0 && k.is_multiple_of(f.drop_every)) {
                w.counts.dropped += 1;
                taken += chunk.len();
                continue;
            }
            let mut frame = chunk.to_vec();
            if f.corrupt_every > 0 && k.is_multiple_of(f.corrupt_every) {
                let i = (k as usize * 7919) % frame.len();
                frame[i] ^= 0x5a;
                w.counts.damaged += 1;
            }
            if f.truncate_every > 0 && k.is_multiple_of(f.truncate_every) && frame.len() > 1 {
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
        })
    }

    fn open(&self, _: &Found) -> io::Result<Box<dyn Frames>> {
        Ok(Box::new(self.pair.end(self.side)?))
    }
}
