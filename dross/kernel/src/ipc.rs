//! Channels and notifications.
//!
//! A channel is a pair of ends; each end reads the messages the other writes (bytes and handles, which move
//! with the message). Writes never block (a full queue is an error); reads block until a message, the peer
//! closing, or a deadline. A notification is a word of bits: signalling ORs bits in, waiting takes them all. A
//! server waits on one notification that its channels (`bind`) and interrupts signal.

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use dross_abi::{Error, CHANNEL_DEPTH};

use crate::object::Entry;
use crate::sched::{WaitCell, WaitError};

pub struct Notification {
    bits: WaitCell<u64>,
}

impl Notification {
    pub fn new() -> Notification {
        Notification { bits: WaitCell::new(0) }
    }

    pub fn signal(&self, bits: u64) {
        if bits != 0 {
            self.bits.update(|b| *b |= bits);
        }
    }

    /// For poll on a notification fd: POLLIN while bits are pending.
    pub fn poll(&self, me: &alloc::sync::Arc<crate::sched::Thread>) -> u16 {
        self.bits.poll(me, |b| if *b != 0 { 1 } else { 0 })
    }

    pub fn unpoll(&self, me: &alloc::sync::Arc<crate::sched::Thread>) {
        self.bits.unregister(me);
    }

    pub fn wait(&self, deadline_tsc: u64) -> Result<u64, WaitError> {
        self.bits.wait(deadline_tsc, |b| if *b != 0 { Some(core::mem::take(b)) } else { None })
    }
}

pub struct Message {
    pub bytes: Vec<u8>,
    pub handles: Vec<Entry>,
}

struct Side {
    queue: VecDeque<Message>,
    bound: Option<(Arc<Notification>, u64)>,
}

struct Pair {
    /// Messages waiting for end i.
    sides: [WaitCell<Side>; 2],
    closed: [AtomicBool; 2],
}

pub struct ChannelEnd {
    pair: Arc<Pair>,
    side: usize,
}

pub enum ReadError {
    Wait(WaitError),
    PeerClosed,
    /// Message bytes and handle count of the waiting message.
    TooSmall(usize, usize),
}

impl ChannelEnd {
    pub fn pair() -> (ChannelEnd, ChannelEnd) {
        let side = || WaitCell::new(Side { queue: VecDeque::new(), bound: None });
        let pair = Arc::new(Pair { sides: [side(), side()], closed: [AtomicBool::new(false), AtomicBool::new(false)] });
        (ChannelEnd { pair: pair.clone(), side: 0 }, ChannelEnd { pair, side: 1 })
    }

    fn peer(&self) -> usize {
        1 - self.side
    }

    pub fn write(&self, m: Message) -> Result<(), (Error, Message)> {
        if self.pair.closed[self.peer()].load(Ordering::Acquire) {
            return Err((Error::PeerClosed, m));
        }
        let mut m = Some(m);
        let bound = self.pair.sides[self.peer()].update(|s| {
            if s.queue.len() >= CHANNEL_DEPTH {
                return Err(());
            }
            s.queue.push_back(m.take().unwrap());
            Ok(s.bound.clone())
        });
        match bound {
            Ok(Some((n, bits))) => n.signal(bits),
            Ok(None) => {}
            Err(()) => return Err((Error::ShouldWait, m.unwrap())),
        }
        Ok(())
    }

    /// Takes the next message if it fits in `max_bytes` and `max_handles`.
    pub fn read(&self, max_bytes: usize, max_handles: usize, deadline_tsc: u64) -> Result<Message, ReadError> {
        let closed = &self.pair.closed[self.peer()];
        let r = self.pair.sides[self.side].wait(deadline_tsc, |s| match s.queue.front() {
            Some(m) if m.bytes.len() > max_bytes || m.handles.len() > max_handles => Some(Err(ReadError::TooSmall(m.bytes.len(), m.handles.len()))),
            Some(_) => Some(Ok(s.queue.pop_front().unwrap())),
            None if closed.load(Ordering::Acquire) => Some(Err(ReadError::PeerClosed)),
            None => None,
        });
        match r {
            Ok(r) => r,
            Err(e) => Err(ReadError::Wait(e)),
        }
    }

    /// Internal RPC receive: consume the response before exposing ordinary signals.
    #[cfg(feature = "unix")]
    pub fn read_transaction(&self, max_bytes: usize, max_handles: usize, deadline: u64) -> Result<Message, ReadError> {
        let closed = &self.pair.closed[self.peer()];
        self.pair.sides[self.side].wait_uninterruptible(deadline, |s| match s.queue.front() {
            Some(m) if m.bytes.len() > max_bytes || m.handles.len() > max_handles => Some(Err(ReadError::TooSmall(m.bytes.len(), m.handles.len()))),
            Some(_) => Some(Ok(s.queue.pop_front().unwrap())),
            None if closed.load(Ordering::Acquire) => Some(Err(ReadError::PeerClosed)),
            None => None,
        }).map_err(ReadError::Wait)?
    }

    #[cfg(feature = "unix")]
    pub fn readiness(&self) -> u16 {
        let readable = self.pair.sides[self.side].with(|s| !s.queue.is_empty());
        let closed = self.pair.closed[self.peer()].load(Ordering::Acquire);
        let writable = !closed && self.pair.sides[self.peer()].with(|s| s.queue.len() < CHANNEL_DEPTH);
        (if readable || closed { 1 } else { 0 }) | (if writable { 4 } else { 0 }) | (if closed { 16 } else { 0 })
    }

    /// Signals `bits` on `n` whenever a message arrives at this end or the peer closes (now too, if one waits).
    pub fn bind(&self, n: Arc<Notification>, bits: u64) {
        let pending = self.pair.sides[self.side].with(|s| {
            s.bound = Some((n.clone(), bits));
            !s.queue.is_empty()
        });
        if pending || self.pair.closed[self.peer()].load(Ordering::Acquire) {
            n.signal(bits);
        }
    }
}

impl Drop for ChannelEnd {
    fn drop(&mut self) {
        self.pair.closed[self.side].store(true, Ordering::Release);
        // Messages still queued here go (with their handles) outside the lock.
        let left = self.pair.sides[self.side].with(|s| core::mem::take(&mut s.queue));
        drop(left);
        let bound = self.pair.sides[self.peer()].update(|s| s.bound.clone());
        if let Some((n, bits)) = bound {
            n.signal(bits);
        }
    }
}
