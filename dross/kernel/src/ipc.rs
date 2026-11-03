//! Channels and notifications.
//!
//! A channel is a pair of ends; each end reads the messages the other writes (bytes and handles, which move
//! with the message). Writes never block (a full queue is an error); reads block until a message, the peer
//! closing, or a deadline. A notification is a word of bits: signalling ORs bits in, waiting takes them all. A
//! server waits on one notification that its channels (`bind`) and interrupts signal.

use alloc::collections::VecDeque;
use alloc::sync::{Arc, Weak};
use spin::Mutex;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use dross_abi::{Error, CHANNEL_DEPTH};

use crate::object::{Entry, Object};
use crate::sched::{WaitCell, WaitError};

pub struct Notification {
    _charge: crate::account::Charge,
    bits: WaitCell<u64>,
    pub signals: Arc<crate::events::Signals>,
}

impl Notification {
    pub fn new() -> Result<Notification, Error> {
        let charge = crate::account::current().reserve(4096)?;
        Ok(Notification { _charge: charge, bits: WaitCell::new(0), signals: crate::events::Signals::new(0) })
    }

    pub fn signal(&self, bits: u64) {
        if bits != 0 {
            self.bits.update(|b| { *b |= bits; self.signals.set(dross_abi::event::PENDING, dross_abi::event::PENDING); });
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
        self.bits.wait(deadline_tsc, |b| if *b != 0 { self.signals.set(dross_abi::event::PENDING, 0); Some(core::mem::take(b)) } else { None })
    }
}

pub struct Message {
    pub bytes: Vec<u8>,
    pub handles: Vec<Entry>,
    _charge: Option<crate::account::Charge>,
}
impl Message {
    pub fn new(bytes: Vec<u8>, handles: Vec<Entry>) -> Self { Self { bytes, handles, _charge: None } }
}
// Queue ownership changes and tracing share this lock. It is never held while
// sleeping or while dropping queued capabilities, whose destructors can recurse.
static GRAPH: Mutex<()> = Mutex::new(());
static PAIRS: Mutex<Vec<Weak<Pair>>> = Mutex::new(Vec::new());


struct Side {
    queue: VecDeque<Message>,
    bound: Option<(Arc<Notification>, u64)>,
}

struct Pair {
    ends: Mutex<[Weak<ChannelEnd>; 2]>,
    _charge: crate::account::Charge,
    /// Messages waiting for end i.
    sides: [WaitCell<Side>; 2],
    closed: [AtomicBool; 2],
    signals: [Arc<crate::events::Signals>; 2],
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
    pub fn try_pair() -> Result<(Arc<ChannelEnd>, Arc<ChannelEnd>), Error> {
        let charge = crate::account::current().reserve(1024)?;
        let _graph = GRAPH.lock();
        let mut registry = PAIRS.lock();
        registry.retain(|p| p.strong_count() != 0);
        registry.try_reserve(1).map_err(|_| Error::NoMemory)?;
        let side = || WaitCell::new(Side { queue: VecDeque::new(), bound: None });
        let pair = Arc::new(Pair { ends: Mutex::new([Weak::new(), Weak::new()]), _charge: charge,
            sides: [side(), side()], closed: [AtomicBool::new(false), AtomicBool::new(false)],
            signals: [crate::events::Signals::new(dross_abi::event::WRITABLE), crate::events::Signals::new(dross_abi::event::WRITABLE)] });
        let a = Arc::new(ChannelEnd { pair: pair.clone(), side: 0 });
        let b = Arc::new(ChannelEnd { pair: pair.clone(), side: 1 });
        *pair.ends.lock() = [Arc::downgrade(&a), Arc::downgrade(&b)];
        registry.push(Arc::downgrade(&pair));
        Ok((a, b))
    }
    pub fn pair() -> (Arc<ChannelEnd>, Arc<ChannelEnd>) { Self::try_pair().expect("boot channel allocation") }

    pub fn signals(&self) -> Arc<crate::events::Signals> { self.pair.signals[self.side].clone() }

    fn peer(&self) -> usize {
        1 - self.side
    }

    pub fn write(&self, mut m: Message) -> Result<(), (Error, Message)> {
        if m._charge.is_none() {
            let bytes = (core::mem::size_of::<Message>() + m.bytes.capacity()
                + m.handles.capacity() * (core::mem::size_of::<Entry>() + 128)) as u64;
            match crate::account::current().reserve(bytes) {
                Ok(charge) => m._charge = Some(charge),
                Err(error) => return Err((error, m)),
            }
        }
        let graph = GRAPH.lock();
        let mut m = Some(m);
        let bound = self.pair.sides[self.peer()].update(|s| {
            // Serialize the close check with draining this queue in Drop. Otherwise a
            // writer can enqueue handles after the last reader has drained it.
            if self.pair.closed[self.peer()].load(Ordering::Acquire) {
                return Err(Error::PeerClosed);
            }
            if s.queue.len() >= CHANNEL_DEPTH {
                return Err(Error::ShouldWait);
            }
            s.queue.try_reserve(1).map_err(|_| Error::NoMemory)?;
            s.queue.push_back(m.take().unwrap());
            self.pair.signals[self.peer()].set(dross_abi::event::READABLE, dross_abi::event::READABLE);
            if s.queue.len() == CHANNEL_DEPTH { self.pair.signals[self.side].set(dross_abi::event::WRITABLE, 0); }
            Ok(s.bound.clone())
        });
        drop(graph);
        match bound {
            Ok(Some((n, bits))) => n.signal(bits),
            Ok(None) => {}
            Err(e) => return Err((e, m.unwrap())),
        }
        Ok(())
    }

    /// Takes the next message if it fits in `max_bytes` and `max_handles`.
    pub fn read(&self, max_bytes: usize, max_handles: usize, deadline_tsc: u64) -> Result<Message, ReadError> {
        let closed = &self.pair.closed[self.peer()];
        let r = self.pair.sides[self.side].wait_serialized(deadline_tsc, true, &GRAPH, |s| match s.queue.front() {
            Some(m) if m.bytes.len() > max_bytes || m.handles.len() > max_handles => Some(Err(ReadError::TooSmall(m.bytes.len(), m.handles.len()))),
            Some(_) => {
                let message = s.queue.pop_front().unwrap();
                if s.queue.is_empty() { s.queue = VecDeque::new(); self.pair.signals[self.side].set(dross_abi::event::READABLE, 0); }
                if !self.pair.closed[self.peer()].load(Ordering::Acquire) {
                    self.pair.signals[self.peer()].set(dross_abi::event::WRITABLE, dross_abi::event::WRITABLE);
                }
                Some(Ok(message))
            },
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
        self.pair.sides[self.side].wait_serialized(deadline, false, &GRAPH, |s| match s.queue.front() {
            Some(m) if m.bytes.len() > max_bytes || m.handles.len() > max_handles => Some(Err(ReadError::TooSmall(m.bytes.len(), m.handles.len()))),
            Some(_) => {
                let message = s.queue.pop_front().unwrap();
                if s.queue.is_empty() { s.queue = VecDeque::new(); self.pair.signals[self.side].set(dross_abi::event::READABLE, 0); }
                if !self.pair.closed[self.peer()].load(Ordering::Acquire) {
                    self.pair.signals[self.peer()].set(dross_abi::event::WRITABLE, dross_abi::event::WRITABLE);
                }
                Some(Ok(message))
            },
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
        // Messages still queued here go (with their handles) outside the lock.
        let graph = GRAPH.lock();
        let left = self.pair.sides[self.side].with(|s| {
            self.pair.closed[self.side].store(true, Ordering::Release);
            self.pair.signals[self.peer()].set(dross_abi::event::PEER_CLOSED | dross_abi::event::WRITABLE, dross_abi::event::PEER_CLOSED);
            core::mem::take(&mut s.queue)
        });
        drop(graph);
        drop(left);
        let bound = self.pair.sides[self.peer()].update(|s| s.bound.clone());
        if let Some((n, bits)) = bound {
            n.signal(bits);
        }
    }
}

/// Trace queued native channel capabilities. A reference outside channel queues
/// (including an in-flight operation) roots an endpoint. Only that endpoint's
/// receive queue is reachable; holding its peer cannot recover a self-queued end.
/// Scratch allocation failure leaves all queues intact.
static COLLECTION_REQUESTED: AtomicBool = AtomicBool::new(false);
static COLLECTING: AtomicBool = AtomicBool::new(false);
pub fn request_collection() { COLLECTION_REQUESTED.store(true, Ordering::Release); }
pub fn has_deferred() -> bool { COLLECTION_REQUESTED.load(Ordering::Acquire) && !COLLECTING.load(Ordering::Acquire) }
pub fn reap_deferred() -> usize { if has_deferred() { collect_cycles() } else { 0 } }
pub fn collect_cycles() -> usize {
    COLLECTION_REQUESTED.store(true, Ordering::Release);
    if COLLECTING.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() { return 0; }
    let mut count = 0;
    for _ in 0..2 {
        COLLECTION_REQUESTED.store(false, Ordering::Release);
        count += collect_once();
        if !COLLECTION_REQUESTED.load(Ordering::Acquire) { break; }
    }
    COLLECTING.store(false, Ordering::Release);
    count
}
fn collect_once() -> usize {
    struct Node { pair: Arc<Pair>, ends: [Option<Arc<ChannelEnd>>; 2], incoming: [usize; 2], marked: [bool; 2] }
    // Snapshot ownership must be dropped after GRAPH, including every error path.
    let mut nodes: Vec<Node> = Vec::new();
    let mut drained: Vec<VecDeque<Message>> = Vec::new();
    let Some(graph) = GRAPH.try_lock() else { COLLECTION_REQUESTED.store(true, Ordering::Release); return 0 };
    {
        let mut registry = PAIRS.lock();
        registry.retain(|p| p.strong_count() != 0);
        if nodes.try_reserve_exact(registry.len()).is_err() { return 0; }
        for weak in registry.iter() {
            if let Some(pair) = weak.upgrade() {
                let ends = { let ends = pair.ends.lock(); [ends[0].upgrade(), ends[1].upgrade()] };
                nodes.push(Node { pair, ends, incoming: [0; 2], marked: [false; 2] });
            }
        }
    }
    let mut pointers = Vec::new();
    let mut work = Vec::new();
    if pointers.try_reserve_exact(nodes.len() * 2).is_err()
        || work.try_reserve_exact(nodes.len() * 2).is_err()
        || drained.try_reserve_exact(nodes.len() * 2).is_err() { return 0; }
    for (index, node) in nodes.iter().enumerate() {
        for side in 0..2 { if let Some(end) = &node.ends[side] { pointers.push((Arc::as_ptr(end) as usize, index * 2 + side)); } }
    }
    pointers.sort_unstable_by_key(|&(pointer, _)| pointer);
    let find = |end: &Arc<ChannelEnd>| pointers.binary_search_by_key(&(Arc::as_ptr(end) as usize), |&(pointer, _)| pointer).ok().map(|i| pointers[i].1);
    for index in 0..nodes.len() {
        for side in 0..2 {
            let pair = nodes[index].pair.clone();
            pair.sides[side].with(|state| {
                for message in &state.queue { for entry in &message.handles {
                    if let Object::Channel(end) = &entry.object { if let Some(id) = find(end) { nodes[id / 2].incoming[id % 2] += 1; } }
                } }
            });
        }
    }
    for (index, node) in nodes.iter_mut().enumerate() {
        for side in 0..2 {
            if node.ends[side].as_ref().is_some_and(|end| Arc::strong_count(end) > node.incoming[side] + 1) {
                node.marked[side] = true; work.push(index * 2 + side);
            }
        }
    }
    while let Some(id) = work.pop() {
        let pair = nodes[id / 2].pair.clone();
        pair.sides[id % 2].with(|state| {
            for message in &state.queue { for entry in &message.handles {
                if let Object::Channel(end) = &entry.object { if let Some(next) = find(end) {
                    if !nodes[next / 2].marked[next % 2] { nodes[next / 2].marked[next % 2] = true; work.push(next); }
                } }
            } }
        });
    }
    let mut count = 0;
    for node in &nodes {
        for side in 0..2 { if !node.marked[side] {
            let queue = node.pair.sides[side].with(|state| core::mem::take(&mut state.queue));
            if !queue.is_empty() {
                count += queue.len(); drained.push(queue);
                node.pair.signals[side].set(dross_abi::event::READABLE, 0);
                if !node.pair.closed[side].load(Ordering::Acquire) {
                    node.pair.signals[1 - side].set(dross_abi::event::WRITABLE, dross_abi::event::WRITABLE);
                }
            }
        } }
    }
    drop(graph);
    drop(drained);
    drop(nodes);
    {
        let mut registry = PAIRS.lock();
        registry.retain(|p| p.strong_count() != 0);
        if registry.is_empty() { *registry = Vec::new(); }
        else if registry.capacity() > 64 && registry.capacity() > registry.len() * 4 {
            let mut compact = Vec::new();
            if compact.try_reserve_exact(registry.len()).is_ok() { compact.extend(registry.iter().cloned()); *registry = compact; }
        }
    }
    count
}
