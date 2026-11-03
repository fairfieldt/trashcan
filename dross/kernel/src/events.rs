//! One-shot object readiness. Lock order: source -> handle life -> subscription -> queue.
//! Queue slots reserve packet capacity at registration; delivery never grows packet storage.
use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use alloc::collections::VecDeque;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use dross_abi::{event::Event, Error, EVENT_LIMIT};
use spin::Mutex;
use crate::sched::WaitCell;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub struct Signals {
    inner: Mutex<SignalState>,
}
struct SignalState {
    bits: u64,
    observers: Vec<Weak<Subscription>>,
}
impl Signals {
    pub fn new(bits: u64) -> Arc<Self> {
        Arc::new(Self { inner: Mutex::new(SignalState { bits, observers: Vec::new() }) })
    }
    pub fn set(&self, mask: u64, bits: u64) {
        let mut state = self.inner.lock();
        state.bits = (state.bits & !mask) | (bits & mask);
        let observed = state.bits;
        state.observers.retain(|weak| {
            let Some(sub) = weak.upgrade() else { return false };
            !sub.deliver(observed)
        });
    }
}

/// Logical source ownership is separate from temporary kernel references.
/// Entry clones preserve identity during transfer; HANDLE_DUP creates a new lease.
pub struct HandleLease {
    pub identity: Arc<HandleLife>,
}
pub struct HandleLife {
    owners: AtomicUsize,
    closed: AtomicBool,
    observers: Mutex<Vec<Weak<Subscription>>>,
}
impl HandleLease {
    pub fn new() -> Self {
        Self { identity: Arc::new(HandleLife { owners: AtomicUsize::new(1), closed: AtomicBool::new(false), observers: Mutex::new(Vec::new()) }) }
    }
}
impl Clone for HandleLease {
    fn clone(&self) -> Self {
        self.identity.owners.fetch_add(1, Ordering::Relaxed);
        Self { identity: self.identity.clone() }
    }
}
impl Drop for HandleLease {
    fn drop(&mut self) {
        if self.identity.owners.fetch_sub(1, Ordering::AcqRel) == 1 { self.identity.close(); }
    }
}
impl HandleLife {
    fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) { return; }
        // Never acquire a source lock while holding the handle observer lock.
        let observers = core::mem::take(&mut *self.observers.lock());
        for weak in observers {
            if let Some(sub) = weak.upgrade() { sub.source_closed(); }
        }
    }
}

struct SubState { active: bool, queued: bool, closed: bool }
struct Subscription {
    id: u64,
    key: u64,
    mask: u64,
    source: Weak<Signals>,
    life: Weak<HandleLife>,
    queue: Weak<EventQueue>,
    state: Mutex<SubState>,
}
impl Subscription {
    /// Called with the source lock held. True removes the observer from the source.
    fn deliver(&self, observed: u64) -> bool {
        let mut state = self.state.lock();
        if !state.active { return true; }
        if observed & self.mask == 0 { return false; }
        state.active = false;
        if let Some(queue) = self.queue.upgrade() {
            state.queued = true;
            queue.inner.update(|q| {
                // Each live registration reserves exactly one possible packet.
                assert!(q.ready.len() < q.slots.len());
                q.ready.push_back(Event { subscription: self.id, key: self.key, observed });
            });
        }
        true
    }
    fn source_closed(&self) {
        let source = self.source.upgrade();
        let mut source = source.as_ref().map(|s| s.inner.lock());
        let mut state = self.state.lock();
        state.active = false;
        state.closed = true;
        if let Some(source) = &mut source { source.observers.retain(|w| w.upgrade().is_some_and(|s| s.id != self.id)); }
        // A packet already queued remains a valid observation of this identity.
    }
    fn detach(&self) {
        let source = self.source.upgrade();
        let mut source = source.as_ref().map(|s| s.inner.lock());
        let mut state = self.state.lock();
        state.active = false;
        state.closed = true;
        if let Some(source) = &mut source { source.observers.retain(|w| w.upgrade().is_some_and(|s| s.id != self.id)); }
        if let Some(queue) = self.queue.upgrade() {
            queue.inner.with(|q| {
                q.ready.retain(|e| e.subscription != self.id);
                for slot in &mut q.slots {
                    if slot.as_ref().is_some_and(|s| s.id == self.id) { *slot = None; break; }
                }
            });
        }
        state.queued = false;
    }
}

struct QueueState {
    slots: Vec<Option<Arc<Subscription>>>,
    ready: VecDeque<Event>,
}
pub struct EventQueue {
    _charge: crate::account::Charge,
    inner: WaitCell<QueueState>,
}
impl EventQueue {
    pub fn new(capacity: usize) -> Result<Self, Error> {
        if capacity == 0 || capacity > EVENT_LIMIT { return Err(Error::InvalidArgs); }
        let charge = crate::account::current().reserve(512 + capacity as u64 * 256)?;
        let mut slots = Vec::new();
        slots.try_reserve_exact(capacity).map_err(|_| Error::NoMemory)?;
        slots.resize_with(capacity, || None);
        let mut ready = VecDeque::new();
        ready.try_reserve_exact(capacity).map_err(|_| Error::NoMemory)?;
        Ok(Self { _charge: charge, inner: WaitCell::new(QueueState { slots, ready }) })
    }
    pub fn subscribe(self: &Arc<Self>, source: Arc<Signals>, life: Arc<HandleLife>, mask: u64, key: u64) -> Result<u64, Error> {
        let mut signals = source.inner.lock();
        signals.observers.retain(|w| w.strong_count() != 0);
        let mut handles = life.observers.lock();
        handles.retain(|w| w.strong_count() != 0);
        if life.closed.load(Ordering::Acquire) { return Err(Error::PeerClosed); }
        if signals.observers.len() >= EVENT_LIMIT || handles.len() >= EVENT_LIMIT { return Err(Error::NoResources); }
        signals.observers.try_reserve(1).map_err(|_| Error::NoMemory)?;
        handles.try_reserve(1).map_err(|_| Error::NoMemory)?;
        let sub = self.inner.with(|q| {
            let slot = q.slots.iter_mut().find(|s| s.is_none()).ok_or(Error::NoResources)?;
            let id = NEXT_ID.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1)).map_err(|_| Error::NoResources)?;
            let sub = Arc::new(Subscription { id, key, mask, source: Arc::downgrade(&source), life: Arc::downgrade(&life), queue: Arc::downgrade(self), state: Mutex::new(SubState { active: true, queued: false, closed: false }) });
            *slot = Some(sub.clone());
            Ok::<_, Error>(sub)
        })?;
        handles.push(Arc::downgrade(&sub));
        if !sub.deliver(signals.bits) { signals.observers.push(Arc::downgrade(&sub)); }
        Ok(sub.id)
    }
    fn get(&self, id: u64) -> Result<Arc<Subscription>, Error> {
        self.inner.with(|q| q.slots.iter().flatten().find(|s| s.id == id).cloned().ok_or(Error::NotFound))
    }
    pub fn rearm(&self, id: u64) -> Result<(), Error> {
        let sub = self.get(id)?;
        // Temporary kernel references do not keep logical source ownership alive.
        let life = sub.life.upgrade().ok_or(Error::PeerClosed)?;
        let source = sub.source.upgrade().ok_or(Error::PeerClosed)?;
        let mut signals = source.inner.lock();
        {
            let mut state = sub.state.lock();
            if state.closed || life.closed.load(Ordering::Acquire) { return Err(Error::PeerClosed); }
            if state.queued { return Err(Error::Busy); }
            if state.active { return Err(Error::Busy); }
            signals.observers.retain(|w| w.strong_count() != 0);
            if signals.observers.len() >= EVENT_LIMIT { return Err(Error::NoResources); }
            signals.observers.try_reserve(1).map_err(|_| Error::NoMemory)?;
            state.active = true;
        }
        if !sub.deliver(signals.bits) { signals.observers.push(Arc::downgrade(&sub)); }
        drop(signals);
        drop(life);
        Ok(())
    }
    pub fn cancel(&self, id: u64) -> Result<(), Error> { self.get(id)?.detach(); Ok(()) }
    /// Copy under the queue lock before consuming, so an output fault loses no packet.
    pub fn wait(&self, deadline: u64, mut copy: impl FnMut(Event) -> Result<(), Error>) -> Result<(), Error> {
        let result = self.inner.wait(deadline, |q| {
            let event = *q.ready.front()?;
            if let Err(e) = copy(event) { return Some(Err(e)); }
            q.ready.pop_front();
            let sub = q.slots.iter().flatten().find(|s| s.id == event.subscription).cloned();
            Some(Ok(sub))
        }).map_err(|e| match e { crate::sched::WaitError::TimedOut => Error::TimedOut, crate::sched::WaitError::Killed => Error::PeerClosed })??;
        if let Some(sub) = result { sub.state.lock().queued = false; }
        Ok(())
    }
}

#[cfg(feature = "audit-test")]
pub fn audit_lifetimes() {
    use dross_abi::event::PENDING;
    let source = Signals::new(0);
    let queue = Arc::new(EventQueue::new(1).unwrap());
    let owner = HandleLease::new();
    let temporary = owner.identity.clone();
    let id = queue.subscribe(source.clone(), temporary.clone(), PENDING, 1).unwrap();
    drop(owner);
    // Holding an internal Arc must not postpone source-handle cancellation.
    assert!(temporary.closed.load(Ordering::Acquire));
    source.set(PENDING, PENDING);
    assert!(queue.inner.with(|q| q.ready.is_empty()));
    assert_eq!(queue.rearm(id), Err(Error::PeerClosed));
    queue.cancel(id).unwrap();
    let owner = HandleLease::new();
    let id = queue.subscribe(source, owner.identity.clone(), PENDING, 2).unwrap();
    drop(owner);
    assert_eq!(queue.inner.with(|q| q.ready.front().unwrap().subscription), id);
    queue.cancel(id).unwrap();
    crate::println!("events: source close cancels before temporary references drop");
}
