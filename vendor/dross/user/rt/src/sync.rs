//! Blocking synchronization between threads of one process. Waiters park in the kernel, so these work in
//! native and Linux programs. `Lock` remains for short critical sections.
use crate::deadline::Deadline;
use crate::{Error, sys};
use alloc::{collections::VecDeque, sync::Arc};
use core::{
    cell::UnsafeCell,
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU32, Ordering},
};

const UNLOCKED: u32 = 0;
const LOCKED: u32 = 1;
const CONTENDED: u32 = 2;

/// A lock whose waiters sleep. A panic ends the process, so there is no poisoning.
pub struct Mutex<T> {
    state: AtomicU32,
    value: UnsafeCell<T>,
}

// SAFETY: access to `value` is serialized by `state`.
unsafe impl<T: Send> Sync for Mutex<T> {}
unsafe impl<T: Send> Send for Mutex<T> {}

impl<T> Mutex<T> {
    pub const fn new(value: T) -> Self {
        Self { state: AtomicU32::new(UNLOCKED), value: UnsafeCell::new(value) }
    }

    pub fn lock(&self) -> MutexGuard<'_, T> {
        if self.state.compare_exchange(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed).is_err() {
            while self.state.swap(CONTENDED, Ordering::Acquire) != UNLOCKED {
                // Any failure only means looking at the word again.
                if sys::park(&self.state, CONTENDED, u64::MAX) == Err(Error::NoMemory) {
                    sys::yield_now();
                }
            }
        }
        MutexGuard { mutex: self }
    }

    pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        self.state
            .compare_exchange(UNLOCKED, LOCKED, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| MutexGuard { mutex: self })
    }

    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }
}

pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<T> Deref for MutexGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the guard holds the lock.
        unsafe { &*self.mutex.value.get() }
    }
}
impl<T> DerefMut for MutexGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: the guard holds the lock.
        unsafe { &mut *self.mutex.value.get() }
    }
}
impl<T> Drop for MutexGuard<'_, T> {
    fn drop(&mut self) {
        if self.mutex.state.swap(UNLOCKED, Ordering::Release) == CONTENDED {
            let _ = sys::unpark(&self.mutex.state, 1);
        }
    }
}

/// Waits for a change made under a `Mutex`. A wait can end with no change, so check in a loop.
#[derive(Default)]
pub struct Condvar {
    sequence: AtomicU32,
}

impl Condvar {
    pub const fn new() -> Self {
        Self { sequence: AtomicU32::new(0) }
    }

    /// Releases the lock, waits for a notification or the deadline, and takes the lock again.
    pub fn wait_until<'a, T>(&self, guard: MutexGuard<'a, T>, deadline: Deadline) -> MutexGuard<'a, T> {
        let mutex = guard.mutex;
        // Read under the lock: a notifier holding it afterwards changes the value, so the park below returns.
        let sequence = self.sequence.load(Ordering::Relaxed);
        drop(guard);
        let _ = sys::park(&self.sequence, sequence, deadline.as_monotonic_ns());
        mutex.lock()
    }

    pub fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        self.wait_until(guard, Deadline::FOREVER)
    }

    pub fn notify_one(&self) {
        self.sequence.fetch_add(1, Ordering::Relaxed);
        let _ = sys::unpark(&self.sequence, 1);
    }

    pub fn notify_all(&self) {
        self.sequence.fetch_add(1, Ordering::Relaxed);
        let _ = sys::unpark(&self.sequence, u64::MAX);
    }
}

struct ChannelState<T> {
    queue: VecDeque<T>,
    senders: usize,
    receiver_gone: bool,
}

struct Shared<T> {
    state: Mutex<ChannelState<T>>,
    ready: Condvar,
}

/// A queue between threads, from any number of senders to one receiver.
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Shared {
        state: Mutex::new(ChannelState { queue: VecDeque::new(), senders: 1, receiver_gone: false }),
        ready: Condvar::new(),
    });
    (Sender(shared.clone()), Receiver(shared))
}

pub struct Sender<T>(Arc<Shared<T>>);

impl<T> Sender<T> {
    /// Returns the value if the receiver is gone or there is no memory to queue it.
    pub fn send(&self, value: T) -> Result<(), T> {
        let mut state = self.0.state.lock();
        if state.receiver_gone || state.queue.try_reserve(1).is_err() {
            return Err(value);
        }
        state.queue.push_back(value);
        drop(state);
        self.0.ready.notify_one();
        Ok(())
    }
}
impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.0.state.lock().senders += 1;
        Self(self.0.clone())
    }
}
impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock();
        state.senders -= 1;
        let last = state.senders == 0;
        drop(state);
        if last {
            self.0.ready.notify_all();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecvError {
    TimedOut,
    /// Every sender is gone and the queue is empty.
    Closed,
}

pub struct Receiver<T>(Arc<Shared<T>>);

impl<T> Receiver<T> {
    pub fn recv_until(&self, deadline: Deadline) -> Result<T, RecvError> {
        let mut state = self.0.state.lock();
        loop {
            if let Some(value) = state.queue.pop_front() {
                return Ok(value);
            }
            if state.senders == 0 {
                return Err(RecvError::Closed);
            }
            if deadline.has_elapsed() {
                return Err(RecvError::TimedOut);
            }
            state = self.0.ready.wait_until(state, deadline);
        }
    }

    pub fn recv(&self) -> Result<T, RecvError> {
        self.recv_until(Deadline::FOREVER)
    }

    /// What is queued now, without waiting.
    pub fn try_iter(&self) -> impl Iterator<Item = T> + '_ {
        core::iter::from_fn(|| self.recv_until(Deadline::IMMEDIATE).ok())
    }

    /// Values until every sender is gone.
    pub fn iter(&self) -> impl Iterator<Item = T> + '_ {
        core::iter::from_fn(|| self.recv().ok())
    }
}
impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock();
        state.receiver_gone = true;
        let queued = core::mem::take(&mut state.queue);
        drop(state);
        drop(queued);
    }
}
