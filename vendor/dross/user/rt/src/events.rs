//! What one thread is waiting for. Make one `Events` per thread: threads stay on their CPU, so work and its wakeups do
//! too. Threads that should take turns over one set of sources share a queue with `try_clone`.
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use dross_abi::{EVENT_BATCH_LIMIT, event, event::Event, rights};
use crate::deadline::Deadline;
use crate::owned::{self, BorrowedHandle, EventQueue, Message, SendError, Untyped};
use crate::Error;
use crate::sys::{self, Result};

pub struct Events {
    queue: EventQueue,
    taken: [Event; EVENT_BATCH_LIMIT],
    /// Registrations for the next `wait` to arm.
    rearm: Vec<u64>,
    /// Registrations that end with their one reply.
    calls: BTreeSet<u64>,
}
// SAFETY: its handle has one owner, and a kernel queue can be waited on from any thread.
unsafe impl Send for Events {}
impl Events {
    /// `capacity` is how many watches and calls it can have at once.
    pub fn new(capacity: usize) -> Result<Self> {
        Self::from_queue(EventQueue::create(capacity)?)
    }
    fn from_queue(queue: EventQueue) -> Result<Self> {
        let mut rearm = Vec::new();
        rearm.try_reserve_exact(EVENT_BATCH_LIMIT).map_err(|_| Error::NoMemory)?;
        Ok(Self { queue, taken: [Event::default(); EVENT_BATCH_LIMIT], rearm, calls: BTreeSet::new() })
    }
    /// For another thread, which then takes turns with this one: an event goes to one of them, and stays with it until
    /// its next `wait`. Watch and hold with the `Events` that took the event.
    pub fn try_clone(&self) -> Result<Self> {
        Self::from_queue(self.queue.try_duplicate(rights::ALL)?)
    }
    /// Reports `key` from `wait` for as long as `source` shows any of `signals`. Forget it before closing `source`.
    pub fn watch<T>(&mut self, source: BorrowedHandle<'_, T>, signals: u64, key: u64) -> Result<u64> {
        sys::object_subscribe(self.queue.as_handle().as_raw(), source.as_raw(), signals, key)
    }
    pub fn forget(&mut self, id: u64) -> Result<()> {
        self.rearm.retain(|&r| r != id);
        self.calls.remove(&id);
        sys::subscription_cancel(self.queue.as_handle().as_raw(), id)
    }
    /// Stops reporting a watch that `wait` just returned until `release`, for while its source is being dealt with.
    pub fn hold(&mut self, id: u64) {
        self.rearm.retain(|&r| r != id);
    }
    pub fn release(&mut self, id: u64) {
        self.rearm.push(id);
    }
    /// Lets any thread add events to this queue.
    pub fn poster(&self) -> Result<Poster> {
        Ok(Poster { queue: self.queue.try_duplicate(rights::WRITE)? })
    }
    /// Sends `message` on `channel`. When its reply is there, `wait` reports `key`, and `owned::receive` takes the reply.
    /// One call at a time on a channel: any message on it ends the registration.
    pub fn call_async(
        &mut self,
        channel: BorrowedHandle<'_, Untyped>,
        message: Message,
        key: u64,
    ) -> core::result::Result<(), SendError> {
        let id = match self.watch(channel, event::READABLE | event::PEER_CLOSED, key) {
            Ok(id) => id,
            Err(error) => return Err(SendError { error, message }),
        };
        if let Err(failure) = owned::try_send(channel, message) {
            let _ = self.forget(id);
            return Err(failure);
        }
        self.calls.insert(id);
        Ok(())
    }
    /// Arms what the last `wait` returned and takes everything waiting, up to `EVENT_BATCH_LIMIT`. A posted event has
    /// subscription 0, and its value in `observed`.
    pub fn wait(&mut self, deadline: Deadline) -> Result<&[Event]> {
        let raw = self.queue.as_handle().as_raw();
        loop {
            // Only as many as it can report on fit in one call; there are more only after many `release`s.
            while self.rearm.len() > EVENT_BATCH_LIMIT {
                let _ = sys::subscription_rearm(raw, self.rearm.pop().unwrap());
            }
            let count = match sys::event_wait(raw, &self.rearm, &mut self.taken, deadline.as_monotonic_ns()) {
                Ok(count) => count,
                Err(Error::TimedOut) => {
                    self.rearm.clear();
                    return Err(Error::TimedOut);
                }
                Err(error) => return Err(error),
            };
            self.rearm.clear();
            let mut kept = 0;
            for i in 0..count {
                let taken = self.taken[i];
                if taken.observed & event::CANCELLED != 0 {
                    // Its source closed while it was out. Nothing more can come of it.
                    let _ = sys::subscription_cancel(raw, taken.subscription);
                    continue;
                }
                if self.calls.remove(&taken.subscription) {
                    let _ = sys::subscription_cancel(raw, taken.subscription);
                } else if taken.subscription != 0 {
                    self.rearm.push(taken.subscription);
                }
                self.taken[kept] = taken;
                kept += 1;
            }
            if kept != 0 {
                return Ok(&self.taken[..kept]);
            }
        }
    }
}

/// Adds events to one `Events`' queue, from any thread.
pub struct Poster {
    queue: EventQueue,
}
// SAFETY: posting is one system call on the handle, from any thread.
unsafe impl Send for Poster {}
unsafe impl Sync for Poster {}
impl Poster {
    /// NoResources while the queue has no room.
    pub fn post(&self, key: u64, value: u64) -> Result<()> {
        sys::event_post(self.queue.as_handle().as_raw(), key, value)
    }
}
