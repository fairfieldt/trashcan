//! Monotonic, one-shot timer objects. The home CPU's APIC services deadlines;
//! registrations are weak and a closed timer cannot keep itself alive.
use alloc::sync::Arc;
use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;
use dross_abi::{event::EXPIRED, Error, FOREVER};
use crate::{cpu, events::Signals, sched, time};

struct State { generation: u64, deadline: u64, armed: bool }
pub struct Timer {
    _charge: crate::account::Charge,
    pub cpu: u32,
    pub slot: AtomicUsize,
    pub signals: Arc<Signals>,
    state: Mutex<State>,
}
impl Timer {
    pub fn new() -> Result<Arc<Self>, Error> {
        let charge = crate::account::current().reserve(4096)?;
        let timer = Arc::new(Self { _charge: charge, cpu: cpu::index(), slot: AtomicUsize::new(usize::MAX), signals: Signals::new(0), state: Mutex::new(State { generation: 0, deadline: FOREVER, armed: false }) });
        let slot = sched::register_timer(&timer)?;
        timer.slot.store(slot, Ordering::Relaxed);
        Ok(timer)
    }
    /// Reset also clears the EXPIRED latch. Already queued observations remain historical.
    pub fn arm(&self, deadline_ns: u64) -> Result<(), Error> {
        let mut state = self.state.lock();
        let generation = state.generation.checked_add(1).ok_or(Error::NoResources)?;
        let deadline = if deadline_ns == FOREVER { FOREVER } else { time::tsc_at_ns(deadline_ns) };
        let expired = deadline != FOREVER && deadline <= cpu::rdtsc();
        state.generation = generation;
        state.deadline = deadline;
        state.armed = deadline != FOREVER && !expired;
        sched::set_timer(self, generation, if state.armed { deadline } else { FOREVER });
        self.signals.set(EXPIRED, if expired { EXPIRED } else { 0 });
        Ok(())
    }
    pub fn cancel(&self) -> Result<(), Error> { self.arm(FOREVER) }
    pub fn expire(&self, generation: u64, deadline: u64, now: u64) {
        let mut state = self.state.lock();
        if state.armed && state.generation == generation && state.deadline == deadline && deadline <= now {
            state.armed = false;
            self.signals.set(EXPIRED, EXPIRED);
        }
    }
}
impl Drop for Timer {
    fn drop(&mut self) { sched::unregister_timer(self); }
}
