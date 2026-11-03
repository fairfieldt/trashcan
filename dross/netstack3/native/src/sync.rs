//! Platform locks for the off-path native worker. They never call into core.
use core::sync::atomic::{AtomicBool, AtomicIsize, AtomicU8, Ordering};

#[derive(Debug, Default)]
pub struct RawMutex(AtomicBool);
// SAFETY: successful acquisition has acquire ordering and unique ownership;
// release makes writes visible. Unlock only follows a successful acquisition.
unsafe impl lock_api::RawMutex for RawMutex {
    const INIT: Self = Self(AtomicBool::new(false));
    type GuardMarker = lock_api::GuardSend;
    fn lock(&self) {
        while !self.try_lock() {
            dross_rt::sys::yield_now();
        }
    }
    fn try_lock(&self) -> bool {
        self.0
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
    unsafe fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

#[derive(Debug, Default)]
pub struct RawRwLock(AtomicIsize);
// SAFETY: -1 is an exclusive writer, nonnegative values count readers.
// All successful acquisitions synchronize with release on the same atomic.
unsafe impl lock_api::RawRwLock for RawRwLock {
    const INIT: Self = Self(AtomicIsize::new(0));
    type GuardMarker = lock_api::GuardSend;
    fn lock_shared(&self) {
        while !self.try_lock_shared() {
            dross_rt::sys::yield_now();
        }
    }
    fn try_lock_shared(&self) -> bool {
        let mut current = self.0.load(Ordering::Relaxed);
        while current >= 0 && current < isize::MAX {
            match self.0.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(next) => current = next,
            }
        }
        false
    }
    unsafe fn unlock_shared(&self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
    fn lock_exclusive(&self) {
        while !self.try_lock_exclusive() {
            dross_rt::sys::yield_now();
        }
    }
    fn try_lock_exclusive(&self) -> bool {
        self.0
            .compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
    unsafe fn unlock_exclusive(&self) {
        self.0.store(0, Ordering::Release);
    }
}

#[derive(Debug, Default)]
pub struct Once(AtomicU8);
// SAFETY: exactly one caller transitions 0->1 and executes initialization;
// successful completion is published as2 with release/acquire ordering.
// A panic poisons state3 so no second initialization can execute.
unsafe impl netstack3_sync::once::Once for Once {
    fn once(&self, f: impl FnOnce()) {
        match self
            .0
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Acquire)
        {
            Ok(_) => {
                struct Poison<'a>(&'a AtomicU8);
                impl Drop for Poison<'_> {
                    fn drop(&mut self) {
                        self.0.store(3, Ordering::Release);
                    }
                }
                let poison = Poison(&self.0);
                f();
                core::mem::forget(poison);
                self.0.store(2, Ordering::Release);
            }
            Err(_) => loop {
                match self.0.load(Ordering::Acquire) {
                    2 => break,
                    3 => panic!("native Once initializer panicked"),
                    _ => dross_rt::sys::yield_now(),
                }
            },
        }
    }
    fn is_completed(&self) -> bool {
        self.0.load(Ordering::Acquire) == 2
    }
}

pub type Mutex<T> = netstack3_sync::Mutex<T, RawMutex>;
