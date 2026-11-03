//! Counts allocator-requested bytes, including upstream fake bindings.
//! This does not impose a quota, include allocator metadata, or measure RSS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingSystem;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);

fn allocated(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
    CALLS.fetch_add(1, Ordering::Relaxed);
}

// SAFETY: Every allocation/deallocation is delegated with the original pointer
// and layout to System. Accounting only uses nonallocating atomic operations.
unsafe impl GlobalAlloc for CountingSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: This is the allocator contract supplied by the caller.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: This is the allocator contract supplied by the caller.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: Pointer/layout are unchanged from the caller's contract.
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: Pointer/layout/new size are passed through unchanged.
        let replacement = unsafe { System.realloc(pointer, layout, size) };
        if !replacement.is_null() {
            if size >= layout.size() {
                allocated(size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - size, Ordering::Relaxed);
                CALLS.fetch_add(1, Ordering::Relaxed);
            }
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: CountingSystem = CountingSystem;

#[derive(Debug)]
pub struct Measurement {
    pub live_before: usize,
    pub live_after: usize,
    pub peak_live: usize,
    pub allocation_calls: usize,
}

/// Used only by the serial proof executable. Concurrent test-runner allocations
/// would make a per-scenario measurement ambiguous, so tests do not call this.
pub fn measure<T>(work: impl FnOnce() -> T) -> (T, Measurement) {
    let before = LIVE.load(Ordering::Relaxed);
    let calls_before = CALLS.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = work();
    let measurement = Measurement {
        live_before: before,
        live_after: LIVE.load(Ordering::Relaxed),
        peak_live: PEAK.load(Ordering::Relaxed),
        allocation_calls: CALLS.load(Ordering::Relaxed) - calls_before,
    };
    (result, measurement)
}
