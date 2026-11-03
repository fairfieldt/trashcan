//! The kernel heap: one block of direct-mapped frames handed to a linked-list allocator.

use linked_list_allocator::LockedHeap;

#[global_allocator]
static HEAP: LockedHeap = LockedHeap::empty();

const HEAP_BYTES: usize = 64 << 20;

pub fn init() {
    let base = crate::mem::alloc_zeroed(HEAP_BYTES / 4096, 2 << 20);
    // SAFETY: a fresh allocation that is never freed.
    unsafe { HEAP.lock().init(base as *mut u8, HEAP_BYTES) };
    crate::println!("heap: {} MiB at {:#x}", HEAP_BYTES >> 20, base);
}
