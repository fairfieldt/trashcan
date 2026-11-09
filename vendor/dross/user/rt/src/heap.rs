//! The program heap. Small allocations come from arenas that are added as needed and kept. A large one is a memory
//! object of its own, returned to the kernel when freed. Running out of the program's memory account fails the
//! allocation, so `try_reserve` reports it.
use crate::{Lock, sys};
use core::{
    alloc::{GlobalAlloc, Layout},
    ptr::{self, NonNull},
    sync::atomic::{AtomicUsize, Ordering},
};
use dross_abi::vm;
use linked_list_allocator::Heap;

const ARENA: usize = 16 << 20;
const MAX_ARENAS: usize = 64;
const LARGE: usize = 1 << 20;
const PAGE: usize = 4096;

struct Arenas {
    heaps: [Heap; MAX_ARENAS],
    count: usize,
}

impl Arenas {
    /// False if there is no room for another of `size` bytes.
    fn add(&mut self, size: usize) -> bool {
        if self.count == MAX_ARENAS {
            return false;
        }
        let Some(at) = map(size) else { return false };
        // SAFETY: a fresh private mapping, used for nothing else.
        unsafe { self.heaps[self.count].init(at, size) };
        self.count += 1;
        true
    }
}

/// Blocks freed recently, by size. unixd allocates for every call it serves, and with a first-fit list under one lock
/// spent 20% of its time there (results/selfhost-perf/cache1).
const CLASSES: [usize; 12] = [16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 2048, 4096];
/// All blocks are this aligned, so their sizes alone decide their class. More alignment goes to the arenas.
const CLASS_ALIGN: usize = 16;
/// What a class keeps, at most: more goes back to the arenas, for any size to use.
const CLASS_BLOCKS: usize = 256;

struct FreeList {
    head: *mut u8,
    count: usize,
}
// SAFETY: free memory nothing else refers to, reached only through its lock.
unsafe impl Send for FreeList {}

pub struct ProgramHeap {
    arenas: Lock<Arenas>,
    large_bytes: AtomicUsize,
    free: [Lock<FreeList>; CLASSES.len()],
}

fn class_of(layout: Layout) -> Option<usize> {
    if layout.align() > CLASS_ALIGN {
        return None;
    }
    CLASSES.iter().position(|&size| layout.size() <= size)
}
fn class_layout(class: usize) -> Layout {
    // SAFETY: the sizes above are multiples of the alignment and not near overflow.
    unsafe { Layout::from_size_align_unchecked(CLASSES[class], CLASS_ALIGN) }
}

fn page_rounded(size: usize) -> usize {
    (size + PAGE - 1) & !(PAGE - 1)
}

fn is_large(layout: Layout) -> bool {
    layout.size() >= LARGE && layout.align() <= PAGE
}

/// Fresh zeroed pages. The mapping keeps the memory object alive.
fn map(size: usize) -> Option<*mut u8> {
    let memory = sys::mem_create(size as u64, 0).ok()?;
    let at = sys::vm_map(memory, 0, size as u64, 0, vm::READ | vm::WRITE);
    let _ = sys::close(memory);
    at.ok().map(|at| at as *mut u8)
}

impl ProgramHeap {
    pub const fn new() -> Self {
        Self {
            arenas: Lock::new(Arenas { heaps: [const { Heap::empty() }; MAX_ARENAS], count: 0 }),
            large_bytes: AtomicUsize::new(0),
            free: [const { Lock::new(FreeList { head: ptr::null_mut(), count: 0 }) }; CLASSES.len()],
        }
    }

    pub fn add_first_arena(&self) -> bool {
        self.arenas.with(|a| a.add(ARENA))
    }

    /// Bytes currently allocated.
    pub fn used(&self) -> usize {
        let small = self.arenas.with(|a| a.heaps[..a.count].iter().map(Heap::used).sum::<usize>());
        let free: usize = self.free.iter().zip(CLASSES).map(|(list, size)| list.with(|l| l.count) * size).sum();
        small - free + self.large_bytes.load(Ordering::Relaxed)
    }

    fn alloc_large(&self, layout: Layout) -> *mut u8 {
        let size = page_rounded(layout.size());
        let Some(at) = map(size) else { return ptr::null_mut() };
        self.large_bytes.fetch_add(size, Ordering::Relaxed);
        at
    }

    fn alloc_small(&self, layout: Layout) -> *mut u8 {
        self.arenas.with(|a| {
            let count = a.count;
            for heap in a.heaps[..count].iter_mut().rev() {
                if let Ok(block) = heap.allocate_first_fit(layout) {
                    return block.as_ptr();
                }
            }
            // Only an over-aligned block can need more than one arena's bytes.
            if !a.add(ARENA.max(page_rounded(layout.size() + layout.align()))) {
                return ptr::null_mut();
            }
            a.heaps[count].allocate_first_fit(layout).map_or(ptr::null_mut(), NonNull::as_ptr)
        })
    }

    unsafe fn free_small(&self, block: *mut u8, layout: Layout) {
        self.arenas.with(|a| {
            let count = a.count;
            let heap = a.heaps[..count].iter_mut().find(|h| (h.bottom()..h.top()).contains(&block));
            // SAFETY: the caller allocated this block with this layout, from this arena.
            unsafe { heap.expect("freed block in an arena").deallocate(NonNull::new_unchecked(block), layout) };
        });
    }
}

// SAFETY: blocks come from distinct arena allocations or fresh mappings, sized and aligned for their layouts.
unsafe impl GlobalAlloc for ProgramHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_large(layout) {
            return self.alloc_large(layout);
        }
        let Some(class) = class_of(layout) else { return self.alloc_small(layout) };
        let block = self.free[class].with(|list| {
            let block = list.head;
            if !block.is_null() {
                // SAFETY: a free block holds the next one.
                list.head = unsafe { *(block as *const *mut u8) };
                list.count -= 1;
            }
            block
        });
        if block.is_null() { self.alloc_small(class_layout(class)) } else { block }
    }

    unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
        if is_large(layout) {
            let size = page_rounded(layout.size());
            self.large_bytes.fetch_sub(size, Ordering::Relaxed);
            let _ = sys::vm_unmap(block as u64, size as u64);
            return;
        }
        let Some(class) = class_of(layout) else {
            // SAFETY: as this function's.
            return unsafe { self.free_small(block, layout) };
        };
        let kept = self.free[class].with(|list| {
            if list.count == CLASS_BLOCKS {
                return false;
            }
            // SAFETY: the caller is done with this block, at least 16 bytes and aligned to 16.
            unsafe { *(block as *mut *mut u8) = list.head };
            list.head = block;
            list.count += 1;
            true
        });
        if !kept {
            // SAFETY: it came from this class.
            unsafe { self.free_small(block, class_layout(class)) };
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if is_large(layout) {
            // The kernel zeroes a memory object when creating it.
            return self.alloc_large(layout);
        }
        // SAFETY: the caller's layout, passed on; the block is `layout.size()` writable bytes.
        unsafe {
            let block = self.alloc(layout);
            if !block.is_null() {
                ptr::write_bytes(block, 0, layout.size());
            }
            block
        }
    }
}
