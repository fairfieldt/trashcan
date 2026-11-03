//! Growable kernel heap. Empty expansion segments are reclaimed at lock-free
//! syscall/scheduler safe points, after cross-CPU translation invalidation.
use core::alloc::{GlobalAlloc, Layout};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, Ordering};
use linked_list_allocator::Heap;
use spin::Mutex;
const INITIAL_BYTES: usize = 4 << 20;
const GROW_BYTES: usize = 4 << 20;
const SEGMENTS: usize = 256;
struct Segment { heap: Heap, base: u64, bytes: usize, extent: usize, retired: bool, reaping: bool }
impl Segment { const fn empty() -> Self { Self { heap: Heap::empty(), base: 0, bytes: 0, extent: 0, retired: false, reaping: false } } }
struct KernelHeap(Mutex<[Segment; SEGMENTS]>);
#[global_allocator]
static HEAP: KernelHeap = KernelHeap(Mutex::new([const { Segment::empty() }; SEGMENTS]));
static RETIRED: AtomicBool = AtomicBool::new(false);
static COLLECTING: AtomicBool = AtomicBool::new(false);
static DIRTY_TABLES: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for KernelHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut segments = self.0.lock();
        for segment in segments.iter_mut().filter(|s| s.bytes != 0) {
            if let Ok(pointer) = segment.heap.allocate_first_fit(layout) {
                segment.retired = false; return pointer.as_ptr();
            }
        }
        if crate::paging::kernel_pml4() == 0 { return core::ptr::null_mut(); }
        let Some(bytes) = layout.size().checked_add(layout.align()).and_then(|n| n.checked_add(4095)).map(|n| n & !4095) else { return core::ptr::null_mut() };
        let bytes = bytes.max(GROW_BYTES);
        if bytes as u64 > crate::mem::free_bytes() { return core::ptr::null_mut(); }
        let index = segments.iter().enumerate().skip(1).find(|(_, s)| s.bytes == 0 && !s.reaping && s.extent >= bytes).map(|(i, _)| i)
            .or_else(|| segments.iter().enumerate().skip(1).find(|(_, s)| s.bytes == 0 && !s.reaping).map(|(i, _)| i));
        let Some(index) = index else { return core::ptr::null_mut() };
        let segment = &mut segments[index];
        if segment.extent < bytes {
            let Some(extent) = bytes.checked_next_power_of_two() else { return core::ptr::null_mut() };
            let Some(base) = crate::mem::reserve_heap_range(extent as u64) else { return core::ptr::null_mut() };
            segment.base = base; segment.extent = extent;
        }
        if crate::mem::map_heap_pages(segment.base, bytes / 4096).is_err() {
            DIRTY_TABLES.store(true, Ordering::Release); RETIRED.store(true, Ordering::Release);
            return core::ptr::null_mut();
        }
        // SAFETY: newly mapped pages; the extent is exclusively owned by this segment.
        unsafe { segment.heap.init(segment.base as *mut u8, bytes); }
        segment.bytes = bytes; segment.retired = false;
        match segment.heap.allocate_first_fit(layout) {
            Ok(pointer) => pointer.as_ptr(),
            Err(_) => { segment.retired = true; RETIRED.store(true, Ordering::Release); core::ptr::null_mut() }
        }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let mut segments = self.0.lock();
        let (index, segment) = segments.iter_mut().enumerate().find(|(_, s)| s.bytes != 0
            && pointer as u64 >= s.base && (pointer as u64) < s.base + s.bytes as u64).expect("kernel heap pointer");
        // SAFETY: GlobalAlloc's caller returns the original pointer and layout.
        unsafe { segment.heap.deallocate(NonNull::new_unchecked(pointer), layout); }
        if index != 0 && segment.heap.used() == 0 { segment.retired = true; RETIRED.store(true, Ordering::Release); }
    }
}
pub fn init() {
    let base = crate::mem::alloc_zeroed(INITIAL_BYTES / 4096, 2 << 20);
    let mut segments = HEAP.0.lock();
    // SAFETY: fresh direct-mapped boot pages, kept as the initial allocator reserve.
    unsafe { segments[0].heap.init(base as *mut u8, INITIAL_BYTES); }
    segments[0].base = base; segments[0].bytes = INITIAL_BYTES; segments[0].extent = INITIAL_BYTES;
    drop(segments);
    crate::println!("heap: {} MiB initial reserve", INITIAL_BYTES >> 20);
}
pub fn has_retired() -> bool { RETIRED.load(Ordering::Acquire) }
/// Must be called without another kernel lock: shootdown may wait for CPUs
/// that are finishing unrelated kernel operations. No allocator lock spans it.
pub fn reap_deferred() {
    if !has_retired() { return; }
    if COLLECTING.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        crate::paging::serve_shootdown();
        return;
    }
    struct Guard;
    impl Drop for Guard { fn drop(&mut self) { COLLECTING.store(false, Ordering::Release); } }
    let _guard = Guard;
    for _ in 0..4 {
        let range = {
            let mut segments = HEAP.0.lock();
            if let Some((index, segment)) = segments.iter_mut().enumerate().skip(1).find(|(_, s)| s.retired) {
                let range = (index, segment.base, segment.bytes);
                segment.heap = Heap::empty(); segment.bytes = 0; segment.retired = false; segment.reaping = true;
                Some(range)
            } else if DIRTY_TABLES.swap(false, Ordering::AcqRel) {
                Some((usize::MAX, 0, 0))
            } else { RETIRED.store(false, Ordering::Release); None }
        };
        let Some((index, base, bytes)) = range else { return };
        crate::mem::free_heap_pages(base, bytes / 4096);
        if index != usize::MAX { HEAP.0.lock()[index].reaping = false; }
    }
}

#[cfg(feature = "audit-test")]
pub fn audit_test() {
    let before = crate::mem::free_bytes();
    let mut large = alloc::vec::Vec::<u8>::new();
    large.try_reserve_exact(80 << 20).unwrap();
    large.resize(80 << 20, 0x5a);
    assert_eq!(large[0], 0x5a); assert_eq!(large[large.len() - 1], 0x5a);
    assert!(crate::mem::free_bytes() < before - (64 << 20));
    drop(large);
    while has_retired() { reap_deferred(); }
    assert_eq!(crate::mem::free_bytes(), before);
    crate::println!("heap: growth beyond 64 MiB and empty-segment reclaim passed");
}
