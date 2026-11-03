//! Physical memory (the `pmm` bitmap over the loader's memory map), the direct map, and MMIO mappings.

use bootinfo::{BootInfo, RegionKind, DIRECT_MAP};
use dross_abi::Error;
use core::sync::atomic::{AtomicU64, Ordering};
use pmm::{Frames, FRAME};
use spin::Mutex;
use x86_64::registers::control::Cr3;
use x86_64::structures::paging::{
    FrameAllocator, Mapper, OffsetPageTable, Page, PageTable, PageTableFlags as F, PhysFrame, Size4KiB,
};
use x86_64::{PhysAddr, VirtAddr};

const EFI_CONVENTIONAL: u32 = 7;

static FRAMES: Mutex<Option<Frames<'static>>> = Mutex::new(None);

/// MMIO mappings go here, uncached, one after another (PML4 slot 384).
const MMIO_BASE: u64 = 0xffff_c000_0000_0000;
static MMIO_NEXT: AtomicU64 = AtomicU64::new(MMIO_BASE);

pub const fn virt(phys: u64) -> u64 {
    DIRECT_MAP + phys
}

pub fn init(info: &BootInfo) {
    let mut usable = 0;
    let mut kernel = 0;
    let mut top = 0;
    for r in info.regions() {
        match r.kind {
            RegionKind::Usable => usable += r.pages,
            RegionKind::Kernel => kernel += r.pages,
            _ => {}
        }
        if matches!(r.kind, RegionKind::Usable | RegionKind::Kernel) {
            top = top.max(r.end());
        }
    }
    crate::println!(
        "memory: {} regions, {} MiB usable, {} KiB handed over by the loader, RAM up to {:#x}, direct map {} GiB",
        info.region_count,
        usable * FRAME >> 20,
        kernel * 4,
        top,
        info.phys_limit >> 30
    );

    // Until the BSP has its own GDT and IDT, the firmware's are live in boot services memory: only
    // conventional memory is handed out before `reclaim_boot_services`.
    let words = Frames::words_for(top);
    let bytes = (words * 8) as u64;
    let at = info
        .regions()
        .iter()
        .find(|r| r.efi_type == EFI_CONVENTIONAL && r.start >= 0x10_0000 && r.pages * FRAME >= bytes)
        .expect("no room for the frame bitmap")
        .start;
    // SAFETY: usable memory nothing else owns yet, in the direct map.
    let bits = unsafe { core::slice::from_raw_parts_mut(virt(at) as *mut u64, words) };
    let mut f = Frames::new(bits);
    for r in info.regions().iter().filter(|r| r.efi_type == EFI_CONVENTIONAL) {
        f.release(r.start, r.end());
    }
    // The first MiB stays out (real-mode leftovers; the AP trampoline page is the loader's).
    f.reserve(0, 0x10_0000);
    f.reserve(at, at + bytes);
    crate::println!("memory: frame bitmap {} KiB at {:#x}, {} MiB free", bytes >> 10, at, f.free_frames() * FRAME >> 20);
    *FRAMES.lock() = Some(f);
}

/// The loader's and boot services' memory, once nothing the firmware set up is in use.
pub fn reclaim_boot_services(info: &BootInfo) {
    let mut f = FRAMES.lock();
    let f = f.as_mut().expect("memory not initialized");
    let before = f.free_frames();
    for r in info.regions().iter().filter(|r| r.kind == RegionKind::Usable && r.efi_type != EFI_CONVENTIONAL) {
        f.release(r.start, r.end());
    }
    f.reserve(0, 0x10_0000);
    crate::println!("memory: {} KiB of loader and boot services memory reclaimed", (f.free_frames() - before) * 4);
}

/// RAM ranges (usable or the kernel's) from the memory map, to keep `MEM_PHYSICAL` off them.
static RAM: Mutex<alloc::vec::Vec<(u64, u64)>> = Mutex::new(alloc::vec::Vec::new());

pub fn record_ram(info: &BootInfo) {
    let mut r = RAM.lock();
    for reg in info.regions().iter().filter(|r| matches!(r.kind, RegionKind::Usable | RegionKind::Kernel)) {
        r.push((reg.start, reg.end()));
    }
}

pub fn overlaps_ram(phys: u64, size: u64) -> bool {
    RAM.lock().iter().any(|&(s, e)| phys < e && phys + size > s)
}

pub fn alloc_phys_zeroed(pages: usize) -> u64 {
    try_alloc_phys_zeroed(pages).expect("boot physical allocation")
}

pub fn try_alloc_phys_zeroed(pages: usize) -> Result<u64, Error> {
    let p = alloc_phys(pages, FRAME, u64::MAX).ok_or(Error::NoMemory)?;
    zero_phys(p, pages);
    Ok(p)
}

pub fn zero_phys(phys: u64, pages: usize) {
    // SAFETY: frames the caller owns, in the direct map.
    unsafe { core::ptr::write_bytes(virt(phys) as *mut u8, 0, pages * FRAME as usize) };
}

pub fn free_phys(phys: u64, pages: usize) {
    FRAMES.lock().as_mut().expect("memory not initialized").free(phys, pages as u64);
}

/// Frees pages given by their direct-map address (from `alloc_zeroed`).
pub fn free_virt(v: u64, pages: usize) {
    free_phys(v - DIRECT_MAP, pages);
}

pub fn free_bytes() -> u64 {
    FRAMES.lock().as_ref().map_or(0, |f| f.free_frames() * FRAME)
}

/// Physical frames: `pages` contiguous, aligned to `align` bytes, below `ceiling`.
pub fn alloc_phys(pages: usize, align: u64, ceiling: u64) -> Option<u64> {
    #[cfg(feature = "audit-test")]
    if ALLOCATIONS_BEFORE_FAILURE.load(Ordering::Relaxed) >= 0 {
        if ALLOCATIONS_BEFORE_FAILURE.fetch_sub(1, Ordering::Relaxed) == 0 { return None; }
    }
    FRAMES.lock().as_mut().expect("memory not initialized").alloc(pages as u64, align, ceiling)
}

#[cfg(feature = "audit-test")]
static ALLOCATIONS_BEFORE_FAILURE: core::sync::atomic::AtomicI64 = core::sync::atomic::AtomicI64::new(-1);

/// Boot-only failure injection: never enabled while user threads can allocate.
#[cfg(feature = "audit-test")]
pub fn fail_after(allocations: i64) {
    ALLOCATIONS_BEFORE_FAILURE.store(allocations, Ordering::Relaxed);
}

#[cfg(feature = "audit-test")]
pub fn audit_test() {
    // Warm the shared stack table first: empty table branches are deliberately
    // retained, but every stack data page must be rolled back.
    let next = STACK_NEXT.load(Ordering::Relaxed);
    if next % (2 << 20) > (2 << 20) - 64 * FRAME {
        // Skip unused virtual addresses so warm-up, failure and retry share one
        // leaf table, independent of the number of CPUs brought up before us.
        STACK_NEXT.store(next.next_multiple_of(2 << 20), Ordering::Relaxed);
    }
    let stack = try_alloc_stack(16).unwrap();
    free_stack(stack, 16);
    let before = free_bytes();
    fail_after(3);
    assert!(matches!(try_alloc_stack(16), Err(Error::NoMemory)));
    fail_after(-1);
    assert_eq!(free_bytes(), before);
    let stack = try_alloc_stack(16).unwrap();
    free_stack(stack, 16);
    assert_eq!(free_bytes(), before);
    crate::println!("memory: stack allocation failure rollback passed");
}

/// Zeroed pages, returned as their direct-map address. Panics when memory runs out.
pub fn alloc_zeroed(pages: usize, align: u64) -> u64 {
    try_alloc_zeroed(pages, align).expect("boot physical allocation")
}

pub fn try_alloc_zeroed(pages: usize, align: u64) -> Result<u64, Error> {
    let p = alloc_phys(pages, align, u64::MAX).ok_or(Error::NoMemory)?;
    let v = virt(p);
    // SAFETY: just allocated, in the direct map.
    unsafe { core::ptr::write_bytes(v as *mut u8, 0, pages * FRAME as usize) };
    Ok(v)
}

struct PageTableFrames;

// SAFETY: hands out unused frames; the mapper zeroes the tables it makes from them.
unsafe impl FrameAllocator<Size4KiB> for PageTableFrames {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        alloc_phys(1, FRAME, u64::MAX).map(|p| PhysFrame::containing_address(PhysAddr::new(p)))
    }
}

static MAPPER_LOCK: Mutex<()> = Mutex::new(());

/// Kernel stacks live here (PML4 slot 416), each with an unmapped guard page below it, so an overflow faults
/// instead of writing into whatever frame lies below.
const STACK_BASE: u64 = 0xffff_d000_0000_0000;
static STACK_NEXT: AtomicU64 = AtomicU64::new(STACK_BASE);

fn kernel_mapper() -> OffsetPageTable<'static> {
    let (pml4, _) = Cr3::read();
    // SAFETY: the kernel half is shared by every PML4; MAPPER_LOCK serializes changes to it.
    unsafe { OffsetPageTable::new(&mut *(virt(pml4.start_address().as_u64()) as *mut PageTable), VirtAddr::new(DIRECT_MAP)) }
}

/// A zeroed kernel stack of `pages` pages; returns its lowest address.
pub fn alloc_stack(pages: usize) -> u64 {
    try_alloc_stack(pages).expect("boot stack allocation")
}

pub fn try_alloc_stack(pages: usize) -> Result<u64, Error> {
    let _guard = MAPPER_LOCK.lock();
    let base = STACK_NEXT.fetch_add((pages as u64 + 1) * FRAME, Ordering::Relaxed) + FRAME;
    let mut mapper = kernel_mapper();
    for i in 0..pages as u64 {
        let Some(frame) = alloc_phys(1, FRAME, u64::MAX) else {
            free_stack_pages(&mut mapper, base, i);
            return Err(Error::NoMemory);
        };
        zero_phys(frame, 1);
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(base + i * FRAME));
        let flags = F::PRESENT | F::WRITABLE | F::NO_EXECUTE;
        // SAFETY: a fresh range of the stack window, to a fresh frame.
        match unsafe { mapper.map_to(page, PhysFrame::containing_address(PhysAddr::new(frame)), flags, &mut PageTableFrames) } {
            Ok(flush) => flush.flush(),
            Err(_) => {
                free_phys(frame, 1);
                free_stack_pages(&mut mapper, base, i);
                return Err(Error::NoMemory);
            }
        }
    }
    Ok(base)
}

fn free_stack_pages(mapper: &mut OffsetPageTable<'_>, base: u64, pages: u64) {
    for i in 0..pages {
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(base + i * FRAME));
        if let Ok((frame, flush)) = mapper.unmap(page) {
            flush.flush();
            free_phys(frame.start_address().as_u64(), 1);
        }
    }
}

/// Frees a stack from `alloc_stack` (its addresses are not reused).
pub fn free_stack(base: u64, pages: usize) {
    let _guard = MAPPER_LOCK.lock();
    let mut mapper = kernel_mapper();
    free_stack_pages(&mut mapper, base, pages as u64);
    // Other CPUs may hold the translation; the addresses are never reused, so stale entries do no harm.
}

/// Maps `size` bytes of device memory at `phys` uncached (PCD|PWT, no execute) and returns the virtual address
/// of `phys`. Mappings are permanent.
pub fn map_mmio(phys: u64, size: u64) -> u64 {
    let _guard = MAPPER_LOCK.lock();
    let first = phys & !(FRAME - 1);
    let pages = (phys + size - first).div_ceil(FRAME);
    let base = MMIO_NEXT.fetch_add(pages * FRAME, Ordering::Relaxed);
    let mut mapper = kernel_mapper();
    for i in 0..pages {
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(base + i * FRAME));
        let frame = PhysFrame::containing_address(PhysAddr::new(first + i * FRAME));
        let flags = F::PRESENT | F::WRITABLE | F::NO_CACHE | F::WRITE_THROUGH | F::NO_EXECUTE;
        // SAFETY: a fresh virtual range in the MMIO window, mapped to device memory.
        unsafe { mapper.map_to(page, frame, flags, &mut PageTableFrames).expect("map_mmio").flush() };
    }
    base + (phys - first)
}

// Dedicated shared-kernel slots 400..415; reclaimed extents can be reused only
// after a full cross-CPU invalidation. No heap allocation occurs in these helpers.
const HEAP_BASE: u64 = 0xffff_c800_0000_0000;
static HEAP_NEXT: AtomicU64 = AtomicU64::new(HEAP_BASE);
pub fn reserve_heap_range(bytes: u64) -> Option<u64> {
    HEAP_NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed,
        |base| base.checked_add(bytes).filter(|end| *end <= HEAP_BASE + (16u64 << 39))).ok()
}
pub fn map_heap_pages(base: u64, pages: usize) -> Result<(), Error> {
    let _guard = MAPPER_LOCK.lock();
    let mut mapper = kernel_mapper();
    for index in 0..pages as u64 {
        let Some(frame) = alloc_phys(1, FRAME, u64::MAX) else { free_stack_pages(&mut mapper, base, index); return Err(Error::NoMemory) };
        zero_phys(frame, 1);
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(base + index * FRAME));
        // SAFETY: unpublished fresh kernel range, to freshly owned frames.
        match unsafe { mapper.map_to(page, PhysFrame::containing_address(PhysAddr::new(frame)), F::PRESENT | F::WRITABLE | F::NO_EXECUTE, &mut PageTableFrames) } {
            Ok(flush) => flush.flush(),
            Err(_) => { free_phys(frame, 1); free_stack_pages(&mut mapper, base, index); return Err(Error::NoMemory); }
        }
    }
    Ok(())
}
pub fn free_heap_pages(base: u64, pages: usize) {
    fn push(head: &mut u64, frame: u64) {
        // SAFETY: dead data, or an empty unlinked page table. Aligned next-frame
        // addresses have PRESENT clear if a CPU still walks an old table.
        unsafe { (virt(frame) as *mut u64).write(*head); }
        *head = frame;
    }
    fn prune(phys: u64, level: u32, head: &mut u64) -> bool {
        // SAFETY: MAPPER_LOCK exclusively owns changes to shared kernel tables.
        let entries = unsafe { &mut *(virt(phys) as *mut PageTable) };
        let mut empty = true;
        for entry in entries.iter_mut() {
            if !entry.flags().contains(F::PRESENT) { continue; }
            if level > 1 && !entry.flags().contains(F::HUGE_PAGE) {
                let child = entry.addr().as_u64();
                if prune(child, level - 1, head) { entry.set_unused(); push(head, child); continue; }
            }
            empty = false;
        }
        empty
    }
    let mut head = 0;
    {
        let _guard = MAPPER_LOCK.lock();
        let mut mapper = kernel_mapper();
        for index in 0..pages as u64 {
            let page = Page::<Size4KiB>::containing_address(VirtAddr::new(base + index * FRAME));
            if let Ok((frame, flush)) = mapper.unmap(page) { flush.flush(); push(&mut head, frame.start_address().as_u64()); }
        }
        let root = Cr3::read().0.start_address().as_u64();
        // SAFETY: this process shares all kernel-half PDPTs; these roots persist.
        let root = unsafe { &mut *(virt(root) as *mut PageTable) };
        for slot in 400..416 { if root[slot].flags().contains(F::PRESENT) { prune(root[slot].addr().as_u64(), 3, &mut head); } }
    }
    let online = crate::smp::online();
    crate::paging::shootdown(if online >= 64 { u64::MAX } else { (1u64 << online) - 1 });
    while head != 0 {
        let frame = head;
        // SAFETY: the physical direct map remains live after the old heap aliases vanish.
        head = unsafe { (virt(frame) as *const u64).read() };
        free_phys(frame, 1);
    }
}
