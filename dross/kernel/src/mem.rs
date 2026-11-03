//! Physical memory (the `pmm` bitmap over the loader's memory map), the direct map, and MMIO mappings.

use bootinfo::{BootInfo, RegionKind, DIRECT_MAP};
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

pub fn free_bytes() -> u64 {
    FRAMES.lock().as_ref().map_or(0, |f| f.free_frames() * FRAME)
}

/// Physical frames: `pages` contiguous, aligned to `align` bytes, below `ceiling`.
pub fn alloc_phys(pages: usize, align: u64, ceiling: u64) -> Option<u64> {
    FRAMES.lock().as_mut().expect("memory not initialized").alloc(pages as u64, align, ceiling)
}

/// Zeroed pages, returned as their direct-map address. Panics when memory runs out.
pub fn alloc_zeroed(pages: usize, align: u64) -> u64 {
    let p = alloc_phys(pages, align, u64::MAX).expect("out of physical memory");
    let v = virt(p);
    // SAFETY: just allocated, in the direct map.
    unsafe { core::ptr::write_bytes(v as *mut u8, 0, pages * FRAME as usize) };
    v
}

struct PageTableFrames;

// SAFETY: hands out unused frames; the mapper zeroes the tables it makes from them.
unsafe impl FrameAllocator<Size4KiB> for PageTableFrames {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        alloc_phys(1, FRAME, u64::MAX).map(|p| PhysFrame::containing_address(PhysAddr::new(p)))
    }
}

static MAPPER_LOCK: Mutex<()> = Mutex::new(());

/// Maps `size` bytes of device memory at `phys` uncached (PCD|PWT, no execute) and returns the virtual address
/// of `phys`. Mappings are permanent.
pub fn map_mmio(phys: u64, size: u64) -> u64 {
    let _guard = MAPPER_LOCK.lock();
    let first = phys & !(FRAME - 1);
    let pages = (phys + size - first).div_ceil(FRAME);
    let base = MMIO_NEXT.fetch_add(pages * FRAME, Ordering::Relaxed);
    let (pml4, _) = Cr3::read();
    // SAFETY: the active PML4, through the direct map; MAPPER_LOCK serializes changes.
    let mut mapper = unsafe { OffsetPageTable::new(&mut *(virt(pml4.start_address().as_u64()) as *mut PageTable), VirtAddr::new(DIRECT_MAP)) };
    for i in 0..pages {
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(base + i * FRAME));
        let frame = PhysFrame::containing_address(PhysAddr::new(first + i * FRAME));
        let flags = F::PRESENT | F::WRITABLE | F::NO_CACHE | F::WRITE_THROUGH | F::NO_EXECUTE;
        // SAFETY: a fresh virtual range in the MMIO window, mapped to device memory.
        unsafe { mapper.map_to(page, frame, flags, &mut PageTableFrames).expect("map_mmio").flush() };
    }
    base + (phys - first)
}
