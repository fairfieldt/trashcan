//! Address spaces. Every process's PML4 shares the kernel half (slots 256-511) with the kernel's own table;
//! those slots all point at PDPTs allocated at boot, so later kernel mappings show up everywhere. The user half
//! (below `USER_TOP`) holds mappings of memory objects, each kept alive by the address space while mapped.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use dross_abi::{vm, Error, USER_TOP};
use spin::Mutex;
use x86_64::structures::paging::{
    FrameAllocator, Mapper, OffsetPageTable, Page, PageTable, PageTableFlags as F, PhysFrame, Size4KiB,
};
use x86_64::{PhysAddr, VirtAddr};

use crate::memory::Memory;
use crate::mem::{self, virt};
use crate::{apic, cpu, interrupts};

const PAGE: u64 = 4096;
static KERNEL_PML4: AtomicU64 = AtomicU64::new(0);

pub fn kernel_pml4() -> u64 {
    KERNEL_PML4.load(Ordering::Relaxed)
}

fn table(phys: u64) -> &'static mut [u64; 512] {
    // SAFETY: page tables are frames in the direct map.
    unsafe { &mut *(virt(phys) as *mut [u64; 512]) }
}

/// After the APs are up: drop the identity map and give every kernel-half slot a PDPT.
pub fn init_kernel_space() {
    let pml4 = x86_64::registers::control::Cr3::read().0.start_address().as_u64();
    let t = table(pml4);
    t[0] = 0;
    for slot in 256..512 {
        if t[slot] & 1 == 0 {
            t[slot] = mem::alloc_phys_zeroed(1) | (F::PRESENT | F::WRITABLE).bits();
        }
    }
    KERNEL_PML4.store(pml4, Ordering::Relaxed);
    x86_64::instructions::tlb::flush_all();
    crate::println!("paging: identity map gone, kernel half shared from {:#x}", pml4);
}

struct TableFrames;

// SAFETY: hands out unused frames; the mapper zeroes the tables it makes from them.
unsafe impl FrameAllocator<Size4KiB> for TableFrames {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        mem::alloc_phys(1, PAGE, u64::MAX).map(|p| PhysFrame::containing_address(PhysAddr::new(p)))
    }
}

struct Mapping {
    len: u64,
    mem: Arc<Memory>,
}

struct Inner {
    mappings: BTreeMap<u64, Mapping>,
    /// Where `map` places mappings the caller doesn't place.
    next: u64,
}

pub struct AddressSpace {
    pml4: u64,
    inner: Mutex<Inner>,
    /// CPUs that have had this space loaded (TLB shootdowns go to them).
    cpus: AtomicU64,
}

const AUTO_BASE: u64 = 0x0000_1000_0000_0000;

impl AddressSpace {
    pub fn new() -> AddressSpace {
        let pml4 = mem::alloc_phys_zeroed(1);
        let k = table(kernel_pml4());
        table(pml4)[256..].copy_from_slice(&k[256..]);
        AddressSpace { pml4, inner: Mutex::new(Inner { mappings: BTreeMap::new(), next: AUTO_BASE }), cpus: AtomicU64::new(0) }
    }

    pub fn pml4(&self) -> u64 {
        self.pml4
    }

    pub fn loaded_on(&self, cpu: u32) {
        self.cpus.fetch_or(1 << cpu, Ordering::Relaxed);
    }

    fn mapper(&self) -> OffsetPageTable<'static> {
        // SAFETY: this space's PML4 in the direct map; callers hold `inner`.
        unsafe { OffsetPageTable::new(&mut *(virt(self.pml4) as *mut PageTable), VirtAddr::new(bootinfo::DIRECT_MAP)) }
    }

    /// Maps `len` bytes of `mem` from `offset` at `addr` (0: anywhere). Returns the address.
    pub fn map(&self, mem: Arc<Memory>, offset: u64, len: u64, addr: u64, flags: u64) -> Result<u64, Error> {
        if len == 0 || offset % PAGE != 0 || len % PAGE != 0 || addr % PAGE != 0 {
            return Err(Error::InvalidArgs);
        }
        if offset.checked_add(len).is_none_or(|e| e > mem.size()) {
            return Err(Error::OutOfRange);
        }
        let mut inner = self.inner.lock();
        let addr = if addr == 0 {
            let a = inner.next;
            // A guard page between automatic mappings.
            inner.next = a + len + PAGE;
            a
        } else {
            addr
        };
        let end = addr.checked_add(len).ok_or(Error::OutOfRange)?;
        if end > USER_TOP || addr < PAGE {
            return Err(Error::OutOfRange);
        }
        // No overlap with the mapping before or after.
        if let Some((&s, m)) = inner.mappings.range(..end).next_back() {
            if s + m.len > addr {
                return Err(Error::Busy);
            }
        }
        let mut pf = F::PRESENT | F::USER_ACCESSIBLE;
        if flags & vm::WRITE != 0 {
            pf |= F::WRITABLE;
        }
        if flags & vm::EXECUTE == 0 {
            pf |= F::NO_EXECUTE;
        }
        if mem.uncached() {
            pf |= F::NO_CACHE | F::WRITE_THROUGH;
        }
        let parent = F::PRESENT | F::WRITABLE | F::USER_ACCESSIBLE;
        let mut mapper = self.mapper();
        for i in 0..len / PAGE {
            let page = Page::<Size4KiB>::containing_address(VirtAddr::new(addr + i * PAGE));
            let frame = PhysFrame::containing_address(PhysAddr::new(mem.page_phys((offset / PAGE + i) as usize)));
            // SAFETY: user-half page in a checked free range, to a frame the memory object owns or may expose.
            unsafe { mapper.map_to_with_table_flags(page, frame, pf, parent, &mut TableFrames).map_err(|_| Error::NoMemory)?.ignore() };
        }
        inner.mappings.insert(addr, Mapping { len, mem });
        Ok(addr)
    }

    /// Removes the whole mapping that starts at `addr`.
    pub fn unmap(&self, addr: u64, len: u64) -> Result<(), Error> {
        let mut inner = self.inner.lock();
        match inner.mappings.get(&addr) {
            Some(m) if m.len == len => {}
            _ => return Err(Error::NotFound),
        }
        let m = inner.mappings.remove(&addr).unwrap();
        let mut mapper = self.mapper();
        for i in 0..m.len / PAGE {
            let page = Page::<Size4KiB>::containing_address(VirtAddr::new(addr + i * PAGE));
            if let Ok((_, f)) = mapper.unmap(page) {
                f.flush();
            }
        }
        shootdown(self.cpus.load(Ordering::Relaxed));
        drop(inner);
        drop(m);
        Ok(())
    }

    /// Runs `f` on user memory [ptr, ptr+len) if one mapping covers it with the needed access. The lock keeps
    /// the mapping in place meanwhile. This space must be the loaded one.
    pub fn with_user<R>(&self, ptr: u64, len: u64, write: bool, f: impl FnOnce(*mut u8) -> R) -> Result<R, Error> {
        let inner = self.inner.lock();
        if len == 0 {
            return Ok(f(ptr as *mut u8));
        }
        let end = ptr.checked_add(len).ok_or(Error::InvalidArgs)?;
        let (&s, m) = inner.mappings.range(..=ptr).next_back().ok_or(Error::InvalidArgs)?;
        if end > s + m.len || (write && !m.mem.writable()) {
            return Err(Error::InvalidArgs);
        }
        Ok(f(ptr as *mut u8))
    }
}

impl Drop for AddressSpace {
    fn drop(&mut self) {
        // Free the user half's page tables; the frames they map belong to memory objects.
        fn free(table_phys: u64, level: u32) {
            if level > 1 {
                for e in table(table_phys).iter() {
                    if e & 1 != 0 && e & F::HUGE_PAGE.bits() == 0 {
                        free(e & 0x000f_ffff_ffff_f000, level - 1);
                    }
                }
            }
            mem::free_phys(table_phys, 1);
        }
        for e in table(self.pml4)[..256].iter() {
            if e & 1 != 0 {
                free(e & 0x000f_ffff_ffff_f000, 3);
            }
        }
        mem::free_phys(self.pml4, 1);
    }
}

/// Flushes the TLBs of the CPUs in `mask` (other than this one) and waits until they have. While waiting,
/// serves requests aimed at this CPU, so two CPUs shooting at each other don't deadlock.
pub fn shootdown(mask: u64) {
    let me = cpu::this().index;
    let mut sent = 0u64;
    for i in 0..cpu::MAX_CPUS {
        if mask & 1 << i == 0 || i as u32 == me {
            continue;
        }
        if let Some(pc) = cpu::get(i) {
            pc.sched.flush_pending.store(true, Ordering::Release);
            apic::send_ipi(pc.apic_id, interrupts::SHOOTDOWN as u64);
            sent |= 1 << i;
        }
    }
    for i in 0..cpu::MAX_CPUS {
        if sent & 1 << i == 0 {
            continue;
        }
        let pc = cpu::get(i).unwrap();
        while pc.sched.flush_pending.load(Ordering::Acquire) {
            serve_shootdown();
            core::hint::spin_loop();
        }
    }
}

pub fn serve_shootdown() {
    let pc = cpu::this();
    if pc.sched.flush_pending.load(Ordering::Acquire) {
        x86_64::instructions::tlb::flush_all();
        pc.sched.flush_pending.store(false, Ordering::Release);
    }
}
