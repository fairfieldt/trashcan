//! Address spaces. Every process's PML4 shares the kernel half (slots 256-511) with the kernel's own table;
//! those slots all point at PDPTs allocated at boot, so later kernel mappings show up everywhere. The user half
//! (below `USER_TOP`) holds mappings of memory objects, each kept alive by the address space while mapped.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
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
/// Internal mapping metadata; never installed as a hardware permission.
pub const FORK_SHARED: u64 = 1 << 63;
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

#[derive(Clone)]
struct Mapping {
    len: u64,
    /// None: reserved address space with nothing behind it yet (Linux `PROT_NONE` reservations).
    mem: Option<Arc<Memory>>,
    /// Offset of the mapping's first byte in `mem`.
    offset: u64,
    /// `vm::READ | WRITE | EXECUTE`; 0 maps nothing (no access).
    flags: u64,
    /// Held while the framebuffer is mapped: the kernel log stays off it.
    _display: Option<crate::fbcon::Claim>,
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

    /// Eager fork for small Unix processes. Shared mappings keep their memory object;
    /// private aliases of the same object share one copy in the child.
    #[cfg(feature = "unix")]
    pub fn fork_into(&self, child: &Self) -> Result<(), Error> {
        let inner = self.inner.lock();
        let mut copies: BTreeMap<usize, Arc<Memory>> = BTreeMap::new();
        for (&addr, m) in &inner.mappings {
            match &m.mem {
                None => { child.reserve(m.len, addr, false)?; }
                Some(mem) => {
                    if !mem.is_ram() { return Err(Error::AccessDenied); }
                    let backing = if m.flags & FORK_SHARED != 0 { mem.clone() } else {
                        let key = Arc::as_ptr(mem) as usize;
                        if let Some(copy) = copies.get(&key) { copy.clone() } else {
                            let copy = Arc::new(Memory::anonymous(mem.size(), false, false)?);
                            let mut buffer = alloc::vec![0; 4096];
                            for off in (0..mem.size()).step_by(4096) { mem.read(off, &mut buffer)?; copy.write(off, &buffer)?; }
                            copies.insert(key, copy.clone()); copy
                        }
                    };
                    child.map(backing, m.offset, m.len, addr, m.flags)?;
                }
            }
        }
        child.inner.lock().next = inner.next;
        Ok(())
    }

    /// vfork shares backing pages while the parent is suspended. The child may
    /// only exec/exit; its independent page tables let exec detach transactionally.
    #[cfg(feature = "unix")]
    pub fn vfork_into(&self, child: &Self) -> Result<(), Error> {
        let inner = self.inner.lock();
        for (&addr, m) in &inner.mappings {
            if let Some(mem) = &m.mem {
                if !mem.is_ram() { return Err(Error::AccessDenied); }
                child.map(mem.clone(), m.offset, m.len, addr, m.flags)?;
            } else { child.reserve(m.len, addr, false)?; }
        }
        child.inner.lock().next = inner.next;
        Ok(())
    }

    /// Install a fully validated exec image without changing this process's CR3.
    /// The caller has excluded sibling threads before committing the replacement.
    #[cfg(feature = "unix")]
    pub fn replace_from(&self, image: &Self) {
        let mut old = self.inner.lock();
        let mut new = image.inner.lock();
        for index in 0..256 { core::mem::swap(&mut table(self.pml4)[index], &mut table(image.pml4)[index]); }
        core::mem::swap(&mut *old, &mut *new);
        drop(new); drop(old);
        x86_64::instructions::tlb::flush_all();
        shootdown(self.cpus.load(Ordering::Relaxed));
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

    /// Writes the page table entries of a mapping at `addr` (or none if it maps nothing).
    fn install(&self, addr: u64, m: &Mapping) -> Result<(), Error> {
        let Some(mem) = &m.mem else { return Ok(()) };
        if m.flags & (vm::READ | vm::WRITE | vm::EXECUTE) == 0 {
            return Ok(());
        }
        let mut pf = F::PRESENT | F::USER_ACCESSIBLE;
        if m.flags & vm::WRITE != 0 {
            pf |= F::WRITABLE;
        }
        if m.flags & vm::EXECUTE == 0 {
            pf |= F::NO_EXECUTE;
        }
        if mem.uncached() {
            pf |= F::NO_CACHE | F::WRITE_THROUGH;
        }
        let parent = F::PRESENT | F::WRITABLE | F::USER_ACCESSIBLE;
        let mut mapper = self.mapper();
        for i in 0..m.len / PAGE {
            let page = Page::<Size4KiB>::containing_address(VirtAddr::new(addr + i * PAGE));
            let frame = PhysFrame::containing_address(PhysAddr::new(mem.page_phys(((m.offset + i * PAGE) / PAGE) as usize)));
            // SAFETY: a user-half page of a checked range, to a frame the memory object owns or may expose.
            if unsafe { mapper.map_to_with_table_flags(page, frame, pf, parent, &mut TableFrames) }.map(|f| f.ignore()).is_err() {
                // A partially installed mapping must not outlive its backing object.
                drop(mapper);
                self.uninstall(addr, i * PAGE);
                return Err(Error::NoMemory);
            }
        }
        Ok(())
    }

    /// Removes the page table entries of [addr, addr+len); the caller shoots down other CPUs' TLBs.
    fn uninstall(&self, addr: u64, len: u64) {
        let mut mapper = self.mapper();
        for i in 0..len / PAGE {
            let page = Page::<Size4KiB>::containing_address(VirtAddr::new(addr + i * PAGE));
            if let Ok((_, f)) = mapper.unmap(page) {
                f.flush();
            }
        }
    }

    fn check_range(addr: u64, len: u64) -> Result<u64, Error> {
        let end = addr.checked_add(len).ok_or(Error::OutOfRange)?;
        if addr % PAGE != 0 || len % PAGE != 0 || len == 0 || end > USER_TOP || addr < PAGE {
            return Err(Error::OutOfRange);
        }
        Ok(end)
    }

    /// Picks a free range of `len` bytes (with a guard page after it), past mappings placed by address.
    fn place(inner: &mut Inner, len: u64) -> u64 {
        let mut a = inner.next;
        while let Some((&s, m)) = inner.mappings.range(..a + len + PAGE).next_back() {
            if s + m.len <= a {
                break;
            }
            a = s + m.len + PAGE;
        }
        inner.next = a + len + PAGE;
        a
    }

    /// Maps `len` bytes of `mem` from `offset` at `addr` (0: anywhere). Returns the address.
    pub fn map(&self, mem: Arc<Memory>, offset: u64, len: u64, addr: u64, flags: u64) -> Result<u64, Error> {
        if len == 0 || offset % PAGE != 0 || len % PAGE != 0 || addr % PAGE != 0 {
            return Err(Error::InvalidArgs);
        }
        if offset.checked_add(len).is_none_or(|e| e > mem.size()) {
            return Err(Error::OutOfRange);
        }
        self.insert(Some(mem), offset, len, addr, flags, false)
    }

    /// Reserves address space with nothing behind it (no access until `protect` gives it memory).
    pub fn reserve(&self, len: u64, addr: u64, replace: bool) -> Result<u64, Error> {
        self.insert(None, 0, len, addr, 0, replace)
    }

    /// Maps, replacing whatever was in the range if `replace` (Linux `MAP_FIXED`).
    pub fn map_replace(&self, mem: Arc<Memory>, offset: u64, len: u64, addr: u64, flags: u64) -> Result<u64, Error> {
        if offset.checked_add(len).is_none_or(|e| e > mem.size()) {
            return Err(Error::OutOfRange);
        }
        self.insert(Some(mem), offset, len, addr, flags, true)
    }

    fn insert(&self, mem: Option<Arc<Memory>>, offset: u64, len: u64, addr: u64, flags: u64, replace: bool) -> Result<u64, Error> {
        let mut inner = self.inner.lock();
        let addr = if addr == 0 { Self::place(&mut inner, len) } else { addr };
        let end = Self::check_range(addr, len)?;
        let removed = if replace {
            self.remove_locked(&mut inner, addr, len)
        } else {
            if let Some((&s, m)) = inner.mappings.range(..end).next_back() {
                if s + m.len > addr {
                    return Err(Error::Busy);
                }
            }
            Vec::new()
        };
        let _display = mem.as_ref().filter(|m| m.is_display()).map(|_| crate::fbcon::Claim::new());
        let m = Mapping { len, mem, offset, flags, _display };
        let r = self.install(addr, &m);
        if r.is_ok() {
            inner.mappings.insert(addr, m);
        }
        drop(inner);
        if !removed.is_empty() || r.is_err() {
            self.finish(removed);
        }
        r.map(|_| addr)
    }

    /// Splits the mapping containing `at` (if any) so that a mapping starts at `at`.
    fn split(inner: &mut Inner, at: u64) {
        let Some((&s, m)) = inner.mappings.range(..at).next_back() else { return };
        if s + m.len <= at {
            return;
        }
        let mut right = m.clone();
        let left_len = at - s;
        right.len = m.len - left_len;
        right.offset = m.offset + left_len;
        inner.mappings.get_mut(&s).unwrap().len = left_len;
        inner.mappings.insert(at, right);
    }

    fn remove_locked(&self, inner: &mut Inner, addr: u64, len: u64) -> Vec<Mapping> {
        let end = addr + len;
        Self::split(inner, addr);
        Self::split(inner, end);
        let keys: Vec<u64> = inner.mappings.range(addr..end).map(|(&k, _)| k).collect();
        let mut out = Vec::new();
        for k in keys {
            let m = inner.mappings.remove(&k).unwrap();
            self.uninstall(k, m.len);
            out.push(m);
        }
        out
    }

    /// After the page tables changed: flush the other CPUs' TLBs, then free what was unmapped. Called without
    /// `inner` held, since another CPU may be spinning on it with interrupts off and couldn't take the IPI.
    fn finish(&self, removed: Vec<Mapping>) {
        shootdown(self.cpus.load(Ordering::Relaxed));
        drop(removed);
    }

    /// Removes the whole mapping that starts at `addr` (dross `VM_UNMAP`).
    pub fn unmap(&self, addr: u64, len: u64) -> Result<(), Error> {
        let mut inner = self.inner.lock();
        match inner.mappings.get(&addr) {
            Some(m) if m.len == len => {}
            _ => return Err(Error::NotFound),
        }
        let removed = self.remove_locked(&mut inner, addr, len);
        drop(inner);
        self.finish(removed);
        Ok(())
    }

    /// Removes whatever is mapped in [addr, addr+len), splitting mappings at the edges (Linux `munmap`).
    pub fn unmap_range(&self, addr: u64, len: u64) -> Result<(), Error> {
        Self::check_range(addr, len)?;
        let mut inner = self.inner.lock();
        let removed = self.remove_locked(&mut inner, addr, len);
        drop(inner);
        self.finish(removed);
        Ok(())
    }

    /// Changes the access of [addr, addr+len), which must be covered by mappings (Linux `mprotect`). Reserved
    /// ranges get zeroed memory when they first become accessible.
    pub fn protect(&self, addr: u64, len: u64, flags: u64) -> Result<(), Error> {
        let end = Self::check_range(addr, len)?;
        let mut inner = self.inner.lock();
        // Covered without holes?
        let mut at = addr;
        for (&s, m) in inner.mappings.range(..end) {
            if s + m.len <= at {
                continue;
            }
            if s > at {
                return Err(Error::NotFound);
            }
            at = s + m.len;
            if at >= end {
                break;
            }
        }
        if at < end {
            return Err(Error::NotFound);
        }
        // Validate the entire range before changing any of its mappings.
        if flags & vm::WRITE != 0 && inner.mappings.range(..end).any(|(&s, m)| {
            s + m.len > addr && m.mem.as_ref().is_some_and(|mm| !mm.writable())
        }) {
            return Err(Error::AccessDenied);
        }
        Self::split(&mut inner, addr);
        Self::split(&mut inner, end);
        let keys: Vec<u64> = inner.mappings.range(addr..end).map(|(&k, _)| k).collect();
        let mut prepared = Vec::new();
        for &k in &keys {
            let mut m = inner.mappings.get(&k).unwrap().clone();
            if m.mem.is_none() && flags != 0 {
                m.mem = Some(Arc::new(Memory::anonymous(m.len, false, false)?));
                m.offset = 0;
            }
            m.flags = flags | (m.flags & FORK_SHARED);
            prepared.push((k, m));
        }
        for (index, (k, m)) in prepared.iter().enumerate() {
            self.uninstall(*k, m.len);
            if let Err(e) = self.install(*k, m) {
                // Page tables are retained on unmap, so restoring the old entries
                // requires no new table frames. Keep new backing alive through the
                // shootdown before returning it to the allocator.
                for (old_k, new) in &prepared[..=index] {
                    self.uninstall(*old_k, new.len);
                    let old = inner.mappings.get(old_k).unwrap();
                    self.install(*old_k, old).expect("restore existing page tables");
                }
                drop(inner);
                self.finish(Vec::new());
                drop(prepared);
                return Err(e);
            }
        }
        for (k, m) in prepared {
            inner.mappings.insert(k, m);
        }
        drop(inner);
        self.finish(Vec::new());
        Ok(())
    }

    /// Runs `f` on user memory [ptr, ptr+len) if mappings cover all of it with the needed access. The lock
    /// keeps them in place meanwhile. This space must be the loaded one.
    pub fn with_user<R>(&self, ptr: u64, len: u64, write: bool, f: impl FnOnce(*mut u8) -> R) -> Result<R, Error> {
        let inner = self.inner.lock();
        if len == 0 {
            return Ok(f(ptr as *mut u8));
        }
        let end = ptr.checked_add(len).ok_or(Error::InvalidArgs)?;
        let need = if write { vm::WRITE } else { vm::READ };
        let mut at = ptr;
        let (&first, _) = inner.mappings.range(..=ptr).next_back().ok_or(Error::InvalidArgs)?;
        for (&s, m) in inner.mappings.range(first..) {
            if s > at {
                return Err(Error::InvalidArgs);
            }
            let writable = m.mem.as_ref().is_some_and(|mm| mm.writable());
            if m.flags & need == 0 || m.mem.is_none() || (write && !writable) {
                return Err(Error::InvalidArgs);
            }
            at = s + m.len;
            if at >= end {
                return Ok(f(ptr as *mut u8));
            }
        }
        Err(Error::InvalidArgs)
    }

    /// The memory object and offset behind a user address (for futexes keyed by memory, and checks).
    pub fn lookup(&self, addr: u64) -> Option<(Arc<Memory>, u64)> {
        self.lookup_mapping(addr).map(|(m, off, _, _)| (m, off))
    }

    /// The memory object behind a user address, the offset there, the mapping's flags and how many bytes of the
    /// mapping follow the address.
    pub fn lookup_mapping(&self, addr: u64) -> Option<(Arc<Memory>, u64, u64, u64)> {
        let inner = self.inner.lock();
        let (&s, m) = inner.mappings.range(..=addr).next_back()?;
        if addr >= s + m.len {
            return None;
        }
        Some((m.mem.clone()?, m.offset + (addr - s), m.flags, s + m.len - addr))
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
