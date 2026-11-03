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

struct TableAllocation { phys: u64, retired: u64, _charge: crate::account::Charge }
struct TableFrames<'a>(&'a AddressSpace);
static NEXT_RETIREMENT: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "audit-test")]
static TABLES_BEFORE_FAILURE: core::sync::atomic::AtomicI64 = core::sync::atomic::AtomicI64::new(-1);

// SAFETY: hands out unused frames; the mapper zeroes the tables it makes from them.
unsafe impl FrameAllocator<Size4KiB> for TableFrames<'_> {
    fn allocate_frame(&mut self) -> Option<PhysFrame<Size4KiB>> {
        #[cfg(feature = "audit-test")]
        if TABLES_BEFORE_FAILURE.load(Ordering::Relaxed) >= 0 {
            if TABLES_BEFORE_FAILURE.fetch_sub(1, Ordering::Relaxed) == 0 { return None; }
        }
        let mut tables = self.0.tables.lock();
        tables.try_reserve(1).ok()?;
        let charge = self.0.account.reserve(PAGE).ok()?;
        let phys = mem::alloc_phys(1, PAGE, u64::MAX)?;
        tables.push(TableAllocation { phys, retired: 0, _charge: charge });
        Some(PhysFrame::containing_address(PhysAddr::new(phys)))
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
    /// Maximum permissions granted when backing entered this address space.
    max_flags: u64,
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
    _root_charge: crate::account::Charge,
    account: Arc<crate::account::Account>,
    tables: Mutex<Vec<TableAllocation>>,
    inner: Mutex<Inner>,
    /// CPUs that have had this space loaded (TLB shootdowns go to them).
    cpus: AtomicU64,
}

const AUTO_BASE: u64 = 0x0000_1000_0000_0000;

/// Fault injection runs before user processes exist and is absent from normal kernels.
#[cfg(feature = "audit-test")]
pub fn audit_test() {
    use x86_64::structures::paging::{Translate, mapper::TranslateResult};
    let present = |space: &AddressSpace, addr| space.mapper().translate_addr(VirtAddr::new(addr)).is_some();
    let s = AddressSpace::new().unwrap();
    let m = Arc::new(Memory::anonymous(2 * PAGE, false, false).unwrap());
    TABLES_BEFORE_FAILURE.store(3, Ordering::Relaxed);
    assert!(matches!(s.map(m, 0, 2 * PAGE, 0x3ff000, vm::READ), Err(Error::NoMemory)));
    assert!(!present(&s, 0x3ff000) && !present(&s, 0x400000));
    assert!(s.lookup(0x3ff000).is_none());
    TABLES_BEFORE_FAILURE.store(-1, Ordering::Relaxed);

    let s = AddressSpace::new().unwrap();
    let m = Arc::new(Memory::anonymous(PAGE, false, false).unwrap());
    s.map(m, 0, PAGE, 0x3ff000, vm::READ).unwrap();
    s.reserve(PAGE, 0x400000, false).unwrap();
    TABLES_BEFORE_FAILURE.store(0, Ordering::Relaxed);
    assert!(matches!(s.protect(0x3ff000, 2 * PAGE, vm::READ | vm::WRITE), Err(Error::NoMemory)));
    TABLES_BEFORE_FAILURE.store(-1, Ordering::Relaxed);
    assert!(present(&s, 0x3ff000) && !present(&s, 0x400000));
    assert_eq!(s.lookup_mapping(0x3ff000).unwrap().2, vm::READ);
    assert!(matches!(s.mapper().translate(VirtAddr::new(0x3ff000)), TranslateResult::Mapped { flags, .. } if !flags.contains(F::WRITABLE)));
    assert!(s.lookup(0x400000).is_none());
    s.protect(0x3ff000, 2 * PAGE, vm::READ | vm::WRITE).unwrap();
    assert!(present(&s, 0x400000));
    assert!(matches!(s.mapper().translate(VirtAddr::new(0x3ff000)), TranslateResult::Mapped { flags, .. } if flags.contains(F::WRITABLE)));
    let account = crate::account::Account::new(4 * PAGE);
    let space = AddressSpace::new().unwrap();
    let memory = Memory::anonymous_paged_in(4 * PAGE, account.clone()).unwrap();
    space.map(Arc::new(memory), 0, 4 * PAGE, 0x100000, vm::READ | vm::WRITE).unwrap();
    space.unmap_range(0x101000, 2 * PAGE).unwrap();
    assert_eq!(account.stats()[0], 2 * PAGE);
    assert!(space.lookup(0x100000).is_some() && space.lookup(0x103000).is_some());
    assert!(space.lookup(0x101000).is_none());
    space.clear();
    assert_eq!(account.stats()[0], 0);
    let explicit = Arc::new(Memory::anonymous_in(4 * PAGE, false, false, account.clone()).unwrap());
    space.map(explicit.clone(), 0, 4 * PAGE, 0x100000, vm::READ).unwrap();
    space.unmap_range(0x101000, 2 * PAGE).unwrap();
    assert_eq!(account.stats()[0], 4 * PAGE);
    space.clear();
    assert_eq!(account.stats()[0], 4 * PAGE);
    drop(explicit);
    assert_eq!(account.stats()[0], 0);
    #[cfg(feature = "unix")]
    {
        let a = AddressSpace::new().unwrap();
        let child = AddressSpace::new().unwrap();
        let parent_account = crate::account::Account::new(4 * PAGE);
        let child_account = crate::account::Account::new(4 * PAGE);
        let memory = Arc::new(Memory::anonymous_paged_in(4 * PAGE, parent_account.clone()).unwrap());
        a.map(memory.clone(), 0, 4 * PAGE, 0x100000, vm::READ | vm::WRITE).unwrap();
        a.map(memory, 0, PAGE, 0x200000, vm::READ | vm::WRITE).unwrap();
        a.unmap_range(0x102000, 2 * PAGE).unwrap();
        a.fork_into(&child, child_account.clone()).unwrap();
        let (first, off) = child.lookup(0x100000).unwrap();
        let (alias, alias_off) = child.lookup(0x200000).unwrap();
        assert_eq!(first.page_phys((off / PAGE) as usize), alias.page_phys((alias_off / PAGE) as usize));
        assert_eq!(child_account.stats()[0], 4 * PAGE);
        child.clear();
        drop(first); drop(alias);
        assert_eq!(child_account.stats()[0], 0);
        a.clear();
        assert_eq!(parent_account.stats()[0], 0);
    }
    let account = crate::account::Account::new(128 << 10);
    let sparse = AddressSpace::new_in(account.clone()).unwrap();
    let page = Arc::new(Memory::anonymous_in(PAGE, false, false, account.clone()).unwrap());
    for index in 1..=16 {
        let addr = index * (1u64 << 39);
        sparse.map(page.clone(), 0, PAGE, addr, vm::READ).unwrap();
        assert!(account.stats()[0] > 2 * PAGE);
        sparse.unmap(addr, PAGE).unwrap();
        assert_eq!(account.stats()[0], 2 * PAGE);
    }
    drop(page); drop(sparse); assert_eq!(account.stats()[0], 0);
    #[cfg(feature = "unix")]
    {
        let account = crate::account::Account::new(128 << 10);
        let running = AddressSpace::new_in(account.clone()).unwrap();
        let image = AddressSpace::new_in(account.clone()).unwrap();
        let memory = Arc::new(Memory::anonymous_in(PAGE, false, false, account.clone()).unwrap());
        running.map(memory.clone(), 0, PAGE, 0x100000, vm::READ).unwrap();
        image.map(memory.clone(), 0, PAGE, 0x4000000000, vm::READ).unwrap();
        running.replace_from(&image);
        drop(image);
        assert!(present(&running, 0x4000000000));
        running.unmap(0x4000000000, PAGE).unwrap();
        assert_eq!(account.stats()[0], 2 * PAGE);
        drop(memory); drop(running);
        assert_eq!(account.stats()[0], 0);
        crate::println!("paging: exec replacement retains live table ownership");
    }
    crate::println!("paging: physical overhead charged and empty branches reclaimed");
    crate::println!("paging: partial anonymous unmap reclamation tests passed");
    let restricted = AddressSpace::new().unwrap();
    let memory = Arc::new(Memory::anonymous(3 * PAGE, false, false).unwrap());
    restricted.map_restricted(memory, 0, 3 * PAGE, 0x100000, vm::READ, vm::READ).unwrap();
    restricted.protect(0x101000, PAGE, 0).unwrap();
    assert!(matches!(restricted.protect(0x100000, 3 * PAGE, vm::READ | vm::WRITE), Err(Error::AccessDenied)));
    assert!(matches!(restricted.protect(0x101000, PAGE, vm::EXECUTE), Err(Error::AccessDenied)));
    restricted.protect(0x100000, 3 * PAGE, vm::READ).unwrap();
    #[cfg(feature = "unix")]
    {
        let child = AddressSpace::new().unwrap();
        restricted.fork_into(&child, crate::account::Account::new(3 * PAGE)).unwrap();
        assert!(matches!(child.protect(0x100000, PAGE, vm::READ | vm::WRITE), Err(Error::AccessDenied)));
        let borrowed = AddressSpace::new().unwrap();
        restricted.vfork_into(&borrowed).unwrap();
        assert!(matches!(borrowed.protect(0x101000, PAGE, vm::READ | vm::EXECUTE), Err(Error::AccessDenied)));
    }
    crate::println!("paging: delegated mapping permission checks passed");
    crate::println!("paging: allocation failure rollback tests passed");
}

impl AddressSpace {
    #[cfg(feature = "audit-test")]
    pub fn new() -> Result<AddressSpace, Error> { Self::new_in(crate::account::current()) }
    pub fn new_in(account: Arc<crate::account::Account>) -> Result<AddressSpace, Error> {
        let charge = account.reserve(PAGE)?;
        let pml4 = mem::try_alloc_phys_zeroed(1)?;
        let k = table(kernel_pml4());
        table(pml4)[256..].copy_from_slice(&k[256..]);
        Ok(AddressSpace { pml4, _root_charge: charge, account, tables: Mutex::new(Vec::new()),
            inner: Mutex::new(Inner { mappings: BTreeMap::new(), next: AUTO_BASE }), cpus: AtomicU64::new(0) })
    }

    /// Every thread has switched away. Only the root/status page remains.
    pub fn clear(&self) {
        let mappings = core::mem::take(&mut self.inner.lock().mappings);
        table(self.pml4)[..256].fill(0);
        let tables = core::mem::take(&mut *self.tables.lock());
        for allocation in tables { mem::free_phys(allocation.phys, 1); }
        drop(mappings);
    }

    /// Unlink empty user table branches while mappings are serialized. Frames
    /// stay reserved until this batch's own cross-CPU invalidation completes.
    fn prune_tables(&self) -> u64 {
        let Ok(batch) = NEXT_RETIREMENT.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1)) else { return 0 };
        let _inner = self.inner.lock();
        let mut allocations = self.tables.lock();
        fn empty(phys: u64, level: u32, allocations: &mut [TableAllocation], batch: u64) -> bool {
            let mut unused = true;
            for entry in table(phys).iter_mut() {
                if *entry & 1 == 0 { continue; }
                if level > 1 && *entry & F::HUGE_PAGE.bits() == 0 {
                    let child = *entry & 0x000f_ffff_ffff_f000;
                    if empty(child, level - 1, allocations, batch) {
                        *entry = 0;
                        allocations.iter_mut().find(|a| a.phys == child).unwrap().retired = batch;
                        continue;
                    }
                }
                unused = false;
            }
            unused
        }
        for entry in &mut table(self.pml4)[..256] {
            if *entry & 1 == 0 { continue; }
            let child = *entry & 0x000f_ffff_ffff_f000;
            if empty(child, 3, &mut allocations, batch) {
                *entry = 0;
                allocations.iter_mut().find(|a| a.phys == child).unwrap().retired = batch;
            }
        }
        batch
    }
    fn release_tables(&self, batch: u64) {
        if batch == 0 { return; }
        let mut allocations = self.tables.lock();
        let mut index = 0;
        while index < allocations.len() {
            if allocations[index].retired == batch {
                let allocation = allocations.swap_remove(index);
                mem::free_phys(allocation.phys, 1);
                drop(allocation);
            } else { index += 1; }
        }
    }

    /// Eager fork for small Unix processes. Shared mappings keep their memory object;
    /// private aliases of the same object share one copy in the child.
    #[cfg(feature = "unix")]
    pub fn fork_into(&self, child: &Self, account: Arc<crate::account::Account>) -> Result<(), Error> {
        let inner = self.inner.lock();
        let mut copies = crate::memory::PageCopies::new();
        for (&addr, m) in &inner.mappings {
            match &m.mem {
                None => { child.reserve(m.len, addr, false)?; }
                Some(mem) => {
                    if !mem.is_ram() { return Err(Error::AccessDenied); }
                    let backing = if m.flags & FORK_SHARED != 0 { mem.clone() } else {
                        Arc::new(mem.copy_for_fork(account.clone(), &mut copies)?)
                    };
                    child.map_restricted(backing, m.offset, m.len, addr, m.flags, m.max_flags)?;
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
                child.map_restricted(mem.clone(), m.offset, m.len, addr, m.flags, m.max_flags)?;
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
        // Table ownership must follow the entries. Otherwise dropping the
        // temporary image frees the new live tables after a successful exec.
        core::mem::swap(&mut *self.tables.lock(), &mut *image.tables.lock());
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
            if unsafe { mapper.map_to_with_table_flags(page, frame, pf, parent, &mut TableFrames(self)) }.map(|f| f.ignore()).is_err() {
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
    fn place(inner: &mut Inner, len: u64) -> Result<u64, Error> {
        let mut a = inner.next;
        while let Some((&s, m)) = inner.mappings.range(..a.checked_add(len).and_then(|v| v.checked_add(PAGE)).ok_or(Error::OutOfRange)?).next_back() {
            if s + m.len <= a {
                break;
            }
            a = s.checked_add(m.len).and_then(|v| v.checked_add(PAGE)).ok_or(Error::OutOfRange)?;
        }
        Self::check_range(a, len)?;
        inner.next = a.checked_add(len).and_then(|v| v.checked_add(PAGE)).ok_or(Error::OutOfRange)?;
        Ok(a)
    }

    /// Maps owned backing with its full permission ceiling.
    pub fn map(&self, mem: Arc<Memory>, offset: u64, len: u64, addr: u64, flags: u64) -> Result<u64, Error> {
        self.map_restricted(mem, offset, len, addr, flags, vm::READ | vm::WRITE | vm::EXECUTE)
    }

    /// Maps delegated backing without permitting later protection changes to add authority.
    pub fn map_restricted(&self, mem: Arc<Memory>, offset: u64, len: u64, addr: u64, flags: u64, max_flags: u64) -> Result<u64, Error> {
        self.map_with_limit(mem, offset, len, addr, flags, max_flags, false)
    }

    /// Reserves address space with nothing behind it (no access until `protect` gives it memory).
    pub fn reserve(&self, len: u64, addr: u64, replace: bool) -> Result<u64, Error> {
        self.insert(None, 0, len, addr, 0, vm::READ | vm::WRITE | vm::EXECUTE, replace)
    }

    /// Maps delegated backing, replacing the range (Linux `MAP_FIXED`).
    pub fn map_replace_restricted(&self, mem: Arc<Memory>, offset: u64, len: u64, addr: u64, flags: u64, max_flags: u64) -> Result<u64, Error> {
        self.map_with_limit(mem, offset, len, addr, flags, max_flags, true)
    }

    fn map_with_limit(&self, mem: Arc<Memory>, offset: u64, len: u64, addr: u64, flags: u64, mut max_flags: u64, replace: bool) -> Result<u64, Error> {
        if len == 0 || offset % PAGE != 0 || len % PAGE != 0 || addr % PAGE != 0
            || flags & !(vm::READ | vm::WRITE | vm::EXECUTE | FORK_SHARED) != 0
            || max_flags & !(vm::READ | vm::WRITE | vm::EXECUTE) != 0 {
            return Err(Error::InvalidArgs);
        }
        if offset.checked_add(len).is_none_or(|e| e > mem.size()) {
            return Err(Error::OutOfRange);
        }
        if !mem.writable() { max_flags &= !vm::WRITE; }
        // x86 present user pages are readable even when only WRITE/EXECUTE was requested.
        let requested = flags & (vm::READ | vm::WRITE | vm::EXECUTE);
        let effective = requested | if requested != 0 { vm::READ } else { 0 };
        if effective & !max_flags != 0 { return Err(Error::AccessDenied); }
        self.insert(Some(mem), offset, len, addr, flags, max_flags, replace)
    }

    fn insert(&self, mem: Option<Arc<Memory>>, offset: u64, len: u64, addr: u64, flags: u64, max_flags: u64, replace: bool) -> Result<u64, Error> {
        let mut inner = self.inner.lock();
        let addr = if addr == 0 { Self::place(&mut inner, len)? } else { addr };
        let end = Self::check_range(addr, len)?;
        let removed = if replace {
            self.remove_locked(&mut inner, addr, len)?
        } else {
            if let Some((&s, m)) = inner.mappings.range(..end).next_back() {
                if s + m.len > addr {
                    return Err(Error::Busy);
                }
            }
            Vec::new()
        };
        let _display = mem.as_ref().filter(|m| m.is_display()).map(|_| crate::fbcon::Claim::new());
        let m = Mapping { len, mem, offset, flags, max_flags, _display };
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
    fn split(inner: &mut Inner, at: u64) -> Result<(), Error> {
        let Some((&s, m)) = inner.mappings.range(..at).next_back() else { return Ok(()) };
        if s + m.len <= at {
            return Ok(());
        }
        let mut right = m.clone();
        let left_len = at - s;
        right.len = m.len - left_len;
        right.offset = m.offset + left_len;
        let mut left = m.clone();
        left.len = left_len;
        if let Some(mem) = &m.mem {
            let left_view = mem.slice(m.offset, left.len)?;
            let right_view = mem.slice(right.offset, right.len)?;
            if let (Some(l), Some(r)) = (left_view, right_view) {
                left.mem = Some(Arc::new(l)); left.offset = 0;
                right.mem = Some(Arc::new(r)); right.offset = 0;
            }
        }
        inner.mappings.insert(s, left);
        inner.mappings.insert(at, right);
        Ok(())
    }

    fn remove_locked(&self, inner: &mut Inner, addr: u64, len: u64) -> Result<Vec<Mapping>, Error> {
        let end = addr + len;
        Self::split(inner, addr)?;
        Self::split(inner, end)?;
        let keys: Vec<u64> = inner.mappings.range(addr..end).map(|(&k, _)| k).collect();
        let mut out = Vec::new();
        for k in keys {
            let m = inner.mappings.remove(&k).unwrap();
            self.uninstall(k, m.len);
            out.push(m);
        }
        Ok(out)
    }

    /// After the page tables changed: flush the other CPUs' TLBs, then free what was unmapped. Called without
    /// `inner` held, since another CPU may be spinning on it with interrupts off and couldn't take the IPI.
    fn finish(&self, removed: Vec<Mapping>) {
        let batch = self.prune_tables();
        shootdown(self.cpus.load(Ordering::Acquire));
        self.release_tables(batch);
        drop(removed);
    }

    /// Release only surviving mappings of this thread-owned backing. Holding
    /// `memory` keeps every frame alive until the final TLB shootdown, without
    /// allocating a temporary removal list. Reused addresses are left alone.
    pub fn release_stack(&self, base: u64, len: u64, memory: &Arc<Memory>) {
        let end = base + len;
        let mut inner = self.inner.lock();
        let mut changed = false;
        loop {
            let key = inner.mappings.range(base..end).find_map(|(&addr, map)| {
                (addr + map.len <= end && map.offset == addr - base
                    && map.mem.as_ref().is_some_and(|m| Arc::ptr_eq(m, memory))).then_some(addr)
            });
            let Some(key) = key else { break };
            let mapping = inner.mappings.remove(&key).unwrap();
            self.uninstall(key, mapping.len);
            drop(mapping);
            changed = true;
        }
        drop(inner);
        if changed { self.finish(Vec::new()); }
    }

    /// Removes the whole mapping that starts at `addr` (dross `VM_UNMAP`).
    pub fn unmap(&self, addr: u64, len: u64) -> Result<(), Error> {
        let mut inner = self.inner.lock();
        match inner.mappings.get(&addr) {
            Some(m) if m.len == len => {}
            _ => return Err(Error::NotFound),
        }
        let removed = self.remove_locked(&mut inner, addr, len)?;
        drop(inner);
        self.finish(removed);
        Ok(())
    }

    /// Removes whatever is mapped in [addr, addr+len), splitting mappings at the edges (Linux `munmap`).
    pub fn unmap_range(&self, addr: u64, len: u64) -> Result<(), Error> {
        Self::check_range(addr, len)?;
        let mut inner = self.inner.lock();
        let removed = self.remove_locked(&mut inner, addr, len)?;
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
        // Validate every mapping's ceiling before changing any part of the range.
        if flags & !(vm::READ | vm::WRITE | vm::EXECUTE) != 0 { return Err(Error::InvalidArgs); }
        let effective = flags | if flags != 0 { vm::READ } else { 0 };
        if inner.mappings.range(..end).any(|(&s, m)| {
            s + m.len > addr && (effective & !m.max_flags != 0
                || (flags & vm::WRITE != 0 && m.mem.as_ref().is_some_and(|mm| !mm.writable())))
        }) {
            return Err(Error::AccessDenied);
        }
        Self::split(&mut inner, addr)?;
        Self::split(&mut inner, end)?;
        let keys: Vec<u64> = inner.mappings.range(addr..end).map(|(&k, _)| k).collect();
        let mut prepared = Vec::new();
        for &k in &keys {
            let mut m = inner.mappings.get(&k).unwrap().clone();
            if m.mem.is_none() && flags != 0 {
                m.mem = Some(Arc::new(Memory::anonymous_paged(m.len)?));
                m.offset = 0;
            }
            m.flags = flags | (m.flags & FORK_SHARED);
            prepared.push((k, m));
        }
        for (index, (k, m)) in prepared.iter().enumerate() {
            self.uninstall(*k, m.len);
            if let Err(e) = self.install(*k, m) {
                // Page tables are retained until this transaction finishes, so restoring the old entries
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
    fn drop(&mut self) { self.clear(); mem::free_phys(self.pml4, 1); }
}

/// Flushes the TLBs of the CPUs in `mask` (other than this one) and waits until they have. While waiting,
/// serves requests aimed at this CPU, so two CPUs shooting at each other don't deadlock.
static SHOOTDOWN_LOCK: Mutex<()> = Mutex::new(());
pub fn shootdown(mask: u64) {
    let _serial = loop {
        if let Some(guard) = SHOOTDOWN_LOCK.try_lock() { break guard; }
        serve_shootdown();
        core::hint::spin_loop();
    };
    // Leaf invalidations performed while unlinking mappings precede pruning
    // their empty parent tables. Flush the local paging-structure caches too,
    // before those table frames can be reused for unrelated data or mappings.
    x86_64::instructions::tlb::flush_all();
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
