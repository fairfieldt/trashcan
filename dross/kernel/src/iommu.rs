//! VT-d DMA remapping. Every device starts blocked (no context entry); a device can DMA only after it is
//! attached to a domain, and then only to the IOVAs mapped there. Domains are objects: a driver gets one for its
//! device from pcid and maps its memory objects into it. Faults are logged, with the device and address.
//!
//! The Mac Pro 6,1 has one unit for all devices, no interrupt remapping, and one reserved region (RMRR) for its
//! USB controllers, which gets an identity-mapped domain so firmware's USB DMA keeps working.

use acpi::dmar::{scopes, Dmar, Remapping};
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use dross_abi::{dma, Error};
use spin::Mutex;

use crate::firmware::Acpi;
use crate::memory::Memory;
use crate::{cpu, mem, println};

const VER: usize = 0x00;
const CAP: usize = 0x08;
const ECAP: usize = 0x10;
const GCMD: usize = 0x18;
const GSTS: usize = 0x1c;
const RTADDR: usize = 0x20;
const CCMD: usize = 0x28;
const FSTS: usize = 0x34;
const FECTL: usize = 0x38;
const FEDATA: usize = 0x3c;
const FEADDR: usize = 0x40;
const FEUADDR: usize = 0x44;

const GCMD_TE: u32 = 1 << 31;
const GCMD_SRTP: u32 = 1 << 30;
const GCMD_WBF: u32 = 1 << 27;
/// GSTS bits that persist and must be written back with each command.
const GSTS_PERSIST: u32 = 0x96ff_ffff;

const PAGE: u64 = 4096;

struct Unit {
    regs: u64,
    cap: u64,
    ecap: u64,
    /// Page-table levels (3: 39-bit, 4: 48-bit IOVAs).
    levels: u32,
    root: u64,
    /// Context table of each bus (physical, 0: none yet).
    context: [u64; 256],
}

impl Unit {
    fn r32(&self, off: usize) -> u32 {
        // SAFETY: the unit's register page, mapped uncached.
        unsafe { ((self.regs as usize + off) as *const u32).read_volatile() }
    }
    fn w32(&self, off: usize, v: u32) {
        // SAFETY: as above.
        unsafe { ((self.regs as usize + off) as *mut u32).write_volatile(v) }
    }
    fn r64(&self, off: usize) -> u64 {
        // SAFETY: as above.
        unsafe { ((self.regs as usize + off) as *const u64).read_volatile() }
    }
    fn w64(&self, off: usize, v: u64) {
        // SAFETY: as above.
        unsafe { ((self.regs as usize + off) as *mut u64).write_volatile(v) }
    }

    fn coherent(&self) -> bool {
        self.ecap & 1 != 0
    }

    /// Makes table writes visible to a unit that doesn't snoop the CPU caches.
    fn flush(&self, virt: u64, len: u64) {
        if self.coherent() {
            return;
        }
        let mut a = virt & !63;
        while a < virt + len {
            // SAFETY: a cache line of a page table in the direct map.
            unsafe { core::arch::asm!("clflush [{}]", in(reg) a) };
            a += 64;
        }
        // SAFETY: orders the flushes before the device is told.
        unsafe { core::arch::asm!("mfence") };
    }

    fn command(&self, bit: u32) {
        let s = self.r32(GSTS) & GSTS_PERSIST;
        self.w32(GCMD, s | bit);
        self.wait_status(bit, true);
    }

    fn wait_status(&self, bit: u32, set: bool) {
        for _ in 0..10_000_000 {
            if (self.r32(GSTS) & bit != 0) == set {
                return;
            }
            core::hint::spin_loop();
        }
        panic!("iommu: status bit {bit:#x} never became {set}");
    }

    fn flush_write_buffer(&self) {
        if self.cap & 1 << 4 != 0 {
            self.command(GCMD_WBF);
            self.wait_status(GCMD_WBF, false);
        }
    }

    /// Global context-cache invalidation, then a global IOTLB flush.
    fn invalidate_all(&self) {
        self.flush_write_buffer();
        self.w64(CCMD, 1 << 63 | 1 << 61);
        while self.r64(CCMD) & 1 << 63 != 0 {
            core::hint::spin_loop();
        }
        self.flush_iotlb();
    }

    fn flush_iotlb(&self) {
        self.flush_write_buffer();
        let iotlb = ((self.ecap >> 8 & 0x3ff) * 16 + 8) as usize;
        // Global invalidation, draining reads and writes.
        self.w64(iotlb, 1 << 63 | 1 << 60 | 1 << 49 | 1 << 48);
        while self.r64(iotlb) & 1 << 63 != 0 {
            core::hint::spin_loop();
        }
    }

    /// The unit can force snooping per mapping (ECAP.SC).
    fn snoop_control(&self) -> bool {
        self.ecap & 1 << 7 != 0
    }

    fn caching_mode(&self) -> bool {
        self.cap & 1 << 7 != 0
    }
}

static UNIT: Mutex<Option<Unit>> = Mutex::new(None);
static NEXT_DOMAIN: AtomicU16 = AtomicU16::new(1);
pub static FAULTS: AtomicU64 = AtomicU64::new(0);

fn table(phys: u64) -> &'static mut [u64; 512] {
    // SAFETY: page tables are frames in the direct map.
    unsafe { &mut *(mem::virt(phys) as *mut [u64; 512]) }
}

pub fn present() -> bool {
    UNIT.lock().is_some()
}

/// Sets up the remapping unit and turns translation on. Devices are blocked until attached.
pub fn init(acpi: &Acpi) {
    let Some(t) = &acpi.dmar else {
        println!("iommu: no DMAR table, DMA is not remapped");
        return;
    };
    let d = Dmar::parse(t).unwrap();
    let units: Vec<_> = d.structures().flatten().filter_map(|s| match s {
        Remapping::Unit { base, include_all, .. } => Some((base, include_all)),
        _ => None,
    }).collect();
    let Some(&(base, _)) = units.first() else { return };
    // One unit covers every device whether it says include-all (the Mac Pro) or lists scopes (QEMU 11).
    assert!(units.len() == 1, "iommu: only one remapping unit is supported: {:x?}", units);
    let regs = mem::map_mmio(base, 0x1000);
    let mut u = Unit { regs, cap: 0, ecap: 0, levels: 0, root: 0, context: [0; 256] };
    u.cap = u.r64(CAP);
    u.ecap = u.r64(ECAP);
    let sagaw = u.cap >> 8 & 0x1f;
    u.levels = if sagaw & 1 << 2 != 0 { 4 } else if sagaw & 1 << 1 != 0 { 3 } else { panic!("iommu: no 3- or 4-level tables (sagaw {sagaw:#x})") };
    let iotlb_end = (u.ecap >> 8 & 0x3ff) * 16 + 16;
    let fro_end = (u.cap >> 24 & 0x3ff) * 16 + ((u.cap >> 40 & 0xff) + 1) * 16;
    assert!(iotlb_end <= 0x1000 && fro_end <= 0x1000, "iommu: registers beyond the first page");
    println!(
        "iommu: unit at {:#x} version {:#x}, {}-level tables, {} domains, coherent {}, snoop control {}, caching mode {}, translation {}",
        base,
        u.r32(VER),
        u.levels,
        1u64 << (4 + 2 * (u.cap & 7)),
        u.coherent(),
        u.snoop_control(),
        u.caching_mode(),
        if u.r32(GSTS) & GCMD_TE != 0 { "already on" } else { "off" }
    );
    u.root = mem::alloc_phys_zeroed(1);
    u.flush(mem::virt(u.root), PAGE);
    u.w64(RTADDR, u.root);
    u.command(GCMD_SRTP);
    u.invalidate_all();

    // Faults to a vector on this CPU.
    let apic = cpu::this().apic_id;
    u.w32(FEDATA, crate::interrupts::IOMMU_FAULT as u32);
    u.w32(FEADDR, 0xfee0_0000 | apic << 12);
    u.w32(FEUADDR, 0);
    u.w32(FECTL, 0);
    *UNIT.lock() = Some(u);

    // Reserved regions: identity-mapped for their devices before translation starts.
    for s in d.structures().flatten() {
        if let Remapping::Reserved { base, limit, scopes: sc, .. } = s {
            let len = (limit + 1 - base).next_multiple_of(PAGE);
            let dom = Arc::new(Domain::new());
            let m = Arc::new(Memory::device(base, len));
            dom.map(m, 0, len, base, dma::READ | dma::WRITE).expect("rmrr map");
            for sc in scopes(sc) {
                if sc.path.len() == 2 {
                    let bdf = (sc.start_bus as u16) << 8 | (sc.path[0] as u16) << 3 | sc.path[1] as u16;
                    let _ = dom.attach(bdf);
                    println!("iommu: reserved {:#x}-{:#x} identity-mapped for {:02x}:{:02x}.{}", base, limit, sc.start_bus, sc.path[0], sc.path[1]);
                }
            }
            core::mem::forget(dom);
        }
    }
    let u = UNIT.lock();
    let u = u.as_ref().unwrap();
    u.command(GCMD_TE);
    println!("iommu: translation on; unattached devices are blocked");
}

/// The fault event interrupt: log and clear every recorded fault.
pub fn fault_interrupt() {
    let g = UNIT.lock();
    let Some(u) = g.as_ref() else { return };
    let fsts = u.r32(FSTS);
    let fro = ((u.cap >> 24 & 0x3ff) * 16) as usize;
    let n = ((u.cap >> 40 & 0xff) + 1) as usize;
    for i in 0..n {
        let hi = u.r64(fro + i * 16 + 8);
        if hi & 1 << 63 == 0 {
            continue;
        }
        let lo = u.r64(fro + i * 16);
        let sid = hi & 0xffff;
        println!(
            "iommu: fault: {:02x}:{:02x}.{} {} at iova {:#x}, reason {:#x}",
            sid >> 8,
            sid >> 3 & 0x1f,
            sid & 7,
            if hi >> 62 & 1 != 0 { "read" } else { "write" },
            lo & !0xfff,
            hi >> 32 & 0xff
        );
        FAULTS.fetch_add(1, Ordering::Relaxed);
        u.w64(fro + i * 16 + 8, 1 << 63);
    }
    // Clear overflow and pending.
    u.w32(FSTS, fsts & 0x7f);
}

struct DomainInner {
    mappings: BTreeMap<u64, (u64, Arc<Memory>)>,
    devices: Vec<u16>,
}

/// A device address space: second-level page tables plus what is mapped in them.
pub struct Domain {
    id: u16,
    root: u64,
    levels: u32,
    inner: Mutex<DomainInner>,
}

impl Domain {
    pub fn new() -> Domain {
        let levels = UNIT.lock().as_ref().map_or(4, |u| u.levels);
        Domain {
            id: NEXT_DOMAIN.fetch_add(1, Ordering::Relaxed),
            root: mem::alloc_phys_zeroed(1),
            levels,
            inner: Mutex::new(DomainInner { mappings: BTreeMap::new(), devices: Vec::new() }),
        }
    }

    pub fn iova_limit(&self) -> u64 {
        1 << (12 + 9 * self.levels)
    }

    fn leaf(&self, u: &Unit, iova: u64) -> *mut u64 {
        let mut t = self.root;
        for level in (1..self.levels).rev() {
            let idx = (iova >> (12 + 9 * level) & 511) as usize;
            let e = &mut table(t)[idx];
            if *e & 3 == 0 {
                let n = mem::alloc_phys_zeroed(1);
                u.flush(mem::virt(n), PAGE);
                *e = n | 3;
                u.flush(e as *mut u64 as u64, 8);
            }
            t = *e & 0x000f_ffff_ffff_f000;
        }
        &mut table(t)[(iova >> 12 & 511) as usize]
    }

    /// Maps `len` bytes of `m` from `offset` at device address `iova`.
    pub fn map(&self, m: Arc<Memory>, offset: u64, len: u64, iova: u64, flags: u64) -> Result<(), Error> {
        if len == 0 || offset % PAGE != 0 || len % PAGE != 0 || iova % PAGE != 0 || flags & 3 == 0 {
            return Err(Error::InvalidArgs);
        }
        if offset.checked_add(len).is_none_or(|e| e > m.size()) || iova.checked_add(len).is_none_or(|e| e > self.iova_limit()) {
            return Err(Error::OutOfRange);
        }
        let g = UNIT.lock();
        let u = g.as_ref().ok_or(Error::NotFound)?;
        let mut inner = self.inner.lock();
        if let Some((&s, &(l, _))) = inner.mappings.range(..iova + len).next_back() {
            if s + l > iova {
                return Err(Error::Busy);
            }
        }
        let mut perm = (flags & dma::READ != 0) as u64 | ((flags & dma::WRITE != 0) as u64) << 1;
        // Snoop: device writes marked no-snoop (a GPU's AGP aperture uses them) still go through the CPU caches,
        // so the CPU never reads a stale cached copy. Linux sets it for VFIO the same way.
        if u.snoop_control() {
            perm |= 1 << 11;
        }
        for i in 0..len / PAGE {
            let e = self.leaf(u, iova + i * PAGE);
            // SAFETY: an entry of this domain's tables.
            unsafe { *e = m.page_phys(((offset + i * PAGE) / PAGE) as usize) | perm };
            u.flush(e as u64, 8);
        }
        inner.mappings.insert(iova, (len, m));
        if u.caching_mode() {
            u.flush_iotlb();
        }
        Ok(())
    }

    pub fn unmap(&self, iova: u64, len: u64) -> Result<(), Error> {
        let g = UNIT.lock();
        let u = g.as_ref().ok_or(Error::NotFound)?;
        let mut inner = self.inner.lock();
        match inner.mappings.get(&iova) {
            Some(&(l, _)) if l == len => {}
            _ => return Err(Error::NotFound),
        }
        for i in 0..len / PAGE {
            let e = self.leaf(u, iova + i * PAGE);
            // SAFETY: as in `map`.
            unsafe { *e = 0 };
            u.flush(e as u64, 8);
        }
        u.flush_iotlb();
        // The memory may go only once the device can no longer reach it.
        let m = inner.mappings.remove(&iova);
        drop(inner);
        drop(g);
        drop(m);
        Ok(())
    }

    /// Points device `bdf` (bus << 8 | dev << 3 | fn) at this domain.
    pub fn attach(&self, bdf: u16) -> Result<(), Error> {
        let mut g = UNIT.lock();
        let u = g.as_mut().ok_or(Error::NotFound)?;
        let bus = (bdf >> 8) as usize;
        if u.context[bus] == 0 {
            let ct = mem::alloc_phys_zeroed(1);
            u.flush(mem::virt(ct), PAGE);
            u.context[bus] = ct;
            let root = table(u.root);
            root[bus * 2] = ct | 1;
            u.flush(&root[bus * 2] as *const u64 as u64, 16);
        }
        let ctx = table(u.context[bus]);
        let i = (bdf & 0xff) as usize * 2;
        if ctx[i] & 1 != 0 {
            return Err(Error::Busy);
        }
        let aw = if self.levels == 4 { 2 } else { 1 };
        ctx[i + 1] = aw | (self.id as u64) << 8;
        ctx[i] = self.root | 1;
        u.flush(&ctx[i] as *const u64 as u64, 16);
        u.invalidate_all();
        self.inner.lock().devices.push(bdf);
        Ok(())
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        let g = UNIT.lock();
        if let Some(u) = g.as_ref() {
            for &bdf in &self.inner.lock().devices {
                let ctx = table(u.context[(bdf >> 8) as usize]);
                let i = (bdf & 0xff) as usize * 2;
                ctx[i] = 0;
                ctx[i + 1] = 0;
                u.flush(&ctx[i] as *const u64 as u64, 16);
            }
            u.invalidate_all();
        }
        drop(g);
        fn free(t: u64, level: u32) {
            if level > 1 {
                for &e in table(t).iter() {
                    if e & 3 != 0 {
                        free(e & 0x000f_ffff_ffff_f000, level - 1);
                    }
                }
            }
            mem::free_phys(t, 1);
        }
        free(self.root, self.levels);
    }
}
