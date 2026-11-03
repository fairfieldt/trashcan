//! The dross backend: a PCI function a driver opened through pcid, on dross (the trashcan microkernel).
//!
//! Config space goes through pcid, which filters writes (IDs, BARs, ROM and the MSI/MSI-X capabilities are its
//! own). BARs are memory objects pcid hands over, mapped uncached. The function has an IOMMU domain of its own;
//! [`DrossDma`] allocates memory objects and maps them there. Interrupts are notifications: pcid points the MSI
//! or an MSI-X vector at one.

#[allow(unused_imports)]
use crate::prelude::*;
use core::ops::Range;
use core::ptr::NonNull;
use core::time::Duration;
use dross_abi::{dma, vm};
use dross_rt::pci::{Bar, Device, Pci};
use dross_rt::sys::{self, Handle};
use dross_rt::Lock;

use crate::dma::{iova_align, DmaAccess, DmaBuffer, DmaSpace};
use crate::irq::{IrqEvent, IrqWait};
use crate::{Error, Mmio, PciAddress, PciDevice, Region, RegionInfo, ResetCapability, Result};

/// Set once a Linux program on dross opened a device: native objects then also get Linux fds.
static LINUX: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// dross starts Linux programs with `DROSS_BOOTSTRAP` naming their bootstrap channel (handle 1).
#[cfg(feature = "std")]
pub fn started_by_dross() -> bool {
    std::env::var_os("DROSS_BOOTSTRAP").is_some()
}

/// Opens `address` through the pcid connection named `pci` in this program's bootstrap message.
#[cfg(feature = "std")]
pub fn open_from_env(address: PciAddress) -> Result<(DrossDevice, DrossDma)> {
    static PCI: std::sync::OnceLock<Option<Pci>> = std::sync::OnceLock::new();
    let pci = PCI.get_or_init(|| {
        let h: u32 = std::env::var("DROSS_BOOTSTRAP").ok()?.parse().ok()?;
        let b = dross_rt::bootstrap::Bootstrap::read(Handle(h)).ok()?;
        b.handle("pci").map(Pci::new)
    });
    let pci = pci.as_ref().ok_or_else(|| Error::NotFound(String::from("no pcid connection in the bootstrap message")))?;
    LINUX.store(true, core::sync::atomic::Ordering::Relaxed);
    open(pci, address)
}

fn os(op: &'static str) -> impl Fn(dross_abi::Error) -> Error {
    move |e| Error::Os { op, errno: e as i64 as i32 }
}

/// Opens `address` through the pcid connection `pci`: the function and its DMA space.
pub fn open(pci: &Pci, address: PciAddress) -> Result<(DrossDevice, DrossDma)> {
    if address.domain != 0 {
        return Err(Error::NotFound(format!("{address}: only segment 0")));
    }
    let bdf = (address.bus as u16) << 8 | (address.device as u16) << 3 | address.function as u16;
    let dev = pci.open(bdf).map_err(os("pcid open"))?;
    let space = dev.dma_space().map_err(os("pcid domain"))?;
    let iova = alloc::sync::Arc::new(Lock::new(crate::iova::IovaAllocator::new(&[dross_rt::pci::DmaSpace::FIRST..IOVA_END])));
    let dma = DrossDma { domain: space.domain(), iova };
    Ok((DrossDevice { address, dev, bars: Lock::new(Default::default()) }, dma))
}

pub struct DrossDevice {
    address: PciAddress,
    dev: Device,
    /// Each BAR mapped once (address, length, memory object), on first use.
    bars: Lock<[Option<(u64, usize, Handle)>; 6]>,
}

impl DrossDevice {
    fn bar(&self, n: u8) -> Result<Bar> {
        let b = *self.dev.info.bars.get(n as usize).ok_or_else(|| Error::Invalid(format!("BAR{n}")))?;
        if b.size == 0 {
            return Err(Error::NotFound(format!("{}: BAR{n}", self.address)));
        }
        Ok(b)
    }

    fn mapped(&self, n: u8) -> Result<(u64, usize)> {
        self.mapped_bar(n).map(|(a, l, _)| (a, l))
    }

    fn mapped_bar(&self, n: u8) -> Result<(u64, usize, Handle)> {
        if let Some(m) = self.bars.with(|b| b.get(n as usize).copied().flatten()) {
            return Ok(m);
        }
        let bar = self.bar(n)?;
        if bar.flags & Bar::IO != 0 {
            return Err(Error::Unsupported(format!("{}: BAR{n} is I/O space", self.address)));
        }
        let mmio = self.dev.map_bar(n as usize).map_err(os("pcid map BAR"))?;
        let m = (mmio.as_ptr() as u64, mmio.len(), mmio.mem);
        // The mapping stays for the life of the process.
        core::mem::forget(mmio);
        self.bars.with(|b| b[n as usize] = Some(m));
        Ok(m)
    }

    fn vector(&self, index: u32) -> Result<IrqEvent> {
        let n = sys::notify_create().map_err(os("notification"))?;
        let ev = self.dev.enable_vector(n, 1, index).map_err(os("pcid MSI"))?;
        Ok(IrqEvent::new(Box::new(Notification { n: ev.notification, _irq: ev.irq, fd: core::sync::atomic::AtomicI32::new(-1) })))
    }
}

impl PciDevice for DrossDevice {
    fn address(&self) -> PciAddress {
        self.address
    }

    fn region_info(&self, region: Region) -> Result<RegionInfo> {
        Ok(match region {
            Region::Config => RegionInfo { region, size: 4096, readable: true, writable: true, mappable: false },
            Region::Bar(n) => {
                let b = self.dev.info.bars.get(n as usize).copied().unwrap_or_default();
                RegionInfo { region, size: b.size, readable: true, writable: true, mappable: b.size != 0 && b.flags & Bar::IO == 0 }
            }
            Region::Rom => RegionInfo { region, size: 0, readable: false, writable: false, mappable: false },
        })
    }

    fn read(&self, region: Region, offset: u64, buf: &mut [u8]) -> Result<()> {
        match region {
            Region::Config => {
                let mut at = 0;
                while at < buf.len() {
                    let off = offset + at as u64;
                    let size = if off % 4 == 0 && buf.len() - at >= 4 { 4 } else if off % 2 == 0 && buf.len() - at >= 2 { 2 } else { 1 };
                    let v = self.dev.config_read(off as u16, size as u8).map_err(os("pcid config read"))?;
                    buf[at..at + size].copy_from_slice(&v.to_le_bytes()[..size]);
                    at += size;
                }
                Ok(())
            }
            Region::Bar(n) => {
                let (base, len) = self.mapped(n)?;
                if offset + buf.len() as u64 > len as u64 {
                    return Err(Error::Invalid(format!("BAR{n} read past {len:#x}")));
                }
                for (i, b) in buf.iter_mut().enumerate() {
                    // SAFETY: inside the mapped BAR.
                    *b = unsafe { ((base + offset + i as u64) as *const u8).read_volatile() };
                }
                Ok(())
            }
            Region::Rom => Err(Error::Unsupported(format!("{}: ROM through pcid", self.address))),
        }
    }

    fn write(&self, region: Region, offset: u64, data: &[u8]) -> Result<()> {
        match region {
            Region::Config => {
                let mut at = 0;
                while at < data.len() {
                    let off = offset + at as u64;
                    let size = if off % 4 == 0 && data.len() - at >= 4 { 4 } else if off % 2 == 0 && data.len() - at >= 2 { 2 } else { 1 };
                    let mut v = [0u8; 4];
                    v[..size].copy_from_slice(&data[at..at + size]);
                    self.dev.config_write(off as u16, size as u8, u32::from_le_bytes(v)).map_err(os("pcid config write"))?;
                    at += size;
                }
                Ok(())
            }
            Region::Bar(n) => {
                let (base, len) = self.mapped(n)?;
                if offset + data.len() as u64 > len as u64 {
                    return Err(Error::Invalid(format!("BAR{n} write past {len:#x}")));
                }
                for (i, b) in data.iter().enumerate() {
                    // SAFETY: inside the mapped BAR.
                    unsafe { ((base + offset + i as u64) as *mut u8).write_volatile(*b) };
                }
                Ok(())
            }
            Region::Rom => Err(Error::Unsupported(format!("{}: ROM through pcid", self.address))),
        }
    }

    fn map(&self, region: Region) -> Result<Mmio> {
        let Region::Bar(n) = region else {
            return Err(Error::Unsupported(format!("{}: mapping {region}", self.address)));
        };
        let (base, len) = self.mapped(n)?;
        // SAFETY: the BAR mapping is never removed.
        Ok(unsafe { Mmio::from_raw(NonNull::new(base as *mut u8).unwrap(), len, Box::new(())) })
    }

    fn enable_msi(&self) -> Result<IrqEvent> {
        self.vector(0)
    }

    fn enable_msix(&self, count: u32) -> Result<Vec<IrqEvent>> {
        (0..count).map(|i| self.vector(i)).collect()
    }

    fn disable_irqs(&self) -> Result<()> {
        self.dev.disable_irqs().map_err(os("pcid disable interrupts"))
    }

    fn reset_capability(&self) -> ResetCapability {
        ResetCapability::None
    }

    #[cfg(feature = "std")]
    fn export_region(&self, region: Region) -> Result<(std::os::fd::OwnedFd, u64, u64)> {
        use std::os::fd::FromRawFd;
        let Region::Bar(n) = region else {
            return Err(Error::Unsupported(format!("{}: exporting {region}", self.address)));
        };
        let bar = self.bar(n)?;
        let (_, _, mem) = self.mapped_bar(n)?;
        let fd = sys::handle_fd(mem).map_err(os("fd for a BAR"))?;
        // SAFETY: a fresh descriptor from the kernel.
        Ok((unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) }, bar.phys & 0xfff, bar.size))
    }
}

/// An interrupt as a notification. Notifications don't count: a wait reports 1 however often it fired.
struct Notification {
    n: Handle,
    _irq: Handle,
    /// The Linux fd for poll, made on first use (Linux programs only).
    fd: core::sync::atomic::AtomicI32,
}

impl IrqWait for Notification {
    fn wait(&self, timeout: Option<Duration>) -> Result<u64> {
        let deadline = timeout.map_or(dross_rt::FOREVER, |t| sys::clock().saturating_add(t.as_nanos() as u64));
        match sys::notify_wait(self.n, deadline) {
            Ok(_) => Ok(1),
            Err(dross_abi::Error::TimedOut) => Err(Error::Timeout),
            Err(e) => Err(os("notification wait")(e)),
        }
    }

    fn take(&self) -> Result<u64> {
        match sys::notify_wait(self.n, 0) {
            Ok(_) => Ok(1),
            Err(dross_abi::Error::TimedOut) => Ok(0),
            Err(e) => Err(os("notification wait")(e)),
        }
    }

    fn raw_fd(&self) -> Option<i32> {
        use core::sync::atomic::Ordering;
        if !LINUX.load(Ordering::Relaxed) {
            return None;
        }
        let fd = self.fd.load(Ordering::Relaxed);
        if fd >= 0 {
            return Some(fd);
        }
        let fd = sys::handle_fd(self.n).ok()?;
        self.fd.store(fd, Ordering::Relaxed);
        Some(fd)
    }
}

/// Device addresses handed out (the domain has 4-level tables; 39 bits keeps to what 3-level units have too).
const IOVA_END: u64 = 1 << 39;

/// The function's IOMMU domain; device addresses come from a first-fit allocator, so freed ones are reused.
pub struct DrossDma {
    domain: Handle,
    iova: alloc::sync::Arc<Lock<crate::iova::IovaAllocator>>,
}

impl DmaSpace for DrossDma {
    fn alloc(&self, size: usize, access: DmaAccess) -> Result<DmaBuffer> {
        let size = size.next_multiple_of(iova_align(size) as usize);
        let flags = match access {
            DmaAccess::DeviceReads => dma::READ,
            DmaAccess::DeviceWrites => dma::WRITE,
            DmaAccess::ReadWrite => dma::READ | dma::WRITE,
        };
        let size64 = size as u64;
        let iova = self.iova.with(|a| a.alloc(size64, iova_align(size)))?;
        let mapped = (|| {
            let mem = sys::mem_create(size64, 0)?;
            let ptr = sys::vm_map(mem, 0, size64, 0, vm::READ | vm::WRITE)?;
            sys::dma_map(self.domain, mem, 0, size64, iova, flags)?;
            Ok::<_, dross_abi::Error>((mem, ptr))
        })();
        let (mem, ptr) = match mapped {
            Ok(v) => v,
            Err(e) => {
                self.iova.with(|a| a.release(iova..iova + size64));
                return Err(os("DMA allocation")(e));
            }
        };
        let (domain, len, allocator) = (self.domain, size, self.iova.clone());
        let release = move || {
            let _ = sys::dma_unmap(domain, iova, size64);
            let _ = sys::vm_unmap(ptr, size64);
            let _ = sys::close(mem);
            allocator.with(|a| a.release(iova..iova + size64));
        };
        // SAFETY: the memory object is mapped here and for the device until `release`.
        Ok(unsafe { DmaBuffer::from_raw(NonNull::new(ptr as *mut u8).unwrap(), len, iova, Box::new(release)) })
    }

    unsafe fn map_external(&self, _ptr: NonNull<u8>, _size: usize, _access: DmaAccess) -> Result<DmaBuffer> {
        // dross maps memory objects, not addresses; peer BARs will come as memory objects from pcid.
        Err(Error::Unsupported(String::from("mapping external memory by address on dross")))
    }

    fn iova_ranges(&self) -> Vec<Range<u64>> {
        vec![dross_rt::pci::DmaSpace::FIRST..IOVA_END]
    }

    fn page_sizes(&self) -> u64 {
        1 << 12
    }
}

impl From<dross_rt::Error> for Error {
    fn from(e: dross_rt::Error) -> Error {
        os("dross")(e)
    }
}
