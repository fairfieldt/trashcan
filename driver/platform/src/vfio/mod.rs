//! The Linux backend: VFIO with the type1 IOMMU (container and group interface).
//!
//! A [`Container`] is one IOMMU domain, and the [`DmaSpace`] of every device in it; each IOMMU group is attached to
//! it ([`Container::attach`]) before its devices are opened ([`Group::open_device`]). Two GPUs in one container
//! share device addresses, which is what peer-to-peer needs.
//!
//! Requirements on the host (see the driver's docs/vfio.md): the IOMMU on (`intel_iommu=on iommu=pt`), every
//! device of the group bound to vfio-pci or to no driver, and, to keep a GPU's state across opens, its reset methods
//! cleared and vfio-pci's `disable_idle_d3` set.

mod sys;

use crate::dma::{LARGE_ALIGN, iova_align};
use crate::iova::{IovaAllocator, intersect};
use crate::irq::IrqWait;
use crate::{
    DmaAccess, DmaBuffer, DmaSpace, Error, IrqEvent, Mmio, PciAddress, PciDevice, Region, RegionInfo, ResetCapability,
    Result,
};
use std::ffi::CString;
use std::fs;
use std::ops::Range;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use sys::*;

const PAGE: usize = 4096;

fn open_path(path: &str, op: &'static str) -> Result<OwnedFd> {
    let c = CString::new(path).unwrap();
    // SAFETY: a valid C string; the fd is owned on success.
    let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(Error::last_os(op));
    }
    // SAFETY: fd is a fresh descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// ioctl with a pointer argument.
fn ioctl_ptr<T>(fd: RawFd, req: libc::c_ulong, arg: &mut T, op: &'static str) -> Result<i32> {
    // SAFETY: arg points to a struct of the layout `req` expects (sys.rs), large enough per its argsz.
    let r = unsafe { libc::ioctl(fd, req, arg as *mut T) };
    if r < 0 { Err(Error::last_os(op)) } else { Ok(r) }
}

fn ioctl_val(fd: RawFd, req: libc::c_ulong, arg: libc::c_ulong, op: &'static str) -> Result<i32> {
    // SAFETY: requests that take an integer argument.
    let r = unsafe { libc::ioctl(fd, req, arg) };
    if r < 0 { Err(Error::last_os(op)) } else { Ok(r) }
}

/// The IOMMU group of a PCI function (sysfs).
pub fn iommu_group(addr: PciAddress) -> Result<u32> {
    let link = fs::read_link(format!("/sys/bus/pci/devices/{addr}/iommu_group"))
        .map_err(|_| Error::NotFound(format!("IOMMU group of {addr} (is the IOMMU on?)")))?;
    link.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| Error::Invalid(format!("IOMMU group link {link:?}")))
}

/// The driver a PCI function is bound to (sysfs), if any.
pub fn driver(addr: PciAddress) -> Option<String> {
    fs::read_link(format!("/sys/bus/pci/devices/{addr}/driver"))
        .ok()
        .and_then(|l| l.file_name().map(|n| n.to_string_lossy().into_owned()))
}

/// The functions in an IOMMU group and their drivers (sysfs).
pub fn group_devices(group: u32) -> Result<Vec<(PciAddress, Option<String>)>> {
    let dir = format!("/sys/kernel/iommu_groups/{group}/devices");
    let mut out = Vec::new();
    for e in fs::read_dir(&dir).map_err(|_| Error::NotFound(dir.clone()))? {
        let name = e.map_err(|_| Error::last_os("readdir"))?.file_name();
        if let Ok(addr) = name.to_string_lossy().parse::<PciAddress>() {
            out.push((addr, driver(addr)));
        }
    }
    out.sort();
    Ok(out)
}

/// Opens one PCI function in a container of its own; device addresses are taken from `dma_window`.
pub fn open(addr: PciAddress, dma_window: Range<u64>) -> Result<(Container, Device)> {
    let container = Container::open(dma_window)?;
    let group = container.attach(iommu_group(addr)?)?;
    let device = group.open_device(addr)?;
    Ok((container, device))
}

/// One IOMMU domain.
#[derive(Clone)]
pub struct Container {
    inner: Arc<ContainerInner>,
}

struct ContainerInner {
    fd: OwnedFd,
    dma_window: Range<u64>,
    state: Mutex<ContainerState>,
}

#[derive(Default)]
struct ContainerState {
    // Set when the first group is attached.
    iommu: Option<IommuInfo>,
    iova: Option<IovaAllocator>,
}

/// What the type1 IOMMU reports.
#[derive(Clone, Debug, Default)]
pub struct IommuInfo {
    pub page_sizes: u64,
    /// Translated device address ranges (half-open), before the DMA window is applied.
    pub iova_ranges: Vec<Range<u64>>,
    /// How many more DMA mappings the container accepts (`dma_entry_limit`), if reported.
    pub dma_mappings_available: Option<u32>,
}

impl Container {
    pub fn open(dma_window: Range<u64>) -> Result<Self> {
        let fd = open_path("/dev/vfio/vfio", "open /dev/vfio/vfio")?;
        let v = ioctl_val(fd.as_raw_fd(), VFIO_GET_API_VERSION, 0, "VFIO_GET_API_VERSION")?;
        if v != VFIO_API_VERSION {
            return Err(Error::Unsupported(format!("VFIO API version {v}")));
        }
        if ioctl_val(fd.as_raw_fd(), VFIO_CHECK_EXTENSION, VFIO_TYPE1v2_IOMMU, "VFIO_CHECK_EXTENSION")? != 1 {
            return Err(Error::Unsupported("VFIO type1v2 IOMMU".into()));
        }
        Ok(Container { inner: Arc::new(ContainerInner { fd, dma_window, state: Mutex::default() }) })
    }

    /// Attaches an IOMMU group; every function in it must be bound to vfio-pci or to no driver.
    pub fn attach(&self, group: u32) -> Result<Arc<Group>> {
        let fd = open_path(&format!("/dev/vfio/{group}"), "open /dev/vfio/GROUP")?;
        let mut status = vfio_group_status { argsz: size_of::<vfio_group_status>() as u32, ..Default::default() };
        ioctl_ptr(fd.as_raw_fd(), VFIO_GROUP_GET_STATUS, &mut status, "VFIO_GROUP_GET_STATUS")?;
        if status.flags & VFIO_GROUP_FLAGS_VIABLE == 0 {
            return Err(Error::Unsupported(format!(
                "IOMMU group {group} is not viable: bind each of its devices to vfio-pci or unbind it"
            )));
        }
        let mut cfd = self.inner.fd.as_raw_fd();
        ioctl_ptr(fd.as_raw_fd(), VFIO_GROUP_SET_CONTAINER, &mut cfd, "VFIO_GROUP_SET_CONTAINER")?;
        let mut state = self.inner.state.lock().unwrap();
        if state.iommu.is_none() {
            ioctl_val(self.inner.fd.as_raw_fd(), VFIO_SET_IOMMU, VFIO_TYPE1v2_IOMMU, "VFIO_SET_IOMMU")?;
            let info = self.query_iommu()?;
            let usable = intersect(&info.iova_ranges, std::slice::from_ref(&self.inner.dma_window));
            if usable.is_empty() {
                return Err(Error::Exhausted(format!("no IOVA in {:#x?}", self.inner.dma_window)));
            }
            state.iova = Some(IovaAllocator::new(&usable));
            state.iommu = Some(info);
        }
        Ok(Arc::new(Group { fd, id: group, container: self.clone() }))
    }

    pub fn iommu_info(&self) -> Option<IommuInfo> {
        self.inner.state.lock().unwrap().iommu.clone()
    }

    fn query_iommu(&self) -> Result<IommuInfo> {
        let fd = self.inner.fd.as_raw_fd();
        let mut head = vfio_iommu_type1_info { argsz: size_of::<vfio_iommu_type1_info>() as u32, ..Default::default() };
        ioctl_ptr(fd, VFIO_IOMMU_GET_INFO, &mut head, "VFIO_IOMMU_GET_INFO")?;
        let mut info = IommuInfo { page_sizes: head.iova_pgsizes, ..Default::default() };
        if head.flags & VFIO_IOMMU_INFO_CAPS == 0 || head.argsz as usize <= size_of::<vfio_iommu_type1_info>() {
            return Ok(info);
        }
        // The kernel said how big the info with its capability chain is; ask again with that much room.
        let mut buf = vec![0u64; (head.argsz as usize).div_ceil(8)];
        buf[0] = head.argsz as u64; // argsz in the low half of the first word (little endian)
        ioctl_ptr(fd, VFIO_IOMMU_GET_INFO, &mut buf[0], "VFIO_IOMMU_GET_INFO")?;
        // SAFETY: buf is head.argsz bytes, initialized.
        let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), head.argsz as usize) };
        let u16_at = |o: usize| u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let mut off = u32_at(16) as usize; // cap_offset
        while off != 0 && off + 8 <= bytes.len() {
            match u16_at(off) {
                VFIO_IOMMU_TYPE1_INFO_CAP_IOVA_RANGE => {
                    let n = u32_at(off + 8) as usize;
                    for i in 0..n {
                        let r = off + 16 + i * 16;
                        if r + 16 <= bytes.len() {
                            // The kernel's ends are inclusive.
                            info.iova_ranges.push(u64_at(r)..u64_at(r + 8).saturating_add(1));
                        }
                    }
                }
                VFIO_IOMMU_TYPE1_INFO_DMA_AVAIL => info.dma_mappings_available = Some(u32_at(off + 8)),
                _ => {}
            }
            off = u32_at(off + 4) as usize;
        }
        Ok(info)
    }

    fn map_raw(&self, ptr: NonNull<u8>, size: usize, access: DmaAccess, owned: Option<Munmap>) -> Result<DmaBuffer> {
        let align = iova_align(size);
        let iova = {
            let mut s = self.inner.state.lock().unwrap();
            s.iova
                .as_mut()
                .ok_or_else(|| Error::Unsupported("no IOMMU group attached yet".into()))?
                .alloc(size as u64, align)?
        };
        let flags = match access {
            DmaAccess::DeviceReads => VFIO_DMA_MAP_FLAG_READ,
            DmaAccess::DeviceWrites => VFIO_DMA_MAP_FLAG_WRITE,
            DmaAccess::ReadWrite => VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE,
        };
        let mut map = vfio_iommu_type1_dma_map {
            argsz: size_of::<vfio_iommu_type1_dma_map>() as u32,
            flags,
            vaddr: ptr.as_ptr() as u64,
            iova,
            size: size as u64,
        };
        let inner = self.inner.clone();
        if let Err(e) = ioctl_ptr(inner.fd.as_raw_fd(), VFIO_IOMMU_MAP_DMA, &mut map, "VFIO_IOMMU_MAP_DMA") {
            inner.state.lock().unwrap().iova.as_mut().unwrap().release(iova..iova + size as u64);
            return Err(e);
        }
        let release = Box::new(move || {
            let mut unmap = vfio_iommu_type1_dma_unmap {
                argsz: size_of::<vfio_iommu_type1_dma_unmap>() as u32,
                flags: 0,
                iova,
                size: size as u64,
            };
            // Only on a bug can this fail; then the pages stay pinned and the range stays allocated.
            if ioctl_ptr(inner.fd.as_raw_fd(), VFIO_IOMMU_UNMAP_DMA, &mut unmap, "VFIO_IOMMU_UNMAP_DMA").is_ok() {
                inner.state.lock().unwrap().iova.as_mut().unwrap().release(iova..iova + size as u64);
                drop(owned);
            } else {
                std::mem::forget(owned);
            }
        });
        // SAFETY: mapped at iova until release runs; owned memory lives until then.
        Ok(unsafe { DmaBuffer::from_raw(ptr, size, iova, release) })
    }
}

impl DmaSpace for Container {
    fn alloc(&self, size: usize, access: DmaAccess) -> Result<DmaBuffer> {
        let size = size.max(1).next_multiple_of(if size >= LARGE_ALIGN { LARGE_ALIGN } else { PAGE });
        // SAFETY: anonymous private mapping; the kernel zeroes it. MAP_POPULATE faults it in before pinning.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(Error::last_os("mmap DMA buffer"));
        }
        let ptr = NonNull::new(p.cast::<u8>()).unwrap();
        self.map_raw(ptr, size, access, Some(Munmap { ptr, len: size }))
    }

    unsafe fn map_external(&self, ptr: NonNull<u8>, size: usize, access: DmaAccess) -> Result<DmaBuffer> {
        if !(ptr.as_ptr() as usize).is_multiple_of(PAGE) || !size.is_multiple_of(PAGE) || size == 0 {
            return Err(Error::Invalid("external DMA mappings are whole pages".into()));
        }
        self.map_raw(ptr, size, access, None)
    }

    fn iova_ranges(&self) -> Vec<Range<u64>> {
        let s = self.inner.state.lock().unwrap();
        s.iommu
            .as_ref()
            .map(|i| intersect(&i.iova_ranges, std::slice::from_ref(&self.inner.dma_window)))
            .unwrap_or_default()
    }

    fn page_sizes(&self) -> u64 {
        self.inner.state.lock().unwrap().iommu.as_ref().map_or(0, |i| i.page_sizes)
    }
}

/// An attached IOMMU group.
pub struct Group {
    fd: OwnedFd,
    id: u32,
    container: Container,
}

impl Group {
    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn container(&self) -> &Container {
        &self.container
    }

    pub fn open_device(self: &Arc<Self>, addr: PciAddress) -> Result<Device> {
        let name = CString::new(addr.to_string()).unwrap();
        // SAFETY: the request takes a C string.
        let dfd = unsafe { libc::ioctl(self.fd.as_raw_fd(), VFIO_GROUP_GET_DEVICE_FD, name.as_ptr()) };
        if dfd < 0 {
            return Err(Error::last_os("VFIO_GROUP_GET_DEVICE_FD"));
        }
        // SAFETY: a fresh descriptor.
        let fd = unsafe { OwnedFd::from_raw_fd(dfd) };
        let mut info = vfio_device_info { argsz: size_of::<vfio_device_info>() as u32, ..Default::default() };
        ioctl_ptr(fd.as_raw_fd(), VFIO_DEVICE_GET_INFO, &mut info, "VFIO_DEVICE_GET_INFO")?;
        if info.flags & VFIO_DEVICE_FLAGS_PCI == 0 {
            return Err(Error::Unsupported(format!("{addr} is not a vfio-pci device")));
        }
        let mut regions = Vec::new();
        for index in 0..info.num_regions {
            let mut r = vfio_region_info { argsz: size_of::<vfio_region_info>() as u32, index, ..Default::default() };
            // Regions a function doesn't have report size 0; an error means the index isn't there at all.
            if ioctl_ptr(fd.as_raw_fd(), VFIO_DEVICE_GET_REGION_INFO, &mut r, "VFIO_DEVICE_GET_REGION_INFO").is_ok() {
                regions.push(r);
            }
        }
        Ok(Device { fd, address: addr, info, regions, _group: self.clone() })
    }
}

/// A vfio-pci function.
pub struct Device {
    fd: OwnedFd,
    address: PciAddress,
    info: vfio_device_info,
    regions: Vec<vfio_region_info>,
    _group: Arc<Group>,
}

impl Device {
    pub fn num_regions(&self) -> u32 {
        self.info.num_regions
    }

    pub fn num_irqs(&self) -> u32 {
        self.info.num_irqs
    }

    /// Vectors behind an interrupt index (0 INTx, 1 MSI, 2 MSI-X, 3 error, 4 request).
    pub fn irq_count(&self, index: u32) -> Result<u32> {
        let mut i = vfio_irq_info { argsz: size_of::<vfio_irq_info>() as u32, index, ..Default::default() };
        ioctl_ptr(self.fd.as_raw_fd(), VFIO_DEVICE_GET_IRQ_INFO, &mut i, "VFIO_DEVICE_GET_IRQ_INFO")?;
        Ok(i.count)
    }

    /// The functions a bus reset of this device would also reset: (IOMMU group, address).
    pub fn hot_reset_dependents(&self) -> Result<Vec<(u32, PciAddress)>> {
        let mut head =
            vfio_pci_hot_reset_info { argsz: size_of::<vfio_pci_hot_reset_info>() as u32, ..Default::default() };
        let fd = self.fd.as_raw_fd();
        match ioctl_ptr(fd, VFIO_DEVICE_GET_PCI_HOT_RESET_INFO, &mut head, "VFIO_DEVICE_GET_PCI_HOT_RESET_INFO") {
            Ok(_) => {}
            Err(Error::Os { errno: libc::ENOSPC, .. }) => {}
            Err(e) => return Err(e),
        }
        let n = head.count as usize;
        let size = size_of::<vfio_pci_hot_reset_info>() + n * size_of::<vfio_pci_dependent_device>();
        let mut buf = vec![0u32; size.div_ceil(4)];
        buf[0] = size as u32;
        ioctl_ptr(fd, VFIO_DEVICE_GET_PCI_HOT_RESET_INFO, &mut buf[0], "VFIO_DEVICE_GET_PCI_HOT_RESET_INFO")?;
        Ok((0..n)
            .map(|i| {
                let d = &buf[3 + i * 2..5 + i * 2];
                let devfn = (d[1] >> 24) as u8;
                (
                    d[0],
                    PciAddress {
                        domain: (d[1] & 0xffff) as u16,
                        bus: ((d[1] >> 16) & 0xff) as u8,
                        device: devfn >> 3,
                        function: devfn & 7,
                    },
                )
            })
            .collect())
    }

    fn region(&self, region: Region) -> Result<&vfio_region_info> {
        let index = match region {
            Region::Bar(n) if n < 6 => VFIO_PCI_BAR0_REGION_INDEX + n as u32,
            Region::Rom => VFIO_PCI_ROM_REGION_INDEX,
            Region::Config => VFIO_PCI_CONFIG_REGION_INDEX,
            Region::Bar(_) => return Err(Error::Invalid(format!("{region}"))),
        };
        self.regions
            .iter()
            .find(|r| r.index == index)
            .ok_or_else(|| Error::NotFound(format!("{region} of {}", self.address)))
    }

    fn check_range(&self, region: Region, offset: u64, len: usize) -> Result<&vfio_region_info> {
        let r = self.region(region)?;
        if offset.checked_add(len as u64).is_none_or(|end| end > r.size) {
            return Err(Error::Invalid(format!("{region} access at {offset:#x}+{len:#x} past {:#x}", r.size)));
        }
        Ok(r)
    }
}

impl PciDevice for Device {
    fn address(&self) -> PciAddress {
        self.address
    }

    fn region_info(&self, region: Region) -> Result<RegionInfo> {
        let r = self.region(region)?;
        Ok(RegionInfo {
            region,
            size: r.size,
            readable: r.flags & VFIO_REGION_INFO_FLAG_READ != 0,
            writable: r.flags & VFIO_REGION_INFO_FLAG_WRITE != 0,
            mappable: r.flags & VFIO_REGION_INFO_FLAG_MMAP != 0,
        })
    }

    fn read(&self, region: Region, offset: u64, buf: &mut [u8]) -> Result<()> {
        let r = self.check_range(region, offset, buf.len())?;
        let mut done = 0;
        while done < buf.len() {
            // SAFETY: writes into buf[done..].
            let n = unsafe {
                libc::pread(
                    self.fd.as_raw_fd(),
                    buf[done..].as_mut_ptr().cast(),
                    buf.len() - done,
                    (r.offset + offset + done as u64) as libc::off_t,
                )
            };
            if n <= 0 {
                return Err(if n == 0 {
                    Error::Invalid(format!("short {region} read"))
                } else {
                    Error::last_os("pread")
                });
            }
            done += n as usize;
        }
        Ok(())
    }

    fn write(&self, region: Region, offset: u64, data: &[u8]) -> Result<()> {
        let r = self.check_range(region, offset, data.len())?;
        let mut done = 0;
        while done < data.len() {
            // SAFETY: reads from data[done..].
            let n = unsafe {
                libc::pwrite(
                    self.fd.as_raw_fd(),
                    data[done..].as_ptr().cast(),
                    data.len() - done,
                    (r.offset + offset + done as u64) as libc::off_t,
                )
            };
            if n <= 0 {
                return Err(if n == 0 {
                    Error::Invalid(format!("short {region} write"))
                } else {
                    Error::last_os("pwrite")
                });
            }
            done += n as usize;
        }
        Ok(())
    }

    fn map(&self, region: Region) -> Result<Mmio> {
        let r = *self.region(region)?;
        if r.flags & VFIO_REGION_INFO_FLAG_MMAP == 0 || r.size == 0 {
            return Err(Error::Unsupported(format!("{region} of {} can't be mapped", self.address)));
        }
        // SAFETY: maps the region's part of the device fd; the kernel checks the range.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                r.size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.fd.as_raw_fd(),
                r.offset as libc::off_t,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(Error::last_os("mmap BAR"));
        }
        let ptr = NonNull::new(p.cast::<u8>()).unwrap();
        // SAFETY: mapped for r.size bytes until the Munmap guard drops.
        Ok(unsafe { Mmio::from_raw(ptr, r.size as usize, Box::new(Munmap { ptr, len: r.size as usize })) })
    }

    fn enable_msi(&self) -> Result<IrqEvent> {
        if self.irq_count(VFIO_PCI_MSI_IRQ_INDEX)? == 0 {
            return Err(Error::Unsupported(format!("{} has no MSI", self.address)));
        }
        // SAFETY: plain eventfd creation.
        let efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if efd < 0 {
            return Err(Error::last_os("eventfd"));
        }
        // SAFETY: a fresh descriptor.
        let efd = unsafe { OwnedFd::from_raw_fd(efd) };
        let mut set = vfio_irq_set_one {
            argsz: size_of::<vfio_irq_set_one>() as u32,
            flags: VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER,
            index: VFIO_PCI_MSI_IRQ_INDEX,
            start: 0,
            count: 1,
            fd: efd.as_raw_fd(),
        };
        ioctl_ptr(self.fd.as_raw_fd(), VFIO_DEVICE_SET_IRQS, &mut set, "VFIO_DEVICE_SET_IRQS")?;
        Ok(IrqEvent::new(Box::new(EventFd(efd))))
    }

    fn enable_msix(&self, count: u32) -> Result<Vec<IrqEvent>> {
        let have = self.irq_count(VFIO_PCI_MSIX_IRQ_INDEX)?;
        if have < count {
            return Err(Error::Unsupported(format!("{} has {have} MSI-X vectors, {count} asked for", self.address)));
        }
        let mut fds = Vec::with_capacity(count as usize);
        for _ in 0..count {
            // SAFETY: plain eventfd creation.
            let efd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
            if efd < 0 {
                return Err(Error::last_os("eventfd"));
            }
            // SAFETY: a fresh descriptor.
            fds.push(unsafe { OwnedFd::from_raw_fd(efd) });
        }
        // struct vfio_irq_set followed by one eventfd per vector.
        let mut buf: Vec<u32> = vec![
            20 + 4 * count,
            VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER,
            VFIO_PCI_MSIX_IRQ_INDEX,
            0,
            count,
        ];
        buf.extend(fds.iter().map(|f| f.as_raw_fd() as u32));
        // The kernel reads the whole buffer from its first dword.
        ioctl_ptr(self.fd.as_raw_fd(), VFIO_DEVICE_SET_IRQS, &mut buf[0], "VFIO_DEVICE_SET_IRQS")?;
        Ok(fds.into_iter().map(|f| IrqEvent::new(Box::new(EventFd(f)))).collect())
    }

    /// Stops MSI and MSI-X delivery; fails only if neither was on.
    fn disable_irqs(&self) -> Result<()> {
        let mut result = Err(Error::Unsupported(format!("{} has no MSI or MSI-X", self.address)));
        for index in [VFIO_PCI_MSI_IRQ_INDEX, VFIO_PCI_MSIX_IRQ_INDEX] {
            if self.irq_count(index).unwrap_or(0) == 0 {
                continue;
            }
            let mut set = vfio_irq_set_one {
                argsz: 20, // no data
                flags: VFIO_IRQ_SET_DATA_NONE | VFIO_IRQ_SET_ACTION_TRIGGER,
                index,
                ..Default::default()
            };
            match ioctl_ptr(self.fd.as_raw_fd(), VFIO_DEVICE_SET_IRQS, &mut set, "VFIO_DEVICE_SET_IRQS") {
                Ok(_) => result = Ok(()),
                Err(e) if result.is_err() => result = Err(e),
                Err(_) => {}
            }
        }
        result
    }

    fn reset_capability(&self) -> ResetCapability {
        if self.info.flags & VFIO_DEVICE_FLAGS_RESET != 0 {
            ResetCapability::Function
        } else if self.hot_reset_dependents().is_ok_and(|d| !d.is_empty()) {
            ResetCapability::Bus
        } else {
            ResetCapability::None
        }
    }
}

struct Munmap {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: owns a mapping; nothing thread-specific.
unsafe impl Send for Munmap {}
unsafe impl Sync for Munmap {}

impl Drop for Munmap {
    fn drop(&mut self) {
        // SAFETY: unmaps exactly what was mapped.
        unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) };
    }
}

struct EventFd(OwnedFd);

impl EventFd {
    fn read_count(&self) -> Result<u64> {
        let mut v = 0u64;
        // SAFETY: reads 8 bytes into v.
        let n = unsafe { libc::read(self.0.as_raw_fd(), (&mut v as *mut u64).cast(), 8) };
        if n == 8 {
            Ok(v)
        } else if n < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EAGAIN) {
            Ok(0)
        } else {
            Err(Error::last_os("read eventfd"))
        }
    }
}

impl IrqWait for EventFd {
    fn wait(&self, timeout: Option<Duration>) -> Result<u64> {
        let ms = timeout.map_or(-1, |t| t.as_millis().min(i32::MAX as u128) as i32);
        let mut p = libc::pollfd { fd: self.0.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        loop {
            // SAFETY: one valid pollfd.
            let r = unsafe { libc::poll(&mut p, 1, ms) };
            if r < 0 {
                if std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                return Err(Error::last_os("poll eventfd"));
            }
            if r == 0 {
                return Err(Error::Timeout);
            }
            let n = self.read_count()?;
            if n > 0 {
                return Ok(n);
            }
        }
    }

    fn take(&self) -> Result<u64> {
        self.read_count()
    }
}
