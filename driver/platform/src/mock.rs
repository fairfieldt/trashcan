//! An in-memory platform for tests: a PCI function whose config space and BARs are plain memory, a DMA space
//! backed by heap allocations at fake device addresses, and interrupts a test raises by hand.

use crate::iova::IovaAllocator;
use crate::irq::IrqWait;
use crate::pci::{CONFIG_CAPABILITIES, CONFIG_STATUS};
use crate::{
    DmaAccess, DmaBuffer, DmaSpace, Error, IrqEvent, Mmio, PciAddress, PciDevice, Region, RegionInfo, ResetCapability,
    Result,
};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const PAGE: usize = 4096;

/// Zeroed, page-aligned heap memory, standing in for a BAR or for pinned DMA pages.
pub struct AlignedMemory {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: owned heap memory.
unsafe impl Send for AlignedMemory {}
unsafe impl Sync for AlignedMemory {}

impl AlignedMemory {
    pub fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len.max(1), PAGE).unwrap();
        // SAFETY: non-zero size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).expect("out of memory");
        AlignedMemory { ptr, layout }
    }

    pub fn into_mmio(self) -> Mmio {
        let (ptr, len) = (self.ptr, self.layout.size());
        // SAFETY: the memory lives as long as the Mmio holds it.
        unsafe { Mmio::from_raw(ptr, len, Box::new(self)) }
    }
}

impl Drop for AlignedMemory {
    fn drop(&mut self) {
        // SAFETY: allocated in new() with this layout.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

/// A PCI function with a 4 KiB config space and memory BARs.
pub struct MockDevice {
    address: PciAddress,
    config: Mutex<Vec<u8>>,
    // (BAR index, its memory), shared with every mapping of it
    bars: Vec<(u8, Arc<Mmio>)>,
    irq: Arc<MockIrq>,
    msi_enabled: Mutex<bool>,
    // (offset of the last standard capability, where the next one goes)
    caps: Mutex<(Option<usize>, usize)>,
}

impl MockDevice {
    /// `bars`: (index, size) of each memory BAR.
    pub fn new(address: PciAddress, vendor_id: u16, device_id: u16, bars: &[(u8, usize)]) -> Self {
        let mut config = vec![0u8; 4096];
        config[0..2].copy_from_slice(&vendor_id.to_le_bytes());
        config[2..4].copy_from_slice(&device_id.to_le_bytes());
        MockDevice {
            address,
            config: Mutex::new(config),
            bars: bars.iter().map(|&(i, size)| (i, Arc::new(AlignedMemory::new(size).into_mmio()))).collect(),
            irq: Arc::new(MockIrq::default()),
            msi_enabled: Mutex::new(false),
            caps: Mutex::new((None, 0x40)),
        }
    }

    /// Writes config space directly, as firmware would have set it up.
    pub fn poke_config(&self, offset: usize, data: &[u8]) {
        self.config.lock().unwrap()[offset..offset + data.len()].copy_from_slice(data);
    }

    /// Adds a standard capability with `body` after its ID and next pointer; returns its offset.
    pub fn add_capability(&self, id: u8, body: &[u8]) -> u16 {
        let mut c = self.config.lock().unwrap();
        let mut caps = self.caps.lock().unwrap();
        let (last, off) = *caps;
        match last {
            Some(l) => c[l + 1] = off as u8,
            None => c[CONFIG_CAPABILITIES as usize] = off as u8,
        }
        c[off] = id;
        c[off + 2..off + 2 + body.len()].copy_from_slice(body);
        *caps = (Some(off), (off + 2 + body.len()).next_multiple_of(4));
        let status = CONFIG_STATUS as usize;
        c[status] |= 1 << 4;
        off as u16
    }

    /// Raises the function's MSI, if the driver enabled it.
    pub fn raise_msi(&self) {
        if *self.msi_enabled.lock().unwrap() {
            self.irq.raise();
        }
    }

    fn bar(&self, region: Region) -> Result<&Arc<Mmio>> {
        match region {
            Region::Bar(n) => self.bars.iter().find(|b| b.0 == n).map(|b| &b.1),
            _ => None,
        }
        .ok_or_else(|| Error::NotFound(format!("{region}")))
    }
}

impl PciDevice for MockDevice {
    fn address(&self) -> PciAddress {
        self.address
    }

    fn region_info(&self, region: Region) -> Result<RegionInfo> {
        let (size, mappable) = match region {
            Region::Config => (4096, false),
            Region::Rom => (0, false),
            Region::Bar(_) => self.bar(region).map(|b| (b.len() as u64, true)).unwrap_or((0, false)),
        };
        Ok(RegionInfo { region, size, readable: size > 0, writable: size > 0, mappable })
    }

    fn read(&self, region: Region, offset: u64, buf: &mut [u8]) -> Result<()> {
        let check = |len: usize| {
            (offset as usize)
                .checked_add(buf.len())
                .filter(|&end| end <= len)
                .ok_or_else(|| Error::Invalid(format!("{region} read at {offset:#x}")))
        };
        match region {
            Region::Config => {
                let c = self.config.lock().unwrap();
                let end = check(c.len())?;
                buf.copy_from_slice(&c[offset as usize..end]);
                Ok(())
            }
            Region::Bar(_) => {
                let mmio = self.bar(region)?;
                check(mmio.len())?;
                for (i, b) in buf.iter_mut().enumerate() {
                    // SAFETY: in bounds (checked); plain memory.
                    *b = unsafe { mmio.as_ptr().add(offset as usize + i).read_volatile() };
                }
                Ok(())
            }
            Region::Rom => Err(Error::NotFound("ROM".into())),
        }
    }

    fn write(&self, region: Region, offset: u64, data: &[u8]) -> Result<()> {
        match region {
            Region::Config => {
                let mut c = self.config.lock().unwrap();
                let end = (offset as usize)
                    .checked_add(data.len())
                    .filter(|&end| end <= c.len())
                    .ok_or_else(|| Error::Invalid(format!("config write at {offset:#x}")))?;
                c[offset as usize..end].copy_from_slice(data);
                Ok(())
            }
            _ => Err(Error::Unsupported(format!("{region} writes through the platform"))),
        }
    }

    fn map(&self, region: Region) -> Result<Mmio> {
        // Another view of the same memory, kept alive by the Arc.
        let shared = self.bar(region)?.clone();
        let (ptr, len) = (NonNull::new(shared.as_ptr()).unwrap(), shared.len());
        // SAFETY: the backing memory lives while `shared` does.
        Ok(unsafe { Mmio::from_raw(ptr, len, Box::new(shared)) })
    }

    fn enable_msi(&self) -> Result<IrqEvent> {
        *self.msi_enabled.lock().unwrap() = true;
        Ok(IrqEvent::new(Box::new(self.irq.clone())))
    }

    fn disable_irqs(&self) -> Result<()> {
        *self.msi_enabled.lock().unwrap() = false;
        Ok(())
    }

    fn reset_capability(&self) -> ResetCapability {
        ResetCapability::None
    }
}

#[derive(Default)]
pub struct MockIrq {
    count: Mutex<u64>,
    cv: Condvar,
}

impl MockIrq {
    pub fn raise(&self) {
        *self.count.lock().unwrap() += 1;
        self.cv.notify_all();
    }
}

impl IrqWait for Arc<MockIrq> {
    fn wait(&self, timeout: Option<Duration>) -> Result<u64> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut n = self.count.lock().unwrap();
        while *n == 0 {
            n = match deadline {
                None => self.cv.wait(n).unwrap(),
                Some(d) => {
                    let left = d.checked_duration_since(Instant::now()).ok_or(Error::Timeout)?;
                    self.cv.wait_timeout(n, left).unwrap().0
                }
            };
        }
        Ok(std::mem::take(&mut *n))
    }

    fn take(&self) -> Result<u64> {
        Ok(std::mem::take(&mut *self.count.lock().unwrap()))
    }
}

/// Heap memory at device addresses from an [`IovaAllocator`]; records what is mapped so tests can check
/// lifetimes.
pub struct MockDma {
    inner: Arc<Mutex<MockDmaInner>>,
    ranges: Vec<Range<u64>>,
}

struct MockDmaInner {
    iova: IovaAllocator,
    mapped: Vec<(u64, usize, DmaAccess)>,
}

impl MockDma {
    pub fn new(ranges: &[Range<u64>]) -> Self {
        MockDma {
            inner: Arc::new(Mutex::new(MockDmaInner { iova: IovaAllocator::new(ranges), mapped: Vec::new() })),
            ranges: ranges.to_vec(),
        }
    }

    /// (iova, size, access) of each live mapping.
    pub fn mappings(&self) -> Vec<(u64, usize, DmaAccess)> {
        self.inner.lock().unwrap().mapped.clone()
    }

    fn map(&self, ptr: NonNull<u8>, size: usize, access: DmaAccess, owned: Option<AlignedMemory>) -> Result<DmaBuffer> {
        let mut g = self.inner.lock().unwrap();
        let iova = g.iova.alloc(size as u64, crate::dma::iova_align(size))?;
        g.mapped.push((iova, size, access));
        let inner = self.inner.clone();
        let release = Box::new(move || {
            let mut g = inner.lock().unwrap();
            g.mapped.retain(|m| m.0 != iova);
            g.iova.release(iova..iova + size as u64);
            drop(owned);
        });
        // SAFETY: the memory lives until release runs (owned) or per map_external's contract.
        Ok(unsafe { DmaBuffer::from_raw(ptr, size, iova, release) })
    }
}

impl DmaSpace for MockDma {
    fn alloc(&self, size: usize, access: DmaAccess) -> Result<DmaBuffer> {
        let size =
            size.max(1).next_multiple_of(if size >= crate::dma::LARGE_ALIGN { crate::dma::LARGE_ALIGN } else { PAGE });
        let mem = AlignedMemory::new(size);
        let ptr = mem.ptr;
        self.map(ptr, size, access, Some(mem))
    }

    unsafe fn map_external(&self, ptr: NonNull<u8>, size: usize, access: DmaAccess) -> Result<DmaBuffer> {
        if !(ptr.as_ptr() as usize).is_multiple_of(PAGE) || !size.is_multiple_of(PAGE) || size == 0 {
            return Err(Error::Invalid("external DMA mappings are whole pages".into()));
        }
        self.map(ptr, size, access, None)
    }

    fn iova_ranges(&self) -> Vec<Range<u64>> {
        self.ranges.clone()
    }

    fn page_sizes(&self) -> u64 {
        PAGE as u64
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;
    use crate::pci::{CAP_ID_EXPRESS, CAP_ID_MSI, CAP_ID_PM, capabilities};

    fn gpu() -> MockDevice {
        MockDevice::new("0000:02:00.0".parse().unwrap(), 0x1002, 0x6798, &[(0, 1 << 20), (2, 256 << 10)])
    }

    #[test]
    fn config_space_and_capability_walk() {
        let d = gpu();
        assert_eq!(d.config_u16(0).unwrap(), 0x1002);
        assert_eq!(d.config_u16(2).unwrap(), 0x6798);
        let pm = d.add_capability(CAP_ID_PM, &[0; 6]);
        let msi = d.add_capability(CAP_ID_MSI, &[0; 12]);
        let exp = d.add_capability(CAP_ID_EXPRESS, &[0; 58]);
        // One extended capability (AER, ID 1) at 0x100 and a second (ID 0xb) at 0x150.
        d.poke_config(0x100, &(1u32 | 0x150 << 20).to_le_bytes());
        d.poke_config(0x150, &0xbu32.to_le_bytes());
        let caps = capabilities(&d).unwrap();
        assert_eq!(caps.standard, vec![(CAP_ID_PM, pm), (CAP_ID_MSI, msi), (CAP_ID_EXPRESS, exp)]);
        assert_eq!(caps.extended, vec![(1, 0x100), (0xb, 0x150)]);
        assert_eq!(caps.find(CAP_ID_MSI), Some(msi));
        d.set_config_u16(4, 0x6).unwrap();
        assert_eq!(d.config_u16(4).unwrap(), 6);
        assert!(d.read(Region::Config, 4095, &mut [0; 2]).is_err());
    }

    #[test]
    fn a_mapped_bar_is_shared_with_region_reads() {
        let d = gpu();
        let regs = d.map(Region::Bar(2)).unwrap();
        regs.write32(0x5428, 0x1800);
        let again = d.map(Region::Bar(2)).unwrap();
        assert_eq!(again.read32(0x5428), 0x1800);
        let mut b = [0u8; 4];
        d.read(Region::Bar(2), 0x5428, &mut b).unwrap();
        assert_eq!(u32::from_le_bytes(b), 0x1800);
        assert!(d.map(Region::Bar(4)).is_err());
    }

    #[test]
    fn msi_counts_and_times_out() {
        let d = gpu();
        d.raise_msi(); // not enabled yet: dropped
        let irq = d.enable_msi().unwrap();
        assert!(matches!(irq.wait(Some(Duration::from_millis(5))), Err(Error::Timeout)));
        d.raise_msi();
        d.raise_msi();
        assert_eq!(irq.wait(Some(Duration::from_secs(1))).unwrap(), 2);
        assert_eq!(irq.take().unwrap(), 0);
        d.disable_irqs().unwrap();
        d.raise_msi();
        assert_eq!(irq.take().unwrap(), 0);
    }

    #[test]
    fn dma_buffers_unmap_on_drop() {
        let dma = MockDma::new(&[0x10_0000..0x20_0000]);
        let a = dma.alloc(100, DmaAccess::ReadWrite).unwrap();
        let b = dma.alloc(8192, DmaAccess::DeviceWrites).unwrap();
        assert_eq!(a.len(), 4096);
        assert_eq!(a.iova(), 0x10_0000);
        assert_eq!(b.iova(), 0x10_1000);
        a.write32(4092, 7);
        assert_eq!(a.read32(4092), 7);
        assert_eq!(dma.mappings().len(), 2);
        drop(a);
        assert_eq!(dma.mappings(), vec![(0x10_1000, 8192, DmaAccess::DeviceWrites)]);
        let c = dma.alloc(4096, DmaAccess::DeviceReads).unwrap();
        assert_eq!(c.iova(), 0x10_0000, "freed range is reused");
        let mem = AlignedMemory::new(4096);
        // SAFETY: mem outlives the mapping.
        let ext = unsafe { dma.map_external(mem.ptr, 4096, DmaAccess::DeviceReads) }.unwrap();
        drop(ext);
        drop(mem);
        assert!(matches!(dma.alloc(1 << 20, DmaAccess::ReadWrite), Err(Error::Exhausted(_))));
    }
}
