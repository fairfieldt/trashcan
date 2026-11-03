//! PCI for drivers: the protocol to pcid, and the platform interface a driver programs against. The types
//! follow `gpud-platform` (driver/platform): a `Device` gives config space, BAR mappings (`Mmio`) and MSIs
//! (`IrqEvent`); a `DmaSpace` gives memory at device addresses (`DmaBuffer`) in the device's own IOMMU domain.

use alloc::vec::Vec;
use dross_abi::{dma, vm, Error};

use crate::sys::{self, Handle};

/// Requests: `op, bdf, a, b, c` as little-endian u32, u32, u64, u64, u64.
pub mod op {
    pub const LIST: u32 = 1;
    pub const FIND: u32 = 2;
    pub const OPEN: u32 = 3;
    pub const CONFIG_READ: u32 = 4;
    pub const CONFIG_WRITE: u32 = 5;
    pub const MAP_BAR: u32 = 6;
    pub const DOMAIN: u32 = 7;
    pub const MSI: u32 = 8;
}

pub const REQUEST_LEN: usize = 32;

#[derive(Clone, Copy, Debug, Default)]
pub struct Request {
    pub op: u32,
    pub bdf: u32,
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

impl Request {
    pub fn encode(&self) -> [u8; REQUEST_LEN] {
        let mut o = [0u8; REQUEST_LEN];
        o[0..4].copy_from_slice(&self.op.to_le_bytes());
        o[4..8].copy_from_slice(&self.bdf.to_le_bytes());
        o[8..16].copy_from_slice(&self.a.to_le_bytes());
        o[16..24].copy_from_slice(&self.b.to_le_bytes());
        o[24..32].copy_from_slice(&self.c.to_le_bytes());
        o
    }

    pub fn decode(b: &[u8]) -> Option<Request> {
        if b.len() != REQUEST_LEN {
            return None;
        }
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        Some(Request {
            op: u32::from_le_bytes(b[0..4].try_into().unwrap()),
            bdf: u32::from_le_bytes(b[4..8].try_into().unwrap()),
            a: u64_at(8),
            b: u64_at(16),
            c: u64_at(24),
        })
    }
}

/// Replies: status (i64, 0 or an `Error`), value (u64), then the op's payload.
pub fn reply(status: Result<u64, Error>, payload: &[u8]) -> Vec<u8> {
    let (s, v) = match status {
        Ok(v) => (0i64, v),
        Err(e) => (e as i64, 0),
    };
    let mut o = Vec::with_capacity(16 + payload.len());
    o.extend_from_slice(&s.to_le_bytes());
    o.extend_from_slice(&v.to_le_bytes());
    o.extend_from_slice(payload);
    o
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Bar {
    pub phys: u64,
    pub size: u64,
    /// Bit 0: I/O space; bit 1: 64-bit; bit 2: prefetchable.
    pub flags: u32,
}

impl Bar {
    pub const IO: u32 = 1;
    pub const MEM64: u32 = 2;
    pub const PREFETCH: u32 = 4;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Info {
    pub bdf: u16,
    pub vendor: u16,
    pub device: u16,
    /// Class, subclass, programming interface.
    pub class: u32,
    pub bars: [Bar; 6],
}

pub const INFO_LEN: usize = 12 + 6 * 20;

impl Info {
    pub fn encode(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(INFO_LEN);
        o.extend_from_slice(&self.bdf.to_le_bytes());
        o.extend_from_slice(&self.vendor.to_le_bytes());
        o.extend_from_slice(&self.device.to_le_bytes());
        o.extend_from_slice(&[0, 0]);
        o.extend_from_slice(&self.class.to_le_bytes());
        for b in &self.bars {
            o.extend_from_slice(&b.phys.to_le_bytes());
            o.extend_from_slice(&b.size.to_le_bytes());
            o.extend_from_slice(&b.flags.to_le_bytes());
        }
        o
    }

    pub fn decode(b: &[u8]) -> Option<Info> {
        if b.len() < INFO_LEN {
            return None;
        }
        let u16_at = |o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let mut i = Info { bdf: u16_at(0), vendor: u16_at(2), device: u16_at(4), class: u32_at(8), bars: Default::default() };
        for (n, bar) in i.bars.iter_mut().enumerate() {
            let o = 12 + n * 20;
            *bar = Bar { phys: u64_at(o), size: u64_at(o + 8), flags: u32_at(o + 16) };
        }
        Some(i)
    }
}

pub fn bdf_string(bdf: u16) -> alloc::string::String {
    alloc::format!("{:02x}:{:02x}.{}", bdf >> 8, bdf >> 3 & 0x1f, bdf & 7)
}

/// A connection to pcid.
pub struct Pci {
    ch: Handle,
}

impl Pci {
    pub fn new(ch: Handle) -> Pci {
        Pci { ch }
    }

    fn call(&self, r: Request, send: &[Handle]) -> sys::Result<(Vec<u8>, Vec<Handle>)> {
        sys::channel_write(self.ch, &r.encode(), send)?;
        let mut buf = alloc::vec![0u8; 64 * 1024];
        let mut hs = [Handle(0); 4];
        let (n, nh) = sys::channel_read(self.ch, &mut buf, &mut hs, crate::FOREVER)?;
        if n < 16 {
            return Err(Error::InvalidArgs);
        }
        let status = i64::from_le_bytes(buf[0..8].try_into().unwrap());
        if status != 0 {
            return Err(Error::from_status(status).unwrap_or(Error::BadSyscall));
        }
        buf.truncate(n);
        Ok((buf, hs[..nh].to_vec()))
    }

    fn value(reply: &[u8]) -> u64 {
        u64::from_le_bytes(reply[8..16].try_into().unwrap())
    }

    pub fn list(&self) -> sys::Result<Vec<Info>> {
        let (r, _) = self.call(Request { op: op::LIST, ..Default::default() }, &[])?;
        Ok(r[16..].chunks_exact(INFO_LEN).filter_map(Info::decode).collect())
    }

    /// The `index`th function with this vendor and device ID.
    pub fn find(&self, vendor: u16, device: u16, index: u64) -> sys::Result<u16> {
        let (r, _) = self.call(Request { op: op::FIND, a: (vendor as u64) << 16 | device as u64, b: index, ..Default::default() }, &[])?;
        Ok(Self::value(&r) as u16)
    }

    /// The first function of this class (class << 16 | subclass << 8 | interface).
    pub fn find_class(&self, class: u32) -> sys::Result<u16> {
        let (r, _) = self.call(Request { op: op::FIND, a: 1 << 63 | class as u64, ..Default::default() }, &[])?;
        Ok(Self::value(&r) as u16)
    }

    /// Takes ownership of a function: nobody else can open it until this connection closes.
    pub fn open(&self, bdf: u16) -> sys::Result<Device<'_>> {
        let (r, _) = self.call(Request { op: op::OPEN, bdf: bdf as u32, ..Default::default() }, &[])?;
        let info = Info::decode(&r[16..]).ok_or(Error::InvalidArgs)?;
        Ok(Device { pci: self, info })
    }
}

/// One PCI function owned by this driver.
pub struct Device<'a> {
    pci: &'a Pci,
    pub info: Info,
}

impl Device<'_> {
    pub fn bdf(&self) -> u16 {
        self.info.bdf
    }

    pub fn config_read(&self, offset: u16, size: u8) -> sys::Result<u32> {
        let (r, _) = self.pci.call(Request { op: op::CONFIG_READ, bdf: self.info.bdf as u32, a: offset as u64, b: size as u64, c: 0 }, &[])?;
        Ok(Pci::value(&r) as u32)
    }

    /// Writes are filtered by pcid: IDs, BARs, the ROM and the MSI/MSI-X capabilities are its own.
    pub fn config_write(&self, offset: u16, size: u8, value: u32) -> sys::Result<()> {
        self.pci.call(Request { op: op::CONFIG_WRITE, bdf: self.info.bdf as u32, a: offset as u64, b: size as u64, c: value as u64 }, &[]).map(|_| ())
    }

    /// Turns on memory decoding and bus mastering.
    pub fn enable(&self) -> sys::Result<()> {
        let cmd = self.config_read(4, 2)?;
        self.config_write(4, 2, cmd | 0x6)
    }

    pub fn map_bar(&self, n: usize) -> sys::Result<Mmio> {
        let bar = self.info.bars.get(n).copied().ok_or(Error::InvalidArgs)?;
        let (_, hs) = self.pci.call(Request { op: op::MAP_BAR, bdf: self.info.bdf as u32, a: n as u64, ..Default::default() }, &[])?;
        let mem = *hs.first().ok_or(Error::InvalidArgs)?;
        let size = sys::mem_size(mem)?;
        let base = sys::vm_map(mem, 0, size, 0, vm::READ | vm::WRITE)?;
        Ok(Mmio { mem, base: base + (bar.phys & 0xfff), len: bar.size as usize })
    }

    /// The device's DMA domain (devices can only reach what is mapped there).
    pub fn dma_space(&self) -> sys::Result<DmaSpace> {
        let (_, hs) = self.pci.call(Request { op: op::DOMAIN, bdf: self.info.bdf as u32, ..Default::default() }, &[])?;
        Ok(DmaSpace { domain: *hs.first().ok_or(Error::InvalidArgs)?, next: DmaSpace::FIRST })
    }

    /// Programs the device's MSI to signal `bits` on `notification` (which stays the driver's).
    pub fn enable_msi(&self, notification: Handle, bits: u64) -> sys::Result<IrqEvent> {
        let n = sys::dup(notification, dross_abi::rights::ALL)?;
        let (_, hs) = self.pci.call(Request { op: op::MSI, bdf: self.info.bdf as u32, a: bits, b: u64::MAX, c: 0 }, &[n])?;
        Ok(IrqEvent { notification, bits, irq: *hs.first().ok_or(Error::InvalidArgs)? })
    }
}

/// A mapped memory BAR.
pub struct Mmio {
    pub mem: Handle,
    base: u64,
    len: usize,
}

impl Mmio {
    fn at(&self, offset: usize, size: usize) -> u64 {
        assert!(offset + size <= self.len, "mmio access at {offset:#x} past {:#x}", self.len);
        self.base + offset as u64
    }
    pub fn read32(&self, offset: usize) -> u32 {
        // SAFETY: inside the mapped BAR.
        unsafe { (self.at(offset, 4) as *const u32).read_volatile() }
    }
    pub fn write32(&self, offset: usize, v: u32) {
        // SAFETY: as above.
        unsafe { (self.at(offset, 4) as *mut u32).write_volatile(v) }
    }
    pub fn read64(&self, offset: usize) -> u64 {
        // SAFETY: as above.
        unsafe { (self.at(offset, 8) as *const u64).read_volatile() }
    }
    pub fn write64(&self, offset: usize, v: u64) {
        // SAFETY: as above.
        unsafe { (self.at(offset, 8) as *mut u64).write_volatile(v) }
    }
    pub fn read8(&self, offset: usize) -> u8 {
        // SAFETY: as above.
        unsafe { (self.at(offset, 1) as *const u8).read_volatile() }
    }
    pub fn len(&self) -> usize {
        self.len
    }
}

/// A device's view of memory.
pub struct DmaSpace {
    domain: Handle,
    next: u64,
}

impl DmaSpace {
    /// IOVAs start above 1 MiB (a zero or tiny device address is more often a bug than a buffer).
    const FIRST: u64 = 1 << 20;

    /// Pinned, zeroed memory mapped for the CPU and at a fresh device address.
    pub fn alloc(&mut self, size: usize) -> sys::Result<DmaBuffer> {
        let size = (size as u64).next_multiple_of(4096);
        let mem = sys::mem_create(size, 0)?;
        let ptr = sys::vm_map(mem, 0, size, 0, vm::READ | vm::WRITE)?;
        let align = if size >= 2 << 20 { 2 << 20 } else { 4096 };
        let iova = self.next.next_multiple_of(align);
        sys::dma_map(self.domain, mem, 0, size, iova, dma::READ | dma::WRITE)?;
        self.next = iova + size + 4096;
        Ok(DmaBuffer { mem, ptr, iova, len: size as usize })
    }
}

pub struct DmaBuffer {
    pub mem: Handle,
    pub ptr: u64,
    pub iova: u64,
    pub len: usize,
}

impl DmaBuffer {
    pub fn as_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }
    pub fn read32(&self, offset: usize) -> u32 {
        assert!(offset + 4 <= self.len);
        // SAFETY: inside the buffer.
        unsafe { ((self.ptr + offset as u64) as *const u32).read_volatile() }
    }
    pub fn write32(&self, offset: usize, v: u32) {
        assert!(offset + 4 <= self.len);
        // SAFETY: inside the buffer.
        unsafe { ((self.ptr + offset as u64) as *mut u32).write_volatile(v) }
    }
    pub fn slice(&self) -> &mut [u8] {
        // SAFETY: the buffer is this mapping; the device may write it concurrently, as with any DMA buffer.
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }
}

/// An interrupt arriving as bits on a notification.
pub struct IrqEvent {
    pub notification: Handle,
    pub bits: u64,
    pub irq: Handle,
}

impl IrqEvent {
    /// Waits for this interrupt (other bits that arrive meanwhile are returned too).
    pub fn wait(&self, deadline: u64) -> sys::Result<u64> {
        sys::notify_wait(self.notification, deadline)
    }
}
