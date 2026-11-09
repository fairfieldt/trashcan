//! PCI for drivers: the protocol to pcid, and the platform interface a driver programs against. The types
//! follow `gpud-platform` (driver/platform): a `Device` gives config space, BAR mappings (`Mmio`) and MSIs
//! (`IrqEvent`); a `DmaSpace` gives memory at device addresses (`DmaBuffer`) in the device's own IOMMU domain.

use alloc::sync::Arc;
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
    /// a: bits, b: CPU (`u64::MAX`: any), c: MSI-X vector index (ignored for MSI); one notification handle.
    pub const MSI: u32 = 8;
    pub const DISABLE_IRQS: u32 = 9;
    /// Same-owner release after a transferred Process handle proves successful exit.
    pub const RELEASE: u32 = 10;
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

/// A connection to pcid; clones share it (requests are serialized).
#[derive(Clone)]
pub struct Pci {
    ch: Arc<crate::Lock<Option<Handle>>>,
    timeout_ms: Option<u64>,
}

impl Pci {
    pub fn new(ch: Handle) -> Pci {
        Pci { ch: Arc::new(crate::Lock::new(Some(ch))), timeout_ms: None }
    }

    /// Bound each reply wait after serialization, not the whole caller operation.
    /// A timeout closes the shared endpoint: pcid replies have no
    /// transaction ID, so the connection cannot safely be reused after a late reply.
    /// Existing `new` callers retain their original unbounded startup behavior.
    pub fn with_timeout(ch: Handle, timeout_ms: u64) -> Pci {
        Pci { ch: Arc::new(crate::Lock::new(Some(ch))), timeout_ms: Some(timeout_ms) }
    }

    fn call(&self, r: Request, send: &[Handle]) -> sys::Result<(Vec<u8>, Vec<Handle>)> {
        let mut buf = alloc::vec![0u8; 64 * 1024];
        let mut hs = [Handle(0); 4];
        let (n, nh) = self.ch.with(|slot| {
            let ch = (*slot).ok_or(Error::PeerClosed)?;
            sys::channel_write(ch, &r.encode(), send)?;
            let deadline = self.timeout_ms.map_or(crate::FOREVER, sys::after_ms);
            let result = sys::channel_read(ch, &mut buf, &mut hs, deadline);
            if result == Err(Error::TimedOut) {
                *slot = None;
                let _ = sys::close(ch);
            }
            result
        })?;
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

    /// Release a quiesced function after the driver exited successfully. The
    /// server verifies the process handle before dropping its retained domain.
    pub fn release(&self, bdf: u16, exited: Handle) -> sys::Result<()> {
        self.ch.with(|slot| {
            let ch = (*slot).ok_or(Error::PeerClosed)?;
            let process = sys::dup(exited, dross_abi::rights::WAIT | dross_abi::rights::TRANSFER)?;
            let request = Request { op: op::RELEASE, bdf: bdf as u32, ..Default::default() }.encode();
            if let Err(error) = sys::channel_write(ch, &request, &[process]) {
                let _ = sys::close(process);
                return Err(error);
            }
            let mut bytes = [0u8; 16]; let mut handles = [Handle(0); 4];
            let result = sys::channel_read(ch, &mut bytes, &mut handles, sys::after_ms(self.timeout_ms.unwrap_or(2000)));
            if result == Err(Error::TimedOut) {
                *slot = None;
                let _ = sys::close(ch);
            }
            let (n, nh) = result?;
            for &handle in &handles[..nh] { let _ = sys::close(handle); }
            if n != 16 || nh != 0 { return Err(Error::InvalidArgs); }
            let status = i64::from_le_bytes(bytes[..8].try_into().unwrap());
            if status == 0 { Ok(()) } else { Err(Error::from_status(status).unwrap_or(Error::BadSyscall)) }
        })
    }

    /// Takes ownership of a function: nobody else can open it until this connection closes.
    pub fn open(&self, bdf: u16) -> sys::Result<Device> {
        let (r, _) = self.call(Request { op: op::OPEN, bdf: bdf as u32, ..Default::default() }, &[])?;
        let info = Info::decode(&r[16..]).ok_or(Error::InvalidArgs)?;
        Ok(Device { pci: self.clone(), info })
    }
}

/// One PCI function owned by this driver.
pub struct Device {
    pci: Pci,
    pub info: Info,
}

impl Device {
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
        Ok(DmaSpace { domain: *hs.first().ok_or(Error::InvalidArgs)?, next: DmaSpace::FIRST, #[cfg(target_arch="riscv64")] physical: None, group: 0, grouped: Vec::new(), live: (0, 0) })
    }

    /// Programs the device's MSI (or MSI-X vector 0) to signal `bits` on `notification` (which stays the
    /// driver's).
    pub fn enable_msi(&self, notification: Handle, bits: u64) -> sys::Result<IrqEvent> {
        self.enable_vector(notification, bits, 0)
    }

    /// MSI-X vector `index` (MSI if the function has no MSI-X: then only index 0).
    pub fn enable_vector(&self, notification: Handle, bits: u64, index: u32) -> sys::Result<IrqEvent> {
        let n = sys::dup(notification, dross_abi::rights::ALL)?;
        let (_, hs) = self.pci.call(Request { op: op::MSI, bdf: self.info.bdf as u32, a: bits, b: u64::MAX, c: index as u64 }, &[n])?;
        Ok(IrqEvent { notification, bits, irq: *hs.first().ok_or(Error::InvalidArgs)?, #[cfg(target_arch="riscv64")] wired: false })
    }

    /// Turns MSI and MSI-X off; the vectors are freed.
    pub fn disable_irqs(&self) -> sys::Result<()> {
        self.pci.call(Request { op: op::DISABLE_IRQS, bdf: self.info.bdf as u32, ..Default::default() }, &[]).map(|_| ())
    }
}

/// A mapped memory BAR.
pub struct Mmio {
    pub mem: Handle,
    base: u64,
    len: usize,
}

impl Mmio {
    /// # Safety
    /// The caller owns a live writable device mapping for the entire range and
    /// retains its memory handle until this view is no longer used.
    pub unsafe fn from_mapping(mem: Handle, base: u64, len: usize) -> Self {
        Self { mem, base, len }
    }

    fn at(&self, offset: usize, size: usize) -> u64 {
        assert!(offset.is_multiple_of(size) && offset.checked_add(size).is_some_and(|end| end <= self.len),
            "mmio access at {offset:#x} past {:#x} or misaligned", self.len);
        self.base + offset as u64
    }
    pub fn write8(&self, offset: usize, value: u8) {
        crate::io::device_fence();
        unsafe { (self.at(offset, 1) as *mut u8).write_volatile(value) };
        crate::io::device_fence();
    }
    pub fn read16(&self, offset: usize) -> u16 {
        crate::io::device_fence();
        let value = unsafe { (self.at(offset, 2) as *const u16).read_volatile() };
        crate::io::device_fence(); value
    }
    pub fn write16(&self, offset: usize, value: u16) {
        crate::io::device_fence();
        unsafe { (self.at(offset, 2) as *mut u16).write_volatile(value) };
        crate::io::device_fence();
    }
    pub fn read32(&self, offset: usize) -> u32 {
        crate::io::device_fence();
        // SAFETY: inside the mapped BAR.
        let value = unsafe { (self.at(offset, 4) as *const u32).read_volatile() };
        crate::io::device_fence(); value
    }
    pub fn write32(&self, offset: usize, v: u32) {
        crate::io::device_fence();
        // SAFETY: as above.
        unsafe { (self.at(offset, 4) as *mut u32).write_volatile(v) }
        crate::io::device_fence();
    }
    pub fn read64(&self, offset: usize) -> u64 {
        crate::io::device_fence();
        // SAFETY: as above.
        let value = unsafe { (self.at(offset, 8) as *const u64).read_volatile() };
        crate::io::device_fence(); value
    }
    pub fn write64(&self, offset: usize, v: u64) {
        crate::io::device_fence();
        // SAFETY: as above.
        unsafe { (self.at(offset, 8) as *mut u64).write_volatile(v) }
        crate::io::device_fence();
    }
    pub fn read8(&self, offset: usize) -> u8 {
        crate::io::device_fence();
        // SAFETY: as above.
        let value = unsafe { (self.at(offset, 1) as *const u8).read_volatile() };
        crate::io::device_fence(); value
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn as_ptr(&self) -> *mut u8 {
        self.base as *mut u8
    }
}

/// A device's view of memory.
pub struct DmaSpace {
    domain: Handle,
    next: u64,
    #[cfg(target_arch="riscv64")]
    physical: Option<(Handle, Vec<(Handle, u64, u64)>)>,
    /// New buffers join this group; zero is the space's lifetime and is never freed.
    group: u32,
    grouped: Vec<(u32, Handle, u64, u64, u64)>,
    live: (usize, u64),
}

impl DmaSpace {
    /// Explicit coherent physical DMA, with no device address isolation.
    /// # Safety
    /// Caller controls the bus master exclusively and retains this space until
    /// halt/BME-off is established. No IOMMU operation falls back to this mode.
    #[cfg(target_arch="riscv64")]
    pub unsafe fn trusted_physical(authority: Handle) -> Self {
        Self { domain: Handle(0), next: Self::FIRST, physical: Some((authority,Vec::new())), group: 0, grouped: Vec::new(), live: (0, 0) }
    }
    /// # Safety
    /// The controller is halted with bus mastering disabled, and all returned
    /// DmaBuffer views must never be used again. Kernel pins survive a crash or
    /// lost owner without this call; reclaim only after proven quiescence.
    #[cfg(target_arch="riscv64")]
    pub unsafe fn reclaim_physical_quiescent(&mut self) -> sys::Result<()> {
        let (authority, retained) = self.physical.as_mut().ok_or(Error::WrongType)?;
        for &(memory,ptr,len) in retained.iter() {
            sys::dma_buffer_reclaim(*authority,memory)?;
            sys::vm_unmap(ptr,len)?;sys::close(memory)?;
        }
        retained.clear();self.grouped.clear();self.live=(0,0);Ok(())
    }

    /// Buffers from now on belong to `group`, for one `free_group`. Returns the previous group.
    pub fn set_group(&mut self, group: u32) -> u32 {
        core::mem::replace(&mut self.group, group)
    }

    /// Unmaps and returns every buffer of a nonzero group.
    /// # Safety
    /// The device can no longer reach these buffers, and their DmaBuffer views must never be used again.
    pub unsafe fn free_group(&mut self, group: u32) -> sys::Result<()> {
        assert_ne!(group, 0);
        let mut at = 0;
        while at < self.grouped.len() {
            let (g, mem, ptr, iova, size) = self.grouped[at];
            if g != group { at += 1; continue; }
            self.grouped.swap_remove(at);
            #[cfg(target_arch="riscv64")]
            if let Some((authority,retained)) = self.physical.as_mut() {
                sys::dma_buffer_reclaim(*authority,mem)?;
                retained.retain(|r|r.0 != mem);
            } else {
                sys::dma_unmap(self.domain, iova, size)?;
            }
            #[cfg(not(target_arch="riscv64"))]
            sys::dma_unmap(self.domain, iova, size)?;
            sys::vm_unmap(ptr, size)?;
            sys::close(mem)?;
            self.live = (self.live.0 - 1, self.live.1 - size);
        }
        Ok(())
    }

    /// Stops tracking all groups; their memory stays allocated for the life of the space.
    pub fn forget_groups(&mut self) {
        self.grouped.clear();
    }

    /// Buffers and bytes this space holds.
    pub fn usage(&self) -> (usize, u64) {
        self.live
    }

    /// The domain, for callers that place mappings themselves.
    pub fn domain(&self) -> Handle {
        #[cfg(target_arch="riscv64")]
        assert!(self.physical.is_none(), "physical DMA has no IOMMU domain");
        self.domain
    }

    /// IOVAs start above 1 MiB (a zero or tiny device address is more often a bug than a buffer).
    pub const FIRST: u64 = 1 << 20;

    /// Pinned, zeroed memory mapped for the CPU and at a fresh device address, readable and writable by the
    /// device.
    pub fn alloc(&mut self, size: usize) -> sys::Result<DmaBuffer> {
        self.alloc_with(size, dma::READ | dma::WRITE)
    }

    /// As `alloc`, with the device's access (`dross_abi::dma` flags).
    pub fn alloc_with(&mut self, size: usize, access: u64) -> sys::Result<DmaBuffer> {
        #[cfg(target_arch="riscv64")]
        if let Some((authority,retained)) = self.physical.as_mut() {
            if access != dma::READ | dma::WRITE { return Err(Error::InvalidArgs); }
            let size = (size as u64).checked_add(4095).ok_or(Error::InvalidArgs)? & !4095;
            retained.try_reserve(1).map_err(|_|Error::NoMemory)?;
            let (mem,iova) = sys::dma_buffer_create(*authority,size)?;
            let ptr = match sys::vm_map(mem,0,size,0,vm::READ|vm::WRITE) {
                Ok(ptr)=>ptr,Err(e)=>{let _=sys::dma_buffer_reclaim(*authority,mem);let _=sys::close(mem);return Err(e);}
            };
            retained.push((mem,ptr,size));
            self.track(mem,ptr,iova,size);
            return Ok(DmaBuffer{mem,domain:Handle(0),ptr,iova,len:size as usize});
        }
        let size = (size as u64).next_multiple_of(4096);
        let mem = sys::mem_create(size, 0)?;
        let ptr = sys::vm_map(mem, 0, size, 0, vm::READ | vm::WRITE)?;
        let align = if size >= 2 << 20 { 2 << 20 } else { 4096 };
        let iova = self.next.next_multiple_of(align);
        sys::dma_map(self.domain, mem, 0, size, iova, access)?;
        self.next = iova + size + 4096;
        self.track(mem, ptr, iova, size);
        Ok(DmaBuffer { mem, domain: self.domain, ptr, iova, len: size as usize })
    }

    fn track(&mut self, mem: Handle, ptr: u64, iova: u64, size: u64) {
        self.live = (self.live.0 + 1, self.live.1 + size);
        if self.group != 0 { self.grouped.push((self.group, mem, ptr, iova, size)); }
    }
}

pub struct DmaBuffer {
    pub mem: Handle,
    /// Zero in explicit physical mode; this is then not an IOMMU handle.
    pub domain: Handle,
    pub ptr: u64,
    pub iova: u64,
    pub len: usize,
}

impl DmaBuffer {
    /// Another view of the same memory, which owns nothing.
    pub fn view(&self) -> DmaBuffer {
        DmaBuffer { mem: self.mem, domain: self.domain, ptr: self.ptr, iova: self.iova, len: self.len }
    }
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
    #[cfg(target_arch="riscv64")]
    wired: bool,
}

impl IrqEvent {
    /// A vector the platform's MSI receiver demultiplexes onto `notification`.
    /// The receiver owns completion, so waiting only drains this notification.
    #[cfg(target_arch="riscv64")]
    pub fn demultiplexed(notification:Handle,bits:u64)->sys::Result<Self> {
        Ok(Self {irq:sys::dup(notification,dross_abi::rights::WAIT)?,notification,bits,wired:false})
    }

    #[cfg(target_arch="riscv64")]
    pub fn wired(resource:Handle,source:u32,notification:Handle,bits:u64,cpu:u32)->sys::Result<Self> {
        Ok(Self {irq:sys::irq_bind(resource,source,notification,bits,cpu)?,notification,bits,wired:true})
    }

    /// Waits for this interrupt (other bits that arrive meanwhile are returned too).
    pub fn wait(&self, deadline: u64) -> sys::Result<u64> {
        // The driver drains events and clears device pending state before
        // waiting. Complete the masked PLIC claim only after that deassertion.
        #[cfg(target_arch="riscv64")]
        if self.wired {
            sys::irq_ack(self.irq)?;
        }
        sys::notify_wait(self.notification, deadline)
    }
}
