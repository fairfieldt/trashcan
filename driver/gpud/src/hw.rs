//! The Southern Islands device: Linux's SI code (C, driver/linux) on gpud's C core (driver/shim), given the platform
//! through gpud_core.h's callbacks. The C side keeps raw pointers to everything in [`Device`], so it owns them for
//! the device's lifetime.

use anyhow::{Result, bail};
use gpud_platform::{DmaAccess, DmaBuffer, DmaSpace, Error, Mmio, PciDevice, Region};
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[repr(C)]
pub struct AmdgpuDevice {
    _opaque: [u8; 0],
}

#[repr(C)]
struct GpudPlatform {
    ctx: *mut c_void,
    log: extern "C" fn(*mut c_void, c_int, *const c_char),
    dma_alloc: extern "C" fn(*mut c_void, usize, *mut *mut c_void, *mut u64) -> c_int,
    dma_free: extern "C" fn(*mut c_void, *mut c_void, u64, usize),
    config_read: extern "C" fn(*mut c_void, c_int, c_int, *mut u32) -> c_int,
    config_write: extern "C" fn(*mut c_void, c_int, c_int, u32) -> c_int,
}

#[repr(C)]
struct GpudDeviceInfo {
    mmio: *mut c_void,
    mmio_size: usize,
    vram: *mut c_void,
    vram_size: usize,
    vram_bar_addr: u64,
    vendor: u16,
    device: u16,
    subsystem_vendor: u16,
    subsystem_device: u16,
    revision: u8,
    bus: u8,
    devfn: u8,
    firmware_dir: *const c_char,
}

unsafe extern "C" {
    fn gpud_device_create(platform: *const GpudPlatform, info: *const GpudDeviceInfo) -> *mut AmdgpuDevice;
    fn gpud_device_init(adev: *mut AmdgpuDevice) -> c_int;
    fn gpud_device_ib_tests(adev: *mut AmdgpuDevice, timeout_ms: c_long) -> c_int;
    fn gpud_irq_process(adev: *mut AmdgpuDevice) -> c_int;
    fn gpud_device_fini(adev: *mut AmdgpuDevice);
    fn gpud_ring_count(adev: *mut AmdgpuDevice) -> c_int;
    fn gpud_ring_name(adev: *mut AmdgpuDevice, ring: c_int) -> *const c_char;
    fn gpud_ring_ready(adev: *mut AmdgpuDevice, ring: c_int) -> c_int;
}

/// What the C side calls back into.
struct Ctx {
    pci: Arc<dyn PciDevice>,
    dma: Arc<dyn DmaSpace>,
    // DMA buffers handed to C, by CPU address.
    buffers: Mutex<HashMap<usize, DmaBuffer>>,
}

extern "C" fn cb_log(_ctx: *mut c_void, level: c_int, msg: *const c_char) {
    // SAFETY: the C side passes a NUL-terminated string.
    let msg = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let tag = match level {
        ..=3 => "error",
        4 => "warning",
        5 | 6 => "info",
        _ => "debug",
    };
    eprintln!("[{tag}] {msg}");
}

extern "C" fn cb_dma_alloc(ctx: *mut c_void, size: usize, cpu: *mut *mut c_void, iova: *mut u64) -> c_int {
    // SAFETY: ctx is the Ctx the Device owns.
    let ctx = unsafe { &*(ctx as *const Ctx) };
    match ctx.dma.alloc(size, DmaAccess::ReadWrite) {
        Ok(buf) => {
            // SAFETY: valid out pointers from the C side.
            unsafe {
                *cpu = buf.as_ptr().cast();
                *iova = buf.iova();
            }
            ctx.buffers.lock().unwrap().insert(buf.as_ptr() as usize, buf);
            0
        }
        Err(e) => {
            eprintln!("[error] DMA allocation of {size} bytes: {e}");
            -12
        }
    }
}

extern "C" fn cb_dma_free(ctx: *mut c_void, cpu: *mut c_void, _iova: u64, _size: usize) {
    // SAFETY: as in cb_dma_alloc.
    let ctx = unsafe { &*(ctx as *const Ctx) };
    ctx.buffers.lock().unwrap().remove(&(cpu as usize));
}

extern "C" fn cb_config_read(ctx: *mut c_void, offset: c_int, size: c_int, value: *mut u32) -> c_int {
    // SAFETY: as in cb_dma_alloc.
    let ctx = unsafe { &*(ctx as *const Ctx) };
    let mut b = [0u8; 4];
    match ctx.pci.read(Region::Config, offset as u64, &mut b[..size as usize]) {
        Ok(()) => {
            // SAFETY: valid out pointer.
            unsafe { *value = u32::from_le_bytes(b) };
            0
        }
        Err(_) => -5,
    }
}

extern "C" fn cb_config_write(ctx: *mut c_void, offset: c_int, size: c_int, value: u32) -> c_int {
    // SAFETY: as in cb_dma_alloc.
    let ctx = unsafe { &*(ctx as *const Ctx) };
    match ctx.pci.write(Region::Config, offset as u64, &value.to_le_bytes()[..size as usize]) {
        Ok(()) => 0,
        Err(_) => -5,
    }
}

struct AdevPtr(*mut AmdgpuDevice);
// SAFETY: the C core serializes its own state; the interrupt thread only calls gpud_irq_process.
unsafe impl Send for AdevPtr {}

pub struct Device {
    adev: *mut AmdgpuDevice,
    irq_stop: Arc<AtomicBool>,
    irq_thread: Option<std::thread::JoinHandle<()>>,
    pub interrupts: Arc<AtomicU64>,
    _platform: Box<GpudPlatform>,
    _info: Box<GpudDeviceInfo>,
    _ctx: Box<Ctx>,
    _mmio: Mmio,
    _vram: Mmio,
    _firmware_dir: CString,
}

pub struct RingStatus {
    pub name: String,
    pub ready: bool,
}

impl Device {
    /// Creates the C device and starts the interrupt thread; nothing is initialized yet.
    pub fn create(pci: Arc<dyn PciDevice>, dma: Arc<dyn DmaSpace>, firmware_dir: &str) -> Result<Self> {
        let mmio = pci.map(Region::Bar(2))?;
        let vram = pci.map(Region::Bar(0))?;
        let ctx = Box::new(Ctx { pci: pci.clone(), dma, buffers: Mutex::new(HashMap::new()) });
        let platform = Box::new(GpudPlatform {
            ctx: &*ctx as *const Ctx as *mut c_void,
            log: cb_log,
            dma_alloc: cb_dma_alloc,
            dma_free: cb_dma_free,
            config_read: cb_config_read,
            config_write: cb_config_write,
        });
        let addr = pci.address();
        let firmware_dir = CString::new(firmware_dir)?;
        let info = Box::new(GpudDeviceInfo {
            mmio: mmio.as_ptr().cast(),
            mmio_size: mmio.len(),
            vram: vram.as_ptr().cast(),
            vram_size: vram.len(),
            vram_bar_addr: (pci.config_u32(0x10)? as u64 & !0xf) | ((pci.config_u32(0x14)? as u64) << 32),
            vendor: pci.config_u16(0)?,
            device: pci.config_u16(2)?,
            subsystem_vendor: pci.config_u16(0x2c)?,
            subsystem_device: pci.config_u16(0x2e)?,
            revision: pci.config_u8(8)?,
            bus: addr.bus,
            devfn: addr.device << 3 | addr.function,
            firmware_dir: firmware_dir.as_ptr(),
        });
        // SAFETY: platform and info live in the returned Device as long as adev does.
        let adev = unsafe { gpud_device_create(&*platform, &*info) };
        if adev.is_null() {
            bail!("gpud_device_create failed");
        }

        let irq = pci.enable_msi()?;
        let irq_stop = Arc::new(AtomicBool::new(false));
        let interrupts = Arc::new(AtomicU64::new(0));
        let (stop, count, ptr) = (irq_stop.clone(), interrupts.clone(), AdevPtr(adev));
        let irq_thread = std::thread::Builder::new().name("gpud-irq".into()).spawn(move || {
            let ptr = ptr;
            while !stop.load(Ordering::Relaxed) {
                match irq.wait(Some(Duration::from_millis(50))) {
                    Ok(n) => {
                        count.fetch_add(n, Ordering::Relaxed);
                        // SAFETY: adev outlives this thread (joined in Drop before fini).
                        unsafe { gpud_irq_process(ptr.0) };
                    }
                    Err(Error::Timeout) => {}
                    Err(e) => {
                        eprintln!("[error] interrupt wait: {e}");
                        break;
                    }
                }
            }
        })?;
        Ok(Device {
            adev,
            irq_stop,
            irq_thread: Some(irq_thread),
            interrupts,
            _platform: platform,
            _info: info,
            _ctx: ctx,
            _mmio: mmio,
            _vram: vram,
            _firmware_dir: firmware_dir,
        })
    }

    /// Every supported IP block through amdgpu's init sequence: golden registers, memory controller and GART,
    /// interrupts, GFX (microcode, CP, RLC, rings) and the async DMA engines.
    pub fn init(&self) -> Result<()> {
        // SAFETY: adev is valid.
        let r = unsafe { gpud_device_init(self.adev) };
        if r != 0 {
            bail!("device init failed: {r}");
        }
        Ok(())
    }

    /// The IB test on every ready ring; returns how many failed.
    pub fn ib_tests(&self, timeout: Duration) -> u32 {
        // SAFETY: adev is valid and initialized.
        unsafe { gpud_device_ib_tests(self.adev, timeout.as_millis() as c_long) as u32 }
    }

    pub fn rings(&self) -> Vec<RingStatus> {
        // SAFETY: adev is valid; ring names are NUL-terminated.
        unsafe {
            (0..gpud_ring_count(self.adev))
                .map(|i| RingStatus {
                    name: CStr::from_ptr(gpud_ring_name(self.adev, i)).to_string_lossy().into_owned(),
                    ready: gpud_ring_ready(self.adev, i) != 0,
                })
                .collect()
        }
    }

    pub fn adev(&self) -> *mut AmdgpuDevice {
        self.adev
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.irq_stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.irq_thread.take() {
            let _ = t.join();
        }
        // SAFETY: adev is valid; the interrupt thread has stopped.
        unsafe { gpud_device_fini(self.adev) };
        let _ = self._ctx.pci.disable_irqs();
    }
}
