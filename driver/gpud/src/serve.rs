//! `gpud serve`: the driver server. Brings the GPU up, then serves clients on a SOCK_SEQPACKET Unix socket
//! (driver/shim/include/gpud_proto.h). Each client gets its own GPU VM (VMID), a buffer file (memfd) whose pages
//! back its host-visible buffers, and a thread. Buffers never move: VRAM buffers come from the client VRAM range,
//! host buffers are pages of the buffer file pinned in the IOMMU, and both are mapped into the client's VM where
//! it asks.

use crate::hw::{AmdgpuDevice, Device};
use crate::proto::{self, *};
use crate::retirement::LastUse;
use anyhow::{Context, Result, bail};
use gpud_platform::iova::IovaAllocator;
use gpud_platform::{DmaAccess, DmaBuffer, DmaSpace, PciDevice};
use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr::NonNull;
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

#[repr(C)]
struct GpudVm {
    _opaque: [u8; 0],
}

#[repr(C)]
struct GpudIb {
    va: u64,
    size_dw: u32,
    flags: u32,
}

unsafe extern "C" {
    fn gpud_vm_create(adev: *mut AmdgpuDevice, vmid: u32) -> *mut GpudVm;
    fn gpud_vm_destroy(vm: *mut GpudVm);
    fn gpud_vm_map(vm: *mut GpudVm, va: u64, size: u64, addr: u64, flags: u64) -> i32;
    fn gpud_vm_unmap(vm: *mut GpudVm, va: u64, size: u64) -> i32;
    fn gpud_vram_alloc(adev: *mut AmdgpuDevice, size: u64, align: u64, offset: *mut u64) -> i32;
    fn gpud_vram_free(adev: *mut AmdgpuDevice, offset: u64, size: u64);
    fn gpud_ring_for(adev: *mut AmdgpuDevice, ip_type: u32, ring: u32) -> i32;
    fn gpud_submit(
        adev: *mut AmdgpuDevice,
        ring: i32,
        vm: *mut GpudVm,
        ibs: *const GpudIb,
        n: u32,
        seq: *mut u64,
    ) -> i32;
    fn gpud_fence_wait(adev: *mut AmdgpuDevice, ring: i32, seq: u64, timeout_ns: i64) -> i32;
    fn gpud_fence_read(adev: *mut AmdgpuDevice, ring: i32) -> u64;
    fn gpud_info(adev: *mut AmdgpuDevice, query: *const u8, out: *mut u8, size: u32) -> i32;
}

const DOMAIN_GTT: u64 = 0x2;
const DOMAIN_VRAM: u64 = 0x4;
const CREATE_NO_CPU_ACCESS: u64 = 1 << 1;
const VA_OP_MAP: u32 = 1;
const VA_OP_UNMAP: u32 = 2;
const VA_OP_CLEAR: u32 = 3;
const VA_OP_REPLACE: u32 = 4;
const VM_PAGE_READABLE: u32 = 1 << 1;
const VM_PAGE_WRITEABLE: u32 = 1 << 2;
const VM_PAGE_PRT: u32 = 1 << 4;
const PTE_VALID: u64 = 1 << 0;
const PTE_SYSTEM: u64 = 1 << 1;
const PTE_SNOOPED: u64 = 1 << 2;
const PTE_READABLE: u64 = 1 << 5;
const PTE_WRITEABLE: u64 = 1 << 6;
const PAGE: u64 = 4096;
/// The page-table fragment size gpud reports to clients (AMDGPU_INFO_DEV_INFO pte_fragment_size).
const FRAGMENT: u64 = 2 << 20;

/// Host memory a client may have in buffers at once (the size of its buffer file; pages exist only when used).
const BO_FILE_SIZE: u64 = 32 << 30;
/// Under dross a memfd's memory is allocated up front, not on first touch, so a client's buffer file is smaller.
const BO_FILE_SIZE_DROSS: u64 = 4 << 30;

fn bo_file_size() -> u64 {
    if gpud_platform::on_dross() { BO_FILE_SIZE_DROSS } else { BO_FILE_SIZE }
}

pub(crate) struct Shared {
    pub(crate) device: Device,
    dma: Arc<dyn DmaSpace>,
    pci: Arc<dyn PciDevice>,
    pub(crate) vmids: Mutex<[bool; 16]>,
    /// Readers protect requests/cleanup; shutdown excludes every device user.
    pub(crate) accepting: RwLock<bool>,
    pub(crate) submitted: Mutex<LastUse>,
    pub(crate) display_live: AtomicBool,
}

// SAFETY: the C core locks its own state (submission, VM tables, VRAM allocator); the raw pointer is the device.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    pub(crate) fn adev(&self) -> *mut AmdgpuDevice {
        self.device.adev()
    }
}

enum Placement {
    Vram { offset: u64 },
    Gtt { file_offset: u64, dma: DmaBuffer },
    Imported { dma: DmaBuffer, mapping: ImportedMapping },
}

/// The driver owns both an fd and its CPU mapping independently of the exporter.
struct ImportedMapping {
    _fd: OwnedFd,
    ptr: *mut c_void,
    size: usize,
}
impl Drop for ImportedMapping {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr, self.size);
        }
    }
}

struct Bo {
    placement: Placement,
    rep: BoRep,
}

struct Client {
    shared: Arc<Shared>,
    vm: *mut GpudVm,
    vmid: u32,
    bo_file: Option<OwnedFd>,
    bo_map: *mut u8,
    bo_space: IovaAllocator,
    bos: HashMap<u32, Bo>,
    next_handle: u32,
    last_fence: LastUse,
}

pub fn run(pci: Arc<dyn PciDevice>, dma: Arc<dyn DmaSpace>, firmware_dir: &str, socket: &str) -> Result<()> {
    // SIGINT and SIGTERM run the teardown before exiting, so the next owner finds the engines stopped; SIGUSR1 logs
    // the GPU's state (for a stuck submission). They are blocked before any thread starts (Device::create starts the
    // interrupt thread): every thread inherits the mask, and only the waiter below sees them, after init.
    // SAFETY: signal mask calls on a local set.
    let signals = unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGUSR1);
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        set
    };
    let device = Device::create(pci.clone(), dma.clone(), firmware_dir)?;
    device.init()?;
    let failed = device.ib_tests(std::time::Duration::from_secs(1));
    if failed > 0 {
        bail!("{failed} IB tests failed");
    }
    let shared = Arc::new(Shared {
        device,
        dma,
        pci,
        vmids: Mutex::new([false; 16]),
        accepting: RwLock::new(true),
        submitted: Mutex::new(LastUse::default()),
        display_live: AtomicBool::new(false),
    });
    crate::display::start(shared.clone());
    let s = shared.clone();
    std::thread::spawn(move || {
        loop {
            let mut sig = 0;
            // SAFETY: waits on the set blocked above.
            unsafe { libc::sigwait(&signals, &mut sig) };
            if sig == libc::SIGUSR1 {
                s.device.dump();
                continue;
            }
            // Once draining, no request may queue more GPU work. If a fence has
            // not completed, retain the process/device/backing; a reset or device
            // detachment is required before the operator may force its exit.
            let mut accepting = s.accepting.write().unwrap();
            *accepting = false;
            if s.display_live.load(Ordering::Acquire) {
                crate::hw::log("error", "display owner must release scanout before driver shutdown; backing retained");
                continue;
            }
            if !s.submitted.lock().unwrap().complete(|ring| unsafe { gpud_fence_read(s.adev(), ring) }) {
                crate::hw::log(
                    "error",
                    "GPU has unfinished work: driver remains alive with backing quarantined; reset/detach before forcing exit",
                );
                continue;
            }
            crate::hw::log("info", &format!("signal {sig}: stopping the drained GPU"));
            s.device.shutdown();
            std::process::exit(0);
        }
    });

    let listener = listen(socket)?;
    crate::supervisor::start(shared.clone());
    eprintln!("[info] gpud serving {} on {socket}", shared.pci.address());
    loop {
        // SAFETY: accept on our listening socket.
        let fd = unsafe {
            libc::accept4(listener.as_raw_fd(), std::ptr::null_mut(), std::ptr::null_mut(), libc::SOCK_CLOEXEC)
        };
        if fd < 0 {
            eprintln!("[error] accept: {}", std::io::Error::last_os_error());
            continue;
        }
        // SAFETY: a fresh descriptor.
        let conn = unsafe { OwnedFd::from_raw_fd(fd) };
        let shared = shared.clone();
        std::thread::spawn(move || {
            if let Err(e) = serve_client(shared, conn) {
                eprintln!("[warning] client: {e:#}");
            }
        });
    }
}

fn listen(path: &str) -> Result<OwnedFd> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir)?;
    }
    let _ = std::fs::remove_file(path);
    // SAFETY: plain socket calls with checked results.
    unsafe {
        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0);
        if fd < 0 {
            bail!("socket: {}", std::io::Error::last_os_error());
        }
        let fd = OwnedFd::from_raw_fd(fd);
        let mut addr: libc::sockaddr_un = std::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as _;
        for (i, b) in path.bytes().enumerate().take(addr.sun_path.len() - 1) {
            addr.sun_path[i] = b as _;
        }
        if libc::bind(fd.as_raw_fd(), (&addr as *const libc::sockaddr_un).cast(), size_of::<libc::sockaddr_un>() as _)
            < 0
        {
            bail!("bind {path}: {}", std::io::Error::last_os_error());
        }
        let c = CString::new(path)?;
        libc::chmod(c.as_ptr(), 0o666);
        if libc::listen(fd.as_raw_fd(), 16) < 0 {
            bail!("listen: {}", std::io::Error::last_os_error());
        }
        Ok(fd)
    }
}

fn recv(fd: RawFd, buf: &mut [u8]) -> Result<(usize, Option<OwnedFd>)> {
    let mut iov = libc::iovec { iov_base: buf.as_mut_ptr().cast(), iov_len: buf.len() };
    let mut control = [0usize; 16];
    unsafe {
        let mut hdr: libc::msghdr = std::mem::zeroed();
        hdr.msg_iov = &mut iov;
        hdr.msg_iovlen = 1;
        hdr.msg_control = control.as_mut_ptr().cast();
        hdr.msg_controllen = size_of_val(&control) as _;
        let n = libc::recvmsg(fd, &mut hdr, libc::MSG_CMSG_CLOEXEC);
        if n < 0 {
            bail!("recvmsg: {}", std::io::Error::last_os_error());
        }
        let mut fds = Vec::new();
        let mut c = libc::CMSG_FIRSTHDR(&hdr);
        while !c.is_null() {
            if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                let bytes = (*c).cmsg_len as usize - libc::CMSG_LEN(0) as usize;
                for i in 0..bytes / size_of::<RawFd>() {
                    fds.push(OwnedFd::from_raw_fd(std::ptr::read_unaligned(libc::CMSG_DATA(c).cast::<RawFd>().add(i))));
                }
            }
            c = libc::CMSG_NXTHDR(&hdr, c);
        }
        if hdr.msg_flags & (libc::MSG_TRUNC | libc::MSG_CTRUNC) != 0 || fds.len() > 1 {
            bail!("oversized GPU request/ancillary data");
        }
        Ok((n as usize, fds.pop()))
    }
}

fn send(fd: RawFd, status: i32, payload: &[u8], pass_fd: Option<RawFd>) -> Result<()> {
    let mut msg = Vec::with_capacity(8 + payload.len());
    msg.extend_from_slice(proto::bytes(&Rep { status, len: payload.len() as u32 }));
    msg.extend_from_slice(payload);
    let mut iov = libc::iovec { iov_base: msg.as_mut_ptr().cast(), iov_len: msg.len() };
    let mut cbuf = [0u8; 64];
    // SAFETY: a msghdr over our buffers; the control message carries one fd when asked.
    unsafe {
        let mut hdr: libc::msghdr = std::mem::zeroed();
        hdr.msg_iov = &mut iov;
        hdr.msg_iovlen = 1;
        if let Some(pfd) = pass_fd {
            hdr.msg_control = cbuf.as_mut_ptr().cast();
            hdr.msg_controllen = libc::CMSG_SPACE(size_of::<RawFd>() as u32) as _;
            let c = libc::CMSG_FIRSTHDR(&hdr);
            (*c).cmsg_level = libc::SOL_SOCKET;
            (*c).cmsg_type = libc::SCM_RIGHTS;
            (*c).cmsg_len = libc::CMSG_LEN(size_of::<RawFd>() as u32) as _;
            std::ptr::write_unaligned(libc::CMSG_DATA(c).cast::<RawFd>(), pfd);
        }
        if libc::sendmsg(fd, &hdr, libc::MSG_NOSIGNAL) < 0 {
            bail!("sendmsg: {}", std::io::Error::last_os_error());
        }
    }
    Ok(())
}

fn serve_client(shared: Arc<Shared>, conn: OwnedFd) -> Result<()> {
    let accepting = shared.accepting.read().unwrap();
    if !*accepting {
        bail!("GPU is draining");
    }
    let vmid = {
        let mut v = shared.vmids.lock().unwrap();
        let id = (1..8).find(|&i| !v[i]).context("no free VMID")?;
        v[id] = true;
        id as u32
    };
    // SAFETY: adev is valid; the VM lives as long as the client.
    let vm = unsafe { gpud_vm_create(shared.adev(), vmid) };
    if vm.is_null() {
        shared.vmids.lock().unwrap()[vmid as usize] = false;
        bail!("VM {vmid}: out of page table memory");
    }
    let mut client = Client {
        shared: shared.clone(),
        vm,
        vmid,
        bo_file: None,
        bo_map: std::ptr::null_mut(),
        bo_space: IovaAllocator::new(std::slice::from_ref(&(0..bo_file_size()))),
        bos: HashMap::new(),
        next_handle: 1,
        last_fence: LastUse::default(),
    };
    drop(accepting);
    eprintln!("[info] client connected: VMID {vmid}");
    let mut buf = vec![0u8; MAX_MSG];
    let r = loop {
        let (n, incoming) = match recv(conn.as_raw_fd(), &mut buf) {
            Ok((0, _)) => break Ok(()),
            Ok(v) => v,
            Err(e) => break Err(e),
        };
        let Some(req) = proto::read::<Req>(&buf[..n], 0) else { break Err(anyhow::anyhow!("short request")) };
        if n != 8 + req.len as usize {
            break Err(anyhow::anyhow!("invalid GPU request length"));
        }
        let payload = &buf[8..n];
        let (status, reply, fd) = match client.handle(req.op, payload, incoming) {
            Ok((reply, fd)) => (0, reply, fd),
            Err(errno) => (-errno, Vec::new(), None),
        };
        if let Err(e) = send(conn.as_raw_fd(), status, &reply, fd) {
            break Err(e);
        }
    };
    client.close();
    shared.vmids.lock().unwrap()[vmid as usize] = false;
    eprintln!("[info] client on VMID {vmid} gone");
    r
}

impl Client {
    fn adev(&self) -> *mut AmdgpuDevice {
        self.shared.adev()
    }

    fn handle(
        &mut self,
        op: u32,
        p: &[u8],
        incoming: Option<OwnedFd>,
    ) -> std::result::Result<(Vec<u8>, Option<RawFd>), i32> {
        let shared = self.shared.clone();
        let accepting = shared.accepting.read().unwrap();
        if !*accepting {
            return Err(libc::ESHUTDOWN);
        }
        if op != BO_IMPORT && incoming.is_some() {
            return Err(libc::EINVAL);
        }
        match op {
            BO_IMPORT => self
                .bo_import(proto::read(p, 0).ok_or(libc::EINVAL)?, incoming.ok_or(libc::EINVAL)?)
                .map(|r| (proto::bytes(&r).to_vec(), None)),
            HELLO => self.hello(),
            INFO => {
                let q: InfoReq = proto::read(p, 0).ok_or(libc::EINVAL)?;
                let size = u32::from_le_bytes(q.query[8..12].try_into().unwrap());
                if size == 0 || size as usize > MAX_MSG - 64 {
                    return Err(libc::EINVAL);
                }
                let mut out = vec![0u8; size as usize];
                // SAFETY: query is a 32-byte drm_amdgpu_info; out has size bytes.
                let r = unsafe { gpud_info(self.adev(), q.query.as_ptr(), out.as_mut_ptr(), size) };
                if r != 0 { Err(-r) } else { Ok((out, None)) }
            }
            BO_CREATE => {
                self.bo_create(proto::read(p, 0).ok_or(libc::EINVAL)?).map(|r| (proto::bytes(&r).to_vec(), None))
            }
            BO_FREE => {
                let h: BoHandle = proto::read(p, 0).ok_or(libc::EINVAL)?;
                self.bo_free(h.handle).map(|_| (Vec::new(), None))
            }
            BO_QUERY => {
                let h: BoHandle = proto::read(p, 0).ok_or(libc::EINVAL)?;
                let bo = self.bos.get(&h.handle).ok_or(libc::ENOENT)?;
                Ok((proto::bytes(&bo.rep).to_vec(), None))
            }
            VA => self.va(proto::read(p, 0).ok_or(libc::EINVAL)?).map(|_| (Vec::new(), None)),
            SUBMIT => self.submit(p).map(|f| (proto::bytes(&FenceRep { fence: f }).to_vec(), None)),
            WAIT => {
                let w: WaitReq = proto::read(p, 0).ok_or(libc::EINVAL)?;
                // SAFETY: adev is valid; the ring index is checked by the C side.
                let r = unsafe {
                    gpud_fence_wait(
                        self.adev(),
                        fence_ring(w.fence),
                        fence_seq(w.fence),
                        if w.timeout_ns < 0 { 500_000_000 } else { w.timeout_ns.min(500_000_000) },
                    )
                };
                if r != 0 { Err(-r) } else { Ok((Vec::new(), None)) }
            }
            RING => {
                let q: RingReq = proto::read(p, 0).ok_or(libc::EINVAL)?;
                // SAFETY: adev is valid.
                let ring = unsafe { gpud_ring_for(self.adev(), q.ip_type, q.ring) };
                if ring < 0 {
                    return Err(-ring);
                }
                // SAFETY: as above.
                let last = unsafe { gpud_fence_read(self.adev(), ring) };
                Ok((proto::bytes(&RingRep { ring, pad: 0, last_fence: fence(ring, last) }).to_vec(), None))
            }
            _ => Err(libc::EINVAL),
        }
    }

    fn hello(&mut self) -> std::result::Result<(Vec<u8>, Option<RawFd>), i32> {
        if self.bo_file.is_none() {
            let name = CString::new("gpud-buffers").unwrap();
            // SAFETY: memfd_create, ftruncate and mmap with checked results.
            unsafe {
                let fd = libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC);
                if fd < 0 {
                    return Err(libc::ENOMEM);
                }
                let fd = OwnedFd::from_raw_fd(fd);
                if libc::ftruncate(fd.as_raw_fd(), bo_file_size() as libc::off_t) < 0 {
                    return Err(libc::ENOMEM);
                }
                let p = libc::mmap(
                    std::ptr::null_mut(),
                    bo_file_size() as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                );
                if p == libc::MAP_FAILED {
                    return Err(libc::ENOMEM);
                }
                self.bo_map = p.cast();
                self.bo_file = Some(fd);
            }
        }
        let pci = &self.shared.pci;
        let a = pci.address();
        let cfg16 = |o: u16| pci.config_u16(o).unwrap_or(0);
        let rep = HelloRep {
            version: VERSION,
            vmid: self.vmid,
            bo_file_size: bo_file_size(),
            pci_domain: a.domain,
            pci_bus: a.bus,
            pci_dev: a.device,
            pci_func: a.function,
            revision: pci.config_u8(8).unwrap_or(0),
            vendor_id: cfg16(0),
            device_id: cfg16(2),
            subvendor_id: cfg16(0x2c),
            subdevice_id: cfg16(0x2e),
        };
        Ok((proto::bytes(&rep).to_vec(), self.bo_file.as_ref().map(|f| f.as_raw_fd())))
    }

    fn bo_create(&mut self, req: BoCreateReq) -> std::result::Result<BoRep, i32> {
        if self.bo_file.is_none() || req.size == 0 {
            return Err(libc::EINVAL);
        }
        let size = req.size.checked_add(PAGE - 1).ok_or(libc::EINVAL)? & !(PAGE - 1);
        let align = req.alignment.max(PAGE).checked_next_power_of_two().ok_or(libc::EINVAL)?;
        // VRAM only for buffers the client promises never to map (NO_CPU_ACCESS: RADV's device-local memory
        // types): no VRAM is CPU-visible to clients, and anything else may be mapped later.
        let want_vram = req.domains & DOMAIN_VRAM != 0 && req.flags & CREATE_NO_CPU_ACCESS != 0;
        let mut placement = None;
        if want_vram {
            // amdgpu's VRAM manager hands out buffers this large in 2 MiB blocks (amdgpu_vram_mgr_new), so their
            // physical address shares the 2 MiB alignment RADV gives their virtual address (pte_fragment_size) and
            // the page tables map them in 2 MiB fragments. With only the requested alignment the fragments shrink
            // to the common alignment of the two, and the TLB covers that much less.
            let align = if size >= FRAGMENT { align.max(FRAGMENT) } else { align };
            let mut off = 0;
            // SAFETY: adev is valid.
            if unsafe { gpud_vram_alloc(self.adev(), size, align, &mut off) } == 0 {
                placement = Some(Placement::Vram { offset: off });
            }
        }
        if placement.is_none() {
            if req.domains & (DOMAIN_GTT | DOMAIN_VRAM) == 0 {
                return Err(libc::EINVAL);
            }
            let off = self.bo_space.alloc(size, align.min(2 << 20)).map_err(|_| libc::ENOMEM)?;
            // SAFETY: bo_map covers the whole buffer file; the range stays mapped until the buffer is freed.
            let ptr = NonNull::new(unsafe { self.bo_map.add(off as usize) }).unwrap();
            // SAFETY: as above.
            match unsafe { self.shared.dma.map_external(ptr, size as usize, DmaAccess::ReadWrite) } {
                Ok(dma) => placement = Some(Placement::Gtt { file_offset: off, dma }),
                Err(e) => {
                    eprintln!("[error] mapping a {size}-byte buffer: {e}");
                    self.bo_space.release(off..off + size);
                    return Err(libc::ENOMEM);
                }
            }
        }
        let placement = placement.unwrap();
        let (domain, mmap_offset) = match &placement {
            Placement::Vram { .. } => (DOMAIN_VRAM as u32, NO_MMAP),
            Placement::Gtt { file_offset, .. } => (DOMAIN_GTT as u32, *file_offset),
            Placement::Imported { .. } => unreachable!(),
        };
        let handle = self.next_handle;
        self.next_handle += 1;
        let rep = BoRep { handle, domain, size, mmap_offset, alignment: align, domains: req.domains, flags: req.flags };
        self.bos.insert(handle, Bo { placement, rep });
        Ok(rep)
    }

    fn bo_import(&mut self, req: BoImportReq, fd: OwnedFd) -> std::result::Result<BoRep, i32> {
        if self.bo_file.is_none()
            || req.size == 0
            || req.size > 64 << 20
            || !req.offset.is_multiple_of(PAGE)
            || !req.size.is_multiple_of(PAGE)
        {
            return Err(libc::EINVAL);
        }
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.as_raw_fd(), &mut stat) } != 0
            || stat.st_size < 0
            || req.offset.checked_add(req.size).is_none_or(|end| end > stat.st_size as u64)
        {
            return Err(libc::EINVAL);
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                req.size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                req.offset as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(libc::ENOMEM);
        }
        let mapping = ImportedMapping { _fd: fd, ptr, size: req.size as usize };
        let dma = unsafe {
            self.shared.dma.map_external(NonNull::new(ptr.cast()).unwrap(), req.size as usize, DmaAccess::ReadWrite)
        }
        .map_err(|_| libc::ENOMEM)?;
        let handle = self.next_handle;
        self.next_handle = handle.checked_add(1).ok_or(libc::ENOMEM)?;
        let rep = BoRep {
            handle,
            domain: DOMAIN_GTT as u32,
            size: req.size,
            mmap_offset: NO_MMAP,
            alignment: PAGE,
            domains: DOMAIN_GTT,
            flags: 0,
        };
        self.bos.insert(handle, Bo { placement: Placement::Imported { dma, mapping }, rep });
        Ok(rep)
    }

    /// Frees a buffer once every ring this client used has finished its work.
    fn bo_free(&mut self, handle: u32) -> std::result::Result<(), i32> {
        if !self.bos.contains_key(&handle) {
            return Err(libc::ENOENT);
        }
        let deadline = Instant::now() + Duration::from_millis(500);
        while !self.last_fence.complete(|ring| unsafe { gpud_fence_read(self.adev(), ring) }) {
            if Instant::now() >= deadline {
                // Leave the BO, file offset and DMA mapping installed. Neither
                // this handle nor its physical range may be recycled yet.
                return Err(libc::ETIMEDOUT);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        let bo = self.bos.remove(&handle).unwrap();
        match bo.placement {
            // SAFETY: adev is valid; the range came from gpud_vram_alloc.
            Placement::Vram { offset } => unsafe { gpud_vram_free(self.adev(), offset, bo.rep.size) },
            Placement::Imported { dma, mapping } => {
                drop(dma);
                drop(mapping);
            }
            Placement::Gtt { file_offset, dma } => {
                drop(dma);
                if let Some(f) = &self.bo_file {
                    // SAFETY: returns the pages of a range nobody maps any more.
                    unsafe {
                        libc::fallocate(
                            f.as_raw_fd(),
                            libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                            file_offset as libc::off_t,
                            bo.rep.size as libc::off_t,
                        )
                    };
                }
                self.bo_space.release(file_offset..file_offset + bo.rep.size);
            }
        }
        Ok(())
    }

    fn va(&mut self, req: VaReq) -> std::result::Result<(), i32> {
        let (va, size) = (req.va, req.size);
        if !va.is_multiple_of(PAGE) || !size.is_multiple_of(PAGE) || !req.offset.is_multiple_of(PAGE) {
            return Err(libc::EINVAL);
        }
        // A partially-resident (sparse) range reserves address space with nothing behind it; gpud leaves it
        // unmapped (SI's PRT aperture isn't set up).
        let prt = req.flags & VM_PAGE_PRT != 0;
        let r = match req.operation {
            VA_OP_MAP | VA_OP_REPLACE if prt => {
                // SAFETY: vm is valid.
                unsafe { gpud_vm_unmap(self.vm, va, size) }
            }
            VA_OP_UNMAP | VA_OP_CLEAR => {
                // SAFETY: vm is valid.
                unsafe { gpud_vm_unmap(self.vm, va, size) }
            }
            VA_OP_MAP | VA_OP_REPLACE => {
                let bo = self.bos.get(&req.handle).ok_or(libc::ENOENT)?;
                if req.offset.checked_add(size).is_none_or(|end| end > bo.rep.size) {
                    return Err(libc::EINVAL);
                }
                let mut flags = PTE_VALID;
                if req.flags & VM_PAGE_READABLE != 0 {
                    flags |= PTE_READABLE;
                }
                if req.flags & VM_PAGE_WRITEABLE != 0 {
                    flags |= PTE_WRITEABLE;
                }
                let addr = match &bo.placement {
                    Placement::Vram { offset } => offset + req.offset,
                    Placement::Gtt { dma, .. } | Placement::Imported { dma, .. } => {
                        flags |= PTE_SYSTEM | PTE_SNOOPED;
                        dma.iova() + req.offset
                    }
                };
                // SAFETY: vm is valid.
                unsafe { gpud_vm_map(self.vm, va, size, addr, flags) }
            }
            _ => return Err(libc::EINVAL),
        };
        if r != 0 { Err(-r) } else { Ok(()) }
    }

    fn submit(&mut self, p: &[u8]) -> std::result::Result<u64, i32> {
        let req: SubmitReq = proto::read(p, 0).ok_or(libc::EINVAL)?;
        if !(1..=16).contains(&req.num_ibs)
            || req.num_deps > 16
            || p.len() != 16 + req.num_ibs as usize * 16 + req.num_deps as usize * 8
        {
            return Err(libc::EINVAL);
        }
        let ibs: Vec<GpudIb> = (0..req.num_ibs as usize)
            .map(|i| {
                proto::read::<IbDesc>(p, 16 + i * 16).map(|d| GpudIb { va: d.va, size_dw: d.size_dw, flags: d.flags })
            })
            .collect::<Option<_>>()
            .ok_or(libc::EINVAL)?;
        let deps_at = 16 + req.num_ibs as usize * 16;
        // SAFETY: adev is valid.
        let ring = unsafe { gpud_ring_for(self.adev(), req.ip_type, req.ring) };
        if ring < 0 {
            return Err(-ring);
        }
        for i in 0..req.num_deps as usize {
            let dep: u64 = proto::read(p, deps_at + i * 8).ok_or(libc::EINVAL)?;
            // Work queued earlier on the same ring runs first anyway; other rings are waited for.
            if fence_ring(dep) != ring {
                // SAFETY: adev is valid.
                let result = unsafe { gpud_fence_wait(self.adev(), fence_ring(dep), fence_seq(dep), 500_000_000) };
                if result != 0 {
                    return Err(-result);
                }
            }
        }
        let mut seq = 0;
        // SAFETY: adev and vm are valid; ibs has num_ibs entries.
        let r = unsafe { gpud_submit(self.adev(), ring, self.vm, ibs.as_ptr(), ibs.len() as u32, &mut seq) };
        if r != 0 {
            return Err(-r);
        }
        let f = fence(ring, seq);
        self.last_fence.record(ring, seq);
        self.shared.submitted.lock().unwrap().record(ring, seq);
        Ok(f)
    }

    fn close(&mut self) {
        let mut reported = false;
        // Disconnected clients keep their VMID, file mapping and every BO until
        // all last uses finish. The bounded VMID pool bounds quarantined clients.
        // A hung device requires explicit detach/reset; closing the socket alone
        // never licenses reuse of memory still reachable by the GPU.
        loop {
            let shared = self.shared.clone();
            let _access = shared.accepting.read().unwrap();
            if self.last_fence.complete(|ring| unsafe { gpud_fence_read(self.adev(), ring) }) {
                let handles: Vec<_> = self.bos.keys().copied().collect();
                for handle in handles {
                    self.bo_free(handle).expect("completed fences cannot fail retirement");
                }
                if !self.bo_map.is_null() {
                    unsafe { libc::munmap(self.bo_map.cast::<c_void>(), bo_file_size() as usize) };
                    self.bo_map = std::ptr::null_mut();
                }
                unsafe { gpud_vm_destroy(self.vm) };
                self.vm = std::ptr::null_mut();
                return;
            }
            if !reported {
                crate::hw::log(
                    "warning",
                    &format!("VMID {} disconnected with GPU work pending: backing quarantined", self.vmid),
                );
                reported = true;
            }
            drop(_access);
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
