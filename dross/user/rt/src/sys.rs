//! System calls (`dross-abi` describes the interface).

use dross_abi::{sys as n, Error};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Handle(pub u32);

pub type Result<T> = core::result::Result<T, Error>;

#[cfg(not(target_arch = "x86_64"))]
fn syscall(_: u64, _: [u64; 6]) -> Result<u64> {
    // Host builds of crates that use this (gpud-platform's dross backend) never run on dross.
    Err(Error::BadSyscall)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn syscall(nr: u64, a: [u64; 6]) -> Result<u64> {
    let (status, value): (i64, u64);
    // SAFETY: the kernel's system call ABI; it preserves everything but rax, rdx, rcx and r11.
    unsafe {
        core::arch::asm!(
            "syscall",
            // NATIVE: so the call also works from a Linux program (gpud-platform's backend in gpud, tbd, ethd).
            inlateout("rax") (nr | dross_abi::NATIVE) as i64 => status,
            in("rdi") a[0], in("rsi") a[1], inlateout("rdx") a[2] => value, in("r10") a[3], in("r8") a[4], in("r9") a[5],
            out("rcx") _, out("r11") _,
            options(nostack),
        );
    }
    if status == 0 { Ok(value) } else { Err(Error::from_status(status).unwrap_or(Error::BadSyscall)) }
}

pub fn log(s: &str) {
    let _ = syscall(n::LOG, [s.as_ptr() as u64, s.len() as u64, 0, 0, 0, 0]);
}

pub fn exit(code: i64) -> ! {
    let _ = syscall(n::PROCESS_EXIT, [code as u64, 0, 0, 0, 0, 0]);
    loop {
        let _ = syscall(n::THREAD_EXIT, [0; 6]);
    }
}

pub fn thread_exit() -> ! {
    let _ = syscall(n::THREAD_EXIT, [0; 6]);
    unreachable!()
}

/// `cpu`: `u64::MAX` lets the kernel choose.
pub fn thread_create(entry: u64, stack_top: u64, arg: u64, cpu: u64) -> Result<Handle> {
    syscall(n::THREAD_CREATE, [entry, stack_top, arg, cpu, 0, 0]).map(|h| Handle(h as u32))
}

pub fn yield_now() {
    let _ = syscall(n::YIELD, [0; 6]);
}

/// Nanoseconds since boot.
pub fn clock() -> u64 {
    syscall(n::CLOCK, [0; 6]).unwrap_or(0)
}

pub fn cpu_count() -> u32 {
    syscall(n::CPU_COUNT, [0; 6]).unwrap_or(1) as u32
}

pub fn close(h: Handle) -> Result<()> {
    syscall(n::HANDLE_CLOSE, [h.0 as u64, 0, 0, 0, 0, 0]).map(|_| ())
}

pub fn dup(h: Handle, rights: u32) -> Result<Handle> {
    syscall(n::HANDLE_DUP, [h.0 as u64, rights as u64, 0, 0, 0, 0]).map(|h| Handle(h as u32))
}

pub fn mem_create(size: u64, flags: u64) -> Result<Handle> {
    syscall(n::MEM_CREATE, [size, flags, 0, 0, 0, 0]).map(|h| Handle(h as u32))
}

pub fn mem_size(h: Handle) -> Result<u64> {
    syscall(n::MEM_SIZE, [h.0 as u64, 0, 0, 0, 0, 0])
}

pub fn mem_physical(resource: Handle, phys: u64, size: u64) -> Result<Handle> {
    syscall(n::MEM_PHYSICAL, [resource.0 as u64, phys, size, 0, 0, 0]).map(|h| Handle(h as u32))
}

/// Returns the address (`addr` 0: the kernel chooses).
pub fn vm_map(mem: Handle, offset: u64, len: u64, addr: u64, flags: u64) -> Result<u64> {
    syscall(n::VM_MAP, [mem.0 as u64, offset, len, addr, flags, 0])
}

pub fn vm_unmap(addr: u64, len: u64) -> Result<()> {
    syscall(n::VM_UNMAP, [addr, len, 0, 0, 0, 0]).map(|_| ())
}

/// Starts the ELF image in `image` as a new process with `bootstrap` (moved) as its handle 1.
pub fn spawn(image: Handle, name: &str, bootstrap: Option<Handle>, cpu: u64) -> Result<Handle> {
    syscall(n::SPAWN, [image.0 as u64, name.as_ptr() as u64, name.len() as u64, bootstrap.map_or(0, |h| h.0 as u64), cpu, 0]).map(|h| Handle(h as u32))
}

/// Waits for a process to end; returns its exit code.
pub fn process_wait(p: Handle, deadline: u64) -> Result<i64> {
    syscall(n::PROCESS_WAIT, [p.0 as u64, deadline, 0, 0, 0, 0]).map(|c| c as i64)
}

pub fn channel_create() -> Result<(Handle, Handle)> {
    syscall(n::CHANNEL_CREATE, [0; 6]).map(|v| (Handle(v as u32), Handle((v >> 32) as u32)))
}

/// Sends bytes and handles (the handles move to the reader).
pub fn channel_write(ch: Handle, bytes: &[u8], handles: &[Handle]) -> Result<()> {
    syscall(n::CHANNEL_WRITE, [ch.0 as u64, bytes.as_ptr() as u64, bytes.len() as u64, handles.as_ptr() as u64, handles.len() as u64, 0]).map(|_| ())
}

/// Receives into the buffers; returns (bytes, handles) received.
pub fn channel_read(ch: Handle, bytes: &mut [u8], handles: &mut [Handle], deadline: u64) -> Result<(usize, usize)> {
    syscall(n::CHANNEL_READ, [ch.0 as u64, bytes.as_mut_ptr() as u64, bytes.len() as u64, handles.as_mut_ptr() as u64, handles.len() as u64, deadline])
        .map(|v| ((v & 0xffff_ffff) as usize, (v >> 32) as usize))
}

pub fn channel_bind(ch: Handle, notification: Handle, bits: u64) -> Result<()> {
    syscall(n::CHANNEL_BIND, [ch.0 as u64, notification.0 as u64, bits, 0, 0, 0]).map(|_| ())
}

pub fn notify_create() -> Result<Handle> {
    syscall(n::NOTIFY_CREATE, [0; 6]).map(|h| Handle(h as u32))
}

pub fn notify_signal(h: Handle, bits: u64) -> Result<()> {
    syscall(n::NOTIFY_SIGNAL, [h.0 as u64, bits, 0, 0, 0, 0]).map(|_| ())
}

/// Waits for any bit; returns (and clears) them all.
pub fn notify_wait(h: Handle, deadline: u64) -> Result<u64> {
    syscall(n::NOTIFY_WAIT, [h.0 as u64, deadline, 0, 0, 0, 0])
}

pub fn resource_sub(r: Handle, base: u64, size: u64) -> Result<Handle> {
    syscall(n::RESOURCE_SUB, [r.0 as u64, base, size, 0, 0, 0]).map(|h| Handle(h as u32))
}

pub fn ioport_enable(r: Handle, port: u16, count: u32) -> Result<()> {
    syscall(n::IOPORT_ENABLE, [r.0 as u64, port as u64, count as u64, 0, 0, 0]).map(|_| ())
}

/// An MSI vector that signals `bits` on `notification`. Returns the interrupt and the (address, data) to
/// program into the device.
pub fn irq_msi(irq: Handle, notification: Handle, bits: u64, cpu: u64) -> Result<(Handle, u64, u32)> {
    let mut out = [0u8; 12];
    let h = syscall(n::IRQ_MSI, [irq.0 as u64, notification.0 as u64, bits, cpu, out.as_mut_ptr() as u64, 0])?;
    Ok((Handle(h as u32), u64::from_le_bytes(out[..8].try_into().unwrap()), u32::from_le_bytes(out[8..].try_into().unwrap())))
}

pub fn dma_domain_create(iommu: Handle) -> Result<Handle> {
    syscall(n::DMA_DOMAIN_CREATE, [iommu.0 as u64, 0, 0, 0, 0, 0]).map(|h| Handle(h as u32))
}

pub fn dma_attach(domain: Handle, iommu: Handle, bdf: u16) -> Result<()> {
    syscall(n::DMA_ATTACH, [domain.0 as u64, iommu.0 as u64, bdf as u64, 0, 0, 0]).map(|_| ())
}

pub fn dma_map(domain: Handle, mem: Handle, offset: u64, len: u64, iova: u64, flags: u64) -> Result<()> {
    syscall(n::DMA_MAP, [domain.0 as u64, mem.0 as u64, offset, len, iova, flags]).map(|_| ())
}

pub fn dma_unmap(domain: Handle, iova: u64, len: u64) -> Result<()> {
    syscall(n::DMA_UNMAP, [domain.0 as u64, iova, len, 0, 0, 0]).map(|_| ())
}

/// Starts a static Linux executable with `argv` and `envp`; `bootstrap` (moved) becomes its native handle 1.
pub fn spawn_linux(image: Handle, argv: &[&str], envp: &[&str], bootstrap: Option<Handle>) -> Result<Handle> {
    let join = |v: &[&str]| v.iter().flat_map(|s| s.bytes().chain(core::iter::once(0))).collect::<alloc::vec::Vec<u8>>();
    let (a, e) = (join(argv), join(envp));
    syscall(n::SPAWN_LINUX, [image.0 as u64, a.as_ptr() as u64, a.len() as u64, e.as_ptr() as u64, e.len() as u64, bootstrap.map_or(0, |h| h.0 as u64)])
        .map(|h| Handle(h as u32))
}

/// From a Linux program: an fd for a notification (polls readable while bits are pending) or a memory object
/// (mappable).
pub fn handle_fd(h: Handle) -> Result<i32> {
    syscall(n::HANDLE_FD, [h.0 as u64, 0, 0, 0, 0, 0]).map(|fd| fd as i32)
}

pub fn system_exit(system: Handle, code: u64) -> ! {
    let _ = syscall(n::SYSTEM_EXIT, [system.0 as u64, code, 0, 0, 0, 0]);
    exit(code as i64)
}

/// Deadline `ms` milliseconds from now.
pub fn after_ms(ms: u64) -> u64 {
    clock() + ms * 1_000_000
}

/// Secure entropy from a kernel built with the Unix feature. No fallback to time.
pub fn random(bytes: &mut [u8]) -> Result<()> {
    syscall(n::RANDOM, [bytes.as_mut_ptr() as u64, bytes.len() as u64, 0, 0, 0, 0]).map(|_| ())
}

/// Registers `service` as the kernel's network service (netd's end gets the sockets of Linux programs). Needs the
/// system resource; kernels built with the Unix feature only.
pub fn unix_net_register(system: Handle, service: Handle) -> Result<()> {
    syscall(n::UNIX_NET_REGISTER, [system.0 as u64, service.0 as u64, 0, 0, 0, 0]).map(|_| ())
}
