//! System calls (`dross-abi` describes the interface).

use core::sync::atomic::AtomicU32;
use dross_abi::{sys as n, Error};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Handle(pub u32);

pub type Result<T> = core::result::Result<T, Error>;

#[cfg(not(any(target_arch = "x86_64", target_arch = "riscv64", all(target_arch = "aarch64", not(target_vendor = "apple")))))]
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

/// All the RAM there is, in bytes.
pub fn memory_total() -> u64 {
    syscall(n::MEMORY_TOTAL, [0; 6]).unwrap_or(0)
}

/// Supervisor collection of unreachable native queued-capability cycles.
pub fn channel_collect(system: Handle) -> Result<u64> { syscall(n::CHANNEL_COLLECT, [system.0 as u64, 0, 0, 0, 0, 0]) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMemoryInfo {
    pub physical_free: u64,
    pub heap_committed: u64,
    pub heap_used: u64,
    pub stack_tables: u64,
    /// Deferred heap or stack-table reclamation is still pending.
    pub reclaim_pending: bool,
}
/// Supervisor-only global counters. Fields are sequential samples, not an atomic snapshot.
pub fn system_memory_info(system: Handle) -> Result<SystemMemoryInfo> {
    let mut values = [0u64; 5];
    syscall(n::SYSTEM_MEMORY_INFO, [system.0 as u64, values.as_mut_ptr() as u64, 0, 0, 0, 0])?;
    Ok(SystemMemoryInfo { physical_free: values[0], heap_committed: values[1], heap_used: values[2],
        stack_tables: values[3], reclaim_pending: values[4] != 0 })
}
/// Private coordinator boundary. PREFLIGHT does not authorize PCI Restore;
/// only a successful ENTER return carrying KERNEL_READY does. Drivers, PCI,
/// AML/RTC and watchdog ownership must already be prepared by the caller.
pub fn system_suspend(system: Handle, request: &dross_abi::pm::Request) -> Result<u64> {
    syscall(n::SYSTEM_SUSPEND, [system.0 as u64, request as *const _ as u64, dross_abi::pm::Request::BYTES as u64, 0, 0, 0])
}

pub fn close(h: Handle) -> Result<()> {
    syscall(n::HANDLE_CLOSE, [h.0 as u64, 0, 0, 0, 0, 0]).map(|_| ())
}

pub fn dup(h: Handle, rights: u32) -> Result<Handle> {
    syscall(n::HANDLE_DUP, [h.0 as u64, rights as u64, 0, 0, 0, 0]).map(|h| Handle(h as u32))
}

/// Creates a child of the current default budget; requires the system resource.
pub fn account_create(system: Handle, limit: u64) -> Result<Handle> {
    syscall(n::ACCOUNT_CREATE, [system.0 as u64, limit, 0, 0, 0, 0]).map(|h| Handle(h as u32))
}

/// Derive a backing budget. Parent 0 means this process's own default account.
pub fn account_child(parent: Handle, limit: u64) -> Result<Handle> {
    syscall(n::ACCOUNT_CHILD, [parent.0 as u64, limit, 0, 0, 0, 0]).map(|h| Handle(h as u32))
}

pub fn account_info(account: Handle) -> Result<[u64; 3]> {
    let mut info = [0u64; 3];
    syscall(n::ACCOUNT_INFO, [account.0 as u64, info.as_mut_ptr() as u64, 0, 0, 0, 0])?;
    Ok(info)
}

pub fn mem_create_in(account: Handle, size: u64, flags: u64) -> Result<Handle> {
    syscall(n::MEM_CREATE_IN, [account.0 as u64, size, flags, 0, 0, 0]).map(|h| Handle(h as u32))
}

/// A thread whose user stack is allocated and reclaimed by the kernel.
pub fn thread_start(entry: u64, arg: u64, cpu: u64, stack_bytes: u64) -> Result<Handle> {
    syscall(n::THREAD_START, [entry, arg, cpu, stack_bytes, 0, 0]).map(|h| Handle(h as u32))
}
pub fn thread_wait(thread: Handle, deadline_ns: u64) -> Result<()> {
    syscall(n::THREAD_WAIT, [thread.0 as u64, deadline_ns, 0, 0, 0, 0]).map(|_| ())
}

pub fn timer_create() -> Result<Handle> {
    syscall(n::TIMER_CREATE, [0; 6]).map(|h| Handle(h as u32))
}
/// Arm/reset a monotonic absolute deadline. FOREVER disarms and clears EXPIRED.
pub fn timer_arm(timer: Handle, deadline_ns: u64) -> Result<()> {
    syscall(n::TIMER_ARM, [timer.0 as u64, deadline_ns, 0, 0, 0, 0]).map(|_| ())
}
pub fn timer_cancel(timer: Handle) -> Result<()> {
    syscall(n::TIMER_CANCEL, [timer.0 as u64, 0, 0, 0, 0, 0]).map(|_| ())
}

pub fn event_queue_create(capacity: usize) -> Result<Handle> {
    syscall(n::EVENT_QUEUE_CREATE, [capacity as u64, 0, 0, 0, 0, 0]).map(|h| Handle(h as u32))
}
pub fn object_subscribe(queue: Handle, source: Handle, signals: u64, key: u64) -> Result<u64> {
    syscall(n::OBJECT_SUBSCRIBE, [queue.0 as u64, source.0 as u64, signals, key, 0, 0])
}
pub fn subscription_rearm(queue: Handle, subscription: u64) -> Result<()> {
    syscall(n::SUBSCRIPTION_REARM, [queue.0 as u64, subscription, 0, 0, 0, 0]).map(|_| ())
}
pub fn subscription_cancel(queue: Handle, subscription: u64) -> Result<()> {
    syscall(n::SUBSCRIPTION_CANCEL, [queue.0 as u64, subscription, 0, 0, 0, 0]).map(|_| ())
}
/// Arms `rearm` again and takes every event waiting into `out`, at least one or TimedOut. `rearm` may not be longer than `out`.
pub fn event_wait(queue: Handle, rearm: &[u64], out: &mut [dross_abi::event::Event], deadline_ns: u64) -> Result<usize> {
    syscall(n::EVENT_WAIT, [queue.0 as u64, rearm.as_ptr() as u64, rearm.len() as u64, out.as_mut_ptr() as u64, out.len() as u64, deadline_ns]).map(|count| count as usize)
}
pub fn event_post(queue: Handle, key: u64, value: u64) -> Result<()> {
    syscall(n::EVENT_POST, [queue.0 as u64, key, value, 0, 0, 0]).map(|_| ())
}
/// How many messages the other end has read, ever, wrapping, and how many of this end's wait for it.
pub fn channel_progress(channel: Handle) -> Result<(u32, u32)> {
    syscall(n::CHANNEL_PROGRESS, [channel.0 as u64, 0, 0, 0, 0, 0]).map(|v| ((v >> 32) as u32, v as u32))
}
pub fn channel_signal_peer(channel: Handle, mask: u64, bits: u64) -> Result<()> {
    syscall(n::CHANNEL_SIGNAL_PEER, [channel.0 as u64, mask, bits, 0, 0, 0]).map(|_| ())
}
pub fn event_queue_wait(queue: Handle, deadline_ns: u64) -> Result<dross_abi::event::Event> {
    let mut event = dross_abi::event::Event::default();
    syscall(n::EVENT_QUEUE_WAIT, [queue.0 as u64, &mut event as *mut _ as u64, deadline_ns, 0, 0, 0])?;
    Ok(event)
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

/// The memory object mapped at `addr`, the address's offset in it, and how many bytes of the mapping follow.
pub fn vm_memory(addr: u64) -> Result<(Handle, u64, u64)> {
    let mut out = [0u64; 2];
    let h = syscall(n::VM_MEMORY, [addr, out.as_mut_ptr() as u64, 0, 0, 0, 0])?;
    Ok((Handle(h as u32), out[0], out[1]))
}

/// Starts the ELF image in `image` as a new process with `bootstrap` (moved) as its handle 1.
pub fn spawn(image: Handle, name: &str, bootstrap: Option<Handle>, cpu: u64) -> Result<Handle> {
    syscall(n::SPAWN, [image.0 as u64, name.as_ptr() as u64, name.len() as u64, bootstrap.map_or(0, |h| h.0 as u64), cpu, 0]).map(|h| Handle(h as u32))
}

/// Select an existing account as the child's shared/default funding account.
pub fn spawn_in(account: Handle, image: Handle, name: &str, bootstrap: Option<Handle>, cpu: u64) -> Result<Handle> {
    syscall(n::SPAWN, [image.0 as u64, name.as_ptr() as u64, name.len() as u64, bootstrap.map_or(0, |h| h.0 as u64), cpu, account.0 as u64]).map(|h| Handle(h as u32))
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

/// Read a data-message prefix with a cursor shared by all endpoint readers.
/// Handle-bearing messages are rejected without consumption. Empty messages
/// return zero; PeerClosed means that all queued bytes have been consumed.
pub fn channel_read_bytes(ch: Handle, bytes: &mut [u8], deadline: u64) -> Result<usize> {
    syscall(n::CHANNEL_READ_BYTES, [ch.0 as u64, bytes.as_mut_ptr() as u64, bytes.len() as u64, deadline, 0, 0]).map(|n| n as usize)
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

/// Sleeps while `word` holds `expected`, until `unpark` or the deadline. ShouldWait if it doesn't hold it.
/// A wake may have no matching change, so check the word in a loop.
pub fn park(word: &AtomicU32, expected: u32, deadline: u64) -> Result<()> {
    park_address(word.as_ptr() as u64, expected, deadline)
}

/// `park` by address. The kernel checks the address.
pub fn park_address(address: u64, expected: u32, deadline: u64) -> Result<()> {
    syscall(n::PARK, [address, expected as u64, deadline, 0, 0, 0]).map(|_| ())
}

/// Wakes up to `count` threads parked on `word`, oldest first. Returns how many.
pub fn unpark(word: &AtomicU32, count: u64) -> Result<u64> {
    syscall(n::UNPARK, [word.as_ptr() as u64, count, 0, 0, 0, 0])
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

pub fn spawn_linux_in(account: Handle, image: Handle, argv: &[&str], envp: &[&str], bootstrap: Option<Handle>) -> Result<Handle> {
    let join = |v: &[&str]| v.iter().flat_map(|s| s.bytes().chain(core::iter::once(0))).collect::<alloc::vec::Vec<u8>>();
    let (argv, envp) = (join(argv), join(envp));
    let args = dross_abi::LinuxSpawnArgs { image: image.0 as u64, argv: argv.as_ptr() as u64, argv_bytes: argv.len() as u64,
        envp: envp.as_ptr() as u64, envp_bytes: envp.len() as u64, bootstrap: bootstrap.map_or(0, |h| h.0 as u64) };
    syscall(n::SPAWN_LINUX_IN, [account.0 as u64, &args as *const _ as u64, 0, 0, 0, 0]).map(|h| Handle(h as u32))
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
    clock().saturating_add(ms.saturating_mul(1_000_000))
}

/// Secure kernel entropy; ShouldWait until fresh boot entropy is available.
/// No fallback to a clock, counter or fixed seed.
pub fn random(bytes: &mut [u8]) -> Result<()> {
    syscall(n::RANDOM, [bytes.as_mut_ptr() as u64, bytes.len() as u64, 0, 0, 0, 0]).map(|_| ())
}

/// Registers `service` as the kernel's network service (netd's end gets the sockets of Linux programs). Needs the
/// system resource; kernels built with the Unix feature only.
pub fn unix_mount(system: Handle, service: Handle, path: &str) -> Result<()> {
    syscall(n::UNIX_MOUNT, [system.0 as u64, service.0 as u64, path.as_ptr() as u64, path.len() as u64, 0, 0]).map(|_| ())
}

/// The memory of a file in the Unix tree, to start as a program. Needs the system resource; Unix kernels only.
pub fn unix_image(system: Handle, path: &str) -> Result<Handle> {
    syscall(n::UNIX_IMAGE, [system.0 as u64, path.as_ptr() as u64, path.len() as u64, 0, 0, 0]).map(|h| Handle(h as u32))
}

/// Memory with no frames yet: a fault on it asks on `pager` (`u64 id, u64 cluster`), for `mem_supply` to answer.
pub fn mem_create_paged(size: u64, pager: Handle, id: u64) -> Result<Handle> {
    syscall(n::MEM_CREATE_PAGED, [size, pager.0 as u64, id, 0, 0, 0]).map(|h| Handle(h as u32))
}

/// Gives paged memory its pages at `offset`, from `source`.
pub fn mem_supply(memory: Handle, offset: u64, source: &[u8]) -> Result<()> {
    syscall(n::MEM_SUPPLY, [memory.0 as u64, offset, source.as_ptr() as u64, source.len() as u64, 0, 0]).map(|_| ())
}

/// Tells those waiting on the cluster at `offset` that it can't be had.
pub fn mem_supply_failed(memory: Handle, offset: u64) -> Result<()> {
    syscall(n::MEM_SUPPLY, [memory.0 as u64, offset, 0, 0, 0, 0]).map(|_| ())
}

pub fn unix_net_register(system: Handle, service: Handle) -> Result<()> {
    syscall(n::UNIX_NET_REGISTER, [system.0 as u64, service.0 as u64, 0, 0, 0, 0]).map(|_| ())
}

/// Start a Linux process with a userspace syscall server. The endpoint is retained,
/// bootstrap is moved, and descendants inherit the server through fork and exec.
pub fn spawn_linux_served(endpoint: Handle, image: Handle, argv: &[&str], envp: &[&str], bootstrap: Option<Handle>, account: Option<Handle>) -> Result<Handle> {
    let join = |v: &[&str]| v.iter().flat_map(|s| s.bytes().chain(core::iter::once(0))).collect::<alloc::vec::Vec<u8>>();
    let (argv, envp) = (join(argv), join(envp));
    let args = dross_abi::LinuxSpawnArgs { image: image.0 as u64, argv: argv.as_ptr() as u64, argv_bytes: argv.len() as u64,
        envp: envp.as_ptr() as u64, envp_bytes: envp.len() as u64, bootstrap: bootstrap.map_or(0, |h| h.0 as u64) };
    syscall(n::SPAWN_LINUX_SERVED, [endpoint.0 as u64, &args as *const _ as u64, account.map_or(0, |h| h.0 as u64), 0, 0, 0]).map(|h| Handle(h as u32))
}

/// Construct an address space; executable loading belongs to its supervisor.
pub fn process_create(account: Option<Handle>, endpoint: Option<Handle>, name: &str, bootstrap: Option<Handle>) -> Result<Handle> {
    syscall(n::PROCESS_CREATE, [account.map_or(0,|h|h.0 as u64), endpoint.map_or(0,|h|h.0 as u64), name.as_ptr() as u64, name.len() as u64, bootstrap.map_or(0,|h|h.0 as u64), 0]).map(|h|Handle(h as u32))
}
pub fn process_map(process: Handle, memory: Handle, mapping: dross_abi::task::Map) -> Result<u64> {
    syscall(n::PROCESS_MAP, [process.0 as u64, memory.0 as u64, &mapping as *const _ as u64, 0, 0, 0])
}
pub fn process_start(process: Handle, entry: u64, stack: u64, tls: u64, cpu: u64) -> Result<Handle> {
    syscall(n::PROCESS_START, [process.0 as u64, entry, stack, tls, cpu, 0]).map(|h|Handle(h as u32))
}

pub fn thread_self() -> u64 { syscall(n::THREAD_SELF, [0; 6]).expect("native thread identity") }
pub fn process_terminate(process: Handle, code: i64) -> Result<()> { syscall(n::PROCESS_TERMINATE, [process.0 as u64, code as u64, 0, 0, 0, 0]).map(|_|()) }
pub fn thread_terminate(thread: Handle) -> Result<()> { syscall(n::THREAD_TERMINATE, [thread.0 as u64, 0, 0, 0, 0, 0]).map(|_|()) }
pub fn process_info(process: Handle) -> Result<dross_abi::task::Info> {
    let mut info = dross_abi::task::Info::default(); syscall(n::PROCESS_INFO, [process.0 as u64, &mut info as *mut _ as u64, 0, 0, 0, 0])?; Ok(info)
}
pub fn process_self_id() -> u32 {
    syscall(n::PROCESS_SELF_ID, [0; 6]).unwrap_or(0) as u32
}
pub fn process_vm(process: Handle, operation: u64, address: u64, length: u64, flags: u64) -> Result<u64> { syscall(n::PROCESS_VM, [process.0 as u64, operation, address, length, flags, 0]) }
pub fn process_start_context(process: Handle, context: &dross_abi::task::Context, cpu: u64) -> Result<Handle> {
    syscall(n::PROCESS_START_CONTEXT, [process.0 as u64, context as *const _ as u64, cpu, 0, 0, 0]).map(|h|Handle(h as u32))
}

pub fn process_copy_space(source: Handle, target: Handle, shared: bool) -> Result<()> { syscall(n::PROCESS_COPY_SPACE, [source.0 as u64, target.0 as u64, shared as u64, 0, 0, 0]).map(|_|()) }
pub fn process_start_snapshot(process: Handle, snapshot: Handle, context: &dross_abi::task::Context) -> Result<Handle> { syscall(n::PROCESS_START_SNAPSHOT, [process.0 as u64, snapshot.0 as u64, context as *const _ as u64, 0, 0, 0]).map(|h|Handle(h as u32)) }
pub fn thread_interrupt(thread: Handle) -> Result<()> { syscall(n::THREAD_INTERRUPT, [thread.0 as u64, 0, 0, 0, 0, 0]).map(|_|()) }

/// Low-level native escape for system services implementing typed capability protocols.
#[doc(hidden)]
pub fn raw(number: u64, arguments: [u64; 6]) -> Result<u64> { syscall(number, arguments) }
#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn syscall(nr: u64, a: [u64; 6]) -> Result<u64> {
    let status: i64; let value: u64;
    unsafe { core::arch::asm!("ecall", in("a7") nr | dross_abi::NATIVE,
        inlateout("a0") a[0] => status, inlateout("a1") a[1] => value,
        in("a2") a[2], in("a3") a[3], in("a4") a[4], in("a5") a[5], options(nostack)); }
    if status == 0 { Ok(value) } else { Err(Error::from_status(status).unwrap_or(Error::BadSyscall)) }
}

#[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
#[inline(always)]
fn syscall(nr: u64, a: [u64; 6]) -> Result<u64> {
    let status: i64; let value: u64;
    unsafe { core::arch::asm!("svc #0", in("x8") nr | dross_abi::NATIVE,
        inlateout("x0") a[0] => status, inlateout("x1") a[1] => value,
        in("x2") a[2], in("x3") a[3], in("x4") a[4], in("x5") a[5], options(nostack)); }
    if status == 0 { Ok(value) } else { Err(Error::from_status(status).unwrap_or(Error::BadSyscall)) }
}

/// Bind a device's wired interrupt capability to a notification.
pub fn irq_bind(resource:Handle,source:u32,notification:Handle,bits:u64,cpu:u32)->Result<Handle> {
    syscall(n::IRQ_BIND,[resource.0 as u64,source as u64,notification.0 as u64,bits,cpu as u64,0]).map(|h|Handle(h as u32))
}
pub fn irq_ack(irq:Handle)->Result<()> {syscall(n::IRQ_ACK,[irq.0 as u64,0,0,0,0,0]).map(|_|())}

/// Explicit coherent physical DMA for a trusted bus master without an IOMMU.
/// The kernel retains backing until dma_buffer_reclaim, even if all handles
/// close or this process exits. The driver must quiesce before reclamation.
pub fn dma_buffer_create(resource:Handle,size:u64)->Result<(Handle,u64)> {
    let mut physical=0u64;
    let memory=syscall(n::DMA_BUFFER,[resource.0 as u64,dross_abi::dma::physical::ALLOCATE,size,&mut physical as *mut _ as u64,0,0])?;
    Ok((Handle(memory as u32),physical))
}
pub fn dma_buffer_reclaim(resource:Handle,memory:Handle)->Result<()> {
    syscall(n::DMA_BUFFER,[resource.0 as u64,dross_abi::dma::physical::RECLAIM,memory.0 as u64,0,0,0]).map(|_|())
}

/// Copy immutable target-native extended state; this does not delegate write authority.
pub fn execution_read(snapshot: Handle, offset: u64, bytes: &mut [u8]) -> Result<()> {
    syscall(n::EXECUTION_READ, [snapshot.0 as u64,offset,bytes.as_mut_ptr() as u64,bytes.len() as u64,0,0]).map(|_|())
}
pub fn process_icache(process: Handle, start: u64, end: u64) -> Result<()> {
    syscall(n::PROCESS_ICACHE, [process.0 as u64,start,end,0,0,0]).map(|_|())
}

pub fn process_barrier(process: Handle) -> Result<()> {
    syscall(n::PROCESS_BARRIER,[process.0 as u64,0,0,0,0,0]).map(|_|())
}

/// Finish a forwarded call with `value` if there is one, then wait for the next message. Bytes, handles, and
/// whether the kernel's forwarding path wrote it.
pub fn personality_next(service: Handle, finish: Option<(Handle, u64)>, bytes: &mut [u8], handles: &mut [Handle]) -> Result<(usize, usize, bool)> {
    let (reply, value) = finish.map_or((0, 0), |(reply, value)| (reply.0 as u64, value));
    let result = syscall(n::PERSONALITY_NEXT, [service.0 as u64, reply, value, bytes.as_mut_ptr() as u64, bytes.len() as u64 | (handles.len() as u64) << 32, handles.as_mut_ptr() as u64])?;
    Ok((result as u32 as usize, ((result >> 32) & 0xffff) as usize, result & dross_abi::personality::NEXT_FORWARDED != 0))
}
