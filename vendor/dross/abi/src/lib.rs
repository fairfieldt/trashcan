//! The system call interface of dross, shared by the kernel and the runtime.
//!
//! `syscall` with the number in `rax` and up to six arguments in `rdi, rsi, rdx, r10, r8, r9`. The kernel returns
//! a status in `rax` (0 or a negative `Error`) and a value in `rdx`. Kernel objects are named by handles: small
//! integers private to a process, each carrying rights. Handle 0 is never valid.
#![no_std]
pub mod personality;
pub mod pm;
pub mod task;
pub mod io;

/// How much paged memory comes at a time (MEM_CREATE_PAGED), counted from its start.
pub const PAGER_CLUSTER: u64 = 1 << 20;

pub mod sys {
    /// Read up to MSG_BYTES bytes from a channel, preserving an unread message
    /// tail for every reader. Rejects messages carrying handles. A zero-length
    /// message returns zero; PeerClosed is reported only after queued data.
    pub const CHANNEL_READ_BYTES: u64 = 113;
    /// The caller's process ID; grants no handle authority.
    pub const PROCESS_SELF_ID: u64 = 114;
    /// Read bytes of an immutable target-native execution snapshot.
    pub const EXECUTION_READ: u64 = 105;
    /// Synchronize instruction caches for an executable user address range.
    pub const PROCESS_ICACHE: u64 = 106;
    /// Complete a full memory fence on every application CPU.
    pub const PROCESS_BARRIER: u64 = 107;
    /// (reply channel WRITE, caller address, &mut bytes, length): copy from the caller stopped behind a forwarded
    /// call's reply channel, without waking it. WrongType asks for the copy through the caller instead.
    pub const PERSONALITY_READ: u64 = 108;
    /// (reply channel WRITE, caller address, &bytes, length): as PERSONALITY_READ, to the caller.
    pub const PERSONALITY_WRITE: u64 = 109;
    /// (service READ, reply channel WRITE or 0, value, &mut bytes, max bytes | max handles << 32, &mut handles): finish
    /// a forwarded call with a value and close its channel, then take the next message; that accepts a forwarded call.
    /// Returns bytes | handles << 32 | personality::NEXT_FORWARDED.
    pub const PERSONALITY_NEXT: u64 = 110;
    /// (System READ, after identity, &mut task::Snapshot): next live task identity, or 0.
    pub const TASK_NEXT: u64 = 101;
    pub const PROCESS_SUPERVISOR: u64 = 100;
    pub const CHARGE_RELEASE: u64 = 99;
    pub const SYSTEM_CPU_TIME: u64 = 98;
    pub const PROCESS_STOP_OTHERS: u64 = 95;
    pub const MEM_READ: u64 = 96;
    pub const MEM_WRITE: u64 = 97;
    pub const SYSTEM_RESET: u64 = 94;
    pub const OBJECT_INFO: u64 = 77;
    pub const ACCOUNT_RESERVE: u64 = 78;
    pub const CHARGE_SPLIT: u64 = 79;
    pub const MEM_SLICE: u64 = 80;
    pub const MEM_COMPOSE: u64 = 81;
    pub const PROCESS_MAP_EXT: u64 = 82;
    pub const PROCESS_LOOKUP: u64 = 83;
    pub const PROCESS_READ: u64 = 84;
    pub const PROCESS_WRITE: u64 = 85;
    pub const PROCESS_HANDLE_INSERT: u64 = 86;
    pub const THREAD_INFO: u64 = 87;
    pub const PROCESS_PARK: u64 = 88;
    pub const PROCESS_WAKE: u64 = 89;
    pub const LOG_READ: u64 = 90;
    pub const SYSTEM_INFO: u64 = 91;
    pub const THREAD_SETTINGS: u64 = 92;
    pub const PERSONALITY_REGISTER: u64 = 93;

    /// (source process READ|WRITE|MAP, destination WRITE|MAP, mode 0 private/1 shared): clone mappings.
    pub const PROCESS_COPY_SPACE: u64 = 74;
    /// (process EXECUTE|WRITE, execution snapshot EXECUTE, &task::Context, &StartupActions or 0): start with saved FPU state.
    pub const PROCESS_START_SNAPSHOT: u64 = 75;
    /// (thread SIGNAL): request its supervisor's generic user-return upcall.
    pub const THREAD_INTERRUPT: u64 = 76;
    /// (process SIGNAL, exit code): terminate an owned process.
    pub const PROCESS_TERMINATE: u64 = 68;
    /// (process READ, &mut task::Info): stable identity and execution/memory counters.
    pub const PROCESS_INFO: u64 = 69;
    /// Current native thread identity, for runtimes managing their own call contexts.
    pub const THREAD_SELF: u64 = 70;
    /// (process MAP|WRITE, operation, address, length, flags): address-space controls.
    pub const PROCESS_VM: u64 = 71;
    /// (process EXECUTE|WRITE, &task::Context, cpu or MAX, &StartupActions or 0): start with explicit user registers.
    pub const PROCESS_START_CONTEXT: u64 = 72;
    /// (thread SIGNAL): terminate a controlled thread.
    pub const THREAD_TERMINATE: u64 = 73;
    /// (account WRITE or 0, personality channel RW or 0, name, length, bootstrap moved or 0): empty process.
    pub const PROCESS_CREATE: u64 = 65;
    /// (process MAP|WRITE, memory MAP, &task::Map): map memory into a controlled address space.
    pub const PROCESS_MAP: u64 = 66;
    /// (process EXECUTE|WRITE, entry, stack, TLS, cpu or MAX, &StartupActions or 0): start a thread, return its status capability.
    pub const PROCESS_START: u64 = 67;
    /// (endpoint READ|WRITE|TRANSFER, &LinuxSpawnArgs, account WRITE|TRANSFER or 0): served launch.
    /// The image must allow READ|MAP|TRANSFER; returned process rights are those granted by the server.
    pub const SPAWN_LINUX_SERVED: u64 = 64;
    pub const LOG: u64 = 0;
    pub const PROCESS_EXIT: u64 = 1;
    pub const THREAD_EXIT: u64 = 2;
    pub const THREAD_CREATE: u64 = 3;
    pub const YIELD: u64 = 4;
    pub const CLOCK: u64 = 5;
    pub const HANDLE_CLOSE: u64 = 6;
    pub const HANDLE_DUP: u64 = 7;
    pub const MEM_CREATE: u64 = 8;
    pub const MEM_PHYSICAL: u64 = 9;
    pub const VM_MAP: u64 = 10;
    pub const VM_UNMAP: u64 = 11;
    pub const SPAWN: u64 = 12;
    pub const CHANNEL_CREATE: u64 = 13;
    pub const CHANNEL_WRITE: u64 = 14;
    pub const CHANNEL_READ: u64 = 15;
    pub const CHANNEL_BIND: u64 = 16;
    pub const NOTIFY_CREATE: u64 = 17;
    pub const NOTIFY_SIGNAL: u64 = 18;
    pub const NOTIFY_WAIT: u64 = 19;
    pub const RESOURCE_SUB: u64 = 20;
    pub const IOPORT_ENABLE: u64 = 21;
    pub const IRQ_MSI: u64 = 22;
    pub const DMA_DOMAIN_CREATE: u64 = 23;
    pub const DMA_ATTACH: u64 = 24;
    pub const DMA_MAP: u64 = 25;
    pub const DMA_UNMAP: u64 = 26;
    pub const CPU_COUNT: u64 = 27;
    /// All the RAM there is, in bytes.
    pub const MEMORY_TOTAL: u64 = 111;
    pub const MEM_SIZE: u64 = 28;
    pub const PROCESS_WAIT: u64 = 29;
    /// (system resource, code): ends a QEMU run (0: success); halts on hardware for now. The Unix server's disks finish first.
    pub const SYSTEM_EXIT: u64 = 30;
    /// (image, argv, argv_len, envp, envp_len, bootstrap): starts a static Linux executable (argv and envp are
    /// NUL-separated strings); `bootstrap` (moved, 0 for none) becomes its native handle 1.
    pub const SPAWN_LINUX: u64 = 31;
    /// (handle) from a Linux process: an fd for the object. A notification's polls readable while bits are pending;
    /// a memory object's can be mapped with `mmap` (a BAR handed to a client, for example).
    /// With the Unix feature, a read/write channel becomes a duplex byte-stream fd.
    pub const HANDLE_FD: u64 = 32;
    /// Opt-in Unix IP service registration: (system resource, channel). Supervisor only.
    pub const UNIX_NET_REGISTER: u64 = 33;
    /// (buffer, length): cryptographically secure bytes, or ShouldWait before seeding.
    pub const RANDOM: u64 = 34;
    /// (system resource, channel, path, path length): a read-only filesystem the program at the channel's other
    /// end serves appears at the path for Linux programs (`unix` kernels; kernel/src/linux/unix/mount.rs).
    pub const UNIX_MOUNT: u64 = 35;
    /// (address, where to write two u64s): a handle to the memory object mapped at the address, with the rights
    /// the mapping has (write only if it is writable); written: the address's offset in it and how many bytes of
    /// the mapping follow the address (for DMA to memory a program mapped by address, such as a memfd).
    pub const VM_MEMORY: u64 = 36;
    /// (system resource, byte limit): new child of the caller's default budget; no RAM is reserved.
    pub const ACCOUNT_CREATE: u64 = 37;
    /// (account, output): writes charged bytes, peak charged bytes, byte limit (three u64s).
    /// Account 0 means the caller's default backing account.
    pub const ACCOUNT_INFO: u64 = 38;
    /// (account, size, flags): MEM_CREATE funded by an explicitly supplied account.
    pub const MEM_CREATE_IN: u64 = 39;
    pub const EVENT_QUEUE_CREATE: u64 = 40;
    pub const OBJECT_SUBSCRIBE: u64 = 41;
    pub const SUBSCRIPTION_REARM: u64 = 42;
    pub const SUBSCRIPTION_CANCEL: u64 = 43;
    pub const EVENT_QUEUE_WAIT: u64 = 44;
    pub const TIMER_CREATE: u64 = 45;
    pub const TIMER_ARM: u64 = 46;
    pub const TIMER_CANCEL: u64 = 47;
    pub const THREAD_START: u64 = 48;
    pub const THREAD_WAIT: u64 = 49;
    pub const ACCOUNT_CHILD: u64 = 50;
    pub const SPAWN_LINUX_IN: u64 = 51;
    /// (system resource): trace unreachable native channel queues; returns packets reclaimed.
    pub const CHANNEL_COLLECT: u64 = 52;
    /// (system resource with READ, output): five u64 memory counters; API.md.
    pub const SYSTEM_MEMORY_INFO: u64 = 53;
    /// (system resource, path, path length): a handle to the memory of the file at the path in the Unix tree, as a
    /// Linux program sees it, so that init can start a program from a root on disk (`unix` kernels).
    pub const UNIX_IMAGE: u64 = 54;
    /// (size, channel, id): memory with no frames yet. A fault on it writes `u64 id, u64 cluster` to the channel, for
    /// the holder of the other end to answer with MEM_SUPPLY, and waits (`unix` kernels).
    pub const MEM_CREATE_PAGED: u64 = 55;
    /// (memory, offset, source, length): gives paged memory those pages, whole clusters, waking whoever waits. Pages it
    /// already has keep their contents. With no source: that cluster can't be had, and those waiting fail.
    pub const MEM_SUPPLY: u64 = 56;
    /// (address of an aligned u32, expected value, deadline): sleeps while the word holds the value, until
    /// `UNPARK` or the deadline. ShouldWait if it doesn't hold the value. A wake may have no matching change.
    pub const PARK: u64 = 57;
    /// (address, count): wakes up to `count` threads parked on the word, oldest first; returns how many.
    pub const UNPARK: u64 = 58;
    /// Private system PM boundary: (system WRITE, &pm::Request, 64). See pm.
    pub const SYSTEM_SUSPEND: u64 = 59;
    // 60 was SYSTEM_FRAMEBUFFER_IO, removed.
    /// (queue, &[subscription], count, &mut [Event], capacity, deadline): arms each subscription again, then takes every
    /// event waiting, up to capacity and `EVENT_BATCH_LIMIT`; the count of them, or TimedOut. One that can't be armed again
    /// is reported as an event with `event::CANCELLED`, so the count may not exceed the capacity.
    pub const EVENT_WAIT: u64 = 61;
    /// (queue, key, value): an event from this program, as (subscription 0, key, value). It takes a registration's place
    /// until it is taken; NoResources if there is none free.
    pub const EVENT_POST: u64 = 62;
    /// (channel, mask, bits): sets `event::USER` bits on the other end, where it can see them with no message.
    /// A service publishes its state this way; PeerClosed if there's no one there.
    pub const CHANNEL_SIGNAL_PEER: u64 = 63;
    /// (channel): how many messages the other end has read, ever, in the high 32 bits, wrapping, and how many of
    /// this end's wait for it, in the low 32. A writer can then hold back what would only go stale in the queue.
    pub const CHANNEL_PROGRESS: u64 = 112;
    /// (IRQ resource, source, notification, bits, CPU): bind a platform wired interrupt.
    pub const IRQ_BIND: u64 = 102;
    /// (interrupt with WRITE): re-enable a masked wired interrupt after servicing it.
    pub const IRQ_ACK: u64 = 103;
    /// Trusted coherent physical DMA, distinct from an isolated IOMMU domain.
    /// (resource WRITE, op, size or memory, &physical for ALLOCATE).
    pub const DMA_BUFFER: u64 = 104;

}

/// Set in the syscall-number register for a native dross call from a Linux process (whose plain numbers are Linux's).
pub const NATIVE: u64 = 0x4000_0000;

/// Negative status codes.
#[repr(i64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    BadSyscall = -1,
    BadHandle = -2,
    WrongType = -3,
    AccessDenied = -4,
    InvalidArgs = -5,
    NoMemory = -6,
    OutOfRange = -7,
    ShouldWait = -8,
    TimedOut = -9,
    PeerClosed = -10,
    BufferTooSmall = -11,
    Busy = -12,
    BadElf = -13,
    NotFound = -14,
    NoResources = -15,
}

impl Error {
    pub fn from_status(s: i64) -> Option<Error> {
        use Error::*;
        Some(match s {
            -1 => BadSyscall,
            -2 => BadHandle,
            -3 => WrongType,
            -4 => AccessDenied,
            -5 => InvalidArgs,
            -6 => NoMemory,
            -7 => OutOfRange,
            -8 => ShouldWait,
            -9 => TimedOut,
            -10 => PeerClosed,
            -11 => BufferTooSmall,
            -12 => Busy,
            -13 => BadElf,
            -14 => NotFound,
            -15 => NoResources,
            _ => return None,
        })
    }
}

/// Rights a handle carries. `HANDLE_DUP` can only remove rights.
pub mod rights {
    pub const READ: u32 = 1 << 0;
    pub const WRITE: u32 = 1 << 1;
    pub const EXECUTE: u32 = 1 << 2;
    pub const MAP: u32 = 1 << 3;
    /// May be sent over a channel.
    pub const TRANSFER: u32 = 1 << 4;
    pub const DUPLICATE: u32 = 1 << 5;
    pub const SIGNAL: u32 = 1 << 6;
    pub const WAIT: u32 = 1 << 7;
    pub const ALL: u32 = 0xff;
}

/// `VM_MAP` and `MEM_CREATE` flags.
pub mod vm {
    /// Keep this backing shared when a supervisor clones address spaces.
    pub const INHERIT_SHARED: u64 = 1 << 16;
    pub const READ: u64 = 1 << 0;
    pub const WRITE: u64 = 1 << 1;
    pub const EXECUTE: u64 = 1 << 2;
    /// `MEM_CREATE`: physically contiguous.
    pub const CONTIGUOUS: u64 = 1 << 8;
    /// Range-owned resident anonymous backing, permitting partial-unmap reclamation.
    pub const PAGED: u64 = 1 << 10;
    /// `MEM_CREATE`: below 4 GiB.
    pub const BELOW_4G: u64 = 1 << 9;
}

/// Where user space ends (exclusive).
pub const USER_TOP: u64 = layout::USER_TOP;
pub mod layout;

/// Channel limits.
pub const MSG_BYTES: usize = 64 * 1024;
pub const MSG_HANDLES: usize = 64;
pub const CHANNEL_DEPTH: usize = 256;

/// The deadline meaning "forever" for waits.
pub const FOREVER: u64 = u64::MAX;

/// `DMA_MAP` flags.
pub mod dma {
    pub const READ: u64 = 1 << 0;
    pub const WRITE: u64 = 1 << 1;
    pub mod physical {
        pub const ALLOCATE: u64 = 0;
        /// Caller guarantees the controller can no longer access this backing.
        pub const RECLAIM: u64 = 1;
        pub const MAX_ALLOCATION: u64 = 16 << 20;
    }
}

/// Events one `EVENT_WAIT` takes and registrations it arms.
pub const EVENT_BATCH_LIMIT: usize = 64;
/// Registrations per queue, charged to whoever creates it. A service has one for each of its clients.
pub const EVENT_QUEUE_LIMIT: usize = 4096;
/// Bounded active observers per object, and registrations per source handle.
pub const EVENT_LIMIT: usize = 256;
pub mod event {
    pub const READABLE: u64 = 1 << 0;
    pub const WRITABLE: u64 = 1 << 1;
    pub const PEER_CLOSED: u64 = 1 << 2;
    pub const PENDING: u64 = 1 << 3;
    pub const EXITED: u64 = 1 << 4;
    pub const EXPIRED: u64 = 1 << 5;
    /// Bits the other end of a channel sets with `CHANNEL_SIGNAL_PEER`. Their meaning is the protocol's.
    pub const USER: u64 = 0xff << 8;
    /// `EVENT_WAIT` couldn't arm this subscription again: its handle has closed.
    pub const CANCELLED: u64 = 1 << 62;
    /// What a network service publishes about a socket, in `USER`.
    pub const SOCKET_READABLE: u64 = 1 << 8;
    pub const SOCKET_WRITABLE: u64 = 1 << 9;
    pub const SOCKET_ERROR: u64 = 1 << 10;
    pub const SOCKET_HANGUP: u64 = 1 << 11;
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Event {
        pub subscription: u64,
        pub key: u64,
        pub observed: u64,
    }
}

/// Timer objects reserved on each CPU. Expiry is processed in bounded interrupt batches.
pub const TIMER_LIMIT: usize = 256;

/// SPAWN_LINUX_IN(account, &args): selected default/shared funding for a Linux image.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LinuxSpawnArgs {
    pub image: u64,
    pub argv: u64,
    pub argv_bytes: u64,
    pub envp: u64,
    pub envp_bytes: u64,
    pub bootstrap: u64,
}
