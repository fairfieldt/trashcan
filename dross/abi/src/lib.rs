//! The system call interface of dross, shared by the kernel and the runtime.
//!
//! `syscall` with the number in `rax` and up to six arguments in `rdi, rsi, rdx, r10, r8, r9`. The kernel returns
//! a status in `rax` (0 or a negative `Error`) and a value in `rdx`. Kernel objects are named by handles: small
//! integers private to a process, each carrying rights. Handle 0 is never valid.
#![no_std]

pub mod sys {
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
    pub const MEM_SIZE: u64 = 28;
    pub const PROCESS_WAIT: u64 = 29;
    /// (system resource, code): ends a QEMU run (0: success); halts on hardware for now.
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
}

/// Set in `rax` for a native dross call from a Linux process (whose plain numbers are Linux's).
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
    pub const READ: u64 = 1 << 0;
    pub const WRITE: u64 = 1 << 1;
    pub const EXECUTE: u64 = 1 << 2;
    /// `MEM_CREATE`: physically contiguous.
    pub const CONTIGUOUS: u64 = 1 << 8;
    /// `MEM_CREATE`: below 4 GiB.
    pub const BELOW_4G: u64 = 1 << 9;
}

/// Where user space ends (exclusive).
pub const USER_TOP: u64 = 0x0000_8000_0000_0000;

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
}

/// Bounded registrations per queue, active observers per object, and registrations per source handle.
pub const EVENT_LIMIT: usize = 256;
pub mod event {
    pub const READABLE: u64 = 1 << 0;
    pub const WRITABLE: u64 = 1 << 1;
    pub const PEER_CLOSED: u64 = 1 << 2;
    pub const PENDING: u64 = 1 << 3;
    pub const EXITED: u64 = 1 << 4;
    pub const EXPIRED: u64 = 1 << 5;
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
