//! Generic process construction; no executable format or Unix state on this interface.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Map {
    pub offset: u64,
    pub length: u64,
    pub address: u64,
    pub flags: u64,
}

pub const UNMAP: u64 = 0;
pub const PROTECT: u64 = 1;
pub const RESERVE: u64 = 2;
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct Info {
    pub pid: u64,
    pub threads: u64,
    pub cpu_ns: u64,
    pub started_ns: u64,
    pub virtual_bytes: u64,
    pub mapped_bytes: u64,
}
/// Register words in the target architecture's native interrupt frame.
#[cfg(any(target_arch = "riscv64", target_arch = "aarch64"))]
pub const REGISTER_WORDS: usize = 35;
#[cfg(not(any(target_arch = "riscv64", target_arch = "aarch64")))]
pub const REGISTER_WORDS: usize = 22;
pub const CONTEXT_WORDS: usize = REGISTER_WORDS + 1;
pub const CONTEXT_BYTES: usize = CONTEXT_WORDS * 8;
/// Target-native registers followed by TLS. The kernel sanitizes privileged state.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Context {
    pub registers: [u64; REGISTER_WORDS],
    pub tls: u64,
}
impl Default for Context {
    fn default() -> Self {
        Self {
            registers: [0; REGISTER_WORDS],
            tls: 0,
        }
    }
}

/// User-word stores in the target address space, applied before a new thread
/// is published or made runnable. Flag bits 0 and 1 enable the respective TID stores.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct StartupActions {
    pub exit_word: u64,
    pub parent_tid: u64,
    pub child_tid: u64,
    pub flags: u64,
}

/// Read-only native task metadata. Enumeration grants no task capabilities.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Snapshot {
    pub pid: u64,
    pub threads: u64,
    pub cpu_ns: u64,
    pub started_ns: u64,
    pub virtual_bytes: u64,
    pub mapped_bytes: u64,
    pub runnable: u64,
    pub name: [u8; 32],
}
