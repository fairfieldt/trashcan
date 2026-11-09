//! Native virtual address layout. Board MMIO addresses are not user addresses.
#[cfg(target_arch = "riscv64")]
pub const DIRECT_MAP: u64 = 0xffff_ffc0_0000_0000;
#[cfg(not(target_arch = "riscv64"))]
pub const DIRECT_MAP: u64 = 0xffff_8000_0000_0000;
// RV64 currently retains the supervisor-only identity mapping of its linked
// image above 2 GiB. Keep native mappings below it on every Sv39 process root.
#[cfg(target_arch = "riscv64")]
pub const USER_TOP: u64 = 0x8000_0000;
#[cfg(not(target_arch = "riscv64"))]
pub const USER_TOP: u64 = 0x0000_8000_0000_0000;
pub const STACK_TOP: u64 = USER_TOP - 0x10000;
#[cfg(target_arch = "riscv64")]
pub const AUTO_BASE: u64 = 0x1000_0000;
#[cfg(not(target_arch = "riscv64"))]
pub const AUTO_BASE: u64 = 0x0000_1000_0000_0000;
