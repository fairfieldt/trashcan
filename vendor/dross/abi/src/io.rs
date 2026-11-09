//! Native ordering for device registers and coherent DMA buffers. Kept with
//! the shared interface so libraries need not select a program runtime.
//! These barriers do not provide cache maintenance for non-coherent devices.

/// Order device-register accesses against each other and normal memory.
/// x86 retains its existing volatile/uncached MMIO path. FU740 needs explicit
/// I/O-domain fences; volatile alone supplies no such hardware ordering.
#[inline(always)]
pub fn device_fence() {
    #[cfg(target_arch = "riscv64")]
    unsafe {
        core::arch::asm!("fence iorw, iorw", options(nostack));
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dsb sy", options(nostack));
    }
}

/// Publish coherent descriptor contents before their ownership/cycle store.
#[inline(always)]
pub fn dma_release() {
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Release);
    #[cfg(target_arch = "riscv64")]
    unsafe {
        core::arch::asm!("fence rw, w", options(nostack));
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dmb oshst", options(nostack));
    }
}

/// Observe coherent descriptor contents after its ownership/cycle load.
#[inline(always)]
pub fn dma_acquire() {
    #[cfg(target_arch = "riscv64")]
    unsafe {
        core::arch::asm!("fence r, rw", options(nostack));
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("dmb oshld", options(nostack));
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::Acquire);
}

/// A 32-bit device register load. On AArch64 this is one plain load, the only form a hypervisor can decode when it
/// emulates the access; a volatile read may compile to one with write-back.
#[inline(always)]
pub fn read32(address: u64) -> u32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let value: u32;
        core::arch::asm!("ldr {v:w}, [{a}]", v = out(reg) value, a = in(reg) address, options(nostack));
        value
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        (address as *const u32).read_volatile()
    }
}

/// A 32-bit device register store. See `read32`.
#[inline(always)]
pub fn write32(address: u64, value: u32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("str {v:w}, [{a}]", v = in(reg) value, a = in(reg) address, options(nostack));
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        (address as *mut u32).write_volatile(value)
    }
}

/// An 8-bit device register load. See `read32`.
#[inline(always)]
pub fn read8(address: u64) -> u8 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let value: u32;
        core::arch::asm!("ldrb {v:w}, [{a}]", v = out(reg) value, a = in(reg) address, options(nostack));
        value as u8
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        (address as *const u8).read_volatile()
    }
}

/// An 8-bit device register store. See `read32`.
#[inline(always)]
pub fn write8(address: u64, value: u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("strb {v:w}, [{a}]", v = in(reg) value as u32, a = in(reg) address, options(nostack));
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        (address as *mut u8).write_volatile(value)
    }
}

/// A 64-bit device register load. See `read32`.
#[inline(always)]
pub fn read64(address: u64) -> u64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let value: u64;
        core::arch::asm!("ldr {v}, [{a}]", v = out(reg) value, a = in(reg) address, options(nostack));
        value
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        (address as *const u64).read_volatile()
    }
}

/// A 64-bit device register store. See `read32`.
#[inline(always)]
pub fn write64(address: u64, value: u64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("str {v}, [{a}]", v = in(reg) value, a = in(reg) address, options(nostack));
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        (address as *mut u64).write_volatile(value)
    }
}
