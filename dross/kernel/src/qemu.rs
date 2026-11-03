//! Ending a run: under QEMU, `isa-debug-exit` at 0xf4 (exit status 33 for success, 35 for failure); on
//! hardware, a platform reset through the FADT's reset register after a second (so ethd's network log can send
//! the last lines), which on a test machine returns to Linux (dross is booted once with `grub-reboot`).

use core::sync::atomic::{AtomicU64, Ordering};
use x86_64::instructions::port::Port;

/// The FADT reset register if it is an I/O port: port << 8 | value.
static RESET: AtomicU64 = AtomicU64::new(0);

pub fn set_reset(port: u16, value: u8) {
    RESET.store((port as u64) << 8 | value as u64, Ordering::Relaxed);
}

pub fn exit(success: bool) -> ! {
    reset(if success { "done" } else { "failed" }, success, 1000)
}

/// Resets the machine after `delay_ms` (or ends QEMU with `success`).
pub fn reset(why: &str, success: bool, delay_ms: u64) -> ! {
    if crate::cpu::Features::detect().hypervisor {
        // SAFETY: QEMU's debug-exit port; real hardware never gets here.
        unsafe { Port::<u32>::new(0xf4).write(if success { 0x10 } else { 0x11 }) };
    }
    let r = RESET.load(Ordering::Relaxed);
    if r != 0 {
        crate::log::emergency(format_args!("resetting in {delay_ms} ms ({why})"));
        x86_64::instructions::interrupts::enable();
        crate::time::delay_us(delay_ms * 1000);
        x86_64::instructions::interrupts::disable();
        // SAFETY: the reset register the firmware describes; this ends everything.
        unsafe { Port::<u8>::new((r >> 8) as u16).write(r as u8) };
    }
    crate::cpu::halt_forever()
}
