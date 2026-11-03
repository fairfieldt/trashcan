//! Ends a QEMU run with a result (`isa-debug-exit` at 0xf4: exit status 33 for success, 35 for failure).
//! Only under a hypervisor; on hardware this halts.

use x86_64::instructions::port::Port;

pub fn exit(success: bool) -> ! {
    if crate::cpu::Features::detect().hypervisor {
        // SAFETY: QEMU's debug-exit port; real hardware never gets here.
        unsafe { Port::<u32>::new(0xf4).write(if success { 0x10 } else { 0x11 }) };
    }
    crate::cpu::halt_forever()
}
