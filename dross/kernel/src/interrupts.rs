//! Interrupt and exception entry. The stubs (build.rs) push an error code (zero when the CPU gives none) and
//! the vector; `isr_common` saves the registers and calls `isr_dispatch` with them.

use core::arch::global_asm;

use crate::{apic, cpu, println, smp};

global_asm!(include_str!(concat!(env!("OUT_DIR"), "/isr.s")));
global_asm!(
    r#"
isr_common:
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    mov rdi, rsp
    cld
    call isr_dispatch
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rbp
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    add rsp, 16
    iretq
"#
);

pub const TIMER: u8 = 0x30;
pub const IPI_TEST: u8 = 0xf0;
pub const APIC_ERROR: u8 = 0xfe;
pub const SPURIOUS: u8 = 0xff;

#[repr(C)]
pub struct Frame {
    pub r15: u64,
    pub r14: u64,
    pub r13: u64,
    pub r12: u64,
    pub r11: u64,
    pub r10: u64,
    pub r9: u64,
    pub r8: u64,
    pub rbp: u64,
    pub rdi: u64,
    pub rsi: u64,
    pub rdx: u64,
    pub rcx: u64,
    pub rbx: u64,
    pub rax: u64,
    pub vector: u64,
    pub error: u64,
    pub rip: u64,
    pub cs: u64,
    pub rflags: u64,
    pub rsp: u64,
    pub ss: u64,
}

const EXCEPTIONS: [&str; 32] = [
    "divide error", "debug", "NMI", "breakpoint", "overflow", "bound range", "invalid opcode", "device not available",
    "double fault", "coprocessor segment overrun", "invalid TSS", "segment not present", "stack fault",
    "general protection", "page fault", "reserved", "x87 floating point", "alignment check", "machine check",
    "SIMD floating point", "virtualization", "control protection", "reserved", "reserved", "reserved", "reserved",
    "reserved", "reserved", "hypervisor injection", "VMM communication", "security", "reserved",
];

#[unsafe(no_mangle)]
extern "C" fn isr_dispatch(f: &mut Frame) {
    match f.vector as u8 {
        v @ 0..32 => fatal(f, EXCEPTIONS[v as usize]),
        TIMER => {
            apic::timer_tick();
            apic::eoi();
        }
        IPI_TEST => {
            smp::ipi_received();
            apic::eoi();
        }
        APIC_ERROR => {
            println!("cpu{}: local APIC error {:#x}", cpu::index(), apic::error_status());
            apic::eoi();
        }
        // Spurious interrupts take no EOI.
        SPURIOUS => {}
        v => {
            println!("cpu{}: unexpected interrupt {:#x}", cpu::index(), v);
            apic::eoi();
        }
    }
}

fn fatal(f: &Frame, name: &str) -> ! {
    let cr2 = x86_64::registers::control::Cr2::read_raw();
    crate::log::emergency(format_args!(
        "cpu{}: {} (vector {}, error {:#x}) at {:#x}, cr2 {:#x}, rsp {:#x}, rflags {:#x}\n  \
         rax {:#x} rbx {:#x} rcx {:#x} rdx {:#x} rsi {:#x} rdi {:#x} rbp {:#x}\n  \
         r8 {:#x} r9 {:#x} r10 {:#x} r11 {:#x} r12 {:#x} r13 {:#x} r14 {:#x} r15 {:#x}",
        cpu::index(), name, f.vector, f.error, f.rip, cr2, f.rsp, f.rflags,
        f.rax, f.rbx, f.rcx, f.rdx, f.rsi, f.rdi, f.rbp, f.r8, f.r9, f.r10, f.r11, f.r12, f.r13, f.r14, f.r15
    ));
    crate::qemu::exit(false)
}

pub fn enable() {
    x86_64::instructions::interrupts::enable();
}
