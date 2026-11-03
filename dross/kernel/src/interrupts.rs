//! Interrupt and exception entry. The stubs (build.rs) push an error code (zero when the CPU gives none) and
//! the vector; `isr_common` swaps GS if it came from user mode, saves the registers and calls `isr_dispatch`.
//! `isr_return` is also how system calls and new threads leave the kernel.

use core::arch::global_asm;

use crate::{apic, cpu, println, sched, smp};

global_asm!(include_str!(concat!(env!("OUT_DIR"), "/isr.s")));
global_asm!(
    r#"
isr_common:
    test byte ptr [rsp + 24], 3
    jz 1f
    swapgs
1:
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
.global isr_return
isr_return:
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
    test byte ptr [rsp + 8], 3
    jz 2f
    swapgs
2:
    iretq
"#
);

pub const TIMER: u8 = 0x30;
pub const IPI_TEST: u8 = 0xf0;
pub const RESCHED: u8 = 0xf1;
pub const SHOOTDOWN: u8 = 0xf2;
pub const IOMMU_FAULT: u8 = 0xf3;
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
    let from_user = f.cs & 3 != 0;
    match f.vector as u8 {
        v @ 0..32 if from_user => user_fault(f, EXCEPTIONS[v as usize]),
        v @ 0..32 => fatal(f, EXCEPTIONS[v as usize]),
        TIMER => {
            apic::TIMER_IRQS[cpu::index() as usize].fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            apic::eoi();
            sched::timer_interrupt();
        }
        RESCHED => {
            apic::eoi();
            sched::request_resched();
        }
        SHOOTDOWN => {
            apic::eoi();
            crate::paging::serve_shootdown();
        }
        IOMMU_FAULT => {
            crate::iommu::fault_interrupt();
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
            if !crate::irq::dispatch(v) {
                println!("cpu{}: unexpected interrupt {:#x}", cpu::index(), v);
            }
            apic::eoi();
        }
    }
    // Only interrupts that arrive in user mode or in the idle loop (the kernel runs with interrupts off) get here,
    // so switching threads is safe.
    sched::preempt_point();
    if from_user && sched::current().killed() {
        sched::exit_current();
    }
}

/// An exception in user mode ends the process.
fn user_fault(f: &Frame, name: &str) {
    let cr2 = x86_64::registers::control::Cr2::read_raw();
    let p = sched::current_process();
    println!(
        "{}: {} (vector {}, error {:#x}) at {:#x}, cr2 {:#x}, rsp {:#x}: process killed",
        p.name, name, f.vector, f.error, f.rip, cr2, f.rsp
    );
    p.kill(-(f.vector as i64) - 1000);
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
