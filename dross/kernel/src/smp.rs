//! Starting the other CPUs: INIT, then STARTUP IPIs to a trampoline page below 1 MiB that goes from real mode
//! straight to long mode on the kernel's page tables, then into `ap_main` on a stack of its own.

use acpi::madt::Madt;
use bootinfo::{BootInfo, DIRECT_MAP};
use core::arch::global_asm;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::cpu::{self, APIC_IDS, MAX_CPUS};
use crate::firmware::Acpi;
use crate::interrupts::IPI_TEST;
use crate::{apic, mem, println, time};

const AP_STACK_PAGES: usize = 16;

global_asm!(
    r#"
.section .rodata.ap_trampoline, "a"
.global AP_START, AP_END, AP_JUMP, AP_GDT, AP_GDTR, AP_LONG, AP_CR3, AP_STACK, AP_ENTRY, AP_ARG
.code16
AP_START:
    cli
    cld
    movw %cs, %ax
    movw %ax, %ds
    movl %cr4, %eax
    orl $0x20, %eax
    movl %eax, %cr4
    movl (AP_CR3 - AP_START), %eax
    movl %eax, %cr3
    movl $0xc0000080, %ecx
    rdmsr
    orl $0x900, %eax
    wrmsr
    lgdtl (AP_GDTR - AP_START)
    movl %cr0, %eax
    orl $0x80000001, %eax
    movl %eax, %cr0
    .byte 0x66, 0xea
AP_JUMP:
    .long 0
    .word 0x08
.balign 8
AP_GDT:
    .quad 0
    .quad 0x00af9a000000ffff
    .quad 0x00cf92000000ffff
AP_GDTR:
    .word 23
    .long 0
.balign 8
AP_CR3:
    .quad 0
AP_STACK:
    .quad 0
AP_ENTRY:
    .quad 0
AP_ARG:
    .quad 0
.code64
AP_LONG:
    movw $0x10, %ax
    movw %ax, %ds
    movw %ax, %es
    movw %ax, %ss
    xorl %eax, %eax
    movw %ax, %fs
    movw %ax, %gs
    movq AP_STACK(%rip), %rsp
    movq AP_ARG(%rip), %rdi
    movq AP_ENTRY(%rip), %rax
    xorl %ebp, %ebp
    callq *%rax
    ud2
AP_END:
.text
"#,
    options(att_syntax)
);

unsafe extern "C" {
    static AP_START: u8;
    static AP_END: u8;
    static AP_JUMP: u8;
    static AP_GDT: u8;
    static AP_GDTR: u8;
    static AP_LONG: u8;
    static AP_CR3: u8;
    static AP_STACK: u8;
    static AP_ENTRY: u8;
    static AP_ARG: u8;
}

/// Offset of a trampoline label from its start.
fn off(label: &u8) -> usize {
    // SAFETY: both are symbols in the same section.
    label as *const u8 as usize - unsafe { &AP_START as *const u8 as usize }
}

static STARTED: AtomicBool = AtomicBool::new(false);
/// The starting AP's TSC as it reports in, and how long its timer self-test took (us, MAX: never fired).
static AP_TSC: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static TIMER_US: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static ONLINE: AtomicU32 = AtomicU32::new(1);
static IPIS: [AtomicU32; MAX_CPUS] = [const { AtomicU32::new(0) }; MAX_CPUS];

extern "sysv64" fn ap_main(index: u64) -> ! {
    let apic_id = cpu::cpuid(1, 0).1 >> 24;
    cpu::init_cpu(index as u32, apic_id);
    apic::init_local();
    assert_eq!(apic::id(), apic_id);
    crate::sched::init_cpu();
    TIMER_US.store(apic::timer_selftest().unwrap_or(u64::MAX), Ordering::Relaxed);
    ONLINE.fetch_add(1, Ordering::AcqRel);
    AP_TSC.store(cpu::rdtsc(), Ordering::Release);
    STARTED.store(true, Ordering::Release);
    crate::sched::idle()
}

/// CPUs up so far (indices 0..online).
pub fn online() -> u32 {
    ONLINE.load(Ordering::Acquire)
}

pub fn ipi_received() {
    IPIS[cpu::index() as usize].fetch_add(1, Ordering::AcqRel);
}

/// Starts every enabled CPU in the MADT, one at a time. Returns the number online, the BSP included.
pub fn start_aps(info: &BootInfo, acpi: &Acpi) -> u32 {
    let page = info.trampoline;
    assert!(page < 0x10_0000 && page & 0xfff == 0 && info.pml4 < 1 << 32);
    // SAFETY: the loader's trampoline page (identity-mapped) and the trampoline image in .rodata.
    unsafe {
        let len = &AP_END as *const u8 as usize - &AP_START as *const u8 as usize;
        assert!(len <= 4096);
        core::ptr::copy_nonoverlapping(&AP_START as *const u8, (DIRECT_MAP + page) as *mut u8, len);
        let at = |label: &u8| (DIRECT_MAP + page + off(label) as u64) as *mut u8;
        // Linear addresses: the far jump's target and the GDT's base.
        (at(&AP_JUMP) as *mut u32).write_unaligned((page as usize + off(&AP_LONG)) as u32);
        (at(&AP_GDTR).add(2) as *mut u32).write_unaligned((page as usize + off(&AP_GDT)) as u32);
        (at(&AP_CR3) as *mut u64).write(info.pml4);
        (at(&AP_ENTRY) as *mut u64).write(ap_main as *const () as u64);
    }
    let set = |label: &u8, v: u64| unsafe { ((DIRECT_MAP + page + off(label) as u64) as *mut u64).write_volatile(v) };

    let bsp = apic::id();
    let madt = Madt::parse(&acpi.madt).unwrap();
    let mut index = 1u32;
    let t0 = cpu::rdtsc();
    for apic_id in madt.cpus().filter(|&id| id != bsp) {
        if index as usize >= MAX_CPUS {
            println!("smp: more than {MAX_CPUS} cpus, ignoring the rest");
            break;
        }
        let stack = mem::alloc_stack(AP_STACK_PAGES);
        set(unsafe { &AP_STACK }, stack + (AP_STACK_PAGES * 4096) as u64);
        set(unsafe { &AP_ARG }, index as u64);
        STARTED.store(false, Ordering::Release);
        apic::send_ipi(apic_id, apic::ICR_INIT);
        time::delay_us(10_000);
        let mut up = false;
        for _ in 0..2 {
            apic::send_ipi(apic_id, apic::ICR_STARTUP | page >> 12);
            // Up to 100 ms per STARTUP (an AP takes microseconds; QEMU's TCG longer).
            let deadline = cpu::rdtsc() + time::tsc_hz() / 2;
            while cpu::rdtsc() < deadline {
                if STARTED.load(Ordering::Acquire) {
                    // The AP's TSC against ours at (nearly) the same moment.
                    let skew = AP_TSC.load(Ordering::Acquire) as i64 - cpu::rdtsc() as i64;
                    let t = TIMER_US.load(Ordering::Relaxed);
                    println!("smp: cpu {index} apic id {apic_id}: tsc {skew:+} vs cpu 0, 1 ms timer took {}", if t == u64::MAX { alloc::string::String::from("forever") } else { alloc::format!("{t} us") });
                    up = true;
                    break;
                }
                core::hint::spin_loop();
            }
            if up {
                break;
            }
        }
        if !up {
            println!("smp: cpu with apic id {apic_id} did not start");
            continue;
        }
        index += 1;
    }
    let online = ONLINE.load(Ordering::Acquire);
    println!("smp: {} of {} cpus online in {:.1} ms", online, madt.cpus().count(), time::ms_since(t0));
    online
}

/// Sends a fixed IPI to every other CPU and waits for each to take it.
pub fn ipi_check() {
    let online = ONLINE.load(Ordering::Acquire) as usize;
    let t0 = cpu::rdtsc();
    for i in 1..online {
        apic::send_ipi(APIC_IDS[i].load(Ordering::Acquire), IPI_TEST as u64);
    }
    for i in 1..online {
        let mut n = 0;
        while IPIS[i].load(Ordering::Acquire) == 0 {
            assert!(n < 100_000, "cpu{i} did not take its IPI");
            time::delay_us(1);
            n += 1;
        }
    }
    println!("smp: {} cpus took an IPI in {:.3} ms", online - 1, time::ms_since(t0));
}
