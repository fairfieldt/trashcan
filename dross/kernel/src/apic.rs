//! Local APICs in x2APIC mode (MSRs, no MMIO), the I/O APICs (all inputs masked until a driver asks for one)
//! and the legacy 8259s (masked for good).

use acpi::madt::{Entry, Madt};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use x86_64::instructions::port::Port;

use crate::cpu::{rdmsr, rdtsc, wrmsr, Features};
use crate::firmware::Acpi;
use crate::interrupts::{APIC_ERROR, SPURIOUS, TIMER};
use crate::{println, time};

const IA32_APIC_BASE: u32 = 0x1b;
const APIC_ENABLE: u64 = 1 << 11;
const APIC_X2: u64 = 1 << 10;
const IA32_TSC_DEADLINE: u32 = 0x6e0;

const X2_ID: u32 = 0x802;
const X2_VERSION: u32 = 0x803;
const X2_TPR: u32 = 0x808;
const X2_EOI: u32 = 0x80b;
const X2_SVR: u32 = 0x80f;
const X2_ESR: u32 = 0x828;
const X2_ICR: u32 = 0x830;
const X2_LVT_TIMER: u32 = 0x832;
const X2_LVT_LINT0: u32 = 0x835;
const X2_LVT_LINT1: u32 = 0x836;
const X2_LVT_ERROR: u32 = 0x837;
const X2_TIMER_INITIAL: u32 = 0x838;
const X2_TIMER_CURRENT: u32 = 0x839;
const X2_TIMER_DIVIDE: u32 = 0x83e;

const LVT_MASKED: u64 = 1 << 16;
const LVT_NMI: u64 = 4 << 8;
const LVT_TIMER_TSC_DEADLINE: u64 = 2 << 17;

pub const ICR_INIT: u64 = 5 << 8 | 1 << 14;
pub const ICR_STARTUP: u64 = 6 << 8 | 1 << 14;

static USE_DEADLINE: AtomicBool = AtomicBool::new(false);
/// APIC timer counts (divide by 16) per 2^32 TSC ticks, for one-shot mode.
static COUNTS_PER_TSC: AtomicU64 = AtomicU64::new(0);

pub fn id() -> u32 {
    rdmsr(X2_ID) as u32
}

pub fn eoi() {
    // SAFETY: x2APIC EOI.
    unsafe { wrmsr(X2_EOI, 0) };
}

pub fn error_status() -> u32 {
    // SAFETY: writing ESR latches the errors so they can be read.
    unsafe { wrmsr(X2_ESR, 0) };
    rdmsr(X2_ESR) as u32
}

/// `icr`: delivery mode and vector bits of the ICR's low half.
pub fn send_ipi(apic_id: u32, icr: u64) {
    // SAFETY: x2APIC ICR, one write sends.
    unsafe { wrmsr(X2_ICR, (apic_id as u64) << 32 | icr) };
}

/// x2APIC mode on, spurious vector set, LINT0 masked, LINT1 as NMI (as the MADT has it on the Mac Pro),
/// errors to `APIC_ERROR`, timer masked.
pub fn init_local() {
    let base = rdmsr(IA32_APIC_BASE);
    // SAFETY: xAPIC enable then x2APIC (the order the SDM requires), then the local vector table.
    unsafe {
        wrmsr(IA32_APIC_BASE, base | APIC_ENABLE);
        wrmsr(IA32_APIC_BASE, base | APIC_ENABLE | APIC_X2);
        wrmsr(X2_TPR, 0);
        wrmsr(X2_TIMER_DIVIDE, 0b0011);
        let mode = if USE_DEADLINE.load(Ordering::Relaxed) { LVT_TIMER_TSC_DEADLINE } else { 0 };
        wrmsr(X2_LVT_TIMER, mode | TIMER as u64);
        wrmsr(X2_LVT_LINT0, LVT_MASKED);
        wrmsr(X2_LVT_LINT1, LVT_NMI);
        wrmsr(X2_LVT_ERROR, APIC_ERROR as u64);
        wrmsr(X2_SVR, 1 << 8 | SPURIOUS as u64);
        wrmsr(X2_ESR, 0);
    }
}

fn ioapic_mask_all(base: u64) -> u32 {
    let regs = crate::mem::map_mmio(base, 0x20);
    let sel = regs as *mut u32;
    let win = (regs + 0x10) as *mut u32;
    // SAFETY: the I/O APIC's index/data window, just mapped.
    unsafe {
        sel.write_volatile(1);
        let entries = (win.read_volatile() >> 16 & 0xff) + 1;
        for i in 0..entries {
            sel.write_volatile(0x10 + 2 * i);
            win.write_volatile(1 << 16);
        }
        entries
    }
}

pub fn init(acpi: &Acpi) {
    let madt = Madt::parse(&acpi.madt).unwrap();
    if madt.flags & 1 != 0 {
        // SAFETY: the 8259 data ports: mask every line.
        unsafe {
            Port::<u8>::new(0x21).write(0xff);
            Port::<u8>::new(0xa1).write(0xff);
        }
    }
    for e in madt.entries().flatten() {
        if let Entry::IoApic { id, address, gsi_base } = e {
            let n = ioapic_mask_all(address as u64);
            println!("apic: ioapic {} at {:#x}, gsi {}-{}, all masked", id, address, gsi_base, gsi_base + n - 1);
        }
    }
    // x2APIC on first (the timer registers are MSRs), then the timer mode, then the final LVT.
    init_local();
    timer_calibrate();
    init_local();
    println!(
        "apic: cpu0 x2apic id {}, version {:#x}, timer {}",
        id(),
        rdmsr(X2_VERSION) & 0xff,
        if USE_DEADLINE.load(Ordering::Relaxed) { "tsc deadline" } else { "one-shot" }
    );
}

/// Chooses the timer mode: TSC deadline where the CPU has it (the Mac Pro), otherwise one-shot counts,
/// calibrated against the TSC.
fn timer_calibrate() {
    if Features::detect().tsc_deadline {
        USE_DEADLINE.store(true, Ordering::Relaxed);
        return;
    }
    // SAFETY: the local APIC timer registers, masked while counting.
    unsafe {
        wrmsr(X2_TIMER_DIVIDE, 0b0011);
        wrmsr(X2_LVT_TIMER, LVT_MASKED | TIMER as u64);
        let t0 = rdtsc();
        wrmsr(X2_TIMER_INITIAL, u32::MAX as u64);
        time::delay_us(10_000);
        let counted = u32::MAX as u64 - rdmsr(X2_TIMER_CURRENT);
        let ticks = rdtsc() - t0;
        wrmsr(X2_TIMER_INITIAL, 0);
        COUNTS_PER_TSC.store(((counted as u128) << 32) as u64 / ticks.max(1) as u64, Ordering::Relaxed);
    }
}

/// Fires the timer interrupt at TSC `deadline` (or right away if it has passed).
pub fn timer_arm(deadline: u64) {
    if USE_DEADLINE.load(Ordering::Relaxed) {
        // SAFETY: the LVT is in deadline mode.
        unsafe { wrmsr(IA32_TSC_DEADLINE, deadline) };
        return;
    }
    let now = rdtsc();
    let ticks = deadline.saturating_sub(now) as u128;
    let counts = ((ticks * COUNTS_PER_TSC.load(Ordering::Relaxed) as u128) >> 32).clamp(1, u32::MAX as u128);
    // SAFETY: one-shot mode: writing the initial count starts it.
    unsafe { wrmsr(X2_TIMER_INITIAL, counts as u64) };
}

pub fn timer_disarm() {
    // SAFETY: zero stops either mode.
    unsafe {
        if USE_DEADLINE.load(Ordering::Relaxed) {
            wrmsr(IA32_TSC_DEADLINE, 0);
        } else {
            wrmsr(X2_TIMER_INITIAL, 0);
        }
    }
}
