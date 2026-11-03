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
const LVT_TIMER_PERIODIC: u64 = 1 << 17;
const LVT_TIMER_TSC_DEADLINE: u64 = 2 << 17;

pub const ICR_INIT: u64 = 5 << 8 | 1 << 14;
pub const ICR_STARTUP: u64 = 6 << 8 | 1 << 14;

static USE_DEADLINE: AtomicBool = AtomicBool::new(false);
/// TSC ticks (deadline mode) or APIC timer counts (periodic mode) per timer tick.
static PERIOD: AtomicU64 = AtomicU64::new(0);
static NEXT_DEADLINE: AtomicU64 = AtomicU64::new(0);
static TICKS: AtomicU64 = AtomicU64::new(0);

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
        wrmsr(X2_LVT_TIMER, LVT_MASKED | TIMER as u64);
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
    init_local();
    println!("apic: cpu0 x2apic id {}, version {:#x}", id(), rdmsr(X2_VERSION) & 0xff);
}

static LAST_TICK: AtomicU64 = AtomicU64::new(0);
/// Shortest TSC interval between two ticks since the last `timer_start`.
static MIN_INTERVAL: AtomicU64 = AtomicU64::new(u64::MAX);

pub fn timer_tick() {
    TICKS.fetch_add(1, Ordering::Relaxed);
    let now = rdtsc();
    let last = LAST_TICK.swap(now, Ordering::Relaxed);
    if last != 0 {
        MIN_INTERVAL.fetch_min(now - last, Ordering::Relaxed);
    }
    if USE_DEADLINE.load(Ordering::Relaxed) {
        let next = NEXT_DEADLINE.fetch_add(PERIOD.load(Ordering::Relaxed), Ordering::Relaxed) + PERIOD.load(Ordering::Relaxed);
        // SAFETY: arms the next deadline.
        unsafe { wrmsr(IA32_TSC_DEADLINE, next) };
    }
}

fn timer_start(hz: u64) {
    LAST_TICK.store(0, Ordering::Relaxed);
    MIN_INTERVAL.store(u64::MAX, Ordering::Relaxed);
    if Features::detect().tsc_deadline {
        USE_DEADLINE.store(true, Ordering::Relaxed);
        let period = time::tsc_hz() / hz;
        PERIOD.store(period, Ordering::Relaxed);
        let first = rdtsc() + period;
        NEXT_DEADLINE.store(first, Ordering::Relaxed);
        // SAFETY: the LVT must be in deadline mode before the deadline MSR is written.
        unsafe {
            wrmsr(X2_LVT_TIMER, LVT_TIMER_TSC_DEADLINE | TIMER as u64);
            core::arch::asm!("mfence");
            wrmsr(IA32_TSC_DEADLINE, first);
        }
    } else {
        // Count the APIC timer (divide by 16) against the TSC for 10 ms, then run it periodic.
        // SAFETY: the local APIC timer registers.
        unsafe {
            wrmsr(X2_TIMER_DIVIDE, 0b0011);
            wrmsr(X2_LVT_TIMER, LVT_MASKED | TIMER as u64);
            wrmsr(X2_TIMER_INITIAL, u32::MAX as u64);
            time::delay_us(10_000);
            let counted = u32::MAX as u64 - rdmsr(X2_TIMER_CURRENT);
            let period = counted * 100 / hz;
            PERIOD.store(period, Ordering::Relaxed);
            wrmsr(X2_LVT_TIMER, LVT_TIMER_PERIODIC | TIMER as u64);
            wrmsr(X2_TIMER_INITIAL, period);
        }
    }
}

fn timer_stop() {
    // SAFETY: masks the timer and clears both kinds of arming.
    unsafe {
        wrmsr(X2_LVT_TIMER, LVT_MASKED | TIMER as u64);
        wrmsr(X2_TIMER_INITIAL, 0);
        if USE_DEADLINE.load(Ordering::Relaxed) {
            wrmsr(IA32_TSC_DEADLINE, 0);
        }
    }
}

/// Runs the timer at 1 kHz for 200 ms and checks the ticks against the TSC, then stops it (the kernel arms
/// timers only when something waits). On hardware the count must be within 10% and no tick early. Under
/// QEMU's TCG on macOS the host coalesces QEMU's timer wakeups: ticks come 2-18 ms late, missed ones are
/// dropped and the next comes on the grid right after a late one. There the check is only that ticks arrive.
pub fn timer_check() {
    const HZ: u64 = 1000;
    const MS: u64 = 200;
    let h0 = time::hpet_count();
    let t0 = TICKS.load(Ordering::Relaxed);
    timer_start(HZ);
    let end = rdtsc() + time::tsc_hz() * MS / 1000;
    while rdtsc() < end {
        x86_64::instructions::interrupts::enable_and_hlt();
    }
    timer_stop();
    let ticks = TICKS.load(Ordering::Relaxed) - t0;
    let hpet_us = (time::hpet_count() - h0) as u128 * time::hpet_fs() as u128 / 1_000_000_000;
    let min_us = MIN_INTERVAL.load(Ordering::Relaxed).saturating_mul(1_000_000) / time::tsc_hz();
    println!(
        "apic: timer ({}) {} ticks at {} Hz in {} us of HPET time, shortest interval {} us",
        if USE_DEADLINE.load(Ordering::Relaxed) { "tsc deadline" } else { "periodic" },
        ticks,
        HZ,
        hpet_us,
        min_us
    );
    let expect = MS * HZ / 1000;
    assert!(ticks >= 2 && ticks <= expect * 11 / 10 + 1, "timer: {ticks} ticks, expected {expect}");
    if !Features::detect().hypervisor {
        assert!(ticks >= expect * 9 / 10, "timer: {ticks} ticks, expected {expect}");
        assert!(min_us * HZ >= 900_000, "timer: a tick came after {min_us} us");
    }
}
