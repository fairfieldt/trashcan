//! Time: the HPET as the reference, the TSC (invariant on the target) as the clock everything reads.

use acpi::hpet::Hpet;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::cpu::rdtsc;
use crate::firmware::Acpi;
use crate::println;

static TSC_HZ: AtomicU64 = AtomicU64::new(0);
static TSC_ZERO: AtomicU64 = AtomicU64::new(0);
static HPET_REGS: AtomicU64 = AtomicU64::new(0);
static HPET_FS: AtomicU64 = AtomicU64::new(0);

const HPET_CAP: usize = 0x0;
const HPET_CONFIG: usize = 0x10;
const HPET_COUNTER: usize = 0xf0;

fn hpet_reg(off: usize) -> *mut u64 {
    (HPET_REGS.load(Ordering::Relaxed) as usize + off) as *mut u64
}

/// The HPET main counter.
pub fn hpet_count() -> u64 {
    // SAFETY: mapped in `init`.
    unsafe { hpet_reg(HPET_COUNTER).read_volatile() }
}

/// Femtoseconds per HPET tick.
pub fn hpet_fs() -> u64 {
    HPET_FS.load(Ordering::Relaxed)
}

pub fn tsc_hz() -> u64 {
    TSC_HZ.load(Ordering::Relaxed)
}

/// Nanoseconds since the kernel calibrated the TSC (0 before).
pub fn now_ns() -> u64 {
    let hz = tsc_hz();
    if hz == 0 {
        return 0;
    }
    ((rdtsc() - TSC_ZERO.load(Ordering::Relaxed)) as u128 * 1_000_000_000 / hz as u128) as u64
}

pub fn ms_since(tsc: u64) -> f64 {
    (rdtsc() - tsc) as f64 * 1000.0 / tsc_hz() as f64
}

pub fn delay_us(us: u64) {
    let end = rdtsc() + tsc_hz() * us / 1_000_000;
    while rdtsc() < end {
        core::hint::spin_loop();
    }
}

/// TSC ticks over `ms` milliseconds of HPET time.
fn measure(ms: u64) -> u64 {
    let fs = HPET_FS.load(Ordering::Relaxed);
    let ticks = ms * 1_000_000_000_000 / fs;
    let h0 = hpet_count();
    let t0 = rdtsc();
    let mut h1;
    loop {
        h1 = hpet_count();
        if h1 - h0 >= ticks {
            break;
        }
    }
    let t1 = rdtsc();
    ((t1 - t0) as u128 * 1_000_000_000_000_000 / ((h1 - h0) as u128 * fs as u128)) as u64
}

pub fn init(acpi: &Acpi) {
    let h = Hpet::parse(acpi.hpet.as_ref().expect("no HPET table")).unwrap();
    assert_eq!(h.base.space, acpi::Gas::MEMORY);
    HPET_REGS.store(crate::mem::map_mmio(h.base.address, 0x400), Ordering::Relaxed);
    // SAFETY: the HPET's registers, just mapped.
    let cap = unsafe { hpet_reg(HPET_CAP).read_volatile() };
    let fs = cap >> 32;
    assert!(fs > 0 && fs <= 100_000_000, "HPET period {fs} fs");
    HPET_FS.store(fs, Ordering::Relaxed);
    // SAFETY: as above. Enable the main counter, legacy routing off.
    unsafe { hpet_reg(HPET_CONFIG).write_volatile(hpet_reg(HPET_CONFIG).read_volatile() & !3 | 1) };

    // Three 10 ms measurements; the median is the frequency.
    let mut m = [measure(10), measure(10), measure(10)];
    m.sort_unstable();
    TSC_ZERO.store(rdtsc(), Ordering::Relaxed);
    TSC_HZ.store(m[1], Ordering::Relaxed);
    println!(
        "time: hpet {:#x}, {} timers, {}.{:06} MHz; tsc {} Hz (spread {} Hz)",
        h.base.address,
        (cap >> 8 & 0x1f) + 1,
        1_000_000_000 / fs,
        1_000_000_000_000_000 / fs % 1_000_000,
        m[1],
        m[2] - m[0]
    );
}
