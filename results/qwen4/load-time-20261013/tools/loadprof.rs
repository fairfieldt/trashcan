//! TEMPORARY load-time counters (timing builds only; never committed).
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

pub const N: usize = 24;
pub static T: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
pub static START: OnceLock<Instant> = OnceLock::new();
pub const NAMES: [&str; N] = [
    "alloc dst", "alloc staging", "staging write", "copy+submit+wait", "free staging", "repack", "expert concat",
    "f32 convert", "fused concat", "other buffers", "device new", "stage new", "bytes", "uploads", "fill", "wait slot",
    "submit", "finish wait", "prefetch wait", "drops", "submits", "c21", "c22", "c23",
];
/// Counters that count things rather than nanoseconds, or that cover the others.
const NOT_PARTS: [usize; 5] = [10, 11, 12, 13, 20];
pub fn start() {
    START.get_or_init(Instant::now);
}
pub fn since() -> f64 {
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}
pub fn add(i: usize, v: u64) {
    T[i].fetch_add(v, Relaxed);
}
pub fn time<R>(i: usize, f: impl FnOnce() -> R) -> R {
    let t = Instant::now();
    let r = f();
    T[i].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    r
}
pub fn report(label: &str, total: f64) {
    let mut s = format!("LOADPROF {label} at {:.2} s: total {total:.2} s", since());
    let mut sum = 0.;
    for (i, n) in NAMES.iter().enumerate() {
        let v = T[i].swap(0, Relaxed);
        if v == 0 {
            continue;
        }
        if i == 12 {
            s += &format!(", {n} {:.2} GB", v as f64 / 1e9);
        } else if i == 13 || i == 20 {
            s += &format!(", {n} {v}");
        } else {
            let secs = v as f64 / 1e9;
            if !NOT_PARTS.contains(&i) {
                sum += secs;
            }
            s += &format!(", {n} {secs:.2} s");
        }
    }
    s += &format!(", unaccounted {:.2} s", total - sum);
    eprintln!("{s}");
}
pub fn mark(label: &str) {
    eprintln!("LOADPROF {label} at {:.2} s", since());
}
