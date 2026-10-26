// Counts SMC LED transactions per second of the portlights node's blinkenlights (same frame/threshold logic).
#[path = "/Users/fairfieldt/dev/claude-work/trashcan-release-20261026/a15/main/lights/src/panel.rs"] #[allow(dead_code)] mod panel;
#[path = "/Users/fairfieldt/dev/claude-work/trashcan-release-20261026/a15/main/lights/src/rng.rs"] #[allow(dead_code)] mod rng;
#[path = "/Users/fairfieldt/dev/claude-work/trashcan-release-20261026/a15/main/lights/src/blinken.rs"] #[allow(dead_code)] mod blinken;
use panel::{COUNT, encode};
fn main() {
    for fences in [10.0f32, 100.0, 1000.0, 10000.0] {
        let s = blinken::Sample { gpu: [fences / 2.0; 2], net: 1000.0 };
        let busy = blinken::busyness(&s).max(0.05);
        let mut b = blinken::Blinken::new(12345);
        let mut shown: Option<[u8; COUNT]> = None;
        let (mut leds, mut writes) = (0u64, 0u64);
        let secs = 600.0; let dt = 0.02;
        let mut t = 0.0;
        while t < secs {
            let f = b.render(dt, busy, &s);
            let mut next = shown.unwrap_or([0; COUNT]);
            let mut n = if shown.is_none() { COUNT as u64 } else { 0 };
            for (i, &v) in f.iter().enumerate() {
                let want = encode(v); let have = next[i];
                if want != have && (want.abs_diff(have) >= 3 || want == 0 || want == 255) { next[i] = want; n += 1; }
            }
            if n > 0 { writes += 1; leds += n; shown = Some(next); }
            t += dt;
        }
        let lps = leds as f32 / secs;
        println!("fences/s {fences:6}: busy {busy:.2}, {:.1} writes/s, {:.1} LED changes/s, SMC busy ~{:.1}% at 0.4 ms/LED", writes as f32 / secs, lps, lps * 0.4e-3 * 100.0);
    }
}
