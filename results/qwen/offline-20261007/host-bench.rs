// rustc --edition=2021 -O results/qwen/offline-20261007/host-bench.rs -o target/tp-host-bench
// CPU RAM only, not mapped D700 memory or end-to-end inference.
#[path = "rejected-host-fusion.rs"]
mod fusion;
#[path = "candidate-source/host.rs"]
mod host;
use std::{hint::black_box as bb, time::Instant};

#[inline(never)]
fn old_sum(out: &mut [f32], a: &[f32], b: &[f32]) {
    out.copy_from_slice(a);
    for (a, b) in out.iter_mut().zip(b) { *a += b; }
}

#[inline(never)]
fn old_norm(resid: &mut [f32], sum: &mut [f32], remote: &[f32], w: &[f32], out: &mut [f32]) {
    for (a, b) in sum.iter_mut().zip(remote) { *a += b; }
    for (a, b) in resid.iter_mut().zip(sum) { *a += *b; }
    for (x, o) in resid.chunks_exact(w.len()).zip(out.chunks_exact_mut(w.len())) {
        let mut acc = [0f32; 8];
        for c in x.chunks_exact(8) {
            for j in 0..8 { acc[j] += c[j] * c[j]; }
        }
        let ss = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        let inv = 1.0 / (ss / w.len() as f32 + 1e-6).sqrt();
        for ((o, &v), &wv) in o.iter_mut().zip(x).zip(w) { *o = v * inv * wv; }
    }
}

fn values(n: usize, shift: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 37 + shift) % 101) as f32 / 101. - 0.5).collect()
}

fn bench(n: usize, fused: bool, iterations: usize, norm: bool) -> f64 {
    let d = 5120;
    let a = values(d * n, 3);
    let b = values(d * n, 11);
    let remote = values(d * n, 29);
    let weights = values(d, 17);
    let mut resid = values(d * n, 31);
    let mut sum = vec![0.; d * n];
    let mut out = sum.clone();
    let mut second = sum.clone();
    let start = Instant::now();
    for _ in 0..iterations {
        if fused {
            host::sum_partials(bb(&mut sum), bb(&[&a, &b]), 0);
            if norm { fusion::add_norm(bb(&mut resid), bb(&sum), Some(bb(&remote)), bb(&weights), 1e-6, bb(&mut out)); }
        } else {
            old_sum(bb(&mut sum), bb(&a), bb(&b));
            if norm { old_norm(bb(&mut resid), bb(&mut sum), bb(&remote), bb(&weights), bb(&mut out)); }
        }
        if norm { second.copy_from_slice(bb(&out)); }
        bb(&second);
    }
    start.elapsed().as_secs_f64() * 1e6 / iterations as f64
}

fn main() {
    println!("case,n,rep,baseline_us,candidate_us");
    for (norm, widths) in [(true, &[1, 3, 8][..]), (false, &[1, 3, 256][..])] {
        for &n in widths {
            let iters = if n == 256 { 1000 } else { 10000 };
            for rep in 0..9 {
                let (old, new) = if rep % 2 == 0 {
                    let old = bench(n, false, iters, norm);
                    (old, bench(n, true, iters, norm))
                } else {
                    let new = bench(n, true, iters, norm);
                    (bench(n, false, iters, norm), new)
                };
                println!("{},{n},{rep},{old:.6},{new:.6}", if norm {"sum_norm_copy"} else {"sum_only"});
            }
        }
    }
}
