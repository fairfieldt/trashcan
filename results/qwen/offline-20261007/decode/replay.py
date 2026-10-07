#!/usr/bin/env python3
"""Model the archived Rust MTP controller using saved aggregate costs/acceptance.

This does not run inference or predict D700 throughput. It samples possible round
outcomes from saved prefix-acceptance histograms; the original round order and
per-round durations are unavailable. Short final tails use serial cost because
these records have no one-draft timings. Every mode gets the same assumptions.
"""
import json
from pathlib import Path
import subprocess
import tempfile


EXPERIMENT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "reference-inputs"


def records(name):
    return {r["case"]["name"]: r for line in (DATA / name).read_text().splitlines()
            if line.strip() for r in [json.loads(line)]}


plain = records("final-corpus-plain.json")
mtp = records("final-corpus-mtp.json")
cases = []
for name, record in mtp.items():
    stats = record["mtp"]
    first, second = stats["accepted_at"]
    rounds = stats["rounds"]
    # Point estimates only. Warmup and thermal variation are not separated in
    # these mixed-session observations; do not infer confidence intervals.
    serial_s = 1 / plain[name]["tg_tok_s"]
    draft_s = sum(stats[k] for k in (
        "draft_ms_per_round", "verify_ms_per_round", "rollback_ms_per_round")) / 1000
    cases.append(f"({json.dumps(name)}, {serial_s!r}, {draft_s!r}, "
                 f"[{rounds - first}, {first - second}, {second}])")

source = (EXPERIMENT / "candidate-source/spec.rs").read_text()
# Compile the original candidate policy preserved from source.tar.gz, so later
# engine changes cannot silently alter this historical replay.
policy = source[source.index("#[derive(Default)]\nstruct Cost"):
                source.index("\npub struct Spec")]
harness = r'''
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

fn episode(budget: usize, policy: usize, serial: f64, draft: f64,
           hist: [usize; 3], seed: u64) -> (f64, f64) {
    let mut controller = Adaptive::default();
    let (mut tokens, mut seconds, mut serial_steps, mut rounds) = (0, 0., 0, 0);
    let mut state = seed;
    while tokens + 1 < budget {
        let tail = tokens + 3 >= budget;
        let mode = if tail { 0 } else if policy == 2 { controller.choose() } else { policy };
        let (elapsed, committed) = if mode == 0 { (serial, 1) } else {
            let r = uniform(&mut state) * hist.iter().sum::<usize>() as f64;
            (draft, if r < hist[0] as f64 { 1 } else if r < (hist[0] + hist[1]) as f64 { 2 } else { 3 })
        };
        if policy == 2 && !tail { controller.observe(mode, elapsed, committed); }
        tokens += committed;
        seconds += elapsed;
        serial_steps += usize::from(mode == 0);
        rounds += 1;
    }
    (seconds, serial_steps as f64 / rounds as f64)
}

fn main() {
    let cases: &[(&str, f64, f64, [usize; 3])] = &[CASES];
    println!("case,tokens,serial_model_tok_s,fixed_model_tok_s,adaptive_model_tok_s,adaptive_serial_round_fraction");
    for &(name, serial, draft, hist) in cases {
        for budget in [128, 2048] {
            let mut sum = [0.; 3];
            let mut serial_fraction = 0.;
            for seed in 0..1024 {
                for (policy, seconds) in sum.iter_mut().enumerate() {
                    let (time, fraction) = episode(budget, policy, serial, draft, hist, seed);
                    *seconds += time;
                    if policy == 2 { serial_fraction += fraction; }
                }
            }
            println!("{name},{budget},{:.3},{:.3},{:.3},{:.4}",
                1024. * budget as f64 / sum[0], 1024. * budget as f64 / sum[1],
                1024. * budget as f64 / sum[2], serial_fraction / 1024.);
        }
    }
}
'''.replace("CASES", ",\n".join(cases))

with tempfile.TemporaryDirectory(prefix="qwen-policy-") as directory:
    directory = Path(directory)
    path = directory / "replay.rs"
    path.write_text(policy + harness)
    binary = directory / "replay"
    subprocess.run(["rustc", "--edition=2021", "-O", str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
