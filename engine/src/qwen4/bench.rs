//! GPU stage timing without a chain: stages of a layer range fed token embeddings (and n-gram rows when they own
//! the PLE layer), prefill chunks then decode steps back to back, so the D700 clocks stay up.
use super::pipeline::{Local, Stage};
use anyhow::Result;

type Profile = Vec<(Vec<f64>, Vec<(usize, &'static str, f64)>)>;

fn collect(local: &Local) -> Profile {
    local
        .stages
        .iter()
        .filter_map(|s| match s {
            Stage::Gpu(g) => Some((g.profile.clone(), g.phases.clone())),
            _ => None,
        })
        .collect()
}

fn add(sum: &mut Profile, cur: Profile, k: f64) {
    if sum.is_empty() {
        *sum = cur
            .iter()
            .map(|(p, ph)| (vec![0.; p.len()], ph.iter().map(|x| (x.0, x.1, 0.)).collect()))
            .collect();
    }
    for ((sp, sph), (p, ph)) in sum.iter_mut().zip(cur) {
        for (a, b) in sp.iter_mut().zip(p) {
            *a += b / k;
        }
        for (a, b) in sph.iter_mut().zip(ph) {
            a.2 += b.2 / k;
        }
    }
}

fn json(v: &Profile) -> Vec<serde_json::Value> {
    v.iter()
        .map(|(p, ph)| {
            serde_json::json!({"layer_ms": p,
                "phases": ph.iter().map(|x| serde_json::json!([x.0, x.1, x.2])).collect::<Vec<_>>()})
        })
        .collect()
}

/// Run a prompt through the local stages only (chunks of `batch`), writing the last chunk's last-token
/// activations in the reference dump format, for comparing a layer range against `qwen4 ref`.
pub fn stage_dump(local: &mut Local, out: &std::path::Path, tokens: &[u32], batch: usize) -> Result<()> {
    let mut caps = super::cpu::Captures::new();
    let chunks: Vec<_> = tokens.chunks(batch).collect();
    let mut pos = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        let x = super::embed(&local.gguf, &local.c, chunk)?;
        let ple = super::ple_rows(&local.gguf, &local.c, &tokens[..pos], chunk)?;
        let dump = if i + 1 == chunks.len() { Some(&mut caps) } else { None };
        local.forward(x, ple.as_deref(), pos, chunk.len(), dump)?;
        pos += chunk.len();
    }
    std::fs::create_dir_all(out)?;
    let mut index = String::new();
    for (name, v) in &caps {
        std::fs::write(out.join(format!("{name}.f32")), bytemuck::cast_slice(v))?;
        index.push_str(&format!("{name}\t{},1,1,1\t{}\n", v.len(), v.len()));
    }
    std::fs::write(out.join("index.tsv"), index)?;
    Ok(())
}

/// Per-layer and per-phase GPU milliseconds of the local stages: `prefill` chunks of `batch` tokens from position 0
/// (after one warm-up chunk), then `decode` single-token steps after `start` tokens.
pub fn stage_bench(
    local: &mut Local,
    decode: usize,
    start: usize,
    prefill: usize,
    batch: usize,
) -> Result<serde_json::Value> {
    // A fixed token stream; timing does not depend on the text.
    let stream: Vec<u32> = (0..(start + decode).max(batch) as u32)
        .map(|i| 1000 + (i * 7919) % 50000)
        .collect();
    let run = |local: &mut Local, toks: &[u32], pos: usize| -> Result<()> {
        let x = super::embed(&local.gguf, &local.c, toks)?;
        let ple = super::ple_rows(&local.gguf, &local.c, &stream[..pos], toks)?;
        local.forward(x, ple.as_deref(), pos, toks.len(), None)?;
        Ok(())
    };
    run(local, &stream[..batch], 0)?;
    let mut pre = Vec::new();
    let t0 = std::time::Instant::now();
    for _ in 0..prefill {
        run(local, &stream[..batch], 0)?;
        add(&mut pre, collect(local), prefill as f64);
    }
    let pre_wall = t0.elapsed().as_secs_f64() / prefill.max(1) as f64;
    // The chunk that ends the `start` prefill is timed too: at long contexts QSA attends sparsely there.
    let mut late = Vec::new();
    let mut late_pos = 0;
    for chunk in (0..start).step_by(batch) {
        let n = batch.min(start - chunk);
        run(local, &stream[chunk..chunk + n], chunk)?;
        late_pos = chunk;
    }
    if start > 0 {
        add(&mut late, collect(local), 1.);
    }
    let mut dec = Vec::new();
    let t0 = std::time::Instant::now();
    for i in 0..decode {
        run(local, &stream[start + i..start + i + 1], start + i)?;
        add(&mut dec, collect(local), decode as f64);
    }
    let dec_wall = t0.elapsed().as_secs_f64() / decode.max(1) as f64;
    // Stages holding head rows without ending the model: wall time to score one hidden state.
    let h = vec![0.01f32; local.c.d];
    let mut slices = Vec::new();
    for s in &mut local.stages {
        if let Stage::Gpu(g) = s {
            if g.head_slice(&h, None)?.is_some() {
                let t0 = std::time::Instant::now();
                for _ in 0..16 {
                    g.head_slice(&h, None)?;
                }
                slices.push(t0.elapsed().as_secs_f64() / 16. * 1e3);
            }
        }
    }
    let mib: Vec<f64> = local
        .stages
        .iter()
        .filter_map(|s| match s {
            Stage::Gpu(g) => Some(g.device_mib()),
            _ => None,
        })
        .collect();
    Ok(
        serde_json::json!({"batch": batch, "prefill_chunks": prefill, "prefill_wall_ms": pre_wall * 1e3,
        "prefill": json(&pre), "decode_steps": decode, "decode_start": start, "decode_wall_ms": dec_wall * 1e3,
        "decode": json(&dec), "device_mib": mib, "late_chunk_pos": late_pos, "late_chunk": json(&late),
        "head_slice_wall_ms": slices}),
    )
}
