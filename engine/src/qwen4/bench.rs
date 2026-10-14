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

/// The head's candidate kernels (`qwen4_head_topk`, `qwen4_head_merge`) against qwen35's `topk` and `merge` on GPU
/// `gpu`: random logits on a grid of 1/64 (many exact ties, some -0 and +0, equal maxima in different chunks) for
/// the head slices of six Macs, a third, the whole vocabulary and a partial chunk, 1, 2 and 4 rows. Every row's 64
/// candidates must match bit for bit between the two pairs and a CPU sort (score, then the lower id). One JSON line
/// per shape with the milliseconds of each pair (`iterations` back to back).
pub fn head_topk_check(gpu: usize, iterations: u32) -> Result<()> {
    use anyhow::{ensure, Context};
    use ash::vk;
    let inst = std::rc::Rc::new(crate::gpu::Instance::new()?);
    let pd = *inst.amd_gpus()?.get(gpu).context("no such GPU")?;
    let d = crate::vulkan::Device::new(inst, pd)?;
    let q = unsafe {
        d.dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)?
    };
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for rows in [21504usize, 11776, 82944, 248320, 1000] {
        for cols in [1usize, 2, 4] {
            let chunks = rows.div_ceil(1024);
            let mut logits: Vec<f32> = (0..rows * cols)
                .map(|_| match next() {
                    r if r % 97 == 0 => -0.0,
                    r if r % 89 == 0 => 0.0,
                    r => ((r >> 20) % 1024) as f32 / 64. - 8.,
                })
                .collect();
            for c in 0..cols {
                for k in 0..5 {
                    logits[c * rows + (k * 4099 + c) % rows] = 9.0;
                }
            }
            let x = d.upload(bytemuck::cast_slice(&logits))?;
            let scratch = d.buffer(cols * chunks * 64 * 8, false, false, None)?;
            let outs = [d.buffer(cols * 64 * 8, true, false, None)?, d.buffer(cols * 64 * 8, true, false, None)?];
            let params = d.upload(bytemuck::cast_slice(&[0u32, cols as u32]))?;
            let push = |a: u64, c: u64, u: [u32; 16]| {
                let mut p = crate::vulkan::Push::default();
                (p.ptr[0], p.ptr[2], p.ptr[6], p.u) = (a, c, params.address, u);
                p
            };
            let mut ms = [0f64; 2];
            for (k, (topk, merge)) in [("topk", "merge"), ("qwen4_head_topk", "qwen4_head_merge")].into_iter().enumerate() {
                let cmd = d.begin()?;
                unsafe {
                    d.dev.cmd_reset_query_pool(cmd, q, 0, 2);
                    d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
                }
                for _ in 0..iterations {
                    let mut u = [0u32; 16];
                    (u[0], u[1], u[2], u[15]) = (rows as u32, 64, (chunks * 64) as u32, cols as u32);
                    d.dispatch(cmd, topk, push(x.address, scratch.address, u), chunks as u32, cols as u32);
                    let mut u = [0u32; 16];
                    (u[0], u[1], u[15]) = ((chunks * 64) as u32, 64, cols as u32);
                    d.dispatch(cmd, merge, push(scratch.address, outs[k].address, u), 1, cols as u32);
                }
                unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1) };
                d.finish(cmd)?;
                d.run_reusable(cmd)?;
                d.run(cmd)?;
                let mut t = [0u64; 2];
                unsafe {
                    d.dev.get_query_pool_results(q, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?
                };
                ms[k] = (t[1] - t[0]) as f64 * d.period as f64 / 1e6 / iterations as f64;
            }
            let (old, new) = (outs[0].read(cols * 64 * 8)?, outs[1].read(cols * 64 * 8)?);
            ensure!(old == new, "rows {rows} cols {cols}: the candidates differ from topk/merge");
            for c in 0..cols {
                let row = &logits[c * rows..(c + 1) * rows];
                let mut want: Vec<(f32, u32)> = row.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
                want.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                for (r, &(v, id)) in want[..64].iter().enumerate() {
                    let at = (c * 64 + r) * 8;
                    let got = (u32::from_le_bytes(new[at..at + 4].try_into()?), u32::from_le_bytes(new[at + 4..at + 8].try_into()?));
                    ensure!(got == (v.to_bits(), id), "rows {rows} col {c} rank {r}: {got:?} against ({v}, {id})");
                }
            }
            println!(
                "{}",
                serde_json::json!({"gpu": gpu, "rows": rows, "cols": cols, "chunks": chunks, "topk_merge_ms": ms[0],
                    "qwen4_head_ms": ms[1], "identical": true})
            );
        }
    }
    unsafe { d.dev.destroy_query_pool(q, None) };
    Ok(())
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
        let last = i + 1 == chunks.len();
        let out = local.forward(x, ple.as_deref(), pos, chunk.len(), last.then_some(&mut caps))?;
        if let (true, super::gpu::Output::Final(h)) = (last, out) {
            local.head_slices(&h, Some(&mut caps))?;
        }
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
    // Per dispatch (TRASHCAN_Q4_PROFILE=1): stage, label, mean ms and mean gap before it over the decode steps.
    let mut disp: Vec<Vec<(String, f64, f64)>> = Vec::new();
    let mut late_disp = Vec::new();
    let profile = |local: &Local, disp: &mut Vec<Vec<(String, f64, f64)>>, pos: usize, n: usize, k: f64| -> Result<()> {
        let mut i = 0;
        for s in &local.stages {
            if let Stage::Gpu(g) = s {
                let p = g.dispatch_profile(pos, n)?;
                if disp.len() <= i {
                    disp.push(p.iter().map(|x| (x.0.clone(), 0., 0.)).collect());
                }
                for (a, b) in disp[i].iter_mut().zip(&p) {
                    a.1 += b.1 / k;
                    a.2 += b.2 / k;
                }
                i += 1;
            }
        }
        Ok(())
    };
    // The late chunk's dispatches (its graph ran last).
    if start > 0 {
        profile(local, &mut late_disp, late_pos, batch.min(start - late_pos), 1.)?;
    }
    // The last timed prefill chunk's dispatches (batches of a few tokens).
    let mut pre_disp = Vec::new();
    if prefill > 0 {
        run(local, &stream[..batch], 0)?;
        profile(local, &mut pre_disp, 0, batch, 1.)?;
    }
    let t0 = std::time::Instant::now();
    for i in 0..decode {
        run(local, &stream[start + i..start + i + 1], start + i)?;
        add(&mut dec, collect(local), decode as f64);
        profile(local, &mut disp, start + i, 1, decode as f64)?;
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
    let mut together = 0.;
    if slices.len() > 1 {
        let t0 = std::time::Instant::now();
        for _ in 0..16 {
            local.head_slices(&h, None)?;
        }
        together = t0.elapsed().as_secs_f64() / 16. * 1e3;
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
        "head_slice_wall_ms": slices, "head_slices_together_wall_ms": together,
        "dispatches": disp.iter().map(|v| v.iter().map(|x| serde_json::json!([x.0, x.1, x.2])).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        "prefill_dispatches": pre_disp.iter().map(|v| v.iter().map(|x| serde_json::json!([x.0, x.1, x.2]))
            .collect::<Vec<_>>()).collect::<Vec<_>>(),
        "late_dispatches": late_disp.iter().map(|v| v.iter().map(|x| serde_json::json!([x.0, x.1, x.2]))
            .collect::<Vec<_>>()).collect::<Vec<_>>()}),
    )
}

/// Diagnostic (results/qwen4/decode-kernels-20261014): what a barrier between dispatches costs on GPU `gpu`. Each case
/// repeats a short sequence of dispatches `iterations` times with timestamps around every dispatch, after each either
/// the engine's barrier (`full`: shader writes visible to later shaders; RADV writes back and invalidates the L2 for it
/// on GFX6), an execution-only barrier (`exec`) or none. Kernels: a one-workgroup activation (`tiny`), the same over
/// 1 MiB (`writer`, 4096 workgroups), the MoE top-k (one wave, 2.7 KB of code) and the Q8_0 matrix-vector kernel on a
/// 6144 x 2560 matrix (`mv`, 16.7 MB). Prints one JSON line per case: per step its mean µs and the mean gap before it.
pub fn barrier_bench(gpu: usize, iterations: u32) -> Result<()> {
    use anyhow::Context;
    use ash::vk;
    let inst = std::rc::Rc::new(crate::gpu::Instance::new()?);
    let pd = *inst.amd_gpus()?.get(gpu).context("no such GPU")?;
    let d = crate::vulkan::Device::new(inst, pd)?;
    let cap = 4096u32;
    let q = unsafe {
        d.dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(cap), None)?
    };
    let params = d.upload(bytemuck::cast_slice(&[0u32, 1, 0, 0]))?;
    let small = d.upload(bytemuck::cast_slice(&[0.5f32; 64]))?;
    let big = d.upload(bytemuck::cast_slice(&vec![0.25f32; 1 << 18]))?;
    let logits: Vec<f32> = (0..513u32).map(|i| ((i * 7919) % 513) as f32 / 64. - 4.).collect();
    let logits = d.upload(bytemuck::cast_slice(&logits))?;
    let ids = d.buffer(64 * 4, false, false, None)?;
    let wts = d.buffer(64 * 4, false, false, None)?;
    let (rows, k) = (6144usize, 2560usize);
    let blocks = rows * k / 32;
    let mat = d.upload(&vec![0u8; blocks * 34])?;
    let x = d.upload(bytemuck::cast_slice(&vec![0.01f32; k]))?;
    let y = d.buffer(rows * 4, false, false, None)?;
    const STREAM: usize = 256 << 20;
    let stream = d.buffer(STREAM, false, false, None)?;
    let sink = d.buffer(4096 * 256 * 4, false, false, None)?;
    let push = |ptr: [u64; 6], u: [u32; 16]| crate::vulkan::Push {
        ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], params.address, 0],
        u,
    };
    let act = |n: usize| {
        let mut u = [0u32; 16];
        (u[0], u[9], u[12], u[15]) = (n as u32, 0, 1f32.to_bits(), 1);
        u
    };
    // A step: dispatch `kind`, then barrier `mode` (0 full, 1 exec, 2 lite, 3 light: `Device::compute_barrier`; 4 none).
    let step = |cmd: vk::CommandBuffer, kind: &str, mode: u8| -> Result<()> {
        match kind {
            "tiny" => d.dispatch(cmd, "qwen4_act", push([small.address, 0, 0, 0, 0, 0], act(64)).defer_barrier(true), 1, 1),
            "writer" => {
                d.dispatch(cmd, "qwen4_act", push([big.address, 0, 0, 0, 0, 0], act(1 << 18)).defer_barrier(true), 4096, 1)
            }
            "topk" => {
                let mut u = [0u32; 16];
                (u[0], u[1], u[2], u[15]) = (512, 1, 10, 1);
                d.dispatch(cmd, "qwen4_topk", push([logits.address, 0, ids.address, wts.address, 0, 0], u).defer_barrier(true), 1, 1)
            }
            "mv" => {
                let mut u = [0u32; 16];
                (u[0], u[1], u[3], u[15]) = (rows as u32, k as u32, (blocks / 2) as u32, 1);
                let p = push([0, x.address, y.address, 0, 0, 0], u).defer_barrier(true);
                d.dispatch_matrix(cmd, "qwen4_q8_mvw4_r4", p, (rows / 16) as u32, 1, &mat)?
            }
            s if s.starts_with("stream") => {
                // stream{2}_{workgroups}: STREAM bytes, uvec4 (or uvec2) loads.
                let (shader, wg) = match s.strip_prefix("stream2_") {
                    Some(w) => ("qwen4_stream_probe2", w),
                    None => ("qwen4_stream_probe", &s[6..]),
                };
                let mut u = [0u32; 16];
                u[0] = (STREAM / 16) as u32;
                let p = push([0, 0, sink.address, 0, 0, 0], u).defer_barrier(true);
                d.dispatch_matrix(cmd, shader, p, wg.parse::<u32>()?, 1, &stream)?
            }
            s if s.starts_with("rows") || s.starts_with("tiled") => {
                // {rows,tiled}{q8,iq4}_{workgroups}: STREAM bytes as matrix rows read the decode kernels' way (4 rows
                // of 2560 bytes, 512 per step; 8 rows of 1280 bytes, 256 per step), or the same steps contiguous.
                let (pat, wg) = s.split_once('_').context("pattern")?;
                let (shader, rowbytes) = if pat.ends_with("q8") { ("qwen4_stream_probe2", 2560) } else { ("qwen4_stream_probe1", 1280) };
                let mut u = [0u32; 16];
                (u[0], u[1], u[9]) = ((STREAM / 16) as u32, rowbytes, if pat.starts_with("rows") { 1 } else { 2 });
                let p = push([0, 0, sink.address, 0, 0, 0], u).defer_barrier(true);
                d.dispatch_matrix(cmd, shader, p, wg.parse::<u32>()?, 1, &stream)?
            }
            _ => unreachable!(),
        }
        if mode < 4 {
            d.compute_barrier(cmd, mode);
        }
        Ok(())
    };
    let cases: [(&str, &[&str], u8, u32); 35] = [
        ("rows-q8-640", &["rowsq8_640"], 0, 6),
        ("tiled-q8-640", &["tiledq8_640"], 0, 6),
        ("rows-q8-1280", &["rowsq8_1280"], 0, 6),
        ("tiled-q8-1280", &["tiledq8_1280"], 0, 6),
        ("rows-iq4-640", &["rowsiq4_640"], 0, 6),
        ("tiled-iq4-640", &["tilediq4_640"], 0, 6),
        ("rows-iq4-1280", &["rowsiq4_1280"], 0, 6),
        ("tiled-iq4-1280", &["tilediq4_1280"], 0, 6),
        ("stream-320", &["stream320"], 0, 6),
        ("stream-640", &["stream640"], 0, 6),
        ("stream-1280", &["stream1280"], 0, 6),
        ("stream-2560", &["stream2560"], 0, 6),
        ("stream2-640", &["stream2_640"], 0, 6),
        ("stream2-1280", &["stream2_1280"], 0, 6),
        ("stream2-2560", &["stream2_2560"], 0, 6),
        ("tiny-full", &["tiny"], 0, iterations),
        ("tiny-exec", &["tiny"], 1, iterations),
        ("tiny-lite", &["tiny"], 2, iterations),
        ("tiny-light", &["tiny"], 3, iterations),
        ("tiny-none", &["tiny"], 4, iterations),
        ("writer-tiny-full", &["writer", "tiny"], 0, iterations / 2),
        ("writer-tiny-lite", &["writer", "tiny"], 2, iterations / 2),
        ("topk-full", &["topk"], 0, iterations),
        ("topk-exec", &["topk"], 1, iterations),
        ("topk-lite", &["topk"], 2, iterations),
        ("topk-light", &["topk"], 3, iterations),
        ("mv-full", &["mv"], 0, iterations / 4),
        ("mv-lite", &["mv"], 2, iterations / 4),
        ("mv-topk-full", &["mv", "topk"], 0, iterations / 4),
        ("mv-topk-exec", &["mv", "topk"], 1, iterations / 4),
        ("mv-topk-lite", &["mv", "topk"], 2, iterations / 4),
        ("mv-topk-light", &["mv", "topk"], 3, iterations / 4),
        ("mv-tiny-full", &["mv", "tiny"], 0, iterations / 4),
        ("mv-tiny-exec", &["mv", "tiny"], 1, iterations / 4),
        ("mv-tiny-lite", &["mv", "tiny"], 2, iterations / 4),
    ];
    let us = d.period as f64 / 1e3;
    for (name, seq, mode, n) in cases {
        let n = n.min(cap / (2 * seq.len() as u32)).max(2);
        let cmd = d.begin()?;
        unsafe { d.dev.cmd_reset_query_pool(cmd, q, 0, cap) };
        d.barrier(cmd);
        let mut i = 0;
        for _ in 0..n {
            for kind in seq {
                unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, i) };
                step(cmd, kind, mode)?;
                unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, i + 1) };
                i += 2;
            }
        }
        d.finish(cmd)?;
        d.run_reusable(cmd)?;
        d.run(cmd)?;
        let mut t = vec![0u64; i as usize];
        unsafe { d.dev.get_query_pool_results(q, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
        // The first repetition is left out (cold).
        let per = 2 * seq.len();
        let mut out = Vec::new();
        for (s, kind) in seq.iter().enumerate() {
            let (mut dur, mut gap, mut cnt) = (0f64, 0f64, 0f64);
            for r in 1..n as usize {
                let a = r * per + 2 * s;
                dur += t[a + 1].wrapping_sub(t[a]) as f64 * us;
                gap += t[a].wrapping_sub(t[a - 1]) as f64 * us;
                cnt += 1.;
            }
            out.push(serde_json::json!({"kernel": kind, "us": dur / cnt, "gap_us": gap / cnt}));
        }
        let total = t[i as usize - 1].wrapping_sub(t[per - 1]) as f64 * us / (n - 1) as f64;
        println!("{}", serde_json::json!({"gpu": gpu, "case": name, "iterations": n, "us_per_iteration": total, "steps": out}));
    }
    // Which caches a barrier invalidates (qwen4_cache_probe): fill a buffer with 1, read it in every workgroup through
    // the vector and scalar caches, write 2 to it from one workgroup after an execution-only barrier, then the barrier
    // under test, then read it again: workgroups that still see 1 read a stale vector-cache line or scalar-cache entry.
    // With 4096 words every workgroup reads its own 1 KB of 16 KB, with 256 words all read the same 1 KB (so that
    // every CU has it cached, as in `Device::light_barriers_ok`). Each barrier is also timed after a one-workgroup
    // dispatch, as above.
    let probes = [(4096usize, d.buffer(4096 * 4, false, false, None)?), (256, d.buffer(256 * 4, false, false, None)?)];
    let res = d.buffer(2 * 1280 * 4, true, false, None)?;
    let flags = vk::AccessFlags::empty();
    let variants: [(&str, vk::AccessFlags, vk::AccessFlags, vk::PipelineStageFlags); 9] = [
        ("full", vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER),
        ("exec", flags, flags, vk::PipelineStageFlags::COMPUTE_SHADER),
        ("none-uniform", flags, vk::AccessFlags::UNIFORM_READ, vk::PipelineStageFlags::COMPUTE_SHADER),
        ("none-shader_read", flags, vk::AccessFlags::SHADER_READ, vk::PipelineStageFlags::COMPUTE_SHADER),
        ("write-none", vk::AccessFlags::SHADER_WRITE, flags, vk::PipelineStageFlags::COMPUTE_SHADER),
        ("none-indirect", flags, vk::AccessFlags::INDIRECT_COMMAND_READ, vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::COMPUTE_SHADER),
        ("none-transfer_read", flags, vk::AccessFlags::TRANSFER_READ, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER),
        ("none-memory_read", flags, vk::AccessFlags::MEMORY_READ, vk::PipelineStageFlags::COMPUTE_SHADER),
        ("write-uniform", vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::UNIFORM_READ, vk::PipelineStageFlags::COMPUTE_SHADER),
    ];
    let barrier = |cmd: vk::CommandBuffer, src: vk::AccessFlags, dst: vk::AccessFlags, stage: vk::PipelineStageFlags| {
        let b = [vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(dst)];
        let b: &[vk::MemoryBarrier] = if src.is_empty() && dst.is_empty() { &[] } else { &b };
        unsafe {
            d.dev.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, stage, vk::DependencyFlags::empty(), b, &[], &[])
        };
    };
    for (name, src, dst, stage) in variants {
        let mut stale = Vec::new();
        for (words, probe) in &probes {
            let probe_push = |kind: u32, value: f32| {
                let mut u = [0u32; 16];
                (u[0], u[9], u[12], u[15]) = (*words as u32, kind, value.to_bits(), 1);
                push([probe.address, probe.address, res.address, 0, 0, probe.address], u).defer_barrier(true)
            };
            let cmd = d.begin()?;
            d.barrier(cmd);
            d.dispatch(cmd, "qwen4_cache_probe", probe_push(1, 1.0), 1, 1);
            d.compute_barrier(cmd, 0);
            d.dispatch(cmd, "qwen4_cache_probe", probe_push(0, 0.0), 1280, 1);
            d.compute_barrier(cmd, 1);
            d.dispatch(cmd, "qwen4_cache_probe", probe_push(1, 2.0), 1, 1);
            barrier(cmd, src, dst, stage);
            d.dispatch(cmd, "qwen4_cache_probe", probe_push(0, 0.0), 1280, 1);
            d.host_barrier(cmd, true);
            d.finish(cmd)?;
            d.run(cmd)?;
            let r: Vec<f32> = bytemuck::cast_slice(&res.read(2 * 1280 * 4)?).to_vec();
            stale.push(r.chunks(2).filter(|c| c[0] != 128.0).count());
            stale.push(r.chunks(2).filter(|c| c[1] != 2.0).count());
        }
        // Timing: a one-workgroup dispatch followed by this barrier.
        let n = 400u32;
        let cmd = d.begin()?;
        unsafe { d.dev.cmd_reset_query_pool(cmd, q, 0, 2) };
        d.barrier(cmd);
        for k in 0..n {
            if k == 1 {
                unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0) };
            }
            d.dispatch(cmd, "qwen4_act", push([small.address, 0, 0, 0, 0, 0], act(64)).defer_barrier(true), 1, 1);
            barrier(cmd, src, dst, stage);
        }
        unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1) };
        d.finish(cmd)?;
        d.run(cmd)?;
        let mut t = [0u64; 2];
        unsafe { d.dev.get_query_pool_results(q, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
        let per = t[1].wrapping_sub(t[0]) as f64 * us / (n - 1) as f64;
        println!(
            "{}",
            serde_json::json!({"gpu": gpu, "barrier": name, "workgroups": 1280, "vector_stale": stale[0],
                "scalar_stale": stale[1], "vector_stale_1k": stale[2], "scalar_stale_1k": stale[3],
                "us_per_tiny_dispatch": per})
        );
    }
    unsafe { d.dev.destroy_query_pool(q, None) };
    Ok(())
}
