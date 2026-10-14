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

/// One step of the bounded gate's single-GPU check on GPU `gpu` (`qwen4 gate-check`). The gate word lives in host
/// memory, `cached` (host-cached, coherent: the engine's host buffers) or not (host-visible, coherent, not cached);
/// one lane polls it (`shader`: `qwen4_gate_clock`, volatile loads, or `qwen4_gate_nt`, nontemporal loads) until it
/// reads 1, for at most `bound_ms` (shader clock at 838.8 MHz, with a poll count behind it that alone ends within
/// `bound_ms` at the fastest poll rate seen, 0.40 µs). Steps:
/// - `release`: `reps` runs, each opened by the host 200 µs after the command processor reached the gate; stops at the
///   first run whose lane did not see the write. Release latency: host write to the GPU timestamp after the gate
///   (calibrated timestamps: `TRASHCAN_TP_PROFILE=1`), and host write to the host seeing the end event.
/// - `never`: one run never opened, which must end by its bound.
/// One JSON line.
pub fn gate_check(gpu: usize, step: &str, cached: bool, shader: &str, bound_ms: f64, reps: usize) -> Result<()> {
    use anyhow::{bail, ensure, Context};
    use ash::vk;
    use serde_json::json;
    use std::time::{Duration, Instant};
    ensure!((1.0..=100.0).contains(&bound_ms), "bounds of 1-100 ms only");
    ensure!(matches!(shader, "qwen4_gate_clock" | "qwen4_gate_nt"), "shader qwen4_gate_clock or qwen4_gate_nt");
    const TICKS_PER_S: f64 = 838.8e6;
    const FASTEST_POLL_S: f64 = 0.40e-6;
    let inst = std::rc::Rc::new(crate::gpu::Instance::new()?);
    let pd = *inst.amd_gpus()?.get(gpu).context("no such GPU")?;
    let d = crate::vulkan::Device::new(inst, pd)?;
    ensure!(d.shader_clock, "the device has no shader clock");
    let (gate, memory) = if cached {
        (d.buffer(64, true, false, None)?, "host-cached, coherent".to_string())
    } else {
        let (b, f) = d.uncached_host_buffer(64)?;
        (b, format!("{f:?}"))
    };
    let q = unsafe {
        d.dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(3), None)?
    };
    let event = || unsafe { d.dev.create_event(&vk::EventCreateInfo::default(), None) };
    let (started, done) = (event()?, event()?);
    let fence = unsafe { d.dev.create_fence(&vk::FenceCreateInfo::default(), None)? };
    let word = |i: usize| unsafe { std::ptr::read_volatile((gate.ptr as *const u32).add(i)) };
    let open = || unsafe {
        std::ptr::write_volatile(gate.ptr as *mut u32, 1);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    };
    let polls = (bound_ms / 1e3 / FASTEST_POLL_S) as u32;
    let ticks = ((bound_ms / 1e3 * TICKS_PER_S) as u64 >> 16) as u32;
    let cmd = d.begin()?;
    unsafe {
        d.dev.cmd_reset_query_pool(cmd, q, 0, 3);
        d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
        d.dev.cmd_set_event(cmd, started, vk::PipelineStageFlags::TOP_OF_PIPE);
    }
    let mut p = crate::vulkan::Push::default();
    p.ptr[0] = gate.address;
    (p.u[0], p.u[1]) = (polls, ticks);
    d.dispatch(cmd, shader, p, 1, 1);
    unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1) };
    // As a stage would before reading the input the host wrote.
    d.barrier(cmd);
    d.host_barrier(cmd, false);
    unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 2) };
    d.host_barrier(cmd, true);
    unsafe { d.dev.cmd_set_event(cmd, done, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER) };
    d.finish(cmd)?;
    // One run: (how, polls, gate ms on the GPU, release to GPU past the gate µs, release to the end seen µs).
    let run = |release: bool| -> Result<(u32, u32, f64, Option<f64>, Option<f64>)> {
        gate.write(&[0u8; 64])?;
        // Drains a write-combined mapping too.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        unsafe {
            d.dev.reset_event(started)?;
            d.dev.reset_event(done)?;
            d.dev.reset_fences(&[fence])?;
        }
        let t0 = Instant::now();
        d.submit(cmd, None, None, fence)?;
        while !unsafe { d.dev.get_event_status(started)? } {
            ensure!(t0.elapsed() < Duration::from_secs(1), "the gate's command buffer did not start");
            std::hint::spin_loop();
        }
        let mut released = None;
        if release {
            let at = Instant::now();
            while at.elapsed() < Duration::from_micros(200) {
                std::hint::spin_loop();
            }
            released = Some((Instant::now(), crate::vulkan::host_ns()));
            open();
        }
        let mut seen = None;
        let wait = Instant::now();
        while wait.elapsed() < Duration::from_secs_f64(bound_ms / 1e3 + 0.5) {
            if unsafe { d.dev.get_event_status(done)? } {
                seen = Some(Instant::now());
                break;
            }
            std::hint::spin_loop();
        }
        unsafe { d.dev.wait_for_fences(&[fence], true, 10_000_000_000)? };
        let mut t = [0u64; 3];
        unsafe { d.dev.get_query_pool_results(q, 0, &mut t, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
        let period = d.period as f64;
        let gate_ms = (t[1] - t[0]) as f64 * period / 1e6;
        let gpu_us = match (released, d.calibrate()) {
            (Some((_, host)), Some((tick, now))) => {
                let past_gate = now as f64 - (tick as i64 - t[1] as i64) as f64 * period;
                Some((past_gate - host as f64) / 1e3)
            }
            _ => None,
        };
        let host_us = released.zip(seen).map(|((r, _), s)| (s - r).as_secs_f64() * 1e6);
        Ok((word(1), word(2), gate_ms, gpu_us, host_us))
    };
    let median = |mut v: Vec<f64>| -> Option<f64> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get(v.len() / 2).copied()
    };
    let p99 = |mut v: Vec<f64>| -> Option<f64> {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get((v.len() * 99 / 100).min(v.len().saturating_sub(1))).copied()
    };
    let base = json!({"gpu": gpu, "step": step, "shader": shader, "memory": memory, "bound_ms": bound_ms,
        "polls_bound": polls, "ticks_bound": (ticks as u64) << 16});
    let result = (|| -> Result<serde_json::Value> {
        match step {
            "release" => {
                let (mut gpu_us, mut host_us, mut poll_counts) = (Vec::new(), Vec::new(), Vec::new());
                for i in 0..reps {
                    let (how, polls, gate_ms, g, h) = run(true)?;
                    if how != 1 {
                        return Ok(json!({"seen": i, "reps": reps, "failed_rep": {"how": how, "polls": polls,
                            "gate_ms": gate_ms}}));
                    }
                    gpu_us.extend(g);
                    host_us.extend(h);
                    poll_counts.push(polls as f64);
                }
                Ok(json!({"seen": reps, "reps": reps,
                    "release_to_gpu_past_gate_us": {"median": median(gpu_us.clone()), "p99": p99(gpu_us)},
                    "release_to_end_seen_us": {"median": median(host_us.clone()), "p99": p99(host_us)},
                    "polls_median": median(poll_counts)}))
            }
            "never" => {
                let (how, polls, gate_ms, _, _) = run(false)?;
                Ok(json!({"how": how, "polls": polls, "gate_ms": gate_ms,
                    "ticks": word(3) as u64 | (word(4) as u64) << 32, "ended_by_bound": how == 2 || how == 3}))
            }
            _ => bail!("step release or never"),
        }
    })();
    // Whatever happened, nothing may wait on the gate.
    open();
    unsafe {
        let _ = d.dev.device_wait_idle();
        d.dev.free_command_buffers(d.pool, &[cmd]);
        d.dev.destroy_fence(fence, None);
        d.dev.destroy_event(started, None);
        d.dev.destroy_event(done, None);
        d.dev.destroy_query_pool(q, None);
    }
    let mut out = base;
    for (k, v) in result?.as_object().context("result")? {
        out[k] = v.clone();
    }
    println!("{out}");
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
            .collect::<Vec<_>>()).collect::<Vec<_>>()}),
    )
}
