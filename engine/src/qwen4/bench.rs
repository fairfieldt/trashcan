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

/// How the host makes its write of the gate word visible (`qwen4 gate-check --fence`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HostFence {
    /// A full fence (x86 `mfence`, which also drains write-combining buffers), as in rounds 1 and 2.
    Mfence,
    /// `sfence`: drains the write-combining buffers (for non-cached, write-combined memory).
    Sfence,
    /// `clflush` of the gate's line, then `mfence` (for host-cached memory).
    Clflush,
}

impl std::str::FromStr for HostFence {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "mfence" => Self::Mfence,
            "sfence" => Self::Sfence,
            "clflush" => Self::Clflush,
            _ => anyhow::bail!("fence mfence, sfence or clflush"),
        })
    }
}

/// Write 1 to the gate word at `ptr` and make the write visible as `fence` says.
fn open_gate(ptr: *mut u32, fence: HostFence) {
    unsafe {
        std::ptr::write_volatile(ptr, 1);
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_clflush, _mm_mfence, _mm_sfence};
            match fence {
                HostFence::Mfence => _mm_mfence(),
                HostFence::Sfence => _mm_sfence(),
                HostFence::Clflush => {
                    _mm_clflush(ptr as *const u8);
                    _mm_mfence();
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = fence;
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }
    }
}

/// Percentiles of `v` (nearest rank): min, p10, p50, p90, p99, max.
fn spread(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| s[((s.len() as f64 * q) as usize).min(s.len() - 1)];
    serde_json::json!({"n": s.len(), "min": s[0], "p10": at(0.1), "p50": at(0.5), "p90": at(0.9), "p99": at(0.99),
        "max": s[s.len() - 1]})
}

/// The bounded gate's single-GPU check (`qwen4 gate-check`), one step per run.
pub struct GateCheck {
    pub gpu: usize,
    /// `release`: `reps` runs, each opened by the host 200 µs after the command processor reached the gate (stops
    /// after `max_fail` runs whose gate did not see the write); `never`: one run never opened, which must end by its
    /// bound; `open`: `reps` runs with the gate already open (the cost of a gate per graph); `busy`: `reps` runs never
    /// opened, back to back (the GPU's power while gates wait).
    pub step: String,
    /// The gate word's memory: host-cached and coherent (the engine's host buffers), or host-visible, coherent and
    /// not cached (RADV: write-combined GTT).
    pub cached: bool,
    /// `qwen4_gate_clock` (volatile loads), `qwen4_gate_nt` (nontemporal loads), `qwen4_gate_atomic` (device-scope
    /// atomic loads), `qwen4_gate_sweep` / `qwen4_gate_sweep_nt` (L2 sweep between polls), or `none` (no gate, for
    /// the `open` step's baseline).
    pub shader: String,
    pub bound_ms: f64,
    pub reps: usize,
    pub fence: HostFence,
    /// The sweep shaders read `sweep_kib` KiB of device-local memory, one dword every `stride` bytes, between polls.
    pub sweep_kib: usize,
    pub stride: usize,
    pub max_fail: usize,
    /// The release step opens each gate 200 µs plus a uniformly random 0-`jitter_us` µs after it started, so that
    /// releases fall at every phase of the sweep and poll cycle.
    pub jitter_us: u64,
}

/// One run of the gate check: how the gate ended and what it measured.
struct GateRun {
    how: u32,
    polls: u32,
    /// Shader-clock ticks: in the gate, in its poll loads, in its sweeps; fewest and most ticks of one poll load.
    ticks: u64,
    load_ticks: u64,
    fewest: u32,
    most: u32,
    sweep_ticks: u64,
    /// GPU µs from the timestamp before the gate to the one after it, and to the one after the input barriers.
    gate_us: f64,
    graph_us: f64,
    /// Release (host write) to: the GPU timestamp after the gate, the gate's own clock at the open poll (from the
    /// timestamp before the gate, so without the dispatch's launch), the host seeing the end event.
    to_gpu_us: Option<f64>,
    to_shader_us: Option<f64>,
    to_host_us: Option<f64>,
    /// Host µs of the write and its fence; started event seen to end event seen.
    write_us: Option<f64>,
    started_to_done_us: Option<f64>,
}

/// One step of the bounded gate's single-GPU check on GPU `c.gpu`. The gate word (word 0 of a page) lives in host
/// memory; the gate polls it until it reads 1, for at most `bound_ms` (shader clock at 838.8 MHz, with a poll count
/// behind it that alone ends within `bound_ms` at the fastest poll rate seen, 0.40 µs). The command buffer is a gated
/// stage's: a timestamp, an event the command processor sets just before the gate, the gate, a timestamp, the
/// barriers a stage needs before reading host input, a timestamp and an end event. Release latencies need calibrated
/// timestamps (`TRASHCAN_TP_PROFILE=1`). One JSON line.
pub fn gate_check(c: &GateCheck) -> Result<()> {
    use anyhow::{bail, ensure, Context};
    use ash::vk;
    use serde_json::json;
    use std::time::{Duration, Instant};
    const TICKS_PER_S: f64 = 838.8e6;
    const FASTEST_POLL_S: f64 = 0.40e-6;
    let sweep = c.shader.starts_with("qwen4_gate_sweep");
    ensure!((1.0..=100.0).contains(&c.bound_ms), "bounds of 1-100 ms only");
    ensure!(
        matches!(c.shader.as_str(), "qwen4_gate_clock" | "qwen4_gate_nt" | "qwen4_gate_atomic" | "none") || sweep,
        "shader qwen4_gate_clock, qwen4_gate_nt, qwen4_gate_atomic, qwen4_gate_sweep, qwen4_gate_sweep_nt or none"
    );
    ensure!(c.shader != "none" || c.step == "open", "no gate only for the open step");
    ensure!(c.stride >= 64 && c.stride % 64 == 0, "stride a multiple of 64 bytes");
    let lines = c.sweep_kib * 1024 / c.stride;
    ensure!(!sweep || lines % 512 == 0, "the sweep reads a multiple of 512 lines");
    ensure!(c.sweep_kib <= 64 << 10, "sweeps of at most 64 MiB");
    let inst = std::rc::Rc::new(crate::gpu::Instance::new()?);
    let pd = *inst.amd_gpus()?.get(c.gpu).context("no such GPU")?;
    let d = crate::vulkan::Device::new(inst, pd)?;
    ensure!(d.shader_clock, "the device has no shader clock");
    let (gate, memory) = if c.cached {
        (d.buffer(4096, true, false, None)?, "host-cached, coherent".to_string())
    } else {
        let (b, f) = d.uncached_host_buffer(4096)?;
        (b, format!("{f:?}"))
    };
    // Device-local; never written, so it reads as whatever the allocation held.
    let lines_buf = if sweep && lines > 0 { Some(d.buffer(lines * c.stride, false, false, None)?) } else { None };
    let q = unsafe {
        d.dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(3), None)?
    };
    let event = || unsafe { d.dev.create_event(&vk::EventCreateInfo::default(), None) };
    let (started, done) = (event()?, event()?);
    let fence = unsafe { d.dev.create_fence(&vk::FenceCreateInfo::default(), None)? };
    let word = |i: usize| unsafe { std::ptr::read_volatile((gate.ptr as *const u32).add(i)) };
    let open = || open_gate(gate.ptr as *mut u32, c.fence);
    let ticks = ((c.bound_ms / 1e3 * TICKS_PER_S) as u64 >> 16) as u32;
    // The gate's command buffer with a bound of `polls` polls (and `bound_ms` of shader clock).
    let record = |polls: u32| -> Result<vk::CommandBuffer> {
        let cmd = d.begin()?;
        unsafe {
            d.dev.cmd_reset_query_pool(cmd, q, 0, 3);
            d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
            d.dev.cmd_set_event(cmd, started, vk::PipelineStageFlags::TOP_OF_PIPE);
        }
        if c.shader != "none" {
            let mut p = crate::vulkan::Push::default();
            p.ptr[0] = gate.address;
            p.ptr[1] = lines_buf.as_ref().map_or(0, |b| b.address);
            (p.u[0], p.u[1], p.u[2], p.u[3]) = (polls, ticks, lines as u32, (c.stride / 4) as u32);
            d.dispatch(cmd, &c.shader, p, 1, 1);
        }
        unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1) };
        // As a stage would before reading the input the host wrote.
        d.barrier(cmd);
        d.host_barrier(cmd, false);
        unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 2) };
        d.host_barrier(cmd, true);
        unsafe { d.dev.cmd_set_event(cmd, done, vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER) };
        d.finish(cmd)?;
        Ok(cmd)
    };
    // Where the gate writes how it ended: words 1-4 (the first line) or, for the sweep shaders, words 16-25.
    let at = if sweep { 16 } else { 1 };
    let seed = std::cell::Cell::new(crate::vulkan::host_ns() | 1);
    let jitter = || {
        let mut x = seed.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        seed.set(x);
        if c.jitter_us == 0 { 0 } else { x % (c.jitter_us + 1) }
    };
    let run = |cmd: vk::CommandBuffer, release: bool, preopen: bool| -> Result<GateRun> {
        gate.write(&[0u8; 128])?;
        if preopen {
            open();
        }
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
        let start_seen = Instant::now();
        let mut released = None;
        let mut write_us = None;
        if release {
            let delay = Duration::from_micros(200 + jitter());
            while start_seen.elapsed() < delay {
                std::hint::spin_loop();
            }
            let (when, ns) = (Instant::now(), crate::vulkan::host_ns());
            open();
            write_us = Some(when.elapsed().as_secs_f64() * 1e6);
            released = Some((when, ns));
        }
        let mut seen = None;
        let wait = Instant::now();
        while wait.elapsed() < Duration::from_secs_f64(c.bound_ms / 1e3 + 0.5) {
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
        let (how, polls, ticks) = if c.shader == "none" {
            (0, 0, 0)
        } else {
            (word(at), word(at + 1), word(at + 2) as u64 | (word(at + 3) as u64) << 32)
        };
        let (load_ticks, fewest, most, sweep_ticks) = if sweep {
            (word(20) as u64 | (word(21) as u64) << 32, word(22), word(23), word(24) as u64 | (word(25) as u64) << 32)
        } else {
            (0, 0, 0, 0)
        };
        let gate_us = (t[1] - t[0]) as f64 * period / 1e3;
        // The shader clock must agree with the command processor's timestamps; otherwise its bound means nothing.
        let ticks_us = ticks as f64 / TICKS_PER_S * 1e6;
        ensure!(
            c.shader == "none" || (ticks_us - gate_us).abs() <= 0.2 * gate_us + 30.0,
            "the shader clock disagrees with the GPU timestamps ({ticks_us:.1} against {gate_us:.1} us): stop"
        );
        let (mut to_gpu_us, mut to_shader_us) = (None, None);
        if let (Some((_, host)), Some((tick, now))) = (released, d.calibrate()) {
            let host_of = |gpu: u64| now as f64 - (tick as i64 - gpu as i64) as f64 * period;
            to_gpu_us = Some((host_of(t[1]) - host as f64) / 1e3);
            to_shader_us = Some((host_of(t[0]) + ticks as f64 / TICKS_PER_S * 1e9 - host as f64) / 1e3);
        }
        Ok(GateRun {
            how,
            polls,
            ticks,
            load_ticks,
            fewest,
            most,
            sweep_ticks,
            gate_us,
            graph_us: (t[2] - t[0]) as f64 * period / 1e3,
            to_gpu_us,
            to_shader_us,
            to_host_us: released.zip(seen).map(|((r, _), s)| (s - r).as_secs_f64() * 1e6),
            write_us,
            started_to_done_us: seen.map(|s| (s - start_seen).as_secs_f64() * 1e6),
        })
    };
    let us = |t: u64| t as f64 / TICKS_PER_S * 1e6;
    // Per poll: the poll load's µs (mean, fewest, most) and the sweep's µs (mean), for the sweep shaders.
    let poll_stats = |r: &GateRun| -> serde_json::Value {
        let n = r.polls.max(1) as f64;
        let mut v = json!({"how": r.how, "polls": r.polls, "gate_ms": r.gate_us / 1e3, "ticks": r.ticks,
            "us_per_poll": us(r.ticks) / n});
        if sweep {
            v["load_us_mean"] = json!(us(r.load_ticks) / (n + 1.0));
            v["load_us_fewest"] = json!(us(r.fewest as u64));
            v["load_us_most"] = json!(us(r.most as u64));
            v["sweep_us_mean"] = json!(us(r.sweep_ticks) / n);
        }
        v
    };
    let round = |v: &[f64]| v.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>();
    // The poll bound: `bound_ms` at the fastest poll seen (0.40 us). A sweep shader polls far slower, so its bound
    // is also `bound_ms` at the rate a 16-poll run shows in the command processor's timestamps: the poll count alone
    // then ends the gate within about its bound even if the shader clock failed.
    let mut polls = (c.bound_ms / 1e3 / FASTEST_POLL_S) as u32;
    let mut calibration = serde_json::Value::Null;
    if sweep && lines > 0 {
        let cal = record(16)?;
        let r = run(cal, false, false);
        open();
        unsafe {
            let _ = d.dev.device_wait_idle();
            d.dev.free_command_buffers(d.pool, &[cal]);
        }
        let r = r?;
        ensure!(r.how == 2 && r.polls == 16, "the 16-poll run ended with how {} after {} polls", r.how, r.polls);
        let per_poll_s = r.gate_us / 1e6 / 16.0;
        polls = polls.min((c.bound_ms / 1e3 / per_poll_s) as u32).max(1);
        calibration = json!({"polls": 16, "gate_us": r.gate_us, "us_per_poll": per_poll_s * 1e6, "ticks": r.ticks});
    }
    let cmd = record(polls)?;
    let base = json!({"gpu": c.gpu, "step": c.step, "jitter_us": c.jitter_us, "shader": c.shader, "memory": memory, "fence": format!("{:?}", c.fence),
        "bound_ms": c.bound_ms, "polls_bound": polls, "ticks_bound": (ticks as u64) << 16,
        "sweep": if sweep { json!({"kib": c.sweep_kib, "stride": c.stride, "lines": lines}) } else { serde_json::Value::Null },
        "calibration": calibration});
    let result = (|| -> Result<serde_json::Value> {
        match c.step.as_str() {
            "release" => {
                let (mut gpu, mut shader, mut host, mut write, mut npolls, mut load) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let (mut failed, mut runs, mut first_fail) = (0, 0, serde_json::Value::Null);
                for _ in 0..c.reps {
                    let r = run(cmd, true, false)?;
                    runs += 1;
                    if r.how != 1 {
                        if failed == 0 {
                            first_fail = poll_stats(&r);
                        }
                        failed += 1;
                        if failed >= c.max_fail {
                            break;
                        }
                        continue;
                    }
                    gpu.extend(r.to_gpu_us);
                    shader.extend(r.to_shader_us);
                    host.extend(r.to_host_us);
                    write.extend(r.write_us);
                    npolls.push(r.polls as f64);
                    if sweep {
                        load.push(us(r.load_ticks) / (r.polls as f64 + 1.0));
                    }
                }
                Ok(json!({"runs": runs, "seen": runs - failed, "failed": failed, "first_failed_run": first_fail,
                    "release_to_gpu_past_gate_us": spread(&gpu), "release_to_shader_open_us": spread(&shader),
                    "release_to_end_seen_us": spread(&host), "write_and_fence_us": spread(&write),
                    "polls": spread(&npolls), "load_us_mean": spread(&load),
                    "raw_release_to_gpu_past_gate_us": round(&gpu), "raw_release_to_shader_open_us": round(&shader)}))
            }
            "never" => {
                let r = run(cmd, false, false)?;
                let mut v = poll_stats(&r);
                v["ended_by_bound"] = json!(r.how == 2 || r.how == 3);
                Ok(v)
            }
            "open" => {
                let (mut gate_us, mut graph_us, mut host_us, mut npolls) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let mut not_open = 0;
                for _ in 0..c.reps {
                    let r = run(cmd, false, true)?;
                    if c.shader != "none" && r.how != 1 {
                        not_open += 1;
                    }
                    gate_us.push(r.gate_us);
                    graph_us.push(r.graph_us);
                    host_us.extend(r.started_to_done_us);
                    npolls.push(r.polls as f64);
                }
                Ok(json!({"runs": c.reps, "not_open": not_open, "gate_gpu_us": spread(&gate_us),
                    "graph_gpu_us": spread(&graph_us), "started_to_done_host_us": spread(&host_us), "polls": spread(&npolls)}))
            }
            "busy" => {
                // Never-released gates back to back (each ends by its bound), for the GPU's power while it waits.
                let utc = || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
                let (start, t) = (utc(), Instant::now());
                let (mut ended, mut opened, mut polls) = (0, 0, Vec::new());
                for _ in 0..c.reps {
                    let r = run(cmd, false, false)?;
                    if r.how == 2 || r.how == 3 {
                        ended += 1;
                    } else {
                        opened += 1;
                    }
                    polls.push(us(r.ticks) / r.polls.max(1) as f64);
                }
                Ok(json!({"runs": c.reps, "ended_by_bound": ended, "not_ended_by_bound": opened,
                    "seconds": t.elapsed().as_secs_f64(), "utc_start": start, "utc_end": utc(), "us_per_poll": spread(&polls)}))
            }
            _ => bail!("step release, never, open or busy"),
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

/// The gate shaders' machine code (`qwen4 gate-isa`): each shader compiled as the engine compiles it (its pipeline
/// layout, 128 bytes of push constants), with RADV's statistics and internal representations
/// (VK_KHR_pipeline_executable_properties), on GPU `gpu`, or on RADV's null device when `RADV_FORCE_FAMILY` is set
/// (no GPU access). Prints text.
pub fn gate_isa(gpu: usize, names: &[String]) -> Result<()> {
    use anyhow::{ensure, Context};
    use ash::vk;
    let inst = crate::gpu::Instance::new()?;
    let pd = if std::env::var_os("RADV_FORCE_FAMILY").is_some() {
        unsafe { inst.instance.enumerate_physical_devices()? }
            .into_iter()
            .find(|&pd| unsafe { inst.instance.get_physical_device_properties(pd) }.vendor_id == 0x1002)
            .context("no RADV null device")?
    } else {
        *inst.amd_gpus()?.get(gpu).context("no such GPU")?
    };
    let props = unsafe { inst.instance.get_physical_device_properties(pd) };
    let driver = {
        let mut drv = vk::PhysicalDeviceDriverProperties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut drv);
        unsafe { inst.instance.get_physical_device_properties2(pd, &mut p2) };
        format!("{:?} {:?}", drv.driver_name_as_c_str().unwrap_or_default(), drv.driver_info_as_c_str().unwrap_or_default())
    };
    println!("# device {:?} ({driver})", props.device_name_as_c_str().unwrap_or_default());
    let family = inst.compute_family(pd).map(|f| f.0).unwrap_or(0);
    let mut pep = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
    let mut clock = vk::PhysicalDeviceShaderClockFeaturesKHR::default().shader_subgroup_clock(true);
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true)
        .vulkan_memory_model(true)
        .vulkan_memory_model_device_scope(true);
    let features = vk::PhysicalDeviceFeatures::default().shader_int16(true).shader_int64(true);
    let extensions = [ash::khr::pipeline_executable_properties::NAME.as_ptr(), ash::khr::shader_clock::NAME.as_ptr()];
    let priority = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priority)];
    let ci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut f11)
        .push_next(&mut f12)
        .push_next(&mut pep)
        .push_next(&mut clock);
    let dev = unsafe { inst.instance.create_device(pd, &ci, None)? };
    let one = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    let eight: Vec<_> = (0..8).map(|b| vk::DescriptorSetLayoutBinding { binding: b, ..one[0] }).collect();
    let set = |b: &[vk::DescriptorSetLayoutBinding]| unsafe {
        dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(b), None)
    };
    let sets = [set(&one)?, set(&one)?, set(&eight)?];
    let range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(128)];
    let layout = unsafe {
        dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets).push_constant_ranges(&range), None)?
    };
    let pe = ash::khr::pipeline_executable_properties::Device::new(&inst.instance, &dev);
    for name in names {
        let bytes = crate::vulkan::gate_spirv(name).with_context(|| format!("no gate shader {name}"))?;
        ensure!(!bytes.is_empty(), "shaders missing: build with glslc");
        let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
        let module = unsafe { dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)? };
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let ci = [vk::ComputePipelineCreateInfo::default()
            .flags(
                vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR,
            )
            .stage(stage)
            .layout(layout)];
        let pipe = unsafe { dev.create_compute_pipelines(vk::PipelineCache::null(), &ci, None) }
            .map_err(|(_, e)| anyhow::anyhow!("{name}: {e}"))?[0];
        let execs = unsafe { pe.get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipe))? };
        for (i, e) in execs.iter().enumerate() {
            println!("\n######## {name}: executable {i} {:?} ({:?})", e.name_as_c_str().unwrap_or_default(),
                e.description_as_c_str().unwrap_or_default());
            let info = vk::PipelineExecutableInfoKHR::default().pipeline(pipe).executable_index(i as u32);
            for s in unsafe { pe.get_pipeline_executable_statistics(&info)? } {
                let v = unsafe {
                    match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => format!("{}", s.value.b32),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => format!("{}", s.value.i64),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => format!("{}", s.value.u64),
                        _ => format!("{}", s.value.f64),
                    }
                };
                println!("stat {:?} = {v}", s.name_as_c_str().unwrap_or_default());
            }
            // Sizes first, then the text itself.
            let mut count = 0u32;
            let f = pe.fp().get_pipeline_executable_internal_representations_khr;
            unsafe { f(pe.device(), &info, &mut count, std::ptr::null_mut()) }.result()?;
            let mut reps = vec![vk::PipelineExecutableInternalRepresentationKHR::default(); count as usize];
            unsafe { f(pe.device(), &info, &mut count, reps.as_mut_ptr()) }.result()?;
            let mut data: Vec<Vec<u8>> = reps.iter().map(|r| vec![0u8; r.data_size]).collect();
            for (r, d) in reps.iter_mut().zip(data.iter_mut()) {
                r.p_data = d.as_mut_ptr().cast();
            }
            unsafe { f(pe.device(), &info, &mut count, reps.as_mut_ptr()) }.result()?;
            for (r, d) in reps.iter().zip(&data) {
                println!("\n==== {name}: {:?} ({:?})", r.name_as_c_str().unwrap_or_default(),
                    r.description_as_c_str().unwrap_or_default());
                if r.is_text == vk::TRUE {
                    println!("{}", String::from_utf8_lossy(d).trim_end_matches('\0'));
                } else {
                    println!("({} bytes, binary)", d.len());
                }
            }
        }
        unsafe {
            dev.destroy_pipeline(pipe, None);
            dev.destroy_shader_module(module, None);
        }
    }
    unsafe {
        dev.destroy_pipeline_layout(layout, None);
        for s in sets {
            dev.destroy_descriptor_set_layout(s, None);
        }
        dev.destroy_device(None);
    }
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
