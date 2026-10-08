//! `trashcan qwen-kernel-check` (docs/tp8-optimization.md T1): every kernel of the TP8 plan on random inputs,
//! against a CPU reference and, on 64-lane GPUs (the D700s), against the kernels it replaces, with timings.
//!
//! The engine's kernels assume 64-lane subgroups; the new ones are written for any subgroup size, so with
//! `--any-device` (lavapipe, a GPU with another wave size) they run against the CPU references alone. One JSON line
//! per case: errors against the reference and the old kernels, whether the outputs equal the old kernels' bit for
//! bit, microseconds per call, and `pass`. The command fails if any case fails.
use crate::{
    gpu::Instance,
    vulkan::{Buffer, Device, Push},
};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{rc::Rc, time::Instant};

/// Shaders a relaxed device compiles: the plan's kernels written for any subgroup size.
const RELAXED: &[&str] = &[
    "qwen_attn_decode",
    "qwen_attn_merge",
    "qwen_topk_wave",
    "qwen_ffn_gu1",
    "qwen_ffn_gu2",
    "qwen_ffn_gu3",
    "qwen_ffn_gu4",
    "qwen_gdn_conv",
    "qwen_gated_norm_conv",
];

/// Uniform values in [-1, 1) from a xorshift state.
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
    }
}

struct Ctx {
    d: Rc<Device>,
    gpu: usize,
    /// 64-lane subgroups: the old kernels run too.
    native: bool,
    /// Subgroup size; kernels written for any size take 4 rows per subgroup.
    lanes: usize,
    iterations: usize,
}

impl Ctx {
    fn upload_f32(&self, v: &[f32]) -> Result<Buffer> {
        self.d.upload(bytemuck::cast_slice(v))
    }
    fn zeros(&self, floats: usize) -> Result<Buffer> {
        self.upload_f32(&vec![0f32; floats.max(1)])
    }
    fn download_f32(&self, b: &Buffer, floats: usize) -> Result<Vec<f32>> {
        if floats == 0 {
            return Ok(Vec::new());
        }
        Ok(b.download_range(0, floats * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    /// Record `body` once to check it and `iterations` times to time it: (microseconds per call).
    fn time(&self, body: &dyn Fn(ash::vk::CommandBuffer)) -> Result<f64> {
        let cmd = self.d.begin()?;
        body(cmd);
        self.d.finish(cmd)?;
        self.d.run(cmd)?;
        if self.iterations == 0 {
            return Ok(0.0);
        }
        let cmd = self.d.begin()?;
        for _ in 0..self.iterations {
            body(cmd);
        }
        self.d.finish(cmd)?;
        self.d.run_reusable(cmd)?;
        let start = Instant::now();
        self.d.run(cmd)?;
        Ok(start.elapsed().as_secs_f64() * 1e6 / self.iterations as f64)
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn bitwise(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

pub fn run(any_device: bool, only: &[String], iterations: usize) -> Result<()> {
    let inst = Rc::new(Instance::new()?);
    let devices = if any_device { inst.all_devices()? } else { inst.amd_gpus()? };
    ensure!(!devices.is_empty(), "no Vulkan devices found{}", if any_device { "" } else { " (--any-device tries others)" });
    let want = |name: &str| only.is_empty() || only.iter().any(|o| o == name);
    let mut failed = 0;
    for (gpu, pd) in devices.into_iter().enumerate() {
        let native = Device::subgroup_size(&inst, pd) == 64;
        let d = if native { Device::new(inst.clone(), pd)? } else { Device::new_relaxed(inst.clone(), pd, RELAXED)? };
        let ctx = Ctx { d, gpu, native, lanes: Device::subgroup_size(&inst, pd) as usize, iterations };
        let mut rows = Vec::new();
        if want("attention") {
            rows.extend(attention(&ctx)?);
        }
        if want("topk") {
            rows.extend(topk(&ctx)?);
        }
        if want("ffn") {
            rows.extend(ffn(&ctx)?);
        }
        if want("gdn") {
            rows.extend(gdn(&ctx)?);
        }
        for row in rows {
            failed += usize::from(row["pass"] != true);
            println!("{row}");
        }
    }
    ensure!(failed == 0, "{failed} kernel check(s) failed");
    Ok(())
}

/// A1: `qwen_attn_decode` + `qwen_attn_merge` (with its output gate) against an f64 reference and, natively, against
/// `attention_partial*` + `attention_merge*` + `qwen_gate`.
fn attention(ctx: &Ctx) -> Result<Vec<Value>> {
    const DIM: usize = 256;
    let mut rows = Vec::new();
    // (query heads per KV head, KV heads, keys before the batch, columns, partitions): TP8, TP4/TP2 and the pipeline,
    // decode and verification, short and long contexts, and partitions with no keys.
    let cases = [(3, 1, 520, 1, 64), (3, 1, 520, 3, 64), (6, 1, 600, 1, 64), (6, 2, 37, 2, 16), (6, 4, 1500, 1, 64), (3, 1, 8000, 1, 128), (2, 1, 5, 4, 32), (4, 1, 300, 8, 32)];
    for (case, &(g, kv, pos, n, parts)) in cases.iter().enumerate() {
        let heads = g * kv;
        let cap = pos + n + 3;
        let mut rng = Rng::new(17 + case as u32);
        let q: Vec<f32> = (0..n * heads * DIM).map(|_| rng.next() * 0.25).collect();
        let k16: Vec<half::f16> = (0..cap * kv * DIM).map(|_| half::f16::from_f32(rng.next() * 2.0)).collect();
        let v16: Vec<half::f16> = (0..cap * kv * DIM).map(|_| half::f16::from_f32(rng.next())).collect();
        let big: Vec<f32> = (0..n * heads * 2 * DIM).map(|_| rng.next() * 3.0).collect();
        // f64 reference with the output gate.
        let mut reference = vec![0f32; n * heads * DIM];
        for t in 0..n {
            for h in 0..heads {
                let kvh = h / g;
                let qr = &q[(t * heads + h) * DIM..][..DIM];
                let keys = pos + t + 1;
                let scores: Vec<f64> = (0..keys)
                    .map(|j| (0..DIM).map(|i| qr[i] as f64 * k16[(j * kv + kvh) * DIM + i].to_f64()).sum())
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let den: f64 = w.iter().sum();
                for i in 0..DIM {
                    let o: f64 = (0..keys).map(|j| w[j] * v16[(j * kv + kvh) * DIM + i].to_f64()).sum::<f64>() / den;
                    let gate = big[t * 2 * heads * DIM + h * 2 * DIM + DIM + i] as f64;
                    reference[(t * heads + h) * DIM + i] = (o / (1.0 + (-gate).exp())) as f32;
                }
            }
        }
        let d = &ctx.d;
        let qb = ctx.upload_f32(&q)?;
        let kc = d.upload(bytemuck::cast_slice(&k16.iter().map(|x| x.to_bits()).collect::<Vec<u16>>()))?;
        let vc = d.upload(bytemuck::cast_slice(&v16.iter().map(|x| x.to_bits()).collect::<Vec<u16>>()))?;
        let bigb = ctx.upload_f32(&big)?;
        let params = d.upload(bytemuck::cast_slice(&[pos as u32, n as u32, 0u32, 0]))?;
        let scratch = ctx.zeros(n * heads * 128 * (DIM + 4))?;
        let out = ctx.zeros(n * heads * DIM)?;
        let mut u = [0u32; 16];
        u[2] = parts as u32;
        u[4] = heads as u32;
        u[5] = kv as u32;
        u[6] = DIM as u32;
        u[7] = cap as u32;
        let kernel = crate::qwen::gpu::DECODE_ATTN_HEADS.iter().find(|v| v.0 == g).map(|v| v.1).unwrap();
        let new_us = ctx.time(&|cmd| {
            d.dispatch(cmd, kernel, Push { ptr: [qb.address, kc.address, vc.address, 0, scratch.address, 0, params.address, 0], u }, (kv * parts) as u32, n as u32);
            d.dispatch(cmd, "qwen_attn_merge", Push { ptr: [0, bigb.address, 0, out.address, scratch.address, 0, params.address, 0], u }, heads as u32, n as u32);
        })?;
        let new = ctx.download_f32(&out, n * heads * DIM)?;
        let scale = reference.iter().fold(0f32, |a, b| a.max(b.abs()));
        let err = max_abs_diff(&new, &reference);
        let mut row = json!({"check": "attention", "gpu": ctx.gpu, "case": {"g": g, "kv_heads": kv, "pos": pos, "n": n, "parts": parts},
            "new_us": new_us, "max_err_vs_reference": err, "reference_scale": scale});
        let mut pass = err <= 2e-4 * scale.max(1e-3) && new.iter().all(|x| x.is_finite());
        if ctx.native {
            let (partial, merge) = crate::vulkan::attention_shaders(parts as u32)?;
            let old_out = ctx.zeros(n * heads * DIM)?;
            let old_us = ctx.time(&|cmd| {
                d.dispatch(cmd, partial, Push { ptr: [qb.address, kc.address, vc.address, old_out.address, scratch.address, 0, params.address, 0], u }, (heads * parts) as u32, n as u32);
                d.dispatch(cmd, merge, Push { ptr: [0, 0, 0, old_out.address, scratch.address, 0, params.address, 0], u }, heads as u32, n as u32);
                let mut g = [0u32; 16];
                g[0] = (heads * DIM) as u32;
                g[6] = DIM as u32;
                d.dispatch(cmd, "qwen_gate", Push { ptr: [old_out.address, bigb.address, 0, 0, 0, 0, params.address, 0], u: g }, (heads * DIM).div_ceil(256) as u32, n as u32);
            })?;
            let old = ctx.download_f32(&old_out, n * heads * DIM)?;
            let vs_old = max_abs_diff(&new, &old);
            row["old_us"] = json!(old_us);
            row["max_err_vs_old"] = json!(vs_old);
            row["max_err_old_vs_reference"] = json!(max_abs_diff(&old, &reference));
            pass &= vs_old <= 2e-4 * scale.max(1e-3);
        }
        row["pass"] = json!(pass);
        rows.push(row);
    }
    Ok(rows)
}

/// K1: the head's selection passes with `qwen_topk_wave` against the exact top 64 per column (highest score, ties by
/// lower id) and, natively, against `qwen_topk`'s candidate lists bit for bit, pass by pass.
fn topk(ctx: &Ctx) -> Result<Vec<Value>> {
    const TOP: usize = 64;
    let mut rows = Vec::new();
    // (vocabulary rows, columns): a TP8 rank's head, a draft head's share, a verification batch, one partial chunk.
    for (case, &(rows_n, cols)) in [(31040usize, 1usize), (8192, 1), (31040, 3), (1000, 2)].iter().enumerate() {
        let mut rng = Rng::new(91 + case as u32);
        // Quantized scores, so ties are common.
        let logits: Vec<f32> = (0..rows_n * cols).map(|_| (rng.next() * 400.0).round() / 16.0).collect();
        let mut expect = Vec::new();
        for t in 0..cols {
            let mut c: Vec<(f32, u32)> = (0..rows_n).map(|i| (logits[t * rows_n + i], i as u32)).collect();
            c.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            c.truncate(TOP);
            expect.push(c);
        }
        let d = &ctx.d;
        let src = ctx.upload_f32(&logits)?;
        let chunks = rows_n.div_ceil(1024);
        let scratch = ctx.zeros(cols * chunks * TOP * 2)?;
        let scratch2 = ctx.zeros(cols * chunks.div_ceil(16) * TOP * 2)?;
        let top = ctx.zeros(cols * TOP * 2)?;
        let params = d.upload(bytemuck::cast_slice(&[0u32, cols as u32, 0, 0]))?;
        // The passes of `Rec::head`: the best 64 of each 1024 logits, then of each 1024 of those, until one list.
        let passes = |kernel: &'static str| {
            let mut out = Vec::new();
            let (mut src_a, mut kind, mut entries, mut groups, mut pass) = (src.address, 0u32, rows_n, chunks, 0);
            loop {
                let dst = match (groups, pass) {
                    (1, _) => &top,
                    (_, 0) => &scratch,
                    _ => &scratch2,
                };
                let mut u = [0u32; 16];
                u[0] = entries as u32;
                u[2] = (groups * TOP) as u32;
                u[9] = kind;
                u[15] = u32::from(cols == 1);
                out.push((kernel, Push { ptr: [src_a, 0, dst.address, 0, 0, 0, params.address, 0], u }, groups as u32, dst));
                if groups == 1 {
                    return out;
                }
                (src_a, kind, entries, pass) = (dst.address, 1, groups * TOP, pass + 1);
                groups = entries.div_ceil(1024);
            }
        };
        let new_passes = passes("qwen_topk_wave");
        let new_us = ctx.time(&|cmd| {
            for (k, p, g, _) in &new_passes {
                d.dispatch(cmd, k, *p, *g, cols as u32);
            }
        })?;
        let read = |b: &Buffer| -> Result<Vec<Vec<(f32, u32)>>> {
            let bytes = b.download_range(0, cols * TOP * 8)?;
            Ok(bytes.chunks_exact(TOP * 8).map(|c| c.chunks_exact(8).map(|e| (f32::from_le_bytes(e[..4].try_into().unwrap()), u32::from_le_bytes(e[4..].try_into().unwrap()))).collect()).collect())
        };
        let got = read(&top)?;
        let mut pass = got == expect;
        let mut row = json!({"check": "topk", "gpu": ctx.gpu, "case": {"rows": rows_n, "cols": cols}, "new_us": new_us, "exact": pass});
        if ctx.native {
            // Run each pass of both kernels on the same input and compare every intermediate list bit for bit.
            let old_passes = passes("qwen_topk");
            let old_us = ctx.time(&|cmd| {
                for (k, p, g, _) in &old_passes {
                    d.dispatch(cmd, k, *p, *g, cols as u32);
                }
            })?;
            let old = read(&top)?;
            let same = old.iter().zip(&got).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.0.to_bits() == y.0.to_bits() && x.1 == y.1));
            row["old_us"] = json!(old_us);
            row["bitwise_vs_old"] = json!(same);
            pass &= same;
        }
        row["pass"] = json!(pass);
        rows.push(row);
    }
    Ok(rows)
}

/// Random Q4_0 matrix of `rows` x `k` in GGUF blocks: (repacked bytes for the GPU, dequantized rows for the CPU).
fn q4_matrix(rng: &mut Rng, rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut raw = Vec::with_capacity(rows * k / 32 * 18);
    let mut deq = vec![0f32; rows * k];
    for r in 0..rows {
        for b in 0..k / 32 {
            let d = half::f16::from_f32(rng.next() * 0.02 + 0.03 * rng.next().signum());
            raw.extend_from_slice(&d.to_le_bytes());
            let mut q = [0u8; 16];
            for (j, byte) in q.iter_mut().enumerate() {
                let (lo, hi) = (((rng.next() + 1.0) * 8.0) as u8 & 15, ((rng.next() + 1.0) * 8.0) as u8 & 15);
                *byte = lo | hi << 4;
                deq[r * k + b * 32 + j] = (lo as f32 - 8.0) * d.to_f32();
                deq[r * k + b * 32 + 16 + j] = (hi as f32 - 8.0) * d.to_f32();
            }
            raw.extend_from_slice(&q);
        }
    }
    (crate::quant::repack_q4(&raw, k, rows, 0, rows, 0, k), deq)
}

/// F1: `qwen_ffn_guN` against an f64 reference and, natively, against `matvecq4` / `qwen_q4cN` for gate and up then
/// `qwen_swiglu`, bit for bit.
fn ffn(ctx: &Ctx) -> Result<Vec<Value>> {
    let mut rows_out = Vec::new();
    // (rows, k, columns): TP8 and TP4 FFN shards, decode and verification widths.
    for (case, &(rows, k, n)) in [(2176usize, 5120usize, 1usize), (2176, 5120, 3), (4352, 5120, 2), (1088, 5120, 4), (2176, 5120, 1)].iter().enumerate() {
        let mut rng = Rng::new(301 + case as u32);
        let (gate_w, gate_ref) = q4_matrix(&mut rng, rows, k);
        let (up_w, up_ref) = q4_matrix(&mut rng, rows, k);
        let x: Vec<f32> = (0..n * k).map(|_| rng.next()).collect();
        let mut reference = vec![0f32; n * rows];
        for t in 0..n {
            for r in 0..rows {
                let dot = |w: &[f32]| (0..k).map(|i| w[r * k + i] as f64 * x[t * k + i] as f64).sum::<f64>();
                let (g, u) = (dot(&gate_ref), dot(&up_ref));
                reference[t * rows + r] = (g / (1.0 + (-g).exp()) * u) as f32;
            }
        }
        let d = &ctx.d;
        let (gate, up) = (d.upload_matrix(&gate_w)?, d.upload_matrix(&up_w)?);
        let xb = ctx.upload_f32(&x)?;
        let act = ctx.zeros(n * rows)?;
        let params = d.upload(bytemuck::cast_slice(&[0u32, n as u32, 0, 0]))?;
        let mut u = [0u32; 16];
        u[0] = rows as u32;
        u[1] = k as u32;
        u[3] = (rows * k / 32) as u32;
        let kernel = ["qwen_ffn_gu1", "qwen_ffn_gu2", "qwen_ffn_gu3", "qwen_ffn_gu4"][n - 1];
        let per_group = 4 * 256 / ctx.lanes;
        let new_us = ctx.time(&|cmd| {
            d.dispatch_matrix(cmd, kernel, Push { ptr: [0, xb.address, act.address, up.address, 0, 0, params.address, 0], u }, rows.div_ceil(per_group) as u32, 1, &gate).unwrap();
        })?;
        let new = ctx.download_f32(&act, n * rows)?;
        let scale = reference.iter().fold(0f32, |a, b| a.max(b.abs()));
        let err = max_abs_diff(&new, &reference);
        let mut row = json!({"check": "ffn", "gpu": ctx.gpu, "case": {"rows": rows, "k": k, "n": n}, "new_us": new_us, "max_err_vs_reference": err, "reference_scale": scale});
        let mut pass = err <= 1e-4 * scale.max(1e-3) && new.iter().all(|v| v.is_finite());
        if ctx.native {
            let (g_out, u_out, old_act) = (ctx.zeros(n * rows)?, ctx.zeros(n * rows)?, ctx.zeros(n * rows)?);
            let mat = ["matvecq4", "qwen_q4c2", "qwen_q4c3", "qwen_q4c4"][n - 1];
            let old_us = ctx.time(&|cmd| {
                d.dispatch_matrix(cmd, mat, Push { ptr: [0, xb.address, g_out.address, 0, 0, 0, params.address, 0], u }.defer_barrier(true), rows.div_ceil(16) as u32, 1, &gate).unwrap();
                d.dispatch_matrix(cmd, mat, Push { ptr: [0, xb.address, u_out.address, 0, 0, 0, params.address, 0], u }, rows.div_ceil(16) as u32, 1, &up).unwrap();
                let mut s = [0u32; 16];
                s[0] = rows as u32;
                d.dispatch(cmd, "qwen_swiglu", Push { ptr: [g_out.address, u_out.address, old_act.address, 0, 0, 0, params.address, 0], u: s }, rows.div_ceil(256) as u32, n as u32);
            })?;
            let old = ctx.download_f32(&old_act, n * rows)?;
            row["old_us"] = json!(old_us);
            row["bitwise_vs_old"] = json!(bitwise(&new, &old));
            row["max_err_vs_old"] = json!(max_abs_diff(&new, &old));
            pass &= bitwise(&new, &old);
        }
        row["pass"] = json!(pass);
        rows_out.push(row);
    }
    Ok(rows_out)
}

/// G1: `qwen_gdn_conv` + `qwen_gated_norm_conv` against an f64 reference of the convolution, recurrence and gated norm
/// and, natively, against `qwen_conv` + `qwen_gdn` + `qwen_gated_norm` bit for bit: the output, the state, the
/// convolution window and, for verification batches, both kinds of snapshots.
fn gdn(ctx: &Ctx) -> Result<Vec<Value>> {
    const DIM: usize = 128;
    let mut rows_out = Vec::new();
    // (key heads, value heads, columns, position, snapshots): TP8, TP4 and the pipeline; decode, verification, a
    // fresh sequence.
    for (case, &(kh, vh, n, pos, snaps)) in [(2usize, 6usize, 1usize, 10usize, false), (2, 6, 3, 10, true), (4, 12, 2, 0, true), (16, 48, 1, 5, false), (2, 6, 8, 3, true)].iter().enumerate() {
        let (kd, vd) = (kh * DIM, vh * DIM);
        let cd = 2 * kd + vd;
        let mut rng = Rng::new(701 + case as u32);
        let mut gen = |len: usize, scale: f32| (0..len).map(|_| rng.next() * scale).collect::<Vec<f32>>();
        let raw = gen(n * cd, 1.0);
        let window = gen(3 * cd, 1.0);
        let conv_w = gen(cd * 4, 0.5);
        let ab = gen(n * 2 * vh, 3.0);
        let dt = gen(vh, 1.0);
        let a_neg: Vec<f32> = gen(vh, 1.0).iter().map(|x| -x.exp()).collect();
        let state = gen(vh * DIM * DIM, 0.1);
        let z = gen(n * vd, 2.0);
        let norm_w: Vec<f32> = gen(DIM, 0.1).iter().map(|x| 1.0 + x).collect();
        let eps = 1e-6f32;
        // f64 reference.
        let fresh = pos == 0;
        let mut win: Vec<f64> = window.iter().map(|&x| if fresh { 0.0 } else { x as f64 }).collect();
        let mut st: Vec<f64> = state.iter().map(|&x| if fresh { 0.0 } else { x as f64 }).collect();
        let (mut out, mut st_snaps, mut win_snaps) = (vec![0f64; n * vd], Vec::new(), Vec::new());
        for t in 0..n {
            let mut cv = vec![0f64; cd];
            for ch in 0..cd {
                let x3 = raw[t * cd + ch] as f64;
                let w = &conv_w[ch * 4..ch * 4 + 4];
                let s = w[0] as f64 * win[ch] + w[1] as f64 * win[cd + ch] + w[2] as f64 * win[2 * cd + ch] + w[3] as f64 * x3;
                cv[ch] = s / (1.0 + (-s).exp());
                (win[ch], win[cd + ch], win[2 * cd + ch]) = (win[cd + ch], win[2 * cd + ch], x3);
            }
            for h in 0..vh {
                let k_head = h % kh;
                let q = &cv[k_head * DIM..][..DIM];
                let k = &cv[kd + k_head * DIM..][..DIM];
                let qn = 1.0 / (q.iter().map(|x| x * x).sum::<f64>() + eps as f64).sqrt() / (DIM as f64).sqrt();
                let kn = 1.0 / (k.iter().map(|x| x * x).sum::<f64>() + eps as f64).sqrt();
                let x = ab[t * 2 * vh + h] as f64 + dt[h] as f64;
                let decay = ((if x > 20.0 { x } else { (1.0 + x.exp()).ln() }) * a_neg[h] as f64).exp();
                let beta = 1.0 / (1.0 + (-(ab[t * 2 * vh + vh + h] as f64)).exp());
                for i in 0..DIM {
                    let row = &mut st[(h * DIM + i) * DIM..][..DIM];
                    row.iter_mut().for_each(|s| *s *= decay);
                    let sk: f64 = row.iter().zip(k).map(|(s, k)| s * k * kn).sum();
                    let delta = (cv[2 * kd + h * DIM + i] - sk) * beta;
                    row.iter_mut().zip(k).for_each(|(s, k)| *s += k * kn * delta);
                    out[t * vd + h * DIM + i] = row.iter().zip(q).map(|(s, q)| s * q * qn).sum();
                }
            }
            if t + 1 < n {
                st_snaps.extend(st.iter().copied());
                win_snaps.extend(win.iter().copied());
            }
        }
        let mut gated = vec![0f32; n * vd];
        for t in 0..n {
            for h in 0..vh {
                let o = &out[t * vd + h * DIM..][..DIM];
                let inv = 1.0 / (o.iter().map(|x| x * x).sum::<f64>() / DIM as f64 + eps as f64).sqrt();
                for i in 0..DIM {
                    let zz = z[t * vd + h * DIM + i] as f64;
                    gated[t * vd + h * DIM + i] = (o[i] * inv * norm_w[i] as f64 * (zz / (1.0 + (-zz).exp()))) as f32;
                }
            }
        }
        let d = &ctx.d;
        let params = d.upload(bytemuck::cast_slice(&[pos as u32, n as u32, 0, 0]))?;
        let (rawb, abb, zb, wb) = (ctx.upload_f32(&raw)?, ctx.upload_f32(&ab)?, ctx.upload_f32(&z)?, ctx.upload_f32(&norm_w)?);
        let wdt: Vec<f32> = conv_w.iter().chain(&dt).chain(&a_neg).copied().collect();
        let wdtb = d.upload_matrix(bytemuck::cast_slice(&wdt))?;
        // Mutable buffers of one run: state, window, output, state and window snapshots.
        let fresh_buffers = || -> Result<[Buffer; 5]> {
            Ok([ctx.upload_f32(&state)?, ctx.upload_f32(&window)?, ctx.zeros(n * vd)?, ctx.zeros(n.max(2) * vh * DIM * DIM)?, ctx.zeros(n.max(2) * 3 * cd)?])
        };
        let new_b = fresh_buffers()?;
        let (snap, snap_conv) = if snaps { (new_b[3].address, new_b[4].address) } else { (0, 0) };
        let mut u = [0u32; 16];
        u[0] = cd as u32;
        u[4] = vh as u32;
        u[5] = kh as u32;
        u[6] = DIM as u32;
        u[12] = eps.to_bits();
        let mut un = [0u32; 16];
        un[0] = vd as u32;
        un[1] = cd as u32;
        un[2] = (2 * kd) as u32;
        un[6] = DIM as u32;
        un[12] = eps.to_bits();
        let per_group = 4 * 256 / ctx.lanes;
        // One run only: the kernels update the state in place.
        let cmd = d.begin()?;
        d.dispatch_matrix(cmd, "qwen_gdn_conv", Push { ptr: [rawb.address, abb.address, new_b[0].address, new_b[2].address, new_b[1].address, snap, params.address, snap_conv], u }, (vh * DIM / per_group) as u32, 1, &wdtb)?;
        d.dispatch(cmd, "qwen_gated_norm_conv", Push { ptr: [new_b[2].address, zb.address, wb.address, rawb.address, new_b[1].address, snap_conv, params.address, 0], u: un }, vh as u32, n as u32);
        d.finish(cmd)?;
        d.run(cmd)?;
        let got = |b: &[Buffer; 5]| -> Result<[Vec<f32>; 5]> {
            Ok([
                ctx.download_f32(&b[0], vh * DIM * DIM)?,
                ctx.download_f32(&b[1], 3 * cd)?,
                ctx.download_f32(&b[2], n * vd)?,
                ctx.download_f32(&b[3], (n - 1) * vh * DIM * DIM)?,
                ctx.download_f32(&b[4], (n - 1) * 3 * cd)?,
            ])
        };
        let new = got(&new_b)?;
        let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let refs = [f(&st), f(&win), gated.clone(), f(&st_snaps), f(&win_snaps)];
        let names = ["state", "window", "output", "state_snapshots", "window_snapshots"];
        let mut errors = serde_json::Map::new();
        let mut pass = true;
        for (i, name) in names.iter().enumerate() {
            if i >= 3 && !snaps {
                continue;
            }
            let scale = refs[i].iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-3);
            let err = max_abs_diff(&new[i], &refs[i]);
            errors.insert(name.to_string(), json!(err / scale));
            pass &= err <= 1e-4 * scale && new[i].iter().all(|v| v.is_finite());
        }
        let mut row = json!({"check": "gdn", "gpu": ctx.gpu, "case": {"k_heads": kh, "v_heads": vh, "n": n, "pos": pos, "snapshots": snaps}, "rel_err_vs_reference": errors});
        if ctx.native {
            let old_b = fresh_buffers()?;
            let (snap, snap_conv) = if snaps { (old_b[3].address, old_b[4].address) } else { (0, 0) };
            let conv_out = ctx.zeros(n * cd)?;
            let (cwb, dtab) = (ctx.upload_f32(&conv_w)?, ctx.upload_f32(&dt.iter().chain(&a_neg).copied().collect::<Vec<f32>>())?);
            let mut uc = [0u32; 16];
            uc[0] = cd as u32;
            let mut ug = [0u32; 16];
            ug[0] = vd as u32;
            ug[6] = DIM as u32;
            ug[12] = eps.to_bits();
            let cmd = d.begin()?;
            d.dispatch(cmd, "qwen_conv", Push { ptr: [rawb.address, cwb.address, old_b[1].address, conv_out.address, snap_conv, 0, params.address, 0], u: uc }, cd.div_ceil(256) as u32, 1);
            d.dispatch(cmd, "qwen_gdn", Push { ptr: [conv_out.address, abb.address, old_b[0].address, old_b[2].address, dtab.address, snap, params.address, 0], u }, (vh * DIM / 16) as u32, 1);
            d.dispatch(cmd, "qwen_gated_norm", Push { ptr: [old_b[2].address, zb.address, wb.address, 0, 0, 0, params.address, 0], u: ug }, vh as u32, n as u32);
            d.finish(cmd)?;
            d.run(cmd)?;
            let old = got(&old_b)?;
            let mut same = serde_json::Map::new();
            for (i, name) in names.iter().enumerate() {
                if i >= 3 && !snaps {
                    continue;
                }
                let eq = bitwise(&new[i], &old[i]);
                same.insert(name.to_string(), json!(eq));
                pass &= eq;
            }
            row["bitwise_vs_old"] = Value::Object(same);
            if ctx.iterations > 0 {
                // Timing on throwaway state: the outputs above were taken from single runs.
                let t_new = ctx.time(&|cmd| {
                    d.dispatch_matrix(cmd, "qwen_gdn_conv", Push { ptr: [rawb.address, abb.address, new_b[0].address, new_b[2].address, new_b[1].address, 0, params.address, 0], u }, (vh * DIM / 16) as u32, 1, &wdtb).unwrap();
                    d.dispatch(cmd, "qwen_gated_norm_conv", Push { ptr: [new_b[2].address, zb.address, wb.address, rawb.address, new_b[1].address, 0, params.address, 0], u: un }, vh as u32, n as u32);
                })?;
                let t_old = ctx.time(&|cmd| {
                    d.dispatch(cmd, "qwen_conv", Push { ptr: [rawb.address, cwb.address, old_b[1].address, conv_out.address, 0, 0, params.address, 0], u: uc }, cd.div_ceil(256) as u32, 1);
                    d.dispatch(cmd, "qwen_gdn", Push { ptr: [conv_out.address, abb.address, old_b[0].address, old_b[2].address, dtab.address, 0, params.address, 0], u }, (vh * DIM / 16) as u32, 1);
                    d.dispatch(cmd, "qwen_gated_norm", Push { ptr: [old_b[2].address, zb.address, wb.address, 0, 0, 0, params.address, 0], u: ug }, vh as u32, n as u32);
                })?;
                row["new_us"] = json!(t_new);
                row["old_us"] = json!(t_old);
            }
        }
        row["pass"] = json!(pass);
        rows_out.push(row);
    }
    Ok(rows_out)
}
