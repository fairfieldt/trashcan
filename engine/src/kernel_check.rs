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
const RELAXED: &[&str] = &["qwen_attn_decode", "qwen_attn_merge", "qwen_topk_wave", "qwen_ffn_gu1", "qwen_ffn_gu2", "qwen_ffn_gu3", "qwen_ffn_gu4"];

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
        Ok(bytemuck::cast_slice(&b.download_range(0, floats * 4)?).to_vec())
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
