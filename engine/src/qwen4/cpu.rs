//! Straightforward f32 CPU forward of a qwen4exp layer range: the executable specification the GPU kernels are
//! checked against (docs/qwen4.md), and a stage that runs anywhere. Captured names follow llama.cpp's graph
//! callbacks; a name the graph uses twice in a layer (the second hyper-connection mixer) gets `#2`.
use super::dequant_row;
use super::Config;
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::qwen::vector;
use anyhow::{ensure, Result};
use half::{bf16, f16};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

/// Reproduce ggml's CPU rounding instead of computing in f32: activations quantized to Q8_0 (blocks of 32,
/// f16 scale, ties to even) before a Q8_0/Q4_0/Q5_0 product and to BF16 before a BF16 one, f16 K/V and
/// indexer caches, f16 queries and attention probabilities. Validation only: it isolates formula errors from
/// llama.cpp's own quantization noise.
pub static EMULATE_GGML: AtomicBool = AtomicBool::new(false);

fn emulate() -> bool {
    EMULATE_GGML.load(Ordering::Relaxed)
}

/// The QSA K/V cache rounding to reproduce: 0 as stored by `EMULATE_GGML` (f32, or f16), 1 Q8_0 K and V,
/// 2 f16 K and Q8_0 V (the GPU's `KvCache` formats).
static KV_CACHE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_kv_cache(kv: super::gpu::KvCache) {
    use super::gpu::KvCache;
    let mode = match kv {
        KvCache::F16 => 0,
        KvCache::Q8 => 1,
        KvCache::K16V8 => 2,
    };
    KV_CACHE.store(mode, Ordering::Relaxed);
}

/// Round to Q8_0 per 32 values as ggml's SIMD quantizer does: d = amax / 127, q = x / d to nearest even, f16 d.
fn round_q8(x: &mut [f32]) {
    for b in x.chunks_mut(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.;
        let id = if d != 0. { 1. / d } else { 0. };
        let dh = f16::from_f32(d).to_f32();
        for v in b {
            *v = (*v * id).round_ties_even() * dh;
        }
    }
}

/// Round a stored K or V row the way the cache being reproduced keeps it.
fn round_kv(x: &mut [f32], is_v: bool) {
    match (KV_CACHE.load(Ordering::Relaxed), is_v) {
        (1, _) | (2, true) => round_q8(x),
        (2, false) => {
            for v in x {
                *v = f16::from_f32(*v).to_f32();
            }
        }
        _ => round_f16(x),
    }
}

fn round_f16(x: &mut [f32]) {
    if emulate() {
        for v in x {
            *v = f16::from_f32(*v).to_f32();
        }
    }
}

/// The activation rounding ggml's CPU backend applies before a product with weights of type `ty`.
fn round_activations(ty: GgmlType, x: &[f32]) -> Option<Vec<f32>> {
    if !emulate() {
        return None;
    }
    let mut y = x.to_vec();
    match ty {
        GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q5_0 => {
            for b in y.chunks_exact_mut(32) {
                let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.;
                let id = if d != 0. { 1. / d } else { 0. };
                let dh = f16::from_f32(d).to_f32();
                for v in b {
                    *v = (*v * id).round_ties_even() * dh;
                }
            }
        }
        GgmlType::BF16 => {
            for v in &mut y {
                *v = bf16::from_f32(*v).to_f32();
            }
        }
        GgmlType::F16 => round_f16(&mut y),
        _ => return None,
    }
    Some(y)
}

/// y[t][row] = W[row] . x[t] over a whole tensor, rows split across threads. When emulating, a product with at
/// least 32 rows, 32 columns and 32 tokens runs unrounded in f32, as ggml's BLAS backend takes those over.
pub(crate) fn matmul(g: &Gguf, t: &TensorInfo, x: &[f32], n: usize) -> Vec<f32> {
    let blas = n >= 32 && t.dims[0] >= 32 && t.dims[1] >= 32;
    match round_activations(t.ty, x).filter(|_| !blas) {
        Some(xr) => matmul_rows(g, t, &xr, n),
        None => matmul_rows(g, t, x, n),
    }
}

/// `qwen::cpu::matmul` with this module's dequantization (which adds the IQ4 types).
fn matmul_rows(g: &Gguf, t: &TensorInfo, x: &[f32], n: usize) -> Vec<f32> {
    let rows: u64 = t.dims[1..].iter().product();
    let rows = rows as usize;
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let per = rows.div_ceil(threads);
    let parts: Vec<Vec<f32>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|i| {
                s.spawn(move || {
                    let (lo, hi) = (i * per, ((i + 1) * per).min(rows));
                    if lo >= hi {
                        return Vec::new();
                    }
                    matvec_rows(g, t, lo, hi - lo, x, n)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut y = vec![0f32; n * rows];
    for (i, part) in parts.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
        let lo = i * per;
        let width = part.len() / n;
        for tok in 0..n {
            y[tok * rows + lo..tok * rows + lo + width].copy_from_slice(&part[tok * width..(tok + 1) * width]);
        }
    }
    y
}

/// Last-token rows of named intermediates.
pub type Captures = BTreeMap<String, Vec<f32>>;

enum State {
    /// Previous `conv - 1` inputs of the short convolution and one `state x state` matrix per value head
    /// (`s[head][v][k]`).
    Recurrent { conv: Vec<f32>, s: Vec<f32> },
    /// f32 K/V `[pos][kv_head][dim]`, raw indexer keys `[pos][idx_dim]` and the pooled, normed and rotated key of
    /// every complete block `[block][idx_dim]`.
    Attention {
        k: Vec<f32>,
        v: Vec<f32>,
        kraw: Vec<f32>,
        pooled: Vec<f32>,
    },
}

pub struct CpuStage {
    c: Config,
    pub first: usize,
    pub last: usize,
    states: Vec<State>,
    /// History of the PLE convolution, `[(conv - 1) * ngram][hc * d]`, oldest first (empty unless this stage owns
    /// the PLE layer).
    ple_conv: Vec<f32>,
    pub head: bool,
    /// The latest verification batch: the recurrent states before it, its input rows, n-gram rows and position,
    /// and its output rows (speculative decoding; `rollback` replays the kept tokens).
    verified: Option<Verified>,
}

struct Verified {
    states: Vec<Vec<f32>>,
    x: Vec<f32>,
    ple: Option<Vec<f32>>,
    pos: usize,
    n: usize,
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for i in 0..chunks {
        for j in 0..8 {
            acc[j] += a[i * 8 + j] * b[i * 8 + j];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub(crate) fn rms_norm(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let mean = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1. / ((mean as f32) + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v *= inv * w.map_or(1., |w| w[i]);
    }
}

pub(crate) fn silu(x: f32) -> f32 {
    x / (1. + (-x).exp())
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// y[t][r] = W[row0 + r] . x[t] for `rows` rows of a (possibly 3-D, expert-stacked) tensor, on this thread.
fn matvec_rows(g: &Gguf, t: &TensorInfo, row0: usize, rows: usize, x: &[f32], n: usize) -> Vec<f32> {
    let rounded = round_activations(t.ty, x);
    let x = rounded.as_deref().unwrap_or(x);
    let k = t.dims[0] as usize;
    let mut w = vec![0f32; k];
    let mut y = vec![0f32; n * rows];
    for r in 0..rows {
        dequant_row(g, t, row0 + r, &mut w).unwrap();
        for tok in 0..n {
            y[tok * rows + r] = dot(&w, &x[tok * k..(tok + 1) * k]);
        }
    }
    y
}

/// The rotation of the first `rot` dims of a head (NeoX pairs i, i + rot/2) at `pos`.
fn rotate(c: &Config, x: &mut [f32], pos: usize) {
    let half = c.rot / 2;
    for i in 0..half {
        let (sn, cs) = c.rope_angle(pos, i).sin_cos();
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cs - b * sn;
        x[i + half] = a * sn + b * cs;
    }
}

/// Output of a hyper-connection mixer: the sublayer input `n x d` and, unless it is the final mixer, the per-stream
/// write weights `n x hc`.
pub(crate) struct Mixed {
    pub input: Vec<f32>,
    pub write: Option<Vec<f32>>,
}

impl CpuStage {
    /// Layers `first..last`; `head` adds the final mixer and LM head.
    pub fn new(c: &Config, first: usize, last: usize, context: usize, head: bool) -> Self {
        let states = (first..last)
            .map(|il| {
                if c.recurrent(il) {
                    State::Recurrent {
                        conv: vec![0.; (c.conv - 1) * c.conv_dim()],
                        s: vec![0.; c.v_heads * c.state * c.state],
                    }
                } else {
                    let kv = context * c.kv_heads * c.head_dim;
                    State::Attention {
                        k: vec![0.; kv],
                        v: vec![0.; kv],
                        kraw: vec![0.; context * c.idx_dim],
                        pooled: vec![0.; context.div_ceil(c.kpool) * c.idx_dim],
                    }
                }
            })
            .collect();
        let ple_conv = match &c.ngram {
            Some(p) if (first..last).contains(&p.layer) => vec![0.; (p.conv - 1) * p.ngram * c.width()],
            _ => Vec::new(),
        };
        Self {
            c: c.clone(),
            first,
            last,
            states,
            ple_conv,
            head,
            verified: None,
        }
    }

    fn recurrent(&self) -> Vec<Vec<f32>> {
        let mut v = vec![self.ple_conv.clone()];
        for st in &self.states {
            if let State::Recurrent { conv, s } = st {
                v.push(conv.clone());
                v.push(s.clone());
            }
        }
        v
    }

    fn set_recurrent(&mut self, saved: &[Vec<f32>]) {
        self.ple_conv.copy_from_slice(&saved[0]);
        let mut it = saved[1..].iter();
        for st in &mut self.states {
            if let State::Recurrent { conv, s } = st {
                conv.copy_from_slice(it.next().unwrap());
                s.copy_from_slice(it.next().unwrap());
            }
        }
    }

    /// `forward` for a verification batch: keeps what `rollback` needs.
    pub fn verify(&mut self, g: &Gguf, x: &mut [f32], ple: Option<&[f32]>, pos: usize, n: usize) -> Result<()> {
        ensure!(pos > 0, "a verification batch continues a sequence");
        let states = self.recurrent();
        let input = x.to_vec();
        self.forward(g, x, ple, pos, n, &mut None)?;
        self.verified = Some(Verified { states, x: input, ple: ple.map(|p| p.to_vec()), pos, n });
        Ok(())
    }

    /// Keep tokens `0..keep` of the latest verification batch: restore the states before it and replay them.
    pub fn rollback(&mut self, g: &Gguf, keep: usize) -> Result<()> {
        let v = self.verified.take().ok_or_else(|| anyhow::anyhow!("rollback without a verification batch"))?;
        ensure!(keep >= 1 && keep <= v.n, "rollback to {keep} of a {}-token batch", v.n);
        if keep < v.n {
            self.set_recurrent(&v.states);
            let w = self.c.width();
            let mut x = v.x[..keep * w].to_vec();
            let ple = v.ple.as_ref().map(|p| &p[..keep * self.c.d]);
            self.forward(g, &mut x, ple, v.pos, keep, &mut None)?;
        }
        Ok(())
    }

    /// The final mixer's output for every row of `x` (a verification batch's final hidden states).
    pub fn final_rows(&self, g: &Gguf, x: &[f32]) -> Result<Vec<f32>> {
        let w = self.c.width();
        let mut out = Vec::with_capacity(x.len() / w * self.c.d);
        for row in x.chunks_exact(w) {
            out.extend(self.final_hidden(g, row, &mut None)?);
        }
        Ok(out)
    }

    /// Run the layers over `n` tokens at `pos` in place on the `n x hc x d` residual. `ple` holds the gathered
    /// n-gram rows (`n x d`) when this stage owns the PLE layer. Position 0 starts a new sequence.
    pub fn forward(
        &mut self,
        g: &Gguf,
        x: &mut [f32],
        ple: Option<&[f32]>,
        pos: usize,
        n: usize,
        cap: &mut Option<&mut Captures>,
    ) -> Result<()> {
        let c = self.c.clone();
        let w = c.width();
        ensure!(x.len() == n * w, "residual length mismatch");
        if pos == 0 {
            self.ple_conv.fill(0.);
            for s in &mut self.states {
                if let State::Recurrent { conv, s } = s {
                    conv.fill(0.);
                    s.fill(0.);
                }
            }
        }
        let mut record = |name: String, rows: &[f32], width: usize| {
            if let Some(m) = cap.as_mut() {
                m.insert(name, rows[rows.len() - width..].to_vec());
            }
        };
        for il in self.first..self.last {
            if c.ngram.as_ref().is_some_and(|p| p.layer == il) {
                let emb = ple.ok_or_else(|| anyhow::anyhow!("layer {il} needs the n-gram rows"))?;
                ensure!(emb.len() == n * c.d, "n-gram rows length mismatch");
                record("ple_embd".into(), emb, c.d);
                Self::ple(g, &c, il, x, emb, n, &mut self.ple_conv, &mut record)?;
            }
            let m = Self::hc_mix(
                g,
                &c,
                &format!("blk.{il}.hc_attn"),
                x,
                n,
                &format!("-{il}"),
                &mut record,
            )?;
            let y = match &mut self.states[il - self.first] {
                State::Recurrent { conv, s } => Self::delta_net(g, &c, il, &m.input, n, conv, s, &mut record)?,
                State::Attention { k, v, kraw, pooled } => {
                    Self::attention(g, &c, il, &m.input, pos, n, k, v, kraw, pooled, 0, &mut record)?
                }
            };
            Self::combine(&c, x, &y, m.write.as_deref().unwrap(), n);
            record(format!("hc_combine-{il}"), x, w);
            let m = Self::hc_mix(
                g,
                &c,
                &format!("blk.{il}.hc_ffn"),
                x,
                n,
                &format!("-{il}#2"),
                &mut record,
            )?;
            let y = Self::moe(g, &c, il, &m.input, n, &mut record)?;
            record(format!("ffn_out-{il}"), &y, c.d);
            Self::combine(&c, x, &y, m.write.as_deref().unwrap(), n);
            record(format!("l_last-{il}"), x, w);
        }
        Ok(())
    }

    /// x_s += w_s * y for every stream s.
    pub(crate) fn combine(c: &Config, x: &mut [f32], y: &[f32], write: &[f32], n: usize) {
        for t in 0..n {
            for s in 0..c.hc {
                let ws = write[t * c.hc + s];
                let row = &mut x[(t * c.hc + s) * c.d..(t * c.hc + s + 1) * c.d];
                for (xv, yv) in row.iter_mut().zip(&y[t * c.d..(t + 1) * c.d]) {
                    *xv += ws * yv;
                }
            }
        }
    }

    /// Hyper-connection mixer `prefix_{norm,down,up,inject}`: RMS-norm each stream with its gamma, a read gate
    /// through the rank bottleneck, the mean of the gated streams, and the write weights `2 sigmoid(inject / hc)`.
    pub(crate) fn hc_mix(
        g: &Gguf,
        c: &Config,
        prefix: &str,
        x: &[f32],
        n: usize,
        tag: &str,
        record: &mut impl FnMut(String, &[f32], usize),
    ) -> Result<Mixed> {
        let t = |name: &str| -> Result<&TensorInfo> {
            let full = format!("{prefix}_{name}.weight");
            g.tensor(&full).ok_or_else(|| anyhow::anyhow!("missing {full}"))
        };
        let (d, hc, w) = (c.d, c.hc, c.width());
        let gamma = vector(g, t("norm")?)?;
        let mut xn = x.to_vec();
        for (i, row) in xn.chunks_exact_mut(d).enumerate() {
            let s = i % hc;
            rms_norm(row, Some(&gamma[s * d..(s + 1) * d]), c.eps);
        }
        record(format!("hc_norm{tag}"), &xn, w);
        let lo: Vec<f32> = matmul(g, t("down")?, &xn, n)
            .into_iter()
            .map(|v| silu(v / hc as f32))
            .collect();
        let gate = matmul(g, t("up")?, &lo, n);
        record(format!("hc_gate{tag}"), &gate, w);
        let mut input = vec![0f32; n * d];
        for tok in 0..n {
            for s in 0..hc {
                let base = tok * w + s * d;
                for i in 0..d {
                    input[tok * d + i] += xn[base + i] * sigmoid(gate[base + i]);
                }
            }
            for v in &mut input[tok * d..(tok + 1) * d] {
                *v /= hc as f32;
            }
        }
        record(format!("hc_mixed{tag}"), &input, d);
        let write = if tag.is_empty() {
            None
        } else {
            let inject = matmul(g, t("inject")?, &xn, n);
            record(format!("hc_inject{tag}"), &inject, hc);
            Some(inject.iter().map(|v| 2. * sigmoid(v / hc as f32)).collect())
        };
        Ok(Mixed { input, write })
    }

    /// The n-gram module: per-stream sigmoid gates of a shared value, then a dilated causal depthwise convolution
    /// of the normed gated value; both are added to the residual.
    #[allow(clippy::too_many_arguments)]
    fn ple(
        g: &Gguf,
        c: &Config,
        il: usize,
        x: &mut [f32],
        emb: &[f32],
        n: usize,
        hist: &mut [f32],
        record: &mut impl FnMut(String, &[f32], usize),
    ) -> Result<()> {
        let p = c.ngram.as_ref().unwrap();
        let t = |name: &str| c.layer_tensor(g, il, name);
        let v = |name: &str| -> Result<Vec<f32>> { vector(g, t(name)?) };
        let (d, hc, w) = (c.d, c.hc, c.width());
        let mut key = matmul(g, t("ple_key.weight")?, emb, n);
        let value = matmul(g, t("ple_value.weight")?, emb, n);
        let (nk, nq, nc) = (
            v("ple_norm_key.weight")?,
            v("ple_norm_query.weight")?,
            v("ple_norm_conv.weight")?,
        );
        let mut gate = vec![0f32; n * hc];
        let mut gated = vec![0f32; n * w];
        for tok in 0..n {
            for s in 0..hc {
                let r = tok * w + s * d;
                rms_norm(&mut key[r..r + d], Some(&nk[s * d..(s + 1) * d]), c.eps);
                let mut q = x[r..r + d].to_vec();
                rms_norm(&mut q, Some(&nq[s * d..(s + 1) * d]), c.eps);
                let a = dot(&key[r..r + d], &q) / (d as f32).sqrt();
                let gs = sigmoid(a.signum() * a.abs().max(1e-6).sqrt() * f32::from(a != 0.));
                gate[tok * hc + s] = gs;
                for i in 0..d {
                    gated[r + i] = gs * value[tok * d + i];
                }
            }
        }
        record(format!("ple_gate-{il}"), &gate, hc);
        record(format!("ple_gated_value-{il}"), &gated, w);
        let mut normed = gated.clone();
        for (i, row) in normed.chunks_exact_mut(d).enumerate() {
            let s = i % hc;
            rms_norm(row, Some(&nc[s * d..(s + 1) * d]), c.eps);
        }
        // Taps t - (conv-1-k) * ngram over [history | this batch].
        let kw = vector(g, t("ple_conv1d.weight")?)?;
        let span = (p.conv - 1) * p.ngram;
        let input = |i: usize, ch: usize| -> f32 {
            if i < span {
                hist[i * w + ch]
            } else {
                normed[(i - span) * w + ch]
            }
        };
        let mut conv = vec![0f32; n * w];
        for tok in 0..n {
            for ch in 0..w {
                let mut sum = 0.;
                for k in 0..p.conv {
                    sum += kw[ch * p.conv + k] * input(tok + span - (p.conv - 1 - k) * p.ngram, ch);
                }
                conv[tok * w + ch] = silu(sum);
            }
        }
        let next: Vec<f32> = (n..n + span)
            .flat_map(|i| (0..w).map(move |ch| (i, ch)))
            .map(|(i, ch)| input(i, ch))
            .collect();
        hist.copy_from_slice(&next);
        record(format!("ple_conv_out-{il}"), &conv, w);
        for ((xv, gv), cv) in x.iter_mut().zip(&gated).zip(&conv) {
            *xv += gv + cv;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn delta_net(
        g: &Gguf,
        c: &Config,
        il: usize,
        h: &[f32],
        n: usize,
        conv: &mut [f32],
        s: &mut [f32],
        record: &mut impl FnMut(String, &[f32], usize),
    ) -> Result<Vec<f32>> {
        let t = |name: &str| c.layer_tensor(g, il, name);
        let (cd, kd, vd, sd) = (c.conv_dim(), c.key_dim(), c.value_dim(), c.state);
        let mixed = matmul(g, t("attn_qkv.weight")?, h, n);
        record(format!("linear_attn_qkv_mixed-{il}"), &mixed, cd);
        let z = matmul(g, t("attn_gate.weight")?, h, n);
        record(format!("z-{il}"), &z, vd);
        let beta: Vec<f32> = matmul(g, t("ssm_beta.weight")?, h, n)
            .into_iter()
            .map(sigmoid)
            .collect();
        record(format!("beta_sigmoid-{il}"), &beta, c.v_heads);
        let alpha = matmul(g, t("ssm_alpha.weight")?, h, n);
        let dt = vector(g, t("ssm_dt.bias")?)?;
        let a = vector(g, t("ssm_a")?)?;
        let gate: Vec<f32> = alpha
            .iter()
            .enumerate()
            .map(|(i, x)| softplus(x + dt[i % c.v_heads]) * a[i % c.v_heads])
            .collect();
        record(format!("gate-{il}"), &gate, c.v_heads);

        let w = vector(g, t("ssm_conv1d.weight")?)?;
        let hist = c.conv - 1;
        let input = |i: usize, ch: usize| -> f32 {
            if i < hist {
                conv[i * cd + ch]
            } else {
                mixed[(i - hist) * cd + ch]
            }
        };
        let mut qkv = vec![0f32; n * cd];
        for tok in 0..n {
            for ch in 0..cd {
                let mut sum = 0.;
                for k in 0..c.conv {
                    sum += w[ch * c.conv + k] * input(tok + k, ch);
                }
                qkv[tok * cd + ch] = silu(sum);
            }
        }
        let next: Vec<f32> = (n..n + hist)
            .flat_map(|i| (0..cd).map(move |ch| (i, ch)))
            .map(|(i, ch)| input(i, ch))
            .collect();
        conv.copy_from_slice(&next);
        record(format!("conv_output_silu-{il}"), &qkv, cd);

        let mut out = vec![0f32; n * vd];
        let scale = 1. / (sd as f32).sqrt();
        // L2-normalized q and k of every token and key head, `[tok][q | k]`.
        let mut qk = vec![0f32; n * 2 * kd];
        for tok in 0..n {
            for part in 0..2 {
                for kh in 0..c.k_heads {
                    let src = &qkv[tok * cd + part * kd + kh * sd..][..sd];
                    let inv = 1. / (src.iter().map(|x| x * x).sum::<f32>() + c.eps).sqrt();
                    for (d, x) in qk[tok * 2 * kd + part * kd + kh * sd..][..sd].iter_mut().zip(src) {
                        *d = x * inv;
                    }
                }
            }
        }
        record(
            format!("q_conv_predelta-{il}"),
            &qk[(n - 1) * 2 * kd..(n - 1) * 2 * kd + kd],
            kd,
        );
        record(format!("k_conv_predelta-{il}"), &qk[(n - 1) * 2 * kd + kd..], kd);
        // Value heads evolve independently: one thread per group of heads, tokens in order. Value head vh reads
        // key head vh % k_heads (ggml tiles the key heads; the converter reorders the value heads to match).
        let threads = std::thread::available_parallelism()
            .map_or(8, |n| n.get())
            .min(c.v_heads);
        let mut heads: Vec<(usize, &mut [f32])> = s.chunks_mut(sd * sd).enumerate().collect();
        let per = c.v_heads.div_ceil(threads);
        let results: Vec<(usize, Vec<f32>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = heads
                .chunks_mut(per)
                .map(|group| {
                    let (qk, qkv, gate, beta) = (&qk, &qkv, &gate, &beta);
                    scope.spawn(move || {
                        let mut res = Vec::new();
                        for (vh, m) in group.iter_mut() {
                            let vh = *vh;
                            let kh = vh % c.k_heads;
                            let mut o = vec![0f32; n * sd];
                            for tok in 0..n {
                                let q = &qk[tok * 2 * kd + kh * sd..][..sd];
                                let k = &qk[tok * 2 * kd + kd + kh * sd..][..sd];
                                let v = &qkv[tok * cd + 2 * kd + vh * sd..][..sd];
                                let decay = gate[tok * c.v_heads + vh].exp();
                                let b = beta[tok * c.v_heads + vh];
                                for i in 0..sd {
                                    let r = &mut m[i * sd..(i + 1) * sd];
                                    for x in r.iter_mut() {
                                        *x *= decay;
                                    }
                                    let delta = (v[i] - dot(r, k)) * b;
                                    for (x, kj) in r.iter_mut().zip(k) {
                                        *x += kj * delta;
                                    }
                                    o[tok * sd + i] = dot(r, q) * scale;
                                }
                            }
                            res.push((vh, o));
                        }
                        res
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        for (vh, o) in results {
            for tok in 0..n {
                out[tok * vd + vh * sd..][..sd].copy_from_slice(&o[tok * sd..(tok + 1) * sd]);
            }
        }
        record(format!("attn_output-{il}"), &out, vd);

        // Per-head RMS norm (shared weight), gated by sigmoid(z): the one difference from qwen35's DeltaNet.
        let norm = vector(g, t("ssm_norm.weight")?)?;
        for (head, zh) in out.chunks_exact_mut(sd).zip(z.chunks_exact(sd)) {
            rms_norm(head, Some(&norm), c.eps);
            for (o, zz) in head.iter_mut().zip(zh) {
                *o *= sigmoid(*zz);
            }
        }
        record(format!("final_output-{il}"), &out, vd);
        let y = matmul(g, t("ssm_out.weight")?, &out, n);
        record(format!("linear_attn_out-{il}"), &y, c.d);
        Ok(y)
    }

    /// Qwen Sparse Attention: gated GQA over the complete 4-token blocks the indexer ranks highest (all of them
    /// while at most `idx_budget / kpool` are complete) plus the incomplete tail.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention(
        g: &Gguf,
        c: &Config,
        il: usize,
        h: &[f32],
        pos: usize,
        n: usize,
        kc: &mut [f32],
        vc: &mut [f32],
        kraw: &mut [f32],
        pooled: &mut [f32],
        kv_first: usize,
        record: &mut impl FnMut(String, &[f32], usize),
    ) -> Result<Vec<f32>> {
        let t = |name: &str| c.layer_tensor(g, il, name);
        let (hd, nh, nkv, id, ih, kp) = (c.head_dim, c.heads, c.kv_heads, c.idx_dim, c.idx_heads, c.kpool);
        ensure!((pos + n) * nkv * hd <= kc.len(), "context exhausted");

        // Indexer: raw keys are cached; a block's pooled key is the mean of its raw keys, normed and rotated to
        // the block's first position.
        let k_raw = matmul(g, t("indexer.k_proj.weight")?, h, n);
        record(format!("indexer_k_raw-{il}"), &k_raw, id);
        kraw[pos * id..(pos + n) * id].copy_from_slice(&k_raw);
        round_f16(&mut kraw[pos * id..(pos + n) * id]);
        let k_norm = vector(g, t("indexer.k_norm.weight")?)?;
        for b in (pos + 1) / kp..(pos + n + 1) / kp {
            let mut m = vec![0f32; id];
            for p in b * kp..(b + 1) * kp {
                for (a, v) in m.iter_mut().zip(&kraw[p * id..(p + 1) * id]) {
                    *a += v;
                }
            }
            for a in &mut m {
                *a /= kp as f32;
            }
            rms_norm(&mut m, Some(&k_norm), c.eps);
            rotate(c, &mut m, b * kp);
            round_f16(&mut m);
            pooled[b * id..(b + 1) * id].copy_from_slice(&m);
        }
        let mut iq = matmul(g, t("indexer.q_proj.weight")?, h, n);
        let q_norm = vector(g, t("indexer.q_norm.weight")?)?;
        for tok in 0..n {
            for hh in 0..ih {
                let r = &mut iq[(tok * ih + hh) * id..(tok * ih + hh + 1) * id];
                rms_norm(r, Some(&q_norm), c.eps);
                rotate(c, r, pos + tok);
            }
        }
        record(format!("indexer_q-{il}"), &iq, ih * id);

        let qg = matmul(g, t("attn_q.weight")?, h, n);
        let k = matmul(g, t("attn_k.weight")?, h, n);
        let v = matmul(g, t("attn_v.weight")?, h, n);
        let qn = vector(g, t("attn_q_norm.weight")?)?;
        let kn = vector(g, t("attn_k_norm.weight")?)?;
        let mut q = vec![0f32; n * nh * hd];
        let mut gate = vec![0f32; n * nh * hd];
        for tok in 0..n {
            for head in 0..nh {
                let src = &qg[(tok * nh + head) * 2 * hd..];
                let dst = (tok * nh + head) * hd;
                q[dst..dst + hd].copy_from_slice(&src[..hd]);
                gate[dst..dst + hd].copy_from_slice(&src[hd..2 * hd]);
                rms_norm(&mut q[dst..dst + hd], Some(&qn), c.eps);
                rotate(c, &mut q[dst..dst + hd], pos + tok);
            }
            for head in 0..nkv {
                let at = (tok * nkv + head) * hd;
                let mut kk = k[at..at + hd].to_vec();
                rms_norm(&mut kk, Some(&kn), c.eps);
                rotate(c, &mut kk, pos + tok);
                let slot = ((pos + tok) * nkv + head) * hd;
                kc[slot..slot + hd].copy_from_slice(&kk);
                vc[slot..slot + hd].copy_from_slice(&v[at..at + hd]);
                round_kv(&mut kc[slot..slot + hd], false);
                round_kv(&mut vc[slot..slot + hd], true);
            }
        }
        record(format!("Qcur-{il}"), &q, nh * hd);
        // Scores of the last token's complete blocks, padded with -inf to llama.cpp's pool count (multiples of 64).
        let complete = (pos + n) / kp;
        let mut scores = Self::block_scores(c, &iq[(n - 1) * ih * id..], pooled, complete);
        scores.resize((complete + 1).next_multiple_of(64).max(64), f32::NEG_INFINITY);
        let width = scores.len();
        record(format!("indexer_score-{il}"), &scores, width);
        let kpos = (pos + n - 1) * nkv * hd;
        record(format!("Kcur-{il}#2"), &kc[kpos..kpos + nkv * hd], nkv * hd);

        let scale = 1. / (hd as f32).sqrt();
        let mut o = vec![0f32; n * nh * hd];
        let (kc, vc, pooled, iq, q) = (&*kc, &*vc, &*pooled, &iq, &q);
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        std::thread::scope(|scope| {
            for (part, rows) in o.chunks_mut(n.div_ceil(threads) * nh * hd).enumerate() {
                scope.spawn(move || {
                    let first = part * n.div_ceil(threads);
                    for (i, row) in rows.chunks_exact_mut(nh * hd).enumerate() {
                        let tok = first + i;
                        let mut visible = Self::select(c, &iq[tok * ih * id..(tok + 1) * ih * id], pooled, pos + tok);
                        // Positions before `kv_first` hold no keys (an MTP block filled from a later position).
                        visible.retain(|&j| j >= kv_first);
                        for (head, dst) in row.chunks_exact_mut(hd).enumerate() {
                            let kh = head / (nh / nkv);
                            let mut qv = q[(tok * nh + head) * hd..][..hd].to_vec();
                            round_f16(&mut qv);
                            let scores: Vec<f32> = visible
                                .iter()
                                .map(|&j| dot(&qv, &kc[(j * nkv + kh) * hd..][..hd]) * scale)
                                .collect();
                            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let w: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                            let sum: f32 = w.iter().sum();
                            let mut probs: Vec<f32> = w.iter().map(|x| x / sum).collect();
                            round_f16(&mut probs);
                            for (&j, pj) in visible.iter().zip(&probs) {
                                for (x, vv) in dst.iter_mut().zip(&vc[(j * nkv + kh) * hd..][..hd]) {
                                    *x += pj * vv;
                                }
                            }
                        }
                    }
                });
            }
        });
        record(format!("attn_pregate-{il}"), &o, nh * hd);
        for (x, gt) in o.iter_mut().zip(&gate) {
            *x *= sigmoid(*gt);
        }
        record(format!("attn_gated-{il}"), &o, nh * hd);
        let y = matmul(g, t("attn_output.weight")?, &o, n);
        record(format!("attn_output-{il}"), &y, c.d);
        Ok(y)
    }

    /// Positions a token at `p` attends to: every complete block while there are at most `top_blocks`, otherwise
    /// the best-scoring ones (score = sum over indexer heads of relu(q . pooled) / sqrt(idx_dim); ties go to the
    /// earlier block), then the incomplete tail ending at `p`.
    /// Indexer scores of the first `complete` blocks for one token's indexer query (heads x idx_dim).
    pub(crate) fn block_scores(c: &Config, iq: &[f32], pooled: &[f32], complete: usize) -> Vec<f32> {
        let id = c.idx_dim;
        let scale = 1. / (id as f32).sqrt();
        (0..complete)
            .map(|b| {
                let kb = &pooled[b * id..(b + 1) * id];
                iq.chunks_exact(id).map(|q| dot(q, kb).max(0.)).sum::<f32>() * scale
            })
            .collect()
    }

    pub(crate) fn select(c: &Config, iq: &[f32], pooled: &[f32], p: usize) -> Vec<usize> {
        let kp = c.kpool;
        let complete = (p + 1) / kp;
        let mut blocks: Vec<usize> = (0..complete).collect();
        if complete > c.top_blocks() {
            let scores = Self::block_scores(c, iq, pooled, complete);
            blocks.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
            blocks.truncate(c.top_blocks());
            blocks.sort_unstable();
        }
        let mut out: Vec<usize> = blocks.iter().flat_map(|b| b * kp..(b + 1) * kp).collect();
        out.extend(complete * kp..=p);
        out
    }

    /// Softmax router over the experts, top `used` renormalized, SwiGLU experts, plus the sigmoid-gated shared
    /// expert. Experts run in parallel; each token sums its experts in rank order.
    pub(crate) fn moe(
        g: &Gguf,
        c: &Config,
        il: usize,
        u: &[f32],
        n: usize,
        record: &mut impl FnMut(String, &[f32], usize),
    ) -> Result<Vec<f32>> {
        let t = |name: &str| c.layer_tensor(g, il, name);
        let (d, ne, k, ff) = (c.d, c.experts, c.used, c.ff_exp);
        let logits = matmul(g, t("ffn_gate_inp.weight")?, u, n);
        record(format!("ffn_moe_logits-{il}"), &logits, ne);
        let mut probs = logits.clone();
        for row in probs.chunks_exact_mut(ne) {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        record(format!("ffn_moe_probs-{il}"), &probs, ne);
        // (expert, weight) per token, in rank order.
        let mut chosen = Vec::with_capacity(n * k);
        for row in probs.chunks_exact(ne) {
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.sort_by(|a, b| row[*b].total_cmp(&row[*a]).then(a.cmp(b)));
            let sum = idx[..k].iter().map(|&e| row[e]).sum::<f32>().max(6.103515625e-5);
            chosen.extend(idx[..k].iter().map(|&e| (e, row[e] / sum)));
        }
        record(
            format!("ffn_moe_weights_norm-{il}"),
            &chosen.iter().map(|x| x.1).collect::<Vec<_>>(),
            k,
        );
        // Tokens of each expert, then every expert on one thread.
        let mut work: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (slot, (e, _)) in chosen.iter().enumerate() {
            work.entry(*e).or_default().push(slot);
        }
        let work: Vec<(usize, Vec<usize>)> = work.into_iter().collect();
        let (gate_t, up_t, down_t) = (
            t("ffn_gate_exps.weight")?,
            t("ffn_up_exps.weight")?,
            t("ffn_down_exps.weight")?,
        );
        ensure!(
            down_t.dims[0] as usize == ff,
            "padded expert down projections are not supported here"
        );
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let results: Vec<Vec<(usize, Vec<f32>, Vec<f32>)>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|i| {
                    let work = &work;
                    s.spawn(move || {
                        let mut out = Vec::new();
                        for (e, slots) in work.iter().skip(i).step_by(threads) {
                            let x: Vec<f32> = slots
                                .iter()
                                .flat_map(|slot| u[slot / k * d..(slot / k + 1) * d].to_vec())
                                .collect();
                            let m = slots.len();
                            let gt = matvec_rows(g, gate_t, e * ff, ff, &x, m);
                            let up = matvec_rows(g, up_t, e * ff, ff, &x, m);
                            let act: Vec<f32> = gt.iter().zip(&up).map(|(a, b)| silu(*a) * b).collect();
                            let down = matvec_rows(g, down_t, e * d, d, &act, m);
                            for (j, slot) in slots.iter().enumerate() {
                                out.push((
                                    *slot,
                                    act[j * ff..(j + 1) * ff].to_vec(),
                                    down[j * d..(j + 1) * d].to_vec(),
                                ));
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut act = vec![0f32; n * k * ff];
        let mut down = vec![0f32; n * k * d];
        for (slot, a, y) in results.into_iter().flatten() {
            act[slot * ff..(slot + 1) * ff].copy_from_slice(&a);
            down[slot * d..(slot + 1) * d].copy_from_slice(&y);
        }
        record(format!("ffn_moe_swiglu-{il}"), &act, k * ff);
        record(format!("ffn_moe_down-{il}"), &down, k * d);
        let mut out = vec![0f32; n * d];
        for tok in 0..n {
            for j in 0..k {
                let (_, wgt) = chosen[tok * k + j];
                let y = &down[(tok * k + j) * d..(tok * k + j + 1) * d];
                for (o, v) in out[tok * d..(tok + 1) * d].iter_mut().zip(y) {
                    *o += wgt * v;
                }
            }
        }
        record(format!("ffn_moe_out-{il}"), &out, d);

        let gs = matmul(g, t("ffn_gate_shexp.weight")?, u, n);
        let us = matmul(g, t("ffn_up_shexp.weight")?, u, n);
        let a: Vec<f32> = gs.iter().zip(&us).map(|(a, b)| silu(*a) * b).collect();
        let sh = matmul(g, t("ffn_down_shexp.weight")?, &a, n);
        record(format!("ffn_shexp-{il}"), &sh, d);
        let gw = vector(g, t("ffn_gate_inp_shexp.weight")?)?;
        for tok in 0..n {
            let sg = sigmoid(dot(&gw, &u[tok * d..(tok + 1) * d]));
            for i in 0..d {
                out[tok * d + i] += sg * sh[tok * d + i];
            }
        }
        Ok(out)
    }

    /// Final mixer (the output norm) and full logits of the last token.
    pub fn logits(&self, g: &Gguf, x: &[f32], cap: &mut Option<&mut Captures>) -> Result<Vec<f32>> {
        let w = self.c.width();
        let last = &x[x.len() - w..];
        let mut record = |name: String, rows: &[f32], width: usize| {
            if let Some(m) = cap.as_mut() {
                m.insert(name, rows[rows.len() - width..].to_vec());
            }
        };
        let m = Self::hc_mix(g, &self.c, "output_hc", last, 1, "", &mut record)?;
        record("result_norm".into(), &m.input, self.c.d);
        Ok(matmul(g, g.tensor("output.weight").unwrap(), &m.input, 1))
    }

    /// The final hidden state of the last token (the final mixer's output), which the head reads.
    pub fn final_hidden(&self, g: &Gguf, x: &[f32], cap: &mut Option<&mut Captures>) -> Result<Vec<f32>> {
        let w = self.c.width();
        let mut record = |name: String, rows: &[f32], width: usize| {
            if let Some(m) = cap.as_mut() {
                m.insert(name, rows[rows.len() - width..].to_vec());
            }
        };
        let m = Self::hc_mix(g, &self.c, "output_hc", &x[x.len() - w..], 1, "", &mut record)?;
        record("result_norm".into(), &m.input, self.c.d);
        Ok(m.input)
    }

    /// Logits of the head rows `row0..row1` for a final hidden state.
    pub fn head_logits(g: &Gguf, h: &[f32], row0: usize, row1: usize) -> Vec<f32> {
        matmul(g, g.tensor("output.weight").unwrap(), h, 1)[row0..row1].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            d: 8,
            layers: 4,
            heads: 2,
            kv_heads: 1,
            head_dim: 8,
            rot: 4,
            rope_base: 1e4,
            eps: 1e-6,
            interval: 4,
            conv: 4,
            k_heads: 1,
            v_heads: 1,
            state: 8,
            vocab: 16,
            hc: 4,
            hc_rank: 2,
            experts: 4,
            used: 2,
            ff_exp: 4,
            ff_shexp: 4,
            idx_heads: 1,
            idx_dim: 4,
            idx_budget: 8,
            kpool: 4,
            ngram: None,
        }
    }

    #[test]
    fn selection_is_dense_until_the_budget_then_ranks_blocks() {
        let c = config();
        let pooled: Vec<f32> = (0..4).flat_map(|b| [b as f32, 0., 0., 0.]).collect();
        // Up to two complete blocks (8 tokens) plus the tail: every visible position.
        assert_eq!(
            CpuStage::select(&c, &[1., 0., 0., 0.], &pooled, 10),
            (0..=10).collect::<Vec<_>>()
        );
        // Three complete blocks: keep the two best (blocks 2 and 1), then the tail.
        assert_eq!(
            CpuStage::select(&c, &[1., 0., 0., 0.], &pooled, 12),
            vec![4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
        // A negative query rectifies every score to zero: ties keep the earliest blocks.
        assert_eq!(
            CpuStage::select(&c, &[-1., 0., 0., 0.], &pooled, 11),
            (0..8).collect::<Vec<_>>()
        );
    }
}
