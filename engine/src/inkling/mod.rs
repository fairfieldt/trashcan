//! Inkling 2 Mini (`inkling2` GGUFs from Unsloth, `inkling2_loop` from upstream llama.cpp): a looped MoE
//! transformer. 24 layers of weights; layers `core_start..core_end` run `passes` times, so a token takes 48 layer
//! applications ("slots", each with its own K/V cache and convolution state). Block attention residuals replace the
//! summed residual, attention has no RoPE but a learned relative-position bias, short causal convolutions act on K,
//! V and both sublayer outputs, and passes after the first gate their inputs with messages from the previous pass.
//! docs/inkling.md has the math; llama.cpp's `src/models/inkling2*.cpp` is the reference.
//!
//! The GPUs run the projections, attention and experts tensor-parallel (`gpu`); the host keeps the residual, the
//! block states, the residual mixes, the sublayer-output convolutions and the loop controllers (`group`).
pub mod chat;
pub mod cli;
pub mod draft;
pub mod spec;
pub mod gpu;
pub mod group;
pub mod vision;


use crate::gguf::{GgmlType, Gguf, TensorInfo};
use anyhow::{bail, ensure, Context, Result};

/// Which GGUF layout the file uses: Unsloth's root files or upstream llama.cpp's `inkling2_loop`. The tensors are
/// the same bytes; the loop tensors are named differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Unsloth,
    Upstream,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub layout: Layout,
    pub d: usize,
    /// Physical layers (24).
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rel: usize,
    pub rel_extent: usize,
    pub window: usize,
    pub experts: usize,
    pub used: usize,
    pub ff: usize,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub conv: usize,
    pub block: usize,
    pub route_scale: f32,
    pub situ_beta: f32,
    pub situ_linear: f32,
    pub softcap: f32,
    /// Rows of the embedding and head, and the tokens of the vocabulary (the rest is padding).
    pub vocab_padded: usize,
    pub vocab: usize,
    pub core_start: usize,
    pub core_end: usize,
    pub passes: usize,
    pub gate_rank: usize,
}

/// One layer application in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct App {
    /// Cache slot (llama.cpp's layer index `il`): physical index for the prelude, the first pass and the coda, then
    /// `layers + (pass - 1) * core + (phys - core_start)` for later passes.
    pub slot: usize,
    pub phys: usize,
    /// Loop pass (0 outside the core).
    pub pass: usize,
}

impl Config {
    pub fn new(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        let layout = match arch {
            "inkling2" => Layout::Unsloth,
            "inkling2_loop" => Layout::Upstream,
            other => bail!("expected an inkling2 model, found {other}"),
        };
        let k = |s: &str| format!("{arch}.{s}");
        let u = |s: &str| -> Result<usize> { Ok(g.get_u64(&k(s))? as usize) };
        let f = |s: &str| -> Result<f32> { Ok(g.get_f64(&k(s))? as f32) };
        let first = |s: &str| -> Result<usize> {
            let v = g.get(&k(s))?;
            Ok(match v.as_arr() {
                Some(a) => a.first().and_then(|x| x.as_u64()).context("empty array")? as usize,
                None => v.as_u64().context("not a number")? as usize,
            })
        };
        let (core_start, core_end, passes, gate_rank) = match layout {
            Layout::Unsloth => (u("loop.core_start")?, u("loop.core_end")?, u("num_loops")?, u("loop.gate_rank")?),
            Layout::Upstream => (u("loop.start")?, u("loop.end")?, u("loop.count")?, u("loop.gate_rank")?),
        };
        let blocks = u("block_count")?;
        let layers = match layout {
            Layout::Unsloth => blocks - (passes - 1) * (core_end - core_start),
            Layout::Upstream => blocks,
        };
        let c = Self {
            layout,
            d: u("embedding_length")?,
            layers,
            heads: first("attention.head_count")?,
            kv_heads: first("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rel: u("attention.relative_length")?,
            rel_extent: u("attention.relative_extent")?,
            window: u("attention.sliding_window")?,
            experts: u("expert_count")?,
            used: u("expert_used_count")?,
            ff: u("expert_feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            log_floor: f("attention.log_scale_floor")?,
            log_alpha: f("attention.log_scale_alpha")?,
            conv: u("shortconv.l_cache")?,
            block: u("attn_res.block_size")?,
            route_scale: f("expert_weights_scale")?,
            situ_beta: f("activation.situ_beta")?,
            situ_linear: f("activation.situ_linear_beta")?,
            softcap: f("final_logit_softcapping").unwrap_or(0.),
            vocab_padded: u("vocab_size")?,
            vocab: u("vocab_size_unpadded")?,
            core_start,
            core_end,
            passes,
            gate_rank,
        };
        ensure!(
            c.d == 2304 && c.head_dim == 128 && c.conv == 4 && c.rel == 16 && c.heads % c.kv_heads == 0
                && c.core_start % c.block == 0 && c.core_end % c.block == 0 && c.core_end <= c.layers && c.passes >= 2
                && c.window <= c.rel_extent && c.rel_extent == 1024,
            "unsupported inkling2 shape {c:?}"
        );
        // Every application of a physical layer has the same kind, as the pattern says.
        let pattern = g.get(&k("attention.sliding_window_pattern"))?.as_arr().context("sliding window pattern")?;
        for a in c.schedule() {
            let idx = if layout == Layout::Upstream { a.phys } else { a.slot };
            let swa = pattern.get(idx).and_then(|v| v.as_u64()).context("short sliding window pattern")? != 0;
            ensure!(swa == c.swa(a.phys), "unexpected sliding window pattern at slot {}", a.slot);
        }
        Ok(c)
    }

    /// Sliding-window layers; the others (every `block`-th, 5, 11, 17, 23) are global.
    pub fn swa(&self, phys: usize) -> bool {
        phys % self.block != self.block - 1
    }
    pub fn core(&self) -> usize {
        self.core_end - self.core_start
    }
    pub fn slots(&self) -> usize {
        self.layers + (self.passes - 1) * self.core()
    }
    pub fn looped(&self, phys: usize) -> bool {
        (self.core_start..self.core_end).contains(&phys)
    }
    /// The layer applications of one token in order.
    pub fn schedule(&self) -> Vec<App> {
        let mut v: Vec<App> = (0..self.core_start).map(|p| App { slot: p, phys: p, pass: 0 }).collect();
        for pass in 0..self.passes {
            for p in self.core_start..self.core_end {
                let slot = if pass == 0 { p } else { self.layers + (pass - 1) * self.core() + p - self.core_start };
                v.push(App { slot, phys: p, pass });
            }
        }
        v.extend((self.core_end..self.layers).map(|p| App { slot: p, phys: p, pass: 0 }));
        v
    }
    pub fn t<'a>(&self, g: &'a Gguf, name: &str) -> Result<&'a TensorInfo> {
        g.tensor(name).with_context(|| format!("missing {name}"))
    }
    pub fn lt<'a>(&self, g: &'a Gguf, phys: usize, name: &str) -> Result<&'a TensorInfo> {
        self.t(g, &format!("blk.{phys}.{name}"))
    }
    /// The name of a loop controller tensor: `ctrl` 0..4 = q, k, v, ffn; `part` "down", "gain" or "forget".
    pub fn ctrl_name(&self, phys: usize, ctrl: usize, part: &str) -> String {
        match self.layout {
            Layout::Upstream => format!("blk.{phys}.loop_{}_ctrl_{part}.weight", ["q", "k", "v", "ffn"][ctrl]),
            Layout::Unsloth => format!("blk.{phys}.loop_{}_{part}.weight", ["attn", "k", "v", "mlp"][ctrl]),
        }
    }
    /// The four message queries (q, k, v, ffn) of core layer `phys`.
    pub fn msg_queries(&self, g: &Gguf, phys: usize) -> Result<[Vec<f32>; 4]> {
        let d = self.d;
        match self.layout {
            Layout::Upstream => {
                let all = vector(g, self.lt(g, phys, "loop_msg_query.weight")?)?;
                ensure!(all.len() == 4 * d, "loop_msg_query shape");
                Ok(std::array::from_fn(|c| all[c * d..(c + 1) * d].to_vec()))
            }
            Layout::Unsloth => {
                let q = vector(g, self.t(g, "loop_res_query.weight")?)?;
                let kv = vector(g, self.t(g, "loop_kv_res_query.weight")?)?;
                let i = phys - self.core_start;
                ensure!(q.len() == 2 * self.core() * d && kv.len() == q.len(), "loop query shapes");
                let row = |v: &[f32], r: usize| v[r * d..(r + 1) * d].to_vec();
                Ok([row(&q, 2 * i), row(&kv, 2 * i), row(&kv, 2 * i + 1), row(&q, 2 * i + 1)])
            }
        }
    }
}

/// Dequantize one row of a 2-D tensor (rows of `dims[0]` elements).
pub fn dequant_row(g: &Gguf, t: &TensorInfo, row: usize, out: &mut [f32]) -> Result<()> {
    if t.ty == GgmlType::NVFP4 {
        let k = t.dims[0] as usize;
        ensure!(out.len() == k && k % 64 == 0, "row length mismatch for {}", t.name);
        let per_row = k / 64 * 36;
        let data = &g.data(t)[row * per_row..(row + 1) * per_row];
        for (o, b) in out.chunks_exact_mut(64).zip(data.chunks_exact(36)) {
            o.copy_from_slice(&crate::quant::nvfp4_block(b.try_into().unwrap()));
        }
        return Ok(());
    }
    crate::qwen::dequant_row(g, t, row, out)
}

/// A whole tensor as f32 (norms, small f32 or BF16 matrices).
pub fn vector(g: &Gguf, t: &TensorInfo) -> Result<Vec<f32>> {
    crate::qwen::vector(g, t)
}

/// RMS norm of each row of `x` (width `w.len()` or `d`) times `w` (none: no weight), eps 1e-6 as llama.cpp's
/// `ggml_rms_norm` uses here. Eight running sums in a fixed order, so every machine gets the same bits (with or
/// without AVX: `group` inlines these loops into functions compiled for it).
#[inline(always)]
pub fn rms_into(x: &[f32], w: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    let d = out.len();
    debug_assert_eq!(x.len(), d);
    let mut acc = [0f32; 8];
    for c in x.chunks_exact(8) {
        for j in 0..8 {
            acc[j] += c[j] * c[j];
        }
    }
    let ss = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    let inv = 1.0 / (ss / d as f32 + eps).sqrt();
    match w {
        Some(w) => out.iter_mut().zip(x).zip(w).for_each(|((o, &v), &wv)| *o = v * inv * wv),
        None => out.iter_mut().zip(x).for_each(|(o, &v)| *o = v * inv),
    }
}

#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// Most states a residual mix takes (the block states and the current residual).
pub const MIX_STATES: usize = 16;

/// Attention over depth for one token: `normed` are the RMS-normed states (earlier blocks, then the current
/// residual; at most `MIX_STATES`); the result is their softmax mix by the query.
#[inline(always)]
pub fn mix(normed: &[&[f32]], query: &[f32], out: &mut [f32]) {
    let scale = 1.0 / (query.len() as f32).sqrt();
    let mut e = [0f32; MIX_STATES];
    let e = &mut e[..normed.len()];
    let mut m = f32::NEG_INFINITY;
    for (x, s) in e.iter_mut().zip(normed) {
        *x = dot(s, query) * scale;
        m = m.max(*x);
    }
    let mut sum = 0f32;
    for x in e.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    for (i, (s, w)) in normed.iter().zip(e.iter()).enumerate() {
        let p = w / sum;
        if i == 0 {
            out.iter_mut().zip(s.iter()).for_each(|(o, v)| *o = p * v);
        } else {
            out.iter_mut().zip(s.iter()).for_each(|(o, v)| *o += p * v);
        }
    }
}

/// Causal depthwise convolution with a residual, of one token's `y` (pre-convolution values) given the three
/// inputs before it (`prev`, oldest first), as llama.cpp's `convolve` without dynamic taps. `taps` are the weights
/// by tap (`conv_taps`): `taps[i * len + c]` is on `prev[i][c]` for i < 3 and on `y[c]` for i = 3.
#[inline(always)]
pub fn conv_row(y: &[f32], prev: [&[f32]; 3], taps: &[f32], out: &mut [f32]) {
    let n = y.len();
    let scale = 1.0 / 5f32.sqrt();
    let (t0, rest) = taps.split_at(n);
    let (t1, rest) = rest.split_at(n);
    let (t2, t3) = rest.split_at(n);
    let (a, b, cc) = (&prev[0][..n], &prev[1][..n], &prev[2][..n]);
    let (t3, out) = (&t3[..n], &mut out[..n]);
    for c in 0..n {
        let conv = t0[c] * a[c] + t1[c] * b[c] + t2[c] * cc[c] + t3[c] * y[c];
        out[c] = (y[c] + conv) * scale;
    }
}

/// Convolution weights `w` [len][4] (GGUF order: `w[c][3]` on the current input) by tap: [4][len].
pub fn conv_taps(w: &[f32]) -> Vec<f32> {
    let n = w.len() / 4;
    (0..4 * n).map(|k| w[(k % n) * 4 + k / n]).collect()
}

/// The host's weights of one physical layer.
pub struct HostLayer {
    pub attn_query: Option<Vec<Vec<f32>>>,
    pub ffn_query: Vec<f32>,
    /// The sublayer-output convolutions' weights by tap (`conv_taps`).
    pub attn_post_conv: Vec<f32>,
    pub ffn_post_conv: Vec<f32>,
    /// Core layers: the message queries for q, k, v, ffn (the GPUs hold the controllers).
    pub msg: Option<[Vec<f32>; 4]>,
}

pub struct HostWeights {
    pub layers: Vec<HostLayer>,
    pub embd_norm: Vec<f32>,
    pub out_norm: Vec<f32>,
    pub out_query: Vec<f32>,
}

impl HostWeights {
    pub fn load(g: &Gguf, c: &Config) -> Result<Self> {
        let d = c.d;
        let mut layers = Vec::new();
        for p in 0..c.layers {
            let v = |n: &str| -> Result<Vec<f32>> { vector(g, c.lt(g, p, n)?) };
            let attn_query = if p > 0 {
                let q = v("attn_res_query.weight")?;
                ensure!(q.len() == d * if c.swa(p) { 1 } else { 3 }, "attn_res_query shape at {p}");
                Some(q.chunks_exact(d).map(<[f32]>::to_vec).collect())
            } else {
                None
            };
            let msg = if c.looped(p) { Some(c.msg_queries(g, p)?) } else { None };
            layers.push(HostLayer {
                attn_query,
                ffn_query: v("ffn_res_query.weight")?,
                attn_post_conv: conv_taps(&v("attn_post_conv.weight")?),
                ffn_post_conv: conv_taps(&v("ffn_post_conv.weight")?),
                msg,
            });
        }
        Ok(Self {
            layers,
            embd_norm: vector(g, c.t(g, "token_embd_norm.weight")?)?,
            out_norm: vector(g, c.t(g, "output_norm.weight")?)?,
            out_query: vector(g, c.t(g, "output_res_query.weight")?)?,
        })
    }
}

/// Token embeddings, RMS-normed with `token_embd_norm` (images skip this: `group` takes their rows as they are).
pub fn embed(g: &Gguf, c: &Config, w: &HostWeights, tokens: &[u32]) -> Result<Vec<f32>> {
    let t = c.t(g, "token_embd.weight")?;
    let mut x = vec![0.; tokens.len() * c.d];
    let mut row = vec![0.; c.d];
    for (o, &id) in x.chunks_exact_mut(c.d).zip(tokens) {
        ensure!((id as usize) < c.vocab, "invalid token {id}");
        dequant_row(g, t, id as usize, &mut row)?;
        rms_into(&row, Some(&w.embd_norm), c.eps, o);
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conv_row_takes_the_history_oldest_first() {
        // Two channels with the GGUF weights [1, 2, 3, 4] and [0, 0, 0, 1].
        let taps = conv_taps(&[1., 2., 3., 4., 0., 0., 0., 1.]);
        assert_eq!(taps, vec![1., 0., 2., 0., 3., 0., 4., 1.]);
        let mut out = [0f32; 2];
        conv_row(&[5., 7.], [&[10., 0.], &[20., 0.], &[30., 0.]], &taps, &mut out);
        assert!((out[0] - (5. + 10. + 40. + 90. + 20.) / 5f32.sqrt()).abs() < 1e-4);
        assert!((out[1] - 14. / 5f32.sqrt()).abs() < 1e-5);
    }
    #[test]
    fn mix_of_one_state_is_that_state() {
        let s = vec![1f32; 16];
        let mut out = vec![0f32; 16];
        mix(&[&s], &vec![0.3; 16], &mut out);
        assert_eq!(out, s);
    }
}
