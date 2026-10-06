//! Straightforward f32 CPU forward of a layer range: the executable specification the GPU kernels are
//! checked against, and a stage that runs anywhere (the distributed protocol is testable without GPUs).
use super::{dequant_row, rope_angle, vector, Config};
use crate::gguf::{Gguf, TensorInfo};
use anyhow::Result;
use std::collections::BTreeMap;

/// Last-token rows of named intermediates, in llama.cpp's callback names.
pub type Captures = BTreeMap<String, Vec<f32>>;

enum State {
    /// Previous `conv - 1` inputs of the short convolution and one `state x state` matrix per value head,
    /// stored as `s[head][v][k]`.
    Recurrent { conv: Vec<f32>, s: Vec<f32> },
    /// f32 K/V for every position, `[pos][kv_head][dim]`.
    Attention { k: Vec<f32>, v: Vec<f32> },
}

pub struct CpuStage {
    c: Config,
    pub first: usize,
    pub last: usize,
    states: Vec<State>,
    pub head: bool,
}

/// y[t][row] = W[row] . x[t], rows split across threads.
pub fn matmul(g: &Gguf, t: &TensorInfo, x: &[f32], n: usize) -> Vec<f32> {
    let k = t.dims[0] as usize;
    let rows = t.dims[1] as usize;
    assert_eq!(x.len(), n * k, "{}: input length", t.name);
    let mut y = vec![0f32; n * rows];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let per = rows.div_ceil(threads);
    let parts: Vec<Vec<f32>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|i| {
                s.spawn(move || {
                    let (lo, hi) = (i * per, ((i + 1) * per).min(rows));
                    let mut w = vec![0f32; k];
                    let mut out = vec![0f32; n * hi.saturating_sub(lo)];
                    for r in lo..hi {
                        dequant_row(g, t, r, &mut w).unwrap();
                        for tok in 0..n {
                            out[tok * (hi - lo) + r - lo] = dot(&w, &x[tok * k..(tok + 1) * k]);
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (i, part) in parts.iter().enumerate().filter(|(_, p)| !p.is_empty()) {
        let lo = i * per;
        let width = part.len() / n;
        for tok in 0..n {
            y[tok * rows + lo..tok * rows + lo + width]
                .copy_from_slice(&part[tok * width..(tok + 1) * width]);
        }
    }
    y
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Eight independent partial sums keep the loop vectorizable.
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

fn rms_norm(x: &mut [f32], w: Option<&[f32]>, eps: f32) {
    let mean = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1. / ((mean as f32) + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v *= inv * w.map_or(1., |w| w[i]);
    }
}

fn silu(x: f32) -> f32 {
    x / (1. + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        x.exp().ln_1p()
    }
}

impl CpuStage {
    /// Layers `first..last`; `head` adds the final norm and LM head.
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
                    }
                }
            })
            .collect();
        Self {
            c: c.clone(),
            first,
            last,
            states,
            head,
        }
    }

    /// Run the layers over `n` tokens at `pos` in place. Position 0 starts a new sequence.
    pub fn forward(
        &mut self,
        g: &Gguf,
        x: &mut [f32],
        pos: usize,
        n: usize,
        cap: &mut Option<&mut Captures>,
    ) -> Result<()> {
        if pos == 0 {
            for s in &mut self.states {
                if let State::Recurrent { conv, s } = s {
                    conv.fill(0.);
                    s.fill(0.);
                }
            }
        }
        let c = self.c.clone();
        let d = c.d;
        let mut record = |name: String, rows: &[f32], width: usize| {
            if let Some(m) = cap.as_mut() {
                m.insert(name, rows[rows.len() - width..].to_vec());
            }
        };
        for il in self.first..self.last {
            let t = |name: &str| c.layer_tensor(g, il, name);
            let v = |name: &str| -> Result<Vec<f32>> { vector(g, t(name)?) };
            let mut h = x.to_vec();
            let attn_norm = v("attn_norm.weight")?;
            for row in h.chunks_exact_mut(d) {
                rms_norm(row, Some(&attn_norm), c.eps);
            }
            record(format!("attn_norm-{il}"), &h, d);
            let out = match &mut self.states[il - self.first] {
                State::Recurrent { conv, s } => {
                    Self::delta_net(g, &c, il, &h, n, conv, s, &mut record)?
                }
                State::Attention { k, v } => {
                    Self::attention(g, &c, il, &h, pos, n, k, v, &mut record)?
                }
            };
            for (xv, o) in x.iter_mut().zip(&out) {
                *xv += o;
            }
            record(format!("attn_residual-{il}"), x, d);
            let mut h = x.to_vec();
            let post = v("post_attention_norm.weight")?;
            for row in h.chunks_exact_mut(d) {
                rms_norm(row, Some(&post), c.eps);
            }
            record(format!("attn_post_norm-{il}"), &h, d);
            let gate = matmul(g, t("ffn_gate.weight")?, &h, n);
            let up = matmul(g, t("ffn_up.weight")?, &h, n);
            let act: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| silu(*a) * b).collect();
            record(format!("ffn_swiglu-{il}"), &act, c.ff);
            let down = matmul(g, t("ffn_down.weight")?, &act, n);
            record(format!("ffn_out-{il}"), &down, d);
            for (xv, o) in x.iter_mut().zip(&down) {
                *xv += o;
            }
            record(format!("l_out-{il}"), x, d);
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

        // Causal depthwise convolution over [previous inputs, this batch], then SiLU.
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
        let l2 = |v: &[f32]| -> Vec<f32> {
            let inv = 1. / (v.iter().map(|x| x * x).sum::<f32>() + c.eps).sqrt();
            v.iter().map(|x| x * inv).collect()
        };
        for tok in 0..n {
            let row = &qkv[tok * cd..(tok + 1) * cd];
            let q: Vec<Vec<f32>> = (0..c.k_heads).map(|kh| l2(&row[kh * sd..(kh + 1) * sd])).collect();
            let k: Vec<Vec<f32>> = (0..c.k_heads)
                .map(|kh| l2(&row[kd + kh * sd..kd + (kh + 1) * sd]))
                .collect();
            if tok == n - 1 {
                record(format!("q_conv_predelta-{il}"), &q.concat(), kd);
                record(format!("k_conv_predelta-{il}"), &k.concat(), kd);
            }
            for vh in 0..c.v_heads {
                // ggml broadcasts the key heads by tiling: value head vh reads key head vh % k_heads.
                let (q, k) = (&q[vh % c.k_heads], &k[vh % c.k_heads]);
                let v = &row[2 * kd + vh * sd..2 * kd + (vh + 1) * sd];
                let decay = gate[tok * c.v_heads + vh].exp();
                let b = beta[tok * c.v_heads + vh];
                let m = &mut s[vh * sd * sd..(vh + 1) * sd * sd];
                for i in 0..sd {
                    let r = &mut m[i * sd..(i + 1) * sd];
                    for x in r.iter_mut() {
                        *x *= decay;
                    }
                    let delta = (v[i] - dot(r, k)) * b;
                    for (x, kj) in r.iter_mut().zip(k) {
                        *x += kj * delta;
                    }
                    out[tok * vd + vh * sd + i] = dot(r, q) * scale;
                }
            }
        }
        record(format!("attn_output-{il}"), &out, vd);

        // Per-head RMS norm, shared weight, gated by SiLU(z).
        let norm = vector(g, t("ssm_norm.weight")?)?;
        for (head, zh) in out.chunks_exact_mut(sd).zip(z.chunks_exact(sd)) {
            rms_norm(head, Some(&norm), c.eps);
            for (o, zz) in head.iter_mut().zip(zh) {
                *o *= silu(*zz);
            }
        }
        record(format!("final_output-{il}"), &out, vd);
        let y = matmul(g, t("ssm_out.weight")?, &out, n);
        record(format!("linear_attn_out-{il}"), &y, c.d);
        Ok(y)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        g: &Gguf,
        c: &Config,
        il: usize,
        h: &[f32],
        pos: usize,
        n: usize,
        kc: &mut [f32],
        vc: &mut [f32],
        record: &mut impl FnMut(String, &[f32], usize),
    ) -> Result<Vec<f32>> {
        let t = |name: &str| c.layer_tensor(g, il, name);
        let (hd, nh, nkv) = (c.head_dim, c.heads, c.kv_heads);
        let qg = matmul(g, t("attn_q.weight")?, h, n);
        let k = matmul(g, t("attn_k.weight")?, h, n);
        let v = matmul(g, t("attn_v.weight")?, h, n);
        let qn = vector(g, t("attn_q_norm.weight")?)?;
        let kn = vector(g, t("attn_k_norm.weight")?)?;
        let rotate = |x: &mut [f32], p: usize| {
            let half = c.rot / 2;
            for i in 0..half {
                let (sn, cs) = rope_angle(c, p, i).sin_cos();
                let (a, b) = (x[i], x[i + half]);
                x[i] = a * cs - b * sn;
                x[i + half] = a * sn + b * cs;
            }
        };
        let mut q = vec![0f32; n * nh * hd];
        let mut gate = vec![0f32; n * nh * hd];
        for tok in 0..n {
            for head in 0..nh {
                let src = &qg[(tok * nh + head) * 2 * hd..];
                let dst = (tok * nh + head) * hd;
                q[dst..dst + hd].copy_from_slice(&src[..hd]);
                gate[dst..dst + hd].copy_from_slice(&src[hd..2 * hd]);
                rms_norm(&mut q[dst..dst + hd], Some(&qn), c.eps);
                rotate(&mut q[dst..dst + hd], pos + tok);
            }
            for head in 0..nkv {
                let at = (tok * nkv + head) * hd;
                let mut kk = k[at..at + hd].to_vec();
                rms_norm(&mut kk, Some(&kn), c.eps);
                rotate(&mut kk, pos + tok);
                let slot = ((pos + tok) * nkv + head) * hd;
                kc[slot..slot + hd].copy_from_slice(&kk);
                vc[slot..slot + hd].copy_from_slice(&v[at..at + hd]);
            }
        }
        record(format!("Qcur-{il}"), &q, nh * hd);
        let kpos = (pos + n - 1) * nkv * hd;
        record(format!("Kcur-{il}"), &kc[kpos..kpos + nkv * hd], nkv * hd);
        let scale = 1. / (hd as f32).sqrt();
        let mut o = vec![0f32; n * nh * hd];
        for tok in 0..n {
            for head in 0..nh {
                let kh = head / (nh / nkv);
                let qv = &q[(tok * nh + head) * hd..][..hd];
                let scores: Vec<f32> = (0..=pos + tok)
                    .map(|p| dot(qv, &kc[(p * nkv + kh) * hd..][..hd]) * scale)
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = w.iter().sum();
                let dst = &mut o[(tok * nh + head) * hd..][..hd];
                for (p, wp) in w.iter().enumerate() {
                    for (j, x) in dst.iter_mut().enumerate() {
                        *x += wp / sum * vc[(p * nkv + kh) * hd + j];
                    }
                }
            }
        }
        record(format!("attn_pregate-{il}"), &o, nh * hd);
        for (x, gt) in o.iter_mut().zip(&gate) {
            *x *= sigmoid(*gt);
        }
        record(format!("attn_gated-{il}"), &o, nh * hd);
        let y = matmul(g, t("attn_output.weight")?, &o, n);
        record(format!("attn_output-{il}"), &y, c.d);
        Ok(y)
    }

    /// Final norm and full logits of the last token.
    pub fn logits(&self, g: &Gguf, x: &[f32], cap: &mut Option<&mut Captures>) -> Result<Vec<f32>> {
        let d = self.c.d;
        let mut h = x[x.len() - d..].to_vec();
        let norm = vector(g, g.tensor("output_norm.weight").unwrap())?;
        rms_norm(&mut h, Some(&norm), self.c.eps);
        if let Some(m) = cap.as_mut() {
            m.insert("result_norm".into(), h.clone());
        }
        Ok(matmul(g, g.tensor("output.weight").unwrap(), &h, 1))
    }
}
