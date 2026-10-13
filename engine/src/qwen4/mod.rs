//! Qwen3.8-Flash-Next (`qwen4exp` GGUFs): 48 layers of Gated DeltaNet with a Qwen Sparse Attention layer every
//! fourth block, a 512-expert MoE in every layer, four hyper-connection residual streams instead of norms, and a
//! hashed n-gram embedding added at layer 1. See docs/qwen4.md.
//!
//! The residual is `hc` streams of `d` floats per token (stream-major, `[token][stream][d]`), all initialised to
//! the token embedding. The n-gram rows are gathered on the host (`Ngram`) and enter the stage owning the PLE
//! layer as `n x d` floats.
pub mod bench;
pub mod cli;
pub mod cpu;
pub mod draft;
pub mod gate;
pub mod gpu;
pub mod mtp;
pub mod net;
pub mod ngram;
pub mod pipeline;
pub mod spec;
pub mod trace;

use crate::gguf::{Gguf, TensorInfo, Value};
use anyhow::{ensure, Context, Result};

#[derive(Clone, Debug)]
pub struct Config {
    pub d: usize,
    /// Trunk layers (without the MTP block).
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Rotated dimensions at the start of each attention and indexer head (partial NeoX rotation).
    pub rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub interval: usize,
    pub conv: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub state: usize,
    pub vocab: usize,
    /// Hyper-connection streams and the rank of the read-gate bottleneck.
    pub hc: usize,
    pub hc_rank: usize,
    pub experts: usize,
    pub used: usize,
    pub ff_exp: usize,
    pub ff_shexp: usize,
    /// QSA indexer: heads, head size, token budget and pooled block size.
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub idx_budget: usize,
    pub kpool: usize,
    pub ngram: Option<Ngram>,
}

/// Hashed n-gram embedding ("PLE"): `(ngram - 1) * per_gram` rows of `dim` from one shared table per token.
#[derive(Clone, Debug)]
pub struct Ngram {
    pub layer: usize,
    pub ngram: usize,
    pub per_gram: usize,
    pub conv: usize,
    pub eos: u32,
    pub dim: usize,
    pub mult: Vec<u64>,
    pub offsets: Vec<u64>,
    pub sizes: Vec<u64>,
}

fn u64s(g: &Gguf, key: &str) -> Result<Vec<u64>> {
    g.get(key)?
        .as_arr()
        .with_context(|| format!("{key} is not an array"))?
        .iter()
        .map(|v| v.as_u64().with_context(|| format!("{key}: not an integer")))
        .collect()
}

impl Config {
    pub fn new(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        ensure!(arch == "qwen4exp", "expected a qwen4exp model, found {arch}");
        let u = |k: &str| -> Result<usize> { Ok(g.get_u64(&format!("qwen4exp.{k}"))? as usize) };
        let mtp = g.get_u64("qwen4exp.nextn_predict_layers").unwrap_or(0) as usize;
        let ratios = u64s(g, "qwen4exp.attention.compress_ratios")?;
        let kpool = ratios.iter().copied().find(|r| *r > 0).unwrap_or(0) as usize;
        ensure!(
            ratios.iter().all(|r| *r == 0 || *r as usize == kpool),
            "QSA layers must share one compress ratio"
        );
        let ngram = match g.get("qwen4exp.ple.layers") {
            Ok(v) => {
                let layers: Vec<u64> = v
                    .as_arr()
                    .context("ple layers")?
                    .iter()
                    .filter_map(Value::as_u64)
                    .collect();
                ensure!(layers.len() == 1, "exactly one PLE layer is supported");
                let n = Ngram {
                    layer: layers[0] as usize,
                    ngram: u("ple.ngram_size")?,
                    per_gram: u("ple.heads_per_ngram")?,
                    conv: u("ple.conv_kernel")?,
                    eos: u("ple.eos_token_id")? as u32,
                    dim: u("embedding_length_per_layer_input")?,
                    mult: u64s(g, "qwen4exp.ple.layer_multipliers")?,
                    offsets: u64s(g, "qwen4exp.ple.head_offsets")?,
                    sizes: u64s(g, "qwen4exp.ple.head_vocab_sizes")?,
                };
                let heads = n.heads();
                ensure!(
                    n.ngram >= 2
                        && n.mult.len() >= n.ngram
                        && n.offsets.len() >= heads
                        && n.sizes.len() >= heads
                        && n.sizes.iter().all(|s| *s > 0),
                    "invalid PLE hash constants"
                );
                Some(n)
            }
            Err(_) => None,
        };
        let c = Self {
            d: u("embedding_length")?,
            layers: u("block_count")? - mtp,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rot: u("rope.dimension_count")?,
            rope_base: g.get_f64("qwen4exp.rope.freq_base")? as f32,
            eps: g.get_f64("qwen4exp.attention.layer_norm_rms_epsilon")? as f32,
            interval: u("full_attention_interval").unwrap_or(4),
            conv: u("ssm.conv_kernel")?,
            k_heads: u("ssm.group_count")?,
            v_heads: u("ssm.time_step_rank")?,
            state: u("ssm.state_size")?,
            // The head's rows; a metadata-only shard falls back to the tokenizer's vocabulary.
            vocab: match g.tensor("output.weight") {
                Some(t) => t.dims[1] as usize,
                None => g
                    .get("tokenizer.ggml.tokens")?
                    .as_arr()
                    .context("tokenizer tokens")?
                    .len(),
            },
            hc: u("hyper_connection.count")?,
            hc_rank: u("hyper_connection.low_rank")?,
            experts: u("expert_count")?,
            used: u("expert_used_count")?,
            ff_exp: u("expert_feed_forward_length")?,
            ff_shexp: u("expert_shared_feed_forward_length")?,
            idx_heads: u("attention.indexer.head_count")?,
            idx_dim: u("attention.indexer.key_length")?,
            idx_budget: u("attention.indexer.top_k")?,
            kpool,
            ngram,
        };
        ensure!(
            u("attention.value_length")? == c.head_dim
                && u("ssm.inner_size")? == c.v_heads * c.state
                && c.v_heads % c.k_heads == 0
                && c.heads % c.kv_heads == 0
                && c.rot % 2 == 0
                && c.rot <= c.idx_dim.min(c.head_dim)
                && c.conv == 4
                && c.hc > 1
                && c.kpool > 1
                && c.idx_budget % c.kpool == 0,
            "unsupported qwen4exp shape {c:?}"
        );
        if let Some(n) = &c.ngram {
            ensure!(
                n.heads() * n.dim == c.d && n.layer < c.layers && c.recurrent(n.layer),
                "unsupported PLE shape"
            );
        }
        // Text positions repeat in all M-RoPE sections, so the interleaved layout reduces to a NeoX rotation of
        // the first `rot` dims when the sections cover exactly the rotated pairs.
        let sections = u64s(g, "qwen4exp.rope.dimension_sections")?;
        ensure!(
            sections.iter().sum::<u64>() as usize == c.rot / 2,
            "unsupported rope sections"
        );
        Ok(c)
    }
    /// Gated DeltaNet layers; the others are QSA.
    pub fn recurrent(&self, il: usize) -> bool {
        (il + 1) % self.interval != 0
    }
    pub fn key_dim(&self) -> usize {
        self.k_heads * self.state
    }
    pub fn value_dim(&self) -> usize {
        self.v_heads * self.state
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// Width of the residual of one token: all hyper-connection streams.
    pub fn width(&self) -> usize {
        self.hc * self.d
    }
    /// Complete pooled blocks that QSA always selects: up to `idx_budget` tokens.
    pub fn top_blocks(&self) -> usize {
        self.idx_budget / self.kpool
    }
    pub fn layer_tensor<'a>(&self, g: &'a Gguf, il: usize, name: &str) -> Result<&'a TensorInfo> {
        let full = format!("blk.{il}.{name}");
        g.tensor(&full).with_context(|| format!("missing {full}"))
    }
    /// Rotation angle of pair `i` (dims i, i + rot/2) at `pos`.
    pub fn rope_angle(&self, pos: usize, i: usize) -> f32 {
        pos as f32 * self.rope_base.powf(-2. * i as f32 / self.rot as f32)
    }
}

impl Ngram {
    pub fn heads(&self) -> usize {
        (self.ngram - 1) * self.per_gram
    }
    /// Table rows of `tokens`, `heads()` per token, given the tokens before them (`history`, oldest first, may be
    /// empty at the start of a sequence). An end-of-sequence token cuts the context at and before it, and missing
    /// predecessors read as end-of-sequence; a token never cuts its own context.
    pub fn rows(&self, history: &[u32], tokens: &[u32]) -> Vec<u32> {
        let prev = self.ngram - 1;
        let mut seq: Vec<u32> = history[history.len().saturating_sub(prev)..].to_vec();
        let start = seq.len();
        seq.extend_from_slice(tokens);
        let mut out = Vec::with_capacity(tokens.len() * self.heads());
        let mut ctx = vec![0u64; self.ngram];
        for i in start..seq.len() {
            ctx[0] = seq[i] as u64;
            let mut cut = false;
            for s in 1..self.ngram {
                let t = if cut || i < s { None } else { Some(seq[i - s]) };
                cut = cut || t.is_none() || t == Some(self.eos);
                ctx[s] = if cut { self.eos as u64 } else { t.unwrap() as u64 };
            }
            for n in 2..=self.ngram {
                let mut mixed = ctx[0].wrapping_mul(self.mult[0]);
                for j in 1..n {
                    mixed ^= ctx[j].wrapping_mul(self.mult[j]);
                }
                for h in (n - 2) * self.per_gram..(n - 1) * self.per_gram {
                    out.push((mixed % self.sizes[h] + self.offsets[h]) as u32);
                }
            }
        }
        out
    }
}

/// The hashed n-gram table, read from the GGUF map (dequantized row by row).
pub fn gather(g: &Gguf, rows: &[u32], out: &mut [f32]) -> Result<()> {
    let t = g
        .tensor("per_layer_token_embd.weight")
        .context("missing per_layer_token_embd")?;
    let dim = t.dims[0] as usize;
    ensure!(out.len() == rows.len() * dim, "n-gram output length");
    for (o, &r) in out.chunks_exact_mut(dim).zip(rows) {
        ensure!((r as u64) < t.dims[1], "n-gram row {r} out of range");
        crate::qwen::dequant_row(g, t, r as usize, o)?;
    }
    Ok(())
}

/// The non-linear IQ4_NL values (also used by IQ4_XS).
pub const IQ4_VALUES: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// Dequantize one row: IQ4_NL and IQ4_XS here, every other type through `qwen::dequant_row`.
pub fn dequant_row(g: &Gguf, t: &TensorInfo, row: usize, out: &mut [f32]) -> Result<()> {
    use crate::gguf::GgmlType;
    use half::f16;
    let k = t.dims[0] as usize;
    match t.ty {
        GgmlType::IQ4NL => {
            let rb = k / 32 * 18;
            let b = &g.data(t)[row * rb..(row + 1) * rb];
            for (o, s) in out.chunks_exact_mut(32).zip(b.chunks_exact(18)) {
                let d = f16::from_le_bytes([s[0], s[1]]).to_f32();
                for j in 0..16 {
                    o[j] = d * IQ4_VALUES[(s[2 + j] & 15) as usize] as f32;
                    o[j + 16] = d * IQ4_VALUES[(s[2 + j] >> 4) as usize] as f32;
                }
            }
        }
        GgmlType::IQ4XS => {
            let rb = k / 256 * 136;
            let b = &g.data(t)[row * rb..(row + 1) * rb];
            for (o, s) in out.chunks_exact_mut(256).zip(b.chunks_exact(136)) {
                let d = f16::from_le_bytes([s[0], s[1]]).to_f32();
                let sh = u16::from_le_bytes([s[2], s[3]]) as u32;
                for ib in 0..8 {
                    let ls = ((s[4 + ib / 2] as u32 >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
                    let dl = d * (ls as i32 - 32) as f32;
                    let q = &s[8 + ib * 16..8 + (ib + 1) * 16];
                    for j in 0..16 {
                        o[ib * 32 + j] = dl * IQ4_VALUES[(q[j] & 15) as usize] as f32;
                        o[ib * 32 + j + 16] = dl * IQ4_VALUES[(q[j] >> 4) as usize] as f32;
                    }
                }
            }
        }
        GgmlType::Q2_0 => {
            let rb = k / 64 * 18;
            let b = &g.data(t)[row * rb..(row + 1) * rb];
            for (o, s) in out.chunks_exact_mut(64).zip(b.chunks_exact(18)) {
                let d = f16::from_le_bytes([s[0], s[1]]).to_f32();
                for (j, v) in o.iter_mut().enumerate() {
                    *v = d * (((s[2 + j / 4] >> (2 * (j % 4))) & 3) as i32 - 1) as f32;
                }
            }
        }
        _ => crate::qwen::dequant_row(g, t, row, out)?,
    }
    Ok(())
}

/// The n-gram rows of `tokens` after `history`, gathered from the GGUF (`None` when the model has no PLE).
pub fn ple_rows(g: &Gguf, c: &Config, history: &[u32], tokens: &[u32]) -> Result<Option<Vec<f32>>> {
    let Some(p) = &c.ngram else { return Ok(None) };
    let rows = p.rows(history, tokens);
    let mut out = vec![0f32; rows.len() * p.dim];
    gather(g, &rows, &mut out)?;
    Ok(Some(out))
}

/// Run the whole model on the CPU reference path: dump the prompt's last-token activations in the format of
/// `tools/dump-activations` and decode `greedy` tokens.
pub fn reference(model: &std::path::Path, out: &std::path::Path, greedy: usize, tokens: &[u32]) -> Result<Vec<u32>> {
    let g = Gguf::open(model)?;
    let c = Config::new(&g)?;
    let mut stage = cpu::CpuStage::new(&c, 0, c.layers, tokens.len() + greedy + 1, true);
    let mut caps = cpu::Captures::new();
    let start = std::time::Instant::now();
    let mut x = embed(&g, &c, tokens)?;
    caps.insert("hc_init".into(), x[x.len() - c.width()..].to_vec());
    let ple = ple_rows(&g, &c, &[], tokens)?;
    stage.forward(&g, &mut x, ple.as_deref(), 0, tokens.len(), &mut Some(&mut caps))?;
    let logits = stage.logits(&g, &x, &mut Some(&mut caps))?;
    caps.insert("result_output".into(), logits.clone());
    std::fs::create_dir_all(out)?;
    let mut index = String::new();
    for (name, v) in &caps {
        std::fs::write(out.join(format!("{name}.f32")), bytemuck::cast_slice(v))?;
        index.push_str(&format!("{name}\t{},1,1,1\t{}\n", v.len(), v.len()));
    }
    std::fs::write(out.join("index.tsv"), index)?;
    std::fs::write(out.join("logits.f32"), bytemuck::cast_slice(&logits))?;
    eprintln!(
        "prompt: {} tokens, {:.2} s",
        tokens.len(),
        start.elapsed().as_secs_f64()
    );
    let argmax = |l: &[f32]| (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b }) as u32;
    let mut seq = tokens.to_vec();
    let mut ids = vec![argmax(&logits)];
    while ids.len() < greedy {
        let id = *ids.last().unwrap();
        let mut x = embed(&g, &c, &[id])?;
        let ple = ple_rows(&g, &c, &seq, &[id])?;
        stage.forward(&g, &mut x, ple.as_deref(), seq.len(), 1, &mut None)?;
        seq.push(id);
        ids.push(argmax(&stage.logits(&g, &x, &mut None)?));
    }
    std::fs::write(
        out.join("greedy.txt"),
        ids.iter().map(|x| format!("{x}\n")).collect::<String>(),
    )?;
    Ok(ids)
}

/// Hyper-connection residual of `tokens`: every stream starts as the token embedding.
pub fn embed(g: &Gguf, c: &Config, tokens: &[u32]) -> Result<Vec<f32>> {
    let t = g.tensor("token_embd.weight").context("missing token_embd")?;
    let mut x = vec![0.; tokens.len() * c.width()];
    for (row, &id) in x.chunks_exact_mut(c.width()).zip(tokens) {
        ensure!((id as usize) < t.dims[1] as usize, "invalid token {id}");
        crate::qwen::dequant_row(g, t, id as usize, &mut row[..c.d])?;
        for s in 1..c.hc {
            row.copy_within(..c.d, s * c.d);
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::Ngram;

    /// The GPU's IQ4_XS layout (quant::Kind::Iq4xs) decoded as qwen4_expert.glsl does (header words, then the
    /// sub-block's 4 nibble words) gives the same values as the reference decoding of the original superblock.
    #[test]
    fn iq4xs_repack_matches_the_shader_decode() {
        let mut sb = [0u8; 136];
        sb[..2].copy_from_slice(&half::f16::from_f32(0.75).to_le_bytes());
        sb[2..4].copy_from_slice(&0b1001_1100_0110_0011u16.to_le_bytes());
        for (i, b) in sb[4..].iter_mut().enumerate() {
            *b = (i * 37 + 11) as u8;
        }
        // Reference decoding (ggml dequantize_row_iq4_xs).
        let d = 0.75f32;
        let sh = u16::from_le_bytes([sb[2], sb[3]]) as u32;
        let mut want = [0f32; 256];
        for ib in 0..8 {
            let ls = ((sb[4 + ib / 2] as u32 >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
            for j in 0..16 {
                let q = sb[8 + ib * 16 + j];
                want[ib * 32 + j] = d * (ls as i32 - 32) as f32 * super::IQ4_VALUES[(q & 15) as usize] as f32;
                want[ib * 32 + j + 16] = d * (ls as i32 - 32) as f32 * super::IQ4_VALUES[(q >> 4) as usize] as f32;
            }
        }
        let p = crate::quant::Planes::of(crate::quant::Kind::Iq4xs, sb.len());
        let (packed, stride) = (p.pack(&sb), p.body_at / 4);
        let words: Vec<u32> = packed
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        for blk in 0..8usize {
            let (h0, h1, ib) = (words[(blk >> 3) * 2], words[(blk >> 3) * 2 + 1], blk as u32 & 7);
            let ls = ((h1 >> (8 * (ib >> 1) + 4 * (ib & 1))) & 15) | (((h0 >> (16 + 2 * ib)) & 3) << 4);
            let scale = half::f16::from_bits(h0 as u16).to_f32() * (ls as i32 - 32) as f32;
            for part in 0..4 {
                let q = words[stride + blk * 4 + part];
                for j in 0..4 {
                    let lo = super::IQ4_VALUES[((q >> (8 * j)) & 15) as usize] as f32 * scale;
                    let hi = super::IQ4_VALUES[((q >> (8 * j + 4)) & 15) as usize] as f32 * scale;
                    assert_eq!(lo, want[blk * 32 + 4 * part + j]);
                    assert_eq!(hi, want[blk * 32 + 16 + 4 * part + j]);
                }
            }
        }
    }

    /// A Q2_0 block repacked like the IQ4_NL/Q4_0 experts (scale array, then 4 code words) and decoded as
    /// qwen4_expert.glsl does (word w holds elements 16w..16w+15) gives ggml's dequantize_row_q2_0 values.
    #[test]
    fn q2_0_repack_matches_the_shader_decode() {
        let mut blk = [0u8; 18];
        blk[..2].copy_from_slice(&half::f16::from_f32(-0.375).to_le_bytes());
        for (i, b) in blk[2..].iter_mut().enumerate() {
            *b = (i * 53 + 7) as u8;
        }
        let want: Vec<f32> = (0..64)
            .map(|j| -0.375 * (((blk[2 + j / 4] >> (2 * (j % 4))) & 3) as i32 - 1) as f32)
            .collect();
        let p = crate::quant::Planes::of(crate::quant::Kind::Q4h, blk.len());
        let (packed, stride) = (p.pack(&blk), p.body_at / 4);
        let scale = half::f16::from_le_bytes([packed[0], packed[1]]).to_f32();
        let words: Vec<u32> = packed
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        for part in 0..4 {
            let q = words[stride + part];
            for e in 0..16 {
                let code = (q >> (2 * e)) & 3;
                assert_eq!(scale * (code as i32 - 1) as f32, want[16 * part + e]);
            }
        }
    }

    fn ngram() -> Ngram {
        Ngram {
            layer: 1,
            ngram: 3,
            per_gram: 2,
            conv: 4,
            eos: 9,
            dim: 4,
            mult: vec![3, 5, 7],
            offsets: vec![0, 11, 24, 41],
            sizes: vec![11, 13, 17, 19],
        }
    }

    #[test]
    fn hashes_follow_the_reference_cut_rules() {
        let n = ngram();
        // Token 4 after [1, 2]: bigram (4, 2), trigram (4, 2, 1).
        let big = (4u64 * 3) ^ (2 * 5);
        let tri = big ^ 7;
        let want = vec![
            (big % 11) as u32,
            (big % 13 + 11) as u32,
            (tri % 17 + 24) as u32,
            (tri % 19 + 41) as u32,
        ];
        assert_eq!(n.rows(&[1, 2], &[4]), want);
        // The same token in one call with its history gives the same rows.
        assert_eq!(&n.rows(&[], &[1, 2, 4])[8..], &want[..]);
        // At the start everything before the token reads as EOS; an EOS cuts everything before it.
        let start = n.rows(&[], &[4]);
        let eos = (4u64 * 3) ^ (9 * 5);
        assert_eq!(start[0], (eos % 11) as u32);
        assert_eq!(start[2], ((eos ^ (9 * 7)) % 17 + 24) as u32);
        assert_eq!(n.rows(&[1, 9], &[4]), start);
        // An EOS token does not cut its own context.
        let own = (9u64 * 3) ^ (2 * 5);
        assert_eq!(n.rows(&[1, 2], &[9])[0], (own % 11) as u32);
    }
}
