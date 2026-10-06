//! Qwen3.5-architecture models (`qwen35` GGUFs: Qwen3.8-27B, Qwen3.5-9B): Gated DeltaNet layers with a
//! gated full-attention layer every fourth block, dense SwiGLU FFNs and an untied LM head.
//!
//! The forward pass is split into stages, each owning a contiguous range of layers on one device (a CPU
//! reference or one GPU). Only the hidden state crosses a stage boundary, so the same stages run on one
//! machine or along a chain of machines (`net`).
pub mod chat;
pub mod cpu;
pub mod gpu;
pub mod native;
pub mod net;
pub mod pipeline;
pub mod spec;

use crate::gguf::{GgmlType, Gguf, TensorInfo};
use anyhow::{ensure, Context, Result};
use half::{bf16, f16};

#[derive(Clone, Debug)]
pub struct Config {
    /// Hidden size.
    pub d: usize,
    /// Trunk layers (the GGUF block count minus the MTP layers).
    pub layers: usize,
    pub ff: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Rotated dimensions at the start of each head (partial NeoX rotation).
    pub rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub interval: usize,
    pub conv: usize,
    /// Gated DeltaNet key/query heads, value heads and their common head size.
    pub k_heads: usize,
    pub v_heads: usize,
    pub state: usize,
    pub vocab: usize,
}

impl Config {
    pub fn new(g: &Gguf) -> Result<Self> {
        let arch = g.get_str("general.architecture")?;
        ensure!(arch == "qwen35", "expected a qwen35 model, found {arch}");
        let u = |k: &str| -> Result<usize> { Ok(g.get_u64(&format!("qwen35.{k}"))? as usize) };
        let mtp = g.get_u64("qwen35.nextn_predict_layers").unwrap_or(0) as usize;
        let c = Self {
            d: u("embedding_length")?,
            layers: u("block_count")? - mtp,
            ff: u("feed_forward_length")?,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rot: u("rope.dimension_count")?,
            rope_base: g.get_f64("qwen35.rope.freq_base")? as f32,
            eps: g.get_f64("qwen35.attention.layer_norm_rms_epsilon")? as f32,
            interval: u("full_attention_interval").unwrap_or(4),
            conv: u("ssm.conv_kernel")?,
            k_heads: u("ssm.group_count")?,
            v_heads: u("ssm.time_step_rank")?,
            state: u("ssm.state_size")?,
            vocab: g.tensor("output.weight").context("missing output.weight")?.dims[1] as usize,
        };
        ensure!(
            u("attention.value_length")? == c.head_dim
                && u("ssm.inner_size")? == c.v_heads * c.state
                && c.v_heads % c.k_heads == 0
                && c.heads % c.kv_heads == 0
                && c.rot % 2 == 0
                && c.rot <= c.head_dim
                && c.conv == 4,
            "unsupported qwen35 shape {c:?}"
        );
        // Text positions repeat in all M-RoPE sections, so the sections only need to cover the
        // rotated pairs; the interleaved layout then reduces to an ordinary NeoX rotation.
        let sections = g
            .get("qwen35.rope.dimension_sections")?
            .as_arr()
            .context("invalid rope sections")?;
        let pairs: u64 = sections.iter().filter_map(|v| v.as_u64()).sum();
        ensure!(pairs as usize == c.rot / 2, "unsupported rope sections");
        Ok(c)
    }
    /// Gated DeltaNet layers; the others are gated full attention.
    pub fn recurrent(&self, il: usize) -> bool {
        (il + 1) % self.interval != 0
    }
    pub fn key_dim(&self) -> usize {
        self.k_heads * self.state
    }
    pub fn value_dim(&self) -> usize {
        self.v_heads * self.state
    }
    /// Channels of the short convolution: q, k and v of the DeltaNet input projection.
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    pub fn layer_tensor<'a>(&self, g: &'a Gguf, il: usize, name: &str) -> Result<&'a TensorInfo> {
        let full = format!("blk.{il}.{name}");
        g.tensor(&full).with_context(|| format!("missing {full}"))
    }
}

/// The MTP (nextn) layer index of a qwen35 GGUF (the model itself or an `--mtp` sidecar), checked against
/// the trunk's shape.
pub fn mtp_layer(g: &Gguf, c: &Config) -> Result<usize> {
    let blocks = g.get_u64("qwen35.block_count")? as usize;
    let nextn = g.get_u64("qwen35.nextn_predict_layers").unwrap_or(0) as usize;
    ensure!(nextn == 1, "expected one MTP (nextn) layer, found {nextn}");
    let il = blocks - 1;
    let eh = c.layer_tensor(g, il, "nextn.eh_proj.weight")?;
    let q = c.layer_tensor(g, il, "attn_q.weight")?;
    ensure!(
        eh.dims == [2 * c.d as u64, c.d as u64] && q.dims == [c.d as u64, (2 * c.heads * c.head_dim) as u64],
        "the MTP layer does not match this model"
    );
    Ok(il)
}

/// Dequantize one row of `k` elements of a 2-D tensor (ggml row-major: rows of `dims[0]` elements).
pub fn dequant_row(g: &Gguf, t: &TensorInfo, row: usize, out: &mut [f32]) -> Result<()> {
    let k = t.dims[0] as usize;
    ensure!(out.len() == k, "row length mismatch for {}", t.name);
    let data = g.data(t);
    let bytes = |block: usize, size: usize| {
        let per_row = k / block * size;
        &data[row * per_row..(row + 1) * per_row]
    };
    match t.ty {
        GgmlType::F32 => {
            for (o, b) in out.iter_mut().zip(bytes(1, 4).chunks_exact(4)) {
                *o = f32::from_le_bytes(b.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (o, b) in out.iter_mut().zip(bytes(1, 2).chunks_exact(2)) {
                *o = f16::from_le_bytes(b.try_into().unwrap()).to_f32();
            }
        }
        GgmlType::BF16 => {
            for (o, b) in out.iter_mut().zip(bytes(1, 2).chunks_exact(2)) {
                *o = bf16::from_le_bytes(b.try_into().unwrap()).to_f32();
            }
        }
        GgmlType::Q4_0 => {
            for (o, b) in out.chunks_exact_mut(32).zip(bytes(32, 18).chunks_exact(18)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                for j in 0..16 {
                    o[j] = ((b[2 + j] & 15) as i32 - 8) as f32 * d;
                    o[j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_1 => {
            for (o, b) in out.chunks_exact_mut(32).zip(bytes(32, 20).chunks_exact(20)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                let m = f16::from_le_bytes([b[2], b[3]]).to_f32();
                for j in 0..16 {
                    o[j] = (b[4 + j] & 15) as f32 * d + m;
                    o[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
                }
            }
        }
        GgmlType::Q8_0 => {
            for (o, b) in out.chunks_exact_mut(32).zip(bytes(32, 34).chunks_exact(34)) {
                let d = f16::from_le_bytes([b[0], b[1]]).to_f32();
                for j in 0..32 {
                    o[j] = b[2 + j] as i8 as f32 * d;
                }
            }
        }
        GgmlType::Q6K => out.copy_from_slice(&crate::quant::q6_row(bytes(256, 210))),
        other => anyhow::bail!("{} has unsupported type {other}", t.name),
    }
    Ok(())
}

/// A vector tensor (norm weights, biases) as f32.
pub fn vector(g: &Gguf, t: &TensorInfo) -> Result<Vec<f32>> {
    let n: u64 = t.dims.iter().product();
    let mut v = vec![0.; n as usize];
    let flat = TensorInfo {
        dims: vec![n],
        ..t.clone()
    };
    dequant_row(g, &flat, 0, &mut v)?;
    Ok(v)
}

/// Token embedding rows, dequantized on the CPU.
pub fn embed(g: &Gguf, c: &Config, tokens: &[u32]) -> Result<Vec<f32>> {
    let t = g.tensor("token_embd.weight").context("missing token_embd")?;
    let mut x = vec![0.; tokens.len() * c.d];
    for (row, &id) in x.chunks_exact_mut(c.d).zip(tokens) {
        ensure!((id as usize) < t.dims[1] as usize, "invalid token {id}");
        dequant_row(g, t, id as usize, row)?;
    }
    Ok(x)
}

/// Rotation angles of the partial NeoX RoPE: pair `i` (dims i, i + rot/2) at `pos`.
pub fn rope_angle(c: &Config, pos: usize, i: usize) -> f32 {
    pos as f32 * c.rope_base.powf(-2. * i as f32 / c.rot as f32)
}
