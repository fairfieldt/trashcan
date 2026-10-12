//! One GPU holding a contiguous range of qwen4exp layers (optionally the n-gram module of layer 1, and the final
//! mixer with the LM head). As in the qwen35 stages, each batch size records one command buffer per layer, replayed
//! with a new position; the residual (all hyper-connection streams) enters and leaves through host buffers.
//!
//! Every mixer applies the previous sublayer's write (`x_s += w_s * y`) in its norm kernel, so a layer ends with a
//! pending combine that the next mixer, the head or the stage output resolves.
//!
//! Dense matrices are Q4_0 repacked with f32 scales (`quant::repack_q4`, the Gemma kernels) or Q8_0 with f16 scales;
//! experts are Q4_0, IQ4_NL (4.5 bits per weight) or IQ4_XS (4.25), at their GGUF size, stacked into one gate/up and
//! one down matrix per layer with the shared expert appended as expert `experts`. Attention is dense, so a stage
//! accepts at most `idx_budget + kpool - 1` visible tokens (the sparse block selection is not implemented here yet).
use super::{cpu::Captures, Config};
use crate::{
    gguf::{GgmlType, Gguf, TensorInfo},
    quant,
    qwen::{dequant_row, vector},
    vulkan::{attention_shaders, q4_decode_shader, Buffer, Device, Push},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::{collections::HashMap, rc::Rc};

pub const TOP_K: usize = crate::qwen::gpu::TOP_K;
const GRAPH_CACHE_CAPACITY: usize = 6;
/// Phase timestamps per layer (n-gram module, two mixers, sublayer (sparse QSA: projections, block selection,
/// attention), router, experts).
const MARKS: usize = 8;
/// Batches of at least this many tokens group expert slots by expert (each expert's rows decoded once).
const GROUPED_MIN: usize = 16;

/// A dense matrix: Q4_0 repacked with f32 scales (the Gemma kernels) or Q8_0 with f16 scales (`qwen4_q8_*`, the
/// word offset of its blocks).
enum Dense {
    Q4(Buffer),
    Q8(Buffer, u32),
}

/// A hyper-connection mixer: gamma `[hc][d]`, bottleneck down `rank x hc*d` and up `hc*d x rank` (Q4_0), write
/// weights `hc x hc*d` (f32; absent for the final mixer).
struct Mix {
    norm: Buffer,
    down: Dense,
    up: Dense,
    inject: Option<Buffer>,
    /// The bottleneck rows followed by the write rows in one matrix, for the fused small-batch kernels.
    fused: Option<Dense>,
}

enum Mixer {
    Delta {
        qkv: Dense,
        z: Dense,
        ab: Buffer,
        dt_a: Buffer,
        conv_w: Buffer,
        conv: Buffer,
        state: Buffer,
        norm: Buffer,
        out: Dense,
    },
    Attention {
        q: Dense,
        k: Dense,
        v: Dense,
        q_norm: Buffer,
        k_norm: Buffer,
        out: Dense,
        kc: Buffer,
        vc: Buffer,
        /// The indexer, for contexts past the dense range: f32 projections, norm weights, the raw key ring
        /// `[context][idx_dim]` and pooled block keys `[context / kpool][idx_dim]`.
        index: Option<Index>,
    },
}

struct Index {
    q: Buffer,
    k: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    raw: Buffer,
    pooled: Buffer,
}

struct Layer {
    il: usize,
    attn_mix: Mix,
    ffn_mix: Mix,
    mixer: Mixer,
    /// Router rows and the shared-expert gate as row `experts` (f32).
    router: Buffer,
    /// Experts in Q4_0 with f16 scales (`quant::repack_q4h`) and the word offsets of their blocks.
    gate_up: Buffer,
    down: Buffer,
    gu_stride: u32,
    down_stride: u32,
    /// Expert formats (qwen4_expert.glsl): 0 Q4_0, 1 IQ4_NL, 2 IQ4_XS.
    gu_fmt: u32,
    down_fmt: u32,
}

/// The GPU expert format of a GGUF type and the bytes of one row of `k` weights.
fn expert_format(t: &TensorInfo) -> Result<(u32, usize)> {
    let k = t.dims[0] as usize;
    Ok(match t.ty {
        GgmlType::Q4_0 => (0, k / 32 * 18),
        GgmlType::IQ4NL => (1, k / 32 * 18),
        GgmlType::IQ4XS => (2, k / 256 * 136),
        GgmlType::Q2_0 => (3, k / 64 * 18),
        other => anyhow::bail!(
            "{} is {other}; GPU experts are Q4_0, IQ4_NL or IQ4_XS (docs/qwen4.md)",
            t.name
        ),
    })
}

/// Raw rows `row0..row0 + rows` of an expert tensor in its GGUF format.
fn expert_rows<'a>(g: &'a Gguf, t: &TensorInfo, row0: usize, rows: usize) -> Result<&'a [u8]> {
    let rb = expert_format(t)?.1;
    Ok(&g.data(t)[row0 * rb..(row0 + rows) * rb])
}

/// Experts repacked for the GPU: scales or headers, then the nibble blocks (and the blocks' word offset).
fn expert_pack(fmt: u32, raw: &[u8]) -> (Vec<u8>, usize) {
    if fmt == 2 {
        quant::repack_iq4xs(raw)
    } else {
        quant::repack_q4h(raw)
    }
}

struct Ple {
    key: Dense,
    value: Dense,
    /// `[key | query | conv]` norm weights, each `hc x d`.
    norms: Buffer,
    conv_w: Buffer,
    hist: Buffer,
}

/// Rows `row0..row0 + rows` of the LM head. The stage that ends the model also runs the final mixer (`mix`); other
/// stages score the final hidden state they are given (`head_slice`).
struct Head {
    mix: Option<Mix>,
    w: Buffer,
    logits: Buffer,
    scratch: Buffer,
    top: Buffer,
    chunks: usize,
    row0: usize,
    rows: usize,
    /// The final hidden state of the last token (written by the ending stage, read by the others).
    h: Buffer,
}

struct Graph {
    gpu: Rc<Device>,
    cmds: Vec<vk::CommandBuffer>,
    captures: Vec<(String, Buffer, usize)>,
    /// Phase timestamps after the layer stamps: (layer index in the stage, phase name).
    marks: Vec<(usize, &'static str)>,
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { self.gpu.dev.free_command_buffers(self.gpu.pool, &self.cmds) };
    }
}

/// The format of the QSA K/V caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCache {
    F16,
    /// Q8 K and V rows: int8 values with an f16 scale per 32 (`qwen4_kv_store`).
    Q8,
    /// f16 K, Q8 V.
    K16V8,
}

impl KvCache {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "f16" => Self::F16,
            "q8" => Self::Q8,
            "k16v8" => Self::K16V8,
            other => anyhow::bail!("unknown K/V cache format {other} (f16, q8, k16v8)"),
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8 => "q8",
            Self::K16V8 => "k16v8",
        }
    }
    fn k_q8(self) -> bool {
        self == Self::Q8
    }
    fn v_q8(self) -> bool {
        self != Self::F16
    }
    /// Bytes of one cache row (one head at one position).
    fn row_bytes(q8: bool, dim: usize) -> usize {
        if q8 {
            dim + dim / 16
        } else {
            dim * 2
        }
    }
}

pub enum Output {
    Hidden(Vec<f32>),
    /// The best candidates of this stage's head rows and the final hidden state of the last token.
    Top(Vec<(f32, u32)>, Vec<f32>),
}

/// Merge candidate lists from head slices: best first, ties to the lower id, at most `TOP_K`.
pub fn merge_top(mut a: Vec<(f32, u32)>, b: Vec<(f32, u32)>) -> Vec<(f32, u32)> {
    a.extend(b);
    a.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    a.truncate(TOP_K);
    a
}

pub struct GpuStage {
    c: Config,
    pub first: usize,
    pub last: usize,
    pub gpu: Rc<Device>,
    layers: Vec<Layer>,
    ple: Option<Ple>,
    head: Option<Head>,
    max_batch: usize,
    context: usize,
    kv: KvCache,
    /// Rows of the raw indexer-key ring: the batch and the incomplete block before it.
    ring: usize,
    input: Buffer,
    ple_in: Buffer,
    output: Buffer,
    params: Buffer,
    params_host: Buffer,
    x: Buffer,
    xn: Buffer,
    lo: Buffer,
    /// Fused mixer outputs `[t][rank + 8]` (bottleneck | write weights | inverse norms), alternating per mixer.
    mo: [Buffer; 2],
    hgate: Buffer,
    u: Buffer,
    w: Buffer,
    y: Buffer,
    big: Buffer,
    conv_out: Buffer,
    z: Buffer,
    ab: Buffer,
    o: Buffer,
    kb: Buffer,
    vb: Buffer,
    /// Normed and rotated K, and V, of the batch before the Q8 cache store.
    kq: Buffer,
    vq: Buffer,
    rq: Buffer,
    scratch: Buffer,
    logits: Buffer,
    ids: Buffer,
    wts: Buffer,
    act: Buffer,
    /// Prefill grouping: per-expert offsets, slot indices grouped by expert, and weighted per-slot outputs.
    offsets: Buffer,
    perm: Buffer,
    ys: Buffer,
    ik: Buffer,
    iq: Buffer,
    iqr: Buffer,
    scores: Buffer,
    sel: Buffer,
    pe: Buffer,
    pk: Buffer,
    pv: Buffer,
    gv: Buffer,
    pn: Buffer,
    fence: vk::Fence,
    graphs: HashMap<(usize, u32, bool), Graph>,
    /// Command buffers scoring a given hidden state with this stage's head rows (by dump).
    head_graphs: HashMap<bool, Graph>,
    queries: vk::QueryPool,
    /// GPU milliseconds of each layer's submission in the last forward (the last includes the head or output).
    pub profile: Vec<f64>,
    /// GPU milliseconds per (layer index in the stage, phase) in the last forward.
    pub phases: Vec<(usize, &'static str, f64)>,
}

/// The pipeline of an expert kernel for a format (the variants are specializations of one shader).
fn expert_shader(name: &'static str, fmt: u32) -> &'static str {
    match (name, fmt) {
        ("qwen4_moe_gu", 1) => "qwen4_moe_gu_iq4nl",
        ("qwen4_moe_gu", 2) => "qwen4_moe_gu_iq4xs",
        ("qwen4_moe_down", 1) => "qwen4_moe_down_iq4nl",
        ("qwen4_moe_down", 2) => "qwen4_moe_down_iq4xs",
        ("qwen4_moe_down", 3) => "qwen4_moe_down_q2",
        ("qwen4_moe_gu_grouped", 1) => "qwen4_moe_gu_grouped_iq4nl",
        ("qwen4_moe_gu_grouped", 2) => "qwen4_moe_gu_grouped_iq4xs",
        ("qwen4_moe_down_grouped", 1) => "qwen4_moe_down_grouped_iq4nl",
        ("qwen4_moe_down_grouped", 2) => "qwen4_moe_down_grouped_iq4xs",
        ("qwen4_moe_down_grouped", 3) => "qwen4_moe_down_grouped_q2",
        _ => name,
    }
}

fn partitions(pos: usize, n: usize) -> u32 {
    if n > 8 {
        return 0;
    }
    match pos + n {
        0..=32 => 8,
        33..=256 => 16,
        257..=1024 => 32,
        _ => 64,
    }
}

/// Raw Q4_0 rows `row0..row0 + rows` of a tensor (any rank; rows of `dims[0]`).
fn q4_rows<'a>(g: &'a Gguf, t: &TensorInfo, row0: usize, rows: usize) -> Result<&'a [u8]> {
    ensure!(
        t.ty == GgmlType::Q4_0,
        "{} is {}; the qwen4 GPU stage needs Q4_0 matrices (docs/qwen4.md)",
        t.name,
        t.ty
    );
    let rb = t.dims[0] as usize / 32 * 18;
    Ok(&g.data(t)[row0 * rb..(row0 + rows) * rb])
}

/// Every row of a tensor as f32 (any type `dequant_row` reads).
fn rows_f32(g: &Gguf, t: &TensorInfo) -> Result<Vec<f32>> {
    let k = t.dims[0] as usize;
    let n: u64 = t.dims[1..].iter().product();
    let mut out = vec![0f32; n as usize * k];
    for (r, row) in out.chunks_exact_mut(k).enumerate() {
        dequant_row(g, t, r, row)?;
    }
    Ok(out)
}

impl GpuStage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: Rc<Device>,
        g: &Gguf,
        c: &Config,
        first: usize,
        last: usize,
        head: Option<(usize, usize)>,
        context: usize,
        max_batch: usize,
        kv: KvCache,
    ) -> Result<Self> {
        ensure!(
            c.state == 128 && c.head_dim % 64 == 0 && c.head_dim <= 512,
            "unsupported head sizes"
        );
        ensure!(
            c.hc <= 4 && c.experts % 64 == 0 && c.experts <= 512,
            "unsupported mixer or router shape"
        );
        ensure!(
            context % c.kpool == 0 && c.idx_dim <= 128 && c.idx_heads * c.idx_dim <= 512,
            "unsupported indexer"
        );
        // Past this many visible tokens QSA attends to selected blocks; below it every block is selected. Q8 caches
        // are read only by the sparse kernels, which select every block while there are few enough.
        let sparse = kv != KvCache::F16 || context > c.idx_budget + c.kpool - 1;
        let ring = (max_batch + c.kpool).next_multiple_of(c.kpool);
        let (d, w) = (c.d, c.width());
        let t = |il: usize, name: &str| c.layer_tensor(g, il, name);
        let q4 = |t: &TensorInfo| -> Result<Dense> {
            let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
            Ok(match t.ty {
                GgmlType::Q8_0 => {
                    let (packed, stride) = quant::repack_q8h(g.data(t));
                    Dense::Q8(gpu.upload_matrix(&packed)?, stride as u32)
                }
                _ => Dense::Q4(gpu.upload_matrix(&quant::repack_q4(q4_rows(g, t, 0, n)?, k, n, 0, n, 0, k))?),
            })
        };
        let f32s = |v: &[f32]| gpu.upload(bytemuck::cast_slice(v));
        let mat32 = |v: &[f32]| gpu.upload_matrix(bytemuck::cast_slice(v));
        let vecb = |t: &TensorInfo| -> Result<Buffer> { f32s(&vector(g, t)?) };
        let mix = |prefix: &str, inject: bool| -> Result<Mix> {
            let tt = |name: &str| {
                let full = format!("{prefix}_{name}.weight");
                g.tensor(&full).with_context(|| format!("missing {full}"))
            };
            let fused = if inject && c.hc == 4 {
                let (dn, inj) = (tt("down")?, tt("inject")?);
                match (dn.ty, inj.ty) {
                    (GgmlType::Q4_0, GgmlType::Q4_0) => {
                        let mut raw = g.data(dn).to_vec();
                        raw.extend_from_slice(g.data(inj));
                        let (k, n) = (dn.dims[0] as usize, dn.dims[1] as usize + inj.dims[1] as usize);
                        Some(Dense::Q4(gpu.upload_matrix(&quant::repack_q4(&raw, k, n, 0, n, 0, k))?))
                    }
                    (GgmlType::Q8_0, GgmlType::Q8_0) => {
                        let mut raw = g.data(dn).to_vec();
                        raw.extend_from_slice(g.data(inj));
                        let (packed, stride) = quant::repack_q8h(&raw);
                        Some(Dense::Q8(gpu.upload_matrix(&packed)?, stride as u32))
                    }
                    _ => None,
                }
            } else {
                None
            };
            Ok(Mix {
                norm: vecb(tt("norm")?)?,
                down: q4(tt("down")?)?,
                up: q4(tt("up")?)?,
                inject: if inject {
                    Some(mat32(&rows_f32(g, tt("inject")?)?)?)
                } else {
                    None
                },
                fused,
            })
        };
        let mut layers = Vec::new();
        for il in first..last {
            let mixer = if c.recurrent(il) {
                let mut ab = rows_f32(g, t(il, "ssm_alpha.weight")?)?;
                ab.extend(rows_f32(g, t(il, "ssm_beta.weight")?)?);
                let mut dt_a = vector(g, t(il, "ssm_dt.bias")?)?;
                dt_a.extend(vector(g, t(il, "ssm_a")?)?);
                Mixer::Delta {
                    qkv: q4(t(il, "attn_qkv.weight")?)?,
                    z: q4(t(il, "attn_gate.weight")?)?,
                    ab: mat32(&ab)?,
                    dt_a: f32s(&dt_a)?,
                    conv_w: vecb(t(il, "ssm_conv1d.weight")?)?,
                    conv: gpu.buffer((c.conv - 1) * c.conv_dim() * 4, false, false, None)?,
                    state: gpu.buffer(c.v_heads * c.state * c.state * 4, false, false, None)?,
                    norm: vecb(t(il, "ssm_norm.weight")?)?,
                    out: q4(t(il, "ssm_out.weight")?)?,
                }
            } else {
                let (kbytes, vbytes) = (
                    context * c.kv_heads * KvCache::row_bytes(kv.k_q8(), c.head_dim),
                    context * c.kv_heads * KvCache::row_bytes(kv.v_q8(), c.head_dim),
                );
                Mixer::Attention {
                    q: q4(t(il, "attn_q.weight")?)?,
                    k: q4(t(il, "attn_k.weight")?)?,
                    v: q4(t(il, "attn_v.weight")?)?,
                    q_norm: vecb(t(il, "attn_q_norm.weight")?)?,
                    k_norm: vecb(t(il, "attn_k_norm.weight")?)?,
                    out: q4(t(il, "attn_output.weight")?)?,
                    kc: gpu.buffer(kbytes, false, false, None)?,
                    vc: gpu.buffer(vbytes, false, false, None)?,
                    index: if sparse {
                        Some(Index {
                            q: mat32(&rows_f32(g, t(il, "indexer.q_proj.weight")?)?)?,
                            k: mat32(&rows_f32(g, t(il, "indexer.k_proj.weight")?)?)?,
                            q_norm: vecb(t(il, "indexer.q_norm.weight")?)?,
                            k_norm: vecb(t(il, "indexer.k_norm.weight")?)?,
                            raw: gpu.buffer(ring * c.idx_dim * 4, false, false, None)?,
                            pooled: gpu.buffer(context / c.kpool * c.idx_dim * 4, false, false, None)?,
                        })
                    } else {
                        None
                    },
                }
            };
            // Experts stacked as [gate rows | up rows] per expert, the shared expert last.
            let (ff, ne) = (c.ff_exp, c.experts);
            ensure!(c.ff_shexp == ff, "the shared expert must match the routed width");
            let (gt, ut, dt) = (
                t(il, "ffn_gate_exps.weight")?,
                t(il, "ffn_up_exps.weight")?,
                t(il, "ffn_down_exps.weight")?,
            );
            let (gs, us, ds) = (
                t(il, "ffn_gate_shexp.weight")?,
                t(il, "ffn_up_shexp.weight")?,
                t(il, "ffn_down_shexp.weight")?,
            );
            // The shared expert is stacked with the routed ones, so it must share their formats.
            let (gu_fmt, down_fmt) = (expert_format(gt)?.0, expert_format(dt)?.0);
            ensure!(gu_fmt != 3, "{} is Q2_0; the GPU reads Q2_0 for down projections only", gt.name);
            for (a, b) in [(gt, ut), (gt, gs), (gt, us), (dt, ds)] {
                ensure!(
                    a.ty == b.ty,
                    "{} is {} but {} is {}: quantize them alike",
                    a.name,
                    a.ty,
                    b.name,
                    b.ty
                );
            }
            let mut gu = Vec::with_capacity((ne + 1) * 2 * ff * expert_format(gt)?.1);
            for e in 0..ne {
                gu.extend_from_slice(expert_rows(g, gt, e * ff, ff)?);
                gu.extend_from_slice(expert_rows(g, ut, e * ff, ff)?);
            }
            gu.extend_from_slice(expert_rows(g, gs, 0, ff)?);
            gu.extend_from_slice(expert_rows(g, us, 0, ff)?);
            let (packed, gu_stride) = expert_pack(gu_fmt, &gu);
            drop(gu);
            let gate_up = gpu.upload_matrix(&packed)?;
            drop(packed);
            let mut dn = Vec::with_capacity((ne + 1) * d * expert_format(dt)?.1);
            dn.extend_from_slice(expert_rows(g, dt, 0, ne * d)?);
            dn.extend_from_slice(expert_rows(g, ds, 0, d)?);
            let (packed, down_stride) = expert_pack(down_fmt, &dn);
            drop(dn);
            let down = gpu.upload_matrix(&packed)?;
            drop(packed);
            let mut router = rows_f32(g, t(il, "ffn_gate_inp.weight")?)?;
            router.extend(vector(g, t(il, "ffn_gate_inp_shexp.weight")?)?);
            layers.push(Layer {
                il,
                attn_mix: mix(&format!("blk.{il}.hc_attn"), true)?,
                ffn_mix: mix(&format!("blk.{il}.hc_ffn"), true)?,
                mixer,
                router: mat32(&router)?,
                gate_up,
                down,
                gu_stride: gu_stride as u32,
                down_stride: down_stride as u32,
                gu_fmt,
                down_fmt,
            });
        }
        let ple = match &c.ngram {
            Some(p) if (first..last).contains(&p.layer) => {
                let il = p.layer;
                let mut norms = vector(g, t(il, "ple_norm_key.weight")?)?;
                norms.extend(vector(g, t(il, "ple_norm_query.weight")?)?);
                norms.extend(vector(g, t(il, "ple_norm_conv.weight")?)?);
                Some(Ple {
                    key: q4(t(il, "ple_key.weight")?)?,
                    value: q4(t(il, "ple_value.weight")?)?,
                    norms: f32s(&norms)?,
                    conv_w: f32s(&vector(g, t(il, "ple_conv1d.weight")?)?)?,
                    hist: gpu.buffer((p.conv - 1) * p.ngram * w * 4, false, false, None)?,
                })
            }
            _ => None,
        };
        let ends = last == c.layers;
        ensure!(head.is_some() || !ends, "the stage that ends the model needs head rows");
        let head = match head {
            Some((row0, row1)) => {
                let t = g.tensor("output.weight").context("missing output.weight")?;
                ensure!(t.ty == GgmlType::Q6K, "output.weight must be Q6_K, found {}", t.ty);
                ensure!(row0 < row1 && row1 <= c.vocab, "head rows {row0}..{row1} outside the vocabulary");
                let rows = row1 - row0;
                let chunks = rows.div_ceil(1024);
                ensure!(chunks <= 256, "vocabulary too large for one candidate merge");
                let rb = d / 256 * 210;
                Some(Head {
                    mix: if ends { Some(mix("output_hc", false)?) } else { None },
                    w: gpu.upload_matrix(&quant::repack_q6(&g.data(t)[row0 * rb..row1 * rb]))?,
                    logits: gpu.buffer(rows * 4, false, false, None)?,
                    scratch: gpu.buffer(chunks * TOP_K * 8, false, false, None)?,
                    top: gpu.buffer(TOP_K * 8, true, false, None)?,
                    chunks,
                    row0,
                    rows,
                    h: gpu.buffer(d * 4, true, false, None)?,
                })
            }
            None => None,
        };
        let alloc = |n: usize| gpu.buffer(n * max_batch * 4, false, false, None);
        let qd = c.heads * c.head_dim;
        let slots = c.used + 1;
        let fence = unsafe { gpu.dev.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let queries = unsafe {
            gpu.dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count((2 + MARKS) as u32 * layers.len() as u32),
                None,
            )?
        };
        Ok(Self {
            first,
            last,
            layers,
            ple,
            head,
            max_batch,
            context,
            kv,
            ring,
            input: gpu.buffer(w * max_batch * 4, true, false, None)?,
            ple_in: gpu.buffer(d * max_batch * 4, true, false, None)?,
            output: gpu.buffer(w * max_batch * 4, true, false, None)?,
            params: gpu.buffer(16, false, false, None)?,
            params_host: gpu.buffer(16, true, false, None)?,
            x: alloc(w)?,
            xn: alloc(w)?,
            lo: alloc(c.hc_rank)?,
            mo: [alloc(c.hc_rank + 8)?, alloc(c.hc_rank + 8)?],
            hgate: alloc(w)?,
            u: alloc(d)?,
            w: alloc(c.hc)?,
            y: alloc(d)?,
            big: alloc(c.conv_dim().max(2 * qd))?,
            conv_out: alloc(c.conv_dim())?,
            z: alloc(c.value_dim())?,
            ab: alloc(2 * c.v_heads)?,
            o: alloc(c.value_dim().max(qd))?,
            kb: alloc(c.kv_heads * c.head_dim)?,
            vb: alloc(c.kv_heads * c.head_dim)?,
            kq: if kv.k_q8() {
                alloc(c.kv_heads * c.head_dim)?
            } else {
                gpu.buffer(4, false, false, None)?
            },
            vq: if kv.v_q8() {
                alloc(c.kv_heads * c.head_dim)?
            } else {
                gpu.buffer(4, false, false, None)?
            },
            rq: alloc(qd)?,
            scratch: gpu.buffer(c.heads * 64 * (c.head_dim + 2) * 4 * 8, false, false, None)?,
            logits: alloc(c.experts + 1)?,
            ids: alloc(slots)?,
            wts: alloc(slots)?,
            act: alloc(slots * c.ff_exp)?,
            offsets: gpu.buffer((c.experts + 2) * 4, false, false, None)?,
            perm: alloc(slots)?,
            ys: if max_batch >= GROUPED_MIN {
                alloc(slots * c.d)?
            } else {
                gpu.buffer(4, false, false, None)?
            },
            ik: alloc(c.idx_dim)?,
            iq: alloc(c.idx_heads * c.idx_dim)?,
            iqr: alloc(c.idx_heads * c.idx_dim)?,
            scores: if sparse {
                alloc(context / c.kpool)?
            } else {
                gpu.buffer(4, false, false, None)?
            },
            sel: if sparse {
                alloc(c.top_blocks())?
            } else {
                gpu.buffer(4, false, false, None)?
            },
            pe: alloc(d)?,
            pk: alloc(w)?,
            pv: alloc(d)?,
            gv: alloc(w)?,
            pn: alloc(w)?,
            fence,
            graphs: HashMap::new(),
            head_graphs: HashMap::new(),
            queries,
            profile: Vec::new(),
            phases: Vec::new(),
            c: c.clone(),
            gpu,
        })
    }

    /// Indexer upkeep and block selection for one QSA layer: raw keys into the ring, pooled keys of completed blocks,
    /// the normed and rotated indexer queries, then each token's selected blocks in `sel`.
    fn index_select(
        &self,
        current: &std::cell::Cell<vk::CommandBuffer>,
        ix: &Index,
        attn: &[u32; 16],
        n: usize,
    ) -> Result<()> {
        let gpu = &self.gpu;
        let c = &self.c;
        let push = |ptr: [u64; 6], u: [u32; 16]| Push {
            ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], self.params.address, 0],
            u,
        };
        let (id, ih, d) = (c.idx_dim, c.idx_heads, c.d);
        let matf32 = |wt: &Buffer, y: &Buffer, rows: usize| -> Result<()> {
            let mut u = [0; 16];
            u[0] = rows as u32;
            u[1] = d as u32;
            gpu.dispatch_matrix(
                current.get(),
                "qwen_matf32",
                push([0, self.u.address, y.address, 0, 0, 0], u),
                rows.div_ceil(4) as u32,
                n.div_ceil(8) as u32,
                wt,
            )
        };
        matf32(&ix.k, &self.ik, id)?;
        let mut u = [0; 16];
        u[6] = id as u32;
        u[10] = self.ring as u32;
        gpu.dispatch(
            current.get(),
            "qwen4_idx_store",
            push([self.ik.address, 0, ix.raw.address, 0, 0, 0], u),
            1,
            n as u32,
        );
        let mut u = [0; 16];
        u[1] = c.rot as u32;
        u[6] = id as u32;
        u[7] = self.context as u32;
        u[8] = c.eps.to_bits();
        u[9] = c.kpool as u32;
        u[10] = self.ring as u32;
        u[12] = c.rope_base.to_bits();
        gpu.dispatch(
            current.get(),
            "qwen4_idx_pool",
            push([ix.raw.address, ix.k_norm.address, ix.pooled.address, 0, 0, 0], u),
            1,
            n as u32,
        );
        matf32(&ix.q, &self.iq, ih * id)?;
        let mut u = [0; 16];
        u[1] = c.rot as u32;
        u[2] = (ih * id) as u32;
        u[3] = id as u32;
        u[4] = ih as u32;
        u[6] = id as u32;
        u[7] = self.context as u32;
        u[8] = c.eps.to_bits();
        u[12] = c.rope_base.to_bits();
        u[13] = 1f32.to_bits();
        gpu.dispatch(
            current.get(),
            "qwen_qk",
            push([self.iq.address, ix.q_norm.address, self.iqr.address, 0, 0, 0], u),
            ih as u32,
            n as u32,
        );
        let mut u = [0; 16];
        u[2] = c.top_blocks() as u32;
        u[4] = ih as u32;
        u[6] = id as u32;
        u[7] = self.context as u32;
        u[9] = c.kpool as u32;
        let _ = attn;
        // Scores over every block a recorded graph may see (the position changes on replay; groups past it exit).
        gpu.dispatch(
            current.get(),
            "qwen4_idx_score",
            push([self.iqr.address, ix.pooled.address, self.scores.address, 0, 0, 0], u),
            (self.context / c.kpool).div_ceil(64) as u32,
            n.div_ceil(8) as u32,
        );
        gpu.dispatch(
            current.get(),
            "qwen4_idx_select",
            push([0, 0, self.scores.address, self.sel.address, 0, 0], u),
            1,
            n as u32,
        );
        Ok(())
    }

    /// Score the hidden state at `input` with this stage's head rows: logits, chunk top-k, merged top-k in `hd.top`.
    fn score_head(&self, graph: &mut Graph, cmd: vk::CommandBuffer, hd: &Head, input: u64, dump: bool) -> Result<()> {
        let gpu = &self.gpu;
        let d = self.c.d;
        let push = |ptr: [u64; 6], u: [u32; 16]| Push {
            ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], self.params.address, 0],
            u,
        };
        let mut u = [0; 16];
        u[0] = hd.rows as u32;
        u[1] = d as u32;
        u[3] = (hd.rows * d / 16) as u32;
        u[9] = 1;
        u[15] = 1;
        gpu.dispatch_matrix(
            cmd,
            "headvec",
            push([0, input, hd.logits.address, 0, 0, 0], u),
            hd.rows.div_ceil(16) as u32,
            1,
            &hd.w,
        )?;
        if dump {
            let host = gpu.buffer(hd.rows * 4, true, false, None)?;
            gpu.copy(cmd, &hd.logits, 0, &host, 0, hd.rows * 4);
            let name = if hd.rows == self.c.vocab {
                "result_output".to_string()
            } else {
                format!("result_output@{}", hd.row0)
            };
            graph.captures.push((name, host, hd.rows));
        }
        let mut u = [0; 16];
        u[0] = hd.rows as u32;
        u[1] = TOP_K as u32;
        u[2] = (hd.chunks * TOP_K) as u32;
        u[15] = 1;
        gpu.dispatch(
            cmd,
            "topk",
            push([hd.logits.address, 0, hd.scratch.address, 0, 0, 0], u),
            hd.chunks as u32,
            1,
        );
        let mut u = [0; 16];
        u[0] = (hd.chunks * TOP_K) as u32;
        u[1] = TOP_K as u32;
        u[15] = 1;
        gpu.dispatch(
            cmd,
            "merge",
            push([hd.scratch.address, 0, hd.top.address, 0, 0, 0], u),
            1,
            1,
        );
        Ok(())
    }

    /// The merged candidates of a head (ids offset to the vocabulary).
    fn read_top(hd: &Head) -> Result<Vec<(f32, u32)>> {
        let bytes = hd.top.read(TOP_K * 8)?;
        let mut top: Vec<(f32, u32)> = bytes
            .chunks_exact(8)
            .map(|s| {
                (
                    f32::from_le_bytes(s[..4].try_into().unwrap()),
                    u32::from_le_bytes(s[4..].try_into().unwrap()) + hd.row0 as u32,
                )
            })
            .collect();
        top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        Ok(top)
    }

    /// Score a final hidden state (from the stage that ends the model) with this stage's head rows. `None` when the
    /// stage holds no rows or ends the model itself (its rows were scored in `forward`).
    pub fn head_slice(&mut self, h: &[f32], dump: Option<&mut Captures>) -> Result<Option<Vec<(f32, u32)>>> {
        let dumping = dump.is_some();
        if !self.head_submit(h, dumping)? {
            return Ok(None);
        }
        self.head_collect(dump)
    }

    /// Start scoring `h` with this stage's head rows; false when the stage has none to score (see `head_slice`).
    /// Stages on different GPUs can score at the same time; `head_collect` waits for the result.
    pub fn head_submit(&mut self, h: &[f32], dump: bool) -> Result<bool> {
        let d = self.c.d;
        let Some(hd) = self.head.as_ref().filter(|hd| hd.mix.is_none()) else {
            return Ok(false);
        };
        ensure!(h.len() == d, "hidden state length mismatch");
        hd.h.write(bytemuck::cast_slice(h))?;
        if !self.head_graphs.contains_key(&dump) {
            let cmd = self.gpu.begin()?;
            let mut graph = Graph {
                gpu: self.gpu.clone(),
                cmds: vec![cmd],
                captures: Vec::new(),
                marks: Vec::new(),
            };
            self.gpu.barrier(cmd);
            self.score_head(&mut graph, cmd, hd, hd.h.address, dump)?;
            self.gpu.finish(cmd)?;
            self.head_graphs.insert(dump, graph);
        }
        unsafe {
            self.gpu.dev.reset_fences(&[self.fence])?;
        }
        self.gpu.submit(self.head_graphs[&dump].cmds[0], None, None, self.fence)?;
        Ok(true)
    }

    /// Wait for `head_submit` and return its candidates (with vocabulary ids).
    pub fn head_collect(&mut self, dump: Option<&mut Captures>) -> Result<Option<Vec<(f32, u32)>>> {
        unsafe {
            self.gpu.dev.wait_for_fences(&[self.fence], true, 600_000_000_000)?;
        }
        if let Some(m) = dump {
            for (name, host, width) in &self.head_graphs[&true].captures {
                m.insert(name.clone(), bytemuck::cast_slice(&host.read(width * 4)?).to_vec());
            }
        }
        Ok(Some(Self::read_top(self.head.as_ref().context("no head rows")?)?))
    }

    pub fn device_mib(&self) -> f64 {
        self.gpu.device_bytes.get() as f64 / 1048576.
    }

    fn record(&self, n: usize, parts: u32, dump: bool) -> Result<Graph> {
        let gpu = &self.gpu;
        let c = &self.c;
        let (d, w, hc) = (c.d, c.width(), c.hc);
        let current = std::cell::Cell::new(gpu.begin()?);
        let mut graph = Graph {
            gpu: gpu.clone(),
            cmds: vec![current.get()],
            captures: Vec::new(),
            marks: Vec::new(),
        };
        let cmd = current.get();
        // Timestamps around every layer's submission (the last one includes the head or the output copy).
        let stamp = |cmd: vk::CommandBuffer, index: usize, end: bool| unsafe {
            let stage = if end {
                vk::PipelineStageFlags::BOTTOM_OF_PIPE
            } else {
                vk::PipelineStageFlags::TOP_OF_PIPE
            };
            gpu.dev.cmd_write_timestamp(cmd, stage, self.queries, index as u32);
        };
        unsafe {
            gpu.dev
                .cmd_reset_query_pool(cmd, self.queries, 0, (2 + MARKS) as u32 * self.layers.len() as u32)
        };
        let base = 2 * self.layers.len();
        let mark = |graph: &mut Graph, layer: usize, name: &'static str| {
            stamp(current.get(), base + graph.marks.len(), true);
            graph.marks.push((layer, name));
        };
        stamp(cmd, 0, false);
        gpu.barrier(cmd);
        gpu.copy(cmd, &self.params_host, 0, &self.params, 0, 16);
        gpu.copy(cmd, &self.input, 0, &self.x, 0, n * w * 4);
        if self.ple.is_some() {
            gpu.copy(cmd, &self.ple_in, 0, &self.pe, 0, n * d * 4);
        }
        let push = |ptr: [u64; 6], u: [u32; 16]| Push {
            ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], self.params.address, 0],
            u,
        };
        let eps = c.eps.to_bits();
        let capture = |graph: &mut Graph, name: String, b: &Buffer, width: usize| -> Result<()> {
            if dump {
                let host = gpu.buffer(width * 4, true, false, None)?;
                gpu.copy(current.get(), b, (n - 1) * width * 4, &host, 0, width * 4);
                graph.captures.push((name, host, width));
            }
            Ok(())
        };
        // Q4_0 projections use the Gemma kernels (GEMM for prefill, multi-column for small batches), Q8_0 ones the
        // qwen4_q8 kernels.
        let matrix = |wt: &Dense, x: &Buffer, y: &Buffer, k: usize, rows: usize, defer: bool| -> Result<()> {
            let mut u = [0; 16];
            u[0] = rows as u32;
            u[1] = k as u32;
            let (wt, (shader, gx, gy)) = match wt {
                Dense::Q4(b) => {
                    u[3] = (rows * k / 32) as u32;
                    (
                        b,
                        match n {
                            1 => (q4_decode_shader(rows, k), rows.div_ceil(16), 1),
                            2 => ("multivec4x2", rows.div_ceil(16), 1),
                            3 => ("multivec4x3", rows.div_ceil(16), 1),
                            4..=8 => ("multivec4", rows.div_ceil(16), n.div_ceil(4)),
                            _ => ("gemm", rows.div_ceil(64), n.div_ceil(64)),
                        },
                    )
                }
                Dense::Q8(b, stride) => {
                    u[3] = *stride;
                    (
                        b,
                        match n {
                            1 => ("qwen4_q8_mv", rows.div_ceil(16), 1),
                            2..=8 => ("qwen4_q8_mv4", rows.div_ceil(16), n.div_ceil(4)),
                            _ => ("qwen4_q8_gemm", rows.div_ceil(64), n.div_ceil(64)),
                        },
                    )
                }
            };
            gpu.dispatch_matrix(
                current.get(),
                shader,
                push([0, x.address, y.address, 0, 0, 0], u).defer_barrier(defer),
                gx as u32,
                gy as u32,
                wt,
            )
        };
        // Small f32 projections: one wave per row and eight tokens.
        let matf32 = |wt: &Buffer, x: &Buffer, y: &Buffer, k: usize, rows: usize, defer: bool| -> Result<()> {
            let mut u = [0; 16];
            u[0] = rows as u32;
            u[1] = k as u32;
            gpu.dispatch_matrix(
                current.get(),
                "qwen_matf32",
                push([0, x.address, y.address, 0, 0, 0], u).defer_barrier(defer),
                rows.div_ceil(4) as u32,
                n.div_ceil(8) as u32,
                wt,
            )
        };
        let act = |b: &Buffer, width: usize, kind: u32, scale: f32| {
            let mut u = [0; 16];
            u[0] = width as u32;
            u[9] = kind;
            u[12] = scale.to_bits();
            gpu.dispatch(
                current.get(),
                "qwen4_act",
                push([b.address, 0, 0, 0, 0, 0], u),
                width.div_ceil(64) as u32,
                n as u32,
            );
        };
        // Where the pending write weights are: (buffer address, row stride, offset).
        let wloc = std::cell::Cell::new((self.w.address, hc as u32, 0u32));
        let mo_cur = std::cell::Cell::new(0usize);
        let fused = n <= 8
            && self
                .layers
                .iter()
                .all(|l| l.attn_mix.fused.is_some() && l.ffn_mix.fused.is_some());
        // Norm all streams into xn (applying the pending combine first), or with `norm` None only combine.
        let hc_norm = |pending: bool, norm: Option<&Buffer>| {
            let mut u = [0; 16];
            u[0] = d as u32;
            u[2] = wloc.get().1;
            u[4] = hc as u32;
            u[10] = wloc.get().2;
            u[11] = u32::from(pending);
            u[12] = eps;
            gpu.dispatch(
                current.get(),
                "qwen4_hc_norm",
                push(
                    [
                        self.x.address,
                        self.y.address,
                        norm.map_or(0, |_| self.xn.address),
                        norm.map_or(0, |b| b.address),
                        wloc.get().0,
                        0,
                    ],
                    u,
                ),
                1,
                n as u32,
            );
        };
        // A mixer: sublayer input in u and, with an inject matrix, write weights in w (or the fused output).
        let mixer = |graph: &mut Graph, m: &Mix, pending: bool, tag: &str| -> Result<()> {
            if let (true, Some(fm)) = (fused, &m.fused) {
                let (cur, next) = (mo_cur.get(), 1 - mo_cur.get());
                let (mo_next, mo_prev) = (&self.mo[next], &self.mo[cur]);
                let (wb, stride, q8) = match fm {
                    Dense::Q4(b) => (b, ((c.hc_rank + 4) * w / 32) as u32, false),
                    Dense::Q8(b, st) => (b, *st, true),
                };
                let mut u = [0; 16];
                u[0] = d as u32;
                u[1] = w as u32;
                u[2] = c.hc_rank as u32;
                u[3] = stride;
                u[11] = u32::from(pending);
                u[12] = eps;
                gpu.dispatch_matrix(
                    current.get(),
                    if q8 { "qwen4_mix_down_q8" } else { "qwen4_mix_down" },
                    push(
                        [
                            self.x.address,
                            self.y.address,
                            mo_next.address,
                            mo_prev.address,
                            m.norm.address,
                            0,
                        ],
                        u,
                    ),
                    (c.hc_rank + 4).div_ceil(16) as u32,
                    n as u32,
                    wb,
                )?;
                let (ub, ustride, uq8) = match &m.up {
                    Dense::Q4(b) => (b, (w * c.hc_rank / 32) as u32, false),
                    Dense::Q8(b, st) => (b, *st, true),
                };
                let mut u = [0; 16];
                u[0] = d as u32;
                u[2] = c.hc_rank as u32;
                u[3] = ustride;
                u[11] = u32::from(pending);
                gpu.dispatch_matrix(
                    current.get(),
                    if uq8 { "qwen4_mix_up_q8" } else { "qwen4_mix_up" },
                    push(
                        [
                            self.x.address,
                            mo_next.address,
                            self.u.address,
                            mo_prev.address,
                            m.norm.address,
                            self.y.address,
                        ],
                        u,
                    ),
                    d.div_ceil(16) as u32,
                    n as u32,
                    ub,
                )?;
                capture(graph, format!("hc_mixed{tag}"), &self.u, d)?;
                mo_cur.set(next);
                wloc.set((mo_next.address, (c.hc_rank + 8) as u32, c.hc_rank as u32));
                return Ok(());
            }
            hc_norm(pending, Some(&m.norm));
            capture(graph, format!("hc_norm{tag}"), &self.xn, w)?;
            matrix(&m.down, &self.xn, &self.lo, w, c.hc_rank, false)?;
            act(&self.lo, c.hc_rank, 0, 1. / hc as f32);
            matrix(&m.up, &self.lo, &self.hgate, c.hc_rank, w, false)?;
            capture(graph, format!("hc_gate{tag}"), &self.hgate, w)?;
            let mut u = [0; 16];
            u[0] = d as u32;
            u[4] = hc as u32;
            gpu.dispatch(
                current.get(),
                "qwen4_hc_mix",
                push([self.xn.address, self.hgate.address, self.u.address, 0, 0, 0], u),
                d.div_ceil(256) as u32,
                n as u32,
            );
            capture(graph, format!("hc_mixed{tag}"), &self.u, d)?;
            if let Some(inj) = &m.inject {
                matf32(inj, &self.xn, &self.w, w, hc, false)?;
                capture(graph, format!("hc_inject{tag}"), &self.w, hc)?;
                act(&self.w, hc, 1, 1. / hc as f32);
                wloc.set((self.w.address, hc as u32, 0));
            }
            Ok(())
        };
        let mut pending = false;
        for l in &self.layers {
            let il = l.il;
            if let (Some(p), true) = (&self.ple, c.ngram.as_ref().is_some_and(|p| p.layer == il)) {
                hc_norm(pending, None);
                pending = false;
                capture(&mut graph, "ple_embd".into(), &self.pe, d)?;
                matrix(&p.key, &self.pe, &self.pk, d, w, true)?;
                matrix(&p.value, &self.pe, &self.pv, d, d, false)?;
                let mut u = [0; 16];
                u[0] = d as u32;
                u[4] = hc as u32;
                u[12] = eps;
                gpu.dispatch(
                    current.get(),
                    "qwen4_ple_gate",
                    push(
                        [
                            self.pk.address,
                            self.pv.address,
                            self.x.address,
                            p.norms.address,
                            self.gv.address,
                            self.pn.address,
                        ],
                        u,
                    ),
                    1,
                    n as u32,
                );
                capture(&mut graph, format!("ple_gated_value-{il}"), &self.gv, w)?;
                let mut u = [0; 16];
                u[0] = w as u32;
                gpu.dispatch(
                    current.get(),
                    "qwen4_ple_conv",
                    push(
                        [
                            self.x.address,
                            self.gv.address,
                            self.pn.address,
                            p.conv_w.address,
                            p.hist.address,
                            0,
                        ],
                        u,
                    ),
                    w.div_ceil(256) as u32,
                    1,
                );
            }
            let li = il - self.first;
            if c.ngram.as_ref().is_some_and(|p| p.layer == il) && self.ple.is_some() {
                mark(&mut graph, li, "ngram");
            }
            mixer(&mut graph, &l.attn_mix, pending, &format!("-{il}"))?;
            mark(&mut graph, li, "mixer");
            match &l.mixer {
                Mixer::Delta {
                    qkv,
                    z,
                    ab,
                    dt_a,
                    conv_w,
                    conv,
                    state,
                    norm,
                    out,
                } => {
                    let (cd, vd) = (c.conv_dim(), c.value_dim());
                    matrix(qkv, &self.u, &self.big, d, cd, true)?;
                    matrix(z, &self.u, &self.z, d, vd, true)?;
                    matf32(ab, &self.u, &self.ab, d, 2 * c.v_heads, false)?;
                    capture(&mut graph, format!("linear_attn_qkv_mixed-{il}"), &self.big, cd)?;
                    capture(&mut graph, format!("z-{il}"), &self.z, vd)?;
                    let mut u = [0; 16];
                    u[0] = cd as u32;
                    gpu.dispatch(
                        current.get(),
                        "qwen_conv",
                        push(
                            [
                                self.big.address,
                                conv_w.address,
                                conv.address,
                                self.conv_out.address,
                                0,
                                0,
                            ],
                            u,
                        ),
                        cd.div_ceil(256) as u32,
                        1,
                    );
                    capture(&mut graph, format!("conv_output_silu-{il}"), &self.conv_out, cd)?;
                    let mut u = [0; 16];
                    u[0] = cd as u32;
                    u[4] = c.v_heads as u32;
                    u[5] = c.k_heads as u32;
                    u[6] = c.state as u32;
                    u[12] = eps;
                    gpu.dispatch(
                        current.get(),
                        "qwen_gdn",
                        push(
                            [
                                self.conv_out.address,
                                self.ab.address,
                                state.address,
                                self.o.address,
                                dt_a.address,
                                0,
                            ],
                            u,
                        ),
                        (c.v_heads * c.state / crate::qwen::gpu::GDN_ROWS_PER_GROUP) as u32,
                        1,
                    );
                    capture(&mut graph, format!("attn_output-{il}"), &self.o, vd)?;
                    let mut u = [0; 16];
                    u[0] = vd as u32;
                    u[6] = c.state as u32;
                    u[12] = eps;
                    gpu.dispatch(
                        current.get(),
                        "qwen4_gated_norm",
                        push([self.o.address, self.z.address, norm.address, 0, 0, 0], u),
                        c.v_heads as u32,
                        n as u32,
                    );
                    capture(&mut graph, format!("final_output-{il}"), &self.o, vd)?;
                    matrix(out, &self.o, &self.y, vd, d, false)?;
                    capture(&mut graph, format!("linear_attn_out-{il}"), &self.y, d)?;
                }
                Mixer::Attention {
                    q,
                    k,
                    v,
                    q_norm,
                    k_norm,
                    out,
                    kc,
                    vc,
                    index,
                } => {
                    let (hd, kvd) = (c.head_dim, c.kv_heads * c.head_dim);
                    let qd = c.heads * hd;
                    matrix(q, &self.u, &self.big, d, 2 * qd, true)?;
                    matrix(k, &self.u, &self.kb, d, kvd, true)?;
                    matrix(v, &self.u, &self.vb, d, kvd, false)?;
                    for (src, norm, out, cache, heads, stride, width, scale, defer) in [
                        (
                            &self.big,
                            Some(q_norm),
                            Some(&self.rq),
                            None,
                            c.heads,
                            2 * hd,
                            2 * qd,
                            1. / (hd as f32).sqrt(),
                            true,
                        ),
                        (
                            &self.kb,
                            Some(k_norm),
                            self.kv.k_q8().then_some(&self.kq),
                            (!self.kv.k_q8()).then_some(kc),
                            c.kv_heads,
                            hd,
                            kvd,
                            1.,
                            true,
                        ),
                        (
                            &self.vb,
                            None,
                            self.kv.v_q8().then_some(&self.vq),
                            (!self.kv.v_q8()).then_some(vc),
                            c.kv_heads,
                            hd,
                            kvd,
                            1f32,
                            false,
                        ),
                    ] {
                        let mut u = [0; 16];
                        u[1] = if norm.is_some() { c.rot as u32 } else { 0 };
                        u[2] = width as u32;
                        u[3] = stride as u32;
                        u[4] = heads as u32;
                        u[6] = hd as u32;
                        u[7] = self.context as u32;
                        u[8] = eps;
                        u[12] = c.rope_base.to_bits();
                        u[13] = scale.to_bits();
                        gpu.dispatch(
                            current.get(),
                            "qwen_qk",
                            push(
                                [
                                    src.address,
                                    norm.map_or(0, |b| b.address),
                                    out.map_or(0, |b| b.address),
                                    0,
                                    0,
                                    cache.map_or(0, |b| b.address),
                                ],
                                u,
                            )
                            .defer_barrier(defer),
                            heads as u32,
                            n as u32,
                        );
                    }
                    if self.kv != KvCache::F16 {
                        let mut u = [0; 16];
                        u[5] = c.kv_heads as u32;
                        u[6] = hd as u32;
                        u[7] = self.context as u32;
                        u[11] = self.kv.k_q8() as u32 | (self.kv.v_q8() as u32) << 1;
                        gpu.dispatch(
                            current.get(),
                            "qwen4_kv_store",
                            push([self.kq.address, self.vq.address, kc.address, vc.address, 0, 0], u),
                            1,
                            n as u32,
                        );
                    }
                    capture(&mut graph, format!("Qcur-{il}"), &self.rq, qd)?;
                    let mut u = [0; 16];
                    u[4] = c.heads as u32;
                    u[5] = c.kv_heads as u32;
                    u[6] = hd as u32;
                    u[7] = self.context as u32;
                    if let Some(ix) = index {
                        mark(&mut graph, li, "qkv");
                        self.index_select(&current, ix, &u, n)?;
                        mark(&mut graph, li, "index");
                        capture(
                            &mut graph,
                            format!("indexer_q-{il}"),
                            &self.iqr,
                            c.idx_heads * c.idx_dim,
                        )?;
                        let mut u = u;
                        u[9] = c.kpool as u32;
                        u[2] = c.top_blocks() as u32;
                        let ptr = [
                            self.rq.address,
                            kc.address,
                            vc.address,
                            self.o.address,
                            self.scratch.address,
                            self.sel.address,
                        ];
                        if parts == 0 {
                            gpu.dispatch(
                                current.get(),
                                match self.kv {
                                    KvCache::F16 => "qwen4_attention_sparse_full",
                                    KvCache::Q8 => "qwen4_attention_sparse_full_q8",
                                    KvCache::K16V8 => "qwen4_attention_sparse_full_k16v8",
                                },
                                push(ptr, u),
                                c.heads as u32,
                                n as u32,
                            );
                        } else {
                            let (_, merge) = attention_shaders(parts)?;
                            let partial = match (self.kv, parts) {
                                (KvCache::F16, 8) => "qwen4_attention_sparse8",
                                (KvCache::F16, 16) => "qwen4_attention_sparse16",
                                (KvCache::F16, 64) => "qwen4_attention_sparse64",
                                (KvCache::F16, _) => "qwen4_attention_sparse",
                                (KvCache::Q8, 8) => "qwen4_attention_sparse_q8_8",
                                (KvCache::Q8, 16) => "qwen4_attention_sparse_q8_16",
                                (KvCache::Q8, 64) => "qwen4_attention_sparse_q8_64",
                                (KvCache::Q8, _) => "qwen4_attention_sparse_q8",
                                (KvCache::K16V8, 8) => "qwen4_attention_sparse_k16v8_8",
                                (KvCache::K16V8, 16) => "qwen4_attention_sparse_k16v8_16",
                                (KvCache::K16V8, 64) => "qwen4_attention_sparse_k16v8_64",
                                (KvCache::K16V8, _) => "qwen4_attention_sparse_k16v8",
                            };
                            gpu.dispatch(current.get(), partial, push(ptr, u), c.heads as u32 * parts, n as u32);
                            gpu.dispatch(
                                current.get(),
                                merge,
                                push([0, 0, 0, self.o.address, self.scratch.address, 0], u),
                                c.heads as u32,
                                n as u32,
                            );
                        }
                    } else if parts == 0 {
                        gpu.dispatch(
                            current.get(),
                            "attention",
                            push([self.rq.address, kc.address, vc.address, self.o.address, 0, 0], u),
                            c.heads as u32,
                            n as u32,
                        );
                    } else {
                        let (partial, merge) = attention_shaders(parts)?;
                        gpu.dispatch(
                            current.get(),
                            partial,
                            push(
                                [
                                    self.rq.address,
                                    kc.address,
                                    vc.address,
                                    self.o.address,
                                    self.scratch.address,
                                    0,
                                ],
                                u,
                            ),
                            c.heads as u32 * parts,
                            n as u32,
                        );
                        gpu.dispatch(
                            current.get(),
                            merge,
                            push([0, 0, 0, self.o.address, self.scratch.address, 0], u),
                            c.heads as u32,
                            n as u32,
                        );
                    }
                    capture(&mut graph, format!("attn_pregate-{il}"), &self.o, qd)?;
                    let mut u = [0; 16];
                    u[0] = qd as u32;
                    u[6] = hd as u32;
                    gpu.dispatch(
                        current.get(),
                        "qwen_gate",
                        push([self.o.address, self.big.address, 0, 0, 0, 0], u),
                        qd.div_ceil(256) as u32,
                        n as u32,
                    );
                    capture(&mut graph, format!("attn_gated-{il}"), &self.o, qd)?;
                    matrix(out, &self.o, &self.y, qd, d, false)?;
                    capture(&mut graph, format!("attn_output-{il}"), &self.y, d)?;
                }
            }
            mark(&mut graph, li, if c.recurrent(il) { "deltanet" } else { "attention" });
            mixer(&mut graph, &l.ffn_mix, true, &format!("-{il}#2"))?;
            mark(&mut graph, li, "mixer");
            capture(&mut graph, format!("hc_combine-{il}"), &self.x, w)?;
            // MoE: router (with the shared gate as its last row), top-k, experts, weighted sum.
            let (ne, slots, ff) = (c.experts, c.used + 1, c.ff_exp);
            matf32(&l.router, &self.u, &self.logits, d, ne + 1, false)?;
            capture(&mut graph, format!("ffn_moe_logits_all-{il}"), &self.logits, ne + 1)?;
            let mut u = [0; 16];
            u[0] = ne as u32;
            u[2] = c.used as u32;
            gpu.dispatch(
                current.get(),
                "qwen4_topk",
                push([self.logits.address, 0, self.ids.address, self.wts.address, 0, 0], u),
                1,
                n as u32,
            );
            capture(&mut graph, format!("moe_weights-{il}"), &self.wts, slots)?;
            mark(&mut graph, li, "router");
            let (gu_stride, down_stride) = (l.gu_stride, l.down_stride);
            if n < GROUPED_MIN {
                // Small batches: one workgroup per (token, slot), each re-reading its expert's rows.
                let mut u = [0; 16];
                u[0] = ff as u32;
                u[1] = d as u32;
                u[2] = slots as u32;
                u[3] = gu_stride;
                gpu.dispatch_matrix(
                    current.get(),
                    expert_shader("qwen4_moe_gu", l.gu_fmt),
                    push([0, self.u.address, self.act.address, self.ids.address, 0, 0], u),
                    ff.div_ceil(16) as u32,
                    (n * slots) as u32,
                    &l.gate_up,
                )?;
                let mut u = [0; 16];
                u[0] = d as u32;
                u[1] = ff as u32;
                u[2] = slots as u32;
                u[3] = down_stride;
                gpu.dispatch_matrix(
                    current.get(),
                    expert_shader("qwen4_moe_down", l.down_fmt),
                    push(
                        [
                            0,
                            self.act.address,
                            self.y.address,
                            self.ids.address,
                            self.wts.address,
                            0,
                        ],
                        u,
                    ),
                    d.div_ceil(16) as u32,
                    n as u32,
                    &l.down,
                )?;
            } else {
                // Prefill: group the slots by expert, decode each expert's rows once for all of its slots.
                let mut u = [0; 16];
                u[0] = (ne + 1) as u32;
                u[2] = slots as u32;
                gpu.dispatch(
                    current.get(),
                    "qwen4_moe_route",
                    push([self.ids.address, 0, self.offsets.address, self.perm.address, 0, 0], u),
                    1,
                    1,
                );
                let mut u = [0; 16];
                u[0] = ff as u32;
                u[1] = d as u32;
                u[2] = slots as u32;
                u[3] = gu_stride;
                gpu.dispatch_matrix(
                    current.get(),
                    expert_shader("qwen4_moe_gu_grouped", l.gu_fmt),
                    push(
                        [
                            0,
                            self.u.address,
                            self.act.address,
                            self.offsets.address,
                            self.perm.address,
                            0,
                        ],
                        u,
                    ),
                    ff.div_ceil(16) as u32,
                    (ne + 1) as u32,
                    &l.gate_up,
                )?;
                let mut u = [0; 16];
                u[0] = d as u32;
                u[1] = ff as u32;
                u[2] = slots as u32;
                u[3] = down_stride;
                gpu.dispatch_matrix(
                    current.get(),
                    expert_shader("qwen4_moe_down_grouped", l.down_fmt),
                    push(
                        [
                            0,
                            self.act.address,
                            self.ys.address,
                            self.offsets.address,
                            self.perm.address,
                            self.wts.address,
                        ],
                        u,
                    ),
                    d.div_ceil(16) as u32,
                    (ne + 1) as u32,
                    &l.down,
                )?;
                let mut u = [0; 16];
                u[0] = d as u32;
                u[2] = slots as u32;
                gpu.dispatch(
                    current.get(),
                    "qwen4_moe_sum",
                    push([self.ys.address, 0, self.y.address, 0, 0, 0], u),
                    d.div_ceil(256) as u32,
                    n as u32,
                );
            }
            capture(&mut graph, format!("ffn_out-{il}"), &self.y, d)?;
            mark(&mut graph, li, "experts");
            pending = true;
            if l.il + 1 < self.last {
                let li = l.il - self.first;
                stamp(current.get(), 2 * li + 1, true);
                gpu.finish(current.get())?;
                let next = gpu.begin()?;
                stamp(next, 2 * li + 2, false);
                gpu.barrier(next);
                graph.cmds.push(next);
                current.set(next);
            }
        }
        let last = self.last - 1;
        if let Some((hd, m)) = self.head.as_ref().and_then(|hd| Some((hd, hd.mix.as_ref()?))) {
            mixer(&mut graph, m, pending, "")?;
            capture(&mut graph, format!("l_last-{last}"), &self.x, w)?;
            capture(&mut graph, "result_norm".into(), &self.u, d)?;
            gpu.copy(current.get(), &self.u, (n - 1) * d * 4, &hd.h, 0, d * 4);
            self.score_head(&mut graph, current.get(), hd, self.u.address + ((n - 1) * d * 4) as u64, dump)?;
        } else {
            if pending {
                hc_norm(true, None);
            }
            capture(&mut graph, format!("l_last-{last}"), &self.x, w)?;
            gpu.copy(current.get(), &self.x, 0, &self.output, 0, n * w * 4);
        }
        stamp(current.get(), 2 * self.layers.len() - 1, true);
        gpu.finish(current.get())?;
        Ok(graph)
    }

    /// Run `n` tokens at `pos` on the `n x hc x d` residual; `ple` carries the gathered n-gram rows when this stage
    /// owns the PLE layer. Position 0 starts a new sequence.
    pub fn forward(
        &mut self,
        x: &[f32],
        ple: Option<&[f32]>,
        pos: usize,
        n: usize,
        dump: Option<&mut Captures>,
    ) -> Result<Output> {
        let (d, w) = (self.c.d, self.c.width());
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(pos + n <= self.context, "context exhausted");
        ensure!(x.len() == n * w, "residual length mismatch");
        self.input.write(bytemuck::cast_slice(x))?;
        if self.ple.is_some() {
            let p = ple.context("this stage needs the n-gram rows")?;
            ensure!(p.len() == n * d, "n-gram rows length mismatch");
            self.ple_in.write(bytemuck::cast_slice(p))?;
        }
        self.params_host
            .write(bytemuck::cast_slice(&[pos as u32, n as u32, 0, 0]))?;
        let key = (n, partitions(pos, n), dump.is_some());
        if !self.graphs.contains_key(&key) {
            if self.graphs.len() >= GRAPH_CACHE_CAPACITY {
                let victim = *self
                    .graphs
                    .keys()
                    .find(|k| k.0 != 1)
                    .unwrap_or(self.graphs.keys().next().unwrap());
                self.graphs.remove(&victim);
            }
            let graph = self.record(n, key.1, key.2)?;
            self.graphs.insert(key, graph);
        }
        let graph = &self.graphs[&key];
        unsafe {
            self.gpu.dev.reset_fences(&[self.fence])?;
        }
        for (i, &cmd) in graph.cmds.iter().enumerate() {
            let last = i + 1 == graph.cmds.len();
            self.gpu
                .submit(cmd, None, None, if last { self.fence } else { vk::Fence::null() })?;
        }
        unsafe {
            self.gpu.dev.wait_for_fences(&[self.fence], true, 600_000_000_000)?;
        }
        let mut times = vec![0u64; 2 * self.layers.len() + graph.marks.len()];
        unsafe {
            self.gpu.dev.get_query_pool_results(
                self.queries,
                0,
                &mut times,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let period = self.gpu.period as f64 / 1e6;
        let layers = self.layers.len();
        self.profile = times[..2 * layers]
            .chunks_exact(2)
            .map(|t| t[1].wrapping_sub(t[0]) as f64 * period)
            .collect();
        let mut prev = vec![0u64; layers];
        for (li, p) in prev.iter_mut().enumerate() {
            *p = times[2 * li];
        }
        self.phases = graph
            .marks
            .iter()
            .enumerate()
            .map(|(i, &(li, name))| {
                let t = times[2 * layers + i];
                let ms = t.wrapping_sub(prev[li]) as f64 * period;
                prev[li] = t;
                (li, name, ms)
            })
            .collect();
        if let Some(m) = dump {
            for (name, host, width) in &graph.captures {
                let v: Vec<f32> = bytemuck::cast_slice(&host.read(width * 4)?).to_vec();
                m.insert(name.clone(), v);
            }
        }
        Ok(match self.head.as_ref().filter(|hd| hd.mix.is_some()) {
            Some(hd) => Output::Top(Self::read_top(hd)?, bytemuck::cast_slice(&hd.h.read(d * 4)?).to_vec()),
            None => Output::Hidden(bytemuck::cast_slice(&self.output.read(n * w * 4)?).to_vec()),
        })
    }
}

impl Drop for GpuStage {
    fn drop(&mut self) {
        unsafe {
            let _ = self.gpu.dev.device_wait_idle();
        }
        self.graphs.clear();
        self.head_graphs.clear();
        unsafe {
            self.gpu.dev.destroy_fence(self.fence, None);
            self.gpu.dev.destroy_query_pool(self.queries, None);
        }
    }
}
