//! One GPU holding a contiguous range of qwen4exp layers (optionally the n-gram module of layer 1, and the final
//! mixer with the LM head). As in the qwen35 stages, each batch size records one command buffer per layer, replayed
//! with a new position; the residual (all hyper-connection streams) enters and leaves through host buffers.
//!
//! Every mixer applies the previous sublayer's write (`x_s += w_s * y`) in its norm kernel, so a layer ends with a
//! pending combine that the next mixer, the head or the stage output resolves.
//!
//! Dense matrices are Q4_0 repacked with f32 scales (`quant::Kind::Q4`, the Gemma kernels) or Q8_0 with f16 scales;
//! experts are Q4_0, IQ4_NL (4.5 bits per weight) or IQ4_XS (4.25), at their GGUF size, stacked into one gate/up and
//! one down matrix per layer with the shared expert appended as expert `experts`. Attention is dense, so a stage
//! accepts at most `idx_budget + kpool - 1` visible tokens (the sparse block selection is not implemented here yet).
use super::{cpu::Captures, Config};
use crate::{
    gguf::{GgmlType, Gguf, TensorInfo},
    quant::{Kind, Planes},
    qwen::{dequant_row, vector},
    vulkan::{attention_shaders, q4_decode_shader, Buffer, Device, Push, Stamps, Uploader, STAMP_CAPACITY},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::{collections::HashMap, rc::Rc};

pub const TOP_K: usize = crate::qwen::gpu::TOP_K;
const GRAPH_CACHE_CAPACITY: usize = 6;
/// Phase timestamps per layer (n-gram module, two mixers, sublayer (sparse QSA: projections, block selection,
/// attention), router, experts).
const MARKS: usize = 8;
/// Batches of up to this many tokens use the fused mixer kernels, which read the mixer matrices once per token; larger
/// ones the prefill path, whose cost hardly depends on the batch up to 64 tokens (results/qwen4/prefill-mixer-20261012:
/// 16 Q8_0 mixers fused 8.7 against 9.4 ms at 8 tokens, 11.5 against 9.5 at 12). Q4_0 mixers keep the older prefill
/// path, which the fused kernels beat up to 64 tokens (results/qwen4/prefill-batch-20261012).
const FUSED_MAX: usize = 8;
const FUSED_MAX_Q4: usize = 64;
/// K splits of the Q8_0 prefill mixer's bottleneck GEMM (10240 -> 324 rows, 6 row tiles of 64 alone, 96 workgroups
/// per 64 tokens with the splits; 8 and 32 measured the same).
const MIX_SPLITS: usize = 16;
/// Q8_0 prefill batches of at most this many tokens use the lane-per-row GEMM (`qwen4_q8_gemm_s*`): the 64-token
/// tiles of `qwen4_q8_gemm` compute 64 tokens whatever the batch (results/qwen4/prefill-kernels-20261013).
const LANE_GEMM_MAX: usize = 32;
/// K splits of the Q8_0 prefill out projections (2560 rows, 40 row tiles of 64 alone): 8 when the partial sums fit
/// in hgate, else 4.
const OUT_SPLITS: usize = 8;
/// Batches of at least this many tokens group their expert slots by expert (`qwen4_moe_tiles`), so each expert's
/// weights are decoded once per tile of up to `TILE` slots; smaller ones use the per-(token, slot) kernels
/// (results/qwen4/prefill-experts-20261012: 8 layers' experts 28.2 -> 20.5 ms at 16 tokens, 221 -> 84 ms at 128; a
/// tie at 8).
const GROUPED_MIN: usize = 16;
/// Slots per expert tile (`TILE` in qwen4_moe_{gu,down}_tiles.comp).
const TILE: usize = 8;

/// Tiles needed at most for `slots` slots over `experts` experts.
fn max_tiles(slots: usize, experts: usize) -> usize {
    slots.min(experts + slots / TILE)
}
/// Residual chunk of a qwen4_mix_down_sk workgroup (40 blocks of 32, two per stream of 2560).
const SPLIT: usize = 1280;
/// Rows of a verification batch the head stage can score (speculative decoding: the pending token and its drafts).
pub const VERIFY_MAX: usize = 4;

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
    /// Experts with their scales or headers first (`quant::Kind::Q4h`, `Kind::Iq4xs`) and the word offsets of their
    /// blocks.
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

/// How experts are repacked for the GPU: scales or headers, then the nibble blocks.
fn expert_kind(fmt: u32) -> Kind {
    if fmt == 2 {
        Kind::Iq4xs
    } else {
        Kind::Q4h
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

/// Rows `row0..row0 + rows` of the LM head. The stage that ends the model also runs the final mixer (`mix`) and
/// leaves the final hidden state in `h`; every stage scores the final hidden state it is given (`head_slice`).
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
    /// Dispatch timestamps (`TRASHCAN_Q4_PROFILE=1`): the query pool and a label per query.
    stamps: Option<(vk::QueryPool, Vec<String>)>,
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            self.gpu.dev.free_command_buffers(self.gpu.pool, &self.cmds);
            if let Some((pool, _)) = self.stamps {
                self.gpu.dev.destroy_query_pool(pool, None);
            }
        }
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

/// What a batch keeps besides its output: nothing, or (a verification batch) the per-token states for a rollback and
/// the residual rows on the stage that ends the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Plain,
    Verify,
}

/// Forwards of up to this many tokens record the stage as one command buffer, which waits for `start` and sets
/// `done` (see `GpuStage::presubmit`).
pub const GATED_MAX: usize = 8;
/// Stages after which a graph sets its done event: all prior work.
const DONE_STAGES: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
    vk::PipelineStageFlags::COMPUTE_SHADER.as_raw() | vk::PipelineStageFlags::TRANSFER.as_raw(),
);
/// How long `collect` spins on a done event before it waits for the fence instead.
const SPIN: std::time::Duration = std::time::Duration::from_millis(50);

pub enum Output {
    Hidden(Vec<f32>),
    /// The final hidden state of the last token (this stage ends the model); every stage holding head rows scores
    /// it with `head_submit`.
    Final(Vec<f32>),
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
    /// Split mixer partials `[t][chunk][rank + 8]` and the workgroup counter per token group (qwen4_mix_down_sk).
    mp: Buffer,
    mcount: Buffer,
    /// Partial sums of the n-gram gates `[t][group][16]` (qwen4_ple_sums).
    psum: Buffer,
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
    /// Expert tiles of a prefill batch and its slots grouped by expert (`qwen4_moe_tiles`).
    tiles: Buffer,
    list: Buffer,
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
    graphs: HashMap<(usize, u32, bool, Mode), Graph>,
    /// The graph and batch of a submitted forward that `collect` has not waited for yet.
    pending: Option<((usize, u32, bool, Mode), usize)>,
    /// Command buffers scoring given hidden states with this stage's head rows (by dump and row count).
    head_graphs: HashMap<(bool, usize), Graph>,
    /// Rows scored by the last `head_submit`.
    head_n: usize,
    /// Forwards of up to `GATED_MAX` tokens and head scoring wait for the host to set `start` (`hstart`) before they
    /// read their input and set `done` (`hdone`) at their end, which the host polls; work can then be submitted ahead
    /// of its input (`presubmit`, `head_presubmit`), off the critical path (a stage's submission takes 0.1-0.2 ms on
    /// the D700s, results/qwen4/decode-host-20261013).
    start: vk::Event,
    done: vk::Event,
    hstart: vk::Event,
    hdone: vk::Event,
    head_fence: vk::Fence,
    /// A forward submitted ahead that waits for its input: position, tokens, mode.
    ahead: Option<(usize, usize, Mode)>,
    /// Head scoring submitted ahead that waits for its hidden states: rows.
    head_ahead: Option<usize>,
    /// Head scoring submitted and not collected yet.
    head_pending: bool,
    /// The last forward's (head scoring's) end was seen through `done` (`hdone`); its fence is waited for before reuse.
    unretired: bool,
    head_unretired: bool,
    /// Verification batches (speculative decoding, `enable_verify`): per DeltaNet layer the states and convolution
    /// windows after every token but the last, the n-gram convolution history before the batch, and the latest batch.
    snaps: Vec<Option<(Buffer, Buffer)>>,
    ple_hist0: Option<Buffer>,
    verify_max: usize,
    last_verify: Option<usize>,
    /// Tokens of the latest batch (`tail`).
    last_n: usize,
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
        ("qwen4_moe_gu_tiles", 1) => "qwen4_moe_gu_tiles_iq4nl",
        ("qwen4_moe_gu_tiles", 2) => "qwen4_moe_gu_tiles_iq4xs",
        ("qwen4_moe_down_tiles", 1) => "qwen4_moe_down_tiles_iq4nl",
        _ => name,
    }
}

/// The Q8_0 GEMM of a prefill batch of `n` tokens over `rows` rows split in `parts` along K: the lane-per-row kernel
/// up to `LANE_GEMM_MAX` tokens (16 tokens per workgroup, the token tile first), else the 64 x 64 tiles. Returns the
/// pipeline and its workgroups.
fn q8_gemm(rows: usize, n: usize, parts: usize) -> (&'static str, usize, usize) {
    let parts = parts.max(1);
    if n <= LANE_GEMM_MAX {
        ("qwen4_q8_gemm_s16r1p", n.div_ceil(16), rows.div_ceil(64) * parts)
    } else {
        ("qwen4_q8_gemm", rows.div_ceil(64) * parts, n.div_ceil(64))
    }
}

/// Q8_0 kernels for 1 and 2-8 tokens and the rows per workgroup: wide loads (a uvec2 per lane), 4 rows per wave for
/// tall matrices and 2 for shorter ones, which then get more waves (results/qwen4/decode-kernels-20261012).
fn q8_decode_shaders(rows: usize) -> (&'static str, &'static str, usize) {
    if rows >= 4096 {
        ("qwen4_q8_mvw4_r4", "qwen4_q8_mvw4_r4_c4", 16)
    } else {
        ("qwen4_q8_mvw4_r2", "qwen4_q8_mvw4_r2_c4", 8)
    }
}

/// The f32 projection kernel: up to four tokens issue all their loads at once (qwen4_matf32).
fn matf32_shader(n: usize) -> &'static str {
    match n {
        1 => "qwen4_matf32",
        2..=4 => "qwen4_matf32_c4",
        _ => "qwen_matf32",
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
    pub fn new<'a>(
        gpu: Rc<Device>,
        g: &'a Gguf,
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
            c.hc <= 4 && c.d <= 2560 && c.experts == 512 && c.used <= 64,
            "unsupported mixer or router shape"
        );
        ensure!(
            c.d == 2560 && c.ff_exp == 640,
            "the small-batch expert kernels expect 2560-wide inputs and 640-wide experts"
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
        // Weights are repacked straight into staging memory and copied while the next ones are prepared; they are
        // all on the GPU once `finish` returns, at the end.
        let up = std::cell::RefCell::new(Uploader::new(&gpu)?);
        // A dense matrix from its blocks, given as consecutive runs (Q4_0 with f32 scales or Q8_0 with f16 scales).
        let dense = |q8: bool, runs: &[&'a [u8]]| -> Result<Dense> {
            let bytes = runs.iter().map(|r| r.len()).sum();
            Ok(if q8 {
                let p = Planes::of(Kind::Q8h, bytes);
                Dense::Q8(up.borrow_mut().planes(p, runs)?, (p.body_at / 4) as u32)
            } else {
                Dense::Q4(up.borrow_mut().planes(Planes::of(Kind::Q4, bytes), runs)?)
            })
        };
        let q4 = |t: &TensorInfo| -> Result<Dense> {
            match t.ty {
                GgmlType::Q8_0 => dense(true, &[g.data(t)]),
                _ => dense(false, &[q4_rows(g, t, 0, t.dims[1] as usize)?]),
            }
        };
        // Stacked experts and the word offset of their blocks.
        let experts = |fmt: u32, runs: &[&'a [u8]]| -> Result<(Buffer, usize)> {
            let p = Planes::of(expert_kind(fmt), runs.iter().map(|r| r.len()).sum());
            Ok((up.borrow_mut().planes(p, runs)?, p.body_at / 4))
        };
        let f32s = |v: &[f32]| up.borrow_mut().bytes(bytemuck::cast_slice(v));
        let mat32 = f32s;
        let vecb = |t: &TensorInfo| -> Result<Buffer> { f32s(&vector(g, t)?) };
        let mix = |prefix: &str, inject: bool| -> Result<Mix> {
            let tt = |name: &str| {
                let full = format!("{prefix}_{name}.weight");
                g.tensor(&full).with_context(|| format!("missing {full}"))
            };
            let fused = if inject && c.hc == 4 {
                // The bottleneck rows followed by the write rows.
                let (dn, inj) = (tt("down")?, tt("inject")?);
                match (dn.ty, inj.ty) {
                    (GgmlType::Q4_0, GgmlType::Q4_0) => Some(dense(false, &[g.data(dn), g.data(inj)])?),
                    (GgmlType::Q8_0, GgmlType::Q8_0) => Some(dense(true, &[g.data(dn), g.data(inj)])?),
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
            let mut gu = Vec::with_capacity(2 * ne + 2);
            for e in 0..ne {
                gu.push(expert_rows(g, gt, e * ff, ff)?);
                gu.push(expert_rows(g, ut, e * ff, ff)?);
            }
            gu.push(expert_rows(g, gs, 0, ff)?);
            gu.push(expert_rows(g, us, 0, ff)?);
            let (gate_up, gu_stride) = experts(gu_fmt, &gu)?;
            let (down, down_stride) = experts(down_fmt, &[expert_rows(g, dt, 0, ne * d)?, expert_rows(g, ds, 0, d)?])?;
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
                    w: up.borrow_mut().planes(Planes::of(Kind::Q6, rows * rb), &[&g.data(t)[row0 * rb..row1 * rb]])?,
                    logits: gpu.buffer(VERIFY_MAX * rows * 4, false, false, None)?,
                    scratch: gpu.buffer(VERIFY_MAX * chunks * TOP_K * 8, false, false, None)?,
                    top: gpu.buffer(VERIFY_MAX * TOP_K * 8, true, false, None)?,
                    chunks,
                    row0,
                    rows,
                    h: gpu.buffer(VERIFY_MAX * d * 4, true, false, None)?,
                })
            }
            None => None,
        };
        let alloc = |n: usize| gpu.buffer(n * max_batch * 4, false, false, None);
        let qd = c.heads * c.head_dim;
        let slots = c.used + 1;
        let fence = unsafe { gpu.dev.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let head_fence = unsafe { gpu.dev.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let event = || unsafe { gpu.dev.create_event(&vk::EventCreateInfo::default(), None) };
        let (start, done, hstart, hdone) = (event()?, event()?, event()?, event()?);
        let queries = unsafe {
            gpu.dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count((2 + MARKS) as u32 * layers.len() as u32),
                None,
            )?
        };
        let stage = Self {
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
            mp: alloc((w / SPLIT).max(1) * (c.hc_rank + 8))?,
            mcount: up.borrow_mut().bytes(&[0u8; 8])?,
            psum: alloc(16 * d.div_ceil(256))?,
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
            tiles: gpu.buffer((1 + 2 * max_batch * slots) * 4, false, false, None)?,
            list: alloc(slots)?,
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
            pending: None,
            head_graphs: HashMap::new(),
            head_n: 1,
            start,
            done,
            hstart,
            hdone,
            head_fence,
            ahead: None,
            head_ahead: None,
            head_pending: false,
            unretired: false,
            head_unretired: false,
            snaps: Vec::new(),
            ple_hist0: None,
            verify_max: 0,
            last_verify: None,
            last_n: 0,
            queries,
            profile: Vec::new(),
            phases: Vec::new(),
            c: c.clone(),
            gpu,
        };
        up.into_inner().finish()?;
        super::gate::register(&stage.gpu, stage.start);
        super::gate::register(&stage.gpu, stage.hstart);
        Ok(stage)
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
                matf32_shader(n),
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

    /// Score the hidden states of `cols` rows of `hd.h` with this stage's head rows: logits row by row (each exactly as
    /// a single row's), then the chunk top-k and the merge of every row at once into `hd.top`.
    fn score_head(&self, graph: &mut Graph, cmd: vk::CommandBuffer, hd: &Head, cols: usize, dump: bool) -> Result<()> {
        let gpu = &self.gpu;
        let d = self.c.d;
        let push = |ptr: [u64; 6], u: [u32; 16]| Push {
            ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], self.params.address, 0],
            u,
        };
        for row in 0..cols {
            let mut u = [0; 16];
            u[0] = hd.rows as u32;
            u[1] = d as u32;
            u[3] = (hd.rows * d / 16) as u32;
            u[9] = 1;
            u[15] = 1;
            gpu.dispatch_matrix(
                cmd,
                "headvec",
                push(
                    [
                        0,
                        hd.h.address + (row * d * 4) as u64,
                        hd.logits.address + (row * hd.rows * 4) as u64,
                        0,
                        0,
                        0,
                    ],
                    u,
                )
                .defer_barrier(row + 1 < cols),
                hd.rows.div_ceil(16) as u32,
                1,
                &hd.w,
            )?;
        }
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
        u[15] = cols as u32;
        gpu.dispatch(
            cmd,
            "topk",
            push([hd.logits.address, 0, hd.scratch.address, 0, 0, 0], u),
            hd.chunks as u32,
            cols as u32,
        );
        let mut u = [0; 16];
        u[0] = (hd.chunks * TOP_K) as u32;
        u[1] = TOP_K as u32;
        u[15] = cols as u32;
        gpu.dispatch(
            cmd,
            "merge",
            push([hd.scratch.address, 0, hd.top.address, 0, 0, 0], u),
            1,
            cols as u32,
        );
        Ok(())
    }

    /// The merged candidates of a head (ids offset to the vocabulary).
    fn read_top(hd: &Head, row: usize) -> Result<Vec<(f32, u32)>> {
        let bytes = hd.top.read((row + 1) * TOP_K * 8)?;
        let mut top: Vec<(f32, u32)> = bytes[row * TOP_K * 8..]
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
    /// stage holds no rows.
    pub fn head_slice(&mut self, h: &[f32], dump: Option<&mut Captures>) -> Result<Option<Vec<(f32, u32)>>> {
        let dumping = dump.is_some();
        if !self.head_submit(h, dumping)? {
            return Ok(None);
        }
        self.head_collect(dump)
    }

    /// Start scoring `h` with this stage's head rows; false when the stage has none to score (see `head_slice`).
    /// Stages on different GPUs can score at the same time; `head_collect` waits for the result. Releases scoring
    /// submitted ahead (`head_presubmit`) for as many rows.
    pub fn head_submit(&mut self, h: &[f32], dump: bool) -> Result<bool> {
        let d = self.c.d;
        let Some(hd) = self.head.as_ref() else {
            return Ok(false);
        };
        ensure!(self.pending.is_none(), "the stage is still working on a forward");
        // Several rows (a verification batch) are scored one after another.
        let rows = h.len() / d;
        ensure!(
            h.len() == rows * d && (1..=VERIFY_MAX).contains(&rows) && (rows == 1 || !dump),
            "hidden state length mismatch"
        );
        super::trace::mark("hsub0", self.first as u64);
        hd.h.write(bytemuck::cast_slice(h))?;
        self.head_n = rows;
        if let Some(ahead) = self.head_ahead.take() {
            ensure!(ahead == rows && !dump, "head scoring of {rows} rows does not match the {ahead} submitted ahead");
            self.release(self.hstart)?;
        } else {
            self.head_launch((dump, rows), false)?;
        }
        super::trace::mark("hsub", self.first as u64);
        Ok(true)
    }

    /// Submit the scoring of `rows` final hidden states ahead of them (`head_submit` releases it); false when the
    /// stage holds no head rows.
    pub fn head_presubmit(&mut self, rows: usize) -> Result<bool> {
        if self.head.is_none() {
            return Ok(false);
        }
        ensure!((1..=VERIFY_MAX).contains(&rows), "head scoring of {rows} rows");
        ensure!(self.pending.is_none() && !self.head_pending, "the stage is still working");
        self.head_launch((false, rows), true)?;
        self.head_ahead = Some(rows);
        Ok(true)
    }

    /// Record the head graph `key` (dump, rows) if needed and submit it, waiting for `hstart` when `ahead`.
    fn head_launch(&mut self, key: (bool, usize), ahead: bool) -> Result<()> {
        let hd = self.head.as_ref().context("no head rows")?;
        if !self.head_graphs.contains_key(&key) {
            let cmd = self.gpu.begin()?;
            let mut graph = Graph {
                gpu: self.gpu.clone(),
                cmds: vec![cmd],
                captures: Vec::new(),
                marks: Vec::new(),
                stamps: None,
            };
            self.gpu.wait_host_event(cmd, self.hstart);
            self.gpu.barrier(cmd);
            self.score_head(&mut graph, cmd, hd, key.1, key.0)?;
            self.gpu.host_barrier(cmd, true);
            unsafe { self.gpu.dev.cmd_set_event(cmd, self.hdone, DONE_STAGES) };
            self.gpu.finish(cmd)?;
            self.head_graphs.insert(key, graph);
        }
        self.retire_head()?;
        unsafe {
            self.gpu.dev.reset_fences(&[self.head_fence])?;
            self.gpu.dev.reset_event(self.hdone)?;
            if ahead {
                self.gpu.dev.reset_event(self.hstart)?;
            } else {
                self.gpu.dev.set_event(self.hstart)?;
            }
        }
        self.gpu.submit(self.head_graphs[&key].cmds[0], None, None, self.head_fence)?;
        if ahead {
            super::gate::arm(self.hstart);
        }
        self.head_pending = true;
        Ok(())
    }

    /// Wait for the head scoring in progress.
    fn head_wait(&mut self) -> Result<()> {
        ensure!(self.head_ahead.is_none(), "head scoring submitted ahead has no hidden states yet");
        ensure!(std::mem::take(&mut self.head_pending), "no head scoring submitted");
        self.head_unretired = !self.wait_done(self.hdone, self.head_fence)?;
        super::trace::mark("hfence", self.first as u64);
        Ok(())
    }

    /// Wait for `head_submit` and return its candidates (with vocabulary ids).
    pub fn head_collect(&mut self, dump: Option<&mut Captures>) -> Result<Option<Vec<(f32, u32)>>> {
        self.head_wait()?;
        if let Some(m) = dump {
            for (name, host, width) in &self.head_graphs[&(true, 1)].captures {
                m.insert(name.clone(), bytemuck::cast_slice(&host.read(width * 4)?).to_vec());
            }
        }
        Ok(Some(Self::read_top(self.head.as_ref().context("no head rows")?, 0)?))
    }

    /// Wait for `head_submit` and return the candidates of every row it scored.
    pub fn head_collect_rows(&mut self) -> Result<Vec<Vec<(f32, u32)>>> {
        self.head_wait()?;
        let hd = self.head.as_ref().context("no head rows")?;
        (0..self.head_n).map(|r| Self::read_top(hd, r)).collect()
    }

    /// Spin until a gated submission sets `event` at its end; past `SPIN`, wait for its fence instead. Whether the
    /// fence was waited for.
    fn wait_done(&self, event: vk::Event, fence: vk::Fence) -> Result<bool> {
        let start = std::time::Instant::now();
        while !unsafe { self.gpu.dev.get_event_status(event)? } {
            if start.elapsed() > SPIN {
                unsafe { self.gpu.dev.wait_for_fences(&[fence], true, 600_000_000_000)? };
                return Ok(true);
            }
            std::hint::spin_loop();
        }
        Ok(false)
    }

    /// Set the start event of work submitted ahead, once its input is written.
    fn release(&self, event: vk::Event) -> Result<()> {
        super::gate::disarm(event)?;
        unsafe { self.gpu.dev.set_event(event)? };
        Ok(())
    }

    /// Wait for the fence of a forward (head scoring) whose end was seen through its done event, before the fence or
    /// its command buffers are reused.
    fn retire(&mut self) -> Result<()> {
        if std::mem::take(&mut self.unretired) {
            unsafe { self.gpu.dev.wait_for_fences(&[self.fence], true, 600_000_000_000)? };
        }
        Ok(())
    }

    fn retire_head(&mut self) -> Result<()> {
        if std::mem::take(&mut self.head_unretired) {
            unsafe { self.gpu.dev.wait_for_fences(&[self.head_fence], true, 600_000_000_000)? };
        }
        Ok(())
    }

    /// Let work submitted ahead run without its input (garbage results, e.g. after an error elsewhere) and wait until
    /// the stage is idle.
    pub fn abandon(&mut self) -> Result<()> {
        let mut first = Ok(());
        if self.ahead.take().is_some() {
            first = first.and(self.release(self.start));
        }
        if self.head_ahead.take().is_some() {
            first = first.and(self.release(self.hstart));
        }
        unsafe {
            // Whatever the watchdog did, nothing may still wait on the GPU.
            self.gpu.dev.set_event(self.start)?;
            self.gpu.dev.set_event(self.hstart)?;
            self.gpu.dev.queue_wait_idle(self.gpu.queue)?;
        }
        self.pending = None;
        self.head_pending = false;
        self.unretired = false;
        self.head_unretired = false;
        unsafe { self.gpu.dev.reset_fences(&[self.fence, self.head_fence])? };
        first
    }

    pub fn device_mib(&self) -> f64 {
        self.gpu.device_bytes.get() as f64 / 1048576.
    }

    fn record(&self, n: usize, parts: u32, dump: bool, mode: Mode) -> Result<Graph> {
        let verify = mode == Mode::Verify;
        let gpu = &self.gpu;
        let c = &self.c;
        let (d, w, hc) = (c.d, c.width(), c.hc);
        let current = std::cell::Cell::new(gpu.begin()?);
        let mut graph = Graph {
            gpu: gpu.clone(),
            cmds: vec![current.get()],
            captures: Vec::new(),
            marks: Vec::new(),
            stamps: None,
        };
        let cmd = current.get();
        if std::env::var_os("TRASHCAN_Q4_PROFILE").is_some() {
            let pool = unsafe {
                gpu.dev.create_query_pool(
                    &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(STAMP_CAPACITY),
                    None,
                )?
            };
            unsafe { gpu.dev.cmd_reset_query_pool(cmd, pool, 0, STAMP_CAPACITY) };
            *gpu.stamps.borrow_mut() = Some(Stamps { pool, labels: Vec::new(), dispatches: true });
        }
        // Timestamps around every layer's submission (the last one includes the final mixer or the output copy).
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
        let gated = n <= GATED_MAX;
        if gated {
            gpu.wait_host_event(cmd, self.start);
        }
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
                    let (one, four, per) = q8_decode_shaders(rows);
                    (
                        b,
                        match n {
                            1 => (one, rows.div_ceil(per), 1),
                            2..=8 => (four, rows.div_ceil(per), n.div_ceil(4)),
                            _ => q8_gemm(rows, n, 1),
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
        // Q8_0 out projections of prefill batches split along K, their partial sums in hgate (free during the
        // sublayer), then summed in part order into y.
        let splits = if OUT_SPLITS * n * d <= w * self.max_batch { OUT_SPLITS } else { OUT_SPLITS / 2 };
        let out_proj = |wt: &Dense, x: &Buffer, k: usize| -> Result<()> {
            match wt {
                Dense::Q8(b, stride) if n > 8 && k % (64 * splits) == 0 && splits * d <= w => {
                    let mut u = [0; 16];
                    u[0] = d as u32;
                    u[1] = k as u32;
                    u[2] = splits as u32;
                    u[3] = *stride;
                    let (shader, gx, gy) = q8_gemm(d, n, splits);
                    gpu.dispatch_matrix(
                        current.get(),
                        shader,
                        push([0, x.address, self.hgate.address, 0, 0, 0], u),
                        gx as u32,
                        gy as u32,
                        b,
                    )?;
                    let mut u = [0; 16];
                    u[0] = d as u32;
                    u[1] = splits as u32;
                    gpu.dispatch(
                        current.get(),
                        "qwen4_sum_parts",
                        push([self.hgate.address, 0, self.y.address, 0, 0, 0], u),
                        d.div_ceil(256) as u32,
                        n as u32,
                    );
                    Ok(())
                }
                _ => matrix(wt, x, &self.y, k, d, false),
            }
        };
        // Small f32 projections: one wave per row and eight tokens.
        let matf32 = |wt: &Buffer, x: &Buffer, y: &Buffer, k: usize, rows: usize, defer: bool| -> Result<()> {
            let mut u = [0; 16];
            u[0] = rows as u32;
            u[1] = k as u32;
            gpu.dispatch_matrix(
                current.get(),
                matf32_shader(n),
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
        let q8_mixers = self
            .layers
            .iter()
            .all(|l| matches!((&l.attn_mix.up, &l.ffn_mix.up), (Dense::Q8(..), Dense::Q8(..))));
        let fused = n <= if q8_mixers { FUSED_MAX } else { FUSED_MAX_Q4 }
            && self
                .layers
                .iter()
                .all(|l| l.attn_mix.fused.is_some() && l.ffn_mix.fused.is_some());
        // Norm all streams into xn, applying the pending combine first (one workgroup per stream and token).
        let hc_norm = |pending: bool, norm: &Buffer| {
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
                push([self.x.address, self.y.address, self.xn.address, norm.address, wloc.get().0, 0], u),
                hc as u32,
                n as u32,
            );
        };
        // Apply the pending combine without a norm: x_s += w_s * y.
        let combine = || {
            let mut u = [0; 16];
            u[0] = d as u32;
            u[2] = wloc.get().1;
            u[4] = hc as u32;
            u[10] = wloc.get().2;
            gpu.dispatch(
                current.get(),
                "qwen4_combine",
                push([self.x.address, self.y.address, 0, 0, wloc.get().0, 0], u),
                d.div_ceil(256) as u32,
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
                if d == 2 * SPLIT && w % SPLIT == 0 && n <= 8 {
                    // Split over the residual (qwen4_mix_down has 21 workgroups for one token): one token, or four
                    // per workgroup reading the weights once (measured up to 8 tokens; prefill chunks keep
                    // qwen4_mix_down).
                    let groups = if n == 1 { 1 } else { n.div_ceil(4) };
                    gpu.dispatch_matrix(
                        current.get(),
                        match (q8, n) {
                            (true, 1) => "qwen4_mix_down_sk_q8",
                            (false, 1) => "qwen4_mix_down_sk",
                            (true, _) => "qwen4_mix_down_sk4_q8",
                            (false, _) => "qwen4_mix_down_sk4",
                        },
                        push(
                            [
                                self.x.address,
                                self.y.address,
                                self.mp.address,
                                mo_prev.address,
                                m.norm.address,
                                mo_next.address,
                            ],
                            u,
                        )
                        .with_extra(self.mcount.address),
                        (c.hc_rank + 4).div_ceil(16) as u32,
                        (groups * (w / SPLIT)) as u32,
                        wb,
                    )?;
                } else {
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
                }
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
            hc_norm(pending, &m.norm);
            capture(graph, format!("hc_norm{tag}"), &self.xn, w)?;
            // Q8_0 above FUSED_MAX tokens: the bottleneck and write rows (the fused matrix; the final mixer has no
            // write rows) as a split-K GEMM, its partial sums (in hgate) reduced with the activations, then the up
            // projection with the read gates folded into the sublayer input. Fewer tokens (the final mixer) take the
            // matrix-vector kernels below.
            let q8 = match (&m.fused, &m.down, &m.up) {
                _ if n <= FUSED_MAX => None,
                (Some(Dense::Q8(b, st)), _, Dense::Q8(ub, ust)) => Some((b, *st, c.hc_rank + hc, ub, *ust)),
                (None, Dense::Q8(b, st), Dense::Q8(ub, ust)) if m.inject.is_none() && hc == 4 => {
                    Some((b, *st, c.hc_rank, ub, *ust))
                }
                _ => None,
            };
            if let Some((b, stride, rows, ub, ustride)) = q8 {
                let mut u = [0; 16];
                u[0] = rows as u32;
                u[1] = w as u32;
                u[2] = MIX_SPLITS as u32;
                u[3] = stride;
                // Up to LANE_GEMM_MAX tokens the lane-per-row kernel, 8 tokens per workgroup and the row blocks first
                // (16 tokens: 136 against 265 us per mixer).
                let (shader, gx, gy) = if n <= LANE_GEMM_MAX {
                    ("qwen4_q8_gemm_s8r1po", rows.div_ceil(64) * MIX_SPLITS, n.div_ceil(8))
                } else {
                    ("qwen4_q8_gemm", rows.div_ceil(64) * MIX_SPLITS, n.div_ceil(64))
                };
                gpu.dispatch_matrix(
                    current.get(),
                    shader,
                    push([0, self.xn.address, self.hgate.address, 0, 0, 0], u),
                    gx as u32,
                    gy as u32,
                    b,
                )?;
                let mut u = [0; 16];
                u[0] = rows as u32;
                u[1] = MIX_SPLITS as u32;
                u[2] = c.hc_rank as u32;
                gpu.dispatch(
                    current.get(),
                    "qwen4_mix_act",
                    push([self.hgate.address, 0, self.lo.address, self.w.address, 0, 0], u),
                    rows.div_ceil(64) as u32,
                    n as u32,
                );
                let mut u = [0; 16];
                u[0] = w as u32;
                u[1] = c.hc_rank as u32;
                u[3] = ustride;
                // Up to 16 tokens the lane-per-row kernel with the four streams' rows of an element in one lane (151
                // against 217 us per mixer).
                let (shader, gx, gy) = if n <= 16 {
                    ("qwen4_q8_gemm_s4r4om", w / 256, n.div_ceil(4))
                } else {
                    ("qwen4_q8_gemm_mix", w / 64, n.div_ceil(64))
                };
                gpu.dispatch_matrix(
                    current.get(),
                    shader,
                    push([0, self.lo.address, self.u.address, self.xn.address, 0, 0], u),
                    gx as u32,
                    gy as u32,
                    ub,
                )?;
                capture(graph, format!("hc_mixed{tag}"), &self.u, d)?;
                if m.inject.is_some() {
                    wloc.set((self.w.address, hc as u32, 0));
                }
                return Ok(());
            }
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
            gpu.stamp(current.get(), false, || format!("layer {il}"));
            if let (Some(p), true) = (&self.ple, c.ngram.as_ref().is_some_and(|p| p.layer == il)) {
                if pending {
                    combine();
                }
                pending = false;
                capture(&mut graph, "ple_embd".into(), &self.pe, d)?;
                if n <= 8 {
                    matrix(&p.value, &self.pe, &self.pv, d, d, true)?;
                    matrix(&p.key, &self.pe, &self.pk, d, w, false)?;
                } else {
                    matrix(&p.key, &self.pe, &self.pk, d, w, true)?;
                    matrix(&p.value, &self.pe, &self.pv, d, d, false)?;
                }
                let mut u = [0; 16];
                u[0] = d as u32;
                u[4] = hc as u32;
                u[12] = eps;
                for shader in ["qwen4_ple_sums", "qwen4_ple_gate"] {
                    gpu.dispatch(
                        current.get(),
                        shader,
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
                        )
                        .with_extra(self.psum.address),
                        d.div_ceil(256) as u32,
                        n as u32,
                    );
                }
                capture(&mut graph, format!("ple_gated_value-{il}"), &self.gv, w)?;
                if let (true, Some(h0)) = (verify, &self.ple_hist0) {
                    gpu.copy(current.get(), &p.hist, 0, h0, 0, h0.size);
                }
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
                    let (snap, snap_conv) = match (verify, self.snaps.get(il - self.first)) {
                        (true, Some(Some((s, w)))) => (s.address, w.address),
                        _ => (0, 0),
                    };
                    // Decode puts the small projections first: the last dispatch of a group starts its waves last.
                    if n <= 8 {
                        matf32(ab, &self.u, &self.ab, d, 2 * c.v_heads, true)?;
                        matrix(z, &self.u, &self.z, d, vd, true)?;
                        matrix(qkv, &self.u, &self.big, d, cd, false)?;
                    } else {
                        matrix(qkv, &self.u, &self.big, d, cd, true)?;
                        matrix(z, &self.u, &self.z, d, vd, true)?;
                        matf32(ab, &self.u, &self.ab, d, 2 * c.v_heads, false)?;
                    }
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
                                snap_conv,
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
                    if n <= 8 {
                        // Two state rows per wave and vec2 lanes for decode: 52 -> 45 us per layer.
                        gpu.dispatch(
                            current.get(),
                            "qwen4_gdn",
                            push(
                                [
                                    self.conv_out.address,
                                    self.ab.address,
                                    state.address,
                                    self.o.address,
                                    dt_a.address,
                                    snap,
                                ],
                                u,
                            ),
                            (c.v_heads * c.state / 8) as u32,
                            1,
                        );
                    } else {
                        // Prefill: q and k normalized and the gates prepared once per token, then a lane per state
                        // row walks the chunk.
                        gpu.dispatch(
                            current.get(),
                            "qwen4_gdn_prep",
                            push([self.conv_out.address, 0, self.ab.address, 0, dt_a.address, 0], u),
                            (2 * c.k_heads + 1) as u32,
                            n as u32,
                        );
                        gpu.dispatch(
                            current.get(),
                            "qwen4_gdn_rows",
                            push([0, self.conv_out.address, state.address, self.o.address, self.ab.address, 0], u),
                            (c.v_heads * c.state / 64) as u32,
                            1,
                        );
                    }
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
                    out_proj(out, &self.o, vd)?;
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
                    if n <= 8 {
                        matrix(k, &self.u, &self.kb, d, kvd, true)?;
                        matrix(v, &self.u, &self.vb, d, kvd, true)?;
                        matrix(q, &self.u, &self.big, d, 2 * qd, false)?;
                    } else {
                        matrix(q, &self.u, &self.big, d, 2 * qd, true)?;
                        matrix(k, &self.u, &self.kb, d, kvd, true)?;
                        matrix(v, &self.u, &self.vb, d, kvd, false)?;
                    }
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
                    out_proj(out, &self.o, qd)?;
                    capture(&mut graph, format!("attn_output-{il}"), &self.y, d)?;
                }
            }
            mark(&mut graph, li, if c.recurrent(il) { "deltanet" } else { "attention" });
            mixer(&mut graph, &l.ffn_mix, true, &format!("-{il}#2"))?;
            mark(&mut graph, li, "mixer");
            capture(&mut graph, format!("hc_combine-{il}"), &self.x, w)?;
            // MoE: router (with the shared gate as its last row), top-k, experts, weighted sum.
            let (ne, slots, ff) = (c.experts, c.used + 1, c.ff_exp);
            // The batch as a push constant (u[15]), so these kernels start without loading it.
            let nb = n as u32;
            // Logits: one wave per row up to 4 tokens (the same bits as qwen4_matf32); prefill batches multiply in
            // 64 x 64 tiles with the K range split in `parts`, which top-k adds (in `big`, free until the experts).
            let parts = (16 / n.div_ceil(64)).max(1);
            let gemm = n > 4 && d % (32 * parts) == 0 && parts * n * (ne + 1) <= self.big.size / 4;
            let logits = if gemm { &self.big } else { &self.logits };
            let mut u = [0; 16];
            u[0] = (ne + 1) as u32;
            u[1] = d as u32;
            u[15] = nb;
            if n <= 4 && d % (64 * 20) == 0 {
                gpu.dispatch_matrix(
                    current.get(),
                    if n == 1 { "qwen4_router" } else { "qwen4_router_c4" },
                    push([0, self.u.address, self.logits.address, 0, 0, 0], u),
                    (ne + 1).div_ceil(4) as u32,
                    1,
                    &l.router,
                )?;
            } else if gemm {
                u[2] = parts as u32;
                gpu.dispatch_matrix(
                    current.get(),
                    "qwen4_router_gemm",
                    push([0, self.u.address, self.big.address, 0, 0, 0], u),
                    ((ne + 1).div_ceil(64) * parts) as u32,
                    n.div_ceil(64) as u32,
                    &l.router,
                )?;
            } else {
                matf32(&l.router, &self.u, &self.logits, d, ne + 1, false)?;
            }
            if !gemm {
                capture(&mut graph, format!("ffn_moe_logits_all-{il}"), logits, ne + 1)?;
            }
            let mut u = [0; 16];
            u[0] = ne as u32;
            u[1] = if gemm { parts as u32 } else { 1 };
            u[2] = c.used as u32;
            u[15] = nb;
            gpu.dispatch(
                current.get(),
                "qwen4_topk",
                push([logits.address, 0, self.ids.address, self.wts.address, 0, 0], u),
                1,
                n as u32,
            );
            if gemm {
                capture(&mut graph, format!("ffn_moe_logits_all-{il}"), logits, ne + 1)?;
            }
            capture(&mut graph, format!("moe_weights-{il}"), &self.wts, slots)?;
            mark(&mut graph, li, "router");
            let (gu_stride, down_stride) = (l.gu_stride, l.down_stride);
            // Prefill batches group the slots by expert, so each expert's rows are decoded once per tile of slots;
            // small batches use one workgroup per (token, slot), each reading its expert's rows.
            let grouped = n >= GROUPED_MIN && n * slots < 1 << 16 && l.gu_fmt <= 2 && d % 256 == 0 && ff % 128 == 0;
            let tiles = max_tiles(n * slots, ne + 1) as u32;
            if grouped {
                let mut u = [0; 16];
                u[0] = (ne + 1) as u32;
                u[1] = TILE as u32;
                u[2] = slots as u32;
                gpu.dispatch(
                    current.get(),
                    "qwen4_moe_tiles",
                    push([self.ids.address, 0, self.tiles.address, self.list.address, 0, 0], u),
                    1,
                    1,
                );
                let mut u = [0; 16];
                u[0] = ff as u32;
                u[1] = d as u32;
                u[3] = gu_stride;
                gpu.dispatch_matrix(
                    current.get(),
                    expert_shader("qwen4_moe_gu_tiles", l.gu_fmt),
                    push([0, self.u.address, self.act.address, self.tiles.address, self.list.address, 0], u),
                    ff.div_ceil(128) as u32,
                    tiles,
                    &l.gate_up,
                )?;
            } else {
                let mut u = [0; 16];
                u[0] = ff as u32;
                u[1] = d as u32;
                u[2] = slots as u32;
                u[3] = gu_stride;
                u[15] = nb;
                gpu.dispatch_matrix(
                    current.get(),
                    expert_shader("qwen4_moe_gu", l.gu_fmt),
                    push([0, self.u.address, self.act.address, self.ids.address, 0, 0], u),
                    ff.div_ceil(16) as u32,
                    (n * slots) as u32,
                    &l.gate_up,
                )?;
            }
            if grouped && l.down_fmt <= 1 {
                // Each slot's weighted output for a range of rows goes to `big` (free during the experts), then a
                // token's slots are summed in order.
                let rows = (c.conv_dim().max(2 * c.heads * c.head_dim) / slots) / 128 * 128;
                for first in (0..d).step_by(rows) {
                    let m = rows.min(d - first);
                    let mut u = [0; 16];
                    u[0] = d as u32;
                    u[1] = ff as u32;
                    u[2] = m as u32;
                    u[3] = down_stride;
                    u[10] = first as u32;
                    gpu.dispatch_matrix(
                        current.get(),
                        expert_shader("qwen4_moe_down_tiles", l.down_fmt),
                        push(
                            [
                                0,
                                self.act.address,
                                self.big.address,
                                self.tiles.address,
                                self.list.address,
                                self.wts.address,
                            ],
                            u,
                        ),
                        m.div_ceil(128) as u32,
                        tiles,
                        &l.down,
                    )?;
                    let mut u = [0; 16];
                    u[0] = d as u32;
                    u[1] = slots as u32;
                    u[2] = m as u32;
                    u[10] = first as u32;
                    gpu.dispatch(
                        current.get(),
                        "qwen4_moe_sum",
                        push([self.big.address, 0, self.y.address, 0, 0, 0], u),
                        m.div_ceil(256) as u32,
                        n as u32,
                    );
                }
            } else {
                let mut u = [0; 16];
                u[0] = d as u32;
                u[1] = ff as u32;
                u[2] = slots as u32;
                u[3] = down_stride;
                u[15] = nb;
                gpu.dispatch_matrix(
                    current.get(),
                    // Wide loads and the (slot, block) pairs over the lanes (ff = 640 is 20 blocks); Q2_0 per slot.
                    match l.down_fmt {
                        0 => "qwen4_moe_down_w",
                        1 => "qwen4_moe_down_w_iq4nl",
                        _ => expert_shader("qwen4_moe_down", l.down_fmt),
                    },
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
            }
            capture(&mut graph, format!("ffn_out-{il}"), &self.y, d)?;
            mark(&mut graph, li, "experts");
            pending = true;
            if l.il + 1 < self.last {
                let li = l.il - self.first;
                stamp(current.get(), 2 * li + 1, true);
                // Prefill submits a command buffer per layer (each GPU job well inside amdgpu's timeout); a decode
                // step records the stage in one, since every submission costs ~11 us between layers on the D700.
                if n > 8 {
                    gpu.finish(current.get())?;
                    let next = gpu.begin()?;
                    stamp(next, 2 * li + 2, false);
                    gpu.barrier(next);
                    graph.cmds.push(next);
                    current.set(next);
                } else {
                    stamp(current.get(), 2 * li + 2, false);
                }
            }
        }
        let last = self.last - 1;
        if let Some((hd, m)) = self.head.as_ref().and_then(|hd| Some((hd, hd.mix.as_ref()?))) {
            mixer(&mut graph, m, pending, "")?;
            capture(&mut graph, format!("l_last-{last}"), &self.x, w)?;
            capture(&mut graph, "result_norm".into(), &self.u, d)?;
            if verify {
                gpu.copy(current.get(), &self.u, 0, &hd.h, 0, n * d * 4);
            } else {
                gpu.copy(current.get(), &self.u, (n - 1) * d * 4, &hd.h, 0, d * 4);
            }
            if verify {
                gpu.copy(current.get(), &self.x, 0, &self.output, 0, n * w * 4);
            }
        } else {
            if pending {
                combine();
            }
            capture(&mut graph, format!("l_last-{last}"), &self.x, w)?;
            gpu.copy(current.get(), &self.x, 0, &self.output, 0, n * w * 4);
        }
        stamp(current.get(), 2 * self.layers.len() - 1, true);
        if gated {
            gpu.host_barrier(current.get(), true);
            unsafe { gpu.dev.cmd_set_event(current.get(), self.done, DONE_STAGES) };
        }
        gpu.finish(current.get())?;
        graph.stamps = gpu.stamps.borrow_mut().take().map(|s| (s.pool, s.labels));
        Ok(graph)
    }

    /// Dispatch times of the graph that ran `n` tokens at `pos` last (`TRASHCAN_Q4_PROFILE=1`): per dispatch its
    /// label, GPU milliseconds from its start to the end of all work before it, and the gap before its start.
    pub fn dispatch_profile(&self, pos: usize, n: usize) -> Result<Vec<(String, f64, f64)>> {
        let Some(graph) = self.graphs.get(&(n, partitions(pos, n), false, Mode::Plain)) else { return Ok(Vec::new()) };
        let Some((pool, labels)) = &graph.stamps else { return Ok(Vec::new()) };
        let mut t = vec![0u64; labels.len()];
        unsafe {
            self.gpu.dev.get_query_pool_results(
                *pool,
                0,
                &mut t,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?
        };
        let ms = self.gpu.period as f64 / 1e6;
        let mut out = Vec::new();
        let (mut i, mut prev) = (0, t.first().copied().unwrap_or(0));
        while i < labels.len() {
            if labels[i].starts_with("layer ") || i + 1 >= labels.len() {
                out.push((labels[i].clone(), 0., t[i].wrapping_sub(prev) as f64 * ms));
                prev = t[i];
                i += 1;
                continue;
            }
            out.push((
                labels[i].clone(),
                t[i + 1].wrapping_sub(t[i]) as f64 * ms,
                t[i].wrapping_sub(prev) as f64 * ms,
            ));
            prev = t[i + 1];
            i += 2;
        }
        Ok(out)
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
        self.submit(x, ple, pos, n, dump.is_some())?;
        self.collect(dump)
    }

    /// `forward` for a verification batch (speculative decoding): the DeltaNet and n-gram states after every token
    /// are kept for `rollback`, and the stage that ends the model returns the final hidden state of every token and
    /// keeps their residual rows (`residual`). Needs `enable_verify`.
    pub fn verify(&mut self, x: &[f32], ple: Option<&[f32]>, pos: usize, n: usize) -> Result<Output> {
        ensure!(n <= self.verify_max, "verification batch of {n} exceeds the enabled {}", self.verify_max);
        self.submit_mode(x, ple, pos, n, false, Mode::Verify)?;
        self.collect(None)
    }

    /// The last `m` residual rows (`m x hc x d`, fewer when the latest batch was shorter) of the latest batch on the
    /// stage that ends the model: the MTP block's input at the end of a prompt (copied out on its own).
    pub fn tail(&mut self, m: usize) -> Result<Vec<f32>> {
        ensure!(self.last == self.c.layers, "only the stage that ends the model keeps the final residual");
        ensure!(self.pending.is_none(), "the stage is still working on a forward");
        let m = m.min(self.last_n);
        ensure!(m >= 1, "no batch has run");
        self.retire()?;
        let w4 = self.c.width() * 4;
        let cmd = self.gpu.begin()?;
        self.gpu.barrier(cmd);
        self.gpu.copy(cmd, &self.x, (self.last_n - m) * w4, &self.output, 0, m * w4);
        self.gpu.finish(cmd)?;
        unsafe {
            self.gpu.dev.reset_fences(&[self.fence])?;
        }
        self.gpu.submit(cmd, None, None, self.fence)?;
        unsafe {
            self.gpu.dev.wait_for_fences(&[self.fence], true, 600_000_000_000)?;
            self.gpu.dev.free_command_buffers(self.gpu.pool, &[cmd]);
        }
        Ok(bytemuck::cast_slice(&self.output.read(m * w4)?).to_vec())
    }

    /// Keep tokens `0..keep` of the latest verification batch: restore the recurrent states after token `keep - 1`.
    /// The K/V and indexer caches are addressed by position and need nothing.
    pub fn rollback(&mut self, keep: usize) -> Result<()> {
        let n = self.last_verify.context("rollback without a verification batch")?;
        ensure!(keep >= 1 && keep <= n, "rollback to {keep} of a {n}-token batch");
        if keep == n {
            return Ok(());
        }
        ensure!(self.pending.is_none(), "the stage is still working on a forward");
        self.retire()?;
        let c = &self.c;
        let (state, conv) = (c.v_heads * c.state * c.state * 4, (c.conv - 1) * c.conv_dim() * 4);
        let cmd = self.gpu.begin()?;
        self.gpu.barrier(cmd);
        for (l, snap) in self.layers.iter().zip(&self.snaps) {
            if let (Mixer::Delta { conv: cw, state: st, .. }, Some((s, w))) = (&l.mixer, snap) {
                self.gpu.copy(cmd, s, (keep - 1) * state, st, 0, state);
                self.gpu.copy(cmd, w, (keep - 1) * conv, cw, 0, conv);
            }
        }
        // The n-gram convolution keeps its last 9 inputs: those before the batch, then the batch's first `keep`.
        if let (Some(p), Some(h0)) = (&self.ple, &self.ple_hist0) {
            let row = c.width() * 4;
            let rows = h0.size / row;
            self.gpu.copy(cmd, h0, keep * row, &p.hist, 0, (rows - keep) * row);
            self.gpu.copy(cmd, &self.pn, 0, &p.hist, (rows - keep) * row, keep * row);
        }
        self.gpu.finish(cmd)?;
        unsafe {
            self.gpu.dev.reset_fences(&[self.fence])?;
        }
        self.gpu.submit(cmd, None, None, self.fence)?;
        unsafe {
            self.gpu.dev.wait_for_fences(&[self.fence], true, 600_000_000_000)?;
            self.gpu.dev.free_command_buffers(self.gpu.pool, &[cmd]);
        }
        self.last_verify = Some(keep);
        Ok(())
    }

    /// Allocate what verification batches of up to `max` tokens need (`max - 1` state copies per DeltaNet layer).
    pub fn enable_verify(&mut self, max: usize) -> Result<()> {
        ensure!(max <= VERIFY_MAX, "verification batches are limited to {VERIFY_MAX} tokens");
        ensure!(max <= self.max_batch, "verification batches exceed the stage's batch size");
        if max <= self.verify_max {
            return Ok(());
        }
        ensure!(self.pending.is_none(), "the stage is still working on a forward");
        // Verification graphs are recorded again below.
        self.retire()?;
        let c = &self.c;
        let (state, conv) = (c.v_heads * c.state * c.state * 4, (c.conv - 1) * c.conv_dim() * 4);
        let snaps = max.saturating_sub(1).max(1);
        self.snaps = self
            .layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Delta { .. } => Ok(Some((
                    self.gpu.buffer(snaps * state, false, false, None)?,
                    self.gpu.buffer(snaps * conv, false, false, None)?,
                ))),
                Mixer::Attention { .. } => Ok(None),
            })
            .collect::<Result<_>>()?;
        if let (Some(_), Some(ng)) = (&self.ple, &c.ngram) {
            self.ple_hist0 = Some(self.gpu.buffer((ng.conv - 1) * ng.ngram * c.width() * 4, false, false, None)?);
        }
        self.verify_max = max;
        self.graphs.retain(|k, _| k.3 != Mode::Verify);
        Ok(())
    }

    /// The residual rows (`n x hc x d`) of the latest verification batch on the stage that ends the model: the MTP
    /// block's input.
    pub fn residual(&self, n: usize) -> Result<Vec<f32>> {
        ensure!(self.last_verify.is_some_and(|v| n <= v), "no residual rows of a verification batch");
        Ok(bytemuck::cast_slice(&self.output.read(n * self.c.width() * 4)?).to_vec())
    }

    /// Start `forward` without waiting for it; `collect` waits and returns the output. Releases a forward submitted
    /// ahead (`presubmit`) for the same position and tokens.
    pub fn submit(&mut self, x: &[f32], ple: Option<&[f32]>, pos: usize, n: usize, dump: bool) -> Result<()> {
        self.submit_mode(x, ple, pos, n, dump, Mode::Plain)
    }

    pub(crate) fn submit_mode(&mut self, x: &[f32], ple: Option<&[f32]>, pos: usize, n: usize, dump: bool, mode: Mode) -> Result<()> {
        let (d, w) = (self.c.d, self.c.width());
        super::trace::mark("sub0", self.first as u64);
        ensure!(x.len() == n * w, "residual length mismatch");
        let ple = match (&self.ple, ple) {
            (None, _) => None,
            (Some(_), Some(p)) if p.len() == n * d => Some(p),
            (Some(_), Some(_)) => anyhow::bail!("n-gram rows length mismatch"),
            (Some(_), None) => anyhow::bail!("this stage needs the n-gram rows"),
        };
        if let Some(ahead) = self.ahead.take() {
            ensure!(
                ahead == (pos, n, mode) && !dump,
                "a forward of {n} tokens at {pos} does not match the one submitted ahead ({ahead:?})"
            );
            self.write_input(x, ple, pos, n)?;
            self.release(self.start)?;
        } else {
            self.check(pos, n)?;
            self.write_input(x, ple, pos, n)?;
            self.launch((n, partitions(pos, n), dump, mode), false)?;
        }
        super::trace::mark("sub1", self.first as u64);
        Ok(())
    }

    /// Submit the graph of a forward of `n` tokens at `pos` (up to `GATED_MAX`) ahead of its input: it waits on the GPU
    /// until `submit` with the same position and tokens writes the input and releases it, so the submission's own
    /// cost overlaps earlier work. Submit ahead only when the input is on its way (see `gate`).
    pub fn presubmit(&mut self, pos: usize, n: usize, mode: Mode) -> Result<()> {
        ensure!(n <= GATED_MAX, "only forwards of up to {GATED_MAX} tokens are submitted ahead");
        ensure!(mode == Mode::Plain || n <= self.verify_max, "verification batch of {n} exceeds the enabled {}", self.verify_max);
        super::trace::mark("pre0", self.first as u64);
        self.check(pos, n)?;
        self.launch((n, partitions(pos, n), false, mode), true)?;
        self.ahead = Some((pos, n, mode));
        super::trace::mark("pre1", self.first as u64);
        Ok(())
    }

    /// Whether a forward was submitted ahead of its input.
    pub fn presubmitted(&self) -> bool {
        self.ahead.is_some()
    }

    fn check(&self, pos: usize, n: usize) -> Result<()> {
        ensure!(self.pending.is_none(), "the stage is still working on a forward");
        ensure!(self.head_ahead.is_none(), "head scoring submitted ahead still waits for its hidden states");
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(pos + n <= self.context, "context exhausted");
        Ok(())
    }

    fn write_input(&self, x: &[f32], ple: Option<&[f32]>, pos: usize, n: usize) -> Result<()> {
        self.input.write(bytemuck::cast_slice(x))?;
        if let Some(p) = ple {
            self.ple_in.write(bytemuck::cast_slice(p))?;
        }
        self.params_host.write(bytemuck::cast_slice(&[pos as u32, n as u32, 0, 0]))
    }

    /// Record the graph `key` if needed and submit it. A graph of up to `GATED_MAX` tokens waits for `start`, which is
    /// set now unless the submission is `ahead` of its input.
    fn launch(&mut self, key: (usize, u32, bool, Mode), ahead: bool) -> Result<()> {
        // The previous forward's command buffers may be freed below.
        self.retire()?;
        if !self.graphs.contains_key(&key) {
            if self.graphs.len() >= GRAPH_CACHE_CAPACITY {
                let victim = *self
                    .graphs
                    .keys()
                    .find(|k| k.0 != 1)
                    .unwrap_or(self.graphs.keys().next().unwrap());
                self.graphs.remove(&victim);
            }
            let graph = self.record(key.0, key.1, key.2, key.3)?;
            self.graphs.insert(key, graph);
        }
        let gated = key.0 <= GATED_MAX;
        ensure!(gated || !ahead, "only forwards of up to {GATED_MAX} tokens are submitted ahead");
        unsafe {
            self.gpu.dev.reset_fences(&[self.fence])?;
            if gated {
                self.gpu.dev.reset_event(self.done)?;
                if ahead {
                    self.gpu.dev.reset_event(self.start)?;
                } else {
                    self.gpu.dev.set_event(self.start)?;
                }
            }
        }
        let graph = &self.graphs[&key];
        for (i, &cmd) in graph.cmds.iter().enumerate() {
            let last = i + 1 == graph.cmds.len();
            self.gpu
                .submit(cmd, None, None, if last { self.fence } else { vk::Fence::null() })?;
        }
        if ahead {
            super::gate::arm(self.start);
        }
        self.pending = Some((key, key.0));
        Ok(())
    }

    /// Wait for the forward started by `submit`; its output (with the captures when it was submitted with `dump`).
    pub fn collect(&mut self, dump: Option<&mut Captures>) -> Result<Output> {
        let (d, w) = (self.c.d, self.c.width());
        ensure!(self.ahead.is_none(), "a forward submitted ahead has no input yet");
        let (key, n) = self.pending.take().context("no forward submitted")?;
        if key.0 <= GATED_MAX {
            self.unretired = !self.wait_done(self.done, self.fence)?;
        } else {
            unsafe {
                self.gpu.dev.wait_for_fences(&[self.fence], true, 600_000_000_000)?;
            }
        }
        let graph = &self.graphs[&key];
        super::trace::mark("fence", self.first as u64);
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
        if let Some((tick, host)) = super::trace::on().then(|| self.gpu.calibrate()).flatten() {
            // The GPU's start and end of the stage in host time.
            let at = |t: u64| (host as f64 + (t as i64 - tick as i64) as f64 * period * 1e6) as u64;
            super::trace::mark_at("gpu0", self.first as u64, at(times[0]));
            super::trace::mark_at("gpu1", self.first as u64, at(times[2 * layers - 1]));
        }
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
        let verify = key.3 == Mode::Verify;
        self.last_verify = verify.then_some(n);
        self.last_n = n;
        let rows = if verify { n } else { 1 };
        let out = match self.head.as_ref().filter(|hd| hd.mix.is_some()) {
            Some(hd) => Output::Final(read_f32(&hd.h, rows * d)?),
            None => Output::Hidden(read_f32(&self.output, n * w)?),
        };
        super::trace::mark("col", self.first as u64);
        Ok(out)
    }
}

/// `count` floats of a host-visible buffer, copied once.
fn read_f32(b: &Buffer, count: usize) -> Result<Vec<f32>> {
    ensure!(!b.ptr.is_null() && count * 4 <= b.size, "invalid host read");
    let mut v = vec![0f32; count];
    unsafe { std::ptr::copy_nonoverlapping(b.ptr, v.as_mut_ptr().cast::<u8>(), count * 4) };
    Ok(v)
}

impl Drop for GpuStage {
    fn drop(&mut self) {
        unsafe {
            // Work submitted ahead runs on whatever input is there, so that the device can go idle.
            let _ = self.gpu.dev.set_event(self.start);
            let _ = self.gpu.dev.set_event(self.hstart);
            let _ = self.gpu.dev.device_wait_idle();
        }
        super::gate::unregister(&self.gpu);
        self.graphs.clear();
        self.head_graphs.clear();
        unsafe {
            self.gpu.dev.destroy_fence(self.fence, None);
            self.gpu.dev.destroy_fence(self.head_fence, None);
            for e in [self.start, self.done, self.hstart, self.hdone] {
                self.gpu.dev.destroy_event(e, None);
            }
            self.gpu.dev.destroy_query_pool(self.queries, None);
        }
    }
}
