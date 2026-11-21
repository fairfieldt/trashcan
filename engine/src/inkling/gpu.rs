//! One rank of a tensor-parallel Inkling group on one GPU: its query heads with their KV heads, its slice of every
//! expert's intermediate rows and its rows of the vocabulary. The host writes each segment's normed input rows into
//! `hin` and reads the segment's partial output (attention or MoE, before the all-reduce) from `part`.
//!
//! A forward is a sequence of segments: per layer application an attention segment (`rec_attention`) and a MoE
//! segment (`rec_moe`), then the head (`rec_head`). Decode-sized forwards are one event-gated submission; prefill
//! batches submit each segment, and a batch split in micro-batches (`MICRO`) keeps a segment of each queued, so the
//! host works on one micro-batch's output while the GPU runs the other's segment. Each micro-batch has its own
//! `hin`, `part` and parameters; the device scratch is shared (segments run one at a time).
use super::{dequant_row, vector, Config};
use crate::{
    gguf::{GgmlType, Gguf, TensorInfo},
    quant::ue4m3_half,
    vulkan::{Buffer, Device, Push},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use half::f16;
use std::{cell::Cell, rc::Rc};

/// This rank's share of the model.
#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub rank: usize,
    pub ranks: usize,
}

impl Split {
    pub fn heads(&self, c: &Config) -> (usize, usize) {
        let n = c.heads / self.ranks;
        (self.rank * n, n)
    }
    pub fn kv_heads(&self, c: &Config) -> (usize, usize) {
        let n = c.kv_heads / self.ranks;
        (self.rank * n, n)
    }
    pub fn ff(&self, c: &Config) -> (usize, usize) {
        let n = c.ff / self.ranks;
        (self.rank * n, n)
    }
    pub fn vocab(&self, c: &Config) -> (usize, usize) {
        let n = c.vocab_padded / self.ranks;
        (self.rank * n, n)
    }
    pub fn check(&self, c: &Config) -> Result<()> {
        ensure!(
            c.heads % self.ranks == 0 && c.kv_heads % self.ranks == 0 && (c.ff / self.ranks) % (9 * 32) == 0
                && c.vocab_padded % self.ranks == 0 && c.ff % self.ranks == 0,
            "inkling: {} ranks do not divide the heads, KV heads, expert rows (in runs of 9 blocks of 32) and vocabulary",
            self.ranks
        );
        Ok(())
    }
}

/// Bytes of one engine block (32 weights) in the head and body planes of a repacked matrix.
fn engine_sizes(ty: GgmlType) -> Result<(usize, usize)> {
    Ok(match ty {
        GgmlType::Q8_0 => (2, 32),
        GgmlType::NVFP4 => (4, 16),
        other => anyhow::bail!("inkling: unsupported matrix type {other}"),
    })
}

/// Engine block `b` (32 weights) of a GGUF row of `ty`: its head (f16 scales) and body.
fn engine_block(ty: GgmlType, row: &[u8], b: usize, head: &mut [u8], body: &mut [u8]) {
    match ty {
        GgmlType::Q8_0 => {
            let g = &row[b * 34..b * 34 + 34];
            head.copy_from_slice(&g[..2]);
            body.copy_from_slice(&g[2..]);
        }
        GgmlType::NVFP4 => {
            let g: &[u8; 36] = row[b / 2 * 36..b / 2 * 36 + 36].try_into().unwrap();
            let half = b % 2;
            for s in 0..2 {
                head[s * 2..s * 2 + 2].copy_from_slice(&f16::from_f32(ue4m3_half(g[half * 2 + s])).to_le_bytes());
            }
            let nib = |s: usize, w: usize| (g[4 + s * 8 + w % 8] >> (w / 8 * 4)) & 15;
            for j in 0..16 {
                body[j] = nib(half * 2, j) | nib(half * 2 + 1, j) << 4;
            }
        }
        _ => unreachable!(),
    }
}

/// A matrix of `rows` (GGUF row byte slices), each cut to engine blocks `b0..b0 + nb`, repacked for the kernels:
/// every block's head (f16 scales), then from word `stride` every block's body. NVFP4 bodies keep the Q4_0 nibble
/// order with two f16 scales per block, the low nibbles' and the high nibbles'.
pub struct Packed {
    pub bytes: Vec<u8>,
    pub stride: usize,
}

pub fn pack_rows(ty: GgmlType, rows: &[&[u8]], b0: usize, nb: usize) -> Result<Packed> {
    let (hs, bs) = engine_sizes(ty)?;
    let count = rows.len() * nb;
    let body_at = (count * hs).next_multiple_of(64);
    let mut bytes = vec![0u8; body_at + count * bs];
    let (heads, bodies) = bytes.split_at_mut(body_at);
    for (r, row) in rows.iter().enumerate() {
        for i in 0..nb {
            let at = r * nb + i;
            engine_block(ty, row, b0 + i, &mut heads[at * hs..(at + 1) * hs], &mut bodies[at * bs..(at + 1) * bs]);
        }
    }
    Ok(Packed { bytes, stride: body_at / 4 })
}

/// The GGUF bytes of rows `r0..r0 + n` of a 2-D (or stacked 3-D) tensor.
fn rows<'a>(g: &'a Gguf, t: &TensorInfo, r0: usize, n: usize) -> Result<Vec<&'a [u8]>> {
    let (be, bb) = t.ty.block().context("block size")?;
    let per = t.dims[0] as usize / be as usize * bb as usize;
    let data = g.data(t);
    ensure!((r0 + n) * per <= data.len(), "rows out of range in {}", t.name);
    Ok((r0..r0 + n).map(|r| &data[r * per..(r + 1) * per]).collect())
}

/// A repacked matrix on the GPU.
pub struct Matrix {
    pub buf: Buffer,
    pub stride: usize,
    pub rows: usize,
    pub k: usize,
    pub ty: GgmlType,
}

fn upload_packed(gpu: &Rc<Device>, ty: GgmlType, rows_: &[&[u8]], b0: usize, nb: usize) -> Result<Matrix> {
    let p = pack_rows(ty, rows_, b0, nb)?;
    Ok(Matrix { buf: gpu.upload(&p.bytes)?, stride: p.stride, rows: rows_.len(), k: nb * 32, ty })
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

fn upload_f32(gpu: &Rc<Device>, v: &[f32]) -> Result<Buffer> {
    gpu.upload(f32_bytes(v))
}

/// Rows `r0..r0 + n` of a small matrix (F32, F16 or BF16; Q8_0 too) as f32.
fn f32_rows(g: &Gguf, t: &TensorInfo, r0: usize, n: usize) -> Result<Vec<f32>> {
    let k = t.dims[0] as usize;
    let mut v = vec![0f32; n * k];
    for (i, o) in v.chunks_exact_mut(k).enumerate() {
        dequant_row(g, t, r0 + i, o)?;
    }
    Ok(v)
}

pub struct Layer {
    pub swa: bool,
    /// q, gate and r rows of this rank's query heads, stacked (input xq).
    qgr: Matrix,
    wk: Matrix,
    wv: Matrix,
    /// This rank's columns (its query heads) of the output projection.
    wo: Matrix,
    /// f32 weights of inkling_prep: K and V conv taps, q and r clip scales, sink rows, K and V tap rows (global).
    small: Buffer,
    /// f32 rows: sinks, then on global layers K and V tap rows, stacked (inkling_qkv) and apart (prefill).
    rows32: Buffer,
    nrows32: usize,
    sinkw: Buffer,
    tapw: Option<(Buffer, Buffer)>,
    relb: Buffer,
    router: Buffer,
    bias: Buffer,
    gu: Matrix,
    down: Matrix,
    sgu: Buffer,
    sd: Buffer,
    /// Input norms (SWA: attn_norm; global: q, k, v input norms), then ffn_norm.
    norms: Buffer,
    /// Core layers: controller down matrices [4][rank][2d], then gains and forgets [4][d][rank] (forgets from word
    /// `ctrl_stride`).
    ctrl: Option<(Buffer, Buffer, usize)>,
}

/// The cache of one slot: f16 K and V of this rank's KV heads, and the convolution history of K and V.
struct Slot {
    k: Buffer,
    v: Buffer,
    hist: Buffer,
    cap: usize,
}

pub struct Shard {
    pub gpu: Rc<Device>,
    pub split: Split,
    pub c: Config,
    pub layers: Vec<Layer>,
    slots: Vec<Slot>,
    head: Matrix,
    pub max_batch: usize,
    pub context: usize,
    hq: usize,
    hk: usize,
    f: usize,
    qgr_w: usize,
    // Host-visible: input rows, partial outputs and params of each micro-batch, head logits.
    hin: [Buffer; MICRO],
    part: [Buffer; MICRO],
    logits: Buffer,
    params: [Buffer; MICRO],
    /// The micro-batch whose segment is being recorded.
    mb: Cell<usize>,
    // Device scratch.
    raw: Buffer,
    xin: Buffer,
    z: Buffer,
    qgr: Buffer,
    /// K pre-convolution rows [t][C], then from float `max_batch * C` the sinks [t][heads].
    kpre: Buffer,
    vpre: Buffer,
    apart: Buffer,
    aout: Buffer,
    rlog: Buffer,
    ids: Buffer,
    wts: Buffer,
    act: Buffer,
    /// Prefill experts: the tiles and the pairs listed by expert (qwen4_moe_tiles), and each pair's down output.
    tiles: Buffer,
    list: Buffer,
    ypair: Buffer,
    timeline: vk::Semaphore,
    counter: u64,
    /// Per-segment command buffers in flight, with their timeline values.
    cmds: Vec<(u64, vk::CommandBuffer)>,
    /// Event-gated graphs by (tokens, plan), the release and done events they share, and whether one is running.
    graphs: std::collections::HashMap<(usize, Vec<Seg>), Graph>,
    gate: (vk::Event, vk::Event),
    gated: bool,
    /// Profiling (`TRASHCAN_INKLING_PROFILE=1`): a timestamp pool, the labels of the segment in flight, and GPU
    /// milliseconds per kernel and in whole segments.
    stamp_pool: Option<[vk::QueryPool; MICRO]>,
    stamp_labels: [Vec<String>; MICRO],
    pub profile: std::collections::BTreeMap<String, (u64, f64)>,
}

/// Batches up to this many tokens split each token's keys into partitions (decode, verification); larger ones
/// (prefill) take one partition per token and head.
const SPLIT_MAX: usize = 16;

/// Attention partitions per token and head for `n` tokens: the kernel divides the visible keys among them at run
/// time, so a recorded graph serves every position.
pub fn partitions(n: usize, swa: bool) -> usize {
    match (n > SPLIT_MAX, swa) {
        (true, _) => 1,
        (false, true) => 8,
        (false, false) => 32,
    }
}

/// One segment of a forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Seg {
    /// Attention of application `slot` (physical layer `phys`): `kinds` input row sets (1 or 3), with messages.
    Attn { slot: usize, phys: usize, kinds: usize, controlled: bool },
    Moe { phys: usize, controlled: bool },
    /// The head over `rows` rows.
    Head { rows: usize },
}

/// A recorded event-gated forward: one command buffer whose segments wait for the host's release event.
struct Graph {
    cmd: vk::CommandBuffer,
    labels: Vec<String>,
}

/// Pairs per tile of the prefill expert kernels (inkling_moe_mm).
const TILE: usize = 8;

/// Micro-batches of a prefill batch that can be in flight together.
pub const MICRO: usize = 2;

/// Forwards of at most this many tokens are event-gated graphs (one submission per forward); longer ones submit
/// each segment (a gated submission lasts the whole forward, and amdgpu resets a GPU whose job runs past 2 s).
pub const GATED_MAX: usize = 8;

/// Stages after which gated graphs set and reset their events: all prior work.
const GATE_STAGES: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
    vk::PipelineStageFlags::COMPUTE_SHADER.as_raw() | vk::PipelineStageFlags::TRANSFER.as_raw(),
);

impl Shard {
    pub fn new(gpu: Rc<Device>, g: &Gguf, c: &Config, split: Split, context: usize, max_batch: usize) -> Result<Self> {
        split.check(c)?;
        let d = c.d;
        let (h0, hq) = split.heads(c);
        let (g0, hk) = split.kv_heads(c);
        let (f0, f) = split.ff(c);
        let (v0, vn) = split.vocab(c);
        let qgr_w = 2 * hq * 128 + hq * c.rel;
        let mut layers = Vec::new();
        for p in 0..c.layers {
            let lt = |n: &str| c.lt(g, p, n);
            let q = lt("attn_q.weight")?;
            let gate = lt("attn_gate.weight")?;
            let rel = lt("attn_rel.weight")?;
            ensure!(q.ty == GgmlType::Q8_0 && gate.ty == q.ty && rel.ty == q.ty, "inkling: attention projections must be Q8_0 (layer {p})");
            let mut qrows = rows(g, q, h0 * 128, hq * 128)?;
            qrows.extend(rows(g, gate, h0 * 128, hq * 128)?);
            qrows.extend(rows(g, rel, h0 * c.rel, hq * c.rel)?);
            let qgr = upload_packed(&gpu, GgmlType::Q8_0, &qrows, 0, d / 32)?;
            let wkt = lt("attn_k.weight")?;
            let wvt = lt("attn_v.weight")?;
            let wk = upload_packed(&gpu, wkt.ty, &rows(g, wkt, g0 * 128, hk * 128)?, 0, d / 32)?;
            let wv = upload_packed(&gpu, wvt.ty, &rows(g, wvt, g0 * 128, hk * 128)?, 0, d / 32)?;
            let wot = lt("attn_output.weight")?;
            let wo = upload_packed(&gpu, wot.ty, &rows(g, wot, 0, d)?, h0 * 128 / 32, hq * 128 / 32)?;
            let swa = c.swa(p);
            let kc = vector(g, lt("attn_k_conv.weight")?)?;
            let vc = vector(g, lt("attn_v_conv.weight")?)?;
            let ch = g0 * 128 * 4..(g0 + hk) * 128 * 4;
            let mut small = kc[ch.clone()].to_vec();
            small.extend_from_slice(&vc[ch]);
            let qc = vector(g, lt("attn_q_clip.weight")?)?;
            let rc = vector(g, lt("attn_rel_clip.weight")?)?;
            small.extend_from_slice(&qc[h0 * 128..(h0 + hq) * 128]);
            small.extend_from_slice(&rc[h0 * c.rel..(h0 + hq) * c.rel]);
            let sinkr = f32_rows(g, lt("attn_sink_proj.weight")?, h0, hq)?;
            let mut rows32 = sinkr.clone();
            let tapw = if !swa {
                let kt = f32_rows(g, lt("attn_k_conv_proj.weight")?, g0 * 4, hk * 4)?;
                let vt = f32_rows(g, lt("attn_v_conv_proj.weight")?, g0 * 4, hk * 4)?;
                rows32.extend_from_slice(&kt);
                rows32.extend_from_slice(&vt);
                Some((upload_f32(&gpu, &kt)?, upload_f32(&gpu, &vt)?))
            } else {
                None
            };
            let nrows32 = rows32.len() / d;
            let relb = vector(g, lt("attn_rel_bias.weight")?)?;
            ensure!(relb.len() == c.rel * c.rel_extent, "attn_rel_bias shape at {p}");
            let router = vector(g, lt("ffn_gate_inp.weight")?)?;
            let bias = g.tensor(&format!("blk.{p}.exp_probs_b.bias")).map(|t| vector(g, t)).transpose()?.unwrap_or_else(|| vec![0.; c.experts]);
            // Experts: gate rows f0.. and up rows ff + f0.. of every expert, and every expert's down rows cut to the
            // columns f0..f0 + f.
            let gut = lt("ffn_gate_up_exps.weight")?;
            let dt = lt("ffn_down_exps.weight")?;
            ensure!(gut.ty == dt.ty, "inkling: gate/up and down types differ at layer {p}");
            let mut gr = Vec::with_capacity(c.experts * 2 * f);
            for e in 0..c.experts {
                gr.extend(rows(g, gut, e * 2 * c.ff + f0, f)?);
                gr.extend(rows(g, gut, e * 2 * c.ff + c.ff + f0, f)?);
            }
            let gu = upload_packed(&gpu, gut.ty, &gr, 0, d / 32)?;
            drop(gr);
            let down = upload_packed(&gpu, dt.ty, &rows(g, dt, 0, c.experts * d)?, f0 / 32, f / 32)?;
            let scale = |n: &str| -> Result<Vec<f32>> {
                Ok(match g.tensor(&format!("blk.{p}.{n}")) {
                    Some(t) => vector(g, t)?,
                    None => vec![1.; c.experts],
                })
            };
            let mut norms = Vec::new();
            for n in if swa { &["attn_norm.weight"][..] } else { &["attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight"][..] } {
                norms.extend(vector(g, lt(n)?)?);
            }
            norms.extend(vector(g, lt("ffn_norm.weight")?)?);
            let ctrl = if c.looped(p) {
                let (mut down, mut gain, mut forget) = (Vec::new(), Vec::new(), Vec::new());
                for i in 0..4 {
                    let get = |part: &str| -> Result<Vec<f32>> {
                        let name = c.ctrl_name(p, i, part);
                        vector(g, g.tensor(&name).with_context(|| format!("missing {name}"))?)
                    };
                    down.extend(get("down")?);
                    // Gains and forgets transposed to [rank][d], so neighbouring threads read neighbouring weights.
                    let r = c.gate_rank;
                    let tr = |v: Vec<f32>| -> Vec<f32> { (0..r * d).map(|k| v[(k % d) * r + k / d]).collect() };
                    gain.extend(tr(get("gain")?));
                    forget.extend(tr(get("forget")?));
                }
                let stride = gain.len();
                gain.extend(forget);
                Some((upload_f32(&gpu, &down)?, upload_f32(&gpu, &gain)?, stride))
            } else {
                None
            };
            let sgu = upload_f32(&gpu, &scale("ffn_gate_up_exps.scale")?)?;
            let sd = upload_f32(&gpu, &scale("ffn_down_exps.scale")?)?;
            layers.push(Layer {
                swa,
                qgr,
                wk,
                wv,
                wo,
                small: upload_f32(&gpu, &small)?,
                rows32: upload_f32(&gpu, &rows32)?,
                nrows32,
                sinkw: upload_f32(&gpu, &sinkr)?,
                tapw,
                relb: upload_f32(&gpu, &relb)?,
                router: upload_f32(&gpu, &router)?,
                bias: upload_f32(&gpu, &bias)?,
                gu,
                down,
                sgu,
                sd,
                norms: upload_f32(&gpu, &norms)?,
                ctrl,
            });
        }
        let ht = c.t(g, "output.weight")?;
        let head = upload_packed(&gpu, ht.ty, &rows(g, ht, v0, vn)?, 0, d / 32)?;
        let mut slots = Vec::new();
        for a in c.schedule() {
            let cap = if c.swa(a.phys) { (c.window + max_batch).min(context) } else { context };
            let kvb = cap * hk * 128 * 2;
            slots.push((a.slot, Slot {
                k: gpu.buffer(kvb, false, false, None)?,
                v: gpu.buffer(kvb, false, false, None)?,
                hist: gpu.upload(&vec![0u8; 2 * 3 * hk * 128 * 4])?,
                cap,
            }));
        }
        slots.sort_by_key(|s| s.0);
        let slots = slots.into_iter().map(|s| s.1).collect();
        let nb = max_batch;
        let fb = |n: usize| gpu.buffer(n * 4, false, false, None);
        let hb = |n: usize| gpu.buffer(n * 4, true, false, None);
        let timeline = gpu.timeline(false)?;
        let gate = unsafe {
            (gpu.dev.create_event(&vk::EventCreateInfo::default(), None)?, gpu.dev.create_event(&vk::EventCreateInfo::default(), None)?)
        };
        let stamp_pool = if std::env::var("TRASHCAN_INKLING_PROFILE").is_ok() {
            let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(crate::vulkan::STAMP_CAPACITY);
            Some(unsafe { [gpu.dev.create_query_pool(&info, None)?, gpu.dev.create_query_pool(&info, None)?] })
        } else {
            None
        };
        Ok(Self {
            hin: [hb(6 * nb * d)?, hb(6 * nb * d)?],
            raw: fb(6 * nb * d)?,
            z: fb(nb * 3 * c.gate_rank)?,
            part: [hb(nb * d)?, hb(nb * d)?],
            logits: hb(nb.min(16) * vn)?,
            params: [hb(64)?, hb(64)?],
            mb: Cell::new(0),
            xin: fb(3 * nb * d)?,
            qgr: fb(nb * qgr_w)?,
            kpre: fb(nb * hk * 128 + nb * hq + 2 * nb * hk * 4)?,
            vpre: fb(nb * hk * 128)?,
            apart: fb((nb * hq * 130).max(SPLIT_MAX.min(nb) * hq * 32 * 130))?,
            aout: fb(nb * hq * 128)?,
            rlog: fb(nb * c.experts)?,
            ids: fb(nb * c.used)?,
            wts: fb(nb * c.used)?,
            act: fb(nb * c.used * f)?,
            tiles: fb(1 + 2 * ((nb * c.used).div_ceil(TILE) + c.experts))?,
            list: fb(nb * c.used)?,
            ypair: fb(if nb > GATED_MAX { nb * c.used * d } else { 1 })?,
            gpu,
            split,
            c: c.clone(),
            layers,
            slots,
            head,
            max_batch,
            context,
            hq,
            hk,
            f,
            qgr_w,
            stamp_pool,
            graphs: Default::default(),
            gate,
            gated: false,
            stamp_labels: Default::default(),
            profile: Default::default(),
            timeline,
            counter: 0,
            cmds: Vec::new(),
        })
    }

    pub fn device_mib(&self) -> f64 {
        self.gpu.device_bytes.get() as f64 / (1 << 20) as f64
    }

    /// Rows of the vocabulary this rank scores.
    pub fn vocab_rows(&self) -> (usize, usize) {
        self.split.vocab(&self.c)
    }

    /// The host-visible input rows of micro-batch `mb` (`3 * n * d` floats for attention, `n * d` for the MoE and
    /// the head).
    pub fn hin(&self, mb: usize, len: usize) -> &mut [f32] {
        assert!(len <= self.hin[mb].size / 4);
        unsafe { std::slice::from_raw_parts_mut(self.hin[mb].ptr.cast::<f32>(), len) }
    }
    /// This rank's partial output of micro-batch `mb`'s last attention or MoE segment (`n * d` floats).
    pub fn part(&self, mb: usize, len: usize) -> &[f32] {
        assert!(len <= self.part[mb].size / 4);
        unsafe { std::slice::from_raw_parts(self.part[mb].ptr.cast::<f32>(), len) }
    }
    /// This rank's head logits of the last head segment, `rows` rows of `vocab_rows().1`.
    pub fn logits(&self, rows: usize) -> &[f32] {
        let n = rows * self.vocab_rows().1;
        unsafe { std::slice::from_raw_parts(self.logits.ptr.cast::<f32>(), n) }
    }

    /// Set the position and batch of micro-batch `mb`'s next segments.
    pub fn set_params(&self, mb: usize, pos: usize, n: usize) -> Result<()> {
        ensure!(n >= 1 && n <= self.max_batch, "batch of {n} tokens (at most {})", self.max_batch);
        ensure!(pos + n <= self.context, "context of {} tokens exceeded", self.context);
        self.params[mb].write(bytemuck::cast_slice(&[pos as u32, n as u32, 0, 0]))
    }

    fn push(&self, ptr: [u64; 6], extra: u64, u: [u32; 16]) -> Push {
        Push { ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], self.params[self.mb.get()].address, extra], u }
    }
    /// The input rows and the partial output of the micro-batch being recorded.
    fn hin_buf(&self) -> &Buffer {
        &self.hin[self.mb.get()]
    }
    fn part_buf(&self) -> &Buffer {
        &self.part[self.mb.get()]
    }

    /// A Q8_0 matrix times `n` columns of `x` into `y` (decode kernels up to 4 columns, the GEMM above).
    fn matrix(&self, cmd: vk::CommandBuffer, m: &Matrix, x: u64, y: u64, n: usize) -> Result<()> {
        self.matrix_with(cmd, m, x, y, n, 0)
    }

    /// `matrix` over `rows` columns whatever the forward's batch (the head).
    fn matrix_rows(&self, cmd: vk::CommandBuffer, m: &Matrix, x: u64, y: u64, rows: usize) -> Result<()> {
        self.matrix_with(cmd, m, x, y, rows, rows as u32)
    }

    fn matrix_with(&self, cmd: vk::CommandBuffer, m: &Matrix, x: u64, y: u64, n: usize, batch: u32) -> Result<()> {
        ensure!(m.ty == GgmlType::Q8_0, "matrix kernels read Q8_0");
        let mut u = [0u32; 16];
        u[15] = batch;
        u[0] = m.rows as u32;
        u[1] = m.k as u32;
        u[3] = m.stride as u32;
        if n <= 4 {
            let wide = m.rows >= 4096;
            let names = if wide {
                ["qwen4_q8_mvw4_r4", "qwen4_q8_mvw4_r4_c2", "qwen4_q8_mvw4_r4_c3", "qwen4_q8_mvw4_r4_c4"]
            } else {
                ["qwen4_q8_mvw4_r2", "qwen4_q8_mvw4_r2_c2", "qwen4_q8_mvw4_r2_c3", "qwen4_q8_mvw4_r2_c4"]
            };
            let rpw = if wide { 16 } else { 8 };
            let p = self.push([0, x, y, 0, 0, 0], 0, u);
            self.gpu.dispatch_matrix(cmd, names[n - 1], p, m.rows.div_ceil(rpw) as u32, 1, &m.buf)
        } else {
            let p = self.push([0, x, y, 0, 0, 0], 0, u);
            self.gpu.dispatch_matrix(cmd, "qwen4_q8_gemm", p, m.rows.div_ceil(64) as u32, n.div_ceil(64) as u32, &m.buf)
        }
    }

    /// An f32 matrix (`rows` x `k`, in set 0) times `n` columns.
    fn matf32(&self, cmd: vk::CommandBuffer, w: &Buffer, rows: usize, k: usize, x: u64, y: u64, n: usize) -> Result<()> {
        let mut u = [0u32; 16];
        u[0] = rows as u32;
        u[1] = k as u32;
        let p = self.push([0, x, y, 0, 0, 0], 0, u);
        match n {
            1 => self.gpu.dispatch_matrix(cmd, "qwen4_matf32", p, rows.div_ceil(4) as u32, 1, w),
            2..=4 => self.gpu.dispatch_matrix(cmd, "qwen4_matf32_c4", p, rows.div_ceil(4) as u32, 1, w),
            _ => self.gpu.dispatch_matrix(cmd, "qwen_matf32", p, rows.div_ceil(4) as u32, n.div_ceil(8) as u32, w),
        }
    }

    /// Profile the submission being recorded with micro-batch `mb`'s timestamps.
    fn start_stamps(&self, cmd: vk::CommandBuffer, mb: usize) {
        if let Some(pool) = self.stamp_pool.map(|p| p[mb]) {
            unsafe { self.gpu.dev.cmd_reset_query_pool(cmd, pool, 0, crate::vulkan::STAMP_CAPACITY) };
            *self.gpu.stamps.borrow_mut() = Some(crate::vulkan::Stamps { pool, labels: Vec::new(), dispatches: true });
        }
    }

    fn take_stamps(&self) -> Vec<String> {
        self.gpu.stamps.borrow_mut().take().map(|s| s.labels).unwrap_or_default()
    }

    /// Record segment `seg` for `n` tokens.
    fn record(&self, cmd: vk::CommandBuffer, seg: Seg, n: usize) -> Result<()> {
        self.gpu.stamp(cmd, false, || "(segment)".into());
        match seg {
            Seg::Attn { slot, phys, kinds, controlled } => self.rec_attention(cmd, slot, phys, n, kinds, controlled)?,
            Seg::Moe { phys, controlled } => self.rec_moe(cmd, phys, n, controlled)?,
            Seg::Head { rows } => self.rec_head(cmd, rows)?,
        }
        self.gpu.stamp(cmd, true, || "(segment)".into());
        Ok(())
    }

    /// Submit one segment of micro-batch `mb` on its own (prefill batches): the host has written its inputs into
    /// `hin`. It runs after the submissions before it; `wait` takes the returned ticket, before the micro-batch's
    /// next segment is submitted.
    pub fn submit(&mut self, seg: Seg, n: usize, mb: usize) -> Result<u64> {
        let cmd = self.gpu.begin()?;
        self.start_stamps(cmd, mb);
        self.gpu.host_barrier(cmd, false);
        self.mb.set(mb);
        let recorded = self.record(cmd, seg, n);
        self.mb.set(0);
        recorded?;
        self.gpu.host_barrier(cmd, true);
        self.stamp_labels[mb] = self.take_stamps();
        self.gpu.finish(cmd)?;
        self.counter += 1;
        self.gpu.submit(cmd, None, Some((self.timeline, self.counter)), vk::Fence::null())?;
        self.cmds.push((self.counter, cmd));
        self.gated = false;
        Ok(self.counter)
    }

    /// Wait for micro-batch `mb`'s submission `ticket` (`submit`), free the command buffers up to it and collect
    /// its profiling.
    pub fn wait(&mut self, ticket: u64, mb: usize) -> Result<()> {
        let start = std::time::Instant::now();
        while unsafe { self.gpu.dev.get_semaphore_counter_value(self.timeline)? } < ticket {
            ensure!(start.elapsed().as_secs() < 10, "GPU segment did not finish");
            std::hint::spin_loop();
        }
        let gpu = &self.gpu;
        self.cmds.retain(|&(t, cmd)| {
            if t <= ticket {
                unsafe { gpu.dev.free_command_buffers(gpu.pool, &[cmd]) };
            }
            t > ticket
        });
        self.collect_stamps(mb)
    }

    /// Add the timestamps of micro-batch `mb`'s finished submission to the profile.
    fn collect_stamps(&mut self, mb: usize) -> Result<()> {
        let labels = std::mem::take(&mut self.stamp_labels[mb]);
        let (Some(pools), false) = (self.stamp_pool, labels.is_empty()) else { return Ok(()) };
        let mut ticks = vec![0u64; labels.len()];
        unsafe { self.gpu.dev.get_query_pool_results(pools[mb], 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
        let period = self.gpu.period as f64 * 1e-6;
        // Stamps come in (top, bottom) pairs per segment, dispatch and copy.
        let mut open: std::collections::HashMap<&str, u64> = Default::default();
        for (l, &t) in labels.iter().zip(&ticks) {
            let key = l.split(' ').next().unwrap_or(l);
            if let Some(t0) = open.remove(key) {
                let e = self.profile.entry(key.to_owned()).or_default();
                e.0 += 1;
                e.1 += (t.saturating_sub(t0)) as f64 * period;
            } else {
                open.insert(key, t);
            }
        }
        Ok(())
    }

    /// Start the event-gated graph of `plan` for `n` tokens (recorded on first use): its first segment runs at once
    /// (the host has written its inputs), each later one when `release` is called.
    pub fn start_gated(&mut self, plan: &[Seg], n: usize) -> Result<()> {
        ensure!(n <= GATED_MAX, "gated forwards take at most {GATED_MAX} tokens");
        self.retire()?;
        let key = (n, plan.to_vec());
        if !self.graphs.contains_key(&key) {
            let cmd = self.gpu.begin()?;
            self.start_stamps(cmd, 0);
            self.gpu.host_barrier(cmd, false);
            for (j, &seg) in plan.iter().enumerate() {
                if j > 0 {
                    self.gpu.wait_host_event(cmd, self.gate.0);
                    unsafe { self.gpu.dev.cmd_reset_event(cmd, self.gate.0, GATE_STAGES) };
                }
                self.record(cmd, seg, n)?;
                self.gpu.host_barrier(cmd, true);
                unsafe { self.gpu.dev.cmd_set_event(cmd, self.gate.1, GATE_STAGES) };
                self.gpu.barrier(cmd);
            }
            let labels = self.take_stamps();
            self.gpu.finish(cmd)?;
            self.graphs.insert(key.clone(), Graph { cmd, labels });
        }
        let g = &self.graphs[&key];
        unsafe {
            self.gpu.dev.reset_event(self.gate.0)?;
            self.gpu.dev.reset_event(self.gate.1)?;
        }
        self.counter += 1;
        self.gpu.submit(g.cmd, None, Some((self.timeline, self.counter)), vk::Fence::null())?;
        self.stamp_labels[0] = g.labels.clone();
        self.gated = true;
        Ok(())
    }

    /// Let the next segment of the running gated graph start (the host has written its inputs).
    pub fn release(&mut self) -> Result<()> {
        unsafe { self.gpu.dev.set_event(self.gate.0)? };
        Ok(())
    }

    /// Wait for the running gated graph's segment in flight (its done event).
    pub fn wait_gated(&mut self) -> Result<()> {
        let start = std::time::Instant::now();
        while !unsafe { self.gpu.dev.get_event_status(self.gate.1)? } {
            ensure!(start.elapsed().as_secs() < 10, "GPU segment did not finish");
            std::hint::spin_loop();
        }
        unsafe { self.gpu.dev.reset_event(self.gate.1)? };
        Ok(())
    }

    /// Wait for the last submission to retire (after its last segment, or draining a gated graph that an error
    /// left waiting), free per-segment command buffers and collect profiling.
    pub fn retire(&mut self) -> Result<()> {
        let start = std::time::Instant::now();
        while unsafe { self.gpu.dev.get_semaphore_counter_value(self.timeline)? } < self.counter {
            if self.gated {
                unsafe { self.gpu.dev.set_event(self.gate.0)? };
            }
            ensure!(start.elapsed().as_secs() < 10, "a GPU submission did not finish");
            std::hint::spin_loop();
        }
        for (_, cmd) in self.cmds.drain(..) {
            unsafe { self.gpu.dev.free_command_buffers(self.gpu.pool, &[cmd]) };
        }
        self.gated = false;
        (0..MICRO).try_for_each(|mb| self.collect_stamps(mb))
    }

    /// Record the inputs of a sublayer: copy the host's `kinds` rows of `n` tokens (and with `ctrl`, `kinds` message
    /// rows after them) from `hin`, apply the loop controllers `ctrl` (first controller index), then the RMS norms
    /// (vector `norm`, or `norm + j` with `per_kind`) into `xin` ([kinds][n][d]).
    fn inputs(&self, cmd: vk::CommandBuffer, l: &Layer, n: usize, kinds: usize, ctrl: Option<usize>, norm: usize, per_kind: bool) -> Result<()> {
        let c = &self.c;
        let d = c.d;
        // The controllers read their rows from many workgroups: copy them to VRAM first. A norm alone reads the host's
        // rows once.
        let src = if ctrl.is_some() {
            self.gpu.copy(cmd, self.hin_buf(), 0, &self.raw, 0, 2 * kinds * n * d * 4);
            self.gpu.barrier(cmd);
            self.raw.address
        } else {
            self.hin_buf().address
        };
        if let Some(first) = ctrl {
            let (down, gf, stride) = l.ctrl.as_ref().context("controllers of a layer outside the loop")?;
            let m = self.raw.address + (kinds * n * d * 4) as u64;
            let mut u = [0u32; 16];
            u[0] = c.gate_rank as u32;
            u[1] = d as u32;
            u[2] = kinds as u32;
            u[10] = first as u32;
            let p = self.push([0, self.raw.address, m, self.z.address, 0, 0], 0, u);
            self.gpu.dispatch_matrix(cmd, "inkling_ctrl_z", p, c.gate_rank as u32, (n.div_ceil(8) * kinds) as u32, down)?;
            u[3] = *stride as u32;
            let p = self.push([self.raw.address, 0, m, self.z.address, 0, 0], 0, u);
            self.gpu.dispatch_matrix(cmd, "inkling_ctrl_out", p, d.div_ceil(64) as u32, (n.div_ceil(8) * kinds) as u32, gf)?;
        }
        let mut u = [0u32; 16];
        u[1] = d as u32;
        u[10] = norm as u32;
        u[11] = u32::from(!per_kind);
        u[12] = c.eps.to_bits();
        let p = self.push([self.xin.address, src, 0, 0, 0, 0], 0, u);
        self.gpu.dispatch_matrix(cmd, "inkling_norm", p, n as u32, kinds as u32, &l.norms)
    }

    /// Submit the attention segment of application `slot` (physical layer `phys`) for `n` tokens at `pos`. The host
    /// has written into `hin` the mixed inputs: one row set (SWA layers outside later loop passes), or q, k and v
    /// (`3 * n * d`), followed with `controlled` by the q, k and v messages. The partial output projection lands in
    /// `part`.
    fn rec_attention(&self, cmd: vk::CommandBuffer, slot: usize, phys: usize, n: usize, kinds: usize, controlled: bool) -> Result<()> {
        let c = &self.c;
        let d = c.d;
        let (hq, hk) = (self.hq, self.hk);
        let ch = hk * 128;
        let l = &self.layers[phys];
        let s = &self.slots[slot];
        self.inputs(cmd, l, n, kinds, controlled.then_some(0), 0, !l.swa)?;
        let step = if kinds == 3 { (n * d * 4) as u64 } else { 0 };
        let (xq, xk, xv) = (self.xin.address, self.xin.address + step, self.xin.address + 2 * step);
        let sink_off = self.max_batch * ch;
        if n <= 4 {
            let mut u = [0u32; 16];
            u[4] = hq as u32;
            u[5] = hk as u32;
            u[7] = l.nrows32 as u32;
            u[8] = sink_off as u32;
            u[0] = l.qgr.rows as u32;
            u[1] = d as u32;
            u[2] = l.wk.rows as u32;
            u[3] = l.qgr.stride as u32;
            u[6] = l.wv.rows as u32;
            u[10] = (step / 4) as u32;
            u[12] = l.wk.stride as u32;
            u[13] = l.wv.stride as u32;
            let p = self.push([self.vpre.address, xq, self.qgr.address, l.wk.buf.address, l.wv.buf.address, self.kpre.address], l.rows32.address, u);
            let name = ["inkling_qkv", "inkling_qkv_c2", "inkling_qkv_c3", "inkling_qkv_c4"][n - 1];
            let groups = (l.qgr.rows + l.wk.rows + l.wv.rows) / 8 + l.nrows32.div_ceil(4);
            self.gpu.dispatch_matrix(cmd, name, p, groups as u32, 1, &l.qgr.buf)?;
        } else {
            self.matrix(cmd, &l.qgr, xq, self.qgr.address, n)?;
            self.matrix(cmd, &l.wk, xk, self.kpre.address, n)?;
            self.matrix(cmd, &l.wv, xv, self.vpre.address, n)?;
            let at = |floats: usize| self.kpre.address + ((sink_off + floats) * 4) as u64;
            self.matf32(cmd, &l.sinkw, hq, d, xq, at(0), n)?;
            if let Some((kt, vt)) = &l.tapw {
                self.matf32(cmd, kt, hk * 4, d, xk, at(n * hq), n)?;
                self.matf32(cmd, vt, hk * 4, d, xv, at(n * hq + n * hk * 4), n)?;
            }
        }
        let mut u = [0u32; 16];
        u[0] = self.qgr_w as u32;
        u[1] = d as u32;
        u[2] = sink_off as u32;
        u[4] = hq as u32;
        u[5] = hk as u32;
        u[6] = c.log_floor.to_bits();
        u[7] = s.cap as u32;
        u[8] = c.log_alpha.to_bits();
        u[10] = (step / 4) as u32;
        u[11] = u32::from(!l.swa) | (u32::from(n == 1) << 1);
        u[12] = c.qk_clip.to_bits();
        u[13] = c.rel_clip.to_bits();
        let p = self.push([self.qgr.address, xq, self.kpre.address, self.vpre.address, s.hist.address, s.k.address], s.v.address, u);
        self.gpu.dispatch_matrix(cmd, "inkling_prep", p, (n * 2 * hk + n * hq) as u32, 1, &l.small)?;
        if n > 1 {
            let mut u = [0u32; 16];
            u[1] = ch as u32;
            let p = self.push([0, self.kpre.address, self.vpre.address, s.hist.address, 0, 0], 0, u);
            self.gpu.dispatch(cmd, "inkling_kv_hist", p, (2 * ch).div_ceil(256) as u32, 1);
        }
        let window = if l.swa { c.window } else { 0 };
        let parts = partitions(n, l.swa);
        let mut u = [0u32; 16];
        u[1] = self.qgr_w as u32;
        u[2] = parts as u32;
        u[4] = hq as u32;
        u[5] = hk as u32;
        u[7] = s.cap as u32;
        u[8] = window as u32;
        u[12] = c.rel_extent as u32;
        let p = self.push([0, self.qgr.address, self.kpre.address + (sink_off * 4) as u64, s.k.address, s.v.address, self.apart.address], 0, u);
        self.gpu.dispatch_matrix(cmd, "inkling_attn", p, (n * hq) as u32, parts as u32, &l.relb)?;
        let mut u = [0u32; 16];
        u[1] = self.qgr_w as u32;
        u[2] = parts as u32;
        u[4] = hq as u32;
        let p = self.push([self.aout.address, self.qgr.address, 0, 0, 0, self.apart.address], 0, u);
        self.gpu.dispatch(cmd, "inkling_attn_merge", p, (n * hq) as u32, 1);
        self.matrix(cmd, &l.wo, self.aout.address, self.part_buf().address, n)
    }

    /// Submit the MoE segment of physical layer `phys` for `n` tokens: the host has written the mixed input (`n * d`)
    /// into `hin`, followed with `controlled` by the ffn message; the partial expert sum lands in `part`.
    fn rec_moe(&self, cmd: vk::CommandBuffer, phys: usize, n: usize, controlled: bool) -> Result<()> {
        let c = &self.c;
        let d = c.d;
        let l = &self.layers[phys];
        let kinds = if l.swa { 1 } else { 3 };
        self.inputs(cmd, l, n, 1, controlled.then_some(3), kinds, false)?;
        self.matf32(cmd, &l.router, c.experts, d, self.xin.address, self.rlog.address, n)?;
        let mut u = [0u32; 16];
        u[0] = c.experts as u32;
        u[2] = c.used as u32;
        u[12] = c.route_scale.to_bits();
        let p = self.push([0, self.rlog.address, self.ids.address, self.wts.address, 0, 0], 0, u);
        self.gpu.dispatch_matrix(cmd, "inkling_route", p, n as u32, 1, &l.bias)?;
        let q8 = l.gu.ty != GgmlType::NVFP4;
        if n > GATED_MAX {
            // Prefill: group the pairs by expert into tiles, so each tile of up to 8 pairs decodes its expert's weights
            // once.
            let tiles = (n * c.used).div_ceil(TILE) + c.experts;
            let mut u = [0u32; 16];
            u[0] = c.experts as u32;
            u[1] = TILE as u32;
            u[2] = c.used as u32;
            let p = self.push([self.ids.address, 0, self.tiles.address, self.list.address, 0, 0], 0, u);
            self.gpu.dispatch(cmd, "qwen4_moe_tiles", p, 1, 1);
            let mut u = [0u32; 16];
            u[0] = self.f as u32;
            u[1] = d as u32;
            u[3] = l.gu.stride as u32;
            u[12] = c.situ_beta.to_bits();
            u[13] = c.situ_linear.to_bits();
            let p = self.push([0, self.xin.address, self.act.address, self.tiles.address, self.list.address, 0], l.sgu.address, u);
            let name = if q8 { "inkling_moe_mm_gu_q8" } else { "inkling_moe_mm_gu" };
            self.gpu.dispatch_matrix(cmd, name, p, self.f.div_ceil(64) as u32, tiles as u32, &l.gu.buf)?;
            let mut u = [0u32; 16];
            u[0] = d as u32;
            u[1] = self.f as u32;
            u[3] = l.down.stride as u32;
            let p = self.push([0, self.act.address, self.ypair.address, self.tiles.address, self.list.address, self.wts.address], l.sd.address, u);
            let name = if q8 { "inkling_moe_mm_down_q8" } else { "inkling_moe_mm_down" };
            self.gpu.dispatch_matrix(cmd, name, p, d.div_ceil(64) as u32, tiles as u32, &l.down.buf)?;
            let mut u = [0u32; 16];
            u[0] = d as u32;
            u[2] = c.used as u32;
            let p = self.push([0, self.ypair.address, self.part_buf().address, 0, 0, 0], 0, u);
            self.gpu.dispatch(cmd, "inkling_moe_sum", p, d.div_ceil(256) as u32, n as u32);
            return Ok(());
        }
        let (gu_name, down_name) = if q8 { ("inkling_moe_gu_q8", "inkling_moe_down_q8") } else { ("inkling_moe_gu", "inkling_moe_down") };
        let mut u = [0u32; 16];
        u[0] = self.f as u32;
        u[1] = d as u32;
        u[2] = c.used as u32;
        u[3] = l.gu.stride as u32;
        u[12] = c.situ_beta.to_bits();
        u[13] = c.situ_linear.to_bits();
        let p = self.push([0, self.xin.address, self.act.address, self.ids.address, 0, 0], l.sgu.address, u);
        self.gpu.dispatch_matrix(cmd, gu_name, p, self.f.div_ceil(8) as u32, (n * c.used) as u32, &l.gu.buf)?;
        let mut u = [0u32; 16];
        u[0] = d as u32;
        u[1] = self.f as u32;
        u[2] = c.used as u32;
        u[3] = l.down.stride as u32;
        let p = self.push([0, self.act.address, self.part_buf().address, self.wts.address, self.ids.address, 0], l.sd.address, u);
        self.gpu.dispatch_matrix(cmd, down_name, p, d.div_ceil(8) as u32, n as u32, &l.down.buf)
    }

    /// Record the head over `rows` normed rows the host wrote into `hin`: this rank's logits (before the softcap)
    /// land in `logits`.
    fn rec_head(&self, cmd: vk::CommandBuffer, rows: usize) -> Result<()> {
        ensure!(rows <= 16, "at most 16 head rows");
        let d = self.c.d;
        self.gpu.copy(cmd, self.hin_buf(), 0, &self.xin, 0, rows * d * 4);
        self.gpu.barrier(cmd);
        self.matrix_rows(cmd, &self.head, self.xin.address, self.logits.address, rows)
    }

    /// Clear every slot's convolution history (a new sequence).
    pub fn reset(&mut self) -> Result<()> {
        let zeros = vec![0u8; 2 * 3 * self.hk * 128 * 4];
        let staging = self.gpu.buffer(zeros.len(), true, false, None)?;
        staging.write(&zeros)?;
        let cmd = self.gpu.begin()?;
        for s in &self.slots {
            self.gpu.copy(cmd, &staging, 0, &s.hist, 0, zeros.len());
        }
        self.gpu.finish(cmd)?;
        self.gpu.run(cmd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::nvfp4_block;
    #[test]
    fn nvfp4_engine_blocks_match_ggml() {
        let mut x = 0x9e3779b97f4a7c15u64;
        let row: Vec<u8> = (0..4 * 36)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        let p = pack_rows(GgmlType::NVFP4, &[&row], 1, 6).unwrap();
        let nv = crate::quant::NVFP4_VALUES;
        for i in 0..6 {
            let b = 1 + i;
            let want = nvfp4_block(row[b / 2 * 36..b / 2 * 36 + 36].try_into().unwrap());
            for w in 0..32 {
                let byte = p.bytes[p.stride * 4 + i * 16 + w % 16];
                let q = if w < 16 { byte & 15 } else { byte >> 4 };
                let at = i * 4 + w / 16 * 2;
                let d = f16::from_le_bytes([p.bytes[at], p.bytes[at + 1]]).to_f32();
                assert_eq!(nv[q as usize] as f32 * d, want[b % 2 * 32 + w]);
            }
        }
    }
}
