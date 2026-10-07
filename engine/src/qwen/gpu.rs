//! One GPU holding a contiguous range of layers (and optionally the final norm, LM head and MTP block).
//! Each batch shape records one graph, replayed with a new position; the hidden state enters and leaves
//! through host-visible buffers so stages can be chained on other GPUs or machines.
//!
//! Recurrent state: verification batches (up to `verify` columns) make the DeltaNet and convolution
//! kernels also store the state after every column but the last, so `rollback` to an accepted prefix is a
//! device copy. Checkpoint slots copy the whole recurrent state (and the MTP's pending hidden row) to host
//! memory and back; attention KV needs no copy because it is addressed by position.
//!
//! Tensor parallelism (`GpuStage::shard`): the stage holds one rank's share of every layer (`Split`) and runs
//! with the per-rank head counts, so the kernels are unchanged. Each mixer and FFN ends its submission with a
//! partial output projection in host memory; `tp::TpGroup` all-reduces the partials and writes the total
//! that the next submission adds to the replicated residual.
use super::{cpu::Captures, vector, Config};
use crate::{
    gguf::{GgmlType, Gguf},
    quant,
    vulkan::{attention_shaders, q4_decode_shader, Buffer, Device, Push},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::{collections::HashMap, rc::Rc};

pub const TOP_K: usize = 64;
const GRAPH_CACHE_CAPACITY: usize = 24;

impl Tp {
    /// Begin the segments `cmds`: both timelines only grow, so their values start above everything signalled
    /// so far. Returns the number of segments.
    fn start(&mut self, cmds: Vec<vk::CommandBuffer>, head_rows: usize) -> usize {
        self.base = self.done_value.max(self.released) + 1;
        self.head_rows = head_rows;
        self.cmds = cmds;
        self.cmds.len()
    }
}

/// One rank's share of every layer under tensor parallelism: its key heads with the value heads that read
/// them, its query heads with their KV heads, a slice of the FFN and of the vocabulary. The hidden state is
/// replicated. Rank 0 of 1 is the whole layer.
#[derive(Clone, Copy, Debug)]
pub struct Split {
    pub rank: usize,
    pub ranks: usize,
}

/// (first, count) ranges of rows or columns; empty means all of them.
type Ranges = Vec<(usize, usize)>;

impl Split {
    pub const WHOLE: Split = Split { rank: 0, ranks: 1 };

    /// The configuration a rank's kernels see: head counts, FFN width and vocabulary divided by the ranks.
    pub fn local(&self, c: &Config) -> Result<Config> {
        let r = self.ranks;
        ensure!(
            c.k_heads.is_multiple_of(r) && c.kv_heads.is_multiple_of(r) && c.ff.is_multiple_of(32 * r) && c.vocab.is_multiple_of(r),
            "{} ranks do not divide this model's heads, FFN or vocabulary",
            r
        );
        let mut l = c.clone();
        l.k_heads /= r;
        l.v_heads /= r;
        l.heads /= r;
        l.kv_heads /= r;
        l.ff /= r;
        l.vocab /= r;
        Ok(l)
    }

    /// This rank's (first, count) of `total` items split evenly.
    pub fn part(&self, total: usize) -> (usize, usize) {
        (self.rank * total / self.ranks, total / self.ranks)
    }

    /// `ranges` scaled by `unit`, or empty (everything) for the whole layer.
    fn sel(&self, ranges: &[(usize, usize)], unit: usize) -> Ranges {
        if self.ranks == 1 {
            return Vec::new();
        }
        ranges.iter().map(|&(a, n)| (a * unit, n * unit)).collect()
    }

    /// Value heads of this rank as ranges, in the order its DeltaNet kernels see them. Value head h reads key
    /// head h % k_heads, so every group of k_heads value heads contributes this rank's key-head range, and
    /// local value head j then reads local key head j % (local key heads).
    fn value_heads(&self, c: &Config) -> Ranges {
        let (first, count) = self.part(c.k_heads);
        (0..c.v_heads / c.k_heads).map(|b| (b * c.k_heads + first, count)).collect()
    }

    /// DeltaNet input channels [q | k | v] of this rank.
    fn channels(&self, c: &Config) -> Ranges {
        let (first, count) = self.part(c.k_heads);
        let mut r = vec![(first, count), (c.k_heads + first, count)];
        r.extend(self.value_heads(c).into_iter().map(|(h, n)| (2 * c.k_heads + h, n)));
        self.sel(&r, c.state)
    }
}

/// Rows and columns (multiples of 32) of a Q4_0 matrix in GGUF layout: the bytes, then (k, n).
fn cut_q4(data: &[u8], k: usize, n: usize, rows: &[(usize, usize)], cols: &[(usize, usize)]) -> (Vec<u8>, usize, usize) {
    let all_rows = [(0, n)];
    let all_cols = [(0, k)];
    let rows = if rows.is_empty() { &all_rows[..] } else { rows };
    let cols = if cols.is_empty() { &all_cols[..] } else { cols };
    let row_bytes = k / 32 * 18;
    let (kl, nl) = (cols.iter().map(|c| c.1).sum::<usize>(), rows.iter().map(|r| r.1).sum::<usize>());
    let mut out = Vec::with_capacity(nl * kl / 32 * 18);
    for &(r0, rn) in rows {
        for r in r0..r0 + rn {
            let row = &data[r * row_bytes..(r + 1) * row_bytes];
            for &(c0, cn) in cols {
                assert!(c0.is_multiple_of(32) && cn.is_multiple_of(32), "Q4_0 columns split on 32-weight blocks");
                out.extend_from_slice(&row[c0 / 32 * 18..(c0 + cn) / 32 * 18]);
            }
        }
    }
    (out, kl, nl)
}

/// Elements of a vector, or of every `width`-wide row, in the given ranges (empty: all).
fn pick(v: Vec<f32>, width: usize, ranges: &[(usize, usize)]) -> Vec<f32> {
    if ranges.is_empty() {
        return v;
    }
    ranges.iter().flat_map(|&(a, n)| v[a * width..(a + n) * width].iter().copied()).collect()
}

/// Gated attention: the trunk's attention layers and the MTP block.
struct Attn {
    q: Buffer,
    k: Buffer,
    v: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    out: Buffer,
    kc: Buffer,
    vc: Buffer,
}

struct Delta {
    qkv: Buffer,
    z: Buffer,
    /// alpha and beta projections, f32 rows `[alpha(heads) | beta(heads)]`.
    ab: Buffer,
    /// `[dt bias(heads) | A(heads)]`.
    dt_a: Buffer,
    conv_w: Buffer,
    conv: Buffer,
    state: Buffer,
    norm: Buffer,
    out: Buffer,
    /// State and convolution window after each column but the last of a verification batch.
    snaps: Option<(Buffer, Buffer)>,
}

enum Mixer {
    Delta(Delta),
    Attention(Attn),
}

struct Ffn {
    gate: Buffer,
    up: Buffer,
    down: Buffer,
}

struct Layer {
    il: usize,
    attn_norm: Buffer,
    post_norm: Buffer,
    ffn: Ffn,
    mixer: Mixer,
}

struct Head {
    norm: Buffer,
    w: Buffer,
    logits: Buffer,
    scratch: Buffer,
    top: Buffer,
}

/// The MTP block on the head stage; `cpu::CpuMtp` documents the math.
struct Mtp {
    enorm: Buffer,
    hnorm: Buffer,
    eh: Buffer,
    attn_norm: Buffer,
    post_norm: Buffer,
    head_norm: Buffer,
    attn: Attn,
    ffn: Ffn,
    /// Token embeddings of the batch, or of the draft token.
    emb: Buffer,
    /// Row 0: the hidden input of the next token. Rows 1..=n: the target's normed rows of the latest batch.
    hin: Buffer,
    /// `[e | h]` rows, the eh_proj input.
    cat: Buffer,
    /// Draft output, the hidden input of the next chained step.
    out: Buffer,
}

/// One command buffer per layer: each is a separate submission, which keeps every GPU job well inside
/// the kernel's lockup timeout during long prefill chunks.
struct Graph {
    gpu: Rc<Device>,
    cmds: Vec<vk::CommandBuffer>,
    captures: Vec<(String, Buffer, usize)>,
    stamps: Option<Stamps>,
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { self.gpu.dev.free_command_buffers(self.gpu.pool, &self.cmds) };
        if let Some(s) = &self.stamps {
            unsafe { self.gpu.dev.destroy_query_pool(s.pool, None) };
        }
    }
}

/// Profiling only (`TRASHCAN_TP_PROFILE`): a timestamp after every dispatch of a tensor-parallel graph, so each
/// kernel's GPU time is the time since the previous stamp. An empty label marks the start of a submission; the
/// time before it is the gap between submissions.
struct Stamps {
    pool: vk::QueryPool,
    labels: Vec<String>,
}
const STAMPS: u32 = 4096;

fn profiling() -> bool {
    std::env::var_os("TRASHCAN_TP_PROFILE").is_some()
}

pub enum Output {
    Hidden(Vec<f32>),
    /// Candidates per column: the last one, or every column of a verification batch.
    Top(Vec<Vec<(f32, u32)>>),
}

/// One recorded graph: batch, attention partitions, dump, verify, and for a micro-batch of a tensor-parallel batch
/// its params slot, first row in the batch buffers and whether it ends with the LM head.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
    n: usize,
    parts: u32,
    dump: bool,
    verify: bool,
    slot: usize,
    row: usize,
    head: bool,
}

impl Key {
    fn whole(n: usize, parts: u32, dump: bool, verify: bool) -> Self {
        Key { n, parts, dump, verify, slot: 0, row: 0, head: true }
    }
}

/// Micro-batches of a tensor-parallel batch have their own `pos, n` parameters, this many bytes apart.
const PARAMS_STRIDE: usize = 256;
/// Most micro-batches a tensor-parallel batch is split into.
pub const MAX_MICRO: usize = 4;

/// Split a batch of `n` columns into at most `count` micro-batches whose sizes are multiples of 64 (the GEMM tile)
/// except the last: (first row, rows) of each.
pub fn micro_batches(n: usize, count: usize) -> Vec<(usize, usize)> {
    let size = n.div_ceil(count.clamp(1, MAX_MICRO)).next_multiple_of(64).min(n.max(1));
    (0..n).step_by(size).map(|r| (r, size.min(n - r))).collect()
}

/// Tensor-parallel segments: the host buffers of the all-reduces and the timeline the host polls.
struct Tp {
    /// This rank's partial output projection, written by the GPU at the end of each segment.
    part: Buffer,
    /// The all-reduced total, written by the host before the next segment.
    total: Buffer,
    /// Segment `i` of the forward in progress signals `done` = `base + i + 1` when it finishes and, when
    /// submitted ahead, waits for the host to signal `release` = `base + i`.
    done: vk::Semaphore,
    release: vk::Semaphore,
    base: u64,
    /// Highest value signalled on `done` and on `release` so far.
    done_value: u64,
    released: u64,
    /// Graph of the forward in progress (for its captures), and the segments of the forward or draft step in
    /// progress with the vocabulary rows its head ranks.
    current: Option<Key>,
    /// Graphs of the micro-batches of the forward in progress.
    keys: Vec<Key>,
    cmds: Vec<vk::CommandBuffer>,
    head_rows: usize,
}

pub struct GpuStage {
    c: Config,
    last: usize,
    pub gpu: Rc<Device>,
    layers: Vec<Layer>,
    head: Option<Head>,
    mtp: Option<Mtp>,
    max_batch: usize,
    verify: usize,
    context: usize,
    input: Buffer,
    output: Buffer,
    params: Buffer,
    params_host: Buffer,
    x: Buffer,
    h: Buffer,
    y: Buffer,
    /// DeltaNet input projection or the query projection with its gates.
    big: Buffer,
    conv_out: Buffer,
    z: Buffer,
    ab: Buffer,
    o: Buffer,
    kb: Buffer,
    vb: Buffer,
    rq: Buffer,
    scratch: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    fence: vk::Fence,
    graphs: HashMap<Key, (Graph, u64)>,
    drafts: HashMap<(u32, bool, usize), Graph>,
    tick: u64,
    /// Columns of the latest forward, and whether it kept per-column snapshots.
    last_n: usize,
    verified: bool,
    rollbacks: HashMap<usize, vk::CommandBuffer>,
    slots: HashMap<usize, Buffer>,
    /// Seconds spent in submissions and their fence waits.
    pub busy: std::cell::Cell<f64>,
    tp: Option<Tp>,
    /// First vocabulary id of this rank's head rows.
    vocab_start: usize,
}

/// Timing only (`TRASHCAN_TP_VRAM_IO`, wrong results): tensor-parallel segments write their partials to and read
/// their totals from VRAM instead of host memory, to measure what the host-memory traffic costs the GPUs.
fn vram_io() -> bool {
    std::env::var_os("TRASHCAN_TP_VRAM_IO").is_some()
}

/// Partition count of split decode attention, by visible tokens; large batches use the unsplit kernel.
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

fn upload_q4(gpu: &Rc<Device>, g: &Gguf, c: &Config, il: usize, name: &str) -> Result<Buffer> {
    upload_q4_cut(gpu, g, c, il, name, &[], &[])
}

/// A Q4_0 matrix, cut to the given rows and columns (empty: all).
fn upload_q4_cut(gpu: &Rc<Device>, g: &Gguf, c: &Config, il: usize, name: &str, rows: &[(usize, usize)], cols: &[(usize, usize)]) -> Result<Buffer> {
    let t = c.layer_tensor(g, il, name)?;
    ensure!(
        t.ty == GgmlType::Q4_0,
        "{} is {}; use a pure Q4_0 quant (docs/qwen.md)",
        t.name,
        t.ty
    );
    let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
    if rows.is_empty() && cols.is_empty() {
        return gpu.upload_matrix(&quant::repack_q4(g.data(t), k, n, 0, n, 0, k));
    }
    let (data, k, n) = cut_q4(g.data(t), k, n, rows, cols);
    gpu.upload_matrix(&quant::repack_q4(&data, k, n, 0, n, 0, k))
}

fn upload_f32(gpu: &Rc<Device>, v: &[f32]) -> Result<Buffer> {
    gpu.upload(bytemuck::cast_slice(v))
}

fn upload_vector(gpu: &Rc<Device>, g: &Gguf, c: &Config, il: usize, name: &str) -> Result<Buffer> {
    upload_f32(gpu, &vector(g, c.layer_tensor(g, il, name)?)?)
}

/// A gated-attention block: this rank's query heads (`[q | gate]` rows) and KV heads, and the matching
/// columns of the output projection. `c` is the whole model's configuration.
fn load_attn(gpu: &Rc<Device>, g: &Gguf, c: &Config, il: usize, context: usize, split: Split) -> Result<Attn> {
    let (heads, kv_heads) = (split.part(c.heads), split.part(c.kv_heads));
    let kv = context * kv_heads.1 * c.head_dim * 2;
    let kv_rows = split.sel(&[kv_heads], c.head_dim);
    Ok(Attn {
        q: upload_q4_cut(gpu, g, c, il, "attn_q.weight", &split.sel(&[heads], 2 * c.head_dim), &[])?,
        k: upload_q4_cut(gpu, g, c, il, "attn_k.weight", &kv_rows, &[])?,
        v: upload_q4_cut(gpu, g, c, il, "attn_v.weight", &kv_rows, &[])?,
        q_norm: upload_vector(gpu, g, c, il, "attn_q_norm.weight")?,
        k_norm: upload_vector(gpu, g, c, il, "attn_k_norm.weight")?,
        out: upload_q4_cut(gpu, g, c, il, "attn_output.weight", &[], &split.sel(&[heads], c.head_dim))?,
        kc: gpu.buffer(kv, false, false, None)?,
        vc: gpu.buffer(kv, false, false, None)?,
    })
}

fn load_ffn(gpu: &Rc<Device>, g: &Gguf, c: &Config, il: usize, split: Split) -> Result<Ffn> {
    let ff = split.sel(&[split.part(c.ff)], 1);
    Ok(Ffn {
        gate: upload_q4_cut(gpu, g, c, il, "ffn_gate.weight", &ff, &[])?,
        up: upload_q4_cut(gpu, g, c, il, "ffn_up.weight", &ff, &[])?,
        down: upload_q4_cut(gpu, g, c, il, "ffn_down.weight", &[], &ff)?,
    })
}

/// Records one graph: a sequence of command buffers sharing the stage's buffers.
struct Rec<'a> {
    s: &'a GpuStage,
    n: usize,
    dump: bool,
    cmd: vk::CommandBuffer,
    graph: Graph,
    /// Where the mixers' and FFNs' output projections go: `y`, or the host partial under tensor parallelism.
    out: u64,
    /// Params slot, and first row in the batch buffers (`input`, `x`, the MTP embeddings, the partial and the
    /// total), of a micro-batch; 0 for a whole batch.
    slot: usize,
    row: usize,
    stamps: std::cell::RefCell<Option<Stamps>>,
}

impl<'a> Rec<'a> {
    fn new(s: &'a GpuStage, n: usize, dump: bool) -> Result<Self> {
        Self::at(s, n, dump, 0, 0)
    }

    /// A micro-batch of `n` rows starting at `row`, with its params in `slot`.
    fn at(s: &'a GpuStage, n: usize, dump: bool, slot: usize, row: usize) -> Result<Self> {
        let cmd = s.gpu.begin()?;
        let stamps = if s.tp.is_some() && profiling() {
            let ci = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(STAMPS);
            let pool = unsafe { s.gpu.dev.create_query_pool(&ci, None)? };
            unsafe { s.gpu.dev.cmd_reset_query_pool(cmd, pool, 0, STAMPS) };
            Some(Stamps { pool, labels: Vec::new() })
        } else {
            None
        };
        let rows = (row * s.c.d * 4) as u64;
        let r = Self {
            s,
            n,
            dump,
            cmd,
            graph: Graph {
                gpu: s.gpu.clone(),
                cmds: vec![cmd],
                captures: Vec::new(),
                stamps: None,
            },
            out: s.tp.as_ref().filter(|_| !vram_io()).map_or(s.y.address, |t| t.part.address) + rows,
            slot,
            row,
            stamps: std::cell::RefCell::new(stamps),
        };
        r.stamp_at("", vk::PipelineStageFlags::TOP_OF_PIPE);
        s.gpu.barrier(cmd);
        s.gpu.copy(cmd, &s.params_host, slot * PARAMS_STRIDE, &s.params, slot * PARAMS_STRIDE, 16);
        Ok(r)
    }

    /// Byte offset of this micro-batch's rows in the batch buffers.
    fn rows(&self) -> usize {
        self.row * self.s.c.d * 4
    }

    /// This micro-batch's rows of the residual.
    fn x(&self) -> u64 {
        self.s.x.address + self.rows() as u64
    }

    /// End a tensor-parallel segment: the host reads the partial projection after this submission, then
    /// writes the all-reduced total before submitting the next one.
    fn end_segment(&mut self) -> Result<()> {
        self.s.gpu.host_barrier(self.cmd, true);
        self.split()?;
        self.s.gpu.host_barrier(self.cmd, false);
        Ok(())
    }

    fn push(&self, ptr: [u64; 6], u: [u32; 16]) -> Push {
        Push {
            ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], self.s.params.address + (self.slot * PARAMS_STRIDE) as u64, 0],
            u,
        }
    }

    /// Start another submission; its leading barrier orders it after the previous one.
    fn split(&mut self) -> Result<()> {
        self.s.gpu.finish(self.cmd)?;
        self.cmd = self.s.gpu.begin()?;
        self.stamp_at("", vk::PipelineStageFlags::TOP_OF_PIPE);
        self.s.gpu.barrier(self.cmd);
        self.graph.cmds.push(self.cmd);
        Ok(())
    }

    /// Profiling: mark the end of the preceding dispatch as `label`.
    fn stamp(&self, label: &str) {
        self.stamp_at(label, vk::PipelineStageFlags::BOTTOM_OF_PIPE);
    }

    fn stamp_at(&self, label: &str, stage: vk::PipelineStageFlags) {
        if let Some(s) = self.stamps.borrow_mut().as_mut() {
            let i = s.labels.len() as u32;
            if i < STAMPS {
                unsafe { self.s.gpu.dev.cmd_write_timestamp(self.cmd, stage, s.pool, i) };
                s.labels.push(label.to_owned());
            }
        }
    }

    fn finish(mut self) -> Result<Graph> {
        self.s.gpu.finish(self.cmd)?;
        self.graph.stamps = self.stamps.into_inner();
        Ok(self.graph)
    }

    fn copy(&self, a: &Buffer, ao: usize, b: &Buffer, bo: usize, size: usize) {
        self.s.gpu.copy(self.cmd, a, ao, b, bo, size);
        self.stamp("copy");
    }

    /// Under tensor parallelism only the replicated residual `x` and norm output `h` are whole tensors.
    fn capture(&mut self, name: String, b: &Buffer, width: usize) -> Result<()> {
        let whole = self.s.tp.is_none() || b.buf == self.s.x.buf || b.buf == self.s.h.buf;
        if self.dump && whole {
            let host = self.s.gpu.buffer(width * 4, true, false, None)?;
            self.s.gpu.copy(self.cmd, b, (self.n - 1) * width * 4, &host, 0, width * 4);
            self.graph.captures.push((name, host, width));
        }
        Ok(())
    }

    /// Q4_0 projections reuse the Gemma kernels: GEMM for prefill, multi-column for small batches.
    fn matrix(&self, w: &Buffer, x: u64, y: u64, k: usize, rows: usize, defer: bool) -> Result<()> {
        let mut u = [0; 16];
        u[0] = rows as u32;
        u[1] = k as u32;
        u[3] = (rows * k / 32) as u32;
        let n = self.n;
        // Two to four columns use the decode kernels' formulation (qwen_q4c*): each column matches a
        // one-column decode bit for bit, as fast as multivec or slightly faster on Qwen shapes.
        let (shader, gx, gy) = match n {
            1 => (q4_decode_shader(rows, k), rows.div_ceil(16), 1),
            2 => ("qwen_q4c2", rows.div_ceil(16), 1),
            3 => ("qwen_q4c3", rows.div_ceil(16), 1),
            4 => ("qwen_q4c4", rows.div_ceil(16), 1),
            5..=8 => ("multivec4", rows.div_ceil(16), n.div_ceil(4)),
            _ => ("gemm", rows.div_ceil(64), n.div_ceil(64)),
        };
        self.s.gpu.dispatch_matrix(
            self.cmd,
            shader,
            self.push([0, x, y, 0, 0, 0], u).defer_barrier(defer),
            gx as u32,
            gy as u32,
            w,
        )?;
        self.stamp(&format!("{shader} {rows}x{k}"));
        Ok(())
    }

    /// Optional residual add x += y, then h = RMS(x) * w when h is given, its rows `stride` apart
    /// (0: dense). One workgroup per token.
    fn add_norm(&self, x: u64, y: Option<u64>, h: Option<u64>, w: Option<&Buffer>, stride: usize) {
        let mut u = [0; 16];
        u[0] = self.s.c.d as u32;
        u[3] = stride as u32;
        u[11] = u32::from(y.is_some());
        u[12] = self.s.c.eps.to_bits();
        self.s.gpu.dispatch(
            self.cmd,
            "qwen_add_norm",
            self.push([x, y.unwrap_or(0), h.unwrap_or(0), w.map_or(0, |b| b.address), 0, 0], u),
            1,
            self.n as u32,
        );
        self.stamp(if y.is_some() { "add_norm (+total)" } else { "add_norm" });
    }

    /// Gated DeltaNet from `h` into `y`.
    fn delta(&mut self, dl: &Delta, il: usize, verify: bool) -> Result<()> {
        let (s, c) = (self.s, &self.s.c);
        let (d, cd, vd) = (c.d, c.conv_dim(), c.value_dim());
        let eps = c.eps.to_bits();
        self.matrix(&dl.qkv, s.h.address, s.big.address, d, cd, true)?;
        self.matrix(&dl.z, s.h.address, s.z.address, d, vd, true)?;
        let mut u = [0; 16];
        u[0] = (2 * c.v_heads) as u32;
        u[1] = d as u32;
        s.gpu.dispatch_matrix(
            self.cmd,
            "qwen_matf32",
            self.push([0, s.h.address, s.ab.address, 0, 0, 0], u),
            (2 * c.v_heads).div_ceil(4) as u32,
            self.n.div_ceil(8) as u32,
            &dl.ab,
        )?;
        self.stamp("matf32");
        self.capture(format!("linear_attn_qkv_mixed-{il}"), &s.big, cd)?;
        self.capture(format!("z-{il}"), &s.z, vd)?;
        let (snap, snap_conv) = match (&dl.snaps, verify) {
            (Some((a, b)), true) => (a.address, b.address),
            _ => (0, 0),
        };
        let mut u = [0; 16];
        u[0] = cd as u32;
        s.gpu.dispatch(
            self.cmd,
            "qwen_conv",
            self.push([s.big.address, dl.conv_w.address, dl.conv.address, s.conv_out.address, snap_conv, 0], u),
            cd.div_ceil(256) as u32,
            1,
        );
        self.stamp("conv");
        self.capture(format!("conv_output_silu-{il}"), &s.conv_out, cd)?;
        let mut u = [0; 16];
        u[0] = cd as u32;
        u[4] = c.v_heads as u32;
        u[5] = c.k_heads as u32;
        u[6] = c.state as u32;
        u[12] = eps;
        s.gpu.dispatch(
            self.cmd,
            "qwen_gdn",
            self.push([s.conv_out.address, s.ab.address, dl.state.address, s.o.address, dl.dt_a.address, snap], u),
            (c.v_heads * c.state / 16) as u32,
            1,
        );
        self.stamp("gdn");
        self.capture(format!("attn_output-{il}"), &s.o, vd)?;
        let mut u = [0; 16];
        u[0] = vd as u32;
        u[6] = c.state as u32;
        u[12] = eps;
        s.gpu.dispatch(
            self.cmd,
            "qwen_gated_norm",
            self.push([s.o.address, s.z.address, dl.norm.address, 0, 0, 0], u),
            c.v_heads as u32,
            self.n as u32,
        );
        self.stamp("gated_norm");
        self.capture(format!("final_output-{il}"), &s.o, vd)?;
        self.matrix(&dl.out, s.o.address, self.out, vd, d, false)?;
        self.capture(format!("linear_attn_out-{il}"), &s.y, d)
    }

    /// Gated attention from `h` into `y`, caching K and V at the batch positions.
    fn attention(&mut self, a: &Attn, parts: u32, il: usize) -> Result<()> {
        let (s, c) = (self.s, &self.s.c);
        let (d, hd, kvd) = (c.d, c.head_dim, c.kv_heads * c.head_dim);
        let qd = c.heads * hd;
        self.matrix(&a.q, s.h.address, s.big.address, d, 2 * qd, true)?;
        self.matrix(&a.k, s.h.address, s.kb.address, d, kvd, true)?;
        self.matrix(&a.v, s.h.address, s.vb.address, d, kvd, false)?;
        // Q (normed, rotated, pre-scaled into rq), K (normed, rotated into the cache), V (cache).
        for (src, norm, out, cache, heads, stride, width, scale, defer) in [
            (&s.big, Some(&a.q_norm), Some(&s.rq), None, c.heads, 2 * hd, 2 * qd, 1. / (hd as f32).sqrt(), true),
            (&s.kb, Some(&a.k_norm), None, Some(&a.kc), c.kv_heads, hd, kvd, 1., true),
            (&s.vb, None, None, Some(&a.vc), c.kv_heads, hd, kvd, 1f32, false),
        ] {
            let mut u = [0; 16];
            u[1] = if norm.is_some() { c.rot as u32 } else { 0 };
            u[2] = width as u32;
            u[3] = stride as u32;
            u[4] = heads as u32;
            u[6] = hd as u32;
            u[7] = s.context as u32;
            u[8] = c.eps.to_bits();
            u[12] = c.rope_base.to_bits();
            u[13] = scale.to_bits();
            s.gpu.dispatch(
                self.cmd,
                "qwen_qk",
                self.push(
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
                self.n as u32,
            );
            self.stamp("qk");
        }
        self.capture(format!("Qcur-{il}"), &s.rq, qd)?;
        let mut u = [0; 16];
        u[4] = c.heads as u32;
        u[5] = c.kv_heads as u32;
        u[6] = hd as u32;
        u[7] = s.context as u32;
        if parts == 0 {
            s.gpu.dispatch(
                self.cmd,
                "attention",
                self.push([s.rq.address, a.kc.address, a.vc.address, s.o.address, 0, 0], u),
                c.heads as u32,
                self.n as u32,
            );
        } else {
            let (partial, merge) = attention_shaders(parts)?;
            s.gpu.dispatch(
                self.cmd,
                partial,
                self.push([s.rq.address, a.kc.address, a.vc.address, s.o.address, s.scratch.address, 0], u),
                c.heads as u32 * parts,
                self.n as u32,
            );
            s.gpu.dispatch(
                self.cmd,
                merge,
                self.push([0, 0, 0, s.o.address, s.scratch.address, 0], u),
                c.heads as u32,
                self.n as u32,
            );
        }
        self.stamp("attention");
        self.capture(format!("attn_pregate-{il}"), &s.o, qd)?;
        let mut u = [0; 16];
        u[0] = qd as u32;
        u[6] = hd as u32;
        s.gpu.dispatch(
            self.cmd,
            "qwen_gate",
            self.push([s.o.address, s.big.address, 0, 0, 0, 0], u),
            qd.div_ceil(256) as u32,
            self.n as u32,
        );
        self.stamp("gate");
        self.capture(format!("attn_gated-{il}"), &s.o, qd)?;
        self.matrix(&a.out, s.o.address, self.out, qd, d, false)?;
        self.capture(format!("attn_output-{il}"), &s.y, d)
    }

    /// SwiGLU FFN from `h` into `y`.
    fn ffn(&mut self, f: &Ffn, il: usize) -> Result<()> {
        let (s, c) = (self.s, &self.s.c);
        self.matrix(&f.gate, s.h.address, s.gate.address, c.d, c.ff, true)?;
        self.matrix(&f.up, s.h.address, s.up.address, c.d, c.ff, false)?;
        let mut u = [0; 16];
        u[0] = c.ff as u32;
        s.gpu.dispatch(
            self.cmd,
            "qwen_swiglu",
            self.push([s.gate.address, s.up.address, s.act.address, 0, 0, 0], u),
            c.ff.div_ceil(256) as u32,
            self.n as u32,
        );
        self.stamp("swiglu");
        self.capture(format!("ffn_swiglu-{il}"), &s.act, c.ff)?;
        self.matrix(&f.down, s.act.address, self.out, c.ff, c.d, false)
    }

    /// LM head and top candidates of the last `cols` normed rows of `input`, over the first `rows` tokens
    /// of the vocabulary.
    fn head(&mut self, hd: &Head, input: &Buffer, cols: usize, rows: usize) -> Result<()> {
        let (s, c) = (self.s, &self.s.c);
        let (d, n) = (c.d, self.n);
        let chunks = rows.div_ceil(1024);
        let mut u = [0; 16];
        u[0] = rows as u32;
        u[1] = d as u32;
        u[3] = (c.vocab * d / 16) as u32;
        u[9] = 1;
        let single = u32::from(cols == 1);
        u[15] = single;
        let (shader, gy, x) = match cols {
            1 => ("headvec", 1, input.address + ((n - 1) * d * 4) as u64),
            2 => ("multivec4x2", 1, input.address),
            3 => ("multivec4x3", 1, input.address),
            _ => ("multivec4", cols.div_ceil(4), input.address),
        };
        ensure!(cols == 1 || cols == n, "the head ranks the last column or all of them");
        s.gpu.dispatch_matrix(
            self.cmd,
            shader,
            self.push([0, x, hd.logits.address, 0, 0, 0], u),
            rows.div_ceil(16) as u32,
            gy as u32,
            &hd.w,
        )?;
        self.stamp("head");
        if self.dump {
            // A tensor-parallel rank holds a slice of the vocabulary; tp::TpGroup joins the slices.
            let name = if s.tp.is_some() { "result_output_part" } else { "result_output" };
            let host = s.gpu.buffer(rows * 4, true, false, None)?;
            s.gpu.copy(self.cmd, &hd.logits, (cols - 1) * rows * 4, &host, 0, rows * 4);
            self.graph.captures.push((name.into(), host, rows));
        }
        let mut u = [0; 16];
        u[0] = rows as u32;
        u[1] = TOP_K as u32;
        u[2] = (chunks * TOP_K) as u32;
        u[15] = single;
        s.gpu.dispatch(
            self.cmd,
            "topk",
            self.push([hd.logits.address, 0, hd.scratch.address, 0, 0, 0], u),
            chunks as u32,
            cols as u32,
        );
        let mut u = [0; 16];
        u[0] = (chunks * TOP_K) as u32;
        u[1] = TOP_K as u32;
        u[15] = single;
        s.gpu.dispatch(
            self.cmd,
            "merge",
            self.push([hd.scratch.address, 0, hd.top.address, 0, 0, 0], u),
            1,
            cols as u32,
        );
        self.stamp("topk");
        Ok(())
    }

    /// The MTP block from the rows of `cat` into the residual `x`, without its final norm.
    fn mtp_block(&mut self, m: &Mtp, parts: u32) -> Result<()> {
        let s = self.s;
        let d = s.c.d;
        self.matrix(&m.eh, m.cat.address, s.x.address, 2 * d, d, false)?;
        self.add_norm(s.x.address, None, Some(s.h.address), Some(&m.attn_norm), 0);
        self.attention(&m.attn, parts, s.c.layers)?;
        self.add_norm(s.x.address, Some(s.y.address), Some(s.h.address), Some(&m.post_norm), 0);
        self.ffn(&m.ffn, s.c.layers)
    }

    /// `[RMS(emb) * enorm | RMS(hidden) * hnorm]` into `cat`; the embeddings pass through `x`.
    fn mtp_inputs(&mut self, m: &Mtp, hidden: u64) {
        let s = self.s;
        let d = s.c.d;
        self.copy(&m.emb, self.rows(), &s.x, self.rows(), self.n * d * 4);
        self.add_norm(self.x(), None, Some(m.cat.address), Some(&m.enorm), 2 * d);
        self.add_norm(hidden, None, Some(m.cat.address + (d * 4) as u64), Some(&m.hnorm), 2 * d);
    }
}

impl GpuStage {
    /// Layers `first..last`; `head` adds the final norm and LM head, `mtp` (a GGUF holding the nextn
    /// layer) the MTP block. `verify` is the largest verification batch that can be rolled back.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: Rc<Device>,
        g: &Gguf,
        c: &Config,
        first: usize,
        last: usize,
        head: bool,
        context: usize,
        max_batch: usize,
        verify: usize,
        mtp: Option<&Gguf>,
    ) -> Result<Self> {
        Self::build(gpu, g, c, first, last, head, context, max_batch, verify, mtp, Split::WHOLE, false)
    }

    /// One rank of a tensor-parallel group: its share of every layer, of the LM head and, with `mtp`, of the
    /// MTP block. Forward passes and draft steps go through `tp_prepare` or `tp_draft_prepare`, then
    /// `tp_submit_all`, `tp_release` and `tp_wait` per segment (`tp::TpGroup`).
    #[allow(clippy::too_many_arguments)]
    pub fn shard(gpu: Rc<Device>, g: &Gguf, c: &Config, context: usize, max_batch: usize, verify: usize, split: Split, mtp: Option<&Gguf>) -> Result<Self> {
        Self::build(gpu, g, c, 0, c.layers, true, context, max_batch, verify, mtp, split, true)
    }

    /// `gc` is the whole model's configuration; the stage runs with `split`'s share of it.
    #[allow(clippy::too_many_arguments)]
    fn build(
        gpu: Rc<Device>,
        g: &Gguf,
        gc: &Config,
        first: usize,
        last: usize,
        head: bool,
        context: usize,
        max_batch: usize,
        verify: usize,
        mtp: Option<&Gguf>,
        split: Split,
        tp: bool,
    ) -> Result<Self> {
        let local = split.local(gc)?;
        let c = &local;
        ensure!(c.state == 128, "DeltaNet kernels need 128-wide heads");
        ensure!(c.head_dim <= 512 && c.head_dim % 64 == 0, "unsupported head size");
        ensure!(verify <= 8, "verification batches are limited to eight columns");
        ensure!(mtp.is_none() || head, "the MTP block belongs on the head stage");
        let snaps = verify.saturating_sub(1);
        let (channels, value_heads) = (split.channels(gc), split.sel(&split.value_heads(gc), 1));
        let value_dims = split.sel(&split.value_heads(gc), gc.state);
        let mut layers = Vec::new();
        for il in first..last {
            let mixer = if c.recurrent(il) {
                let mut ab = pick(vector(g, c.layer_tensor(g, il, "ssm_alpha.weight")?)?, gc.d, &value_heads);
                ab.extend(pick(vector(g, c.layer_tensor(g, il, "ssm_beta.weight")?)?, gc.d, &value_heads));
                let mut dt_a = pick(vector(g, c.layer_tensor(g, il, "ssm_dt.bias")?)?, 1, &value_heads);
                dt_a.extend(pick(vector(g, c.layer_tensor(g, il, "ssm_a")?)?, 1, &value_heads));
                let conv_w = pick(vector(g, c.layer_tensor(g, il, "ssm_conv1d.weight")?)?, gc.conv, &channels);
                let (state, conv) = (c.v_heads * c.state * c.state * 4, (c.conv - 1) * c.conv_dim() * 4);
                Mixer::Delta(Delta {
                    qkv: upload_q4_cut(&gpu, g, c, il, "attn_qkv.weight", &channels, &[])?,
                    z: upload_q4_cut(&gpu, g, c, il, "attn_gate.weight", &value_dims, &[])?,
                    ab: gpu.upload_matrix(bytemuck::cast_slice(&ab))?,
                    dt_a: upload_f32(&gpu, &dt_a)?,
                    conv_w: upload_f32(&gpu, &conv_w)?,
                    conv: gpu.buffer(conv, false, false, None)?,
                    state: gpu.buffer(state, false, false, None)?,
                    norm: upload_vector(&gpu, g, c, il, "ssm_norm.weight")?,
                    out: upload_q4_cut(&gpu, g, c, il, "ssm_out.weight", &[], &value_dims)?,
                    snaps: if snaps > 0 {
                        Some((gpu.buffer(snaps * state, false, false, None)?, gpu.buffer(snaps * conv, false, false, None)?))
                    } else {
                        None
                    },
                })
            } else {
                Mixer::Attention(load_attn(&gpu, g, gc, il, context, split)?)
            };
            layers.push(Layer {
                il,
                attn_norm: upload_vector(&gpu, g, c, il, "attn_norm.weight")?,
                post_norm: upload_vector(&gpu, g, c, il, "post_attention_norm.weight")?,
                ffn: load_ffn(&gpu, g, gc, il, split)?,
                mixer,
            });
        }
        let cols = verify.max(1);
        let (vocab_start, vocab_rows) = split.part(gc.vocab);
        let head = if head {
            let t = g.tensor("output.weight").context("missing output.weight")?;
            ensure!(t.ty == GgmlType::Q6K, "output.weight must be Q6_K, found {}", t.ty);
            let chunks = c.vocab.div_ceil(1024);
            ensure!(chunks <= 256, "vocabulary too large for one candidate merge");
            let row = gc.d / 256 * 210;
            Some(Head {
                norm: upload_f32(&gpu, &vector(g, g.tensor("output_norm.weight").context("missing output norm")?)?)?,
                w: gpu.upload_matrix(&quant::repack_q6(&g.data(t)[vocab_start * row..(vocab_start + vocab_rows) * row]))?,
                logits: gpu.buffer(cols * c.vocab * 4, false, false, None)?,
                scratch: gpu.buffer(cols * chunks * TOP_K * 8, false, false, None)?,
                top: gpu.buffer(cols * TOP_K * 8, true, false, None)?,
            })
        } else {
            None
        };
        let mtp = match mtp {
            Some(m) => {
                let il = super::mtp_layer(m, gc)?;
                let v = |name: &str| upload_vector(&gpu, m, c, il, name);
                Some(Mtp {
                    enorm: v("nextn.enorm.weight")?,
                    hnorm: v("nextn.hnorm.weight")?,
                    eh: upload_q4(&gpu, m, c, il, "nextn.eh_proj.weight")?,
                    attn_norm: v("attn_norm.weight")?,
                    post_norm: v("post_attention_norm.weight")?,
                    head_norm: v("nextn.shared_head_norm.weight")?,
                    attn: load_attn(&gpu, m, gc, il, context, split)?,
                    ffn: load_ffn(&gpu, m, gc, il, split)?,
                    emb: gpu.buffer(c.d * max_batch * 4, true, false, None)?,
                    hin: gpu.buffer(c.d * (max_batch + 1) * 4, false, false, None)?,
                    cat: gpu.buffer(2 * c.d * max_batch * 4, false, false, None)?,
                    out: gpu.buffer(c.d * 4, false, false, None)?,
                })
            }
            None => None,
        };
        let alloc = |w: usize| gpu.buffer(w * max_batch * 4, false, false, None);
        let fence = unsafe { gpu.dev.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let qd = c.heads * c.head_dim;
        Ok(Self {
            last,
            layers,
            head,
            mtp,
            max_batch,
            verify,
            context,
            input: gpu.buffer(c.d * max_batch * 4, true, false, None)?,
            output: gpu.buffer(c.d * max_batch * 4, true, false, None)?,
            params: gpu.buffer(PARAMS_STRIDE * MAX_MICRO, false, false, None)?,
            params_host: gpu.buffer(PARAMS_STRIDE * MAX_MICRO, true, false, None)?,
            x: alloc(c.d)?,
            h: alloc(c.d)?,
            y: alloc(c.d)?,
            big: alloc(c.conv_dim().max(2 * qd))?,
            conv_out: alloc(c.conv_dim())?,
            z: alloc(c.value_dim())?,
            ab: alloc(2 * c.v_heads)?,
            o: alloc(c.value_dim().max(qd))?,
            kb: alloc(c.kv_heads * c.head_dim)?,
            vb: alloc(c.kv_heads * c.head_dim)?,
            rq: alloc(qd)?,
            scratch: gpu.buffer(c.heads * 64 * (c.head_dim + 2) * 4 * 8, false, false, None)?,
            gate: alloc(c.ff)?,
            up: alloc(c.ff)?,
            act: alloc(c.ff)?,
            fence,
            graphs: HashMap::new(),
            drafts: HashMap::new(),
            tick: 0,
            last_n: 0,
            verified: false,
            rollbacks: HashMap::new(),
            slots: HashMap::new(),
            busy: std::cell::Cell::new(0.),
            tp: if tp {
                Some(Tp {
                    part: gpu.buffer(c.d * max_batch * 4, true, false, None)?,
                    total: gpu.buffer(c.d * max_batch * 4, true, false, None)?,
                    done: gpu.timeline(false)?,
                    release: gpu.timeline(false)?,
                    base: 0,
                    done_value: 0,
                    released: 0,
                    current: None,
                    keys: Vec::new(),
                    cmds: Vec::new(),
                    head_rows: 0,
                })
            } else {
                None
            },
            vocab_start,
            c: c.clone(),
            gpu,
        })
    }

    pub fn device_mib(&self) -> f64 {
        self.gpu.device_bytes.get() as f64 / 1048576.
    }

    fn record(&self, k: Key) -> Result<Graph> {
        if self.tp.is_some() {
            return self.record_tp(k);
        }
        let (n, parts, dump, verify) = (k.n, k.parts, k.dump, k.verify);
        let d = self.c.d;
        let mut r = Rec::new(self, n, dump)?;
        r.copy(&self.input, 0, &self.x, 0, n * d * 4);
        let mut pending = false;
        for l in &self.layers {
            let il = l.il;
            r.add_norm(self.x.address, pending.then_some(self.y.address), Some(self.h.address), Some(&l.attn_norm), 0);
            if pending {
                r.capture(format!("l_out-{}", il - 1), &self.x, d)?;
            }
            r.capture(format!("attn_norm-{il}"), &self.h, d)?;
            match &l.mixer {
                Mixer::Delta(dl) => r.delta(dl, il, verify)?,
                Mixer::Attention(a) => r.attention(a, parts, il)?,
            }
            r.add_norm(self.x.address, Some(self.y.address), Some(self.h.address), Some(&l.post_norm), 0);
            r.capture(format!("attn_residual-{il}"), &self.x, d)?;
            r.capture(format!("attn_post_norm-{il}"), &self.h, d)?;
            r.ffn(&l.ffn, il)?;
            r.capture(format!("ffn_out-{il}"), &self.y, d)?;
            pending = true;
            if l.il + 1 < self.last {
                r.split()?;
            }
        }
        let last = self.last - 1;
        if let Some(hd) = &self.head {
            r.add_norm(self.x.address, pending.then_some(self.y.address), Some(self.h.address), Some(&hd.norm), 0);
            r.capture(format!("l_out-{last}"), &self.x, d)?;
            r.capture("result_norm".into(), &self.h, d)?;
            r.head(hd, &self.h, if verify { n } else { 1 }, self.c.vocab)?;
            if let Some(m) = &self.mtp {
                // Extend the MTP cache over this batch: token p pairs with the target's row p - 1.
                r.split()?;
                r.copy(&self.h, 0, &m.hin, d * 4, n * d * 4);
                r.mtp_inputs(m, m.hin.address);
                r.mtp_block(m, parts)?;
                r.copy(&m.hin, n * d * 4, &m.hin, 0, d * 4);
            }
        } else {
            if pending {
                r.add_norm(self.x.address, Some(self.y.address), None, None, 0);
            }
            r.capture(format!("l_out-{last}"), &self.x, d)?;
            r.copy(&self.x, 0, &self.output, 0, n * d * 4);
        }
        r.finish()
    }

    /// Tensor-parallel graph: every mixer and FFN ends a submission with its partial projection in host
    /// memory, and the next submission adds the all-reduced total to the replicated residual. The last
    /// submission ranks this rank's vocabulary rows (unless `k` is a micro-batch before the last). 2 * layers + 1
    /// submissions. A micro-batch works on its own rows of the batch buffers, so the segments of several
    /// micro-batches can alternate on the queue (`tp_prepare`).
    fn record_tp(&self, k: Key) -> Result<Graph> {
        let (n, parts, dump, verify) = (k.n, k.parts, k.dump, k.verify);
        let d = self.c.d;
        let (hd, tp) = (self.head.as_ref().context("a tensor-parallel rank needs its head rows")?, self.tp.as_ref().unwrap());
        let mut r = Rec::at(self, n, dump, k.slot, k.row)?;
        let (rows, x) = (r.rows(), r.x());
        let total = (if vram_io() { self.y.address } else { tp.total.address }) + rows as u64;
        r.copy(&self.input, rows, &self.x, rows, n * d * 4);
        let mut added = None;
        for l in &self.layers {
            let il = l.il;
            r.add_norm(x, added, Some(self.h.address), Some(&l.attn_norm), 0);
            if added.is_some() {
                r.capture(format!("l_out-{}", il - 1), &self.x, d)?;
            }
            r.capture(format!("attn_norm-{il}"), &self.h, d)?;
            match &l.mixer {
                Mixer::Delta(dl) => r.delta(dl, il, verify)?,
                Mixer::Attention(a) => r.attention(a, parts, il)?,
            }
            r.end_segment()?;
            r.add_norm(x, Some(total), Some(self.h.address), Some(&l.post_norm), 0);
            r.capture(format!("attn_residual-{il}"), &self.x, d)?;
            r.capture(format!("attn_post_norm-{il}"), &self.h, d)?;
            r.ffn(&l.ffn, il)?;
            r.end_segment()?;
            added = Some(total);
        }
        r.add_norm(x, added, Some(self.h.address), Some(&hd.norm), 0);
        r.capture(format!("l_out-{}", self.last - 1), &self.x, d)?;
        r.capture("result_norm".into(), &self.h, d)?;
        if k.head {
            r.head(hd, &self.h, if verify { n } else { 1 }, self.c.vocab)?;
        }
        if let Some(m) = &self.mtp {
            // Extend the MTP layer's KV cache over this batch (token p pairs with the target's row p - 1, which for
            // a later micro-batch the previous one left in row 0). Only its K and V are kept, so the block stops
            // after the attention and needs no all-reduce.
            r.copy(&self.h, 0, &m.hin, d * 4, n * d * 4);
            r.mtp_inputs(m, m.hin.address);
            r.matrix(&m.eh, m.cat.address, x, 2 * d, d, false)?;
            r.add_norm(x, None, Some(self.h.address), Some(&m.attn_norm), 0);
            r.attention(&m.attn, parts, self.c.layers)?;
            r.copy(&m.hin, n * d * 4, &m.hin, 0, d * 4);
        }
        r.finish()
    }

    /// Tensor-parallel MTP draft step: the attention and the FFN each end a segment with their partials, then
    /// the head ranks this rank's share of the first `rows` vocabulary ids (none when `rows` is 0).
    fn record_draft_tp(&self, parts: u32, first: bool, rows: usize) -> Result<Graph> {
        let (m, hd) = (self.mtp.as_ref().context("no MTP block")?, self.head.as_ref().context("no head")?);
        let (d, total) = (self.c.d, self.tp.as_ref().unwrap().total.address);
        let mut r = Rec::new(self, 1, false)?;
        r.mtp_inputs(m, if first { m.hin.address } else { m.out.address });
        r.matrix(&m.eh, m.cat.address, self.x.address, 2 * d, d, false)?;
        r.add_norm(self.x.address, None, Some(self.h.address), Some(&m.attn_norm), 0);
        r.attention(&m.attn, parts, self.c.layers)?;
        r.end_segment()?;
        r.add_norm(self.x.address, Some(total), Some(self.h.address), Some(&m.post_norm), 0);
        r.ffn(&m.ffn, self.c.layers)?;
        r.end_segment()?;
        r.add_norm(self.x.address, Some(total), Some(m.out.address), Some(&m.head_norm), 0);
        if rows > 0 {
            r.head(hd, &m.out, 1, rows)?;
        }
        r.finish()
    }

    /// Start a tensor-parallel forward of `n` tokens at `pos` (see `forward`) as the micro-batches `micro`
    /// ((first row, rows) covering the batch in order; one for the whole batch): write the input rows and find
    /// or record each micro-batch's graph. Their segments alternate: segment `j` is segment `j / m` of
    /// micro-batch `j % m`, so the host can all-reduce one micro-batch while the GPU runs the next. Returns the
    /// number of segments; the caller releases them in order, writing each all-reduced total before the
    /// segment of the same micro-batch that reads it.
    pub fn tp_prepare(&mut self, x: &[f32], pos: usize, n: usize, verify: bool, dump: bool, micro: &[(usize, usize)]) -> Result<usize> {
        let d = self.c.d;
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(pos + n <= self.context, "context exhausted");
        ensure!(x.len() == n * d, "hidden state length mismatch");
        let verify = verify && n > 1;
        ensure!(!verify || n <= self.verify, "verification batch exceeds the configured {} columns", self.verify);
        ensure!(!micro.is_empty() && micro.len() <= MAX_MICRO, "1 to {MAX_MICRO} micro-batches");
        ensure!(micro.len() == 1 || !(verify || dump), "verification and dumps run as one batch");
        ensure!(
            micro.iter().scan(0, |next, &(row, rows)| {
                let ok = row == *next && rows > 0;
                *next = row + rows;
                Some(ok)
            }).all(|ok| ok) && micro.last().is_some_and(|&(row, rows)| row + rows == n),
            "micro-batches must cover the batch in order"
        );
        self.input.write(bytemuck::cast_slice(x))?;
        if let Some(m) = &self.mtp {
            // The input rows are the batch's token embeddings, which the MTP block also reads.
            m.emb.write(bytemuck::cast_slice(x))?;
            if pos == 0 {
                let cmd = self.gpu.begin()?;
                unsafe { self.gpu.dev.cmd_fill_buffer(cmd, m.hin.buf, 0, (d * 4) as u64, 0) };
                self.gpu.finish(cmd)?;
                self.gpu.run(cmd)?;
            }
        }
        let mut keys = Vec::new();
        for (slot, &(row, rows)) in micro.iter().enumerate() {
            self.params_host.write_at(slot * PARAMS_STRIDE, bytemuck::cast_slice(&[(pos + row) as u32, rows as u32, 0, 0]))?;
            let key = Key { n: rows, parts: partitions(pos + row, rows), dump, verify, slot, row, head: slot + 1 == micro.len() };
            self.tick += 1;
            if let Some(e) = self.graphs.get_mut(&key) {
                e.1 = self.tick;
            } else {
                // The graphs of this forward were used last, so they are never the victim.
                if self.graphs.len() >= GRAPH_CACHE_CAPACITY {
                    let victim = *self.graphs.iter().min_by_key(|e| e.1 .1).unwrap().0;
                    self.graphs.remove(&victim);
                }
                let graph = self.record(key)?;
                self.graphs.insert(key, (graph, self.tick));
            }
            keys.push(key);
        }
        self.last_n = n;
        self.verified = verify;
        let lists: Vec<&[vk::CommandBuffer]> = keys.iter().map(|k| self.graphs[k].0.cmds.as_slice()).collect();
        let segments = lists[0].len();
        ensure!(lists.iter().all(|l| l.len() == segments), "micro-batch graphs differ in segments");
        let cmds: Vec<vk::CommandBuffer> = (0..segments).flat_map(|i| lists.iter().map(move |l| l[i])).collect();
        let rows = self.c.vocab;
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        tp.current = keys.last().copied();
        tp.keys = keys;
        Ok(tp.start(cmds, rows))
    }

    /// Start a tensor-parallel MTP draft step at `pos` for the token with embedding `emb` (see `draft`), the
    /// head ranking the first `rows` vocabulary ids (0: all). Returns the number of segments.
    pub fn tp_draft_prepare(&mut self, emb: &[f32], pos: usize, first: bool, rows: usize) -> Result<usize> {
        let m = self.mtp.as_ref().context("this rank has no MTP block")?;
        ensure!(pos < self.context, "context exhausted");
        m.emb.write(bytemuck::cast_slice(emb))?;
        self.params_host.write(bytemuck::cast_slice(&[pos as u32, 1, 0, 0]))?;
        let rows = if rows == 0 { usize::MAX } else { rows };
        let local = rows.saturating_sub(self.vocab_start).min(self.c.vocab);
        let key = (partitions(pos, 1), first, local);
        if !self.drafts.contains_key(&key) {
            let graph = self.record_draft_tp(key.0, key.1, key.2)?;
            self.drafts.insert(key, graph);
        }
        let cmds = self.drafts[&key].cmds.clone();
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        Ok(tp.start(cmds, local))
    }

    /// Submit segment `i` of the forward in progress, to run as soon as the GPU is free.
    pub fn tp_submit(&mut self, i: usize) -> Result<()> {
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        tp.done_value = tp.base + i as u64 + 1;
        self.gpu.submit(tp.cmds[i], None, Some((tp.done, tp.done_value)), vk::Fence::null())
    }

    /// Submit every segment of the forward in progress ahead, each waiting for `tp_release`.
    pub fn tp_submit_all(&mut self) -> Result<()> {
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        for (i, &cmd) in tp.cmds.iter().enumerate() {
            let at = tp.base + i as u64;
            self.gpu.submit(cmd, Some((tp.release, at)), Some((tp.done, at + 1)), vk::Fence::null())?;
        }
        tp.done_value = tp.base + tp.cmds.len() as u64;
        Ok(())
    }

    /// Let segment `i` (submitted ahead) and everything before it run.
    pub fn tp_release(&mut self, i: usize) -> Result<()> {
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        let value = tp.base + i as u64;
        if value > tp.released {
            tp.released = value;
            unsafe { self.gpu.dev.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(tp.release).value(value))? };
        }
        Ok(())
    }

    /// Release every segment submitted ahead, e.g. after a failed exchange, so the queue drains.
    pub fn tp_release_all(&mut self) -> Result<()> {
        let tp = self.tp.as_ref().context("not a tensor-parallel rank")?;
        let last = tp.done_value.saturating_sub(tp.base + 1) as usize;
        self.tp_release(last)
    }

    /// Spin until segment `i` of the forward in progress has finished.
    pub fn tp_wait(&self, i: usize) -> Result<()> {
        let tp = self.tp.as_ref().context("not a tensor-parallel rank")?;
        let value = tp.base + i as u64 + 1;
        let start = std::time::Instant::now();
        while unsafe { self.gpu.dev.get_semaphore_counter_value(tp.done)? } < value {
            ensure!(start.elapsed().as_secs() < 60, "GPU segment did not finish");
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// This rank's partial projection of the `n` tokens from batch row `row` (a micro-batch's rows) of the
    /// latest segment.
    pub fn tp_part(&self, row: usize, n: usize) -> &[f32] {
        let (part, d) = (&self.tp.as_ref().unwrap().part, self.c.d);
        assert!((row + n) * d * 4 <= part.size);
        unsafe { std::slice::from_raw_parts((part.ptr as *const f32).add(row * d), n * d) }
    }

    /// Where the host writes the all-reduced total of the `n` rows from `row` for the next segment.
    pub fn tp_total(&self, row: usize, n: usize) -> *mut f32 {
        let (t, d) = (&self.tp.as_ref().unwrap().total, self.c.d);
        assert!((row + n) * d * 4 <= t.size);
        unsafe { (t.ptr as *mut f32).add(row * d) }
    }

    /// Top candidates of this rank's vocabulary rows after the last segment, with model token ids.
    pub fn tp_top(&self, cols: usize) -> Result<Vec<Vec<(f32, u32)>>> {
        if self.tp.as_ref().is_some_and(|t| t.head_rows == 0) {
            return Ok(vec![Vec::new(); cols]);
        }
        let start = self.vocab_start as u32;
        Ok(self.read_top(cols)?.into_iter().map(|c| c.into_iter().map(|(s, id)| (s, id + start)).collect()).collect())
    }

    /// Profiling: add the GPU time of every kernel of the latest forward to `into` (count, seconds), by label.
    pub fn tp_profile(&self, into: &mut std::collections::BTreeMap<String, (u64, f64)>) -> Result<()> {
        let tp = self.tp.as_ref().context("not a tensor-parallel rank")?;
        for key in &tp.keys {
            let Some(s) = self.graphs.get(key).and_then(|g| g.0.stamps.as_ref()) else { continue };
            let mut ticks = vec![0u64; s.labels.len()];
            if ticks.len() < 2 {
                continue;
            }
            unsafe {
                self.gpu.dev.get_query_pool_results(s.pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?;
            }
            for (i, label) in s.labels.iter().enumerate().skip(1) {
                let seconds = ticks[i].wrapping_sub(ticks[i - 1]) as f64 * self.gpu.period as f64 * 1e-9;
                let e = into.entry(if label.is_empty() { "(between submissions)".into() } else { label.clone() }).or_default();
                e.0 += 1;
                e.1 += seconds;
            }
        }
        Ok(())
    }

    /// The captures of the latest forward (whole tensors only).
    pub fn tp_captures(&self, m: &mut Captures) -> Result<()> {
        let tp = self.tp.as_ref().context("not a tensor-parallel rank")?;
        let graph = &self.graphs[&tp.current.context("no forward in progress")?].0;
        for (name, host, width) in &graph.captures {
            m.insert(name.clone(), bytemuck::cast_slice(&host.read(width * 4)?).to_vec());
        }
        Ok(())
    }

    /// One chained MTP draft step; the first takes the target's pending row, later ones the last output.
    /// The draft head ranks the first `rows` tokens.
    fn record_draft(&self, parts: u32, first: bool, rows: usize) -> Result<Graph> {
        let (m, hd) = (self.mtp.as_ref().context("no MTP block")?, self.head.as_ref().context("no head")?);
        let mut r = Rec::new(self, 1, false)?;
        r.mtp_inputs(m, if first { m.hin.address } else { m.out.address });
        r.mtp_block(m, parts)?;
        r.add_norm(self.x.address, Some(self.y.address), Some(m.out.address), Some(&m.head_norm), 0);
        r.head(hd, &m.out, 1, rows)?;
        r.finish()
    }

    fn submit(&self, cmds: &[vk::CommandBuffer]) -> Result<()> {
        let start = std::time::Instant::now();
        unsafe {
            self.gpu.dev.reset_fences(&[self.fence])?;
        }
        for (i, &cmd) in cmds.iter().enumerate() {
            let last = i + 1 == cmds.len();
            self.gpu
                .submit(cmd, None, None, if last { self.fence } else { vk::Fence::null() })?;
        }
        unsafe {
            self.gpu
                .dev
                .wait_for_fences(&[self.fence], true, 600_000_000_000)?;
        }
        self.busy.set(self.busy.get() + start.elapsed().as_secs_f64());
        Ok(())
    }

    fn read_top(&self, cols: usize) -> Result<Vec<Vec<(f32, u32)>>> {
        let hd = self.head.as_ref().context("no head")?;
        let bytes = hd.top.read(cols * TOP_K * 8)?;
        Ok(bytes
            .chunks_exact(TOP_K * 8)
            .map(|col| {
                let mut top: Vec<(f32, u32)> = col
                    .chunks_exact(8)
                    .map(|s| {
                        (
                            f32::from_le_bytes(s[..4].try_into().unwrap()),
                            u32::from_le_bytes(s[4..].try_into().unwrap()),
                        )
                    })
                    .collect();
                top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                top
            })
            .collect())
    }

    /// Run `n` tokens at `pos`; position 0 starts a new sequence (DeltaNet states restart from zero).
    /// `verify` keeps per-column states for `rollback` and ranks every column at the head. `emb` holds the
    /// batch's token embeddings for the MTP block.
    pub fn forward(
        &mut self,
        x: &[f32],
        pos: usize,
        n: usize,
        verify: bool,
        emb: Option<&[f32]>,
        dump: Option<&mut Captures>,
    ) -> Result<Output> {
        let d = self.c.d;
        ensure!(self.tp.is_none(), "tensor-parallel ranks run through tp::TpGroup");
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(pos + n <= self.context, "context exhausted");
        ensure!(x.len() == n * d, "hidden state length mismatch");
        let verify = verify && n > 1;
        ensure!(!verify || n <= self.verify, "verification batch exceeds the configured {} columns", self.verify);
        if let Some(m) = &self.mtp {
            let emb = emb.context("the MTP block needs the batch's token embeddings")?;
            ensure!(emb.len() == n * d, "embedding length mismatch");
            m.emb.write(bytemuck::cast_slice(emb))?;
            if pos == 0 {
                let cmd = self.gpu.begin()?;
                unsafe { self.gpu.dev.cmd_fill_buffer(cmd, m.hin.buf, 0, (d * 4) as u64, 0) };
                self.gpu.finish(cmd)?;
                self.gpu.run(cmd)?;
            }
        }
        self.input.write(bytemuck::cast_slice(x))?;
        self.params_host
            .write(bytemuck::cast_slice(&[pos as u32, n as u32, 0, 0]))?;
        let key = Key::whole(n, partitions(pos, n), dump.is_some(), verify);
        self.tick += 1;
        if let Some(e) = self.graphs.get_mut(&key) {
            e.1 = self.tick;
        } else {
            if self.graphs.len() >= GRAPH_CACHE_CAPACITY {
                // Every graph has completed: forward waits for its fence before returning.
                let victim = *self.graphs.iter().min_by_key(|e| e.1 .1).unwrap().0;
                self.graphs.remove(&victim);
            }
            let graph = self.record(key)?;
            self.graphs.insert(key, (graph, self.tick));
        }
        let graph = &self.graphs[&key].0;
        self.submit(&graph.cmds)?;
        self.last_n = n;
        self.verified = verify;
        if let Some(m) = dump {
            for (name, host, width) in &graph.captures {
                let v: Vec<f32> = bytemuck::cast_slice(&host.read(width * 4)?).to_vec();
                m.insert(name.clone(), v);
            }
        }
        Ok(match &self.head {
            Some(_) => Output::Top(self.read_top(if verify { n } else { 1 })?),
            None => Output::Hidden(bytemuck::cast_slice(&self.output.read(n * d * 4)?).to_vec()),
        })
    }

    /// One MTP draft step at `pos` for the token with embedding `emb`; `first` starts a chain from the
    /// target's state. Returns the draft's top candidates among the first `rows` tokens (0: all).
    pub fn draft(&mut self, emb: &[f32], pos: usize, first: bool, rows: usize) -> Result<Vec<(f32, u32)>> {
        let rows = if rows == 0 { self.c.vocab } else { rows.min(self.c.vocab) };
        let m = self.mtp.as_ref().context("this stage has no MTP block")?;
        ensure!(pos < self.context, "context exhausted");
        m.emb.write(bytemuck::cast_slice(emb))?;
        self.params_host
            .write(bytemuck::cast_slice(&[pos as u32, 1, 0, 0]))?;
        let key = (partitions(pos, 1), first, rows);
        if !self.drafts.contains_key(&key) {
            let graph = self.record_draft(key.0, key.1, key.2)?;
            self.drafts.insert(key, graph);
        }
        self.submit(&self.drafts[&key].cmds)?;
        Ok(self.read_top(1)?.remove(0))
    }

    fn recurrent(&self) -> impl Iterator<Item = &Delta> {
        self.layers.iter().filter_map(|l| match &l.mixer {
            Mixer::Delta(dl) => Some(dl),
            Mixer::Attention(_) => None,
        })
    }

    /// Keep columns `0..=column` of the latest verification batch.
    pub fn rollback(&mut self, column: usize) -> Result<()> {
        ensure!(column < self.last_n, "rollback beyond the latest batch");
        if column + 1 == self.last_n {
            return Ok(());
        }
        ensure!(self.verified, "only verification batches can be rolled back");
        if !self.rollbacks.contains_key(&column) {
            let (c, d) = (&self.c, self.c.d);
            let (state, conv) = (c.v_heads * c.state * c.state * 4, (c.conv - 1) * c.conv_dim() * 4);
            let cmd = self.gpu.begin()?;
            self.gpu.barrier(cmd);
            for dl in self.recurrent() {
                let (s, w) = dl.snaps.as_ref().context("no snapshot buffers")?;
                self.gpu.copy(cmd, s, column * state, &dl.state, 0, state);
                self.gpu.copy(cmd, w, column * conv, &dl.conv, 0, conv);
            }
            if let Some(m) = &self.mtp {
                self.gpu.copy(cmd, &m.hin, (column + 1) * d * 4, &m.hin, 0, d * 4);
            }
            self.gpu.finish(cmd)?;
            self.rollbacks.insert(column, cmd);
        }
        self.submit(&[self.rollbacks[&column]])?;
        self.last_n = column + 1;
        Ok(())
    }

    /// Copy the recurrent state (and the MTP's pending row) into checkpoint `slot` or back.
    fn checkpoint(&mut self, slot: usize, save: bool) -> Result<()> {
        let (c, d) = (&self.c, self.c.d);
        let (state, conv) = (c.v_heads * c.state * c.state * 4, (c.conv - 1) * c.conv_dim() * 4);
        let size = self.recurrent().count() * (state + conv) + if self.mtp.is_some() { d * 4 } else { 0 };
        if !self.slots.contains_key(&slot) {
            ensure!(save, "empty checkpoint slot");
            let host = self.gpu.buffer(size, true, false, None)?;
            self.slots.insert(slot, host);
        }
        let host = &self.slots[&slot];
        let cmd = self.gpu.begin()?;
        self.gpu.barrier(cmd);
        let mut at = 0;
        let mut copy = |b: &Buffer, len: usize| {
            if save {
                self.gpu.copy(cmd, b, 0, host, at, len);
            } else {
                self.gpu.copy(cmd, host, at, b, 0, len);
            }
            at += len;
        };
        for dl in self.recurrent() {
            copy(&dl.state, state);
            copy(&dl.conv, conv);
        }
        if let Some(m) = &self.mtp {
            copy(&m.hin, d * 4);
        }
        self.gpu.finish(cmd)?;
        self.gpu.run(cmd)
    }

    pub fn save(&mut self, slot: usize) -> Result<()> {
        self.checkpoint(slot, true)
    }

    pub fn restore(&mut self, slot: usize) -> Result<()> {
        self.checkpoint(slot, false)?;
        self.last_n = 0;
        Ok(())
    }
}

impl Drop for GpuStage {
    fn drop(&mut self) {
        unsafe {
            let _ = self.gpu.dev.device_wait_idle();
        }
        self.graphs.clear();
        self.drafts.clear();
        unsafe {
            let cmds: Vec<_> = self.rollbacks.values().copied().collect();
            if !cmds.is_empty() {
                self.gpu.dev.free_command_buffers(self.gpu.pool, &cmds);
            }
            self.gpu.dev.destroy_fence(self.fence, None);
            if let Some(tp) = &self.tp {
                self.gpu.dev.destroy_semaphore(tp.done, None);
                self.gpu.dev.destroy_semaphore(tp.release, None);
            }
        };
    }
}
