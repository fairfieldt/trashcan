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
//! with the per-rank head counts, so the kernels are unchanged. Each mixer and FFN ends its segment with a
//! partial output projection in host memory; `tp::TpGroup` all-reduces the partials and writes the total
//! that the next segment adds to the replicated residual.
use super::{cpu::Captures, vector, Config};
use crate::{
    gguf::{GgmlType, Gguf},
    quant,
    vulkan::{attention_shaders, q4_decode_shader, Buffer, Device, Push, Stamps, STAMP_CAPACITY},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::{collections::HashMap, rc::Rc};

pub const TOP_K: usize = 64;
const GRAPH_CACHE_CAPACITY: usize = 24;

impl Tp {
    /// Begin the `segments` of `cmds`: one command buffer per segment, segment `j` belonging to micro-batch
    /// `j % micro` (on two queues with `dual`), or a single command buffer when event-gated. The timelines only
    /// grow, so their values start above everything signalled so far. Returns `segments`.
    fn start(&mut self, cmds: Vec<vk::CommandBuffer>, segments: usize, head_rows: usize, micro: usize, dual: bool, gated: bool) -> usize {
        for m in 0..MAX_MICRO {
            self.base[m] = self.done_value[m].max(self.released[m]) + 1;
        }
        self.head_rows = head_rows;
        self.cmds = cmds;
        self.segments = segments;
        self.micro = micro;
        self.dual = dual;
        self.gated = gated;
        segments
    }

    /// Micro-batch and index within it of segment `j`.
    fn at(&self, j: usize) -> (usize, u64) {
        (j % self.micro, (j / self.micro) as u64)
    }
}

/// Whether a tensor-parallel forward of `n` columns is event-gated: recorded as one submission whose segments wait
/// for host-set events (a command-processor poll of host memory) instead of one submission per segment released
/// by a timeline semaphore (`TRASHCAN_TP_EVENTS=0`). Only small batches (decode, verification, drafts), which run
/// as one micro-batch: a gated submission lasts a whole forward, and amdgpu kills GPU jobs after about 10 s.
pub fn gated(n: usize) -> bool {
    n <= 8 && std::env::var("TRASHCAN_TP_EVENTS").as_deref() != Ok("0")
}

/// Whether a tensor-parallel forward keeps its residual on the host: the host adds each all-reduced total to it
/// and writes the next segment's normed rows, which the projections read from host memory (cached in L2 after
/// the first read), so the GPUs run no residual norms. Event-gated forwards without captures
/// (`TRASHCAN_TP_HOSTNORM=0`: the GPUs add and norm).
pub fn hostnorm(gated: bool, dump: bool) -> bool {
    gated && !dump && std::env::var("TRASHCAN_TP_HOSTNORM").as_deref() != Ok("0")
}

/// Stages after which gated graphs set and reset their events: all prior work.
const GATE_STAGES: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
    vk::PipelineStageFlags::COMPUTE_SHADER.as_raw() | vk::PipelineStageFlags::TRANSFER.as_raw(),
);

/// The release event of every tensor-parallel rank in this process. A SIGINT, SIGTERM or SIGHUP that arrives
/// while a gated forward waits for the host keeps setting them for half a second, so the GPUs drain their
/// submissions (computing garbage) instead of waiting until amdgpu's job timeout resets them; the signal then
/// takes its default action. RADV's vkSetEvent is a store to the event's mapped memory.
type Gate = (vk::Device, vk::PFN_vkSetEvent, vk::Event);
static GATES: std::sync::Mutex<Vec<Gate>> = std::sync::Mutex::new(Vec::new());

extern "C" fn release_gates(sig: libc::c_int) {
    if let Ok(gates) = GATES.try_lock() {
        // Every segment start resets its rank's release event, so set them until the submissions are through.
        for _ in 0..25_000 {
            for &(dev, set, event) in gates.iter() {
                let _ = unsafe { set(dev, event) };
            }
            let pause = libc::timespec { tv_sec: 0, tv_nsec: 20_000 };
            unsafe { libc::nanosleep(&pause, std::ptr::null_mut()) };
        }
    }
    unsafe {
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

fn register_gate(gpu: &Device, release: vk::Event) {
    static HANDLERS: std::sync::Once = std::sync::Once::new();
    HANDLERS.call_once(|| unsafe {
        for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            libc::signal(sig, release_gates as extern "C" fn(libc::c_int) as libc::sighandler_t);
        }
    });
    let set = gpu.dev.fp_v1_0().set_event;
    GATES.lock().unwrap().push((gpu.dev.handle(), set, release));
}

fn unregister_gates(gpu: &Device) {
    GATES.lock().unwrap().retain(|g| g.0 != gpu.dev.handle());
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
            c.k_heads.is_multiple_of(r)
                && c.heads.is_multiple_of(r)
                && (c.kv_heads.is_multiple_of(r) || r.is_multiple_of(c.kv_heads))
                && c.ff.is_multiple_of(32 * r)
                && c.vocab.is_multiple_of(r),
            "{} ranks do not divide this model's heads, FFN or vocabulary",
            r
        );
        let mut l = c.clone();
        l.k_heads /= r;
        l.v_heads /= r;
        l.heads /= r;
        l.kv_heads = self.kv_part(c.kv_heads).1;
        l.ff /= r;
        l.vocab /= r;
        Ok(l)
    }

    /// This rank's (first, count) of `total` items split evenly.
    pub fn part(&self, total: usize) -> (usize, usize) {
        (self.rank * total / self.ranks, total / self.ranks)
    }

    /// This rank's (first, count) of `total` KV heads. With more ranks than KV heads, the ranks whose query heads
    /// read one KV head each hold (and compute) that head.
    pub fn kv_part(&self, total: usize) -> (usize, usize) {
        if self.ranks <= total {
            self.part(total)
        } else {
            (self.rank * total / self.ranks, 1)
        }
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
    /// `[conv weights | dt bias | A]`, the weight binding of `qwen_gdn_conv`.
    conv_wdt: Buffer,
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
    /// Rows of `w` (the quantized values start after their scales).
    rows: usize,
    /// Candidates of the selection passes: per 1024 logits, then per 1024 of those.
    scratch: Buffer,
    scratch2: Buffer,
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
    /// Timestamps written by every replay, when profiling: the query pool and a label per query.
    stamps: Option<(vk::QueryPool, Rc<Vec<String>>)>,
    /// Tensor-parallel segments (one per command buffer unless event-gated).
    segments: usize,
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

/// A graph's timestamps after a replay: the query pool's identity, the labels, the times in ns since the first,
/// and the first one in host CLOCK_MONOTONIC ns when calibrated.
pub type StampRead = (u64, Rc<Vec<String>>, Vec<f64>, Option<f64>);

/// Profiling level from the environment: 0 off, 1 segment boundaries (`TRASHCAN_TP_PROFILE=FILE`), 2 also every
/// dispatch (`TRASHCAN_TP_PROFILE_DISPATCH=1`).
pub fn profile_level() -> u8 {
    match (std::env::var_os("TRASHCAN_TP_PROFILE"), std::env::var_os("TRASHCAN_TP_PROFILE_DISPATCH")) {
        (None, _) => 0,
        (Some(_), None) => 1,
        (Some(_), Some(_)) => 2,
    }
}

pub enum Output {
    Hidden(Vec<f32>),
    /// Candidates per column: the last one, or every column of a verification batch.
    Top(Vec<Vec<(f32, u32)>>),
}

/// One recorded graph: batch, attention partitions, dump, verify; for a micro-batch of a tensor-parallel batch its
/// params slot, first row in the batch buffers and whether it ends with the LM head; and whether it is event-gated.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
    n: usize,
    parts: u32,
    keys: usize,
    dump: bool,
    verify: bool,
    slot: usize,
    row: usize,
    head: bool,
    gated: bool,
}

impl Key {
    fn whole(n: usize, parts: u32, dump: bool, verify: bool) -> Self {
        Key { n, parts, keys: 0, dump, verify, slot: 0, row: 0, head: true, gated: false }
    }
}

/// Micro-batches of a tensor-parallel batch have their own `pos, n` parameters, this many bytes apart.
const PARAMS_STRIDE: usize = 256;
/// Most micro-batches a tensor-parallel batch is split into.
pub const MAX_MICRO: usize = 2;

/// Split a batch of `n` columns into at most `count` micro-batches whose sizes are multiples of 64 (the GEMM tile)
/// except the last: (first row, rows) of each.
pub fn micro_batches(n: usize, count: usize) -> Vec<(usize, usize)> {
    let size = n.div_ceil(count.clamp(1, MAX_MICRO)).next_multiple_of(64).min(n.max(1));
    (0..n).step_by(size).map(|r| (r, size.min(n - r))).collect()
}

/// Tensor-parallel segments: the host buffers of the all-reduces and the timelines the host polls.
struct Tp {
    /// This rank's partial output projection, written by the GPU at the end of each segment.
    part: Buffer,
    /// The all-reduced total, written by the host before the next segment.
    total: Buffer,
    /// The normed rows of the next segment when the host keeps the residual (`hostnorm`).
    hin: Buffer,
    hostnorm: bool,
    /// Segment `i` of micro-batch `m` of the forward in progress signals `done[m]` = `base[m] + i + 1` when it
    /// finishes and, when submitted ahead, waits for the host to signal `release[m]` = `base[m] + i`. An
    /// event-gated forward signals `done[0]` once, when its submission ends.
    done: [vk::Semaphore; MAX_MICRO],
    release: [vk::Semaphore; MAX_MICRO],
    base: [u64; MAX_MICRO],
    /// Highest value signalled on `done` and on `release` so far.
    done_value: [u64; MAX_MICRO],
    released: [u64; MAX_MICRO],
    /// Graph of the forward in progress (for its captures), and the segments of the forward or draft step in
    /// progress (segment `j` is segment `j / micro` of micro-batch `j % micro`) with the vocabulary rows its head
    /// ranks.
    current: Option<Key>,
    cmds: Vec<vk::CommandBuffer>,
    micro: usize,
    /// Two micro-batches on two queues: the second micro-batch's segment `i` also waits for the first's, and the
    /// first micro-batch's next segment can run beside it.
    dual: bool,
    head_rows: usize,
    /// First token id of the head rows ranked by the step in progress, and whether they are the balanced draft
    /// head's (`tp_draft_head`).
    head_start: usize,
    draft_top: bool,
    /// Timestamps of the graphs in progress (one per micro-batch), when profiling.
    stamps: Vec<(vk::QueryPool, Rc<Vec<String>>)>,
    /// Release and done events of event-gated forwards (`gated`), shared by all segments: each segment after the
    /// first waits for the host to set release and resets it (ordered before its done), and sets done once its
    /// partial is in host memory; the host resets done when it sees it, before releasing the next segment. One
    /// pair instead of one per segment: every event is a buffer object that each submission must list.
    gate: (vk::Event, vk::Event),
    /// The forward or draft step in progress is event-gated, and its number of segments.
    gated: bool,
    segments: usize,
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
    /// Bounded 64-query tiled prefill scores and denominators, separate for each TP queue.
    prefill_scores: Option<Buffer>,

    gate: Buffer,
    up: Buffer,
    act: Buffer,
    fence: vk::Fence,
    graphs: HashMap<Key, (Graph, u64)>,
    /// Draft graphs by (partitions, first step, head rows, balanced draft head).
    drafts: HashMap<(u32, bool, usize, bool), Graph>,
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
    split: Split,
    /// A tensor-parallel rank's share of the draft vocabulary `(vocab, first id, rows)` (`tp_draft_head`).
    draft_head: Option<(usize, usize, Head)>,
}

/// `TRASHCAN_QWEN_GEMM=gemm` selects the original projection kernel for A/B timing.
fn gemm_reference() -> bool {
    std::env::var("TRASHCAN_QWEN_GEMM").is_ok_and(|v| v == "gemm")
}

/// Whether two tensor-parallel micro-batches run on two queues (`TRASHCAN_TP_QUEUES=1`: one queue).
fn two_queues() -> bool {
    std::env::var("TRASHCAN_TP_QUEUES").map_or(true, |v| v != "1")
}

/// Partition count of split decode attention, by visible tokens; large batches use the unsplit kernel.
/// Tensor-parallel ranks hold few query heads, so their split-K attention takes twice the partitions (at most
/// 128) to have enough waves; `TRASHCAN_TP_PARTS=N` sets the multiplier.
fn tp_partitions(pos: usize, n: usize) -> u32 {
    let p = partitions(pos, n);
    let scale = std::env::var("TRASHCAN_TP_PARTS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(2);
    if p == 0 {
        0
    } else {
        (p * scale).min(128)
    }
}

/// The many-rows decode matvec for rows of `k` values (`TRASHCAN_QWEN_SHORTK=8|16`, rows per wave, for k up to 2048:
/// the out-projections of TP8 and TP4 ranks) and its rows per wave; the same bits as `matvecq4`.
fn short_k(k: usize) -> Option<(&'static str, usize)> {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    match *ROWS.get_or_init(|| std::env::var("TRASHCAN_QWEN_SHORTK").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) {
        8 if k <= 2048 => Some(("qwen_matvec_rows8", 8)),
        16 if k <= 2048 => Some(("qwen_matvec_rows16", 16)),
        _ => None,
    }
}

/// Whether a DeltaNet layer of a batch of `n` columns folds its convolution into the recurrence
/// (`TRASHCAN_QWEN_FUSED_GDN=1`): decode and verification batches (up to 8 columns), not when dumping activations,
/// whose convolution output is a captured tensor.
fn fused_gdn(n: usize, dump: bool) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TRASHCAN_QWEN_FUSED_GDN").as_deref() == Ok("1")) && n <= 8 && !dump
}

/// The fused gate/up/SwiGLU kernel for a batch of `n` columns (`TRASHCAN_QWEN_FUSED_FFN=1`, up to four columns: the
/// widths whose projections use `matvecq4` and `qwen_q4cN`, whose bits the fused kernel reproduces).
fn fused_ffn(n: usize) -> Option<&'static str> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("TRASHCAN_QWEN_FUSED_FFN").as_deref() == Ok("1")) {
        return None;
    }
    match n {
        1 => Some("qwen_ffn_gu1"),
        2 => Some("qwen_ffn_gu2"),
        3 => Some("qwen_ffn_gu3"),
        4 => Some("qwen_ffn_gu4"),
        _ => None,
    }
}

/// The head's candidate selection: `qwen_topk`, or with `TRASHCAN_QWEN_TOPK=wave` the same steps with wave-local
/// barriers (`qwen_topk_wave`, the same candidates).
fn topk_kernel() -> &'static str {
    static WAVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *WAVE.get_or_init(|| std::env::var("TRASHCAN_QWEN_TOPK").as_deref() == Ok("wave")) {
        "qwen_topk_wave"
    } else {
        "qwen_topk"
    }
}

/// The decode attention kernel for `heads` local query heads over `kv_heads` KV heads of size `dim`, when
/// `TRASHCAN_QWEN_DECODE_ATTN=1` selects `qwen_attn_decode` and it has a variant for the shape.
fn decode_attention(heads: usize, kv_heads: usize, dim: usize) -> Option<&'static str> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("TRASHCAN_QWEN_DECODE_ATTN").as_deref() == Ok("1")) || dim != 256 || kv_heads == 0 || heads % kv_heads != 0 {
        return None;
    }
    DECODE_ATTN_HEADS.iter().find(|v| v.0 == heads / kv_heads).map(|v| v.1)
}

/// `qwen_attn_decode` pipelines by query heads per KV head.
pub const DECODE_ATTN_HEADS: [(usize, &str); 6] = [
    (1, "qwen_attn_decode_g1"),
    (2, "qwen_attn_decode_g2"),
    (3, "qwen_attn_decode_g3"),
    (4, "qwen_attn_decode_g4"),
    (6, "qwen_attn_decode_g6"),
    (8, "qwen_attn_decode_g8"),
];

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

fn prefill_key_limit(context: usize, pos: usize, n: usize) -> usize {
    (pos + n).max(64).next_power_of_two().min(context.next_multiple_of(64))
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
    let (heads, kv_heads) = (split.part(c.heads), split.kv_part(c.kv_heads));
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
    /// Event-gated tensor-parallel graph (`gated`).
    gated: bool,
    /// Upper bound of visible keys in this cached tiled-attention graph.
    keys: usize,
    /// The normed rows the mixers and FFNs project: `h` (this micro-batch's half of it).
    h_in: u64,
    /// Params slot, and first row in the batch buffers (`input`, `x`, the MTP embeddings, the partial and the
    /// total), of a micro-batch; 0 for a whole batch.
    slot: usize,
    row: usize,
    /// The second micro-batch of a tensor-parallel batch works in the second half of the work buffers.
    alt: bool,
}

impl<'a> Rec<'a> {
    fn new(s: &'a GpuStage, n: usize, dump: bool) -> Result<Self> {
        Self::at(s, n, dump, 0, 0, s.tp.is_some() && gated(n))
    }

    /// A micro-batch of `n` rows starting at `row`, with its params in `slot`; `gated` records an event-gated
    /// tensor-parallel graph.
    fn at(s: &'a GpuStage, n: usize, dump: bool, slot: usize, row: usize, gated: bool) -> Result<Self> {
        let cmd = s.gpu.begin()?;
        let level = if s.tp.is_some() { profile_level() } else { 0 };
        if level > 0 {
            let pool = unsafe {
                s.gpu.dev.create_query_pool(
                    &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(STAMP_CAPACITY),
                    None,
                )?
            };
            unsafe { s.gpu.dev.cmd_reset_query_pool(cmd, pool, 0, STAMP_CAPACITY) };
            *s.gpu.stamps.borrow_mut() = Some(Stamps { pool, labels: Vec::new(), dispatches: level > 1 });
        }
        s.gpu.barrier(cmd);
        s.gpu.stamp(cmd, false, || "begin".into());
        s.gpu.copy(cmd, &s.params_host, slot * PARAMS_STRIDE, &s.params, slot * PARAMS_STRIDE, 16);
        let alt = s.tp.is_some() && slot % 2 == 1;
        let h_half = if alt { s.h.size / 2 } else { 0 };
        Ok(Self {
            s,
            n,
            dump,
            cmd,
            graph: Graph {
                gpu: s.gpu.clone(),
                cmds: vec![cmd],
                captures: Vec::new(),
                stamps: None,
                segments: 1,
            },
            out: s.tp.as_ref().map_or(s.y.address, |t| t.part.address) + (row * s.c.d * 4) as u64,
            gated,
            keys: s.context.next_multiple_of(64),
            h_in: s.h.address + h_half as u64,
            slot,
            row,
            alt,
        })
    }

    /// Byte offset of this micro-batch's rows in the batch buffers.
    fn rows(&self) -> usize {
        self.row * self.s.c.d * 4
    }

    /// This micro-batch's rows of the residual.
    fn x(&self) -> u64 {
        self.s.x.address + self.rows() as u64
    }

    /// This micro-batch's half of a work buffer (`GpuStage::build`): its byte offset, and its address.
    fn half(&self, b: &Buffer) -> usize {
        if self.alt {
            b.size / 2
        } else {
            0
        }
    }

    fn sc(&self, b: &Buffer) -> u64 {
        b.address + self.half(b) as u64
    }

    /// Make this segment's writes visible to the host and, when gated, set its done event.
    fn signal_done(&mut self) -> Result<()> {
        self.s.gpu.host_barrier(self.cmd, true);
        if self.gated {
            let (_, done) = self.s.tp.as_ref().context("not a tensor-parallel rank")?.gate;
            unsafe { self.s.gpu.dev.cmd_set_event(self.cmd, done, GATE_STAGES) };
        }
        Ok(())
    }

    /// End a tensor-parallel segment: the host reads the partial projection after this submission, then
    /// writes the all-reduced total before submitting the next one.
    fn end_segment(&mut self) -> Result<()> {
        self.s.gpu.stamp(self.cmd, true, || "end".into());
        self.signal_done()?;
        // A gated forward stays one command buffer (each one adds its own buffer to the submission).
        if self.gated {
            self.s.gpu.barrier(self.cmd);
        } else {
            self.split()?;
        }
        self.graph.segments += 1;
        if self.gated {
            let (release, _) = self.s.tp.as_ref().unwrap().gate;
            self.s.gpu.wait_host_event(self.cmd, release);
            unsafe { self.s.gpu.dev.cmd_reset_event(self.cmd, release, GATE_STAGES) };
        } else {
            self.s.gpu.host_barrier(self.cmd, false);
        }
        self.s.gpu.stamp(self.cmd, false, || "begin".into());
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
        self.s.gpu.barrier(self.cmd);
        self.graph.cmds.push(self.cmd);
        Ok(())
    }

    fn finish(mut self) -> Result<Graph> {
        self.s.gpu.stamp(self.cmd, true, || "end".into());
        if self.gated {
            self.signal_done()?;
        }
        self.s.gpu.finish(self.cmd)?;
        self.graph.stamps = self.s.gpu.stamps.borrow_mut().take().map(|s| (s.pool, Rc::new(s.labels)));
        Ok(self.graph)
    }

    fn copy(&self, a: &Buffer, ao: usize, b: &Buffer, bo: usize, size: usize) {
        self.s.gpu.copy(self.cmd, a, ao, b, bo, size);
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
        // Tensor-parallel ranks use matvecq4 at every shape: a matvecq4narrow dispatch (512-thread workgroups)
        // only starts once the previous one has finished on the D700s, so independent projections cannot overlap.
        let decode = if self.s.tp.is_some() { "matvecq4" } else { q4_decode_shader(rows, k) };
        let (shader, gx, gy) = match n {
            1 if short_k(k).is_some() => {
                let (kernel, per_wave) = short_k(k).unwrap();
                (kernel, rows.div_ceil(4 * per_wave), 1)
            }
            1 => (decode, rows.div_ceil(16), 1),
            2 => ("qwen_q4c2", rows.div_ceil(16), 1),
            3 => ("qwen_q4c3", rows.div_ceil(16), 1),
            4 => ("qwen_q4c4", rows.div_ceil(16), 1),
            5..=8 => ("multivec4", rows.div_ceil(16), n.div_ceil(4)),
            // gemm2 is gemm with conflict-free weight-tile reads and the next block loaded ahead: identical results.
            _ => (if gemm_reference() { "gemm" } else { "gemm2" }, rows.div_ceil(64), n.div_ceil(64)),
        };
        self.s.gpu.dispatch_matrix(
            self.cmd,
            shader,
            self.push([0, x, y, 0, 0, 0], u).defer_barrier(defer),
            gx as u32,
            gy as u32,
            w,
        )
    }

    /// Optional residual add x += y, then h = RMS(x) * w when h is given, its rows `stride` apart
    /// (0: dense). One workgroup per token.
    fn add_norm(&self, x: u64, y: Option<u64>, h: Option<u64>, w: Option<&Buffer>, stride: usize) {
        // Hidden sizes of 16 or 20 elements per thread have a straight-line variant (same sums).
        let shader = match self.s.c.d {
            4096 => "qwen_add_norm_j16",
            5120 => "qwen_add_norm_j20",
            _ => "qwen_add_norm",
        };
        let mut u = [0; 16];
        u[0] = self.s.c.d as u32;
        u[3] = stride as u32;
        u[11] = u32::from(y.is_some());
        u[12] = self.s.c.eps.to_bits();
        self.s.gpu.dispatch(
            self.cmd,
            shader,
            self.push([x, y.unwrap_or(0), h.unwrap_or(0), w.map_or(0, |b| b.address), 0, 0], u),
            1,
            self.n as u32,
        );
    }

    /// Gated DeltaNet from `h` into `y`.
    fn delta(&mut self, dl: &Delta, il: usize, verify: bool) -> Result<()> {
        let (s, c) = (self.s, &self.s.c);
        let (d, cd, vd) = (c.d, c.conv_dim(), c.value_dim());
        let eps = c.eps.to_bits();
        let (h, big, z, ab, conv_out, o) = (self.h_in, self.sc(&s.big), self.sc(&s.z), self.sc(&s.ab), self.sc(&s.conv_out), self.sc(&s.o));
        // Independent projections share one barrier. The command processor starts a dispatch's workgroups only
        // once the previous dispatch's have all started, so tensor-parallel ranks issue the small f32 alpha/beta
        // projection first, where it runs beside the large ones instead of alone after them.
        let old = s.tp.is_none();
        if old {
            self.matrix(&dl.qkv, h, big, d, cd, true)?;
            self.matrix(&dl.z, h, z, d, vd, true)?;
        }
        let mut u = [0; 16];
        u[0] = (2 * c.v_heads) as u32;
        u[1] = d as u32;
        // Decode and verification batches take 16 loads per lane, larger batches the plain loop: with the 16-load
        // path compiled in, a 9B TP2 prefill was 4% slower.
        s.gpu.dispatch_matrix(
            self.cmd,
            if self.n <= 4 { "qwen_matf32_u16" } else { "qwen_matf32" },
            self.push([0, h, ab, 0, 0, 0], u).defer_barrier(!old),
            (2 * c.v_heads).div_ceil(4) as u32,
            self.n.div_ceil(8) as u32,
            &dl.ab,
        )?;
        if !old {
            self.matrix(&dl.z, h, z, d, vd, true)?;
            self.matrix(&dl.qkv, h, big, d, cd, false)?;
        }
        self.capture(format!("linear_attn_qkv_mixed-{il}"), &s.big, cd)?;
        self.capture(format!("z-{il}"), &s.z, vd)?;
        let (snap, snap_conv) = match (&dl.snaps, verify) {
            (Some((a, b)), true) => (a.address, b.address),
            _ => (0, 0),
        };
        if fused_gdn(self.n, self.dump) {
            // The convolution inside the recurrence; the q and k windows are written by the gated norm, after every
            // wave has read them (docs/tp8-optimization.md G1). The same bits as the three dispatches below.
            let mut u = [0; 16];
            u[0] = cd as u32;
            u[4] = c.v_heads as u32;
            u[5] = c.k_heads as u32;
            u[6] = c.state as u32;
            u[12] = eps;
            s.gpu.dispatch_matrix(
                self.cmd,
                "qwen_gdn_conv",
                self.push([big, ab, dl.state.address, o, dl.conv.address, snap], u).with_extra(snap_conv),
                (c.v_heads * c.state / 16) as u32,
                1,
                &dl.conv_wdt,
            )?;
            self.capture(format!("attn_output-{il}"), &s.o, vd)?;
            let mut u = [0; 16];
            u[0] = vd as u32;
            u[1] = cd as u32;
            u[2] = (2 * c.key_dim()) as u32;
            u[6] = c.state as u32;
            u[12] = eps;
            s.gpu.dispatch(
                self.cmd,
                "qwen_gated_norm_conv",
                self.push([o, z, dl.norm.address, big, dl.conv.address, snap_conv], u),
                c.v_heads as u32,
                self.n as u32,
            );
            self.capture(format!("final_output-{il}"), &s.o, vd)?;
            self.matrix(&dl.out, o, self.out, vd, d, false)?;
            return self.capture(format!("linear_attn_out-{il}"), &s.y, d);
        }
        let mut u = [0; 16];
        u[0] = cd as u32;
        s.gpu.dispatch(
            self.cmd,
            "qwen_conv",
            self.push([big, dl.conv_w.address, dl.conv.address, conv_out, snap_conv, 0], u),
            cd.div_ceil(256) as u32,
            1,
        );
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
            self.push([conv_out, ab, dl.state.address, o, dl.dt_a.address, snap], u),
            (c.v_heads * c.state / 16) as u32,
            1,
        );
        self.capture(format!("attn_output-{il}"), &s.o, vd)?;
        let mut u = [0; 16];
        u[0] = vd as u32;
        u[6] = c.state as u32;
        u[12] = eps;
        s.gpu.dispatch(
            self.cmd,
            "qwen_gated_norm",
            self.push([o, z, dl.norm.address, 0, 0, 0], u),
            c.v_heads as u32,
            self.n as u32,
        );
        self.capture(format!("final_output-{il}"), &s.o, vd)?;
        self.matrix(&dl.out, o, self.out, vd, d, false)?;
        self.capture(format!("linear_attn_out-{il}"), &s.y, d)
    }

    /// Gated attention from `h` into `y`, caching K and V at the batch positions.
    fn attention(&mut self, a: &Attn, parts: u32, il: usize) -> Result<()> {
        let (s, c) = (self.s, &self.s.c);
        let (d, hd, kvd) = (c.d, c.head_dim, c.kv_heads * c.head_dim);
        let qd = c.heads * hd;
        let h = self.h_in;
        let (big, kb, vb, rq, o, scratch) = (self.sc(&s.big), self.sc(&s.kb), self.sc(&s.vb), self.sc(&s.rq), self.sc(&s.o), self.sc(&s.scratch));
        // Tensor-parallel ranks issue the small K and V projections first (see `delta`).
        if s.tp.is_none() {
            self.matrix(&a.q, h, big, d, 2 * qd, true)?;
            self.matrix(&a.k, h, kb, d, kvd, true)?;
            self.matrix(&a.v, h, vb, d, kvd, false)?;
        } else {
            self.matrix(&a.k, h, kb, d, kvd, true)?;
            self.matrix(&a.v, h, vb, d, kvd, true)?;
            self.matrix(&a.q, h, big, d, 2 * qd, false)?;
        }
        // Q (normed, rotated, pre-scaled into rq), K (normed, rotated into the cache), V (cache).
        for (src, norm, out, cache, heads, stride, width, scale, defer) in [
            (big, Some(&a.q_norm), rq, None, c.heads, 2 * hd, 2 * qd, 1. / (hd as f32).sqrt(), true),
            (kb, Some(&a.k_norm), 0, Some(&a.kc), c.kv_heads, hd, kvd, 1., true),
            (vb, None, 0, Some(&a.vc), c.kv_heads, hd, kvd, 1f32, false),
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
                        src,
                        norm.map_or(0, |b| b.address),
                        out,
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
        }
        self.capture(format!("Qcur-{il}"), &s.rq, qd)?;
        let mut u = [0; 16];
        u[4] = c.heads as u32;
        u[5] = c.kv_heads as u32;
        u[6] = hd as u32;
        u[7] = s.context as u32;
        let mut gated = false;
        if parts == 0 && s.prefill_scores.is_some() && self.n >= 64 {
            let score = self.sc(s.prefill_scores.as_ref().unwrap());
            let stride = s.context.next_multiple_of(64);
            u[0] = stride as u32;
            u[1] = (c.heads / c.kv_heads) as u32;
            let tq = 1usize << (64 / (c.heads / c.kv_heads)).ilog2();
            for first in (0..self.n).step_by(64) {
                u[13] = first as u32;
                let nb = (self.n - first).min(64);
                let groups = c.kv_heads * nb.div_ceil(tq);
                let p = self.push([rq, a.kc.address, a.vc.address, o, score, score + (64 * c.heads * stride * 4) as u64], u);
                s.gpu.dispatch(self.cmd, "qwen_attn_scores", p, self.keys.div_ceil(64) as u32, groups as u32);
                s.gpu.dispatch(self.cmd, "qwen_attn_softmax", p, c.heads as u32, nb as u32);
                s.gpu.dispatch(self.cmd, "qwen_attn_values", p, (hd / 64) as u32, groups as u32);
            }
        } else if parts == 0 {
            s.gpu.dispatch(
                self.cmd,
                "attention",
                self.push([rq, a.kc.address, a.vc.address, o, 0, 0], u),
                c.heads as u32,
                self.n as u32,
            );
        } else if let Some(kernel) = decode_attention(c.heads, c.kv_heads, hd) {
            // Every query head of a KV head in one wave, then a merge that also applies the output gate, except when
            // dumping, which captures the ungated output first (docs/tp8-optimization.md A1).
            u[2] = parts;
            s.gpu.dispatch(
                self.cmd,
                kernel,
                self.push([rq, a.kc.address, a.vc.address, 0, scratch, 0], u),
                c.kv_heads as u32 * parts,
                self.n as u32,
            );
            s.gpu.dispatch(
                self.cmd,
                "qwen_attn_merge",
                self.push([0, if self.dump { 0 } else { big }, 0, o, scratch, 0], u),
                c.heads as u32,
                self.n as u32,
            );
            gated = !self.dump;
        } else {
            let (partial, merge) = attention_shaders(parts)?;
            s.gpu.dispatch(
                self.cmd,
                partial,
                self.push([rq, a.kc.address, a.vc.address, o, scratch, 0], u),
                c.heads as u32 * parts,
                self.n as u32,
            );
            s.gpu.dispatch(
                self.cmd,
                merge,
                self.push([0, 0, 0, o, scratch, 0], u),
                c.heads as u32,
                self.n as u32,
            );
        }
        self.capture(format!("attn_pregate-{il}"), &s.o, qd)?;
        if !gated {
            let mut u = [0; 16];
            u[0] = qd as u32;
            u[6] = hd as u32;
            s.gpu.dispatch(
                self.cmd,
                "qwen_gate",
                self.push([o, big, 0, 0, 0, 0], u),
                qd.div_ceil(256) as u32,
                self.n as u32,
            );
        }
        self.capture(format!("attn_gated-{il}"), &s.o, qd)?;
        self.matrix(&a.out, o, self.out, qd, d, false)?;
        self.capture(format!("attn_output-{il}"), &s.y, d)
    }

    /// SwiGLU FFN from `h` into `y`.
    fn ffn(&mut self, f: &Ffn, il: usize) -> Result<()> {
        let (s, c) = (self.s, &self.s.c);
        let (h, gate, up, act) = (self.h_in, self.sc(&s.gate), self.sc(&s.up), self.sc(&s.act));
        if let Some(kernel) = fused_ffn(self.n) {
            // Gate, up and SwiGLU in one dispatch, the same bits (docs/tp8-optimization.md F1).
            let mut u = [0; 16];
            u[0] = c.ff as u32;
            u[1] = c.d as u32;
            u[3] = (c.ff * c.d / 32) as u32;
            s.gpu.dispatch_matrix(self.cmd, kernel, self.push([0, h, act, f.up.address, 0, 0], u), c.ff.div_ceil(16) as u32, 1, &f.gate)?;
            self.capture(format!("ffn_swiglu-{il}"), &s.act, c.ff)?;
            return self.matrix(&f.down, act, self.out, c.ff, c.d, false);
        }
        self.matrix(&f.gate, h, gate, c.d, c.ff, true)?;
        self.matrix(&f.up, h, up, c.d, c.ff, false)?;
        let mut u = [0; 16];
        u[0] = c.ff as u32;
        s.gpu.dispatch(
            self.cmd,
            "qwen_swiglu",
            self.push([gate, up, act, 0, 0, 0], u),
            c.ff.div_ceil(256) as u32,
            self.n as u32,
        );
        self.capture(format!("ffn_swiglu-{il}"), &s.act, c.ff)?;
        self.matrix(&f.down, act, self.out, c.ff, c.d, false)
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
        u[3] = (hd.rows * d / 16) as u32;
        u[9] = 1;
        let single = u32::from(cols == 1);
        u[15] = single;
        let input = self.sc(input);
        let (shader, gy, x) = match cols {
            1 => ("headvec", 1, input + ((n - 1) * d * 4) as u64),
            2 => ("multivec4x2", 1, input),
            3 => ("multivec4x3", 1, input),
            _ => ("multivec4", cols.div_ceil(4), input),
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
        if self.dump {
            // A tensor-parallel rank holds a slice of the vocabulary; tp::TpGroup joins the slices.
            let name = if s.tp.is_some() { "result_output_part" } else { "result_output" };
            let host = s.gpu.buffer(rows * 4, true, false, None)?;
            s.gpu.copy(self.cmd, &hd.logits, (cols - 1) * rows * 4, &host, 0, rows * 4);
            self.graph.captures.push((name.into(), host, rows));
        }
        if s.tp.is_some() {
            // Passes of qwen_topk: the best 64 of each 1024 logits, then of each 1024 of those, until one list is
            // left (at most three passes for 256 chunks). The same candidates as topk and merge, in a fifth of the
            // time.
            let (mut src, mut kind, mut entries, mut groups, mut pass) = (hd.logits.address, 0, rows, chunks, 0);
            loop {
                let dst = match (groups, pass) {
                    (1, _) => hd.top.address,
                    (_, 0) => hd.scratch.address,
                    _ => hd.scratch2.address,
                };
                let mut u = [0; 16];
                u[0] = entries as u32;
                u[2] = (groups * TOP_K) as u32;
                u[9] = kind;
                u[15] = single;
                s.gpu.dispatch(self.cmd, topk_kernel(), self.push([src, 0, dst, 0, 0, 0], u), groups as u32, cols as u32);
                if groups == 1 {
                    return Ok(());
                }
                ensure!(pass < 2, "too many candidate lists");
                (src, kind, entries, pass) = (dst, 1, groups * TOP_K, pass + 1);
                groups = entries.div_ceil(1024);
            }
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
        let stage = Self::build(gpu, g, c, 0, c.layers, true, context, max_batch, verify, mtp, split, true)?;
        register_gate(&stage.gpu, stage.tp.as_ref().unwrap().gate.0);
        Ok(stage)
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
                let conv_wdt: Vec<f32> = conv_w.iter().chain(&dt_a).copied().collect();
                let (state, conv) = (c.v_heads * c.state * c.state * 4, (c.conv - 1) * c.conv_dim() * 4);
                Mixer::Delta(Delta {
                    qkv: upload_q4_cut(&gpu, g, c, il, "attn_qkv.weight", &channels, &[])?,
                    z: upload_q4_cut(&gpu, g, c, il, "attn_gate.weight", &value_dims, &[])?,
                    ab: gpu.upload_matrix(bytemuck::cast_slice(&ab))?,
                    dt_a: upload_f32(&gpu, &dt_a)?,
                    conv_w: upload_f32(&gpu, &conv_w)?,
                    conv_wdt: gpu.upload_matrix(bytemuck::cast_slice(&conv_wdt))?,
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
                rows: vocab_rows,
                logits: gpu.buffer(cols * c.vocab * 4, false, false, None)?,
                scratch: gpu.buffer(cols * chunks * TOP_K * 8, false, false, None)?,
                scratch2: gpu.buffer(cols * chunks.div_ceil(16) * TOP_K * 8, false, false, None)?,
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
        // Under tensor parallelism the second micro-batch has its own half of every buffer a segment works in,
        // so it can run beside the first one on a second queue (`Rec::sc`).
        let halves = if tp { 2 } else { 1 };
        let work = |w: usize| gpu.buffer(w * max_batch * 4 * halves, false, false, None);
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
            h: work(c.d)?,
            y: alloc(c.d)?,
            big: work(c.conv_dim().max(2 * qd))?,
            conv_out: work(c.conv_dim())?,
            z: work(c.value_dim())?,
            ab: work(2 * c.v_heads)?,
            o: work(c.value_dim().max(qd))?,
            kb: work(c.kv_heads * c.head_dim)?,
            vb: work(c.kv_heads * c.head_dim)?,
            rq: work(qd)?,
            // Split-K attention partials: up to 64 partitions, 128 on tensor-parallel ranks (`tp_partitions`), each a
            // record of head_dim + 2 (`attention_partial*`) or + 4 (`qwen_attn_decode`) floats.
            scratch: gpu.buffer(c.heads * if tp { 128 } else { 64 } * (c.head_dim + 4) * 4 * 8 * halves, false, false, None)?,
            // Tiled prefill keeps bounded score scratch separate for each active queue.
            prefill_scores: if max_batch >= 64 && std::env::var("TRASHCAN_QWEN_ATTN").as_deref() == Ok("tiled") {
                Some(gpu.buffer(c.heads * 64 * (context.next_multiple_of(64) + 1) * 4 * halves, false, false, None)?)
            } else { None },
            gate: work(c.ff)?,
            up: work(c.ff)?,
            act: work(c.ff)?,
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
                    hin: gpu.buffer(c.d * 8 * 4, true, false, None)?,
                    hostnorm: false,
                    done: [gpu.timeline(false)?, gpu.timeline(false)?],
                    release: [gpu.timeline(false)?, gpu.timeline(false)?],
                    base: [0; MAX_MICRO],
                    done_value: [0; MAX_MICRO],
                    released: [0; MAX_MICRO],
                    current: None,
                    cmds: Vec::new(),
                    micro: 1,
                    dual: false,
                    head_rows: 0,
                    head_start: 0,
                    draft_top: false,
                    stamps: Vec::new(),
                    gate: unsafe {
                        let e = || gpu.dev.create_event(&vk::EventCreateInfo::default(), None);
                        (e()?, e()?)
                    },
                    gated: false,
                    segments: 0,
                })
            } else {
                None
            },
            vocab_start,
            split,
            draft_head: None,
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
        if k.keys > 0 { r.keys = k.keys; }
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

    /// Tensor-parallel graph: every mixer and FFN ends a segment with its partial projection in host memory, and
    /// the next segment adds the all-reduced total to the replicated residual. The last segment ranks this rank's
    /// vocabulary rows (unless `k` is a micro-batch before the last). 2 * layers + 1 segments, each its own
    /// submission unless event-gated. A micro-batch works on its own rows of the batch buffers and its own half
    /// of the work buffers, so the segments of two micro-batches can alternate or run side by side (`tp_prepare`).
    fn record_tp(&self, k: Key) -> Result<Graph> {
        let (n, parts, dump, verify) = (k.n, k.parts, k.dump, k.verify);
        let d = self.c.d;
        let (hd, tp) = (self.head.as_ref().context("a tensor-parallel rank needs its head rows")?, self.tp.as_ref().unwrap());
        let host = hostnorm(k.gated, dump);
        let mut r = Rec::at(self, n, dump, k.slot, k.row, k.gated)?;
        if k.keys > 0 { r.keys = k.keys; }
        let (rows, x, h) = (r.rows(), r.x(), r.sc(&self.h));
        let total = tp.total.address + rows as u64;
        if host {
            r.h_in = tp.hin.address;
        } else {
            r.copy(&self.input, rows, &self.x, rows, n * d * 4);
        }
        let mut added = None;
        for l in &self.layers {
            let il = l.il;
            if !host {
                r.add_norm(x, added, Some(h), Some(&l.attn_norm), 0);
            }
            if added.is_some() {
                r.capture(format!("l_out-{}", il - 1), &self.x, d)?;
            }
            r.capture(format!("attn_norm-{il}"), &self.h, d)?;
            match &l.mixer {
                Mixer::Delta(dl) => r.delta(dl, il, verify)?,
                Mixer::Attention(a) => r.attention(a, parts, il)?,
            }
            r.end_segment()?;
            if !host {
                r.add_norm(x, Some(total), Some(h), Some(&l.post_norm), 0);
            }
            r.capture(format!("attn_residual-{il}"), &self.x, d)?;
            r.capture(format!("attn_post_norm-{il}"), &self.h, d)?;
            r.ffn(&l.ffn, il)?;
            r.end_segment()?;
            added = Some(total);
        }
        if !host {
            r.add_norm(x, added, Some(h), Some(&hd.norm), 0);
        }
        r.capture(format!("l_out-{}", self.last - 1), &self.x, d)?;
        r.capture("result_norm".into(), &self.h, d)?;
        let normed = if host { &tp.hin } else { &self.h };
        if k.head {
            r.head(hd, normed, if verify { n } else { 1 }, self.c.vocab)?;
        }
        if let Some(m) = &self.mtp {
            // Extend the MTP layer's KV cache over this batch (token p pairs with the target's row p - 1, which for
            // a later micro-batch the previous one left in row 0). Only its K and V are kept, so the block stops
            // after the attention and needs no all-reduce.
            r.h_in = h;
            r.copy(normed, r.half(normed), &m.hin, d * 4, n * d * 4);
            r.mtp_inputs(m, m.hin.address);
            r.matrix(&m.eh, m.cat.address, x, 2 * d, d, false)?;
            r.add_norm(x, None, Some(h), Some(&m.attn_norm), 0);
            r.attention(&m.attn, parts, self.c.layers)?;
            r.copy(&m.hin, n * d * 4, &m.hin, 0, d * 4);
        }
        r.finish()
    }

    /// Tensor-parallel MTP draft step: the attention and the FFN each end a segment with their partials, then
    /// the head ranks this rank's share of the first `rows` vocabulary ids (none when `rows` is 0).
    fn record_draft_tp(&self, parts: u32, first: bool, rows: usize, balanced: bool) -> Result<Graph> {
        let m = self.mtp.as_ref().context("no MTP block")?;
        let hd = if balanced { &self.draft_head.as_ref().context("no draft head")?.2 } else { self.head.as_ref().context("no head")? };
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
    /// micro-batch `j % m`, so the host can all-reduce one micro-batch while the GPU runs the next. A forward of
    /// one micro-batch of at most 8 columns is event-gated (`gated`). Returns the number of segments; the caller
    /// releases them in order, writing each all-reduced total before the segment of the same micro-batch that
    /// reads it.
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
            micro
                .iter()
                .scan(0, |next, &(row, rows)| {
                    let ok = row == *next && rows > 0;
                    *next = row + rows;
                    Some(ok)
                })
                .all(|ok| ok)
                && micro.last().is_some_and(|&(row, rows)| row + rows == n),
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
        let gated = micro.len() == 1 && gated(n);
        let mut keys = Vec::new();
        for (slot, &(row, rows)) in micro.iter().enumerate() {
            self.params_host.write_at(slot * PARAMS_STRIDE, bytemuck::cast_slice(&[(pos + row) as u32, rows as u32, 0, 0]))?;
            let key = Key { n: rows, parts: tp_partitions(pos + row, rows), keys: self.prefill_keys(pos + row, rows), dump, verify, slot, row, head: slot + 1 == micro.len(), gated };
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
        let graphs: Vec<&Graph> = keys.iter().map(|k| &self.graphs[k].0).collect();
        let segments = graphs[0].segments;
        ensure!(graphs.iter().all(|g| g.segments == segments), "micro-batch graphs differ in segments");
        let cmds: Vec<vk::CommandBuffer> =
            if gated { graphs[0].cmds.clone() } else { (0..segments).flat_map(|i| graphs.iter().map(move |g| g.cmds[i])).collect() };
        let stamps = graphs.iter().filter_map(|g| g.stamps.clone()).collect();
        let rows = self.c.vocab;
        let dual = micro.len() == 2 && self.gpu.queue2 != self.gpu.queue && two_queues();
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        tp.current = keys.last().copied();
        tp.stamps = stamps;
        tp.hostnorm = hostnorm(gated, dump);
        tp.head_start = self.vocab_start;
        tp.draft_top = false;
        Ok(tp.start(cmds, segments * micro.len(), rows, micro.len(), dual, gated))
    }

    fn prefill_keys(&self, pos: usize, n: usize) -> usize {
        if self.prefill_scores.is_some() && n >= 64 {
            prefill_key_limit(self.context, pos, n)
        } else { 0 }
    }

    /// Record full-chunk graphs at load, including the visible-key buckets of tiled attention.
    pub fn tp_warm(&mut self, n: usize, micro: &[(usize, usize)]) -> Result<()> {
        ensure!(n > 8, "only prefill batches are recorded ahead");
        for (slot, &(row, rows)) in micro.iter().enumerate() {
            let parts = tp_partitions(row, rows);
            let mut keys = self.prefill_keys(row, rows);
            loop {
                let key = Key { n: rows, parts, keys, dump: false, verify: false, slot, row, head: slot + 1 == micro.len(), gated: false };
                if !self.graphs.contains_key(&key) {
                    if self.graphs.len() >= GRAPH_CACHE_CAPACITY { break; }
                    let graph = self.record(key)?;
                    self.tick += 1;
                    self.graphs.insert(key, (graph, self.tick));
                }
                if keys == 0 || keys >= self.context.next_multiple_of(64) { break; }
                keys = (keys * 2).min(self.context.next_multiple_of(64));
            }
        }
        Ok(())
    }

    /// Start a tensor-parallel MTP draft step at `pos` for the token with embedding `emb` (see `draft`), the
    /// head ranking the first `rows` vocabulary ids (0: all). Returns the number of segments.
    pub fn tp_draft_prepare(&mut self, emb: &[f32], pos: usize, first: bool, rows: usize) -> Result<usize> {
        let m = self.mtp.as_ref().context("this rank has no MTP block")?;
        ensure!(pos < self.context, "context exhausted");
        m.emb.write(bytemuck::cast_slice(emb))?;
        self.params_host.write(bytemuck::cast_slice(&[pos as u32, 1, 0, 0]))?;
        let balanced = self.draft_head.as_ref().filter(|h| h.0 == rows && rows > 0).map(|h| (h.1, h.2.logits.size / 4));
        let (start, local) = match balanced {
            Some(b) => b,
            None => {
                let rows = if rows == 0 { usize::MAX } else { rows };
                (self.vocab_start, rows.saturating_sub(self.vocab_start).min(self.c.vocab))
            }
        };
        let key = (tp_partitions(pos, 1), first, local, balanced.is_some());
        if !self.drafts.contains_key(&key) {
            let graph = self.record_draft_tp(key.0, key.1, key.2, key.3)?;
            self.drafts.insert(key, graph);
        }
        let (cmds, segments) = (self.drafts[&key].cmds.clone(), self.drafts[&key].segments);
        let stamps = self.drafts[&key].stamps.iter().cloned().collect();
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        tp.stamps = stamps;
        tp.hostnorm = false;
        tp.head_start = start;
        tp.draft_top = balanced.is_some();
        Ok(tp.start(cmds, segments, local, 1, false, gated(1)))
    }

    /// Give this rank an equal share of the first `vocab` token ids for MTP drafts (a copy of those head rows),
    /// so draft heads take every rank instead of the one or two holding the low ids. Kept until the draft
    /// vocabulary changes; ignored for the whole vocabulary or one that does not split evenly
    /// (`TRASHCAN_TP_DRAFT_HEAD=0`: off).
    pub fn tp_draft_head(&mut self, g: &Gguf, vocab: usize) -> Result<()> {
        let (rank, ranks) = (self.split.rank, self.split.ranks);
        let full = self.c.vocab * ranks;
        if self.mtp.is_none() || ranks < 2 || vocab == 0 || vocab >= full || !vocab.is_multiple_of(16 * ranks) {
            return Ok(());
        }
        if std::env::var("TRASHCAN_TP_DRAFT_HEAD").as_deref() == Ok("0") || self.draft_head.as_ref().is_some_and(|h| h.0 == vocab) {
            return Ok(());
        }
        self.drafts.clear();
        let rows = vocab / ranks;
        let start = rank * rows;
        let t = g.tensor("output.weight").context("missing output.weight")?;
        let row = self.c.d / 256 * 210;
        let chunks = rows.div_ceil(1024);
        let gpu = &self.gpu;
        let head = Head {
            norm: gpu.buffer(4, false, false, None)?,
            w: gpu.upload_matrix(&quant::repack_q6(&g.data(t)[start * row..(start + rows) * row]))?,
            rows,
            logits: gpu.buffer(rows * 4, false, false, None)?,
            scratch: gpu.buffer(chunks * TOP_K * 8, false, false, None)?,
            scratch2: gpu.buffer(chunks.div_ceil(16) * TOP_K * 8, false, false, None)?,
            top: gpu.buffer(TOP_K * 8, true, false, None)?,
        };
        self.draft_head = Some((vocab, start, head));
        Ok(())
    }

    /// Submit segment `j` of the forward in progress on its queue, waiting for the host's release.
    fn tp_send(&mut self, j: usize) -> Result<()> {
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        let (m, i) = tp.at(j);
        let mut waits = [(tp.release[m], tp.base[m] + i), (vk::Semaphore::null(), 0)];
        let mut count = 1;
        let second = tp.dual && m == 1;
        if second {
            // The second micro-batch continues the first one's recurrent state and attention cache.
            waits[count] = (tp.done[0], tp.base[0] + i + 1);
            count += 1;
        }
        let done = (tp.done[m], tp.base[m] + i + 1);
        tp.done_value[m] = done.1;
        self.gpu.submit_on(if second { self.gpu.queue2 } else { self.gpu.queue }, tp.cmds[j], &waits[..count], &[done])
    }

    /// Whether the forward or draft step in progress is event-gated.
    pub fn tp_gated(&self) -> bool {
        self.tp.as_ref().is_some_and(|t| t.gated)
    }

    /// Submit every segment of the forward in progress ahead, each waiting for `tp_release`; an event-gated
    /// forward is one submission.
    pub fn tp_submit_all(&mut self) -> Result<()> {
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        if tp.gated {
            // The previous submission must have retired before its command buffers and events are reused.
            let (sems, values) = ([tp.done[0]], [tp.done_value[0]]);
            unsafe { self.gpu.dev.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values), 10_000_000_000)? };
            unsafe {
                self.gpu.dev.reset_event(tp.gate.0)?;
                self.gpu.dev.reset_event(tp.gate.1)?;
            }
            tp.done_value[0] = tp.base[0] + tp.segments as u64;
            return self.gpu.submit_all(&tp.cmds, (tp.done[0], tp.done_value[0]));
        }
        let count = tp.cmds.len();
        (0..count).try_for_each(|j| self.tp_send(j))
    }

    /// Let segment `j` (submitted ahead) and the earlier segments of its micro-batch run.
    pub fn tp_release(&mut self, j: usize) -> Result<()> {
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        if tp.gated {
            // Segment 0 starts with the submission; the others wait for the release event.
            if j > 0 {
                unsafe { self.gpu.dev.set_event(tp.gate.0)? };
            }
            return Ok(());
        }
        let (m, i) = tp.at(j);
        let value = tp.base[m] + i;
        if value > tp.released[m] {
            tp.released[m] = value;
            unsafe { self.gpu.dev.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(tp.release[m]).value(value))? };
        }
        Ok(())
    }

    /// Let a gated submission run to its end (its results are garbage): every segment start resets the release
    /// event, so keep setting it until the submission has retired.
    fn drain(&self) -> Result<()> {
        let tp = self.tp.as_ref().context("not a tensor-parallel rank")?;
        let start = std::time::Instant::now();
        loop {
            unsafe { self.gpu.dev.set_event(tp.gate.0)? };
            if unsafe { self.gpu.dev.get_semaphore_counter_value(tp.done[0])? } >= tp.done_value[0] {
                return Ok(());
            }
            ensure!(start.elapsed().as_secs() < 10, "a gated submission did not drain");
        }
    }

    /// Release every segment submitted ahead, e.g. after a failed exchange, so the queues drain.
    pub fn tp_release_all(&mut self) -> Result<()> {
        let tp = self.tp.as_mut().context("not a tensor-parallel rank")?;
        if tp.gated {
            return self.drain();
        }
        for m in 0..tp.micro {
            let value = tp.done_value[m].saturating_sub(1);
            if value >= tp.base[m] && value > tp.released[m] {
                tp.released[m] = value;
                unsafe { self.gpu.dev.signal_semaphore(&vk::SemaphoreSignalInfo::default().semaphore(tp.release[m]).value(value))? };
            }
        }
        Ok(())
    }

    /// Spin until segment `j` of the forward in progress has finished.
    pub fn tp_wait(&self, j: usize) -> Result<()> {
        let tp = self.tp.as_ref().context("not a tensor-parallel rank")?;
        let start = std::time::Instant::now();
        if tp.gated {
            while !unsafe { self.gpu.dev.get_event_status(tp.gate.1)? } {
                ensure!(start.elapsed().as_secs() < 60, "GPU segment did not finish");
                std::hint::spin_loop();
            }
            // Segment j + 1 can only set done again after the host releases it.
            unsafe { self.gpu.dev.reset_event(tp.gate.1)? };
            return Ok(());
        }
        let (m, i) = tp.at(j);
        let value = tp.base[m] + i + 1;
        while unsafe { self.gpu.dev.get_semaphore_counter_value(tp.done[m])? } < value {
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

    /// Whether the forward in progress reads its normed rows from the host (`hostnorm`).
    pub fn tp_hostnorm(&self) -> bool {
        self.tp.as_ref().is_some_and(|t| t.hostnorm)
    }

    /// The first `len` normed rows the next segment reads, when the host keeps the residual (host memory the
    /// GPU reads only while a segment runs).
    #[allow(clippy::mut_from_ref)]
    pub fn tp_hin(&self, len: usize) -> &mut [f32] {
        let t = &self.tp.as_ref().unwrap().hin;
        assert!(len * 4 <= t.size);
        unsafe { std::slice::from_raw_parts_mut(t.ptr as *mut f32, len) }
    }

    /// Where the host writes the all-reduced total of the `n` rows from `row` for the next segment.
    pub fn tp_total(&self, row: usize, n: usize) -> *mut f32 {
        let (t, d) = (&self.tp.as_ref().unwrap().total, self.c.d);
        assert!((row + n) * d * 4 <= t.size);
        unsafe { (t.ptr as *mut f32).add(row * d) }
    }

    /// Timestamps of the latest forward or draft step when profiling, one read per graph (micro-batch).
    pub fn tp_stamps(&self) -> Result<Vec<StampRead>> {
        let Some(tp) = self.tp.as_ref() else { return Ok(Vec::new()) };
        let period = self.gpu.period as f64;
        let mut reads = Vec::new();
        for (pool, labels) in &tp.stamps {
            let mut ticks = vec![0u64; labels.len()];
            unsafe { self.gpu.dev.get_query_pool_results(*pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
            let first = ticks.first().copied().unwrap_or(0);
            let host = self.gpu.calibrate().map(|(t, ns)| ns as f64 - (t as f64 - first as f64) * period);
            let ns = ticks.iter().map(|&t| (t as f64 - first as f64) * period).collect();
            reads.push((vk::Handle::as_raw(*pool), labels.clone(), ns, host));
        }
        Ok(reads)
    }

    /// Top candidates of this rank's vocabulary rows after the last segment, with model token ids.
    pub fn tp_top(&self, cols: usize) -> Result<Vec<Vec<(f32, u32)>>> {
        let tp = self.tp.as_ref().context("not a tensor-parallel rank")?;
        if tp.head_rows == 0 {
            return Ok(vec![Vec::new(); cols]);
        }
        let start = tp.head_start as u32;
        let hd = if tp.draft_top { &self.draft_head.as_ref().context("no draft head")?.2 } else { self.head.as_ref().context("no head")? };
        Ok(read_top(hd, cols)?.into_iter().map(|c| c.into_iter().map(|(s, id)| (s, id + start)).collect()).collect())
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
        read_top(self.head.as_ref().context("no head")?, cols)
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
        let mut key = Key::whole(n, partitions(pos, n), dump.is_some(), verify);
        key.keys = self.prefill_keys(pos, n);
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
        let key = (partitions(pos, 1), first, rows, false);
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

/// Candidates of `cols` columns from a head's selection, best first.
fn read_top(hd: &Head, cols: usize) -> Result<Vec<Vec<(f32, u32)>>> {
    let bytes = hd.top.read(cols * TOP_K * 8)?;
    Ok(bytes
        .chunks_exact(TOP_K * 8)
        .map(|col| {
            let mut top: Vec<(f32, u32)> = col
                .chunks_exact(8)
                .map(|s| (f32::from_le_bytes(s[..4].try_into().unwrap()), u32::from_le_bytes(s[4..].try_into().unwrap())))
                .collect();
            top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            top
        })
        .collect())
}

impl Drop for GpuStage {
    fn drop(&mut self) {
        unsafe {
            // A gated forward cut short must not leave the command processor waiting for its events.
            if let Some(tp) = &self.tp {
                unregister_gates(&self.gpu);
                if tp.gated {
                    let _ = self.drain();
                }
            }
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
                for &s in tp.done.iter().chain(&tp.release) {
                    self.gpu.dev.destroy_semaphore(s, None);
                }
                self.gpu.dev.destroy_event(tp.gate.0, None);
                self.gpu.dev.destroy_event(tp.gate.1, None);
            }
        };
    }
}

#[cfg(test)]
mod prefill_key_tests {
    use super::prefill_key_limit;
    #[test]
    fn visible_key_buckets_cover_microbatch_offsets_and_non_power_of_two_contexts() {
        for (context, pos, n, expected) in [(8192, 0, 256, 256), (8192, 256, 256, 512),
            (8192, 512, 256, 1024), (8192, 7936, 256, 8192), (12288, 8192, 256, 12288)] {
            let limit = prefill_key_limit(context, pos, n);
            assert_eq!(limit, expected);
            assert!(limit >= pos + n && limit <= context);
        }
    }
}
