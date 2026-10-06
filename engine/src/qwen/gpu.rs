//! One GPU holding a contiguous range of layers (and optionally the final norm, LM head and MTP block).
//! Each batch shape records one graph, replayed with a new position; the hidden state enters and leaves
//! through host-visible buffers so stages can be chained on other GPUs or machines.
//!
//! Recurrent state: verification batches (up to `verify` columns) make the DeltaNet and convolution
//! kernels also store the state after every column but the last, so `rollback` to an accepted prefix is a
//! device copy. Checkpoint slots copy the whole recurrent state (and the MTP's pending hidden row) to host
//! memory and back; attention KV needs no copy because it is addressed by position.
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
const GRAPH_CACHE_CAPACITY: usize = 16;

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
    chunks: usize,
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
}
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { self.gpu.dev.free_command_buffers(self.gpu.pool, &self.cmds) };
    }
}

pub enum Output {
    Hidden(Vec<f32>),
    /// Candidates per column: the last one, or every column of a verification batch.
    Top(Vec<Vec<(f32, u32)>>),
}

/// (batch, attention partitions, dump, verify)
type Key = (usize, u32, bool, bool);

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
    drafts: HashMap<(u32, bool), Graph>,
    tick: u64,
    /// Columns of the latest forward, and whether it kept per-column snapshots.
    last_n: usize,
    verified: bool,
    rollbacks: HashMap<usize, vk::CommandBuffer>,
    slots: HashMap<usize, Buffer>,
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
    let t = c.layer_tensor(g, il, name)?;
    ensure!(
        t.ty == GgmlType::Q4_0,
        "{} is {}; use a pure Q4_0 quant (docs/qwen.md)",
        t.name,
        t.ty
    );
    let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
    gpu.upload_matrix(&quant::repack_q4(g.data(t), k, n, 0, n, 0, k))
}

fn upload_f32(gpu: &Rc<Device>, v: &[f32]) -> Result<Buffer> {
    gpu.upload(bytemuck::cast_slice(v))
}

fn upload_vector(gpu: &Rc<Device>, g: &Gguf, c: &Config, il: usize, name: &str) -> Result<Buffer> {
    upload_f32(gpu, &vector(g, c.layer_tensor(g, il, name)?)?)
}

fn load_attn(gpu: &Rc<Device>, g: &Gguf, c: &Config, il: usize, context: usize) -> Result<Attn> {
    let kv = context * c.kv_heads * c.head_dim * 2;
    Ok(Attn {
        q: upload_q4(gpu, g, c, il, "attn_q.weight")?,
        k: upload_q4(gpu, g, c, il, "attn_k.weight")?,
        v: upload_q4(gpu, g, c, il, "attn_v.weight")?,
        q_norm: upload_vector(gpu, g, c, il, "attn_q_norm.weight")?,
        k_norm: upload_vector(gpu, g, c, il, "attn_k_norm.weight")?,
        out: upload_q4(gpu, g, c, il, "attn_output.weight")?,
        kc: gpu.buffer(kv, false, false, None)?,
        vc: gpu.buffer(kv, false, false, None)?,
    })
}

fn load_ffn(gpu: &Rc<Device>, g: &Gguf, c: &Config, il: usize) -> Result<Ffn> {
    Ok(Ffn {
        gate: upload_q4(gpu, g, c, il, "ffn_gate.weight")?,
        up: upload_q4(gpu, g, c, il, "ffn_up.weight")?,
        down: upload_q4(gpu, g, c, il, "ffn_down.weight")?,
    })
}

/// Records one graph: a sequence of command buffers sharing the stage's buffers.
struct Rec<'a> {
    s: &'a GpuStage,
    n: usize,
    dump: bool,
    cmd: vk::CommandBuffer,
    graph: Graph,
}

impl<'a> Rec<'a> {
    fn new(s: &'a GpuStage, n: usize, dump: bool) -> Result<Self> {
        let cmd = s.gpu.begin()?;
        s.gpu.barrier(cmd);
        s.gpu.copy(cmd, &s.params_host, 0, &s.params, 0, 16);
        Ok(Self {
            s,
            n,
            dump,
            cmd,
            graph: Graph {
                gpu: s.gpu.clone(),
                cmds: vec![cmd],
                captures: Vec::new(),
            },
        })
    }

    fn push(&self, ptr: [u64; 6], u: [u32; 16]) -> Push {
        Push {
            ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], self.s.params.address, 0],
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

    fn finish(self) -> Result<Graph> {
        self.s.gpu.finish(self.cmd)?;
        Ok(self.graph)
    }

    fn copy(&self, a: &Buffer, ao: usize, b: &Buffer, bo: usize, size: usize) {
        self.s.gpu.copy(self.cmd, a, ao, b, bo, size);
    }

    fn capture(&mut self, name: String, b: &Buffer, width: usize) -> Result<()> {
        if self.dump {
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
        let (shader, gx, gy) = match n {
            1 => (q4_decode_shader(rows, k), rows.div_ceil(16), 1),
            2 => ("multivec4x2", rows.div_ceil(16), 1),
            3 => ("multivec4x3", rows.div_ceil(16), 1),
            4..=8 => ("multivec4", rows.div_ceil(16), n.div_ceil(4)),
            _ => ("gemm", rows.div_ceil(64), n.div_ceil(64)),
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
        self.capture(format!("final_output-{il}"), &s.o, vd)?;
        self.matrix(&dl.out, s.o.address, s.y.address, vd, d, false)?;
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
        self.capture(format!("attn_gated-{il}"), &s.o, qd)?;
        self.matrix(&a.out, s.o.address, s.y.address, qd, d, false)?;
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
        self.capture(format!("ffn_swiglu-{il}"), &s.act, c.ff)?;
        self.matrix(&f.down, s.act.address, s.y.address, c.ff, c.d, false)
    }

    /// LM head and top candidates of the last `cols` normed rows of `input`.
    fn head(&mut self, hd: &Head, input: &Buffer, cols: usize) -> Result<()> {
        let (s, c) = (self.s, &self.s.c);
        let (d, n) = (c.d, self.n);
        let mut u = [0; 16];
        u[0] = c.vocab as u32;
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
            c.vocab.div_ceil(16) as u32,
            gy as u32,
            &hd.w,
        )?;
        if self.dump {
            let host = s.gpu.buffer(c.vocab * 4, true, false, None)?;
            s.gpu.copy(self.cmd, &hd.logits, (cols - 1) * c.vocab * 4, &host, 0, c.vocab * 4);
            self.graph.captures.push(("result_output".into(), host, c.vocab));
        }
        let mut u = [0; 16];
        u[0] = c.vocab as u32;
        u[1] = TOP_K as u32;
        u[2] = (hd.chunks * TOP_K) as u32;
        u[15] = single;
        s.gpu.dispatch(
            self.cmd,
            "topk",
            self.push([hd.logits.address, 0, hd.scratch.address, 0, 0, 0], u),
            hd.chunks as u32,
            cols as u32,
        );
        let mut u = [0; 16];
        u[0] = (hd.chunks * TOP_K) as u32;
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
        self.copy(&m.emb, 0, &s.x, 0, self.n * d * 4);
        self.add_norm(s.x.address, None, Some(m.cat.address), Some(&m.enorm), 2 * d);
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
        ensure!(c.state == 128, "DeltaNet kernels need 128-wide heads");
        ensure!(c.head_dim <= 512 && c.head_dim % 64 == 0, "unsupported head size");
        ensure!(verify <= 8, "verification batches are limited to eight columns");
        ensure!(mtp.is_none() || head, "the MTP block belongs on the head stage");
        let snaps = verify.saturating_sub(1);
        let mut layers = Vec::new();
        for il in first..last {
            let mixer = if c.recurrent(il) {
                let mut ab = vector(g, c.layer_tensor(g, il, "ssm_alpha.weight")?)?;
                ab.extend(vector(g, c.layer_tensor(g, il, "ssm_beta.weight")?)?);
                let mut dt_a = vector(g, c.layer_tensor(g, il, "ssm_dt.bias")?)?;
                dt_a.extend(vector(g, c.layer_tensor(g, il, "ssm_a")?)?);
                let (state, conv) = (c.v_heads * c.state * c.state * 4, (c.conv - 1) * c.conv_dim() * 4);
                Mixer::Delta(Delta {
                    qkv: upload_q4(&gpu, g, c, il, "attn_qkv.weight")?,
                    z: upload_q4(&gpu, g, c, il, "attn_gate.weight")?,
                    ab: gpu.upload_matrix(bytemuck::cast_slice(&ab))?,
                    dt_a: upload_f32(&gpu, &dt_a)?,
                    conv_w: upload_vector(&gpu, g, c, il, "ssm_conv1d.weight")?,
                    conv: gpu.buffer(conv, false, false, None)?,
                    state: gpu.buffer(state, false, false, None)?,
                    norm: upload_vector(&gpu, g, c, il, "ssm_norm.weight")?,
                    out: upload_q4(&gpu, g, c, il, "ssm_out.weight")?,
                    snaps: if snaps > 0 {
                        Some((gpu.buffer(snaps * state, false, false, None)?, gpu.buffer(snaps * conv, false, false, None)?))
                    } else {
                        None
                    },
                })
            } else {
                Mixer::Attention(load_attn(&gpu, g, c, il, context)?)
            };
            layers.push(Layer {
                il,
                attn_norm: upload_vector(&gpu, g, c, il, "attn_norm.weight")?,
                post_norm: upload_vector(&gpu, g, c, il, "post_attention_norm.weight")?,
                ffn: load_ffn(&gpu, g, c, il)?,
                mixer,
            });
        }
        let cols = verify.max(1);
        let head = if head {
            let t = g.tensor("output.weight").context("missing output.weight")?;
            ensure!(t.ty == GgmlType::Q6K, "output.weight must be Q6_K, found {}", t.ty);
            let chunks = c.vocab.div_ceil(1024);
            ensure!(chunks <= 256, "vocabulary too large for one candidate merge");
            Some(Head {
                norm: upload_f32(&gpu, &vector(g, g.tensor("output_norm.weight").context("missing output norm")?)?)?,
                w: gpu.upload_matrix(&quant::repack_q6(g.data(t)))?,
                logits: gpu.buffer(cols * c.vocab * 4, false, false, None)?,
                scratch: gpu.buffer(cols * chunks * TOP_K * 8, false, false, None)?,
                top: gpu.buffer(cols * TOP_K * 8, true, false, None)?,
                chunks,
            })
        } else {
            None
        };
        let mtp = match mtp {
            Some(m) => {
                let il = super::mtp_layer(m, c)?;
                let v = |name: &str| upload_vector(&gpu, m, c, il, name);
                Some(Mtp {
                    enorm: v("nextn.enorm.weight")?,
                    hnorm: v("nextn.hnorm.weight")?,
                    eh: upload_q4(&gpu, m, c, il, "nextn.eh_proj.weight")?,
                    attn_norm: v("attn_norm.weight")?,
                    post_norm: v("post_attention_norm.weight")?,
                    head_norm: v("nextn.shared_head_norm.weight")?,
                    attn: load_attn(&gpu, m, c, il, context)?,
                    ffn: load_ffn(&gpu, m, c, il)?,
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
            params: gpu.buffer(16, false, false, None)?,
            params_host: gpu.buffer(16, true, false, None)?,
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
            c: c.clone(),
            gpu,
        })
    }

    pub fn device_mib(&self) -> f64 {
        self.gpu.device_bytes.get() as f64 / 1048576.
    }

    fn record(&self, n: usize, parts: u32, dump: bool, verify: bool) -> Result<Graph> {
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
            r.head(hd, &self.h, if verify { n } else { 1 })?;
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

    /// One chained MTP draft step; the first takes the target's pending row, later ones the last output.
    fn record_draft(&self, parts: u32, first: bool) -> Result<Graph> {
        let (m, hd) = (self.mtp.as_ref().context("no MTP block")?, self.head.as_ref().context("no head")?);
        let mut r = Rec::new(self, 1, false)?;
        r.mtp_inputs(m, if first { m.hin.address } else { m.out.address });
        r.mtp_block(m, parts)?;
        r.add_norm(self.x.address, Some(self.y.address), Some(m.out.address), Some(&m.head_norm), 0);
        r.head(hd, &m.out, 1)?;
        r.finish()
    }

    fn submit(&self, cmds: &[vk::CommandBuffer]) -> Result<()> {
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
        let key = (n, partitions(pos, n), dump.is_some(), verify);
        self.tick += 1;
        if let Some(e) = self.graphs.get_mut(&key) {
            e.1 = self.tick;
        } else {
            if self.graphs.len() >= GRAPH_CACHE_CAPACITY {
                // Every graph has completed: forward waits for its fence before returning.
                let victim = *self.graphs.iter().min_by_key(|e| e.1 .1).unwrap().0;
                self.graphs.remove(&victim);
            }
            let graph = self.record(n, key.1, key.2, key.3)?;
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
    /// target's state. Returns the draft's top candidates.
    pub fn draft(&mut self, emb: &[f32], pos: usize, first: bool) -> Result<Vec<(f32, u32)>> {
        let m = self.mtp.as_ref().context("this stage has no MTP block")?;
        ensure!(pos < self.context, "context exhausted");
        m.emb.write(bytemuck::cast_slice(emb))?;
        self.params_host
            .write(bytemuck::cast_slice(&[pos as u32, 1, 0, 0]))?;
        let key = (partitions(pos, 1), first);
        if !self.drafts.contains_key(&key) {
            let graph = self.record_draft(key.0, key.1)?;
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
            self.gpu.dev.destroy_fence(self.fence, None)
        };
    }
}
