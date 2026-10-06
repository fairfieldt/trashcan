//! One GPU holding a contiguous range of layers (and optionally the final norm and LM head). Each batch
//! size records one command buffer, replayed with a new position; the hidden state enters and leaves
//! through host-visible buffers so stages can be chained on other GPUs or machines.
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
const GRAPH_CACHE_CAPACITY: usize = 6;

enum Mixer {
    Delta {
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
    },
    Attention {
        q: Buffer,
        k: Buffer,
        v: Buffer,
        q_norm: Buffer,
        k_norm: Buffer,
        out: Buffer,
        kc: Buffer,
        vc: Buffer,
    },
}

struct Layer {
    il: usize,
    attn_norm: Buffer,
    post_norm: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
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
    Top(Vec<(f32, u32)>),
}

pub struct GpuStage {
    c: Config,
    last: usize,
    pub gpu: Rc<Device>,
    layers: Vec<Layer>,
    head: Option<Head>,
    max_batch: usize,
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
    graphs: HashMap<(usize, u32, bool), Graph>,
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

impl GpuStage {
    pub fn new(
        gpu: Rc<Device>,
        g: &Gguf,
        c: &Config,
        first: usize,
        last: usize,
        head: bool,
        context: usize,
        max_batch: usize,
    ) -> Result<Self> {
        ensure!(c.state == 128, "DeltaNet kernels need 128-wide heads");
        ensure!(c.head_dim <= 512 && c.head_dim % 64 == 0, "unsupported head size");
        let q4 = |il: usize, name: &str| -> Result<Buffer> {
            let t = c.layer_tensor(g, il, name)?;
            ensure!(
                t.ty == GgmlType::Q4_0,
                "{} is {}; use a pure Q4_0 quant (docs/qwen.md)",
                t.name,
                t.ty
            );
            let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
            gpu.upload_matrix(&quant::repack_q4(g.data(t), k, n, 0, n, 0, k))
        };
        let f32s = |v: &[f32]| gpu.upload(bytemuck::cast_slice(v));
        let vec = |il: usize, name: &str| -> Result<Buffer> { f32s(&vector(g, c.layer_tensor(g, il, name)?)?) };
        let mut layers = Vec::new();
        for il in first..last {
            let mixer = if c.recurrent(il) {
                let mut ab = vector(g, c.layer_tensor(g, il, "ssm_alpha.weight")?)?;
                ab.extend(vector(g, c.layer_tensor(g, il, "ssm_beta.weight")?)?);
                let mut dt_a = vector(g, c.layer_tensor(g, il, "ssm_dt.bias")?)?;
                dt_a.extend(vector(g, c.layer_tensor(g, il, "ssm_a")?)?);
                Mixer::Delta {
                    qkv: q4(il, "attn_qkv.weight")?,
                    z: q4(il, "attn_gate.weight")?,
                    ab: gpu.upload_matrix(bytemuck::cast_slice(&ab))?,
                    dt_a: f32s(&dt_a)?,
                    conv_w: vec(il, "ssm_conv1d.weight")?,
                    conv: gpu.buffer((c.conv - 1) * c.conv_dim() * 4, false, false, None)?,
                    state: gpu.buffer(c.v_heads * c.state * c.state * 4, false, false, None)?,
                    norm: vec(il, "ssm_norm.weight")?,
                    out: q4(il, "ssm_out.weight")?,
                }
            } else {
                let kv = context * c.kv_heads * c.head_dim * 2;
                Mixer::Attention {
                    q: q4(il, "attn_q.weight")?,
                    k: q4(il, "attn_k.weight")?,
                    v: q4(il, "attn_v.weight")?,
                    q_norm: vec(il, "attn_q_norm.weight")?,
                    k_norm: vec(il, "attn_k_norm.weight")?,
                    out: q4(il, "attn_output.weight")?,
                    kc: gpu.buffer(kv, false, false, None)?,
                    vc: gpu.buffer(kv, false, false, None)?,
                }
            };
            layers.push(Layer {
                il,
                attn_norm: vec(il, "attn_norm.weight")?,
                post_norm: vec(il, "post_attention_norm.weight")?,
                gate: q4(il, "ffn_gate.weight")?,
                up: q4(il, "ffn_up.weight")?,
                down: q4(il, "ffn_down.weight")?,
                mixer,
            });
        }
        let head = if head {
            let t = g.tensor("output.weight").context("missing output.weight")?;
            ensure!(t.ty == GgmlType::Q6K, "output.weight must be Q6_K, found {}", t.ty);
            let chunks = c.vocab.div_ceil(1024);
            ensure!(chunks <= 256, "vocabulary too large for one candidate merge");
            Some(Head {
                norm: f32s(&vector(g, g.tensor("output_norm.weight").context("missing output norm")?)?)?,
                w: gpu.upload_matrix(&quant::repack_q6(g.data(t)))?,
                logits: gpu.buffer(c.vocab * 4, false, false, None)?,
                scratch: gpu.buffer(chunks * TOP_K * 8, false, false, None)?,
                top: gpu.buffer(TOP_K * 8, true, false, None)?,
                chunks,
            })
        } else {
            None
        };
        let alloc = |w: usize| gpu.buffer(w * max_batch * 4, false, false, None);
        let fence = unsafe { gpu.dev.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let qd = c.heads * c.head_dim;
        Ok(Self {
            last,
            layers,
            head,
            max_batch,
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
            c: c.clone(),
            gpu,
        })
    }

    pub fn device_mib(&self) -> f64 {
        self.gpu.device_bytes.get() as f64 / 1048576.
    }

    fn record(&self, n: usize, parts: u32, dump: bool) -> Result<Graph> {
        let gpu = &self.gpu;
        let c = &self.c;
        let d = c.d;
        let current = std::cell::Cell::new(gpu.begin()?);
        let mut graph = Graph {
            gpu: gpu.clone(),
            cmds: vec![current.get()],
            captures: Vec::new(),
        };
        let cmd = current.get();
        gpu.barrier(cmd);
        gpu.copy(cmd, &self.params_host, 0, &self.params, 0, 16);
        gpu.copy(cmd, &self.input, 0, &self.x, 0, n * d * 4);
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
        // Q4_0 projections reuse the Gemma kernels: GEMM for prefill, multi-column for small batches.
        let matrix = |w: &Buffer, x: &Buffer, y: &Buffer, k: usize, rows: usize, defer: bool| -> Result<()> {
            let mut u = [0; 16];
            u[0] = rows as u32;
            u[1] = k as u32;
            u[3] = (rows * k / 32) as u32;
            let (shader, gx, gy) = match n {
                1 => (q4_decode_shader(rows, k), rows.div_ceil(16), 1),
                2 => ("multivec4x2", rows.div_ceil(16), 1),
                3 => ("multivec4x3", rows.div_ceil(16), 1),
                4..=8 => ("multivec4", rows.div_ceil(16), n.div_ceil(4)),
                _ => ("gemm", rows.div_ceil(64), n.div_ceil(64)),
            };
            gpu.dispatch_matrix(
                current.get(),
                shader,
                push([0, x.address, y.address, 0, 0, 0], u).defer_barrier(defer),
                gx as u32,
                gy as u32,
                w,
            )
        };
        let add_norm = |x: &Buffer, y: Option<&Buffer>, h: Option<&Buffer>, w: Option<&Buffer>| {
            let mut u = [0; 16];
            u[0] = d as u32;
            u[11] = u32::from(y.is_some());
            u[12] = eps;
            gpu.dispatch(
                current.get(),
                "qwen_add_norm",
                push(
                    [
                        x.address,
                        y.map_or(0, |b| b.address),
                        h.map_or(0, |b| b.address),
                        w.map_or(0, |b| b.address),
                        0,
                        0,
                    ],
                    u,
                ),
                1,
                n as u32,
            );
        };
        let mut pending = false;
        for l in &self.layers {
            let il = l.il;
            add_norm(&self.x, pending.then_some(&self.y), Some(&self.h), Some(&l.attn_norm));
            if pending {
                capture(&mut graph, format!("l_out-{}", il - 1), &self.x, d)?;
            }
            capture(&mut graph, format!("attn_norm-{il}"), &self.h, d)?;
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
                    matrix(qkv, &self.h, &self.big, d, cd, true)?;
                    matrix(z, &self.h, &self.z, d, vd, true)?;
                    let mut u = [0; 16];
                    u[0] = (2 * c.v_heads) as u32;
                    u[1] = d as u32;
                    gpu.dispatch_matrix(
                        current.get(),
                        "qwen_matf32",
                        push([0, self.h.address, self.ab.address, 0, 0, 0], u),
                        (2 * c.v_heads).div_ceil(4) as u32,
                        n.div_ceil(8) as u32,
                        ab,
                    )?;
                    capture(&mut graph, format!("linear_attn_qkv_mixed-{il}"), &self.big, cd)?;
                    capture(&mut graph, format!("z-{il}"), &self.z, vd)?;
                    let mut u = [0; 16];
                    u[0] = cd as u32;
                    gpu.dispatch(
                        current.get(),
                        "qwen_conv",
                        push([self.big.address, conv_w.address, conv.address, self.conv_out.address, 0, 0], u),
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
                            [self.conv_out.address, self.ab.address, state.address, self.o.address, dt_a.address, 0],
                            u,
                        ),
                        (c.v_heads * c.state / 16) as u32,
                        1,
                    );
                    capture(&mut graph, format!("attn_output-{il}"), &self.o, vd)?;
                    let mut u = [0; 16];
                    u[0] = vd as u32;
                    u[6] = c.state as u32;
                    u[12] = eps;
                    gpu.dispatch(
                        current.get(),
                        "qwen_gated_norm",
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
                } => {
                    let (hd, kvd) = (c.head_dim, c.kv_heads * c.head_dim);
                    let qd = c.heads * hd;
                    matrix(q, &self.h, &self.big, d, 2 * qd, true)?;
                    matrix(k, &self.h, &self.kb, d, kvd, true)?;
                    matrix(v, &self.h, &self.vb, d, kvd, false)?;
                    // Q (normed, rotated, pre-scaled into rq), K (normed, rotated into the cache), V (cache).
                    for (src, norm, out, cache, heads, stride, width, scale, defer) in [
                        (&self.big, Some(q_norm), Some(&self.rq), None, c.heads, 2 * hd, 2 * qd, 1. / (hd as f32).sqrt(), true),
                        (&self.kb, Some(k_norm), None, Some(kc), c.kv_heads, hd, kvd, 1., true),
                        (&self.vb, None, None, Some(vc), c.kv_heads, hd, kvd, 1f32, false),
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
                    capture(&mut graph, format!("Qcur-{il}"), &self.rq, qd)?;
                    let mut u = [0; 16];
                    u[4] = c.heads as u32;
                    u[5] = c.kv_heads as u32;
                    u[6] = hd as u32;
                    u[7] = self.context as u32;
                    if parts == 0 {
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
                                [self.rq.address, kc.address, vc.address, self.o.address, self.scratch.address, 0],
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
            add_norm(&self.x, Some(&self.y), Some(&self.h), Some(&l.post_norm));
            capture(&mut graph, format!("attn_residual-{il}"), &self.x, d)?;
            capture(&mut graph, format!("attn_post_norm-{il}"), &self.h, d)?;
            matrix(&l.gate, &self.h, &self.gate, d, c.ff, true)?;
            matrix(&l.up, &self.h, &self.up, d, c.ff, false)?;
            let mut u = [0; 16];
            u[0] = c.ff as u32;
            gpu.dispatch(
                current.get(),
                "qwen_swiglu",
                push([self.gate.address, self.up.address, self.act.address, 0, 0, 0], u),
                c.ff.div_ceil(256) as u32,
                n as u32,
            );
            capture(&mut graph, format!("ffn_swiglu-{il}"), &self.act, c.ff)?;
            matrix(&l.down, &self.act, &self.y, c.ff, d, false)?;
            capture(&mut graph, format!("ffn_out-{il}"), &self.y, d)?;
            pending = true;
            // Later layers start a new submission; the leading barrier orders it after this one.
            if l.il + 1 < self.last {
                gpu.finish(current.get())?;
                let next = gpu.begin()?;
                gpu.barrier(next);
                graph.cmds.push(next);
                current.set(next);
            }
        }
        let last = self.last - 1;
        if let Some(hd) = &self.head {
            add_norm(&self.x, pending.then_some(&self.y), Some(&self.h), Some(&hd.norm));
            capture(&mut graph, format!("l_out-{last}"), &self.x, d)?;
            capture(&mut graph, "result_norm".into(), &self.h, d)?;
            let mut u = [0; 16];
            u[0] = c.vocab as u32;
            u[1] = d as u32;
            u[3] = (c.vocab * d / 16) as u32;
            u[9] = 1;
            u[15] = 1;
            gpu.dispatch_matrix(
                current.get(),
                "headvec",
                push([0, self.h.address + ((n - 1) * d * 4) as u64, hd.logits.address, 0, 0, 0], u),
                c.vocab.div_ceil(16) as u32,
                1,
                &hd.w,
            )?;
            if dump {
                let host = gpu.buffer(c.vocab * 4, true, false, None)?;
                gpu.copy(current.get(), &hd.logits, 0, &host, 0, c.vocab * 4);
                graph.captures.push(("result_output".into(), host, c.vocab));
            }
            let mut u = [0; 16];
            u[0] = c.vocab as u32;
            u[1] = TOP_K as u32;
            u[2] = (hd.chunks * TOP_K) as u32;
            u[15] = 1;
            gpu.dispatch(
                current.get(),
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
                current.get(),
                "merge",
                push([hd.scratch.address, 0, hd.top.address, 0, 0, 0], u),
                1,
                1,
            );
        } else {
            if pending {
                add_norm(&self.x, Some(&self.y), None, None);
            }
            capture(&mut graph, format!("l_out-{last}"), &self.x, d)?;
            gpu.copy(current.get(), &self.x, 0, &self.output, 0, n * d * 4);
        }
        gpu.finish(current.get())?;
        Ok(graph)
    }

    /// Run `n` tokens at `pos`; position 0 starts a new sequence (DeltaNet states restart from zero).
    pub fn forward(
        &mut self,
        x: &[f32],
        pos: usize,
        n: usize,
        dump: Option<&mut Captures>,
    ) -> Result<Output> {
        let d = self.c.d;
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(pos + n <= self.context, "context exhausted");
        ensure!(x.len() == n * d, "hidden state length mismatch");
        self.input.write(bytemuck::cast_slice(x))?;
        self.params_host
            .write(bytemuck::cast_slice(&[pos as u32, n as u32, 0, 0]))?;
        let key = (n, partitions(pos, n), dump.is_some());
        if !self.graphs.contains_key(&key) {
            if self.graphs.len() >= GRAPH_CACHE_CAPACITY {
                // Every graph has completed: forward waits for its fence before returning.
                let victim = *self.graphs.keys().find(|k| k.0 != 1).unwrap_or(self.graphs.keys().next().unwrap());
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
            self.gpu
                .dev
                .wait_for_fences(&[self.fence], true, 600_000_000_000)?;
        }
        if let Some(m) = dump {
            for (name, host, width) in &graph.captures {
                let v: Vec<f32> = bytemuck::cast_slice(&host.read(width * 4)?).to_vec();
                m.insert(name.clone(), v);
            }
        }
        Ok(match &self.head {
            Some(hd) => {
                let bytes = hd.top.read(TOP_K * 8)?;
                let mut top: Vec<(f32, u32)> = bytes
                    .chunks_exact(8)
                    .map(|s| {
                        (
                            f32::from_le_bytes(s[..4].try_into().unwrap()),
                            u32::from_le_bytes(s[4..].try_into().unwrap()),
                        )
                    })
                    .collect();
                top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                Output::Top(top)
            }
            None => Output::Hidden(bytemuck::cast_slice(&self.output.read(n * d * 4)?).to_vec()),
        })
    }
}

impl Drop for GpuStage {
    fn drop(&mut self) {
        unsafe {
            let _ = self.gpu.dev.device_wait_idle();
        }
        self.graphs.clear();
        unsafe { self.gpu.dev.destroy_fence(self.fence, None) };
    }
}
