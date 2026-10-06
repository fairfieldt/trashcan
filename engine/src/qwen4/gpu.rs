//! One GPU holding a contiguous range of qwen4exp layers (optionally the n-gram module of layer 1, and the final
//! mixer with the LM head). As in the qwen35 stages, each batch size records one command buffer per layer, replayed
//! with a new position; the residual (all hyper-connection streams) enters and leaves through host buffers.
//!
//! Every mixer applies the previous sublayer's write (`x_s += w_s * y`) in its norm kernel, so a layer ends with a
//! pending combine that the next mixer, the head or the stage output resolves.
//!
//! Matrices are Q4_0, repacked with f32 scales (`quant::repack_q4`); experts are stacked into one gate/up and one
//! down matrix per layer, with the shared expert appended as expert `experts`. Attention is dense, so a stage
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

/// A hyper-connection mixer: gamma `[hc][d]`, bottleneck down `rank x hc*d` and up `hc*d x rank` (Q4_0), write
/// weights `hc x hc*d` (f32; absent for the final mixer).
struct Mix {
    norm: Buffer,
    down: Buffer,
    up: Buffer,
    inject: Option<Buffer>,
}

enum Mixer {
    Delta {
        qkv: Buffer,
        z: Buffer,
        ab: Buffer,
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
    attn_mix: Mix,
    ffn_mix: Mix,
    mixer: Mixer,
    /// Router rows and the shared-expert gate as row `experts` (f32).
    router: Buffer,
    gate_up: Buffer,
    down: Buffer,
}

struct Ple {
    key: Buffer,
    value: Buffer,
    /// `[key | query | conv]` norm weights, each `hc x d`.
    norms: Buffer,
    conv_w: Buffer,
    hist: Buffer,
}

struct Head {
    mix: Mix,
    w: Buffer,
    logits: Buffer,
    scratch: Buffer,
    top: Buffer,
    chunks: usize,
}

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
    pub first: usize,
    pub last: usize,
    pub gpu: Rc<Device>,
    layers: Vec<Layer>,
    ple: Option<Ple>,
    head: Option<Head>,
    max_batch: usize,
    context: usize,
    input: Buffer,
    ple_in: Buffer,
    output: Buffer,
    params: Buffer,
    params_host: Buffer,
    x: Buffer,
    xn: Buffer,
    lo: Buffer,
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
    rq: Buffer,
    scratch: Buffer,
    logits: Buffer,
    ids: Buffer,
    wts: Buffer,
    act: Buffer,
    pe: Buffer,
    pk: Buffer,
    pv: Buffer,
    gv: Buffer,
    pn: Buffer,
    fence: vk::Fence,
    graphs: HashMap<(usize, u32, bool), Graph>,
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
        head: bool,
        context: usize,
        max_batch: usize,
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
            context <= c.idx_budget + c.kpool - 1,
            "dense attention only: context {context} exceeds {} tokens",
            c.idx_budget + c.kpool - 1
        );
        let (d, w) = (c.d, c.width());
        let t = |il: usize, name: &str| c.layer_tensor(g, il, name);
        let q4 = |t: &TensorInfo| -> Result<Buffer> {
            let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
            gpu.upload_matrix(&quant::repack_q4(q4_rows(g, t, 0, n)?, k, n, 0, n, 0, k))
        };
        let f32s = |v: &[f32]| gpu.upload(bytemuck::cast_slice(v));
        let mat32 = |v: &[f32]| gpu.upload_matrix(bytemuck::cast_slice(v));
        let vecb = |t: &TensorInfo| -> Result<Buffer> { f32s(&vector(g, t)?) };
        let mix = |prefix: &str, inject: bool| -> Result<Mix> {
            let tt = |name: &str| {
                let full = format!("{prefix}_{name}.weight");
                g.tensor(&full).with_context(|| format!("missing {full}"))
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
                let kv = context * c.kv_heads * c.head_dim * 2;
                Mixer::Attention {
                    q: q4(t(il, "attn_q.weight")?)?,
                    k: q4(t(il, "attn_k.weight")?)?,
                    v: q4(t(il, "attn_v.weight")?)?,
                    q_norm: vecb(t(il, "attn_q_norm.weight")?)?,
                    k_norm: vecb(t(il, "attn_k_norm.weight")?)?,
                    out: q4(t(il, "attn_output.weight")?)?,
                    kc: gpu.buffer(kv, false, false, None)?,
                    vc: gpu.buffer(kv, false, false, None)?,
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
            let mut gu = Vec::with_capacity((ne + 1) * 2 * ff * d / 32 * 18);
            for e in 0..ne {
                gu.extend_from_slice(q4_rows(g, gt, e * ff, ff)?);
                gu.extend_from_slice(q4_rows(g, ut, e * ff, ff)?);
            }
            gu.extend_from_slice(q4_rows(g, gs, 0, ff)?);
            gu.extend_from_slice(q4_rows(g, us, 0, ff)?);
            let rows = (ne + 1) * 2 * ff;
            let gate_up = gpu.upload_matrix(&quant::repack_q4(&gu, d, rows, 0, rows, 0, d))?;
            drop(gu);
            let mut dn = Vec::with_capacity((ne + 1) * d * ff / 32 * 18);
            dn.extend_from_slice(q4_rows(g, dt, 0, ne * d)?);
            dn.extend_from_slice(q4_rows(g, ds, 0, d)?);
            let rows = (ne + 1) * d;
            let down = gpu.upload_matrix(&quant::repack_q4(&dn, ff, rows, 0, rows, 0, ff))?;
            drop(dn);
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
        let head = if head {
            let t = g.tensor("output.weight").context("missing output.weight")?;
            ensure!(t.ty == GgmlType::Q6K, "output.weight must be Q6_K, found {}", t.ty);
            let chunks = c.vocab.div_ceil(1024);
            ensure!(chunks <= 256, "vocabulary too large for one candidate merge");
            Some(Head {
                mix: mix("output_hc", false)?,
                w: gpu.upload_matrix(&quant::repack_q6(g.data(t)))?,
                logits: gpu.buffer(c.vocab * 4, false, false, None)?,
                scratch: gpu.buffer(chunks * TOP_K * 8, false, false, None)?,
                top: gpu.buffer(TOP_K * 8, true, false, None)?,
                chunks,
            })
        } else {
            None
        };
        let alloc = |n: usize| gpu.buffer(n * max_batch * 4, false, false, None);
        let qd = c.heads * c.head_dim;
        let slots = c.used + 1;
        let fence = unsafe { gpu.dev.create_fence(&vk::FenceCreateInfo::default(), None)? };
        Ok(Self {
            first,
            last,
            layers,
            ple,
            head,
            max_batch,
            context,
            input: gpu.buffer(w * max_batch * 4, true, false, None)?,
            ple_in: gpu.buffer(d * max_batch * 4, true, false, None)?,
            output: gpu.buffer(w * max_batch * 4, true, false, None)?,
            params: gpu.buffer(16, false, false, None)?,
            params_host: gpu.buffer(16, true, false, None)?,
            x: alloc(w)?,
            xn: alloc(w)?,
            lo: alloc(c.hc_rank)?,
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
            rq: alloc(qd)?,
            scratch: gpu.buffer(c.heads * 64 * (c.head_dim + 2) * 4 * 8, false, false, None)?,
            logits: alloc(c.experts + 1)?,
            ids: alloc(slots)?,
            wts: alloc(slots)?,
            act: alloc(slots * c.ff_exp)?,
            pe: alloc(d)?,
            pk: alloc(w)?,
            pv: alloc(d)?,
            gv: alloc(w)?,
            pn: alloc(w)?,
            fence,
            graphs: HashMap::new(),
            c: c.clone(),
            gpu,
        })
    }

    pub fn device_mib(&self) -> f64 {
        self.gpu.device_bytes.get() as f64 / 1048576.
    }

    pub fn owns_ple(&self) -> bool {
        self.ple.is_some()
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
        };
        let cmd = current.get();
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
        // Q4_0 projections: the Gemma kernels (GEMM for prefill, multi-column for small batches).
        let matrix = |wt: &Buffer, x: &Buffer, y: &Buffer, k: usize, rows: usize, defer: bool| -> Result<()> {
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
        // Norm all streams into xn (applying the pending combine first), or with `norm` None only combine.
        let hc_norm = |pending: bool, norm: Option<&Buffer>| {
            let mut u = [0; 16];
            u[0] = d as u32;
            u[4] = hc as u32;
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
                        self.w.address,
                        0,
                    ],
                    u,
                ),
                1,
                n as u32,
            );
        };
        // A mixer: sublayer input in u and, with an inject matrix, write weights in w.
        let mixer = |graph: &mut Graph, m: &Mix, pending: bool, tag: &str| -> Result<()> {
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
            mixer(&mut graph, &l.attn_mix, pending, &format!("-{il}"))?;
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
            mixer(&mut graph, &l.ffn_mix, true, &format!("-{il}#2"))?;
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
            let mut u = [0; 16];
            u[0] = ff as u32;
            u[1] = d as u32;
            u[2] = slots as u32;
            u[3] = ((ne + 1) * 2 * ff * d / 32) as u32;
            gpu.dispatch_matrix(
                current.get(),
                "qwen4_moe_gu",
                push([0, self.u.address, self.act.address, self.ids.address, 0, 0], u),
                ff.div_ceil(16) as u32,
                (n * slots) as u32,
                &l.gate_up,
            )?;
            let mut u = [0; 16];
            u[0] = d as u32;
            u[1] = ff as u32;
            u[2] = slots as u32;
            u[3] = ((ne + 1) * d * ff / 32) as u32;
            gpu.dispatch_matrix(
                current.get(),
                "qwen4_moe_down",
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
            capture(&mut graph, format!("ffn_out-{il}"), &self.y, d)?;
            pending = true;
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
            mixer(&mut graph, &hd.mix, pending, "")?;
            capture(&mut graph, format!("l_last-{last}"), &self.x, w)?;
            capture(&mut graph, "result_norm".into(), &self.u, d)?;
            let mut u = [0; 16];
            u[0] = c.vocab as u32;
            u[1] = d as u32;
            u[3] = (c.vocab * d / 16) as u32;
            u[9] = 1;
            u[15] = 1;
            gpu.dispatch_matrix(
                current.get(),
                "headvec",
                push(
                    [0, self.u.address + ((n - 1) * d * 4) as u64, hd.logits.address, 0, 0, 0],
                    u,
                ),
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
                hc_norm(true, None);
            }
            capture(&mut graph, format!("l_last-{last}"), &self.x, w)?;
            gpu.copy(current.get(), &self.x, 0, &self.output, 0, n * w * 4);
        }
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
        unsafe { self.gpu.dev.destroy_fence(self.fence, None) };
    }
}
