//! Gemma4 tensor-parallel graph; command buffers are immutable between tokens.
use crate::{
    gguf::{GgmlType, Gguf},
    gpu::Instance,
    quant,
    vulkan::{Buffer, Device, Push},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::{collections::HashMap, path::Path, rc::Rc};
const D: usize = 3840;
const F: usize = 7680;
const V: usize = 131072;
const L: usize = 48;
// 96 receive buffers at 128 tokens occupy 180 MiB, within the 256 MiB peer BAR.
const P2P_BATCH: usize = 128;

pub(crate) struct Layer {
    w: HashMap<String, Buffer>,
    pub k: Buffer,
    pub v: Buffer,
    pub dim: usize,
    pub kv: usize,
    pub cap: usize,
    pub sliding: bool,
}
pub(crate) struct Card {
    pub gpu: Rc<Device>,
    pub layers: Vec<Layer>,
    head: Buffer,
    output_norm: Buffer,
    pub rope: Buffer,
    input: Buffer,
    params: Buffer,
    params_host: Buffer,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    nq: Buffer,
    nk: Buffer,
    nv: Buffer,
    rq: Buffer,
    rk: Buffer,
    o: Buffer,
    pub attn_scratch: Buffer,
    gate: Buffer,
    up: Buffer,
    geglu: Buffer,
    partial: Buffer,
    post: Buffer,
    recv: Vec<Buffer>,
    wide_recv: Vec<Buffer>,
    wide_peer: Vec<Buffer>,
    dummy_recv: Buffer,
    relay_recv: Buffer,
    relay_in: Buffer,
    relay_out: Buffer,
    peer: Vec<Buffer>,
    logits: Buffer,
    top_scratch: Buffer,
    top: Buffer,
    own_sem: vk::Semaphore,
    peer_sem: vk::Semaphore,
    fence: vk::Fence,
    queries: vk::QueryPool,
}
struct Capture {
    name: String,
    host: Buffer,
    len: usize,
    merge: u8,
}
struct Graph {
    cmds: Vec<vk::CommandBuffer>,
    captures: Vec<Capture>,
}
pub struct Engine {
    pub gguf: Gguf,
    pub(crate) cards: Vec<Card>,
    graphs: HashMap<(usize, bool, bool, usize), Vec<Graph>>,
    max_batch: usize,
    context: usize,
    pub pos: usize,
    last_batch: usize,
    top_k: usize,
    counter: u64,
    cpu_reduce: bool,
    pub profile: Vec<[f64; 2]>,
    pub gaps: Vec<[f64; 2]>,
    pub timing: [f64; 3],
}

impl Engine {
    pub fn new(path: &Path, context: usize, max_batch: usize, cpu_reduce: bool) -> Result<Self> {
        ensure!(
            context > 0 && context <= 262144 && max_batch > 0 && max_batch <= 512,
            "invalid context (1..262144) or batch (1..512)"
        );
        let gguf = Gguf::open(path)?;
        ensure!(
            gguf.get_str("general.architecture")? == "gemma4",
            "only gemma4 is supported"
        );
        ensure!(
            gguf.get_u64("gemma4.embedding_length")? == D as u64
                && gguf.get_u64("gemma4.block_count")? == L as u64,
            "unsupported Gemma4 dimensions"
        );
        let pattern = gguf
            .get("gemma4.attention.sliding_window_pattern")?
            .as_arr()
            .context("missing sliding pattern")?;
        ensure!(pattern.len() == L, "sliding pattern must have 48 entries");
        validate(&gguf)?;
        let inst = Rc::new(Instance::new()?);
        let devices = inst.amd_gpus()?;
        ensure!(devices.len() == 2, "expected two discrete AMD GPUs");
        let mut cards = Vec::new();
        for (g, pd) in devices.into_iter().enumerate() {
            let gpu = Device::new(inst.clone(), pd)?;
            let mut layers = Vec::new();
            eprintln!("GPU{g}: uploading sharded weights");
            for (il, pat) in pattern.iter().enumerate() {
                let sliding = pat.as_u64().context("invalid sliding pattern")? != 0;
                let dim = if sliding { 256 } else { 512 };
                let kv = if sliding { 4 } else { 1 };
                let cap = if sliding { 1024 + max_batch } else { context };
                let mut w = HashMap::new();
                for name in [
                    "attn_norm",
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_q_norm",
                    "attn_k_norm",
                    "attn_output",
                    "post_attention_norm",
                    "ffn_norm",
                    "ffn_gate",
                    "ffn_up",
                    "ffn_down",
                    "post_ffw_norm",
                    "layer_output_scale",
                ] {
                    if !sliding && name == "attn_v" {
                        continue;
                    }
                    let t = gguf
                        .tensor(&format!("blk.{il}.{name}.weight"))
                        .with_context(|| format!("missing layer {il} {name}"))?;
                    let bytes = if t.ty == GgmlType::Q4_0 {
                        let k = t.dims[0] as usize;
                        let n = t.dims[1] as usize;
                        let (row, rows, col, cols) = match name {
                            "attn_q" | "ffn_gate" | "ffn_up" => (g * n / 2, n / 2, 0, k),
                            "attn_k" | "attn_v" if sliding => (g * n / 2, n / 2, 0, k),
                            "attn_output" | "ffn_down" => (0, n, g * k / 2, k / 2),
                            _ => (0, n, 0, k),
                        };
                        quant::repack_q4(gguf.data(t), k, n, row, rows, col, cols)
                    } else {
                        ensure!(t.ty == GgmlType::F32, "unexpected type for {}", t.name);
                        gguf.data(t).to_vec()
                    };
                    w.insert(
                        name.to_string(),
                        if t.ty == GgmlType::Q4_0 {
                            gpu.upload_matrix(&bytes)?
                        } else {
                            gpu.upload(&bytes)?
                        },
                    );
                }
                layers.push(Layer {
                    w,
                    k: gpu.buffer(cap * kv * dim * 2, false, false, None)?,
                    v: gpu.buffer(cap * kv * dim * 2, false, false, None)?,
                    dim,
                    kv,
                    cap,
                    sliding,
                });
            }
            let emb = gguf
                .tensor("token_embd.weight")
                .context("missing embedding")?;
            ensure!(
                emb.ty == GgmlType::Q6K && emb.dims == [D as u64, (V * 2) as u64],
                "unsupported embedding"
            );
            let half = emb.size as usize / 2;
            let head =
                gpu.upload_matrix(&quant::repack_q6(&gguf.data(emb)[g * half..(g + 1) * half]))?;
            let output_norm = gpu.upload(
                gguf.data(
                    gguf.tensor("output_norm.weight")
                        .context("missing output norm")?,
                ),
            )?;
            let rope = gpu.upload(
                gguf.data(
                    gguf.tensor("rope_freqs.weight")
                        .context("missing rope factors")?,
                ),
            )?;
            let alloc = |n: usize| gpu.buffer(n * max_batch * 4, false, false, None);
            let own_sem = gpu.timeline(true)?;
            let peer_sem = gpu.timeline(false)?;
            let fence = unsafe {
                gpu.dev
                    .create_fence(&vk::FenceCreateInfo::default(), None)?
            };
            let queries = unsafe {
                gpu.dev.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(194),
                    None,
                )?
            };
            cards.push(Card {
                layers,
                head,
                output_norm,
                rope,
                input: gpu.buffer(D * max_batch * 4, true, false, None)?,
                params: gpu.buffer(16, false, false, None)?,
                params_host: gpu.buffer(16, true, false, None)?,
                x: alloc(D)?,
                h: alloc(D)?,
                q: alloc(4096)?,
                k: alloc(1024)?,
                v: alloc(1024)?,
                nq: alloc(4096)?,
                nk: alloc(1024)?,
                nv: alloc(1024)?,
                rq: alloc(4096)?,
                rk: alloc(1024)?,
                o: alloc(4096)?,
                attn_scratch: gpu.buffer(8 * 8 * 32 * 514 * 4, false, false, None)?,
                gate: alloc(F)?,
                up: alloc(F)?,
                geglu: alloc(F)?,
                partial: alloc(D)?,
                post: alloc(D)?,
                recv: (0..if cpu_reduce { 0 } else { 96 })
                    .map(|_| gpu.external_buffer(D * max_batch.min(P2P_BATCH) * 4, None))
                    .collect::<Result<Vec<_>>>()?,
                wide_recv: (0..if !cpu_reduce && max_batch > P2P_BATCH {
                    2
                } else {
                    0
                })
                    .map(|_| gpu.external_buffer(D * max_batch * 4, None))
                    .collect::<Result<Vec<_>>>()?,
                wide_peer: Vec::new(),
                relay_recv: gpu.buffer(D * max_batch * 4, false, false, None)?,
                relay_in: gpu.buffer(D * max_batch * 4, true, false, None)?,
                relay_out: gpu.buffer(D * max_batch * 4, true, false, None)?,
                dummy_recv: gpu.buffer(4, false, false, None)?,
                peer: Vec::new(),
                logits: gpu.buffer(V * max_batch.min(8) * 4, false, false, None)?,
                top_scratch: gpu.buffer(128 * 64 * max_batch.min(8) * 8, false, false, None)?,
                top: gpu.buffer(64 * max_batch.min(8) * 8, true, false, None)?,
                own_sem,
                peer_sem,
                fence,
                queries,
                gpu: gpu.clone(),
            });
        }
        for g in 0..if cpu_reduce { 0 } else { 2 } {
            let other = 1 - g;
            let imported = cards[other]
                .recv
                .iter()
                .map(|b| cards[g].gpu.external_buffer(b.size, Some(b.export()?)))
                .collect::<Result<Vec<_>>>()?;
            cards[g].peer = imported;
            let wide = cards[other]
                .wide_recv
                .iter()
                .map(|b| cards[g].gpu.external_buffer(b.size, Some(b.export()?)))
                .collect::<Result<Vec<_>>>()?;
            cards[g].wide_peer = wide;
            let exp =
                ash::khr::external_semaphore_fd::Device::new(&inst.instance, &cards[other].gpu.dev);
            let fd = unsafe {
                exp.get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(cards[other].own_sem)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
                )?
            };
            let imp =
                ash::khr::external_semaphore_fd::Device::new(&inst.instance, &cards[g].gpu.dev);
            unsafe {
                imp.import_semaphore_fd(
                    &vk::ImportSemaphoreFdInfoKHR::default()
                        .semaphore(cards[g].peer_sem)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                        .fd(fd),
                )?;
            }
        }
        for (g, c) in cards.iter().enumerate() {
            eprintln!(
                "GPU{g}: target allocations {:.1} MiB",
                c.gpu.device_bytes.get() as f64 / 1048576.
            );
        }
        Ok(Self {
            gguf,
            cards,
            graphs: HashMap::new(),
            max_batch,
            context,
            pos: 0,
            last_batch: 0,
            top_k: 64,
            counter: 0,
            cpu_reduce,
            profile: vec![[0.; 2]; 97],
            gaps: vec![[0.; 2]; 97],
            timing: [0.; 3],
        })
    }
    fn uses_relay(&self, _n: usize) -> bool {
        self.cpu_reduce
    }
    fn uses_stream(&self, n: usize) -> bool {
        !self.cpu_reduce && n > P2P_BATCH
    }
    fn record(&self, g: usize, n: usize, dump: bool, all_logits: bool) -> Result<Graph> {
        let c = &self.cards[g];
        let gpu = &c.gpu;
        let mut graph = Graph {
            cmds: vec![],
            captures: vec![],
        };
        let mut cmd = gpu.begin()?;
        unsafe {
            gpu.dev.cmd_reset_query_pool(cmd, c.queries, 0, 194);
            gpu.dev
                .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, c.queries, 0);
        }
        gpu.barrier(cmd);
        gpu.copy(cmd, &c.params_host, 0, &c.params, 0, 16);
        gpu.copy(cmd, &c.input, 0, &c.x, 0, n * D * 4);
        let push = |ptr: [u64; 6], u: [u32; 16]| Push {
            ptr: [
                ptr[0],
                ptr[1],
                ptr[2],
                ptr[3],
                ptr[4],
                ptr[5],
                c.params.address,
                0,
            ],
            u,
        };
        let capture = |cmd,
                       graph: &mut Graph,
                       name: String,
                       b: &Buffer,
                       len: usize,
                       merge: u8|
         -> Result<()> {
            if dump {
                let host = gpu.buffer(len * 4, true, false, None)?;
                gpu.copy(cmd, b, (n - 1) * len * 4, &host, 0, len * 4);
                graph.captures.push(Capture {
                    name,
                    host,
                    len,
                    merge,
                });
            }
            Ok(())
        };
        capture(cmd, &mut graph, "inp_scaled".into(), &c.x, D, 0)?;
        let matrix = |cmd,
                      w: &Buffer,
                      x: &Buffer,
                      y: &Buffer,
                      k: usize,
                      rows: usize,
                      kind: u32,
                      head: bool,
                      defer: bool| {
            let mut u = [0; 16];
            u[0] = rows as u32;
            u[1] = k as u32;
            u[3] = (rows * k / if kind == 1 { 16 } else { 32 }) as u32;
            u[9] = kind;
            let (input, cols) = if head && !all_logits {
                u[15] = 1;
                (x.address + ((n - 1) * k * 4) as u64, 1)
            } else {
                (x.address, n)
            };
            let shader = if kind == 0 && n > 8 {
                "gemm"
            } else if cols > 1 {
                if cols <= 4 {
                    "multivec4"
                } else {
                    "multivec"
                }
            } else if kind == 0 {
                "matvecq4"
            } else {
                "headvec"
            };
            gpu.dispatch_matrix(
                cmd,
                shader,
                push([0, input, y.address, 0, 0, 0], u).defer_barrier(defer),
                if shader == "gemm" {
                    rows.div_ceil(64)
                } else if matches!(shader, "matvecq4" | "headvec" | "multivec4") {
                    rows.div_ceil(16)
                } else {
                    rows.div_ceil(4)
                } as u32,
                if shader == "gemm" {
                    cols.div_ceil(64)
                } else {
                    cols.div_ceil(8)
                } as u32,
                w,
            )
        };
        let norm = |cmd,
                    x: &Buffer,
                    w: Option<&Buffer>,
                    y: &Buffer,
                    heads: usize,
                    dim: usize,
                    recv: Option<&Buffer>,
                    residual: bool,
                    scale: f32,
                    next: Option<&Buffer>| {
            let mut u = [0; 16];
            u[0] = dim as u32;
            u[3] = (heads * dim) as u32;
            u[11] = u32::from(recv.is_some()) | if residual { 2 } else { 0 };
            u[11] |= if next.is_some() { 4 } else { 0 };
            u[12] = scale.to_bits();
            gpu.dispatch_norm(
                cmd,
                push(
                    [
                        x.address,
                        w.map_or(0, |x| x.address),
                        y.address,
                        0,
                        if residual { c.x.address } else { 0 },
                        next.map_or(0, |w| w.address),
                    ],
                    u,
                )
                .with_extra(if next.is_some() { c.h.address } else { 0 }),
                heads as u32,
                n as u32,
                recv.unwrap_or(&c.dummy_recv),
            )
        };
        let receive = |i: usize| {
            if self.uses_relay(n) {
                &c.relay_recv
            } else if self.uses_stream(n) {
                &c.wide_recv[i % 2]
            } else {
                &c.recv[i]
            }
        };
        for il in 0..L {
            let l = &c.layers[il];
            let w = |name: &str| &l.w[name];
            let detail = il == 0 || il == 5;
            if il > 0 {
                norm(
                    cmd,
                    &c.partial,
                    Some(&c.layers[il - 1].w["post_ffw_norm"]),
                    &c.post,
                    1,
                    D,
                    Some(receive(2 * il - 1)),
                    true,
                    self.layer_scale(il - 1)?,
                    Some(w("attn_norm")),
                )?;
                if il - 1 == 0 || il - 1 == 5 {
                    capture(
                        cmd,
                        &mut graph,
                        format!("ffn_post_norm-{}", il - 1),
                        &c.post,
                        D,
                        0,
                    )?;
                }
                capture(cmd, &mut graph, format!("l_out-{}", il - 1), &c.x, D, 0)?;
            }
            if il == 0 {
                norm(
                    cmd,
                    &c.x,
                    Some(w("attn_norm")),
                    &c.h,
                    1,
                    D,
                    None,
                    false,
                    1.,
                    None,
                )?;
            }
            if detail {
                capture(cmd, &mut graph, format!("attn_norm-{il}"), &c.h, D, 0)?;
            }
            matrix(cmd, w("attn_q"), &c.h, &c.q, D, 8 * l.dim, 0, false, true)?;
            matrix(
                cmd,
                w("attn_k"),
                &c.h,
                &c.k,
                D,
                l.kv * l.dim,
                0,
                false,
                true,
            )?;
            if l.sliding {
                matrix(
                    cmd,
                    w("attn_v"),
                    &c.h,
                    &c.v,
                    D,
                    l.kv * l.dim,
                    0,
                    false,
                    false,
                )?;
            } else {
                gpu.copy(cmd, &c.k, 0, &c.v, 0, n * l.kv * l.dim * 4);
            }
            if detail {
                capture(cmd, &mut graph, format!("Qcur-{il}"), &c.q, 8 * l.dim, 1)?;
                if l.sliding {
                    capture(cmd, &mut graph, format!("Kcur-{il}"), &c.k, l.kv * l.dim, 1)?;
                }
                capture(
                    cmd,
                    &mut graph,
                    format!("Vcur-{il}"),
                    &c.v,
                    l.kv * l.dim,
                    if l.sliding { 1 } else { 0 },
                )?;
            }
            for (src, weight, out, normalized, heads, cache, rotate) in [
                (&c.q, Some(w("attn_q_norm")), &c.rq, &c.nq, 8, None, true),
                (
                    &c.k,
                    Some(w("attn_k_norm")),
                    &c.rk,
                    &c.nk,
                    l.kv,
                    Some(&l.k),
                    true,
                ),
                (&c.v, None, &c.nv, &c.nv, l.kv, Some(&l.v), false),
            ] {
                let mut u = [0; 16];
                u[0] = (heads * l.dim) as u32;
                u[6] = l.dim as u32;
                u[7] = l.cap as u32;
                u[14] = u32::from(rotate); // Q/K are independent; V completes the group.
                u[11] = u32::from(cache.is_some())
                    | if rotate { 2 } else { 0 }
                    | if detail { 4 } else { 0 };
                u[12] = if l.sliding { 10000f32 } else { 1000000f32 }.to_bits();
                gpu.dispatch(
                    cmd,
                    "norm_rope",
                    push(
                        [
                            src.address,
                            weight.map_or(0, |w| w.address),
                            out.address,
                            normalized.address,
                            if l.sliding { 0 } else { c.rope.address },
                            cache.map_or(0, |w| w.address),
                        ],
                        u,
                    ),
                    heads as u32,
                    n as u32,
                );
            }
            if detail {
                for (name, b, rows, merge) in [
                    ("Qcur_normed", &c.nq, 8 * l.dim, 1),
                    (
                        "Kcur_normed",
                        &c.nk,
                        l.kv * l.dim,
                        if l.sliding { 1 } else { 0 },
                    ),
                    (
                        "Vcur_normed",
                        &c.nv,
                        l.kv * l.dim,
                        if l.sliding { 1 } else { 0 },
                    ),
                ] {
                    capture(cmd, &mut graph, format!("{name}-{il}"), b, rows, merge)?;
                }
            }
            if detail {
                capture(
                    cmd,
                    &mut graph,
                    format!("Qcur_pos-{il}"),
                    &c.rq,
                    8 * l.dim,
                    1,
                )?;
                capture(
                    cmd,
                    &mut graph,
                    format!("Kcur_pos-{il}"),
                    &c.rk,
                    l.kv * l.dim,
                    if l.sliding { 1 } else { 0 },
                )?;
            }
            let mut u = [0; 16];
            u[4] = 8;
            u[5] = l.kv as u32;
            u[6] = l.dim as u32;
            u[7] = l.cap as u32;
            u[8] = if l.sliding { 1024 } else { 0 };
            gpu.dispatch(
                cmd,
                if n <= 8 {
                    "attention_partial"
                } else {
                    "attention"
                },
                push(
                    [
                        c.rq.address,
                        l.k.address,
                        l.v.address,
                        c.o.address,
                        c.attn_scratch.address,
                        0,
                    ],
                    u,
                ),
                if n <= 8 { 8 * 32 } else { 8 },
                n as u32,
            );
            if n <= 8 {
                gpu.dispatch(
                    cmd,
                    "attention_merge",
                    push([0, 0, 0, c.o.address, c.attn_scratch.address, 0], u),
                    8,
                    n as u32,
                );
            }
            if detail {
                capture(cmd, &mut graph, format!("kqv_out-{il}"), &c.o, 8 * l.dim, 1)?;
            }
            matrix(
                cmd,
                w("attn_output"),
                &c.o,
                &c.partial,
                8 * l.dim,
                D,
                0,
                false,
                true,
            )?;
            self.end_segment(c, cmd, 2 * il, &mut graph, n)?;
            cmd = self.begin_segment(c, 2 * il + 1, n)?;
            norm(
                cmd,
                &c.partial,
                Some(w("post_attention_norm")),
                &c.post,
                1,
                D,
                Some(receive(2 * il)),
                true,
                1.,
                Some(w("ffn_norm")),
            )?;
            if detail {
                capture(
                    cmd,
                    &mut graph,
                    format!("attn_post_norm-{il}"),
                    &c.post,
                    D,
                    0,
                )?;
            }
            capture(cmd, &mut graph, format!("attn_out-{il}"), &c.x, D, 0)?;
            if detail {
                capture(cmd, &mut graph, format!("ffn_norm-{il}"), &c.h, D, 0)?;
            }
            matrix(cmd, w("ffn_gate"), &c.h, &c.gate, D, F, 0, false, true)?;
            matrix(cmd, w("ffn_up"), &c.h, &c.up, D, F, 0, false, false)?;
            if detail {
                capture(cmd, &mut graph, format!("ffn_gate-{il}"), &c.gate, F, 1)?;
                capture(cmd, &mut graph, format!("ffn_up-{il}"), &c.up, F, 1)?;
            }
            let mut u = [0; 16];
            u[0] = F as u32;
            gpu.dispatch(
                cmd,
                "geglu",
                push([c.gate.address, c.up.address, c.geglu.address, 0, 0, 0], u),
                F.div_ceil(256) as u32,
                n as u32,
            );
            if detail {
                capture(cmd, &mut graph, format!("ffn_geglu-{il}"), &c.geglu, F, 1)?;
            }
            matrix(
                cmd,
                w("ffn_down"),
                &c.geglu,
                &c.partial,
                F,
                D,
                0,
                false,
                true,
            )?;
            if detail {
                capture(cmd, &mut graph, format!("ffn_out-{il}"), &c.partial, D, 2)?;
            }
            self.end_segment(c, cmd, 2 * il + 1, &mut graph, n)?;
            cmd = self.begin_segment(c, 2 * il + 2, n)?;
        }
        norm(
            cmd,
            &c.partial,
            Some(&c.layers[L - 1].w["post_ffw_norm"]),
            &c.post,
            1,
            D,
            Some(receive(2 * L - 1)),
            true,
            self.layer_scale(L - 1)?,
            Some(&c.output_norm),
        )?;
        capture(cmd, &mut graph, "l_out-47".into(), &c.x, D, 0)?;
        capture(cmd, &mut graph, "result_norm".into(), &c.h, D, 0)?;
        matrix(cmd, &c.head, &c.h, &c.logits, D, V, 1, true, false)?;
        if dump {
            let host = gpu.buffer(V * 4, true, false, None)?;
            gpu.copy(
                cmd,
                &c.logits,
                if all_logits { (n - 1) * V * 4 } else { 0 },
                &host,
                0,
                V * 4,
            );
            graph.captures.push(Capture {
                name: "result_output".into(),
                host,
                len: V,
                merge: 3,
            });
        }
        let mut u = [0; 16];
        u[0] = V as u32;
        u[1] = self.top_k as u32;
        u[2] = (128 * self.top_k) as u32;
        u[10] = (g * V) as u32;
        u[11] = 3; // Softcap and model suppression.
        u[12] = 258883;
        u[13] = 258882;
        u[15] = if all_logits { 0 } else { 1 };
        gpu.dispatch(
            cmd,
            "topk",
            push([c.logits.address, 0, c.top_scratch.address, 0, 0, 0], u),
            128,
            if all_logits { n } else { 1 } as u32,
        );
        let mut u = [0; 16];
        u[0] = (128 * self.top_k) as u32;
        u[1] = self.top_k as u32;
        u[15] = if all_logits { 0 } else { 1 };
        gpu.dispatch(
            cmd,
            "merge",
            push([c.top_scratch.address, 0, c.top.address, 0, 0, 0], u),
            1,
            if all_logits { n } else { 1 } as u32,
        );
        unsafe {
            gpu.dev.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                c.queries,
                193,
            );
        }
        gpu.finish(cmd)?;
        graph.cmds.push(cmd);
        Ok(graph)
    }
    fn layer_scale(&self, il: usize) -> Result<f32> {
        let t = self
            .gguf
            .tensor(&format!("blk.{il}.layer_output_scale.weight"))
            .context("missing scale")?;
        Ok(f32::from_le_bytes(self.gguf.data(t)[..4].try_into()?))
    }
    fn begin_segment(&self, c: &Card, i: usize, n: usize) -> Result<vk::CommandBuffer> {
        let cmd = c.gpu.begin()?;
        c.gpu.barrier(cmd);
        unsafe {
            c.gpu.dev.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                c.queries,
                (2 * i) as u32,
            );
        }
        if self.uses_relay(n) {
            c.gpu.copy(cmd, &c.relay_in, 0, &c.relay_recv, 0, n * D * 4);
        }
        Ok(cmd)
    }
    fn end_segment(
        &self,
        c: &Card,
        cmd: vk::CommandBuffer,
        i: usize,
        graph: &mut Graph,
        n: usize,
    ) -> Result<()> {
        if !self.uses_relay(n) {
            let peer = if self.uses_stream(n) {
                &c.wide_peer[i % 2]
            } else {
                &c.peer[i]
            };
            c.gpu.copy(cmd, &c.partial, 0, peer, 0, n * D * 4);
        } else {
            c.gpu.copy(cmd, &c.partial, 0, &c.relay_out, 0, n * D * 4);
        }
        unsafe {
            c.gpu.dev.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                c.queries,
                (2 * i + 1) as u32,
            );
        }
        c.gpu.finish(cmd)?;
        graph.cmds.push(cmd);
        Ok(())
    }
    pub fn forward(
        &mut self,
        tokens: &[u32],
        dump: Option<&Path>,
        all_logits: bool,
    ) -> Result<Vec<Vec<(f32, u32)>>> {
        let n = tokens.len();
        ensure!(
            n > 0 && n <= self.max_batch,
            "batch exceeds configured capacity"
        );
        ensure!(self.pos + n <= self.context, "context exhausted");
        ensure!(
            !all_logits || n <= 8,
            "verification limited to eight columns"
        );
        let mut emb = Vec::with_capacity(n * D);
        let t = self.gguf.tensor("token_embd.weight").unwrap();
        let row = D / 256 * 210;
        for &token in tokens {
            ensure!((token as usize) < V * 2, "invalid token {token}");
            let mut e =
                quant::q6_row(&self.gguf.data(t)[token as usize * row..(token as usize + 1) * row]);
            for x in &mut e {
                *x *= (D as f32).sqrt();
            }
            emb.extend(e);
        }
        for c in &self.cards {
            c.input.write(bytemuck::cast_slice(&emb))?;
            c.params_host
                .write(bytemuck::cast_slice(&[self.pos as u32, n as u32, 0, 0]))?;
        }
        let key = (n, dump.is_some(), all_logits, self.top_k);
        if !self.graphs.contains_key(&key) {
            let a = self.record(0, n, dump.is_some(), all_logits)?;
            let b = self.record(1, n, dump.is_some(), all_logits)?;
            self.graphs.insert(key, vec![a, b]);
        }
        let graphs = &self.graphs[&key];
        for c in &self.cards {
            unsafe {
                c.gpu.dev.reset_fences(&[c.fence])?;
            }
        }
        let started = std::time::Instant::now();
        if !self.uses_relay(n) && !self.uses_stream(n) {
            for (g, c) in self.cards.iter().enumerate() {
                c.gpu.submit_sequence(
                    &graphs[g].cmds,
                    c.peer_sem,
                    c.own_sem,
                    self.counter,
                    c.fence,
                )?;
            }
        }
        // Reused wide slots require interleaved submissions: an implicit fence must never
        // point at a later segment on the other GPU. No CPU waits occur in the stream path.
        for i in 0..if self.uses_relay(n) || self.uses_stream(n) {
            97
        } else {
            0
        } {
            for (g, graph) in graphs.iter().enumerate() {
                let c = &self.cards[g];
                let wait = if i > 0 && !self.uses_relay(n) {
                    Some((c.peer_sem, self.counter + i as u64))
                } else {
                    None
                };
                let signal = if i < 96 && !self.uses_relay(n) {
                    Some((c.own_sem, self.counter + i as u64 + 1))
                } else {
                    None
                };
                c.gpu.submit(
                    graph.cmds[i],
                    wait,
                    signal,
                    if i == 96 || self.uses_relay(n) {
                        c.fence
                    } else {
                        vk::Fence::null()
                    },
                )?;
            }
            if self.uses_relay(n) {
                for c in &self.cards {
                    unsafe {
                        c.gpu
                            .dev
                            .wait_for_fences(&[c.fence], true, 60_000_000_000)?;
                        c.gpu.dev.reset_fences(&[c.fence])?;
                    }
                }
                if i < 96 {
                    for g in 0..2 {
                        self.cards[g]
                            .relay_in
                            .copy_host(&self.cards[1 - g].relay_out, n * D * 4)?;
                    }
                }
            }
        }
        self.timing[0] = started.elapsed().as_secs_f64() * 1e3;
        if !self.uses_relay(n) {
            for c in &self.cards {
                unsafe {
                    c.gpu
                        .dev
                        .wait_for_fences(&[c.fence], true, 60_000_000_000)?;
                }
            }
        }
        self.timing[1] = started.elapsed().as_secs_f64() * 1e3 - self.timing[0];
        self.counter += 96;
        self.pos += n;
        self.last_batch = n;
        for (g, c) in self.cards.iter().enumerate() {
            let mut times = [0u64; 194];
            unsafe {
                c.gpu.dev.get_query_pool_results(
                    c.queries,
                    0,
                    &mut times,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
            }
            for i in 0..97 {
                self.profile[i][g] =
                    (times[2 * i + 1] - times[2 * i]) as f64 * c.gpu.period as f64 / 1e6;
                self.gaps[i][g] = if i > 0 {
                    (times[2 * i] - times[2 * i - 1]) as f64 * c.gpu.period as f64 / 1e6
                } else {
                    0.
                };
            }
        }
        if let Some(path) = dump {
            std::fs::create_dir_all(path)?;
            let mut index = String::new();
            for (a, b) in graphs[0].captures.iter().zip(&graphs[1].captures) {
                ensure!(a.name == b.name, "capture order mismatch");
                let mut va: Vec<f32> = a
                    .host
                    .read(a.len * 4)?
                    .chunks_exact(4)
                    .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
                    .collect();
                let vb: Vec<f32> = b
                    .host
                    .read(b.len * 4)?
                    .chunks_exact(4)
                    .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
                    .collect();
                match a.merge {
                    1 | 3 => va.extend(vb),
                    2 => {
                        for (x, y) in va.iter_mut().zip(vb) {
                            *x += y;
                        }
                    }
                    _ => {}
                }
                if a.merge == 3 {
                    for x in &mut va {
                        *x = 30. * (*x / 30.).tanh();
                    }
                    std::fs::write(path.join("logits.f32"), bytemuck::cast_slice(&va))?;
                }
                std::fs::write(
                    path.join(format!("{}.f32", a.name)),
                    bytemuck::cast_slice(&va),
                )?;
                index.push_str(&format!("{}\t{},1,1,1\t{}\n", a.name, va.len(), va.len()));
            }
            std::fs::write(path.join("index.tsv"), index)?;
        }
        let cols = if all_logits { n } else { 1 };
        let mut result = vec![Vec::new(); cols];
        for c in &self.cards {
            let bytes = c.top.read(cols * self.top_k * 8)?;
            for (t, chunk) in bytes.chunks_exact(self.top_k * 8).enumerate() {
                for s in chunk.chunks_exact(8) {
                    result[t].push((
                        f32::from_le_bytes(s[..4].try_into().unwrap()),
                        u32::from_le_bytes(s[4..].try_into().unwrap()),
                    ));
                }
            }
        }
        for col in &mut result {
            col.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            col.truncate(self.top_k);
        }
        self.timing[2] = started.elapsed().as_secs_f64() * 1e3 - self.timing[0] - self.timing[1];
        Ok(result)
    }
    pub fn set_top_k(&mut self, k: usize) -> Result<()> {
        ensure!((1..=64).contains(&k), "top-k must be in 1..64");
        self.top_k = k;
        Ok(())
    }
    pub fn remaining(&self) -> usize {
        self.context - self.pos
    }
    pub fn rewind(&mut self, pos: usize) -> Result<()> {
        ensure!(pos <= self.pos, "cannot rewind into the future");
        ensure!(
            self.pos - pos <= self.last_batch && self.pos - pos <= self.max_batch,
            "rewind exceeds the last batch retained by the sliding cache"
        );
        self.pos = pos;
        Ok(())
    }
    pub fn hidden_at(&self, index: usize) -> Result<Vec<f32>> {
        ensure!(index < self.last_batch, "hidden row out of range");
        let bytes = self.cards[0].h.download_range(index * D * 4, D * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
            .collect())
    }
    pub fn hidden(&self) -> Result<Vec<f32>> {
        self.hidden_at(self.last_batch - 1)
    }
    pub fn embedding(&self, token: u32) -> Result<Vec<f32>> {
        ensure!((token as usize) < V * 2, "invalid token");
        let t = self.gguf.tensor("token_embd.weight").unwrap();
        let row = D / 256 * 210;
        let mut e =
            quant::q6_row(&self.gguf.data(t)[token as usize * row..(token as usize + 1) * row]);
        for x in &mut e {
            *x *= (D as f32).sqrt();
        }
        Ok(e)
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        for c in &self.cards {
            unsafe {
                let _ = c.gpu.dev.device_wait_idle();
            }
        }
        for (g, c) in self.cards.iter().enumerate() {
            for graphs in self.graphs.values() {
                unsafe {
                    c.gpu.dev.free_command_buffers(c.gpu.pool, &graphs[g].cmds);
                }
            }
            unsafe {
                c.gpu.dev.destroy_query_pool(c.queries, None);
                c.gpu.dev.destroy_fence(c.fence, None);
                c.gpu.dev.destroy_semaphore(c.own_sem, None);
                c.gpu.dev.destroy_semaphore(c.peer_sem, None);
            }
        }
    }
}

fn validate(g: &Gguf) -> Result<()> {
    let suppressed = g
        .get("tokenizer.ggml.suppress_tokens")?
        .as_arr()
        .context("invalid suppressed tokens")?;
    ensure!(
        suppressed.len() == 2
            && suppressed
                .iter()
                .all(|v| matches!(v.as_u64(), Some(258882 | 258883)))
            && suppressed[0].as_u64() != suppressed[1].as_u64(),
        "unsupported suppressed tokens"
    );
    for (key, value) in [
        ("feed_forward_length", 15360),
        ("attention.head_count", 16),
        ("attention.key_length", 512),
        ("attention.value_length", 512),
        ("attention.key_length_swa", 256),
        ("attention.value_length_swa", 256),
        ("attention.sliding_window", 1024),
        ("attention.shared_kv_layers", 0),
        ("embedding_length_per_layer_input", 0),
        ("rope.dimension_count", 512),
        ("rope.dimension_count_swa", 256),
    ] {
        ensure!(
            g.get_u64(&format!("gemma4.{key}"))? == value,
            "unsupported {key}"
        );
    }
    for (key, value) in [
        ("rope.freq_base", 1e6),
        ("rope.freq_base_swa", 1e4),
        ("attention.layer_norm_rms_epsilon", 1e-6),
        ("final_logit_softcapping", 30.),
    ] {
        ensure!(
            (g.get_f64(&format!("gemma4.{key}"))? - value).abs() <= value * 1e-6,
            "unsupported {key}"
        );
    }
    let tensor = |name: &str, dims: &[u64], ty: GgmlType| -> Result<()> {
        let t = g.tensor(name).with_context(|| format!("missing {name}"))?;
        ensure!(
            t.ty == ty && t.dims == dims,
            "unsupported tensor {}: {} {:?}, expected {} {:?}",
            name,
            t.ty,
            t.dims,
            ty,
            dims
        );
        Ok(())
    };
    tensor("token_embd.weight", &[3840, 262144], GgmlType::Q6K)?;
    tensor("output_norm.weight", &[3840], GgmlType::F32)?;
    tensor("rope_freqs.weight", &[256], GgmlType::F32)?;
    let pattern = g
        .get("gemma4.attention.sliding_window_pattern")?
        .as_arr()
        .context("invalid sliding pattern")?;
    let heads = g
        .get("gemma4.attention.head_count_kv")?
        .as_arr()
        .context("invalid KV heads")?;
    ensure!(heads.len() == 48, "invalid KV head count");
    for il in 0..48 {
        let swa = pattern[il].as_u64().context("invalid layer kind")? != 0;
        let d = if swa { 256 } else { 512 };
        let kv = if swa { 8 } else { 1 };
        ensure!(
            heads[il].as_u64() == Some(kv),
            "unsupported KV heads layer {il}"
        );
        for name in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            tensor(&format!("blk.{il}.{name}.weight"), &[3840], GgmlType::F32)?;
        }
        for name in ["attn_q_norm", "attn_k_norm"] {
            tensor(&format!("blk.{il}.{name}.weight"), &[d], GgmlType::F32)?;
        }
        tensor(
            &format!("blk.{il}.layer_output_scale.weight"),
            &[1],
            GgmlType::F32,
        )?;
        for (name, k, n) in [
            ("attn_q", 3840, 16 * d),
            ("attn_k", 3840, kv * d),
            ("attn_output", 16 * d, 3840),
            ("ffn_gate", 3840, 15360),
            ("ffn_up", 3840, 15360),
            ("ffn_down", 15360, 3840),
        ] {
            tensor(&format!("blk.{il}.{name}.weight"), &[k, n], GgmlType::Q4_0)?;
        }
        if swa {
            tensor(
                &format!("blk.{il}.attn_v.weight"),
                &[3840, kv * d],
                GgmlType::Q4_0,
            )?;
        }
    }
    Ok(())
}
