//! The Inkling DSpark block drafter (`dflash` GGUF, llama.cpp #32218), replicated on every rank of a group: 3 dense
//! layers over a block of 8 rows (the last sampled token and 7 mask rows) that attend, without a causal mask, to
//! context entries written from the target's hidden states, then the target's head and a recurrent Markov head that
//! refines the logits of each drafted row in turn. Each rank scores its own rows of the vocabulary; the group merges
//! the ranks' first maxima per drafted row. docs/inkling.md, Speculative decoding, has the math.
use super::{
    dequant_row,
    gpu::{pack_rows, Matrix, Split},
    rms_into, vector, Config,
};
use crate::{
    gguf::{GgmlType, Gguf, TensorInfo},
    vulkan::{Buffer, Device, Push},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::rc::Rc;

pub const BLOCK: usize = 8;
/// Draft layers whose attention keeps a sliding window (llama.cpp: the trained 1024, plus the anchor).
const WINDOW: usize = 1025;

/// The host's share of the drafter: the target-feature mixes and the norms that build each committed token's
/// context rows.
pub struct DraftHost {
    /// Softmax of each row of `dspark_fuse`: per draft layer, then the recurrent head's row; 4 target features each.
    fuse: Vec<[f32; 4]>,
    dflare: Vec<Vec<f32>>,
    dspark_norm: Vec<f32>,
    pub mask: Vec<f32>,
    pub layers: usize,
    eps: f32,
}

impl DraftHost {
    pub fn new(g: &Gguf) -> Result<Self> {
        ensure!(g.get_str("general.architecture")? == "dflash", "not a DSpark drafter GGUF");
        let layers = g.get_u64("dflash.block_count")? as usize;
        let targets = g.get("dflash.target_layers")?.as_arr().context("dflash.target_layers")?.len();
        ensure!(targets == 4 && g.get_u64("dflash.block_size")? as usize == BLOCK, "unsupported drafter shape");
        let t = |n: &str| g.tensor(n).with_context(|| format!("missing {n}"));
        let raw = vector(g, t("dspark_fuse.weight")?)?;
        ensure!(raw.len() == 4 * (layers + 1), "dspark_fuse shape");
        let fuse = raw
            .chunks_exact(4)
            .map(|r| {
                let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = r.iter().map(|v| (v - m).exp()).collect();
                let s: f32 = e.iter().sum();
                [e[0] / s, e[1] / s, e[2] / s, e[3] / s]
            })
            .collect();
        Ok(Self {
            fuse,
            dflare: (0..layers).map(|l| vector(g, t(&format!("blk.{l}.dflare_norm.weight"))?)).collect::<Result<_>>()?,
            dspark_norm: vector(g, t("dspark_norm.weight")?)?,
            mask: vector(g, t("mask_embd.weight")?)?,
            layers,
            eps: g.get_f64("dflash.attention.layer_norm_rms_epsilon")? as f32,
        })
    }

    /// The context rows of one committed token from its 4 target features ([4][d]): each draft layer's normed mix
    /// ([layers][d]), and the recurrent head's context.
    pub fn contexts(&self, feats: &[f32], d: usize) -> (Vec<f32>, Vec<f32>) {
        let mix = |row: usize, w: &[f32], out: &mut [f32]| {
            let f = &self.fuse[row];
            let m: Vec<f32> = (0..d).map(|i| f[0] * feats[i] + f[1] * feats[d + i] + f[2] * feats[2 * d + i] + f[3] * feats[3 * d + i]).collect();
            rms_into(&m, Some(w), self.eps, out);
        };
        let mut ctx = vec![0f32; self.layers * d];
        for (l, o) in ctx.chunks_exact_mut(d).enumerate() {
            mix(l, &self.dflare[l], o);
        }
        let mut rnn = vec![0f32; d];
        mix(self.layers, &self.dspark_norm, &mut rnn);
        (ctx, rnn)
    }
}

struct Layer {
    /// attn_norm, then ffn_norm.
    norms: Buffer,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    k_ctx: Matrix,
    v_ctx: Matrix,
    q_norm: Buffer,
    k_norm: Buffer,
    k_ctx_norm: Buffer,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    kc: Buffer,
    vc: Buffer,
    cap: usize,
    window: usize,
}

/// The drafter on one rank's GPU.
pub struct DraftRank {
    gpu: Rc<Device>,
    d: usize,
    heads: usize,
    kv_heads: usize,
    eps: f32,
    rope: f32,
    softcap: f32,
    vocab: usize,
    v0: usize,
    vn: usize,
    layers: Vec<Layer>,
    out_norm: Buffer,
    head: Matrix,
    joint: Buffer,
    joint_b: Buffer,
    hidden: Buffer,
    hidden_b: Buffer,
    w1: Buffer,
    rank: usize,
    // Host-visible: input rows, the rank's first maximum.
    hin: Buffer,
    top: Buffer,
    // Device scratch.
    x: Buffer,
    n: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    att: Buffer,
    f1: Buffer,
    f2: Buffer,
    base: Buffer,
    jin: Buffer,
    jout: Buffer,
    state: Buffer,
    latent: Buffer,
    corr: Buffer,
    bias: Buffer,
    ctx: Buffer,
}

fn q8(gpu: &Rc<Device>, g: &Gguf, t: &TensorInfo) -> Result<Matrix> {
    ensure!(t.ty == GgmlType::Q8_0, "{} must be Q8_0 (use the Q8_0 drafter GGUF)", t.name);
    let k = t.dims[0] as usize;
    let rows_n = t.dims[1] as usize;
    let per = k / 32 * 34;
    let data = g.data(t);
    let rows: Vec<&[u8]> = (0..rows_n).map(|r| &data[r * per..(r + 1) * per]).collect();
    let p = pack_rows(GgmlType::Q8_0, &rows, 0, k / 32)?;
    Ok(Matrix { buf: gpu.upload(&p.bytes)?, stride: p.stride, rows: rows_n, k, ty: GgmlType::Q8_0 })
}

fn f32m(gpu: &Rc<Device>, g: &Gguf, t: &TensorInfo) -> Result<Buffer> {
    let (k, r) = (t.dims[0] as usize, t.dims[1] as usize);
    let mut v = vec![0f32; k * r];
    for (i, o) in v.chunks_exact_mut(k).enumerate() {
        dequant_row(g, t, i, o)?;
    }
    gpu.upload(bytemuck::cast_slice(&v))
}

impl DraftRank {
    pub fn new(gpu: Rc<Device>, g: &Gguf, target: &Gguf, c: &Config, split: Split, context: usize) -> Result<Self> {
        let d = c.d;
        let t = |n: &str| g.tensor(n).with_context(|| format!("missing {n}"));
        let up = |v: &[f32]| gpu.upload(bytemuck::cast_slice(v));
        let layers_n = g.get_u64("dflash.block_count")? as usize;
        let swa = g.get("dflash.attention.sliding_window_pattern")?.as_arr().context("sliding window pattern")?.to_vec();
        let mut layers = Vec::new();
        for l in 0..layers_n {
            let n = |s: &str| format!("blk.{l}.{s}");
            let windowed = swa.get(l).and_then(|v| v.as_u64()).unwrap_or(0) != 0;
            let cap = if windowed { WINDOW + BLOCK } else { context };
            let kvb = cap * 8 * 128 * 2;
            let attn_norm = vector(g, t(&n("attn_norm.weight"))?)?;
            let ffn_norm = vector(g, t(&n("ffn_norm.weight"))?)?;
            let mut norms = attn_norm.clone();
            norms.extend_from_slice(&ffn_norm);
            layers.push(Layer {
                norms: up(&norms)?,
                q: q8(&gpu, g, t(&n("attn_q.weight"))?)?,
                k: q8(&gpu, g, t(&n("attn_k.weight"))?)?,
                v: q8(&gpu, g, t(&n("attn_v.weight"))?)?,
                o: q8(&gpu, g, t(&n("attn_output.weight"))?)?,
                k_ctx: q8(&gpu, g, t(&n("attn_k_ctx.weight"))?)?,
                v_ctx: q8(&gpu, g, t(&n("attn_v_ctx.weight"))?)?,
                q_norm: up(&vector(g, t(&n("attn_q_norm.weight"))?)?)?,
                k_norm: up(&vector(g, t(&n("attn_k_norm.weight"))?)?)?,
                k_ctx_norm: up(&vector(g, t(&n("attn_k_ctx_norm.weight"))?)?)?,
                gate: q8(&gpu, g, t(&n("ffn_gate.weight"))?)?,
                up: q8(&gpu, g, t(&n("ffn_up.weight"))?)?,
                down: q8(&gpu, g, t(&n("ffn_down.weight"))?)?,
                kc: gpu.buffer(kvb, false, false, None)?,
                vc: gpu.buffer(kvb, false, false, None)?,
                cap,
                window: if windowed { WINDOW } else { 0 },
            });
        }
        let (v0, vn) = split.vocab(c);
        let ht = target.tensor("output.weight").context("missing output.weight")?;
        let per = d / 32 * 34;
        let hdata = target.data(ht);
        let rows: Vec<&[u8]> = (v0..v0 + vn).map(|r| &hdata[r * per..(r + 1) * per]).collect();
        let hp = pack_rows(GgmlType::Q8_0, &rows, 0, d / 32)?;
        let w1t = t("markov_w1.weight")?;
        ensure!(w1t.ty == GgmlType::BF16, "markov_w1 must be BF16");
        let rank = w1t.dims[0] as usize;
        let fb = |n: usize| gpu.buffer(n * 4, false, false, None);
        let hb = |n: usize| gpu.buffer(n * 4, true, false, None);
        let ff = layers[0].gate.rows;
        Ok(Self {
            d,
            heads: g.get_u64("dflash.attention.head_count")? as usize,
            kv_heads: g.get_u64("dflash.attention.head_count_kv")? as usize,
            eps: g.get_f64("dflash.attention.layer_norm_rms_epsilon")? as f32,
            rope: g.get_f64("dflash.rope.freq_base")? as f32,
            softcap: g.get_f64("dflash.final_logit_softcapping")? as f32,
            vocab: c.vocab,
            v0,
            vn,
            out_norm: up(&vector(g, t("output_norm.weight")?)?)?,
            head: Matrix { buf: gpu.upload(&hp.bytes)?, stride: hp.stride, rows: vn, k: d, ty: GgmlType::Q8_0 },
            joint: f32m(&gpu, g, t("joint_proj.weight")?)?,
            joint_b: up(&vector(g, t("joint_proj.bias")?)?)?,
            hidden: f32m(&gpu, g, t("dspark_hidden_out.weight")?)?,
            hidden_b: up(&vector(g, t("dspark_hidden_out.bias")?)?)?,
            w1: gpu.upload(g.data(w1t))?,
            rank,
            hin: hb(BLOCK.max(64) * d)?,
            top: hb(2)?,
            x: fb(64 * d)?,
            n: fb(64 * d)?,
            q: fb(64 * 3072)?,
            k: fb(64 * 1024)?,
            v: fb(64 * 1024)?,
            att: fb(BLOCK * 3072)?,
            f1: fb(BLOCK * ff)?,
            f2: fb(BLOCK * ff)?,
            base: fb(BLOCK * vn)?,
            jin: fb(2 * rank + 2 * d)?,
            jout: fb(3 * rank)?,
            state: fb(rank)?,
            latent: fb(rank)?,
            corr: fb(d)?,
            bias: fb(vn)?,
            ctx: fb(d)?,
            layers,
            gpu,
        })
    }

    fn push(&self, ptr: [u64; 6], u: [u32; 16]) -> Push {
        Push { ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], 0, 0], u }
    }

    /// A Q8_0 matrix times `n` rows.
    fn mm(&self, cmd: vk::CommandBuffer, m: &Matrix, x: u64, y: u64, n: usize) -> Result<()> {
        let mut u = [0u32; 16];
        u[0] = m.rows as u32;
        u[1] = m.k as u32;
        u[3] = m.stride as u32;
        u[15] = n as u32;
        let p = self.push([0, x, y, 0, 0, 0], u);
        if n == 1 {
            let (name, rpw) = if m.rows >= 4096 { ("qwen4_q8_mvw4_r4", 16) } else { ("qwen4_q8_mvw4_r2", 8) };
            self.gpu.dispatch_matrix(cmd, name, p, m.rows.div_ceil(rpw) as u32, 1, &m.buf)
        } else {
            self.gpu.dispatch_matrix(cmd, "qwen4_q8_gemm", p, m.rows.div_ceil(64) as u32, n.div_ceil(64) as u32, &m.buf)
        }
    }

    /// An f32 matrix (rows x k) times one row.
    fn mf(&self, cmd: vk::CommandBuffer, w: &Buffer, rows: usize, k: usize, x: u64, y: u64) -> Result<()> {
        let mut u = [0u32; 16];
        u[0] = rows as u32;
        u[1] = k as u32;
        u[15] = 1;
        let p = self.push([0, x, y, 0, 0, 0], u);
        self.gpu.dispatch_matrix(cmd, "qwen4_matf32", p, rows.div_ceil(4) as u32, 1, w)
    }

    fn norm(&self, cmd: vk::CommandBuffer, w: &Buffer, offset: usize, x: u64, y: u64, n: usize) -> Result<()> {
        let mut u = [0u32; 16];
        u[1] = self.d as u32;
        u[10] = offset as u32;
        u[11] = 1;
        u[12] = self.eps.to_bits();
        u[15] = n as u32;
        let p = self.push([y, x, 0, 0, 0, 0], u);
        self.gpu.dispatch_matrix(cmd, "inkling_norm", p, n as u32, 1, w)
    }

    fn ops(&self, cmd: vk::CommandBuffer, mode: u32, a: u64, b: u64, w: Option<&Buffer>, n: usize, k: usize) -> Result<()> {
        let mut u = [0u32; 16];
        u[0] = n as u32;
        u[1] = k as u32;
        u[11] = mode;
        let p = self.push([a, b, 0, 0, 0, 0], u);
        let groups = (n * k).div_ceil(256) as u32;
        match w {
            Some(w) => self.gpu.dispatch_matrix(cmd, "vit_ops", p, groups, 1, w),
            None => {
                self.gpu.dispatch(cmd, "vit_ops", p, groups, 1);
                Ok(())
            }
        }
    }

    /// Per-head norm, NeoX RoPE at positions `pos..pos + n`, and with `cache` the store of K (from `x`) and V
    /// (`vsrc`) into a layer's caches.
    #[allow(clippy::too_many_arguments)]
    fn qk(&self, cmd: vk::CommandBuffer, w: &Buffer, x: u64, heads: usize, pos: usize, n: usize, cache: Option<(&Layer, u64)>) -> Result<()> {
        let mut u = [0u32; 16];
        u[0] = n as u32;
        u[4] = heads as u32;
        u[10] = pos as u32;
        u[12] = self.eps.to_bits();
        u[13] = self.rope.to_bits();
        let (b, c, d) = match cache {
            Some((l, v)) => {
                u[7] = l.cap as u32;
                u[11] = 1;
                (v, l.kc.address, l.vc.address)
            }
            None => (0, 0, 0),
        };
        let p = self.push([x, b, c, d, 0, 0], u);
        self.gpu.dispatch_matrix(cmd, "dspark_qk", p, (n * heads) as u32, 1, w)
    }

    fn run(&self, cmd: vk::CommandBuffer) -> Result<()> {
        self.gpu.finish(cmd)?;
        self.gpu.run(cmd)
    }

    /// Write the context entries of `n` committed tokens at positions `pos..pos + n`: `rows` [n][layers][d], each
    /// layer's normed target-feature mix (`DraftHost::contexts`).
    pub fn inject(&self, rows: &[f32], pos: usize, n: usize) -> Result<()> {
        let d = self.d;
        let nl = self.layers.len();
        ensure!(rows.len() == n * nl * d && n <= 64, "context rows");
        for (l, layer) in self.layers.iter().enumerate() {
            let mine: Vec<f32> = (0..n).flat_map(|t| rows[(t * nl + l) * d..(t * nl + l + 1) * d].iter().copied()).collect();
            self.hin.write(bytemuck::cast_slice(&mine))?;
            let cmd = self.gpu.begin()?;
            self.gpu.host_barrier(cmd, false);
            self.gpu.copy(cmd, &self.hin, 0, &self.x, 0, n * d * 4);
            self.gpu.barrier(cmd);
            self.mm(cmd, &layer.k_ctx, self.x.address, self.k.address, n)?;
            self.mm(cmd, &layer.v_ctx, self.x.address, self.v.address, n)?;
            self.qk(cmd, &layer.k_ctx_norm, self.k.address, self.kv_heads, pos, n, Some((layer, self.v.address)))?;
            self.run(cmd)?;
        }
        Ok(())
    }

    /// Run the block (`rows`: [8][d] input rows, at positions `pos..pos + 8`) through the layers, the output norm and
    /// this rank's head rows (the base logits), and reset the recurrent head with context `rnn` ([d]).
    pub fn block(&self, rows: &[f32], pos: usize, rnn: &[f32]) -> Result<()> {
        let d = self.d;
        let n = BLOCK;
        self.hin.write(bytemuck::cast_slice(rows))?;
        let cmd = self.gpu.begin()?;
        self.gpu.host_barrier(cmd, false);
        self.gpu.copy(cmd, &self.hin, 0, &self.x, 0, n * d * 4);
        self.gpu.barrier(cmd);
        for l in &self.layers {
            self.norm(cmd, &l.norms, 0, self.x.address, self.n.address, n)?;
            self.mm(cmd, &l.q, self.n.address, self.q.address, n)?;
            self.mm(cmd, &l.k, self.n.address, self.k.address, n)?;
            self.mm(cmd, &l.v, self.n.address, self.v.address, n)?;
            self.qk(cmd, &l.q_norm, self.q.address, self.heads, pos, n, None)?;
            self.qk(cmd, &l.k_norm, self.k.address, self.kv_heads, pos, n, None)?;
            let mut u = [0u32; 16];
            u[4] = self.heads as u32;
            u[5] = self.kv_heads as u32;
            u[7] = l.cap as u32;
            u[8] = l.window as u32;
            u[10] = pos as u32;
            let p = self.push([self.att.address, self.q.address, self.k.address, self.v.address, l.kc.address, l.vc.address], u);
            self.gpu.dispatch(cmd, "dspark_attn", p, (n * self.heads) as u32, 1);
            self.mm(cmd, &l.o, self.att.address, self.n.address, n)?;
            self.ops(cmd, 4, self.x.address, self.n.address, None, n, d)?;
            self.norm(cmd, &l.norms, 1, self.x.address, self.n.address, n)?;
            self.mm(cmd, &l.gate, self.n.address, self.f1.address, n)?;
            self.mm(cmd, &l.up, self.n.address, self.f2.address, n)?;
            self.ops(cmd, 5, self.f1.address, self.f2.address, None, n, l.gate.rows)?;
            self.mm(cmd, &l.down, self.f1.address, self.n.address, n)?;
            self.ops(cmd, 4, self.x.address, self.n.address, None, n, d)?;
        }
        // The hidden rows stay in n (normed), the base logits in base.
        self.norm(cmd, &self.out_norm, 0, self.x.address, self.n.address, n)?;
        self.mm(cmd, &self.head, self.n.address, self.base.address, n)?;
        self.run(cmd)?;
        let zero = vec![0f32; self.rank];
        let staging = self.gpu.buffer((zero.len() + d) * 4, true, false, None)?;
        staging.write(bytemuck::cast_slice(&zero))?;
        staging.write_at(zero.len() * 4, bytemuck::cast_slice(rnn))?;
        let cmd = self.gpu.begin()?;
        self.gpu.copy(cmd, &staging, 0, &self.state, 0, self.rank * 4);
        self.gpu.copy(cmd, &staging, self.rank * 4, &self.ctx, 0, d * 4);
        self.run(cmd)
    }

    /// One recurrent step for block row `row` after token `prev`: this rank's first maximum (value, token id).
    pub fn step(&self, row: usize, prev: u32) -> Result<(f32, u32)> {
        let (d, r) = (self.d, self.rank);
        let cmd = self.gpu.begin()?;
        let mut u = [0u32; 16];
        u[0] = r as u32;
        u[1] = d as u32;
        u[2] = row as u32;
        u[10] = prev;
        let p = self.push([self.jin.address, self.w1.address, self.n.address, self.ctx.address, self.state.address, 0], u);
        self.gpu.dispatch(cmd, "dspark_joint_in", p, (2 * r + 2 * d).div_ceil(256) as u32, 1);
        self.mf(cmd, &self.joint, 3 * r, 2 * r + 2 * d, self.jin.address, self.jout.address)?;
        let mut u = [0u32; 16];
        u[0] = r as u32;
        let p = self.push([self.state.address, self.jout.address, self.latent.address, 0, 0, 0], u);
        self.gpu.dispatch_matrix(cmd, "dspark_markov", p, r.div_ceil(256) as u32, 1, &self.joint_b)?;
        self.mf(cmd, &self.hidden, d, r, self.latent.address, self.corr.address)?;
        self.ops(cmd, 0, self.corr.address, 0, Some(&self.hidden_b), 1, d)?;
        self.mm(cmd, &self.head, self.corr.address, self.bias.address, 1)?;
        let mut u = [0u32; 16];
        u[0] = self.vn as u32;
        u[1] = self.vocab as u32;
        u[2] = row as u32;
        u[10] = self.v0 as u32;
        u[12] = self.softcap.to_bits();
        let p = self.push([self.top.address, self.base.address, self.bias.address, 0, 0, 0], u);
        self.gpu.dispatch(cmd, "dspark_argmax", p, 1, 1);
        self.gpu.host_barrier(cmd, true);
        self.run(cmd)?;
        let b = self.top.read(8)?;
        Ok((f32::from_le_bytes(b[..4].try_into().unwrap()), u32::from_le_bytes(b[4..8].try_into().unwrap())))
    }
}
