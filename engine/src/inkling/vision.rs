//! The Inkling vision tower (`mmproj` GGUF, projector type `inkling2v`) on one GPU: a ViT of 23 pre-norm blocks of
//! 1024 (16 heads of 64, 2D RoPE, layer scales, MLP 4096 with erf GELU) behind a 20 x 20 patch embedding, a CLS token
//! and a learned 32 x 32 position table resized bilinearly to the patch grid; then the CLS row is dropped, 2 x 2
//! patches are merged (4096) and a two-layer projector (erf GELU) maps them to the language model's 2304. As
//! llama.cpp's `tools/mtmd/models/inkling2v.cpp`; the linear layers run as Q8_0 (quantized from BF16 at load).
use super::gpu::{pack_rows, Matrix};
use crate::{
    gguf::{GgmlType, Gguf},
    vulkan::{Buffer, Device, Push},
};
use anyhow::{ensure, Context, Result};
use half::f16;
use std::{path::Path, rc::Rc};

const PATCH: usize = 20;
const MERGE: usize = 2;
/// Image sides are padded to multiples of this (a merged token's pixels).
const ALIGN: usize = PATCH * MERGE;
/// Most merged tokens per image (llama.cpp's `set_limit_image_tokens(1, 4096)`).
const MAX_TOKENS: usize = 4096;
/// The patch embedding's 1200 inputs, padded to whole Q8_0 blocks.
const PATCH_K: usize = 1216;

struct Layer {
    ln1: Buffer,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    qb: Buffer,
    kb: Buffer,
    vb: Buffer,
    /// Output bias, then the layer scale.
    o: Matrix,
    o_ls: Buffer,
    ln2: Buffer,
    up: Matrix,
    upb: Buffer,
    down: Matrix,
    down_ls: Buffer,
}

pub struct Vision {
    gpu: Rc<Device>,
    d: usize,
    heads: usize,
    eps: f32,
    layers: Vec<Layer>,
    patch: Matrix,
    patch_b: Option<Vec<f32>>,
    /// The position table [32 * 32][d] (row-major grid) and the CLS row (class embedding plus its position).
    pos: Vec<f32>,
    cls: Vec<f32>,
    pre_ln: Buffer,
    mm1: Matrix,
    mm2: Matrix,
    pub out_d: usize,
}

/// Q8_0 blocks (as GGUF stores them) of one row of f32 values, as ggml's `quantize_row_q8_0_ref`.
fn q8_row(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for b in x.chunks_exact(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend(b.iter().map(|v| (v * id).round() as i8 as u8));
    }
    out
}

/// A weight matrix (rows of `k` values; `k_pad` zero-padded) as a Q8_0 matrix on the GPU.
fn matrix(gpu: &Rc<Device>, rows: &[Vec<f32>], k_pad: usize) -> Result<Matrix> {
    let q: Vec<Vec<u8>> = rows
        .iter()
        .map(|r| {
            let mut v = r.clone();
            v.resize(k_pad, 0.);
            q8_row(&v)
        })
        .collect();
    let refs: Vec<&[u8]> = q.iter().map(Vec::as_slice).collect();
    let p = pack_rows(GgmlType::Q8_0, &refs, 0, k_pad / 32)?;
    Ok(Matrix { buf: gpu.upload(&p.bytes)?, stride: p.stride, rows: rows.len(), k: k_pad, ty: GgmlType::Q8_0 })
}

fn upload(gpu: &Rc<Device>, v: &[f32]) -> Result<Buffer> {
    gpu.upload(bytemuck::cast_slice(v))
}

impl Vision {
    pub fn new(gpu: Rc<Device>, path: &Path) -> Result<Self> {
        let g = Gguf::open(path)?;
        ensure!(g.get_str("clip.vision.projector_type")? == "inkling2v", "{} is not an Inkling vision projector", path.display());
        let u = |k: &str| -> Result<usize> { Ok(g.get_u64(&format!("clip.vision.{k}"))? as usize) };
        let (d, heads, ff, blocks) = (u("embedding_length")?, u("attention.head_count")?, u("feed_forward_length")?, u("block_count")?);
        ensure!(u("patch_size")? == PATCH && u("spatial_merge_size")? == MERGE && d / heads == 64, "unsupported vision tower shape");
        let eps = g.get_f64("clip.vision.attention.layer_norm_epsilon")? as f32;
        let t = |n: &str| g.tensor(n).with_context(|| format!("missing {n}"));
        let vec = |n: &str| -> Result<Vec<f32>> { super::vector(&g, t(n)?) };
        let rows = |n: &str| -> Result<Vec<Vec<f32>>> {
            let w = t(n)?;
            let (k, r) = (w.dims[0] as usize, w.dims[1] as usize);
            let mut out = vec![vec![0f32; k]; r];
            for (i, o) in out.iter_mut().enumerate() {
                super::dequant_row(&g, w, i, o)?;
            }
            Ok(out)
        };
        let cat = |a: Vec<f32>, b: Vec<f32>| -> Vec<f32> { a.into_iter().chain(b).collect() };
        let mut layers = Vec::new();
        for l in 0..blocks {
            let n = |s: &str| format!("v.blk.{l}.{s}");
            layers.push(Layer {
                ln1: upload(&gpu, &cat(vec(&n("ln1.weight"))?, vec(&n("ln1.bias"))?))?,
                q: matrix(&gpu, &rows(&n("attn_q.weight"))?, d)?,
                k: matrix(&gpu, &rows(&n("attn_k.weight"))?, d)?,
                v: matrix(&gpu, &rows(&n("attn_v.weight"))?, d)?,
                qb: upload(&gpu, &vec(&n("attn_q.bias"))?)?,
                kb: upload(&gpu, &vec(&n("attn_k.bias"))?)?,
                vb: upload(&gpu, &vec(&n("attn_v.bias"))?)?,
                o: matrix(&gpu, &rows(&n("attn_out.weight"))?, d)?,
                o_ls: upload(&gpu, &cat(vec(&n("attn_out.bias"))?, vec(&n("ls1.weight"))?))?,
                ln2: upload(&gpu, &cat(vec(&n("ln2.weight"))?, vec(&n("ln2.bias"))?))?,
                up: matrix(&gpu, &rows(&n("ffn_up.weight"))?, d)?,
                upb: upload(&gpu, &vec(&n("ffn_up.bias"))?)?,
                down: matrix(&gpu, &rows(&n("ffn_down.weight"))?, ff)?,
                down_ls: upload(&gpu, &cat(vec(&n("ffn_down.bias"))?, vec(&n("ls2.weight"))?))?,
            });
        }
        // The patch embedding [kx][ky][c][out]: each output row reads the patch as (c, ky, kx), then padding.
        let pw = vec("v.patch_embd.weight")?;
        let per = PATCH * PATCH * 3;
        let prow: Vec<Vec<f32>> = (0..d).map(|o| pw[o * per..(o + 1) * per].to_vec()).collect();
        let cls = vec("v.class_embd")?.iter().zip(vec("v.class_position_embd.weight")?).map(|(a, b)| a + b).collect();
        Ok(Self {
            patch: matrix(&gpu, &prow, PATCH_K)?,
            patch_b: g.tensor("v.patch_embd.bias").map(|t| super::vector(&g, t)).transpose()?,
            pos: vec("v.position_embd.weight")?,
            cls,
            pre_ln: upload(&gpu, &cat(vec("v.pre_ln.weight")?, vec("v.pre_ln.bias")?))?,
            mm1: matrix(&gpu, &rows("mm.1.weight")?, 4 * d)?,
            mm2: { let m = rows("mm.2.weight")?; let k = m[0].len(); matrix(&gpu, &m, k)? },
            out_d: g.tensor("mm.2.weight").context("missing mm.2.weight")?.dims[1] as usize,
            gpu,
            d,
            heads,
            eps,
            layers,
        })
    }

    /// Decode, pad and normalize an image as llama.cpp's inkling2v preprocessing: images within the token limit
    /// keep their size, padded right and bottom to multiples of 40 with -1 before normalization ((x - 127.5) / 127.5);
    /// larger ones are first shrunk with their aspect ratio. Returns the patches [rows * cols][1216] (each as (c, y,
    /// x), zero-padded) and the patch grid (cols, rows).
    pub fn preprocess(bytes: &[u8]) -> Result<(Vec<f32>, usize, usize)> {
        let img = image::load_from_memory(bytes).context("decoding the image")?.to_rgb8();
        let (mut w, mut h) = (img.width() as usize, img.height() as usize);
        let mut rgb = img.into_raw();
        let aligned = |v: usize| v.div_ceil(ALIGN) * ALIGN;
        let max_pixels = MAX_TOKENS * ALIGN * ALIGN;
        if aligned(w) * aligned(h) > max_pixels {
            let s = (max_pixels as f64 / (w * h) as f64).sqrt();
            let (nw, nh) = (((w as f64 * s) as usize / ALIGN * ALIGN).max(ALIGN), ((h as f64 * s) as usize / ALIGN * ALIGN).max(ALIGN));
            rgb = resize_bilinear(&rgb, w, h, nw, nh);
            (w, h) = (nw, nh);
        }
        let (pw, ph) = (aligned(w), aligned(h));
        let (cols, rows) = (pw / PATCH, ph / PATCH);
        let px = |c: usize, y: usize, x: usize| -> f32 {
            let v = if x < w && y < h { rgb[(y * w + x) * 3 + c] as f32 } else { -1.0 };
            (v - 127.5) / 127.5
        };
        let mut patches = vec![0f32; rows * cols * PATCH_K];
        for py in 0..rows {
            for pxi in 0..cols {
                let o = &mut patches[(py * cols + pxi) * PATCH_K..];
                for c in 0..3 {
                    for ky in 0..PATCH {
                        for kx in 0..PATCH {
                            o[(c * PATCH + ky) * PATCH + kx] = px(c, py * PATCH + ky, pxi * PATCH + kx);
                        }
                    }
                }
            }
        }
        Ok((patches, cols, rows))
    }

    /// The position table resized to `cols` x `rows` patches (ggml's bilinear interpolation, pixel centres).
    fn positions(&self, cols: usize, rows: usize) -> Vec<f32> {
        let (d, side) = (self.d, 32usize);
        if cols == side && rows == side {
            return self.pos.clone();
        }
        let mut out = vec![0f32; cols * rows * d];
        let (sf0, sf1) = (cols as f32 / side as f32, rows as f32 / side as f32);
        let clampi = |v: i64| v.clamp(0, side as i64 - 1) as usize;
        for i1 in 0..rows {
            let y = (i1 as f32 + 0.5) / sf1 - 0.5;
            let y0 = y.floor() as i64;
            let dy = (y - clampi(y0) as f32).clamp(0., 1.);
            let (ya, yb) = (clampi(y0), clampi(y0 + 1));
            for i0 in 0..cols {
                let x = (i0 as f32 + 0.5) / sf0 - 0.5;
                let x0 = x.floor() as i64;
                let dx = (x - clampi(x0) as f32).clamp(0., 1.);
                let (xa, xb) = (clampi(x0), clampi(x0 + 1));
                let o = &mut out[(i1 * cols + i0) * d..(i1 * cols + i0 + 1) * d];
                for c in 0..d {
                    let at = |yy: usize, xx: usize| self.pos[(yy * side + xx) * d + c];
                    o[c] = at(ya, xa) * (1. - dx) * (1. - dy) + at(ya, xb) * dx * (1. - dy) + at(yb, xa) * (1. - dx) * dy + at(yb, xb) * dx * dy;
                }
            }
        }
        out
    }

    fn push(&self, ptr: [u64; 6], u: [u32; 16]) -> Push {
        Push { ptr: [ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], 0, 0], u }
    }

    /// A Q8_0 matrix times `n` rows (the forward's batch given through push constant r3).
    fn mm(&self, cmd: ash::vk::CommandBuffer, m: &Matrix, x: u64, y: u64, n: usize) -> Result<()> {
        let mut u = [0u32; 16];
        u[0] = m.rows as u32;
        u[1] = m.k as u32;
        u[3] = m.stride as u32;
        u[15] = n as u32;
        let p = self.push([0, x, y, 0, 0, 0], u);
        self.gpu.dispatch_matrix(cmd, "qwen4_q8_gemm", p, m.rows.div_ceil(64) as u32, n.div_ceil(64) as u32, &m.buf)
    }

    fn ops(&self, cmd: ash::vk::CommandBuffer, mode: u32, a: u64, b: u64, c: u64, w: Option<&Buffer>, n: usize, k: usize) -> Result<()> {
        let mut u = [0u32; 16];
        u[0] = n as u32;
        u[1] = k as u32;
        u[11] = mode;
        let p = self.push([a, b, c, 0, 0, 0], u);
        let groups = (n * k).div_ceil(256) as u32;
        match w {
            Some(w) => self.gpu.dispatch_matrix(cmd, "vit_ops", p, groups, 1, w),
            None => {
                self.gpu.dispatch(cmd, "vit_ops", p, groups, 1);
                Ok(())
            }
        }
    }

    fn ln(&self, cmd: ash::vk::CommandBuffer, w: &Buffer, x: u64, y: u64, n: usize) -> Result<()> {
        let mut u = [0u32; 16];
        u[0] = n as u32;
        u[1] = self.d as u32;
        u[12] = self.eps.to_bits();
        let p = self.push([y, x, 0, 0, 0, 0], u);
        self.gpu.dispatch_matrix(cmd, "vit_ln", p, n as u32, 1, w)
    }

    /// The language-model input rows of an image (`bytes`: PNG or JPEG): [tokens][2304], and the token count.
    pub fn encode(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize)> {
        let (patches, cols, rows) = Self::preprocess(bytes)?;
        let (d, h) = (self.d, self.heads);
        let np = cols * rows;
        let n = np + 1;
        let gpu = &self.gpu;
        let fb = |len: usize| gpu.buffer(len * 4, false, false, None);
        let xin = gpu.upload(bytemuck::cast_slice(&patches))?;
        // The CLS row, then the position rows; the patch embeddings are added on the GPU.
        let mut first = self.cls.clone();
        let mut pos = self.positions(cols, rows);
        if let Some(b) = &self.patch_b {
            for row in pos.chunks_exact_mut(d) {
                row.iter_mut().zip(b).for_each(|(p, b)| *p += b);
            }
        }
        first.extend(pos);
        let x = gpu.upload(bytemuck::cast_slice(&first))?;
        let (hb, q, k, v, o, f, pe) = (fb(n * d)?, fb(n * d)?, fb(n * d)?, fb(n * d)?, fb(n * d)?, fb(n * 4 * d)?, fb(np * d)?);
        let off = |b: &Buffer, rows: usize| b.address + (rows * d * 4) as u64;
        let cmd = gpu.begin()?;
        self.mm(cmd, &self.patch, xin.address, pe.address, np)?;
        self.ops(cmd, 4, off(&x, 1), pe.address, 0, None, np, d)?;
        self.ln(cmd, &self.pre_ln, x.address, x.address, n)?;
        gpu.finish(cmd)?;
        gpu.run(cmd)?;
        for l in &self.layers {
            let cmd = gpu.begin()?;
            self.ln(cmd, &l.ln1, x.address, hb.address, n)?;
            self.mm(cmd, &l.q, hb.address, q.address, n)?;
            self.mm(cmd, &l.k, hb.address, k.address, n)?;
            self.mm(cmd, &l.v, hb.address, v.address, n)?;
            self.ops(cmd, 0, q.address, 0, 0, Some(&l.qb), n, d)?;
            self.ops(cmd, 0, k.address, 0, 0, Some(&l.kb), n, d)?;
            self.ops(cmd, 0, v.address, 0, 0, Some(&l.vb), n, d)?;
            let mut u = [0u32; 16];
            u[0] = n as u32;
            u[1] = cols as u32;
            u[4] = h as u32;
            u[12] = 10000f32.to_bits();
            let p = self.push([q.address, 0, k.address, 0, 0, 0], u);
            gpu.dispatch(cmd, "vit_rope", p, (n * h * 32).div_ceil(256) as u32, 1);
            let mut u = [0u32; 16];
            u[0] = n as u32;
            u[4] = h as u32;
            let p = self.push([o.address, q.address, k.address, v.address, 0, 0], u);
            gpu.dispatch(cmd, "vit_attn", p, n.div_ceil(16) as u32, h as u32);
            self.mm(cmd, &l.o, o.address, hb.address, n)?;
            self.ops(cmd, 2, hb.address, 0, x.address, Some(&l.o_ls), n, d)?;
            // x = attention residual (hb holds it now).
            self.ln(cmd, &l.ln2, hb.address, x.address, n)?;
            self.mm(cmd, &l.up, x.address, f.address, n)?;
            self.ops(cmd, 1, f.address, 0, 0, Some(&l.upb), n, 4 * d)?;
            self.mm(cmd, &l.down, f.address, o.address, n)?;
            self.ops(cmd, 2, o.address, 0, hb.address, Some(&l.down_ls), n, d)?;
            gpu.copy(cmd, &o, 0, &x, 0, n * d * 4);
            gpu.barrier(cmd);
            gpu.finish(cmd)?;
            gpu.run(cmd)?;
        }
        // Drop CLS and merge 2 x 2 patches: [p(2X, 2Y), p(2X + 1, 2Y), p(2X, 2Y + 1), p(2X + 1, 2Y + 1)].
        let enc: Vec<f32> = bytemuck::cast_slice(&x.download_range(d * 4, np * d * 4)?).to_vec();
        let (mc, mr) = (cols / MERGE, rows / MERGE);
        let tokens = mc * mr;
        let mut merged = vec![0f32; tokens * 4 * d];
        for yy in 0..mr {
            for xx in 0..mc {
                let o = &mut merged[(yy * mc + xx) * 4 * d..];
                for (s, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                    let p = (2 * yy + dy) * cols + 2 * xx + dx;
                    o[s * d..(s + 1) * d].copy_from_slice(&enc[p * d..(p + 1) * d]);
                }
            }
        }
        let mbuf = gpu.upload(bytemuck::cast_slice(&merged))?;
        let (m1, m2) = (fb(tokens * self.mm1.rows)?, fb(tokens * self.out_d)?);
        let cmd = gpu.begin()?;
        self.mm(cmd, &self.mm1, mbuf.address, m1.address, tokens)?;
        self.ops(cmd, 3, m1.address, 0, 0, None, tokens, self.mm1.rows)?;
        self.mm(cmd, &self.mm2, m1.address, m2.address, tokens)?;
        gpu.finish(cmd)?;
        gpu.run(cmd)?;
        let out: Vec<f32> = bytemuck::cast_slice(&m2.download()?[..tokens * self.out_d * 4]).to_vec();
        Ok((out, tokens))
    }
}

/// Bilinear resize of an RGB8 image (only for images beyond the token limit).
fn resize_bilinear(src: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        let sy = ((y as f32 + 0.5) * h as f32 / nh as f32 - 0.5).max(0.);
        let (y0, fy) = (sy as usize, sy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..nw {
            let sx = ((x as f32 + 0.5) * w as f32 / nw as f32 - 0.5).max(0.);
            let (x0, fx) = (sx as usize, sx.fract());
            let x1 = (x0 + 1).min(w - 1);
            for c in 0..3 {
                let p = |yy: usize, xx: usize| src[(yy * w + xx) * 3 + c] as f32;
                let v = p(y0, x0) * (1. - fx) * (1. - fy) + p(y0, x1) * fx * (1. - fy) + p(y1, x0) * (1. - fx) * fy + p(y1, x1) * fx * fy;
                out[(y * nw + x) * 3 + c] = v.round().clamp(0., 255.) as u8;
            }
        }
    }
    out
}
