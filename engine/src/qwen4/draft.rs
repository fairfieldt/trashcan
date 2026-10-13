//! The MTP block on the driver's CPU, drafting tokens for speculative decoding (docs/qwen4.md, MTP). The GPUs have
//! no room for it, so it runs on the driver's CPU (incite: 12-core Xeon E5-2697 v2, AVX without FMA) beside the
//! chain. The math is `mtp::CpuMtp`'s; for speed the matrices are repacked at load into int8 or 4-bit codes with an
//! f32 scale per 32 weights (Q8_0 stays Q8_0, IQ4_XS and IQ4_NL become IQ4_NL codes with per-32 scales), the
//! activations are quantized to Q8_0 per 32 (as ggml's CPU backend does) and multiplied in int8 with SSSE3, and
//! the router and indexer projections stay f32. The draft head is the first `vocab` rows of the trunk's LM head,
//! requantized to Q8_0. Work is split over a persistent pool of threads.
use super::{
    cpu::{rms_norm, sigmoid, silu, CpuStage},
    Config, IQ4_VALUES,
};
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};

/// A job for the pool: `tasks` calls of `f(i)`, claimed through `next`.
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    tasks: usize,
    next: AtomicUsize,
    done: AtomicUsize,
}
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

struct Shared {
    slot: Mutex<(u64, Option<Arc<Job>>)>,
    wake: Condvar,
}

/// Worker threads that run the parallel loops of a draft step (spawning threads for every product would cost more
/// than the products).
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

fn work(job: &Job) {
    loop {
        let i = job.next.fetch_add(1, Ordering::Relaxed);
        if i >= job.tasks {
            return;
        }
        // The caller keeps `f` alive until every task is done.
        unsafe { (*job.f)(i) };
        job.done.fetch_add(1, Ordering::Release);
    }
}

impl Pool {
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            slot: Mutex::new((0, None)),
            wake: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|_| {
                let s = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let job = {
                            let mut g = s.slot.lock().unwrap();
                            while g.0 == seen {
                                g = s.wake.wait(g).unwrap();
                            }
                            seen = g.0;
                            match &g.1 {
                                Some(j) => j.clone(),
                                None => return,
                            }
                        };
                        work(&job);
                    }
                })
            })
            .collect();
        Self { shared, workers }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Run `f(0..tasks)` on every thread, the caller's included; returns when all are done.
    pub fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if tasks == 0 {
            return;
        }
        // Lifetime erased: `run` waits for every task before returning.
        let f: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let job = Arc::new(Job {
            f,
            tasks,
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
        });
        if tasks > 1 && !self.workers.is_empty() {
            let mut g = self.shared.slot.lock().unwrap();
            g.0 += 1;
            g.1 = Some(job.clone());
            self.shared.wake.notify_all();
        }
        work(&job);
        while job.done.load(Ordering::Acquire) < tasks {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut g = self.shared.slot.lock().unwrap();
            g.0 += 1;
            g.1 = None;
            self.shared.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// A pointer for writing disjoint parts of one buffer from several tasks.
#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}
impl Out {
    fn new(v: &mut [f32]) -> Self {
        Self(v.as_mut_ptr())
    }
    /// Safety: tasks write disjoint indices within the buffer.
    unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

/// Weights in rows of `k`, with one f32 scale per 32: int8 values (`Q8`) or IQ4_NL codes, 16 bytes per 32 (`Iq4`);
/// `F32` keeps f32 rows.
pub enum Mat {
    Q8 { k: usize, rows: usize, q: Vec<i8>, d: Vec<f32> },
    Iq4 { k: usize, rows: usize, q: Vec<u8>, d: Vec<f32> },
    F32 { k: usize, rows: usize, w: Vec<f32> },
}

impl Mat {
    fn k(&self) -> usize {
        match self {
            Mat::Q8 { k, .. } | Mat::Iq4 { k, .. } | Mat::F32 { k, .. } => *k,
        }
    }
    pub fn rows(&self) -> usize {
        match self {
            Mat::Q8 { rows, .. } | Mat::Iq4 { rows, .. } | Mat::F32 { rows, .. } => *rows,
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Mat::Q8 { q, d, .. } => q.len() + d.len() * 4,
            Mat::Iq4 { q, d, .. } => q.len() + d.len() * 4,
            Mat::F32 { w, .. } => w.len() * 4,
        }
    }

    /// Rows `row0..row0 + rows` of a GGUF tensor (any rank; rows of `dims[0]`).
    pub fn load(g: &Gguf, t: &TensorInfo, row0: usize, rows: usize) -> Result<Self> {
        let k = t.dims[0] as usize;
        ensure!(k % 32 == 0, "{}: rows of {k} are not blocks of 32", t.name);
        let data = g.data(t);
        let f16 = |b: &[u8]| half::f16::from_le_bytes([b[0], b[1]]).to_f32();
        Ok(match t.ty {
            GgmlType::Q8_0 => {
                let rb = k / 32 * 34;
                let src = &data[row0 * rb..(row0 + rows) * rb];
                let mut q = Vec::with_capacity(rows * k);
                let mut d = Vec::with_capacity(rows * k / 32);
                for b in src.chunks_exact(34) {
                    d.push(f16(b));
                    q.extend(b[2..].iter().map(|v| *v as i8));
                }
                Mat::Q8 { k, rows, q, d }
            }
            GgmlType::IQ4NL => {
                let rb = k / 32 * 18;
                let src = &data[row0 * rb..(row0 + rows) * rb];
                let mut q = Vec::with_capacity(rows * k / 2);
                let mut d = Vec::with_capacity(rows * k / 32);
                for b in src.chunks_exact(18) {
                    d.push(f16(b));
                    q.extend_from_slice(&b[2..]);
                }
                Mat::Iq4 { k, rows, q, d }
            }
            GgmlType::IQ4XS => {
                ensure!(k % 256 == 0, "{}: IQ4_XS rows of {k}", t.name);
                let rb = k / 256 * 136;
                let src = &data[row0 * rb..(row0 + rows) * rb];
                let mut q = Vec::with_capacity(rows * k / 2);
                let mut d = Vec::with_capacity(rows * k / 32);
                for s in src.chunks_exact(136) {
                    let dd = f16(s);
                    let sh = u16::from_le_bytes([s[2], s[3]]) as u32;
                    for ib in 0..8 {
                        let ls = ((s[4 + ib / 2] as u32 >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
                        d.push(dd * (ls as i32 - 32) as f32);
                        q.extend_from_slice(&s[8 + ib * 16..8 + (ib + 1) * 16]);
                    }
                }
                Mat::Iq4 { k, rows, q, d }
            }
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                let mut w = vec![0f32; rows * k];
                for (r, row) in w.chunks_exact_mut(k).enumerate() {
                    crate::qwen::dequant_row(g, t, row0 + r, row)?;
                }
                Mat::F32 { k, rows, w }
            }
            other => bail!("{} is {other}; the CPU draft reads Q8_0, IQ4_NL, IQ4_XS and float matrices", t.name),
        })
    }

    /// Q8_0 codes of f32 rows (the draft head).
    fn quantize(k: usize, w: &[f32]) -> Self {
        let rows = w.len() / k;
        let mut q = vec![0i8; w.len()];
        let mut d = vec![0f32; w.len() / 32];
        quantize_q8(w, &mut q, &mut d);
        Mat::Q8 { k, rows, q, d }
    }

    /// IQ4_NL codes of f32 rows: per 32 the scale minimizing the x^2-weighted squared error over 15 candidates, as
    /// ggml's quantizer does without an importance matrix.
    fn quantize_iq4(pool: &Pool, k: usize, w: &[f32]) -> Self {
        let rows = w.len() / k;
        let blocks = w.len() / 32;
        let mut q = vec![0u8; w.len() / 2];
        let mut d = vec![0f32; blocks];
        let (qo, dout) = (OutU8(q.as_mut_ptr()), Out::new(&mut d));
        pool.run(blocks.div_ceil(4096), &|i| {
            for b in i * 4096..((i + 1) * 4096).min(blocks) {
                let (scale, codes) = iq4_block(&w[b * 32..(b + 1) * 32]);
                unsafe { dout.set(b, scale) };
                for j in 0..16 {
                    unsafe { qo.set(b * 16 + j, codes[j] | (codes[j + 16] << 4)) };
                }
            }
        });
        Mat::Iq4 { k, rows, q, d }
    }

    /// Columns `0..k0` and `k0..k` as two matrices (`k0` a multiple of 32).
    fn split_cols(&self, k0: usize) -> (Mat, Mat) {
        let rows = self.rows();
        let k = self.k();
        let part = |a: usize, b: usize| -> Mat {
            match self {
                Mat::Q8 { q, d, .. } => Mat::Q8 {
                    k: b - a,
                    rows,
                    q: (0..rows).flat_map(|r| q[r * k + a..r * k + b].to_vec()).collect(),
                    d: (0..rows).flat_map(|r| d[(r * k + a) / 32..(r * k + b) / 32].to_vec()).collect(),
                },
                Mat::Iq4 { q, d, .. } => Mat::Iq4 {
                    k: b - a,
                    rows,
                    q: (0..rows).flat_map(|r| q[(r * k + a) / 2..(r * k + b) / 2].to_vec()).collect(),
                    d: (0..rows).flat_map(|r| d[(r * k + a) / 32..(r * k + b) / 32].to_vec()).collect(),
                },
                Mat::F32 { w, .. } => Mat::F32 {
                    k: b - a,
                    rows,
                    w: (0..rows).flat_map(|r| w[r * k + a..r * k + b].to_vec()).collect(),
                },
            }
        };
        (part(0, k0), part(k0, k))
    }

    /// The weights as f32 rows.
    fn to_f32(&self) -> Vec<f32> {
        match self {
            Mat::F32 { w, .. } => w.clone(),
            Mat::Q8 { q, d, .. } => q.iter().enumerate().map(|(i, v)| *v as f32 * d[i / 32]).collect(),
            Mat::Iq4 { q, d, .. } => {
                let mut out = vec![0f32; q.len() * 2];
                for (b, blk) in q.chunks_exact(16).enumerate() {
                    for j in 0..16 {
                        out[b * 32 + j] = IQ4_VALUES[(blk[j] & 15) as usize] as f32 * d[b];
                        out[b * 32 + 16 + j] = IQ4_VALUES[(blk[j] >> 4) as usize] as f32 * d[b];
                    }
                }
                out
            }
        }
    }

    /// Row `r` times one token's activations.
    fn dot(&self, r: usize, x: Tok) -> f32 {
        match self {
            Mat::Q8 { k, q, d, .. } => dot_q8(&q[r * k..(r + 1) * k], &d[r * k / 32..(r + 1) * k / 32], x.q, x.d),
            Mat::Iq4 { k, q, d, .. } => dot_iq4(&q[r * k / 2..(r + 1) * k / 2], &d[r * k / 32..(r + 1) * k / 32], x.q, x.d),
            Mat::F32 { k, w, .. } => {
                debug_assert_eq!(x.f.len(), *k, "an f32 matrix needs the f32 activations");
                dot_f32(&w[r * k..(r + 1) * k], x.f)
            }
        }
    }

    /// `out[t * stride + dst0 + r - r0] = W[r] . x[t]` for the rows `r0..r1` and every token of `x`.
    fn rows_into(&self, x: &Act, r0: usize, r1: usize, out: Out, stride: usize, dst0: usize) {
        for r in r0..r1 {
            for t in 0..x.n {
                unsafe { out.set(t * stride + dst0 + r - r0, self.dot(r, x.tok(t))) };
            }
        }
    }
}

#[derive(Clone, Copy)]
struct OutU8(*mut u8);
unsafe impl Send for OutU8 {}
unsafe impl Sync for OutU8 {}
impl OutU8 {
    /// Safety: tasks write disjoint indices within the buffer.
    unsafe fn set(self, i: usize, v: u8) {
        *self.0.add(i) = v;
    }
}

#[derive(Clone, Copy)]
struct OutI8(*mut i8);
unsafe impl Send for OutI8 {}
unsafe impl Sync for OutI8 {}
impl OutI8 {
    /// Safety: tasks write disjoint indices within the buffer.
    unsafe fn set(self, i: usize, v: i8) {
        *self.0.add(i) = v;
    }
}

/// The positions a token at `p` attends to (`CpuStage::select` with the block scores computed in parallel), from
/// `first` (the first position the block holds keys for).
fn visible(c: &Config, pool: &Pool, iq: &[f32], pooled: &[f32], first: usize, p: usize) -> Vec<usize> {
    let (kp, id) = (c.kpool, c.idx_dim);
    let complete = (p + 1) / kp;
    let mut blocks: Vec<usize> = (0..complete).collect();
    let top = c.top_blocks();
    if complete > top {
        const CHUNK: usize = 64;
        let scale = 1. / (id as f32).sqrt();
        let mut scores = vec![0f32; complete];
        let out = Out::new(&mut scores);
        pool.run(complete.div_ceil(CHUNK), &|i| {
            for b in i * CHUNK..((i + 1) * CHUNK).min(complete) {
                let kb = &pooled[b * id..(b + 1) * id];
                let s = iq.chunks_exact(id).map(|q| dot_f32(q, kb).max(0.)).sum::<f32>() * scale;
                unsafe { out.set(b, s) };
            }
        });
        // The best `top` blocks, ties to the earlier block.
        let order = |a: &usize, b: &usize| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b));
        blocks.select_nth_unstable_by(top - 1, order);
        blocks.truncate(top);
        blocks.sort_unstable();
    }
    let mut out: Vec<usize> = blocks.iter().flat_map(|b| b * kp..(b + 1) * kp).filter(|&j| j >= first).collect();
    out.extend((complete * kp).max(first)..=p);
    out
}

/// Softmax attention of one query over the cache rows `j * stride + off` of the visible positions: the output
/// before normalization and the sum of the weights.
fn attend(q: &[f32], vis: &[usize], kc: &[f32], vc: &[f32], stride: usize, off: usize, scale: f32) -> (Vec<f32>, f32) {
    let hd = q.len();
    #[cfg(target_arch = "x86_64")]
    if hd % 8 == 0 && std::is_x86_feature_detected!("avx") {
        return unsafe { simd::attend(q, vis, kc, vc, stride, off, scale) };
    }
    let scores: Vec<f32> = vis.iter().map(|&j| dot_f32(q, &kc[j * stride + off..][..hd]) * scale).collect();
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = w.iter().sum();
    let mut acc = vec![0f32; hd];
    for (&j, wj) in vis.iter().zip(&w) {
        for (a, x) in acc.iter_mut().zip(&vc[j * stride + off..][..hd]) {
            *a += wj * x;
        }
    }
    (acc, sum)
}

/// Rotate the pairs (i, i + half) of `x` by the angles whose sine and cosine `rope` holds (`half` of them).
fn rotate_with(x: &mut [f32], rope: &[(f32, f32)]) {
    let half = rope.len();
    for (i, &(sn, cs)) in rope.iter().enumerate() {
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cs - b * sn;
        x[i + half] = a * sn + b * cs;
    }
}

/// The IQ4_NL code of `v` (nearest table value).
fn iq4_index(v: f32) -> u8 {
    let mut best = 0;
    for (i, t) in IQ4_VALUES.iter().enumerate() {
        if (v - *t as f32).abs() < (v - IQ4_VALUES[best] as f32).abs() {
            best = i;
        }
    }
    best as u8
}

/// Scale and codes of one block of 32 (ggml's quantize_row_iq4_nl_impl with x^2 weights and 7 tries each side).
fn iq4_block(x: &[f32]) -> (f32, [u8; 32]) {
    let mut codes = [0u8; 32];
    let (mut amax, mut max) = (0f32, 0f32);
    for v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = *v;
        }
    }
    if amax < 1e-30 {
        return (0., [iq4_index(0.); 32]);
    }
    let v0 = IQ4_VALUES[0] as f32;
    let fit = |id: f32, codes: &mut [u8; 32]| -> (f32, f32) {
        let (mut sqx, mut sq2) = (0f32, 0f32);
        for (j, v) in x.iter().enumerate() {
            let l = iq4_index(id * v);
            codes[j] = l;
            let (q, wgt) = (IQ4_VALUES[l as usize] as f32, v * v);
            sqx += wgt * q * v;
            sq2 += wgt * q * q;
        }
        (sqx, sq2)
    };
    let (sqx, sq2) = fit(-v0 / max, &mut codes);
    let mut d = if sq2 > 0. { sqx / sq2 } else { 0. };
    let mut best = d * sqx;
    let mut trial = [0u8; 32];
    for itry in -7..=7 {
        let (sqx, sq2) = fit((itry as f32 + v0) / max, &mut trial);
        if sq2 > 0. && sqx * sqx > best * sq2 {
            d = sqx / sq2;
            best = d * sqx;
            codes = trial;
        }
    }
    (d, codes)
}

/// Activations of `n` tokens: f32 and their Q8_0 codes (per 32: `q = round(x / d)`, `d = amax / 127`).
pub struct Act {
    n: usize,
    k: usize,
    f: Vec<f32>,
    q: Vec<i8>,
    d: Vec<f32>,
}

impl Act {
    fn new(x: &[f32], k: usize) -> Self {
        Self { f: x.to_vec(), ..Self::q8(x, k) }
    }

    /// The Q8_0 codes only, for quantized matrices.
    fn q8(x: &[f32], k: usize) -> Self {
        let n = x.len() / k;
        let mut q = vec![0i8; x.len()];
        let mut d = vec![0f32; x.len() / 32];
        quantize_q8(x, &mut q, &mut d);
        Self { n, k, f: Vec::new(), q, d }
    }

    /// Token `t`.
    fn tok(&self, t: usize) -> Tok<'_> {
        let k = self.k;
        Tok {
            f: if self.f.is_empty() { &[] } else { &self.f[t * k..(t + 1) * k] },
            q: &self.q[t * k..(t + 1) * k],
            d: &self.d[t * k / 32..(t + 1) * k / 32],
        }
    }
}

/// One token's activations (`f` may be empty when only quantized matrices read them).
#[derive(Clone, Copy)]
struct Tok<'a> {
    f: &'a [f32],
    q: &'a [i8],
    d: &'a [f32],
}

fn quantize_q8(x: &[f32], q: &mut [i8], d: &mut [f32]) {
    for ((xb, qb), db) in x.chunks_exact(32).zip(q.chunks_exact_mut(32)).zip(d.iter_mut()) {
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = amax / 127.;
        let inv = if s > 0. { 1. / s } else { 0. };
        *db = s;
        for (o, v) in qb.iter_mut().zip(xb) {
            *o = round_i8(v * inv);
        }
    }
}

/// `x.round() as i8` (half away from zero, saturating) without a call into libm per value (baseline x86_64 has no
/// instruction for it): clamp to the i8 range, truncate, then step away from zero when the exact remainder is at least
/// a half.
#[inline]
fn round_i8(x: f32) -> i8 {
    let x = x.clamp(-128., 127.);
    let t = x as i32;
    let f = x - t as f32;
    (t + (f >= 0.5) as i32 - (f <= -0.5) as i32) as i8
}

fn dot_q8(w: &[i8], wd: &[f32], q: &[i8], d: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("ssse3") {
        return unsafe { simd::dot_q8(w, wd, q, d) };
    }
    let mut sum = 0f32;
    for b in 0..wd.len() {
        let s: i32 = (0..32).map(|j| w[b * 32 + j] as i32 * q[b * 32 + j] as i32).sum();
        sum += s as f32 * wd[b] * d[b];
    }
    sum
}

fn dot_iq4(w: &[u8], wd: &[f32], q: &[i8], d: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("ssse3") {
        return unsafe { simd::dot_iq4(w, wd, q, d) };
    }
    let mut sum = 0f32;
    for b in 0..wd.len() {
        let mut s = 0i32;
        for j in 0..16 {
            let c = w[b * 16 + j];
            s += IQ4_VALUES[(c & 15) as usize] as i32 * q[b * 32 + j] as i32;
            s += IQ4_VALUES[(c >> 4) as usize] as i32 * q[b * 32 + 16 + j] as i32;
        }
        sum += s as f32 * wd[b] * d[b];
    }
    sum
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    super::cpu::dot(a, b)
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn block(w0: __m128i, w1: __m128i, a: *const i8) -> __m128i {
        let ones = _mm_set1_epi16(1);
        let a0 = _mm_loadu_si128(a as *const __m128i);
        let a1 = _mm_loadu_si128(a.add(16) as *const __m128i);
        let p0 = _mm_maddubs_epi16(_mm_abs_epi8(w0), _mm_sign_epi8(a0, w0));
        let p1 = _mm_maddubs_epi16(_mm_abs_epi8(w1), _mm_sign_epi8(a1, w1));
        _mm_add_epi32(_mm_madd_epi16(p0, ones), _mm_madd_epi16(p1, ones))
    }

    #[inline]
    #[target_feature(enable = "sse3")]
    unsafe fn hsum(v: __m128) -> f32 {
        let s = _mm_hadd_ps(v, v);
        _mm_cvtss_f32(_mm_hadd_ps(s, s))
    }

    #[target_feature(enable = "ssse3,sse3")]
    pub unsafe fn dot_q8(w: &[i8], wd: &[f32], q: &[i8], d: &[f32]) -> f32 {
        let mut acc = _mm_setzero_ps();
        for b in 0..wd.len() {
            let wp = w.as_ptr().add(b * 32);
            let w0 = _mm_loadu_si128(wp as *const __m128i);
            let w1 = _mm_loadu_si128(wp.add(16) as *const __m128i);
            let s = block(w0, w1, q.as_ptr().add(b * 32));
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_set1_ps(wd[b] * d[b])));
        }
        hsum(acc)
    }

    /// The sum of a vector's lanes in order, as `cpu::dot` sums its eight accumulators.
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn lanes(v: __m256) -> f32 {
        let mut a = [0f32; 8];
        _mm256_storeu_ps(a.as_mut_ptr(), v);
        a.iter().sum()
    }

    /// `super::attend` with eight dimensions per instruction: each key's score accumulates as `cpu::dot` does (lane j
    /// sums dimensions j, j + 8, ...), four keys at a time so that their sums overlap; the values accumulate in the
    /// order of the positions, four per pass over the output. The results equal the scalar code's.
    #[target_feature(enable = "avx")]
    pub unsafe fn attend(
        q: &[f32],
        vis: &[usize],
        kc: &[f32],
        vc: &[f32],
        stride: usize,
        off: usize,
        scale: f32,
    ) -> (Vec<f32>, f32) {
        let hd = q.len();
        let rows = |c: &[f32], j: usize| -> *const f32 {
            assert!(j * stride + off + hd <= c.len());
            c.as_ptr().add(j * stride + off)
        };
        let qp = q.as_ptr();
        let mut scores = Vec::with_capacity(vis.len());
        let mut groups = vis.chunks_exact(4);
        for g in &mut groups {
            let k = [rows(kc, g[0]), rows(kc, g[1]), rows(kc, g[2]), rows(kc, g[3])];
            let mut a = [_mm256_setzero_ps(); 4];
            for i in (0..hd).step_by(8) {
                let x = _mm256_loadu_ps(qp.add(i));
                for (s, kp) in a.iter_mut().zip(&k) {
                    *s = _mm256_add_ps(*s, _mm256_mul_ps(x, _mm256_loadu_ps(kp.add(i))));
                }
            }
            for s in a {
                scores.push(lanes(s) * scale);
            }
        }
        for &j in groups.remainder() {
            let kp = rows(kc, j);
            let mut s = _mm256_setzero_ps();
            for i in (0..hd).step_by(8) {
                s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_loadu_ps(qp.add(i)), _mm256_loadu_ps(kp.add(i))));
            }
            scores.push(lanes(s) * scale);
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = w.iter().sum();
        let mut acc = vec![0f32; hd];
        let ap = acc.as_mut_ptr();
        let mut groups = vis.chunks_exact(4).zip(w.chunks_exact(4));
        for (g, wg) in &mut groups {
            let v = [rows(vc, g[0]), rows(vc, g[1]), rows(vc, g[2]), rows(vc, g[3])];
            let ws = [_mm256_set1_ps(wg[0]), _mm256_set1_ps(wg[1]), _mm256_set1_ps(wg[2]), _mm256_set1_ps(wg[3])];
            for i in (0..hd).step_by(8) {
                let mut s = _mm256_loadu_ps(ap.add(i));
                for (vp, wv) in v.iter().zip(&ws) {
                    s = _mm256_add_ps(s, _mm256_mul_ps(*wv, _mm256_loadu_ps(vp.add(i))));
                }
                _mm256_storeu_ps(ap.add(i), s);
            }
        }
        let done = vis.len() / 4 * 4;
        for (&j, &wj) in vis[done..].iter().zip(&w[done..]) {
            let (vp, wv) = (rows(vc, j), _mm256_set1_ps(wj));
            for i in (0..hd).step_by(8) {
                let s = _mm256_add_ps(_mm256_loadu_ps(ap.add(i)), _mm256_mul_ps(wv, _mm256_loadu_ps(vp.add(i))));
                _mm256_storeu_ps(ap.add(i), s);
            }
        }
        (acc, sum)
    }

    #[target_feature(enable = "ssse3,sse3")]
    pub unsafe fn dot_iq4(w: &[u8], wd: &[f32], q: &[i8], d: &[f32]) -> f32 {
        let table = _mm_loadu_si128(super::IQ4_VALUES.as_ptr() as *const __m128i);
        let m4 = _mm_set1_epi8(15);
        let mut acc = _mm_setzero_ps();
        for b in 0..wd.len() {
            let c = _mm_loadu_si128(w.as_ptr().add(b * 16) as *const __m128i);
            let w0 = _mm_shuffle_epi8(table, _mm_and_si128(c, m4));
            let w1 = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(c, 4), m4));
            let s = block(w0, w1, q.as_ptr().add(b * 32));
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_set1_ps(wd[b] * d[b])));
        }
        hsum(acc)
    }
}

/// `y = W x` for every token of `x`, rows split over the pool in chunks of `chunk`.
fn matvec(pool: &Pool, w: &Mat, x: &Act, chunk: usize) -> Vec<f32> {
    let rows = w.rows();
    debug_assert_eq!(w.k(), x.k);
    let mut y = vec![0f32; x.n * rows];
    let out = Out::new(&mut y);
    pool.run(rows.div_ceil(chunk), &|i| {
        w.rows_into(x, i * chunk, ((i + 1) * chunk).min(rows), out, rows, i * chunk)
    });
    y
}

/// Products of several matrices with the same input in one parallel loop; returns each output.
fn matvecs(pool: &Pool, ws: &[&Mat], x: &Act, chunk: usize) -> Vec<Vec<f32>> {
    let pairs: Vec<(&Mat, &Act)> = ws.iter().map(|w| (*w, x)).collect();
    matvecs_of(pool, &pairs, chunk)
}

/// Products of several matrices, each with its own input, in one parallel loop; returns each output.
fn matvecs_of(pool: &Pool, ws: &[(&Mat, &Act)], chunk: usize) -> Vec<Vec<f32>> {
    let mut outs: Vec<Vec<f32>> = ws.iter().map(|(w, x)| vec![0f32; x.n * w.rows()]).collect();
    let ptrs: Vec<Out> = outs.iter_mut().map(|o| Out::new(o)).collect();
    let mut tasks = Vec::new();
    for (m, (w, _)) in ws.iter().enumerate() {
        for r0 in (0..w.rows()).step_by(chunk) {
            tasks.push((m, r0, (r0 + chunk).min(w.rows())));
        }
    }
    pool.run(tasks.len(), &|i| {
        let (m, r0, r1) = tasks[i];
        let (w, x) = ws[m];
        w.rows_into(x, r0, r1, ptrs[m], w.rows(), r0)
    });
    outs
}

/// A hyper-connection mixer: per-stream norm gamma, bottleneck down and up, write rows (absent for the final mixer).
struct Mix {
    norm: Vec<f32>,
    down: Mat,
    up: Mat,
    inject: Option<Mat>,
}

struct Experts {
    gate: Mat,
    up: Mat,
    down: Mat,
    sgate: Mat,
    sup: Mat,
    sdown: Mat,
    /// Router rows and the shared-expert gate.
    router: Mat,
    shared_gate: Vec<f32>,
}

/// The MTP block on the CPU with its own QSA caches.
pub struct Draft {
    c: Config,
    pub pool: Pool,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    /// `eh_proj` split by input: the embedding half (once per token) and the hidden half (per stream).
    eh_e: Mat,
    eh_h: Mat,
    attn_mix: Mix,
    ffn_mix: Mix,
    head_mix: Mix,
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    iq: Mat,
    ik: Mat,
    iq_norm: Vec<f32>,
    ik_norm: Vec<f32>,
    moe: Experts,
    /// Token embedding rows (the trunk's), dequantized on use.
    emb: Gguf,
    /// Rows `head_row0..vocab` of the LM head, Q8_0 (the driver's GPUs score the rows before, `skip_head_rows`).
    head: Mat,
    head_row0: usize,
    /// Rows `0..head_row0`, for drafts while the driver's GPUs are busy (`best_all`).
    head_lo: Option<Mat>,
    kc: Vec<f32>,
    vc: Vec<f32>,
    kraw: Vec<f32>,
    pooled: Vec<f32>,
    pub context: usize,
    /// The first position the block holds keys for (prompts fill it from their last chunk only).
    pub first: usize,
    /// Milliseconds per phase over the forwards since the last reset: input projection, first mixer, attention,
    /// second mixer, experts, head.
    pub prof: [f64; 8],
}


impl Draft {
    /// The block of `mtp` (blk.{layers}), the token embedding and the first `vocab` head rows of `trunk`.
    pub fn new(trunk: &std::path::Path, mtp: &std::path::Path, context: usize, vocab: usize, threads: usize) -> Result<Self> {
        let g = Gguf::open(trunk)?;
        let mg = Gguf::open(mtp)?;
        let c = Config::new(&g)?;
        let il = c.layers;
        let t = |name: &str| c.layer_tensor(&mg, il, name);
        let m = |name: &str| -> Result<Mat> {
            let t = t(name)?;
            Mat::load(&mg, t, 0, (t.dims[1..].iter().product::<u64>()) as usize)
        };
        let v = |name: &str| -> Result<Vec<f32>> { crate::qwen::vector(&mg, t(name)?) };
        let mix = |prefix: &str, inject: bool| -> Result<Mix> {
            let tt = |name: &str| {
                let full = format!("{prefix}_{name}.weight");
                mg.tensor(&full).with_context(|| format!("missing {full}"))
            };
            let all = |t: &TensorInfo| Mat::load(&mg, t, 0, t.dims[1] as usize);
            Ok(Mix {
                norm: crate::qwen::vector(&mg, tt("norm")?)?,
                down: all(tt("down")?)?,
                up: all(tt("up")?)?,
                inject: if inject { Some(all(tt("inject")?)?) } else { None },
            })
        };
        let pool = Pool::new(threads);
        let ht = g.tensor("output.weight").context("missing output.weight")?;
        let vocab = vocab.min(c.vocab);
        // The head rows, dequantized and requantized to Q8_0 in parallel.
        let d = c.d;
        let mut hw = vec![0f32; vocab * d];
        {
            let out = Out::new(&mut hw);
            let g = &g;
            pool.run(vocab.div_ceil(1024), &|i| {
                let mut row = vec![0f32; d];
                for r in i * 1024..((i + 1) * 1024).min(vocab) {
                    super::dequant_row(g, ht, r, &mut row).unwrap();
                    for (j, x) in row.iter().enumerate() {
                        unsafe { out.set(r * d + j, *x) };
                    }
                }
            });
        }
        let head = Mat::quantize(d, &hw);
        drop(hw);
        let shared_gate = v("ffn_gate_inp_shexp.weight")?;
        let kv = context * c.kv_heads * c.head_dim;
        let mut draft = Self {
            enorm: v("nextn.enorm.weight")?,
            hnorm: v("nextn.hnorm.weight")?,
            eh_e: Mat::F32 { k: 0, rows: 0, w: Vec::new() },
            eh_h: m("nextn.eh_proj.weight")?,
            attn_mix: mix(&format!("blk.{il}.hc_attn"), true)?,
            ffn_mix: mix(&format!("blk.{il}.hc_ffn"), true)?,
            head_mix: mix(&format!("blk.{il}.nextn.hc_head"), false)?,
            q: m("attn_q.weight")?,
            k: m("attn_k.weight")?,
            v: m("attn_v.weight")?,
            o: m("attn_output.weight")?,
            q_norm: v("attn_q_norm.weight")?,
            k_norm: v("attn_k_norm.weight")?,
            iq: m("indexer.q_proj.weight")?,
            ik: m("indexer.k_proj.weight")?,
            iq_norm: v("indexer.q_norm.weight")?,
            ik_norm: v("indexer.k_norm.weight")?,
            moe: Experts {
                gate: m("ffn_gate_exps.weight")?,
                up: m("ffn_up_exps.weight")?,
                down: m("ffn_down_exps.weight")?,
                sgate: m("ffn_gate_shexp.weight")?,
                sup: m("ffn_up_shexp.weight")?,
                sdown: m("ffn_down_shexp.weight")?,
                router: m("ffn_gate_inp.weight")?,
                shared_gate,
            },
            emb: g,
            head,
            head_row0: 0,
            head_lo: None,
            kc: vec![0.; kv],
            vc: vec![0.; kv],
            kraw: vec![0.; context * c.idx_dim],
            pooled: vec![0.; context.div_ceil(c.kpool) * c.idx_dim],
            context,
            first: 0,
            prof: [0.; 8],
            pool,
            c,
        };
        ensure!(
            draft.moe.gate.rows() == draft.c.experts * draft.c.ff_exp && draft.moe.down.k() == draft.c.ff_exp,
            "unexpected expert shapes"
        );
        // Dense matrices as 4-bit codes (IQ4_NL values, f32 scale per 32): no measurable loss of acceptance, a
        // shorter step (results/qwen4/mtp-20261012). The head stays Q8_0: its 4-bit kernel is compute bound here.
        let p = &draft.pool;
        let r = |m: &mut Mat| *m = Mat::quantize_iq4(p, m.k(), &m.to_f32());
        for m in [&mut draft.eh_h, &mut draft.q, &mut draft.k, &mut draft.v, &mut draft.o] {
            r(m);
        }
        for x in [&mut draft.attn_mix, &mut draft.ffn_mix, &mut draft.head_mix] {
            r(&mut x.down);
            r(&mut x.up);
            if let Some(i) = x.inject.as_mut() {
                r(i);
            }
        }
        let (e, h) = draft.eh_h.split_cols(draft.c.d);
        draft.eh_e = e;
        draft.eh_h = h;
        Ok(draft)
    }

    /// Bytes of the weights a draft step reads (dense matrices, 11 experts and the head) and of all weights held.
    pub fn bytes(&self) -> (usize, usize) {
        let dense = [&self.eh_e, &self.eh_h, &self.q, &self.k, &self.v, &self.o, &self.iq, &self.ik, &self.moe.router]
            .iter()
            .map(|m| m.bytes())
            .sum::<usize>()
            + [&self.attn_mix, &self.ffn_mix, &self.head_mix]
                .iter()
                .map(|x| x.down.bytes() + x.up.bytes() + x.inject.as_ref().map_or(0, |i| i.bytes()))
                .sum::<usize>();
        let per_expert = (self.moe.gate.bytes() + self.moe.up.bytes() + self.moe.down.bytes()) / self.c.experts;
        let step = dense + (self.c.used + 1) * per_expert + self.head.bytes();
        let all = dense + self.moe.gate.bytes() + self.moe.up.bytes() + self.moe.down.bytes() + self.head.bytes();
        (step, all)
    }

    pub fn vocab(&self) -> usize {
        self.head_row0 + self.head.rows()
    }

    /// Leave the first `rows` head rows to another scorer (the driver's GPU head slices): `head_best` ranks only
    /// the rest, `best_all` all of them.
    pub fn skip_head_rows(&mut self, rows: usize) {
        let rows = rows.min(self.vocab());
        if self.head_row0 > 0 || rows == 0 {
            return;
        }
        let lo = match &mut self.head {
            Mat::Q8 { k, rows: n, q, d } => {
                *n -= rows;
                Mat::Q8 { k: *k, rows, q: q.drain(..rows * *k).collect(), d: d.drain(..rows * *k / 32).collect() }
            }
            Mat::Iq4 { k, rows: n, q, d } => {
                *n -= rows;
                Mat::Iq4 { k: *k, rows, q: q.drain(..rows * *k / 2).collect(), d: d.drain(..rows * *k / 32).collect() }
            }
            Mat::F32 { k, rows: n, w } => {
                *n -= rows;
                Mat::F32 { k: *k, rows, w: w.drain(..rows * *k).collect() }
            }
        };
        self.head_lo = Some(lo);
        self.head_row0 = rows;
    }

    /// The block's final mixer on the last row of its output residual: the head's input.
    pub fn final_hidden(&self, x: &[f32]) -> Vec<f32> {
        let w = self.c.width();
        self.mix(&self.head_mix, &x[x.len() - w..], 1).0
    }

    /// The best of the CPU head rows for a final hidden state (token id and logit), if it holds any.
    pub fn head_best(&self, hn: &[f32]) -> Option<(u32, f32)> {
        if self.head.rows() == 0 {
            return None;
        }
        let ha = Act::q8(hn, self.c.d);
        let logits = matvec(&self.pool, &self.head, &ha, 256);
        let b = (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
        Some((b as u32 + self.head_row0 as u32, logits[b]))
    }

    /// The best of all draft head rows (token id and logit), on the CPU.
    pub fn best_all(&self, hn: &[f32]) -> Option<(u32, f32)> {
        let lo = self.head_lo.as_ref().map(|m| {
            let logits = matvec(&self.pool, m, &Act::q8(hn, self.c.d), 256);
            let b = (0..logits.len()).fold(0, |b, i| if logits[i] > logits[b] { i } else { b });
            (b as u32, logits[b])
        });
        match (lo, self.head_best(hn)) {
            (Some(l), Some(h)) => Some(if l.1 >= h.1 { l } else { h }),
            (l, h) => l.or(h),
        }
    }

    /// The top candidates (logit, token id; best first) of the CPU head rows, as a head slice returns them.
    pub fn head_top(&self, hn: &[f32]) -> Vec<(f32, u32)> {
        if self.head.rows() == 0 {
            return Vec::new();
        }
        let logits = matvec(&self.pool, &self.head, &Act::q8(hn, self.c.d), 256);
        let row0 = self.head_row0 as u32;
        crate::qwen::pipeline::top_k(&logits).into_iter().map(|(s, id)| (s, id + row0)).collect()
    }

    /// The top candidates of all draft head rows, on the CPU.
    pub fn top_all(&self, hn: &[f32]) -> Vec<(f32, u32)> {
        let lo = self.head_lo.as_ref().map_or_else(Vec::new, |m| {
            crate::qwen::pipeline::top_k(&matvec(&self.pool, m, &Act::q8(hn, self.c.d), 256))
        });
        super::gpu::merge_top(lo, self.head_top(hn))
    }

    fn mix(&self, m: &Mix, x: &[f32], n: usize) -> (Vec<f32>, Option<Vec<f32>>) {
        let c = &self.c;
        let (d, hc, w) = (c.d, c.hc, c.width());
        let mut xn = x.to_vec();
        for (i, row) in xn.chunks_exact_mut(d).enumerate() {
            rms_norm(row, Some(&m.norm[(i % hc) * d..(i % hc + 1) * d]), c.eps);
        }
        let xa = Act::q8(&xn, w);
        let mut first: Vec<&Mat> = vec![&m.down];
        if let Some(inj) = &m.inject {
            first.push(inj);
        }
        let outs = matvecs(&self.pool, &first, &xa, 8);
        let lo: Vec<f32> = outs[0].iter().map(|v| silu(v / hc as f32)).collect();
        let la = Act::q8(&lo, c.hc_rank);
        // The up rows of every stream for a range of the sublayer input, their read gates and the stream mean in one
        // parallel loop.
        const CHUNK: usize = 64;
        let per = d.div_ceil(CHUNK);
        let mut u = vec![0f32; n * d];
        let out = Out::new(&mut u);
        self.pool.run(n * per, &|i| {
            let (t, i0) = (i / per, i % per * CHUNK);
            let x = la.tok(t);
            for j in i0..(i0 + CHUNK).min(d) {
                let mut acc = 0f32;
                for s in 0..hc {
                    let r = s * d + j;
                    acc += xn[t * w + r] * sigmoid(m.up.dot(r, x));
                }
                unsafe { out.set(t * d + j, acc / hc as f32) };
            }
        });
        let write = outs.get(1).map(|inj| inj.iter().map(|v| 2. * sigmoid(v / hc as f32)).collect());
        (u, write)
    }

    /// Sine and cosine of each rotated pair's angle at `pos`.
    fn rope(&self, pos: usize) -> Vec<(f32, f32)> {
        (0..self.c.rot / 2).map(|i| self.c.rope_angle(pos, i).sin_cos()).collect()
    }

    fn rotate(&self, x: &mut [f32], pos: usize) {
        rotate_with(x, &self.rope(pos));
    }

    fn attention(&mut self, u: &[f32], pos: usize, n: usize) -> Vec<f32> {
        let c = self.c.clone();
        let (d, hd, nh, nkv, id, ih, kp) = (c.d, c.head_dim, c.heads, c.kv_heads, c.idx_dim, c.idx_heads, c.kpool);
        let ua = Act::new(u, d);
        // The indexer queries only rank blocks once more are complete than QSA keeps.
        let sparse = (pos + n) / kp > c.top_blocks();
        let mats: Vec<&Mat> = if sparse {
            vec![&self.q, &self.k, &self.v, &self.ik, &self.iq]
        } else {
            vec![&self.q, &self.k, &self.v, &self.ik]
        };
        let t0 = std::time::Instant::now();
        let outs = matvecs(&self.pool, &mats, &ua, 64);
        self.prof[6] += t0.elapsed().as_secs_f64() * 1e3;
        let (qg, k, v, kr) = (&outs[0], &outs[1], &outs[2], &outs[3]);
        let mut iq = if sparse { outs[4].clone() } else { vec![0f32; n * ih * id] };
        self.kraw[pos * id..(pos + n) * id].copy_from_slice(kr);
        for b in (pos + 1) / kp..(pos + n + 1) / kp {
            let mut m = vec![0f32; id];
            for p in b * kp..(b + 1) * kp {
                for (a, x) in m.iter_mut().zip(&self.kraw[p * id..(p + 1) * id]) {
                    *a += x;
                }
            }
            for a in &mut m {
                *a /= kp as f32;
            }
            rms_norm(&mut m, Some(&self.ik_norm), c.eps);
            self.rotate(&mut m, b * kp);
            self.pooled[b * id..(b + 1) * id].copy_from_slice(&m);
        }
        let ropes: Vec<Vec<(f32, f32)>> = (0..n).map(|t| self.rope(pos + t)).collect();
        for t in (0..n).filter(|_| sparse) {
            for h in 0..ih {
                let r = &mut iq[(t * ih + h) * id..(t * ih + h + 1) * id];
                rms_norm(r, Some(&self.iq_norm), c.eps);
                rotate_with(r, &ropes[t]);
            }
        }
        for t in 0..n {
            for h in 0..nkv {
                let at = (t * nkv + h) * hd;
                let mut kk = k[at..at + hd].to_vec();
                rms_norm(&mut kk, Some(&self.k_norm), c.eps);
                rotate_with(&mut kk, &ropes[t]);
                let slot = ((pos + t) * nkv + h) * hd;
                self.kc[slot..slot + hd].copy_from_slice(&kk);
                self.vc[slot..slot + hd].copy_from_slice(&v[at..at + hd]);
            }
        }
        let visible: Vec<Vec<usize>> = (0..n)
            .map(|t| visible(&c, &self.pool, &iq[t * ih * id..(t + 1) * ih * id], &self.pooled, self.first, pos + t))
            .collect();
        let scale = 1. / (hd as f32).sqrt();
        let mut o = vec![0f32; n * nh * hd];
        let out = Out::new(&mut o);
        let (kc, vc, q_norm, eps) = (&self.kc, &self.vc, &self.q_norm, c.eps);
        // Per head: the query's norm and rotation, attention, and the output gate.
        self.pool.run(n * nh, &|i| {
            let (t, h) = (i / nh, i % nh);
            let kh = h / (nh / nkv);
            let src = &qg[(t * nh + h) * 2 * hd..][..2 * hd];
            let mut qv = src[..hd].to_vec();
            rms_norm(&mut qv, Some(q_norm), eps);
            rotate_with(&mut qv, &ropes[t]);
            let (acc, sum) = attend(&qv, &visible[t], kc, vc, nkv * hd, kh * hd, scale);
            for (e, (a, g)) in acc.iter().zip(&src[hd..]).enumerate() {
                unsafe { out.set((t * nh + h) * hd + e, a / sum * sigmoid(*g)) };
            }
        });
        self.prof[7] += t0.elapsed().as_secs_f64() * 1e3;
        matvec(&self.pool, &self.o, &Act::q8(&o, nh * hd), 32)
    }

    fn moe(&self, u: &[f32], n: usize) -> Vec<f32> {
        let c = &self.c;
        let (d, ne, k, ff) = (c.d, c.experts, c.used, c.ff_exp);
        let e = &self.moe;
        let ua = Act::new(u, d);
        let logits = matvec(&self.pool, &e.router, &ua, 16);
        // (expert, weight) per token in rank order; the shared expert last with its sigmoid gate.
        let mut chosen = Vec::with_capacity(n * (k + 1));
        for t in 0..n {
            let row = &logits[t * ne..(t + 1) * ne];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = p.iter().sum();
            let order = |a: &usize, b: &usize| p[*b].total_cmp(&p[*a]).then(a.cmp(b));
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.select_nth_unstable_by(k - 1, order);
            idx.truncate(k);
            idx.sort_by(order);
            let top = idx.iter().map(|&i| p[i] / sum).sum::<f32>().max(6.103515625e-5);
            chosen.extend(idx.iter().map(|&i| (i, p[i] / sum / top)));
            chosen.push((ne, sigmoid(dot_f32(&e.shared_gate, &u[t * d..(t + 1) * d]))));
        }
        let slots = k + 1;
        // Gate and up rows of every (token, slot) in parts of two Q8_0 blocks, each part's SwiGLU activations
        // quantized for the down rows in the same task; then the down rows.
        const PART: usize = 64;
        let parts = ff / PART;
        let mut aq = vec![0i8; n * slots * ff];
        let mut ad = vec![0f32; n * slots * ff / 32];
        let (qo, dout) = (OutI8(aq.as_mut_ptr()), Out::new(&mut ad));
        self.pool.run(n * slots * parts, &|i| {
            let (ts, part) = (i / parts, i % parts);
            let (t, (ex, _)) = (ts / slots, chosen[ts]);
            let x = ua.tok(t);
            let (g, up, r0) = if ex == ne { (&e.sgate, &e.sup, 0) } else { (&e.gate, &e.up, ex * ff) };
            let mut a = [0f32; PART];
            for (j, v) in a.iter_mut().enumerate() {
                let r = r0 + part * PART + j;
                *v = silu(g.dot(r, x)) * up.dot(r, x);
            }
            let (mut q, mut s) = ([0i8; PART], [0f32; PART / 32]);
            quantize_q8(&a, &mut q, &mut s);
            let at = ts * ff + part * PART;
            for (j, v) in q.iter().enumerate() {
                unsafe { qo.set(at + j, *v) };
            }
            for (j, v) in s.iter().enumerate() {
                unsafe { dout.set(at / 32 + j, *v) };
            }
        });
        let mut down = vec![0f32; n * slots * d];
        let out = Out::new(&mut down);
        let dparts = 8;
        let dper = d / dparts;
        self.pool.run(n * slots * dparts, &|i| {
            let (ts, part) = (i / dparts, i % dparts);
            let ex = chosen[ts].0;
            let x = Tok { f: &[], q: &aq[ts * ff..(ts + 1) * ff], d: &ad[ts * ff / 32..(ts + 1) * ff / 32] };
            let (w, r0) = if ex == ne { (&e.sdown, 0) } else { (&e.down, ex * d) };
            for r in r0 + part * dper..r0 + (part + 1) * dper {
                unsafe { out.set(ts * d + r - r0, w.dot(r, x)) };
            }
        });
        let mut y = vec![0f32; n * d];
        for t in 0..n {
            for s in 0..slots {
                let (_, wgt) = chosen[t * slots + s];
                let r = &down[(t * slots + s) * d..(t * slots + s + 1) * d];
                for (o, v) in y[t * d..(t + 1) * d].iter_mut().zip(r) {
                    *o += wgt * v;
                }
            }
        }
        y
    }

    /// Run the block over `tokens` at `pos..` (token i paired with row i of `h`, see `CpuMtp::forward`). Returns the
    /// output residual (`n x hc x d`) and, with `head`, the best draft token of the last row and its logit.
    pub fn forward(&mut self, tokens: &[u32], h: &[f32], pos: usize, head: bool) -> Result<(Vec<f32>, Option<(u32, f32)>)> {
        let c = self.c.clone();
        let (d, hc, w, n) = (c.d, c.hc, c.width(), tokens.len());
        ensure!(n > 0 && h.len() == n * w, "draft input length mismatch");
        ensure!(pos + n <= self.context, "draft context exhausted");
        let et = self.emb.tensor("token_embd.weight").context("missing token_embd")?;
        // x_s = W_e RMS(e) enorm + W_h RMS(h_s) hnorm_s: the embedding half once per token.
        let mut en = vec![0f32; n * d];
        let mut hn = h.to_vec();
        for (i, &id) in tokens.iter().enumerate() {
            ensure!((id as u64) < et.dims[1], "invalid token {id}");
            let e = &mut en[i * d..(i + 1) * d];
            crate::qwen::dequant_row(&self.emb, et, id as usize, e)?;
            rms_norm(e, Some(&self.enorm), c.eps);
            for s in 0..hc {
                let row = &mut hn[(i * hc + s) * d..(i * hc + s + 1) * d];
                rms_norm(row, Some(&self.hnorm[s * d..(s + 1) * d]), c.eps);
            }
        }
        let mut t = std::time::Instant::now();
        let mut lap = |prof: &mut [f64; 8], i: usize| {
            prof[i] += t.elapsed().as_secs_f64() * 1e3;
            t = std::time::Instant::now();
        };
        let (ea, ha) = (Act::q8(&en, d), Act::q8(&hn, d));
        let outs = matvecs_of(&self.pool, &[(&self.eh_e, &ea), (&self.eh_h, &ha)], 32);
        let mut x = outs[1].clone();
        for (r, row) in x.chunks_exact_mut(d).enumerate() {
            for (a, b) in row.iter_mut().zip(&outs[0][r / hc * d..(r / hc + 1) * d]) {
                *a += b;
            }
        }
        lap(&mut self.prof, 0);
        let (u, write) = self.mix(&self.attn_mix, &x, n);
        lap(&mut self.prof, 1);
        let y = self.attention(&u, pos, n);
        CpuStage::combine(&c, &mut x, &y, write.as_deref().unwrap(), n);
        lap(&mut self.prof, 2);
        let (u, write) = self.mix(&self.ffn_mix, &x, n);
        lap(&mut self.prof, 3);
        let y = self.moe(&u, n);
        CpuStage::combine(&c, &mut x, &y, write.as_deref().unwrap(), n);
        lap(&mut self.prof, 4);
        let best = if head {
            let (hn, _) = self.mix(&self.head_mix, &x[(n - 1) * w..], 1);
            self.head_best(&hn)
        } else {
            None
        };
        lap(&mut self.prof, 5);
        Ok((x, best))
    }

    /// Draft logits of the last row of an output residual (validation).
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        let w = self.c.width();
        let (hn, _) = self.mix(&self.head_mix, &x[x.len() - w..], 1);
        matvec(&self.pool, &self.head, &Act::q8(&hn, self.c.d), 256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_positions_match_the_reference_selection() {
        // The model's indexer shape (4 heads of 128, blocks of 4, 512 blocks kept), random pooled keys and queries,
        // with ties between blocks (repeated keys).
        let c = Config {
            d: 8, layers: 1, heads: 1, kv_heads: 1, head_dim: 8, rot: 4, rope_base: 1e4, eps: 1e-6, interval: 4,
            conv: 4, k_heads: 1, v_heads: 1, state: 8, vocab: 16, hc: 4, hc_rank: 2, experts: 4, used: 2, ff_exp: 4,
            ff_shexp: 4, idx_heads: 4, idx_dim: 128, idx_budget: 2048, kpool: 4, ngram: None,
        };
        let pool = Pool::new(4);
        let mut s = 3u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let blocks = 2000;
        let mut pooled: Vec<f32> = (0..blocks * 128).map(|_| rnd()).collect();
        for b in (0..blocks).step_by(7) {
            let src: Vec<f32> = pooled[..128].to_vec();
            pooled[b * 128..(b + 1) * 128].copy_from_slice(&src);
        }
        for p in [100, 2050, 2051, 2052, 3000, 5000, 7999] {
            let iq: Vec<f32> = (0..4 * 128).map(|_| rnd()).collect();
            for first in [0, 37, 2900] {
                let mut want = CpuStage::select(&c, &iq, &pooled, p);
                want.retain(|&j| j >= first);
                assert_eq!(visible(&c, &pool, &iq, &pooled, first, p), want, "p {p} first {first}");
            }
        }
    }

    #[test]
    fn rounding_matches_f32_round() {
        let mut v = vec![0., -0., 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 126.5, -126.5, 127.49999, 127.5, -127.5, 128., -129.,
            f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e10, -1e10];
        v.extend([0.5f32, 1.5, 2.5].iter().flat_map(|x| [x.next_down(), x.next_up(), -x.next_down(), -x.next_up()]));
        let mut s = 1u64;
        for _ in 0..1_000_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 300.);
        }
        for x in v {
            assert_eq!(round_i8(x), x.round() as i8, "{x}");
        }
    }

    #[test]
    fn attention_matches_the_scalar_loop() {
        // 37 positions (groups of four and a remainder) of a cache with two heads of 16.
        let (hd, nkv, n) = (16, 2, 40);
        let mut s = 7u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let kc: Vec<f32> = (0..n * nkv * hd).map(|_| rnd()).collect();
        let vc: Vec<f32> = (0..n * nkv * hd).map(|_| rnd()).collect();
        let q: Vec<f32> = (0..hd).map(|_| rnd()).collect();
        let vis: Vec<usize> = (3..40).collect();
        let (acc, sum) = attend(&q, &vis, &kc, &vc, nkv * hd, hd, 0.25);
        let scores: Vec<f32> = vis.iter().map(|&j| dot_f32(&q, &kc[j * nkv * hd + hd..][..hd]) * 0.25).collect();
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let w: Vec<f32> = scores.iter().map(|x| (x - max).exp()).collect();
        let mut want = vec![0f32; hd];
        for (&j, wj) in vis.iter().zip(&w) {
            for (a, x) in want.iter_mut().zip(&vc[j * nkv * hd + hd..][..hd]) {
                *a += wj * x;
            }
        }
        assert_eq!(sum.to_bits(), w.iter().sum::<f32>().to_bits());
        assert!(acc.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}
