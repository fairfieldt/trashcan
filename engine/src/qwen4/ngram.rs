//! The hashed n-gram table on the driver's host. Rows come from a resident copy (Q4_0 or Q8_0 blocks in RAM) or
//! straight from the GGUF on disk (pread through the page cache, or O_DIRECT page reads), and `bench` measures
//! both for decode (one token's rows) and prefill (a batch). See docs/qwen4.md.
use crate::gguf::{GgmlType, Gguf};
use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

enum Store {
    /// Resident rows in ggml block format.
    Ram(memmap2::MmapMut),
    /// Rows of the file at `offset`; `direct` reads whole 4 KiB pages without the page cache.
    File { file: File, offset: u64, direct: bool },
}

pub struct Table {
    pub dim: usize,
    pub rows: u64,
    pub ty: GgmlType,
    store: Store,
}

fn row_bytes(ty: GgmlType, dim: usize) -> usize {
    let (blk, size) = ty.block().expect("n-gram row type");
    dim / blk as usize * size as usize
}

/// Dequantize one row of `out.len()` values.
fn dequant(ty: GgmlType, b: &[u8], out: &mut [f32]) -> Result<()> {
    match ty {
        GgmlType::BF16 => {
            for (o, s) in out.iter_mut().zip(b.chunks_exact(2)) {
                *o = bf16::from_le_bytes([s[0], s[1]]).to_f32();
            }
        }
        GgmlType::Q8_0 => {
            for (o, s) in out.chunks_exact_mut(32).zip(b.chunks_exact(34)) {
                let d = f16::from_le_bytes([s[0], s[1]]).to_f32();
                for j in 0..32 {
                    o[j] = s[2 + j] as i8 as f32 * d;
                }
            }
        }
        GgmlType::Q4_0 => {
            for (o, s) in out.chunks_exact_mut(32).zip(b.chunks_exact(18)) {
                let d = f16::from_le_bytes([s[0], s[1]]).to_f32();
                for j in 0..16 {
                    o[j] = ((s[2 + j] & 15) as i32 - 8) as f32 * d;
                    o[j + 16] = ((s[2 + j] >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        other => bail!("unsupported n-gram row type {other}"),
    }
    Ok(())
}

/// ggml's reference Q4_0 rounding (`quantize_row_q4_0_ref`).
fn quantize_q4_0(x: &[f32], out: &mut [u8]) {
    for (v, b) in x.chunks_exact(32).zip(out.chunks_exact_mut(18)) {
        let mut max = 0f32;
        for &a in v {
            if a.abs() > max.abs() {
                max = a;
            }
        }
        let d = max / -8.;
        let id = if d != 0. { 1. / d } else { 0. };
        b[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        for j in 0..16 {
            let q0 = ((v[j] * id + 8.5) as i32).min(15) as u8;
            let q1 = ((v[j + 16] * id + 8.5) as i32).min(15) as u8;
            b[2 + j] = q0 | (q1 << 4);
        }
    }
}

impl Table {
    /// The table of a GGUF file, read in place (`file` may be one shard of a split model).
    pub fn open(path: &Path, direct: bool) -> Result<Self> {
        let g = Gguf::open_one(path)?;
        let t = g
            .tensor("per_layer_token_embd.weight")
            .context("no per_layer_token_embd in this file")?;
        let offset = g.file_offset(t);
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true);
        #[cfg(target_os = "linux")]
        if direct {
            use std::os::unix::fs::OpenOptionsExt;
            opts.custom_flags(libc::O_DIRECT);
        }
        let file = opts.open(path)?;
        #[cfg(target_os = "macos")]
        if direct {
            use std::os::unix::io::AsRawFd;
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        }
        Ok(Self {
            dim: t.dims[0] as usize,
            rows: t.dims[1],
            ty: t.ty,
            store: Store::File { file, offset, direct },
        })
    }

    pub fn row_bytes(&self) -> usize {
        row_bytes(self.ty, self.dim)
    }

    /// Drop this file's pages from the page cache (cold-cache measurements).
    pub fn evict(&self) {
        #[cfg(target_os = "linux")]
        if let Store::File { file, .. } = &self.store {
            use std::os::unix::io::AsRawFd;
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    /// A resident copy in `ty` (Q8_0 copies the rows, Q4_0 requantizes them), read sequentially with `threads`.
    pub fn to_ram(&self, ty: GgmlType, threads: usize) -> Result<Self> {
        ensure!(
            matches!(ty, GgmlType::Q8_0 | GgmlType::Q4_0),
            "resident rows must be Q8_0 or Q4_0"
        );
        let (src_rb, dst_rb) = (self.row_bytes(), row_bytes(ty, self.dim));
        let mut map = memmap2::MmapMut::map_anon(self.rows as usize * dst_rb)?;
        #[cfg(target_os = "linux")]
        map.advise(memmap2::Advice::HugePage)?;
        let chunk = 1 << 16; // rows per read
        let next = AtomicUsize::new(0);
        let n_chunks = (self.rows as usize).div_ceil(chunk);
        let dst = map.as_mut_ptr() as usize;
        std::thread::scope(|s| -> Result<()> {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| -> Result<()> {
                        let mut buf = vec![0u8; chunk * src_rb];
                        let mut row = vec![0f32; self.dim];
                        loop {
                            let c = next.fetch_add(1, Ordering::Relaxed);
                            if c >= n_chunks {
                                return Ok(());
                            }
                            let first = c * chunk;
                            let n = chunk.min(self.rows as usize - first);
                            self.read_rows(first as u64, &mut buf[..n * src_rb])?;
                            // Each chunk writes its own disjoint range of the destination.
                            let out = unsafe {
                                std::slice::from_raw_parts_mut((dst + first * dst_rb) as *mut u8, n * dst_rb)
                            };
                            for (s, d) in buf[..n * src_rb].chunks_exact(src_rb).zip(out.chunks_exact_mut(dst_rb)) {
                                if ty == self.ty {
                                    d.copy_from_slice(s);
                                } else {
                                    dequant(self.ty, s, &mut row)?;
                                    quantize_q4_0(&row, d);
                                }
                            }
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap()?;
            }
            Ok(())
        })?;
        Ok(Self {
            dim: self.dim,
            rows: self.rows,
            ty,
            store: Store::Ram(map),
        })
    }

    /// Contiguous rows from the file (buffered reads only).
    fn read_rows(&self, first: u64, out: &mut [u8]) -> Result<()> {
        let Store::File { file, offset, direct } = &self.store else {
            bail!("not a file table")
        };
        ensure!(!direct, "sequential reads need a buffered table");
        file.read_exact_at(out, offset + first * self.row_bytes() as u64)?;
        Ok(())
    }

    /// One row as f32; `page` is scratch for direct reads (8 KiB plus alignment).
    fn row(&self, r: u32, out: &mut [f32], page: &mut [u8]) -> Result<()> {
        ensure!((r as u64) < self.rows, "n-gram row {r} out of range");
        let rb = self.row_bytes();
        match &self.store {
            Store::Ram(map) => dequant(self.ty, &map[r as usize * rb..(r as usize + 1) * rb], out),
            Store::File {
                file,
                offset,
                direct: false,
            } => {
                let mut b = [0u8; 512];
                file.read_exact_at(&mut b[..rb], offset + r as u64 * rb as u64)?;
                dequant(self.ty, &b[..rb], out)
            }
            Store::File {
                file,
                offset,
                direct: true,
            } => {
                let at = offset + r as u64 * rb as u64;
                let start = at & !4095;
                let len = ((at + rb as u64).next_multiple_of(4096) - start) as usize;
                let shift = page.as_ptr().align_offset(4096);
                let buf = &mut page[shift..shift + len];
                file.read_exact_at(buf, start)?;
                let s = (at - start) as usize;
                dequant(self.ty, &buf[s..s + rb], out)
            }
        }
    }

    /// Gather `rows` into `out` (`rows.len() x dim`), split over `threads` scoped threads.
    pub fn gather(&self, rows: &[u32], out: &mut [f32], threads: usize) -> Result<()> {
        ensure!(out.len() == rows.len() * self.dim, "n-gram output length");
        let per = rows.len().div_ceil(threads.max(1)).max(1);
        std::thread::scope(|s| -> Result<()> {
            let handles: Vec<_> = rows
                .chunks(per)
                .zip(out.chunks_mut(per * self.dim))
                .map(|(rs, os)| {
                    s.spawn(move || -> Result<()> {
                        let mut page = vec![0u8; 3 * 4096];
                        for (&r, o) in rs.iter().zip(os.chunks_exact_mut(self.dim)) {
                            self.row(r, o, &mut page)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap()?;
            }
            Ok(())
        })
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// Measure one storage mode. Rows come from hashing `ids` (real token ids, cycled) or are uniformly random.
#[allow(clippy::too_many_arguments)]
pub fn bench(
    meta: &Path,
    table: &Path,
    mode: &str,
    threads: usize,
    ids: Option<&Path>,
    tokens: usize,
    batch: usize,
) -> Result<serde_json::Value> {
    let g = Gguf::open_one(meta)?;
    let c = super::Config::new(&g)?;
    let ng = c.ngram.clone().context("the model has no n-gram embedding")?;
    let direct = mode == "direct";
    let start = Instant::now();
    let file = Table::open(table, direct)?;
    if mode.ends_with("cold") || direct {
        file.evict();
    }
    let t = match mode {
        "ram-q4" => file.to_ram(GgmlType::Q4_0, threads)?,
        "ram-q8" => file.to_ram(GgmlType::Q8_0, threads)?,
        "buffered" | "buffered-cold" | "direct" => file,
        other => bail!("unknown mode {other}"),
    };
    let load = start.elapsed().as_secs_f64();
    // Token stream: real ids (cycled) or random ids.
    let stream: Vec<u32> = match ids {
        Some(p) => {
            let s = std::fs::read_to_string(p)?;
            let v: Vec<u32> = s.split_whitespace().map(|x| x.parse()).collect::<Result<_, _>>()?;
            v.iter().cycle().take(tokens).copied().collect()
        }
        None => {
            let mut x = 0x9e3779b97f4a7c15u64;
            (0..tokens)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x % c.vocab as u64) as u32
                })
                .collect()
        }
    };
    let rows = ng.rows(&[], &stream);
    let heads = ng.heads();
    let mut out = vec![0f32; batch * heads * t.dim];
    // Decode: one token's rows at a time, single-threaded and over `threads` threads.
    let decode = |th: usize, out: &mut [f32]| -> Result<Vec<f64>> {
        let n = tokens.min(4096);
        let mut lat = Vec::with_capacity(n);
        for tok in 0..n {
            let r = &rows[tok * heads..(tok + 1) * heads];
            let s = Instant::now();
            t.gather(r, &mut out[..heads * t.dim], th)?;
            lat.push(s.elapsed().as_secs_f64() * 1e6);
        }
        lat.sort_by(f64::total_cmp);
        Ok(lat)
    };
    let one = decode(1, &mut out)?;
    let many = decode(heads.min(threads), &mut out)?;
    // Prefill: batches of `batch` tokens over `threads` threads, the rows of a batch in token order.
    if direct || mode == "buffered-cold" {
        t.evict();
    }
    let s = Instant::now();
    let mut done = 0;
    while done + batch <= tokens {
        t.gather(&rows[done * heads..(done + batch) * heads], &mut out, threads)?;
        done += batch;
    }
    let prefill = done as f64 / s.elapsed().as_secs_f64();
    let summary = |v: &[f64]| {
        serde_json::json!({"mean_us": v.iter().sum::<f64>() / v.len() as f64, "p50_us": percentile(v, 0.5),
            "p90_us": percentile(v, 0.9), "p99_us": percentile(v, 0.99)})
    };
    Ok(serde_json::json!({
        "mode": mode, "row_type": t.ty.to_string(), "row_bytes": t.row_bytes(), "rows": t.rows,
        "resident_gib": if mode.starts_with("ram") { t.rows as f64 * t.row_bytes() as f64 / (1u64 << 30) as f64 } else { 0. },
        "load_s": load, "threads": threads, "ids": ids.map(|p| p.display().to_string()), "tokens": tokens,
        "decode_1_thread": summary(&one), "decode_parallel": summary(&many), "parallel_threads": heads.min(threads),
        "prefill_batch": batch, "prefill_tok_s": prefill,
    }))
}
