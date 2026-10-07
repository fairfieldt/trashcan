//! Tensor parallelism across GPUs and machines: every GPU of the group holds one rank's share of every layer
//! (`gpu::Split`: key heads with their value heads, query heads with their KV heads, FFN columns and
//! vocabulary rows) and the hidden state is replicated. Each mixer and FFN ends with a partial output
//! projection; the host adds its GPUs' partials, exchanges the sum with the other machine over the lanes
//! (`link`) and writes the total back, two all-reduces per layer (results/qwen/tp4-viability-20261006).
//!
//! Both machines run the same submissions in lockstep. The driver sends each forward's tokens, rollbacks and
//! checkpoints to the other machine on lane 0; the exchanges keep the machines together. Each machine merges
//! its ranks' candidates and the machines exchange theirs. Partials are added in a fixed order (this machine's
//! ranks, then the other machine's sum), and adding the two machines' sums commutes exactly, so every GPU keeps
//! a bitwise-identical residual.
use super::{
    cpu::Captures,
    embed,
    gpu::{micro_batches, GpuStage, Split, MAX_MICRO, TOP_K},
    pipeline::Draft,
    Config,
};
use crate::{generation::Sampler, gguf::Gguf, gpu::Instance, link::Lanes, vulkan::Device};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    net::TcpListener,
    path::Path,
    rc::Rc,
    time::{Duration, Instant},
};

pub type Candidates = Vec<(f32, u32)>;

pub struct TpGroup {
    pub gguf: Gguf,
    pub c: Config,
    shards: Vec<GpuStage>,
    /// The other machine of the group, if any.
    peer: Option<Lanes>,
    /// This machine starts every operation and tells the other one.
    driver: bool,
    sum: Vec<f32>,
    remote: Vec<f32>,
    /// Host threads for large all-reduces; the driver's choice, which it passes on with each forward.
    threads: usize,
    /// Seconds spent submitting segments, waiting for the GPUs, adding partials and exchanging with the
    /// other machine, since the group was created.
    pub seconds: Phases,
}

#[derive(Clone, Copy, Default, Debug, serde::Serialize)]
pub struct Phases {
    pub submit: f64,
    pub wait: f64,
    /// Everything the host does to add partials: `sum` and `total`.
    pub add: f64,
    pub exchange: f64,
    /// Adding this machine's partials.
    pub sum: f64,
    /// Adding the other machine's sum and writing the total for the local GPUs.
    pub total: f64,
    pub allreduces: u64,
}

impl Phases {
    /// Milliseconds spent in each phase since `earlier`, divided by `per`.
    pub fn ms_since(&self, earlier: &Phases, per: f64) -> Value {
        let ms = |a: f64, b: f64| (a - b) * 1e3 / per;
        json!({"submit": ms(self.submit, earlier.submit), "wait": ms(self.wait, earlier.wait), "add": ms(self.add, earlier.add),
            "exchange": ms(self.exchange, earlier.exchange), "sum": ms(self.sum, earlier.sum), "total": ms(self.total, earlier.total),
            "allreduces": self.allreduces - earlier.allreduces})
    }
}

/// Load ranks `first..first + gpus.len()` of `ranks` onto the local `gpus`, with their share of the MTP layer
/// in `mtp` (a GGUF holding it) when given.
#[allow(clippy::too_many_arguments)]
fn shards(model: &Path, gpus: &[usize], first: usize, ranks: usize, context: usize, max_batch: usize, verify: usize, mtp: Option<&Path>) -> Result<(Gguf, Config, Vec<GpuStage>)> {
    let g = Gguf::open(model)?;
    let c = Config::new(&g)?;
    let m = mtp.map(|p| Gguf::open(p).with_context(|| format!("opening {}", p.display()))).transpose()?;
    let inst = Rc::new(Instance::new()?);
    let all = inst.amd_gpus()?;
    let mut shards = Vec::new();
    for (i, &index) in gpus.iter().enumerate() {
        let pd = *all.get(index).with_context(|| format!("GPU {index} not found"))?;
        let split = Split { rank: first + i, ranks };
        let stage = GpuStage::shard(Device::new(inst.clone(), pd)?, &g, &c, context, max_batch, verify, split, m.as_ref())?;
        eprintln!("GPU{index}: rank {} of {ranks}{}, {:.1} MiB allocated", first + i, if m.is_some() { " + MTP" } else { "" }, stage.device_mib());
        shards.push(stage);
    }
    Ok((g, c, shards))
}

/// Micro-batches a forward of `n` tokens is split into, so that the host all-reduces one while the GPUs run
/// another: `TRASHCAN_TP_MICRO` (1: none), by default 2 for batches of at least 128 tokens.
fn micro_count(n: usize) -> usize {
    let want = std::env::var("TRASHCAN_TP_MICRO").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    if n >= 128 {
        usize::clamp(want, 1, MAX_MICRO)
    } else {
        1
    }
}

/// Host threads that add partials and write totals (`TRASHCAN_TP_THREADS`), for all-reduces of at least
/// PARALLEL_MIN elements; decode and verification batches stay on the calling thread.
fn host_threads() -> usize {
    std::env::var("TRASHCAN_TP_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 16)
}
const PARALLEL_MIN: usize = 1 << 16;

/// A host buffer that several threads write in disjoint ranges.
#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// `f(offset, chunk)` over `threads` contiguous chunks of `out`, the last on the calling thread.
fn parallel<F: Fn(usize, &mut [f32]) + Sync>(out: &mut [f32], threads: usize, f: F) {
    if threads <= 1 || out.is_empty() {
        return f(0, out);
    }
    let (len, size) = (out.len(), out.len().div_ceil(threads).next_multiple_of(16));
    std::thread::scope(|s| {
        let f = &f;
        for (i, chunk) in out.chunks_mut(size).enumerate() {
            if (i + 1) * size >= len {
                f(i * size, chunk);
            } else {
                s.spawn(move || f(i * size, chunk));
            }
        }
    });
}

/// Highest scores first, ties by lower id (as the GPU merge orders them), at most TOP_K.
fn best(c: &mut Candidates) {
    c.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    c.truncate(TOP_K);
}

fn encode(columns: &[Candidates]) -> Vec<u8> {
    columns.iter().flatten().flat_map(|(s, id)| [s.to_le_bytes(), id.to_le_bytes()]).flatten().collect()
}

fn decode(bytes: &[u8]) -> Vec<Candidates> {
    bytes
        .chunks_exact(TOP_K * 8)
        .map(|col| col.chunks_exact(8).map(|e| (f32::from_le_bytes(e[..4].try_into().unwrap()), u32::from_le_bytes(e[4..].try_into().unwrap()))).collect())
        .collect()
}

fn message(v: Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&v)?)
}

impl TpGroup {
    /// The driver: its `gpus` take the first ranks and, with `peer`, the other machine's GPUs the rest. With
    /// `mtp` (a GGUF holding the MTP layer) every rank also holds its share of the MTP block; the other machine
    /// then needs its own copy (`qwen-tp-node --mtp`).
    pub fn new(model: &Path, gpus: &[usize], mut peer: Option<Lanes>, context: usize, max_batch: usize, verify: usize, mtp: Option<&Path>) -> Result<Self> {
        let peer_gpus = match peer.as_mut() {
            Some(l) => {
                l.send_message(&message(json!({"probe": true}))?)?;
                let reply: Value = serde_json::from_slice(&l.recv_message()?)?;
                reply["gpus"].as_u64().context("the other machine did not report its GPUs")? as usize
            }
            None => 0,
        };
        let ranks = gpus.len() + peer_gpus;
        ensure!(ranks > 1, "tensor parallelism needs at least two GPUs");
        if let Some(l) = peer.as_mut() {
            let setup = json!({"first": gpus.len(), "ranks": ranks, "context": context, "batch": max_batch, "verify": verify, "mtp": mtp.is_some()});
            l.send_message(&message(json!({ "setup": setup }))?)?;
        }
        eprintln!("tp: {ranks} ranks, {} on this machine{}", gpus.len(), if peer.is_some() { ", the rest on the other" } else { "" });
        let (gguf, c, shards) = shards(model, gpus, 0, ranks, context, max_batch, verify, mtp)?;
        if let Some(l) = peer.as_mut() {
            let reply: Value = serde_json::from_slice(&l.recv_message()?)?;
            ensure!(reply.get("ok").is_some(), "the other machine failed: {}", reply["error"]);
        }
        let len = c.d * max_batch;
        Ok(Self { gguf, c, shards, peer, driver: true, sum: vec![0.; len], remote: vec![0.; len], threads: host_threads(), seconds: Phases::default() })
    }

    /// Forward `tokens` (embedded as `x`) at `pos`: candidates of the last column, or of every column of a
    /// verification batch, which also keeps the per-column recurrent states for `rollback`.
    pub fn forward(&mut self, x: &[f32], pos: usize, tokens: &[u32], verify: bool, dump: Option<&mut Captures>) -> Result<Vec<Candidates>> {
        let micro = if verify || dump.is_some() { 1 } else { micro_count(tokens.len()) };
        let threads = self.threads;
        self.tell(json!({"forward": {"pos": pos, "tokens": tokens, "verify": verify, "dump": dump.is_some(), "micro": micro, "threads": threads}}))?;
        self.run(x, pos, tokens.len(), verify, dump, micro)
    }

    fn run(&mut self, x: &[f32], pos: usize, n: usize, verify: bool, dump: Option<&mut Captures>, micro: usize) -> Result<Vec<Candidates>> {
        let micro = micro_batches(n, micro);
        let mut segments = 0;
        for s in self.shards.iter_mut() {
            segments = s.tp_prepare(x, pos, n, verify, dump.is_some(), &micro)?;
        }
        self.launch(segments, &micro)?;
        let columns = self.candidates(if verify && n > 1 { n } else { 1 })?;
        if let Some(m) = dump {
            // Whole tensors from the first rank; the logits joined from every rank's vocabulary slice, the other
            // machine's (later ranks) sent to the driver.
            self.shards[0].tp_captures(m)?;
            let mut logits = m.remove("result_output_part").unwrap_or_default();
            for s in &self.shards[1..] {
                let mut other = Captures::new();
                s.tp_captures(&mut other)?;
                logits.extend(other.remove("result_output_part").unwrap_or_default());
            }
            match (self.peer.as_mut(), self.driver) {
                (Some(l), true) => logits.extend(l.recv_message()?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))),
                (Some(l), false) => l.send_message(bytemuck::cast_slice(&logits))?,
                (None, _) => {}
            }
            m.insert("result_output".into(), logits);
        }
        Ok(columns)
    }

    /// Merge the ranks' candidates of the last segment, then the machines'. A machine whose ranks ranked no
    /// rows (a draft head limited to low ids) sends empty slots, which sort last.
    fn candidates(&mut self, cols: usize) -> Result<Vec<Candidates>> {
        let mut columns: Vec<Candidates> = vec![Vec::new(); cols];
        for s in &self.shards {
            for (c, top) in columns.iter_mut().zip(s.tp_top(cols)?) {
                c.extend(top);
            }
        }
        columns.iter_mut().for_each(best);
        if let Some(l) = self.peer.as_mut() {
            for c in columns.iter_mut() {
                c.resize(TOP_K, (f32::NEG_INFINITY, u32::MAX));
            }
            let mine = encode(&columns);
            let mut theirs = vec![0u8; mine.len()];
            l.exchange(&mine, &mut theirs)?;
            for (c, t) in columns.iter_mut().zip(decode(&theirs)) {
                c.extend(t);
                best(c);
            }
        }
        for c in columns.iter_mut() {
            c.retain(|e| e.1 != u32::MAX);
        }
        Ok(columns)
    }

    /// Chain up to `k` MTP drafts from `token` at `pos`, each sampled from the candidates among the first
    /// `vocab` token ids (0: all); stops after a token in `stop`.
    pub fn draft(&mut self, token: u32, pos: usize, k: usize, vocab: usize, sampler: &mut Sampler, stop: &[u32]) -> Result<Vec<Draft>> {
        let mut drafts = Vec::new();
        let mut token = token;
        for j in 0..k {
            self.tell(json!({"draft": {"token": token, "pos": pos + j, "first": j == 0, "vocab": vocab}}))?;
            let top = self.draft_step(token, pos + j, j == 0, vocab)?;
            token = sampler.sample(&top);
            drafts.push((token, top));
            if stop.contains(&token) {
                break;
            }
        }
        Ok(drafts)
    }

    fn draft_step(&mut self, token: u32, pos: usize, first: bool, vocab: usize) -> Result<Candidates> {
        let emb = embed(&self.gguf, &self.c, &[token])?;
        let mut segments = 0;
        for s in self.shards.iter_mut() {
            segments = s.tp_draft_prepare(&emb, pos, first, vocab)?;
        }
        self.launch(segments, &[(0, 1)])?;
        Ok(self.candidates(1)?.remove(0))
    }

    /// Run the segments prepared on every rank as the micro-batches `micro`, all-reducing between them.
    fn launch(&mut self, segments: usize, micro: &[(usize, usize)]) -> Result<()> {
        // By default every segment is submitted ahead and released by a semaphore signal, which costs the host
        // far less than a vkQueueSubmit per segment (TRASHCAN_TP_SUBMIT=1 submits each segment instead).
        let ahead = std::env::var_os("TRASHCAN_TP_SUBMIT").is_none();
        if ahead {
            let start = Instant::now();
            for s in self.shards.iter_mut() {
                s.tp_submit_all()?;
            }
            self.seconds.submit += start.elapsed().as_secs_f64();
        }
        let result = self.segments(segments, micro, ahead);
        if result.is_err() && ahead {
            for s in self.shards.iter_mut() {
                let _ = s.tp_release_all();
            }
        }
        result
    }

    /// Run the segments of the forward prepared on every rank, all-reducing between them. Segment `j` belongs to
    /// micro-batch `j % m` of the `m` in `micro`; its all-reduce releases segment `j + m`, the next one of the same
    /// micro-batch, while the GPUs run the segments in between.
    fn segments(&mut self, segments: usize, micro: &[(usize, usize)], ahead: bool) -> Result<()> {
        if ahead && std::env::var_os("TRASHCAN_TP_FREE").is_some() {
            // Timing only: every segment back to back without all-reduces (wrong results), the GPU floor.
            let start = Instant::now();
            for s in self.shards.iter_mut() {
                s.tp_release(segments - 1)?;
            }
            for s in &self.shards {
                s.tp_wait(segments - 1)?;
            }
            self.seconds.wait += start.elapsed().as_secs_f64();
            return Ok(());
        }
        let m = micro.len();
        // The first segment of every micro-batch needs no total.
        self.release(0..m.min(segments), ahead)?;
        for j in 0..segments {
            let start = Instant::now();
            for s in &self.shards {
                s.tp_wait(j)?;
            }
            self.seconds.wait += start.elapsed().as_secs_f64();
            if j + m < segments {
                let (row, n) = micro[j % m];
                self.allreduce(row, n)?;
                self.release(j + m..j + m + 1, ahead)?;
            }
        }
        Ok(())
    }

    /// Let the segments `range` run: release them (submitted ahead) or submit them.
    fn release(&mut self, range: std::ops::Range<usize>, ahead: bool) -> Result<()> {
        let start = Instant::now();
        for s in self.shards.iter_mut() {
            if ahead {
                if let Some(last) = range.clone().last() {
                    s.tp_release(last)?;
                }
            } else {
                for i in range.clone() {
                    s.tp_submit(i)?;
                }
            }
        }
        self.seconds.submit += start.elapsed().as_secs_f64();
        Ok(())
    }

    /// Add every rank's partial projection of rows `row..row + n` of the latest segment and hand their total
    /// to the local ranks.
    fn allreduce(&mut self, row: usize, n: usize) -> Result<()> {
        let len = n * self.c.d;
        let threads = if len >= PARALLEL_MIN { self.threads } else { 1 };
        let start = Instant::now();
        let sum = &mut self.sum[..len];
        let parts: Vec<&[f32]> = self.shards.iter().map(|s| s.tp_part(row, n)).collect();
        parallel(sum, threads, |at, out| {
            out.copy_from_slice(&parts[0][at..at + out.len()]);
            for p in &parts[1..] {
                for (a, b) in out.iter_mut().zip(&p[at..]) {
                    *a += b;
                }
            }
        });
        let added = Instant::now();
        let mut exchanged = added;
        let mut remote = None;
        if let Some(l) = self.peer.as_mut() {
            l.exchange(bytemuck::cast_slice(sum), bytemuck::cast_slice_mut(&mut self.remote[..len]))?;
            exchanged = Instant::now();
            remote = Some(&self.remote[..len]);
        }
        // Add the other machine's sum and write the total for every local GPU in the same pass.
        let totals: Vec<Ptr> = self.shards.iter().map(|s| Ptr(s.tp_total(row, n))).collect();
        parallel(sum, threads, |at, out| {
            if let Some(r) = remote {
                for (a, b) in out.iter_mut().zip(&r[at..]) {
                    *a += b;
                }
            }
            for t in &totals {
                unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), t.0.add(at), out.len()) };
            }
        });
        let (summed, totalled) = ((added - start).as_secs_f64(), exchanged.elapsed().as_secs_f64());
        self.seconds.add += summed + totalled;
        self.seconds.sum += summed;
        self.seconds.total += totalled;
        self.seconds.exchange += (exchanged - added).as_secs_f64();
        self.seconds.allreduces += 1;
        Ok(())
    }

    /// Device memory of each local rank.
    pub fn device_mib(&self) -> Vec<f64> {
        self.shards.iter().map(GpuStage::device_mib).collect()
    }

    /// Keep columns `0..=column` of the latest verification batch.
    pub fn rollback(&mut self, column: usize) -> Result<()> {
        self.tell(json!({ "rollback": column }))?;
        self.shards.iter_mut().try_for_each(|s| s.rollback(column))
    }

    pub fn save(&mut self, slot: usize) -> Result<()> {
        self.tell(json!({ "save": slot }))?;
        self.shards.iter_mut().try_for_each(|s| s.save(slot))
    }

    pub fn restore(&mut self, slot: usize) -> Result<()> {
        self.tell(json!({ "restore": slot }))?;
        self.shards.iter_mut().try_for_each(|s| s.restore(slot))
    }

    /// Pass an operation from the driver to the other machine.
    fn tell(&mut self, v: Value) -> Result<()> {
        if let (true, Some(l)) = (self.driver, self.peer.as_mut()) {
            l.send_message(&message(v)?)?;
        }
        Ok(())
    }

    /// The other machine's side: carry out the driver's operations until it disconnects or says goodbye.
    fn follow(&mut self) -> Result<()> {
        loop {
            let bytes = match self.peer.as_mut().context("no driver")?.recv_message() {
                Ok(b) => b,
                Err(_) => return Ok(()),
            };
            let m: Value = serde_json::from_slice(&bytes)?;
            if let Some(f) = m.get("forward") {
                let tokens: Vec<u32> = serde_json::from_value(f["tokens"].clone())?;
                let x = embed(&self.gguf, &self.c, &tokens)?;
                let pos = f["pos"].as_u64().context("forward position")? as usize;
                let mut dump = f["dump"].as_bool().unwrap_or(false).then(Captures::new);
                let micro = f["micro"].as_u64().unwrap_or(1) as usize;
                self.threads = f["threads"].as_u64().unwrap_or(1).clamp(1, 16) as usize;
                self.run(&x, pos, tokens.len(), f["verify"].as_bool().unwrap_or(false), dump.as_mut(), micro)?;
            } else if let Some(d) = m.get("draft") {
                let num = |k: &str| d[k].as_u64().with_context(|| format!("draft {k}")).map(|v| v as usize);
                self.draft_step(num("token")? as u32, num("pos")?, d["first"].as_bool().unwrap_or(false), num("vocab")?)?;
            } else if let Some(v) = m.get("rollback") {
                self.rollback(v.as_u64().context("rollback column")? as usize)?;
            } else if let Some(v) = m.get("save") {
                self.save(v.as_u64().context("checkpoint slot")? as usize)?;
            } else if let Some(v) = m.get("restore") {
                self.restore(v.as_u64().context("checkpoint slot")? as usize)?;
            } else if m.get("bye").is_some() {
                return Ok(());
            } else {
                bail!("unknown request {m}");
            }
        }
    }
}

impl Drop for TpGroup {
    fn drop(&mut self) {
        let _ = self.tell(json!({"bye": true}));
    }
}

/// The lanes to the other machine of a tensor-parallel group, from `--tp-connect` (TCP) or `--tp-stream`.
pub fn peer(connect: &[String], stream: &[String]) -> Result<Option<Lanes>> {
    ensure!(connect.is_empty() || stream.is_empty(), "give --tp-connect or --tp-stream, not both");
    Ok(if !connect.is_empty() {
        Some(Lanes::connect(connect)?)
    } else if !stream.is_empty() {
        Some(Lanes::streams(stream, false)?)
    } else {
        None
    })
}

/// Serve the other machine's ranks of tensor-parallel groups, one driver at a time, on TCP lanes accepted on
/// `listen` or on the USB4STREAM devices `streams`.
pub fn serve(model: &Path, mtp: Option<&Path>, gpus: &[usize], listen: Option<&str>, streams: &[String]) -> Result<()> {
    let listener = listen.map(|a| TcpListener::bind(a).with_context(|| format!("listening on {a}"))).transpose()?;
    eprintln!("tp-node: {} on {}, GPUs {gpus:?}", model.display(), listen.map_or(format!("{streams:?}"), str::to_owned));
    loop {
        let lanes = match &listener {
            Some(l) => Lanes::accept(l),
            None => Lanes::streams(streams, true),
        };
        if let Err(e) = lanes.and_then(|l| session(model, mtp, gpus, l)) {
            eprintln!("tp-node: session ended: {e:#}");
        }
        if listener.is_none() {
            // Let the driver close its ends of the streams before they reopen.
            std::thread::sleep(Duration::from_millis(500));
        }
    }
}

fn session(model: &Path, mtp: Option<&Path>, gpus: &[usize], mut lanes: Lanes) -> Result<()> {
    eprintln!("tp-node: session over {} {} lanes", lanes.count(), lanes.kind);
    loop {
        let m: Value = serde_json::from_slice(&lanes.recv_message()?)?;
        if m.get("probe").is_some() {
            lanes.send_message(&message(json!({"gpus": gpus.len()}))?)?;
            continue;
        }
        let s = m.get("setup").with_context(|| format!("unexpected request {m}"))?;
        let num = |k: &str| s[k].as_u64().with_context(|| format!("setup {k}")).map(|v| v as usize);
        let loaded = (|| {
            let m = match (s["mtp"].as_bool().unwrap_or(false), mtp) {
                (true, None) => bail!("the driver drafts; start this node with --mtp"),
                (want, m) => m.filter(|_| want),
            };
            shards(model, gpus, num("first")?, num("ranks")?, num("context")?, num("batch")?, num("verify")?, m)
        })();
        let (gguf, c, shards) = match loaded {
            Ok(v) => v,
            Err(e) => {
                lanes.send_message(&message(json!({"error": format!("{e:#}")}))?)?;
                return Err(e);
            }
        };
        lanes.send_message(&message(json!({"ok": true}))?)?;
        let len = c.d * num("batch")?;
        let mut group = TpGroup { gguf, c, shards, peer: Some(lanes), driver: false, sum: vec![0.; len], remote: vec![0.; len], threads: 1, seconds: Phases::default() };
        return group.follow();
    }
}
