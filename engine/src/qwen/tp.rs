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
//!
//! Decode, verification and draft steps (up to 8 columns) are one event-gated submission per rank
//! (`gpu::gated`). A prefill batch of at least 128 tokens runs as two micro-batches whose segments alternate, so
//! the host all-reduces one while the GPUs run the other; other batches release one submission per segment.
use super::{
    cpu::Captures,
    embed,
    gpu::{micro_batches, GpuStage, Split, MAX_MICRO, TOP_K},
    pipeline::Draft,
    vector, Config,
};
use crate::{
    generation::Sampler,
    gguf::Gguf,
    gpu::Instance,
    link::Lanes,
    vulkan::{host_ns, Device},
};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashSet},
    io::Write,
    net::TcpListener,
    path::Path,
    rc::Rc,
    time::{Duration, Instant},
};

/// Benchmark provenance. The latency marker is set by the wrapper holding the PM QoS request;
/// without the wrapper its value is unknown, rather than assumed to be zero.
pub fn host_settings() -> Value {
    let governor = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").ok().map(|s| s.trim().to_owned());
    let affinity = std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:").map(|x| x.trim().to_owned())));
    json!({"cpu_governor":governor,"cpu_affinity":affinity,"cpu_latency_us":std::env::var("TRASHCAN_CPU_LATENCY_US").ok().and_then(|v| v.parse::<u32>().ok())})
}

pub type Candidates = Vec<(f32, u32)>;

pub struct TpGroup {
    pub gguf: Gguf,
    pub c: Config,
    shards: Vec<GpuStage>,
    /// The other machine of the group, if any.
    peer: Option<Lanes>,
    /// Host settings reported by the peer at setup; absent with an older node.
    pub peer_host: Option<Value>,
    /// This machine starts every operation and tells the other one.
    driver: bool,
    sum: Vec<f32>,
    remote: Vec<f32>,
    /// Host threads for large all-reduces; the driver's choice, which it passes on with each forward.
    threads: usize,
    /// Seconds spent submitting segments, waiting for the GPUs, adding partials and exchanging with the
    /// other machine, since the group was created.
    pub seconds: Phases,
    /// Rank of the first local GPU.
    first: usize,
    profile: Option<Profile>,
    /// Residual norm weights in segment order (attention and FFN norm of each layer, then the final norm), and
    /// the residual of a forward that keeps it on the host (`gpu::hostnorm`).
    norms: Vec<Vec<f32>>,
    resid: Vec<f32>,
}

/// The norm weights that start each segment of a forward: per layer the attention and FFN norms, then the
/// final norm.
fn segment_norms(g: &Gguf, c: &Config) -> Result<Vec<Vec<f32>>> {
    let mut norms = Vec::new();
    for il in 0..c.layers {
        norms.push(vector(g, c.layer_tensor(g, il, "attn_norm.weight")?)?);
        norms.push(vector(g, c.layer_tensor(g, il, "post_attention_norm.weight")?)?);
    }
    norms.push(vector(g, g.tensor("output_norm.weight").context("missing output norm")?)?);
    Ok(norms)
}

/// Add `total` (when given) to the host residual rows `resid`, then write their RMS norm times `w` into every
/// local rank's input rows. Eight running sums in a fixed order keep the result identical on every machine
/// without a dependent add per element; separate simple loops vectorize.
fn normed_rows(resid: &mut [f32], total: Option<&[f32]>, w: &[f32], eps: f32, shards: &[GpuStage]) {
    let d = w.len();
    if let Some(total) = total {
        for (r, &t) in resid.iter_mut().zip(total) {
            *r += t;
        }
    }
    let (first, rest) = shards.split_first().expect("a tensor-parallel group has local ranks");
    let out = first.tp_hin(resid.len());
    for (x, o) in resid.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mut acc = [0f32; 8];
        for c in x.chunks_exact(8) {
            for j in 0..8 {
                acc[j] += c[j] * c[j];
            }
        }
        let ss = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        let inv = 1.0 / (ss / d as f32 + eps).sqrt();
        for ((o, &v), &wv) in o.iter_mut().zip(x).zip(w) {
            *o = v * inv * wv;
        }
    }
    for s in rest {
        s.tp_hin(resid.len()).copy_from_slice(out);
    }
}

/// Profiling (`TRASHCAN_TP_PROFILE=FILE`): every forward and draft step writes one JSON line per local rank and
/// graph (one per micro-batch): the GPU timestamps (`gpu::profile_level`) in ns from the first, that first one in
/// host CLOCK_MONOTONIC ns, and the host times at which each of the graph's segments was released, seen finished
/// on this rank, and exchanged (`tools/tp_profile.py`, `tools/tp_timeline.py`). Forwards of at least 64 tokens
/// (prefill) also add their GPU time per kernel to `kernels` (`TpGroup::profile`).
struct Profile {
    out: std::io::BufWriter<std::fs::File>,
    release: Vec<u64>,
    done: Vec<Vec<u64>>,
    exchanged: Vec<u64>,
    labelled: HashSet<(usize, u64)>,
    count: usize,
    /// Per local rank, (count, seconds) by label (`tally`).
    kernels: Vec<BTreeMap<String, (u64, f64)>>,
}

impl Profile {
    fn open(ranks: usize) -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("TRASHCAN_TP_PROFILE") else { return Ok(None) };
        let file = std::fs::File::create(&path).with_context(|| format!("creating {path:?}"))?;
        let (done, kernels) = (vec![Vec::new(); ranks], vec![BTreeMap::new(); ranks]);
        Ok(Some(Self { out: std::io::BufWriter::new(file), release: Vec::new(), done, exchanged: Vec::new(), labelled: HashSet::new(), count: 0, kernels }))
    }
}

/// Add one graph's GPU time to `into` by label (count, seconds): every stamped dispatch and copy from its start
/// to its end (`TRASHCAN_TP_PROFILE_DISPATCH=1`), "(segments)" from each segment's begin to its end, and
/// "(between segments)" from one segment's end to the next one's begin.
fn tally(labels: &[String], ns: &[f64], into: &mut BTreeMap<String, (u64, f64)>) {
    let mut add = |label: &str, t: f64| {
        let e = into.entry(label.to_owned()).or_default();
        e.0 += 1;
        e.1 += t * 1e-9;
    };
    let (mut begin, mut end, mut open) = (None, None, Vec::<(&str, f64)>::new());
    for (label, &t) in labels.iter().zip(ns) {
        match label.as_str() {
            "begin" => {
                if let Some(e) = end {
                    add("(between segments)", t - e);
                }
                begin = Some(t);
            }
            "end" => {
                if let Some(b) = begin.take() {
                    add("(segments)", t - b);
                }
                end = Some(t);
            }
            // A dispatch's labels come in pairs: its start, then its end.
            l => match open.iter().position(|o| o.0 == l) {
                Some(i) => add(l, t - open.swap_remove(i).1),
                None => open.push((l, t)),
            },
        }
    }
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
    /// Adding the other machine's sum and writing the total, or the next segment's normed rows, for the local GPUs.
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
    ensure!(!gpus.is_empty() && gpus.len() <= LOCAL, "1 to {LOCAL} local GPUs");
    let g = Gguf::open(model)?;
    let c = Config::new(&g)?;
    let m = mtp.map(|p| Gguf::open(p).with_context(|| format!("opening {}", p.display()))).transpose()?;
    let inst = Rc::new(Instance::new()?);
    let all = inst.amd_gpus()?;
    let mut shards = Vec::new();
    for (i, &index) in gpus.iter().enumerate() {
        let pd = *all.get(index).with_context(|| format!("GPU {index} not found"))?;
        let split = Split { rank: first + i, ranks };
        let gpu = Device::new(inst.clone(), pd)?;
        if std::env::var("TRASHCAN_TP_ARENA").as_deref() != Ok("0") {
            gpu.use_arena();
        }
        let mut stage = GpuStage::shard(gpu, &g, &c, context, max_batch, verify, split, m.as_ref())?;
        eprintln!("GPU{index}: rank {} of {ranks}{}, {:.1} MiB allocated", first + i, if m.is_some() { " + MTP" } else { "" }, stage.device_mib());
        if max_batch > 8 {
            // A full prompt chunk's graphs, so the first prompt does not wait for them.
            stage.tp_warm(max_batch, &micro_batches(max_batch, micro_count(max_batch)))?;
        }
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
/// PARALLEL_MIN elements; decode and verification batches stay on the calling thread. By default 6 when the
/// group spans two machines, where the all-reduce of one micro-batch is on the path to the next segment, else 1.
fn host_threads(peer: bool) -> usize {
    let default = if peer { 6 } else { 1 };
    std::env::var("TRASHCAN_TP_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(default).clamp(1, 16)
}
const PARALLEL_MIN: usize = 1 << 16;
/// Most local GPUs of a group.
const LOCAL: usize = 8;

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
    /// `mtp` (a GGUF holding the MTP layer) every rank also holds its share of the MTP block and of the draft
    /// head over the first `draft_vocab` ids; the other machine then needs its own copy (`qwen-tp-node --mtp`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &Path,
        gpus: &[usize],
        mut peer: Option<Lanes>,
        context: usize,
        max_batch: usize,
        verify: usize,
        mtp: Option<&Path>,
        draft_vocab: usize,
    ) -> Result<Self> {
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
            let setup = json!({"first": gpus.len(), "ranks": ranks, "context": context, "batch": max_batch, "verify": verify, "mtp": mtp.is_some(), "draft_vocab": draft_vocab});
            l.send_message(&message(json!({ "setup": setup }))?)?;
        }
        eprintln!("tp: {ranks} ranks, {} on this machine{}", gpus.len(), if peer.is_some() { ", the rest on the other" } else { "" });
        let (gguf, c, mut shards) = shards(model, gpus, 0, ranks, context, max_batch, verify, mtp)?;
        for s in shards.iter_mut() {
            s.tp_draft_head(&gguf, draft_vocab)?;
        }
        let mut peer_host = None;
        if let Some(l) = peer.as_mut() {
            let reply: Value = serde_json::from_slice(&l.recv_message()?)?;
            ensure!(reply.get("ok").is_some(), "the other machine failed: {}", reply["error"]);
            peer_host = reply.get("host").cloned();
        }
        let threads = host_threads(peer.is_some());
        let mut group = Self::assemble(gguf, c, shards, peer, 0, max_batch, threads)?;
        group.peer_host = peer_host;
        Ok(group)
    }

    /// A group of the local `shards` (ranks from `first`); the driver when it has no peer or `first` is 0.
    fn assemble(gguf: Gguf, c: Config, shards: Vec<GpuStage>, peer: Option<Lanes>, first: usize, max_batch: usize, threads: usize) -> Result<Self> {
        let (len, rows) = (c.d * max_batch, c.d * max_batch.min(8));
        let profile = Profile::open(shards.len())?;
        let norms = segment_norms(&gguf, &c)?;
        Ok(Self {
            gguf,
            c,
            shards,
            driver: peer.is_none() || first == 0,
            peer,
            peer_host: None,
            sum: vec![0.; len],
            remote: vec![0.; len],
            threads,
            seconds: Phases::default(),
            first,
            profile,
            norms,
            resid: vec![0.; rows],
        })
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
        if self.shards[0].tp_hostnorm() {
            let len = n * self.c.d;
            self.resid[..len].copy_from_slice(x);
            normed_rows(&mut self.resid[..len], None, &self.norms[0], self.c.eps, &self.shards);
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
            s.tp_draft_head(&self.gguf, vocab)?;
            segments = s.tp_draft_prepare(&emb, pos, first, vocab)?;
        }
        self.launch(segments, &[(0, 1)])?;
        Ok(self.candidates(1)?.remove(0))
    }

    /// Run the segments prepared on every rank as the micro-batches `micro`, all-reducing between them.
    fn launch(&mut self, segments: usize, micro: &[(usize, usize)]) -> Result<()> {
        // By default every segment is submitted ahead and released by a semaphore signal, which costs the host
        // far less than a vkQueueSubmit per segment (TRASHCAN_TP_SUBMIT=1 submits each segment instead). An
        // event-gated forward is always one submission.
        let ahead = self.shards[0].tp_gated() || std::env::var_os("TRASHCAN_TP_SUBMIT").is_none();
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
        result?;
        self.write_profile(micro)
    }

    /// One profile line per local rank and graph (micro-batch) for the forward or draft step just finished, with
    /// the host times of that graph's segments.
    fn write_profile(&mut self, micro: &[(usize, usize)]) -> Result<()> {
        let Some(p) = self.profile.as_mut() else { return Ok(()) };
        let m = micro.len();
        let every = |v: &[u64], slot: usize| v.iter().skip(slot).step_by(m).copied().collect::<Vec<_>>();
        let tokens: usize = micro.iter().map(|b| b.1).sum();
        for (r, s) in self.shards.iter().enumerate() {
            for (slot, (pool, labels, ns, host)) in s.tp_stamps()?.into_iter().enumerate() {
                if tokens >= 64 {
                    tally(&labels, &ns, &mut p.kernels[r]);
                }
                let mut line = json!({"step": p.count, "rank": self.first + r, "n": micro[slot].1, "graph": pool, "ns": ns, "host0": host,
                    "release": every(&p.release, slot), "done": every(&p.done[r], slot), "exchanged": every(&p.exchanged, slot)});
                if m > 1 {
                    line["micro"] = json!([slot, m]);
                }
                if p.labelled.insert((r, pool)) {
                    line["labels"] = json!(*labels);
                }
                writeln!(p.out, "{line}")?;
            }
        }
        p.out.flush()?;
        p.count += 1;
        p.release.clear();
        p.exchanged.clear();
        p.done.iter_mut().for_each(Vec::clear);
        Ok(())
    }

    /// Run the segments of the forward prepared on every rank, all-reducing between them. Segment `j` belongs to
    /// micro-batch `j % m` of the `m` in `micro`; its all-reduce releases segment `j + m`, the next one of the same
    /// micro-batch, while the GPUs run the segments in between.
    fn segments(&mut self, segments: usize, micro: &[(usize, usize)], ahead: bool) -> Result<()> {
        if ahead && std::env::var_os("TRASHCAN_TP_FREE").is_some() {
            // Timing only: the segments without all-reduces (wrong results), the GPU floor: back to back when
            // released by semaphore, each released as soon as all ranks finished the previous one when gated.
            let start = Instant::now();
            if self.shards[0].tp_gated() {
                for j in 0..segments {
                    for s in self.shards.iter_mut() {
                        s.tp_release(j)?;
                    }
                    for s in &self.shards {
                        s.tp_wait(j)?;
                    }
                }
            } else {
                for s in self.shards.iter_mut() {
                    s.tp_release_all()?;
                }
                for s in &self.shards {
                    s.tp_wait(segments - 1)?;
                }
            }
            self.seconds.wait += start.elapsed().as_secs_f64();
            return Ok(());
        }
        let m = micro.len();
        // The first segment of every micro-batch needs no total.
        self.release(0..m.min(segments), ahead)?;
        for j in 0..segments {
            let start = Instant::now();
            for (r, s) in self.shards.iter().enumerate() {
                s.tp_wait(j)?;
                if let Some(p) = self.profile.as_mut() {
                    p.done[r].push(host_ns());
                }
            }
            self.seconds.wait += start.elapsed().as_secs_f64();
            if j + m < segments {
                let (row, n) = micro[j % m];
                self.allreduce(row, n, j / m)?;
                if let Some(p) = self.profile.as_mut() {
                    p.exchanged.push(host_ns());
                }
                self.release(j + m..j + m + 1, ahead)?;
            }
        }
        Ok(())
    }

    /// Let the segments `range` run: release them (submitted ahead) or submit them.
    fn release(&mut self, range: std::ops::Range<usize>, ahead: bool) -> Result<()> {
        let start = Instant::now();
        if let Some(p) = self.profile.as_mut() {
            p.release.extend(range.clone().map(|_| host_ns()));
        }
        for s in self.shards.iter_mut() {
            for j in range.clone() {
                if ahead {
                    s.tp_release(j)?;
                } else {
                    s.tp_submit(j)?;
                }
            }
        }
        self.seconds.submit += start.elapsed().as_secs_f64();
        Ok(())
    }

    /// Add every rank's partial projection of rows `row..row + n` of the latest segment (segment `seg` of its
    /// micro-batch) and hand their total, or the normed rows of the residual it updates (`gpu::hostnorm`), to
    /// the local ranks.
    fn allreduce(&mut self, row: usize, n: usize, seg: usize) -> Result<()> {
        let len = n * self.c.d;
        let threads = if len >= PARALLEL_MIN { self.threads } else { 1 };
        let start = Instant::now();
        let sum = &mut self.sum[..len];
        // Fixed arrays rather than vectors: decode runs 128 all-reduces per token.
        let mut parts: [&[f32]; LOCAL] = [&[]; LOCAL];
        for (p, s) in parts.iter_mut().zip(&self.shards) {
            *p = s.tp_part(row, n);
        }
        let parts = &parts[..self.shards.len()];
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
            // While a gated submission waits on the GPUs, a stalled exchange must fail (and drain them) well
            // before amdgpu's job timeout.
            let stall = if self.shards[0].tp_gated() { Duration::from_secs(2) } else { crate::link::STALL };
            l.exchange_within(bytemuck::cast_slice(sum), bytemuck::cast_slice_mut(&mut self.remote[..len]), stall)?;
            exchanged = Instant::now();
            remote = Some(&self.remote[..len]);
        }
        if self.shards[0].tp_hostnorm() {
            if let Some(r) = remote {
                for (a, b) in sum.iter_mut().zip(r) {
                    *a += b;
                }
            }
            normed_rows(&mut self.resid[..len], Some(sum), &self.norms[seg + 1], self.c.eps, &self.shards);
        } else {
            // Add the other machine's sum and write the total for every local GPU in the same pass.
            let mut totals = [Ptr(std::ptr::null_mut()); LOCAL];
            for (t, s) in totals.iter_mut().zip(&self.shards) {
                *t = Ptr(s.tp_total(row, n));
            }
            let totals = &totals[..self.shards.len()];
            parallel(sum, threads, |at, out| {
                if let Some(r) = remote {
                    for (a, b) in out.iter_mut().zip(&r[at..]) {
                        *a += b;
                    }
                }
                for t in totals {
                    unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), t.0.add(at), out.len()) };
                }
            });
        }
        let (summed, totalled) = ((added - start).as_secs_f64(), exchanged.elapsed().as_secs_f64());
        self.seconds.add += summed + totalled;
        self.seconds.sum += summed;
        self.seconds.total += totalled;
        self.seconds.exchange += (exchanged - added).as_secs_f64();
        self.seconds.allreduces += 1;
        Ok(())
    }

    /// Start a new prefill tally while preserving the trace's graph labels and step identities.
    pub fn reset_profile_tally(&mut self) {
        if let Some(p) = &mut self.profile { p.kernels.iter_mut().for_each(BTreeMap::clear); }
    }

    /// Profiling (`TRASHCAN_TP_PROFILE`): GPU milliseconds of the prefill forwards so far per local rank: `busy_ms`
    /// in segments, and per label (count, ms), largest first (every dispatch with `TRASHCAN_TP_PROFILE_DISPATCH=1`).
    pub fn profile(&self) -> Value {
        let Some(p) = &self.profile else { return Value::Null };
        Value::Array(
            p.kernels
                .iter()
                .map(|k| {
                    let mut rows: Vec<_> = k.iter().filter(|r| r.0 != "(segments)").collect();
                    rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
                    let busy = k.get("(segments)").map_or(0., |v| v.1);
                    json!({"busy_ms": busy * 1e3, "kernels": rows.iter().map(|(k, (n, s))| json!([k, n, s * 1e3])).collect::<Vec<_>>()})
                })
                .collect(),
        )
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

    /// The other machine's side: carry out the driver's operations until it says goodbye or the lanes close.
    /// The driver can stay quiet for hours between operations (a server between requests), so the wait for the
    /// next one has no deadline; the messages and exchanges within an operation still fail when the lanes stall.
    fn follow(&mut self) -> Result<()> {
        loop {
            let bytes = self.peer.as_mut().context("no driver")?.recv_message_idle()?;
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
        match lanes.and_then(|l| session(model, mtp, gpus, l)) {
            Ok(()) => eprintln!("tp-node: session ended"),
            Err(e) => eprintln!("tp-node: session ended: {e:#}"),
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
        let (gguf, c, mut shards) = match loaded {
            Ok(v) => v,
            Err(e) => {
                lanes.send_message(&message(json!({"error": format!("{e:#}")}))?)?;
                return Err(e);
            }
        };
        let draft_vocab = s["draft_vocab"].as_u64().unwrap_or(0) as usize;
        for st in shards.iter_mut() {
            st.tp_draft_head(&gguf, draft_vocab)?;
        }
        lanes.send_message(&message(json!({"ok": true, "host": host_settings()}))?)?;
        let mut group = TpGroup::assemble(gguf, c, shards, Some(lanes), num("first")?, num("batch")?, 1)?;
        return group.follow();
    }
}
