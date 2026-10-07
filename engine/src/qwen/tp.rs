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
    vector,
    gpu::{GpuStage, Split, TOP_K},
    pipeline::Draft,
    Config,
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
    collections::HashSet,
    io::Write,
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

/// Every forward and draft step of a profiling run (`TRASHCAN_TP_PROFILE=FILE`), one JSON line per local rank:
/// the GPU timestamps (`gpu::profile_level`) in ns from the first, that first one in host CLOCK_MONOTONIC ns,
/// and the host times at which each segment was released, seen finished on this rank, and exchanged.
struct Profile {
    out: std::io::BufWriter<std::fs::File>,
    release: Vec<u64>,
    done: Vec<Vec<u64>>,
    exchanged: Vec<u64>,
    labelled: HashSet<(usize, u64)>,
    count: usize,
}

impl Profile {
    fn open(ranks: usize) -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("TRASHCAN_TP_PROFILE") else { return Ok(None) };
        let file = std::fs::File::create(&path).with_context(|| format!("creating {path:?}"))?;
        let done = vec![Vec::new(); ranks];
        Ok(Some(Self { out: std::io::BufWriter::new(file), release: Vec::new(), done, exchanged: Vec::new(), labelled: HashSet::new(), count: 0 }))
    }
}

#[derive(Clone, Copy, Default, Debug, serde::Serialize)]
pub struct Phases {
    pub submit: f64,
    pub wait: f64,
    pub add: f64,
    pub exchange: f64,
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
        let gpu = Device::new(inst.clone(), pd)?;
        if std::env::var("TRASHCAN_TP_ARENA").as_deref() != Ok("0") {
            gpu.use_arena();
        }
        let stage = GpuStage::shard(gpu, &g, &c, context, max_batch, verify, split, m.as_ref())?;
        eprintln!("GPU{index}: rank {} of {ranks}{}, {:.1} MiB allocated", first + i, if m.is_some() { " + MTP" } else { "" }, stage.device_mib());
        shards.push(stage);
    }
    Ok((g, c, shards))
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
        if let Some(l) = peer.as_mut() {
            let reply: Value = serde_json::from_slice(&l.recv_message()?)?;
            ensure!(reply.get("ok").is_some(), "the other machine failed: {}", reply["error"]);
        }
        let len = c.d * max_batch;
        let profile = Profile::open(shards.len())?;
        let norms = segment_norms(&gguf, &c)?;
        let rows = c.d * max_batch.min(8);
        Ok(Self {
            gguf,
            c,
            shards,
            peer,
            driver: true,
            sum: vec![0.; len],
            remote: vec![0.; len],
            seconds: Phases::default(),
            first: 0,
            profile,
            norms,
            resid: vec![0.; rows],
        })
    }

    /// Forward `tokens` (embedded as `x`) at `pos`: candidates of the last column, or of every column of a
    /// verification batch, which also keeps the per-column recurrent states for `rollback`.
    pub fn forward(&mut self, x: &[f32], pos: usize, tokens: &[u32], verify: bool, dump: Option<&mut Captures>) -> Result<Vec<Candidates>> {
        self.tell(json!({"forward": {"pos": pos, "tokens": tokens, "verify": verify, "dump": dump.is_some()}}))?;
        self.run(x, pos, tokens.len(), verify, dump)
    }

    fn run(&mut self, x: &[f32], pos: usize, n: usize, verify: bool, dump: Option<&mut Captures>) -> Result<Vec<Candidates>> {
        let mut segments = 0;
        for s in self.shards.iter_mut() {
            segments = s.tp_prepare(x, pos, n, verify, dump.is_some())?;
        }
        if self.shards[0].tp_hostnorm() {
            let len = n * self.c.d;
            self.resid[..len].copy_from_slice(x);
            normed_rows(&mut self.resid[..len], None, &self.norms[0], self.c.eps, &self.shards);
        }
        self.launch(segments, n)?;
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
        self.launch(segments, 1)?;
        Ok(self.candidates(1)?.remove(0))
    }

    /// Run the segments prepared on every rank, all-reducing `n` columns between them.
    fn launch(&mut self, segments: usize, n: usize) -> Result<()> {
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
        let result = self.segments(segments, n, ahead);
        if result.is_err() && ahead {
            for s in self.shards.iter_mut() {
                let _ = s.tp_release_all();
            }
        }
        result?;
        self.write_profile(n)
    }

    /// One profile line per local rank for the forward or draft step just finished.
    fn write_profile(&mut self, n: usize) -> Result<()> {
        let Some(p) = self.profile.as_mut() else { return Ok(()) };
        for (r, s) in self.shards.iter().enumerate() {
            let Some((pool, labels, ns, host)) = s.tp_stamps()? else { continue };
            let mut line = json!({"step": p.count, "rank": self.first + r, "n": n, "graph": pool, "ns": ns, "host0": host,
                "release": p.release, "done": p.done[r], "exchanged": p.exchanged});
            if p.labelled.insert((r, pool)) {
                line["labels"] = json!(*labels);
            }
            writeln!(p.out, "{line}")?;
        }
        p.out.flush()?;
        p.count += 1;
        p.release.clear();
        p.exchanged.clear();
        p.done.iter_mut().for_each(Vec::clear);
        Ok(())
    }

    /// Run the segments of the forward prepared on every rank, all-reducing between them.
    fn segments(&mut self, segments: usize, n: usize, ahead: bool) -> Result<()> {
        if ahead && std::env::var_os("TRASHCAN_TP_FREE").is_some() {
            // Timing only: the segments without all-reduces (wrong results), the GPU floor: back to back when
            // released by semaphore, each released as soon as all ranks finished the previous one when gated.
            let start = Instant::now();
            if self.shards[0].tp_gated() {
                for i in 0..segments {
                    for s in self.shards.iter_mut() {
                        s.tp_release(i)?;
                    }
                    for s in &self.shards {
                        s.tp_wait(i)?;
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
        for i in 0..segments {
            let start = Instant::now();
            if let Some(p) = self.profile.as_mut() {
                p.release.push(host_ns());
            }
            for s in self.shards.iter_mut() {
                if ahead {
                    s.tp_release(i)?;
                } else {
                    s.tp_submit(i)?;
                }
            }
            let submitted = Instant::now();
            for (r, s) in self.shards.iter().enumerate() {
                s.tp_wait(i)?;
                if let Some(p) = self.profile.as_mut() {
                    p.done[r].push(host_ns());
                }
            }
            self.seconds.submit += (submitted - start).as_secs_f64();
            self.seconds.wait += submitted.elapsed().as_secs_f64();
            if i + 1 < segments {
                self.allreduce(n, i)?;
                if let Some(p) = self.profile.as_mut() {
                    p.exchanged.push(host_ns());
                }
            }
        }
        Ok(())
    }

    /// Add every rank's partial projection of segment `seg` and hand the total, or the normed rows of the
    /// residual it updates, to the local ranks.
    fn allreduce(&mut self, n: usize, seg: usize) -> Result<()> {
        let len = n * self.c.d;
        let start = Instant::now();
        let sum = &mut self.sum[..len];
        sum.copy_from_slice(self.shards[0].tp_part(n));
        for s in &self.shards[1..] {
            for (a, b) in sum.iter_mut().zip(s.tp_part(n)) {
                *a += b;
            }
        }
        let added = Instant::now();
        if let Some(l) = self.peer.as_mut() {
            let remote = &mut self.remote[..len];
            // While a gated submission waits on the GPUs, a stalled exchange must fail (and drain them) well
            // before amdgpu's job timeout.
            let stall = if self.shards[0].tp_gated() { Duration::from_secs(2) } else { crate::link::STALL };
            l.exchange_within(bytemuck::cast_slice(sum), bytemuck::cast_slice_mut(remote), stall)?;
            for (a, b) in sum.iter_mut().zip(remote.iter()) {
                *a += b;
            }
        }
        let exchanged = Instant::now();
        if self.shards[0].tp_hostnorm() {
            normed_rows(&mut self.resid[..len], Some(sum), &self.norms[seg + 1], self.c.eps, &self.shards);
        } else {
            for s in &self.shards {
                s.tp_set_total(sum);
            }
        }
        self.seconds.add += (added - start).as_secs_f64() + exchanged.elapsed().as_secs_f64();
        self.seconds.exchange += (exchanged - added).as_secs_f64();
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
                self.run(&x, pos, tokens.len(), f["verify"].as_bool().unwrap_or(false), dump.as_mut())?;
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
        lanes.send_message(&message(json!({"ok": true}))?)?;
        let len = c.d * num("batch")?;
        let profile = Profile::open(shards.len())?;
        let norms = segment_norms(&gguf, &c)?;
        let rows = c.d * num("batch")?.min(8);
        let mut group = TpGroup {
            gguf,
            c,
            shards,
            peer: Some(lanes),
            driver: false,
            sum: vec![0.; len],
            remote: vec![0.; len],
            seconds: Phases::default(),
            first: num("first")?,
            profile,
            norms,
            resid: vec![0.; rows],
        };
        return group.follow();
    }
}
