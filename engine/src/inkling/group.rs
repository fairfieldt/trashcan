//! The host side of a tensor-parallel Inkling group: the residual, the block states and the loop passes, the
//! sublayer-output convolutions and the loop controllers, and the all-reduces that join the ranks' partial outputs
//! (this machine's GPUs, then the other machines over `qwen::tp::Mesh`). Every machine runs the same host math on
//! the same totals in the same order, so the residuals stay bitwise identical; the driver (machine 0) sends each
//! operation down the mesh first.
use super::{
    embed,
    gpu::{Seg, Shard, Split, GATED_MAX, MICRO, WIDE_FROM},
    conv_blocks, mix_blocks, rms_into, rms_inv, App, Config, HostLayer, HostWeights, MIX_STATES,
};
use crate::{
    gguf::Gguf,
    gpu::Instance,
    link::Lanes,
    qwen::tp::{self, host_settings, Mesh},
    qwen::wire::Wire,
    vulkan::Device,
};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    net::TcpListener,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

pub type Candidates = Vec<(f32, u32)>;
/// A prompt slot that takes an image embedding row (`Group::forward_rows`).
pub const IMAGE_ROW: u32 = u32::MAX;
pub const TOP_K: usize = 64;

/// What a forward scores: the last row, every row (verification), or nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    Last,
    All,
    None,
}

/// Seconds spent in each part of the forwards since the group was created.
#[derive(Clone, Copy, Debug, Default)]
pub struct Phases {
    pub host: f64,
    pub gpu: f64,
    /// Waits for attention and MoE segments.
    pub gpu_attn: f64,
    pub gpu_moe: f64,
    /// The host's passes over the rows between segments (convolutions, residual, mixes, input writes), and adding
    /// the local ranks' partials.
    pub host_in: f64,
    pub host_sum: f64,
    pub exchange: f64,
    pub forwards: u64,
    pub tokens: u64,
}

/// Reference residuals for teacher-forced checks: `l_out-SLOT` rows of every token (tools/dump-activations with
/// DUMP_FULL), read from `dir`. A forward then replaces its residual with the reference after every application and
/// records the relative error it had.
pub struct Reference {
    pub dir: PathBuf,
    pub errors: Vec<(String, f64)>,
}

impl Reference {
    pub fn new(dir: &Path) -> Self {
        Self { dir: dir.to_owned(), errors: Vec::new() }
    }
    pub fn read(&self, name: &str) -> Result<Vec<f32>> {
        let bytes = std::fs::read(self.dir.join(format!("{name}.f32"))).with_context(|| format!("reference {name}"))?;
        Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
    /// Compare `ours` (n rows) with the reference `name`, record the relative L2 error and, with `force`, replace
    /// `ours` by the reference.
    fn check(&mut self, name: &str, ours: &mut [f32], force: bool) -> Result<()> {
        let want = self.read(name)?;
        ensure!(want.len() == ours.len(), "reference {name} has {} values, the forward {}", want.len(), ours.len());
        let (mut num, mut den) = (0f64, 0f64);
        for (a, b) in ours.iter().zip(&want) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
        }
        self.errors.push((name.to_owned(), (num / den.max(1e-30)).sqrt()));
        if force {
            ours.copy_from_slice(&want);
        }
        Ok(())
    }
}

pub struct Group {
    pub gguf: Gguf,
    pub c: Config,
    hw: HostWeights,
    shards: Vec<Shard>,
    mesh: Option<Mesh>,
    driver: bool,
    sched: Vec<App>,
    pub pos: usize,
    pub context: usize,
    max_batch: usize,
    attn_hist: Vec<Vec<f32>>,
    ffn_hist: Vec<Vec<f32>>,
    sum: Vec<f32>,
    remote: Vec<f32>,
    pool: Pool,
    pub seconds: Phases,
    /// GPUs on every machine, for reports.
    pub ranks: usize,
    /// The DSpark drafter, when loaded (`enable_draft`), and what speculative decoding keeps of the last forward.
    drafter: Option<Drafter>,
    spec: Spec,
    /// Keep every row's head input (`hidden`) after each forward, for `logits_all` (`inkling logits`).
    keep_logits: bool,
    hidden: Vec<f32>,
}

/// The drafter on this machine's ranks, and the recurrent head's context of the latest committed positions.
struct Drafter {
    host: super::draft::DraftHost,
    ranks: Vec<super::draft::DraftRank>,
    /// `f_rnn` of position p at slot p % (BLOCK + 1), with its position.
    rnn: Vec<(usize, Vec<f32>)>,
}

/// The state of the last forward that speculative decoding needs: every row's target features for the drafter's
/// context ([rows][4][d]), and for a verification batch each slot's sublayer outputs.
/// The host histories stay at the start of the batch until `rollback` commits its prefix.
#[derive(Default)]
struct Spec {
    features: Vec<f32>,
    /// The batch's first position and rows.
    pos: usize,
    n: usize,
    verifying: bool,
    /// Per slot, the pre-convolution rows of its attention and MoE outputs in this batch.
    attn_y: Vec<Vec<f32>>,
    ffn_y: Vec<Vec<f32>>,
    /// Seconds in the drafter's block, its recurrent steps, verification forwards and rollbacks (with the context
    /// writes), and the rounds.
    t_block: f64,
    t_steps: f64,
    t_verify: f64,
    t_rollback: f64,
    rounds: usize,
}

/// The applications whose input residual the drafter reads (execution order), then the final hidden state.
const FEATURE_VISITS: [usize; 3] = [5, 23, 41];

/// Batches of at least this many tokens run as two micro-batches (`Group::drive`).
const MICRO_MIN: usize = 64;

/// The host state of one micro-batch of a forward: `n` rows of the batch with their own input and output buffers
/// on every rank (`mb`).
struct Flow {
    mb: usize,
    /// The flow's first row in the batch.
    row0: usize,
    n: usize,
    plan: Vec<Seg>,
    /// The residual, and the RMS-normed block states before it.
    cur: Vec<f32>,
    normed: Vec<Vec<f32>>,
    /// The prelude's output and its number of blocks.
    prelude: (Vec<f32>, usize),
    /// The bank of the loop pass before the one in progress, RMS-normed: its block after the first core block, and
    /// its output. The loop messages are mixes of the two.
    bank: [Vec<f32>; 2],
    /// The submission of the segment in flight on every local rank.
    tickets: Vec<u64>,
}

/// Parse a request message.
fn message(v: Value) -> Result<Vec<u8>> {
    tp::message(v)
}

fn load_shards(model: &Path, gpus: &[usize], first: usize, ranks: usize, context: usize, max_batch: usize) -> Result<(Gguf, Config, Vec<Shard>)> {
    let gguf = Gguf::open(model)?;
    let c = Config::new(&gguf)?;
    let inst = Rc::new(Instance::new()?);
    let all = inst.amd_gpus()?;
    let mut shards = Vec::new();
    for (i, &gi) in gpus.iter().enumerate() {
        let pd = *all.get(gi).with_context(|| format!("no GPU {gi}"))?;
        let dev = Device::new(inst.clone(), pd)?;
        dev.use_arena();
        let t = Instant::now();
        let s = Shard::new(dev, &gguf, &c, Split { rank: first + i, ranks }, context, max_batch)?;
        eprintln!("inkling: rank {} on GPU {gi} loaded in {:.1} s, {:.0} MiB", first + i, t.elapsed().as_secs_f64(), s.device_mib());
        shards.push(s);
    }
    Ok((gguf, c, shards))
}

impl Group {
    /// The driver: its `gpus` take the first ranks; with `mesh` the other machines' GPUs take the rest in machine order.
    pub fn new(model: &Path, gpus: &[usize], mut mesh: Option<Mesh>, context: usize, max_batch: usize, forced_rows: usize, exact: bool) -> Result<Self> {
        ensure!(max_batch > 0, "--batch must be positive");
        ensure!(forced_rows <= max_batch, "the teacher-forced prompt must fit one batch (--batch)");
        // Each segment uses one micro-batch. Short forwards and teacher-forced prompts run whole.
        let shard_rows = max_batch.div_ceil(MICRO).max(max_batch.min(MICRO_MIN - 1)).max(forced_rows);
        let mut counts = BTreeMap::from([(0usize, gpus.len())]);
        if let Some(m) = mesh.as_mut() {
            m.tell_children(&message(json!({"probe": true}))?)?;
            let mut cabled = BTreeMap::from([(0usize, m.peers())]);
            for reply in m.children_replies(None)? {
                tp::merge_counts(&mut counts, &reply)?;
                tp::merge_cabled(&mut cabled, &reply);
            }
            ensure!(counts.keys().copied().eq(0..counts.len()), "machines {:?} answered the probe", counts.keys().collect::<Vec<_>>());
            m.full = tp::fully_cabled(&cabled, counts.len());
            m.join(counts.len())?;
            // All-reduces of decode and verification batches go in f16, each machine rounding its partial sums (as
            // the Qwen groups' do): over 14,329 positions the engine's KLD from an f32 reference is the same with
            // either wire (0.00130 against 0.00128, standard errors 0.00004; docs/inkling.md), and decode is 2 to 3%
            // faster. Comparisons with references (`exact`) keep exact f32 sums. TRASHCAN_TP_WIRE=f32 or f16 decides
            // for every command.
            m.wire = match std::env::var("TRASHCAN_TP_WIRE").as_deref() {
                Ok("f16") => Wire::F16,
                Ok("f32") => Wire::F32,
                _ if exact => Wire::F32,
                _ => Wire::F16,
            };
        }
        let firsts: Vec<usize> = counts.values().scan(0, |at, &n| { *at += n; Some(*at - n) }).collect();
        let ranks: usize = counts.values().sum();
        if let Some(m) = mesh.as_mut() {
            let setup = json!({"firsts": firsts, "machines": counts.len(), "ranks": ranks, "context": context, "batch": shard_rows, "wire": m.wire.name(), "full": m.full});
            m.tell_children(&message(json!({ "setup": setup }))?)?;
        }
        let wire = mesh.as_ref().map_or("none", |m| m.wire.name());
        eprintln!("inkling: {ranks} ranks over {} machine(s), GPUs per machine {:?}, small all-reduces in {wire}", counts.len(), counts.values().collect::<Vec<_>>());
        let (gguf, c, shards) = load_shards(model, gpus, 0, ranks, context, shard_rows)?;
        if let Some(m) = mesh.as_mut() {
            for reply in m.children_replies(Some(Duration::from_secs(10 * 60))).context("waiting for the other machines to load")? {
                ensure!(reply.get("ok").is_some(), "another machine failed: {}", reply["error"]);
            }
        }
        Self::assemble(gguf, c, shards, mesh, context, max_batch, ranks)
    }

    fn assemble(gguf: Gguf, c: Config, shards: Vec<Shard>, mesh: Option<Mesh>, context: usize, max_batch: usize, ranks: usize) -> Result<Self> {
        let hw = HostWeights::load(&gguf, &c)?;
        let slots = c.slots();
        let d = c.d;
        Ok(Self {
            sched: c.schedule(),
            attn_hist: vec![vec![0.; 3 * d]; slots],
            ffn_hist: vec![vec![0.; 3 * d]; slots],
            sum: vec![0.; max_batch * d],
            remote: vec![0.; max_batch * d],
            pool: Pool::new(HOST_THREADS - 1),
            driver: mesh.as_ref().is_none_or(|m| m.machine == 0),
            gguf,
            hw,
            shards,
            mesh,
            pos: 0,
            context,
            max_batch,
            seconds: Phases::default(),
            ranks,
            drafter: None,
            spec: Spec::default(),
            keep_logits: false,
            hidden: Vec::new(),
            c,
        })
    }

    /// Load the DSpark drafter (`dspark-*.gguf`; the same path on every machine) onto every local rank.
    pub fn enable_draft(&mut self, path: &Path) -> Result<()> {
        self.tell(json!({"draft_load": path.display().to_string()}))?;
        self.load_draft(path)
    }

    fn load_draft(&mut self, path: &Path) -> Result<()> {
        let g = Gguf::open(path)?;
        let host = super::draft::DraftHost::new(&g)?;
        let mut ranks = Vec::new();
        for s in &self.shards {
            ranks.push(super::draft::DraftRank::new(s.gpu.clone(), &g, &self.gguf, &self.c, s.split, self.context)?);
        }
        self.drafter = Some(Drafter { host, ranks, rnn: vec![(usize::MAX, Vec::new()); super::draft::BLOCK + 1] });
        Ok(())
    }

    pub fn drafting(&self) -> bool {
        self.drafter.is_some()
    }

    /// Mean milliseconds per speculative round: block, steps, verify, rollback; and the rounds.
    pub fn draft_profile(&self) -> ([f64; 4], usize) {
        let s = &self.spec;
        let r = s.rounds.max(1) as f64 * 1e-3;
        ([s.t_block / r, s.t_steps / r, s.t_verify / r, s.t_rollback / r], s.rounds)
    }

    /// Write the drafter's context entries of the last forward's first `rows` rows (committed tokens).
    fn commit_context(&mut self, rows: usize) -> Result<()> {
        let d = self.c.d;
        let Some(dr) = self.drafter.as_mut() else { return Ok(()) };
        let nl = dr.host.layers;
        let pos = self.spec.pos;
        for start in (0..rows).step_by(64) {
            let n = (rows - start).min(64);
            let mut ctx = Vec::with_capacity(n * nl * d);
            for t in start..start + n {
                let (c, rnn) = dr.host.contexts(&self.spec.features[t * 4 * d..(t + 1) * 4 * d], d);
                ctx.extend(c);
                let p = pos + t;
                dr.rnn[p % (super::draft::BLOCK + 1)] = (p, rnn);
            }
            let pend = dr.ranks.iter().map(|r| r.inject(&ctx, pos + start, n)).collect::<Result<Vec<_>>>()?;
            super::draft::wait_all(pend)?;
        }
        Ok(())
    }

    /// Draft up to `n_max` tokens after `last` (the latest sampled token, which goes to position `self.pos`).
    pub fn draft(&mut self, last: u32, n_max: usize) -> Result<Vec<u32>> {
        self.tell(json!({"draft": {"token": last, "n": n_max}}))?;
        self.draft_local(last, n_max)
    }

    fn draft_local(&mut self, last: u32, n_max: usize) -> Result<Vec<u32>> {
        use super::draft::BLOCK;
        let pos0 = self.pos;
        ensure!(pos0 > 0 && n_max < BLOCK, "drafting needs a committed token and at most {} drafts", BLOCK - 1);
        let dr = self.drafter.as_ref().context("no drafter loaded")?;
        if n_max == 0 {
            return Ok(Vec::new());
        }
        let (p, rnn) = &dr.rnn[(pos0 - 1) % (BLOCK + 1)];
        ensure!(*p == pos0 - 1, "the drafter's context of position {} is missing", pos0 - 1);
        let mut rows = embed(&self.gguf, &self.c, &self.hw, &[last])?;
        for _ in 1..BLOCK {
            rows.extend_from_slice(&dr.host.mask);
        }
        let rnn = rnn.clone();
        let t0 = std::time::Instant::now();
        let pend = dr.ranks.iter().map(|r| r.block(&rows, pos0, &rnn, last, n_max)).collect::<Result<Vec<_>>>()?;
        super::draft::wait_all(pend)?;
        self.spec.t_block += t0.elapsed().as_secs_f64();
        let mut out = Vec::new();
        for row in 1..=n_max {
            let t1 = std::time::Instant::now();
            let ranks = &self.drafter.as_ref().unwrap().ranks;
            if row > 1 {
                let prev = *out.last().unwrap();
                let pend = ranks.iter().map(|r| r.step(row, prev)).collect::<Result<Vec<_>>>()?;
                super::draft::wait_all(pend)?;
            }
            let mut best = (f32::NEG_INFINITY, u32::MAX);
            for r in ranks {
                let b = r.top()?;
                if b.0 > best.0 || (b.0 == best.0 && b.1 < best.1) {
                    best = b;
                }
            }
            self.spec.t_steps += t1.elapsed().as_secs_f64();
            if let Some(m) = self.mesh.as_mut() {
                for round in 0..m.rounds() {
                    let mine = [best.0.to_le_bytes(), best.1.to_le_bytes()].concat();
                    let mut theirs = [0u8; 8];
                    m.lanes(m.partner(round))?.exchange(&mine, &mut theirs)?;
                    let o = (f32::from_le_bytes(theirs[..4].try_into().unwrap()), u32::from_le_bytes(theirs[4..].try_into().unwrap()));
                    if o.0 > best.0 || (o.0 == best.0 && o.1 < best.1) {
                        best = o;
                    }
                }
            }
            out.push(best.1);
        }
        Ok(out)
    }

    /// Forward a verification batch (the latest sampled token, then drafts): every row's candidates. Its histories
    /// are kept until `rollback` says how many rows to keep.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Candidates>> {
        ensure!(tokens.len() > 1 && tokens.len() <= GATED_MAX, "a verification batch has 2 to {GATED_MAX} tokens");
        let t0 = std::time::Instant::now();
        self.tell(json!({"forward": {"tokens": tokens, "score": "all", "verify": true}}))?;
        let r = self.verify_local(tokens);
        self.spec.t_verify += t0.elapsed().as_secs_f64();
        self.spec.rounds += 1;
        r
    }

    fn verify_local(&mut self, tokens: &[u32]) -> Result<Vec<Candidates>> {
        ensure!(self.drafter.is_some(), "verification batches come from the drafter");
        self.spec.verifying = true;
        let x = embed(&self.gguf, &self.c, &self.hw, tokens)?;
        let r = self.run(x, tokens.len(), Score::All, None, 1);
        self.spec.verifying = r.is_ok();
        r
    }

    /// Keep the first `keep` rows of the last verification batch: the position, every convolution history and the
    /// drafter's context go back to just after them.
    pub fn rollback(&mut self, keep: usize) -> Result<()> {
        let t0 = std::time::Instant::now();
        self.tell(json!({"rollback": keep}))?;
        let r = self.rollback_local(keep);
        self.spec.t_rollback += t0.elapsed().as_secs_f64();
        r
    }

    fn rollback_local(&mut self, keep: usize) -> Result<()> {
        ensure!(self.spec.verifying && keep >= 1 && keep <= self.spec.n, "nothing to roll back");
        let d = self.c.d;
        let n = self.spec.n;
        // Each slot runs once. Verification leaves its history at the pre-round position
        // until the accepted prefix is known, then appends just those rows.
        let appended = keep.min(3) * d;
        let end = keep * d;
        for (hist, rows) in self.attn_hist.iter_mut().zip(&self.spec.attn_y)
            .chain(self.ffn_hist.iter_mut().zip(&self.spec.ffn_y))
        {
            hist.copy_within(appended.., 0);
            hist[3 * d - appended..].copy_from_slice(&rows[end - appended..end]);
        }
        if keep < n {
            for s in self.shards.iter_mut() {
                s.rollback(keep)?;
            }
            self.pos = self.spec.pos + keep;
        }
        self.spec.verifying = false;
        self.commit_context(keep)
    }

    /// GPU milliseconds per kernel of the first local rank since the last call (TRASHCAN_INKLING_PROFILE=1).
    pub fn take_profile(&mut self) -> Vec<(String, u64, f64)> {
        let mut v: Vec<_> = std::mem::take(&mut self.shards[0].profile).into_iter().map(|(k, (n, ms))| (k, n, ms)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    /// The first local GPU (the vision tower runs there).
    pub fn first_gpu(&self) -> std::rc::Rc<crate::vulkan::Device> {
        self.shards[0].gpu.clone()
    }

    pub fn device_mib(&self) -> Vec<f64> {
        self.shards.iter().map(Shard::device_mib).collect()
    }

    fn tell(&mut self, v: Value) -> Result<()> {
        if let (true, Some(m)) = (self.driver, self.mesh.as_mut()) {
            m.tell_children(&message(v)?)?;
        }
        Ok(())
    }

    /// Start a new sequence at position 0.
    pub fn reset(&mut self) -> Result<()> {
        self.tell(json!({"reset": true}))?;
        self.reset_local()
    }

    fn reset_local(&mut self) -> Result<()> {
        self.pos = 0;
        self.spec = Spec::default();
        for h in self.attn_hist.iter_mut().chain(self.ffn_hist.iter_mut()) {
            h.fill(0.);
        }
        self.shards.iter_mut().try_for_each(Shard::reset)
    }

    /// Forward `tokens` at the current position (which then advances), in batches of at most `max_batch`; the
    /// candidates of the scored rows of the last batch.
    pub fn forward(&mut self, tokens: &[u32], score: Score, reference: Option<&mut Reference>) -> Result<Vec<Candidates>> {
        self.forward_rows(tokens, &[], score, reference)
    }

    /// `forward` where every `IMAGE_ROW` entry of `tokens` takes the next row of `rows` (image embeddings, used as
    /// they are) instead of a token embedding.
    pub fn forward_rows(&mut self, tokens: &[u32], rows: &[f32], score: Score, mut reference: Option<&mut Reference>) -> Result<Vec<Candidates>> {
        let d = self.c.d;
        ensure!(tokens.iter().filter(|&&t| t == IMAGE_ROW).count() * d == rows.len(), "image rows do not match their slots");
        let mut out = Vec::new();
        let mut used = 0;
        for (i, chunk) in tokens.chunks(self.max_batch).enumerate() {
            let last = (i + 1) * self.max_batch >= tokens.len();
            let s = if last { score } else { Score::None };
            let force = reference.as_ref().map(|r| r.dir.display().to_string());
            let k = chunk.iter().filter(|&&t| t == IMAGE_ROW).count();
            let mine = &rows[used * d..(used + k) * d];
            used += k;
            // Two micro-batches overlap the host's work with the GPUs'; forcing takes the batch whole.
            let micro = if chunk.len() >= MICRO_MIN && reference.is_none() && s != Score::All { MICRO } else { 1 };
            self.tell(json!({"forward": {"tokens": chunk, "score": score_name(s), "force": force, "image_rows": k, "micro": micro}}))?;
            if k > 0 {
                if let (true, Some(m)) = (self.driver, self.mesh.as_mut()) {
                    m.tell_children(bytemuck::cast_slice(mine))?;
                }
            }
            let x = self.embed_rows(chunk, mine)?;
            out = self.run(x, chunk.len(), s, reference.as_deref_mut(), micro)?;
        }
        Ok(out)
    }

    fn embed_rows(&self, tokens: &[u32], rows: &[f32]) -> Result<Vec<f32>> {
        let d = self.c.d;
        if rows.is_empty() {
            return embed(&self.gguf, &self.c, &self.hw, tokens);
        }
        let mut x = vec![0f32; tokens.len() * d];
        let mut next = 0;
        for (o, &t) in x.chunks_exact_mut(d).zip(tokens) {
            if t == IMAGE_ROW {
                o.copy_from_slice(&rows[next * d..(next + 1) * d]);
                next += 1;
            } else {
                o.copy_from_slice(&embed(&self.gguf, &self.c, &self.hw, &[t])?);
            }
        }
        Ok(x)
    }

    /// One batch of `n` embedded rows `x` at `self.pos`, as `micro` micro-batches.
    fn run(&mut self, x: Vec<f32>, n: usize, score: Score, mut reference: Option<&mut Reference>, micro: usize) -> Result<Vec<Candidates>> {
        let start = Instant::now();
        let (c, d) = (self.c.clone(), self.c.d);
        ensure!(self.pos + n <= self.context, "context of {} tokens exceeded", self.context);
        let gated = n <= GATED_MAX;
        ensure!(
            micro == 1 || (micro <= MICRO && !gated && n >= micro && reference.is_none() && score != Score::All),
            "a forward of {n} tokens cannot run as {micro} micro-batches"
        );
        let mut flows = Vec::with_capacity(micro);
        let mut row0 = 0;
        for mb in 0..micro {
            let rows = (n - row0).div_ceil(micro - mb);
            let mut plan = Vec::with_capacity(2 * self.sched.len() + 1);
            for a in &self.sched {
                let controlled = a.pass > 0;
                let kinds = if controlled || !c.swa(a.phys) { 3 } else { 1 };
                plan.push(Seg::Attn { slot: a.slot, phys: a.phys, kinds, controlled, wide: self.pos + n > WIDE_FROM });
                plan.push(Seg::Moe { phys: a.phys, controlled });
            }
            // The scored rows are the last micro-batch's.
            if mb + 1 == micro {
                match score {
                    Score::Last => plan.push(Seg::Head { rows: 1 }),
                    Score::All => plan.push(Seg::Head { rows }),
                    Score::None => {}
                }
            }
            for s in &self.shards {
                s.set_params(mb, self.pos + row0, rows)?;
            }
            flows.push(Flow {
                mb,
                row0,
                n: rows,
                plan,
                cur: x[row0 * d..(row0 + rows) * d].to_vec(),
                normed: Vec::new(),
                prelude: (Vec::new(), 0),
                bank: [Vec::new(), Vec::new()],
                tickets: vec![0; self.shards.len()],
            });
            row0 += rows;
        }
        if self.drafter.is_some() {
            self.spec.features.clear();
            self.spec.features.resize(n * 4 * d, 0.);
            self.spec.pos = self.pos;
            self.spec.n = n;
            let slots = self.attn_hist.len();
            self.spec.attn_y.resize_with(slots, Vec::new);
            self.spec.ffn_y.resize_with(slots, Vec::new);
        }
        let result = self.drive(&mut flows, gated, reference.as_deref_mut()).and_then(|()| match score {
            Score::None => Ok(Vec::new()),
            Score::Last => self.candidates(1),
            Score::All => self.candidates(n),
        });
        if result.is_err() || gated {
            for s in self.shards.iter_mut() {
                s.retire()?;
            }
        }
        let result = result?;
        if self.drafter.is_some() {
            // The final hidden state of every row: the drafter's fourth feature.
            for f in &flows {
                let mut cur_n = vec![0f32; f.cur.len()];
                for (a, b) in f.cur.chunks_exact(d).zip(cur_n.chunks_exact_mut(d)) {
                    super::rms_into(a, None, 1e-6, b);
                }
                for t in 0..f.n {
                    let states: Vec<&[f32]> = f.normed.iter().map(|b| &b[t * d..(t + 1) * d]).chain([&cur_n[t * d..(t + 1) * d]]).collect();
                    let mut m = vec![0.; d];
                    super::mix(&states, &self.hw.out_query, &mut m);
                    let at = ((f.row0 + t) * 4 + 3) * d;
                    super::rms_into(&m, Some(&self.hw.out_norm), c.eps, &mut self.spec.features[at..at + d]);
                }
            }
        }
        if self.keep_logits {
            self.keep_hidden(&flows, n);
        }
        self.pos += n;
        if self.drafter.is_some() && !self.spec.verifying {
            self.commit_context(n)?;
        }
        self.seconds.forwards += 1;
        self.seconds.tokens += n as u64;
        self.seconds.host += start.elapsed().as_secs_f64();
        Ok(result)
    }

    /// Run the flows' segments. Segment `j` of every flow runs in flow order (the K/V cache and the convolution
    /// histories of a slot take the rows in order); while the GPUs run one flow's segment, the host all-reduces the
    /// other's output and prepares its next segment, which then waits in the queue.
    fn drive(&mut self, flows: &mut [Flow], gated: bool, mut reference: Option<&mut Reference>) -> Result<()> {
        let d = self.c.d;
        for f in flows.iter_mut() {
            self.advance(f, None, Some(0), false);
            self.launch(f, 0, gated)?;
        }
        let segments = flows.iter().map(|f| f.plan.len()).max().unwrap_or(0);
        for j in 0..segments {
            for f in flows.iter_mut() {
                if j >= f.plan.len() {
                    continue;
                }
                self.wait(f, j, gated)?;
                let head = matches!(f.plan[j], Seg::Head { .. });
                let addend = !head && self.allreduce(f.mb, f.n * d)?;
                let next = (j + 1 < f.plan.len()).then_some(j + 1);
                match reference.as_deref_mut() {
                    // Teacher forcing: the residual after an application is compared and replaced before the next
                    // segment's inputs are made from it.
                    Some(r) => {
                        self.advance(f, Some(j), None, addend);
                        if let Seg::Moe { .. } = f.plan[j] {
                            r.check(&format!("l_out-{}", self.sched[j / 2].slot), &mut f.cur, true)?;
                        }
                        self.advance(f, None, next, false);
                    }
                    None => self.advance(f, Some(j), next, addend),
                }
                // The drafter reads the residual entering some applications.
                if self.drafter.is_some() {
                    if let Some(nj) = next {
                        if nj % 2 == 0 {
                            if let Some(k) = FEATURE_VISITS.iter().position(|&v| v == nj / 2) {
                                for t in 0..f.n {
                                    let at = ((f.row0 + t) * 4 + k) * d;
                                    self.spec.features[at..at + d].copy_from_slice(&f.cur[t * d..(t + 1) * d]);
                                }
                            }
                        }
                    }
                }
                if let Some(next) = next {
                    self.launch(f, next, gated)?;
                }
            }
        }
        Ok(())
    }

    /// Start segment `j` of `f` on every local rank (its inputs are in their `hin`): a submission queued behind the
    /// ones in flight, or the next segment of the gated graph.
    fn launch(&mut self, f: &mut Flow, j: usize, gated: bool) -> Result<()> {
        let g = Instant::now();
        let seg = f.plan[j];
        for (s, ticket) in self.shards.iter_mut().zip(f.tickets.iter_mut()) {
            if !gated {
                *ticket = s.submit(seg, f.n, f.mb)?;
            } else if j == 0 {
                s.start_gated(&f.plan, f.n)?;
            } else {
                s.release()?;
            }
        }
        self.gpu_time(seg, g.elapsed().as_secs_f64());
        Ok(())
    }

    /// Wait for segment `j` of `f` on every local rank.
    fn wait(&mut self, f: &Flow, j: usize, gated: bool) -> Result<()> {
        let g = Instant::now();
        for (s, &ticket) in self.shards.iter_mut().zip(&f.tickets) {
            if gated {
                s.wait_gated()?;
            } else {
                s.wait(ticket, f.mb)?;
            }
        }
        self.gpu_time(f.plan[j], g.elapsed().as_secs_f64());
        Ok(())
    }

    fn gpu_time(&mut self, seg: Seg, dt: f64) {
        self.seconds.gpu += dt;
        match seg {
            Seg::Attn { .. } => self.seconds.gpu_attn += dt,
            Seg::Moe { .. } => self.seconds.gpu_moe += dt,
            Seg::Head { .. } => {}
        }
    }

    /// One pass over the rows of `f` (`Pass`): take the all-reduced output of its segment `done` (in `self.sum`, to
    /// which `self.remote` is still to be added with `addend`) into the residual, then write the inputs of its
    /// segment `next` into the local ranks' `hin`.
    fn advance(&mut self, f: &mut Flow, done: Option<usize>, next: Option<usize>, addend: bool) {
        let h0 = Instant::now();
        let (c, d, n) = (&self.c, self.c.d, f.n);
        let len = n * d;
        // The segments with their layer applications (the head has none).
        let done = done.map(|j| (f.plan[j], self.sched.get(j / 2).copied()));
        let next = next.map(|j| (f.plan[j], self.sched.get(j / 2).copied()));
        // The output's convolution: its weights and the slot's history.
        let (taps, hist): (&[f32], Option<&mut Vec<f32>>) = match done {
            Some((Seg::Attn { slot, phys, .. }, _)) => (&self.hw.layers[phys].attn_post_conv, Some(&mut self.attn_hist[slot])),
            Some((Seg::Moe { phys, .. }, Some(app))) => (&self.hw.layers[phys].ffn_post_conv, Some(&mut self.ffn_hist[app.slot])),
            _ => (&[], None),
        };
        // An attention output at the start of a block: the residual becomes a block state.
        let push = matches!(done, Some((Seg::Attn { phys, .. }, _)) if phys % c.block == 0);
        if push {
            f.normed.push(vec![0.; len]);
        }
        // The loop core's first segment: the first pass keeps the prelude's output; later ones take the previous
        // pass's bank (its block after the first core block, and its output) and restart from the prelude.
        let pass_start = match next {
            Some((Seg::Attn { phys, .. }, Some(app))) if phys == c.core_start => Some(app.pass > 0),
            _ => None,
        };
        match pass_start {
            Some(false) => f.prelude = (vec![0.; len], f.normed.len()),
            Some(true) => {
                f.bank[0] = std::mem::take(&mut f.normed[f.prelude.1 + 1]);
                f.bank[1].resize(len, 0.);
                f.normed.truncate(f.prelude.1);
            }
            None => {}
        }
        let layer = |seg: Seg| match seg {
            Seg::Attn { phys, .. } | Seg::Moe { phys, .. } => Some(&self.hw.layers[phys]),
            Seg::Head { .. } => None,
        };
        let pass = Pass {
            d,
            n,
            eps: c.eps,
            done: done.map(|x| x.0).filter(|_| hist.is_some()),
            sum: &self.sum[..len],
            remote: addend.then_some(&self.remote[..len]),
            hist: hist.as_deref().map_or(&[], |h| h.as_slice()),
            taps,
            push,
            pass_start,
            next: next.map(|x| x.0),
            swa: matches!(next, Some((Seg::Attn { phys, .. }, _)) if c.swa(phys)),
            layer: next.and_then(|x| layer(x.0)),
            out_query: &self.hw.out_query,
            out_norm: &self.hw.out_norm,
            cur: SendPtr(f.cur.as_mut_ptr()),
            normed: f.normed.iter_mut().map(|v| SendPtr(v.as_mut_ptr())).collect(),
            prelude: SendPtr(f.prelude.0.as_mut_ptr()),
            bank: [SendPtr(f.bank[0].as_mut_ptr()), SendPtr(f.bank[1].as_mut_ptr())],
            hin: self.shards.iter().map(|s| SendPtr(s.inputs_mut(f.mb, n, 6 * len + 6 * n).0.as_mut_ptr())).collect(),
            fence: self.shards.iter().any(|s| s.inputs_mut(f.mb, n, 0).1),
        };
        // SAFETY: every task takes its own rows of the flow's state and of `hin`; `sum`, `remote` and the history are
        // only read.
        if (2..=GATED_MAX).contains(&n) || n >= POOL_MIN {
            // Verification batches use one task per row, on workers that poll between the segments' passes. Prefill
            // gives each worker several rows.
            let tasks = if n <= GATED_MAX { n } else { HOST_THREADS.min(n / 4) };
            let per = n.div_ceil(tasks);
            self.pool.run_with(tasks, n <= GATED_MAX && row_spin(), |i| unsafe { pass.rows((i * per).min(n), ((i + 1) * per).min(n)) });
        } else {
            unsafe { pass.rows(0, n) };
        }
        // The history becomes the last three outputs before the convolution.
        if hist.is_some() {
            if self.spec.verifying {
                // A verification batch keeps every row's output for `rollback`.
                let ys = match done {
                    Some((Seg::Attn { slot, .. }, _)) => &mut self.spec.attn_y[slot],
                    Some((Seg::Moe { .. }, Some(app))) => &mut self.spec.ffn_y[app.slot],
                    _ => unreachable!(),
                };
                ys.resize(n * d, 0.);
                for (t, row) in ys.chunks_exact_mut(d).enumerate() {
                    pass.load_y(t as isize, row);
                }
            } else {
                drop(pass);
                // The history's last rows move up, the batch's last rows follow them.
                let hist = hist.unwrap();
                let kept = 3usize.saturating_sub(n);
                if kept > 0 {
                    hist.copy_within((3 - kept) * d.., 0);
                }
                let remote = addend.then_some(&self.remote[..len]);
                for (k, row) in hist.chunks_exact_mut(d).enumerate().skip(kept) {
                    y_row(&self.sum[..len], remote, n + k - 3, d, row);
                }
            }
        }
        self.seconds.host_in += h0.elapsed().as_secs_f64();
    }

    /// Add the local ranks' partials of micro-batch `mb` (rank order) into `self.sum` and all-reduce with the other
    /// machines. Returns whether `self.remote` holds an addend still to be added to `self.sum`.
    fn allreduce(&mut self, mb: usize, len: usize) -> Result<bool> {
        let ts = Instant::now();
        let sum = &mut self.sum[..len];
        let parts: Vec<&[f32]> = self.shards.iter().map(|s| s.part(mb, len)).collect();
        let add = |at: usize, out: &mut [f32]| {
            if let [a, b] = parts[..] {
                // Two local ranks: one pass over both partials.
                for ((o, x), y) in out.iter_mut().zip(&a[at..]).zip(&b[at..]) {
                    *o = x + y;
                }
                return;
            }
            out.copy_from_slice(&parts[0][at..at + out.len()]);
            for p in &parts[1..] {
                for (a, b) in out.iter_mut().zip(&p[at..]) {
                    *a += b;
                }
            }
        };
        if len < POOL_MIN * self.c.d {
            add(0, sum);
        } else {
            let per = len.div_ceil(SUM_THREADS).next_multiple_of(16);
            let out = SendPtr(sum.as_mut_ptr());
            // SAFETY: the tasks write disjoint ranges of `sum`.
            self.pool.run(len.div_ceil(per), |i| add(i * per, unsafe { std::slice::from_raw_parts_mut(out.at(i * per), per.min(len - i * per)) }));
        }
        self.seconds.host_sum += ts.elapsed().as_secs_f64();
        let mut addend = false;
        if let Some(m) = self.mesh.as_mut() {
            let e = Instant::now();
            addend = m.all_reduce(&mut self.sum[..len], &mut self.remote[..len], Duration::from_secs(5), 1, self.c.d)?;
            self.seconds.exchange += e.elapsed().as_secs_f64();
        }
        Ok(addend)
    }

    /// Keep every row's head input after each forward (on every machine), for `logits_all`.
    pub fn keep_logits(&mut self, on: bool) -> Result<()> {
        self.tell(json!({"keep_logits": on}))?;
        self.keep_logits = on;
        self.hidden.clear();
        Ok(())
    }

    /// The head input of every row of the last forward: the output mix of the block states and the residual, normed
    /// with `output_norm` (what the head segment would receive).
    fn keep_hidden(&mut self, flows: &[Flow], n: usize) {
        let d = self.c.d;
        self.hidden.clear();
        self.hidden.resize(n * d, 0.);
        let (mut cur_n, mut m) = (vec![0f32; d], vec![0f32; d]);
        for f in flows {
            for t in 0..f.n {
                rms_into(&f.cur[t * d..(t + 1) * d], None, 1e-6, &mut cur_n);
                let states: Vec<&[f32]> = f.normed.iter().map(|b| &b[t * d..(t + 1) * d]).chain([&cur_n[..]]).collect();
                super::mix(&states, &self.hw.out_query, &mut m);
                let at = (f.row0 + t) * d;
                rms_into(&m, Some(&self.hw.out_norm), self.c.eps, &mut self.hidden[at..at + d]);
            }
        }
    }

    /// Full-vocabulary softcapped logits of every row of the last forward (`keep_logits` must be on): every rank runs
    /// the head over the kept rows, eight at a time, and the machines gather their slices. On the driver the result
    /// is [rank][row][rows of the rank's vocabulary slice] over all ranks (vocabulary id = rank * slice + index,
    /// including the padded rows); the other machines return an empty vector. Outside the decode and prefill paths.
    pub fn logits_all(&mut self) -> Result<Vec<f32>> {
        self.tell(json!({"logits_all": true}))?;
        self.logits_local()
    }

    fn logits_local(&mut self) -> Result<Vec<f32>> {
        const ROWS: usize = 8;
        let d = self.c.d;
        ensure!(self.keep_logits && !self.hidden.is_empty(), "no kept rows: keep_logits, then forward");
        let n = self.hidden.len() / d;
        let vn = self.shards[0].vocab_rows().1;
        let local = self.shards.len();
        let first = self.shards[0].vocab_rows().0 / vn;
        let machines = self.mesh.as_ref().map_or(1, |m| m.machines);
        let me = self.mesh.as_ref().map_or(0, |m| m.machine);
        ensure!(self.ranks == machines * local && first == me * local, "full logits need the same number of GPUs on every machine");
        let per_rank = n * vn;
        let mut all = vec![0f32; self.ranks * per_rank];
        let cap = self.c.softcap;
        for s in self.shards.iter_mut() {
            s.retire()?;
        }
        for r0 in (0..n).step_by(ROWS) {
            let rows = ROWS.min(n - r0);
            let mut tickets = Vec::with_capacity(local);
            for s in self.shards.iter_mut() {
                s.hin(0, rows * d).copy_from_slice(&self.hidden[r0 * d..(r0 + rows) * d]);
                tickets.push(s.submit(Seg::Head { rows }, rows, 0)?);
            }
            for (i, (s, t)) in self.shards.iter_mut().zip(tickets).enumerate() {
                s.wait(t, 0)?;
                let out = &mut all[(first + i) * per_rank + r0 * vn..][..rows * vn];
                out.copy_from_slice(s.logits(rows));
                if cap > 0. {
                    out.iter_mut().for_each(|x| *x = cap * (*x / cap).tanh());
                }
            }
        }
        // Recursive doubling: in round k every machine holds the slices of its block of 2^k machines and swaps them
        // with its partner's block.
        if let Some(m) = self.mesh.as_mut() {
            let block = local * per_rank;
            for round in 0..m.rounds() {
                let partner = m.partner(round);
                let size = (1usize << round) * block;
                let (mine, theirs) = ((me >> round << round) * block, (partner >> round << round) * block);
                let (lo, hi) = all.split_at_mut(mine.max(theirs));
                let (out, inp) = if mine < theirs { (&lo[mine..mine + size], &mut hi[..size]) } else { (&hi[..size], &mut lo[theirs..theirs + size]) };
                m.lanes(partner)?.exchange(bytemuck::cast_slice(out), bytemuck::cast_slice_mut(inp))?;
            }
        }
        Ok(if self.driver { all } else { Vec::new() })
    }

    /// The best TOP_K candidates of each of `rows` scored rows: softcapped logits of the real vocabulary from every
    /// local rank, merged, then merged with the other machines' in the all-reduce rounds.
    fn candidates(&mut self, rows: usize) -> Result<Vec<Candidates>> {
        let (cap, vocab) = (self.c.softcap, self.c.vocab);
        // Every local rank's best of every row, several rows at a time on the pool.
        let ranks = self.shards.len();
        let tops: Vec<Mutex<Candidates>> = (0..ranks * rows).map(|_| Mutex::new(Vec::new())).collect();
        let logits: Vec<_> = self.shards.iter().map(|s| (s.vocab_rows(), s.logits(rows))).collect();
        let job = |i: usize| {
            let (((v0, vn), logits), r) = (logits[i / rows], i % rows);
            let start = r * vn;
            let row = &logits[start..start + vn.min(vocab.saturating_sub(v0))];
            let mut top = top_k(row, v0);
            // The softcap keeps the order, so only the selected logits take it.
            if cap > 0. {
                top.iter_mut().for_each(|e| e.0 = cap * (e.0 / cap).tanh());
            }
            *tops[i].lock().unwrap() = top;
        };
        if rows == 1 {
            (0..ranks).for_each(job);
        } else {
            self.pool.run(ranks * rows, job);
        }
        let mut cols: Vec<Candidates> = vec![Vec::new(); rows];
        for (i, top) in tops.into_iter().enumerate() {
            cols[i % rows].extend(top.into_inner().unwrap());
        }
        cols.iter_mut().for_each(best);
        if let Some(m) = self.mesh.as_mut() {
            for round in 0..m.rounds() {
                for c in cols.iter_mut() {
                    c.resize(TOP_K, (f32::NEG_INFINITY, u32::MAX));
                }
                let mine = encode(&cols);
                let mut theirs = vec![0u8; mine.len()];
                m.lanes(m.partner(round))?.exchange(&mine, &mut theirs)?;
                for (c, t) in cols.iter_mut().zip(decode(&theirs)) {
                    c.extend(t);
                    best(c);
                }
            }
        }
        for c in cols.iter_mut() {
            c.retain(|e| e.1 != u32::MAX);
        }
        Ok(cols)
    }

    /// The other machines' side: carry out the driver's operations until it says goodbye.
    fn follow(&mut self) -> Result<()> {
        loop {
            let mesh = self.mesh.as_mut().context("no driver")?;
            let bytes = mesh.hear()?;
            mesh.tell_children(&bytes)?;
            let m: Value = serde_json::from_slice(&bytes)?;
            if let Some(f) = m.get("forward").filter(|f| f["verify"].as_bool().unwrap_or(false)) {
                let tokens: Vec<u32> = serde_json::from_value(f["tokens"].clone())?;
                self.verify_local(&tokens)?;
            } else if let Some(v) = m.get("draft_load") {
                self.load_draft(Path::new(v.as_str().context("drafter path")?))?;
            } else if let Some(v) = m.get("draft") {
                let token = v["token"].as_u64().context("draft token")? as u32;
                self.draft_local(token, v["n"].as_u64().context("draft count")? as usize)?;
            } else if let Some(v) = m.get("rollback") {
                self.rollback_local(v.as_u64().context("rollback rows")? as usize)?;
            } else if let Some(f) = m.get("forward") {
                let tokens: Vec<u32> = serde_json::from_value(f["tokens"].clone())?;
                let score = parse_score(f["score"].as_str().unwrap_or("last"))?;
                let mut reference = f["force"].as_str().map(|p| Reference::new(Path::new(p)));
                let rows: Vec<f32> = if f["image_rows"].as_u64().unwrap_or(0) > 0 {
                    let mesh = self.mesh.as_mut().context("no driver")?;
                    let bytes = mesh.hear()?;
                    mesh.tell_children(&bytes)?;
                    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
                } else {
                    Vec::new()
                };
                let x = self.embed_rows(&tokens, &rows)?;
                let micro = f["micro"].as_u64().unwrap_or(1) as usize;
                self.run(x, tokens.len(), score, reference.as_mut(), micro)?;
            } else if let Some(v) = m.get("keep_logits") {
                self.keep_logits = v.as_bool().unwrap_or(false);
                self.hidden.clear();
            } else if m.get("logits_all").is_some() {
                self.logits_local()?;
            } else if m.get("reset").is_some() {
                self.reset_local()?;
            } else if m.get("bye").is_some() {
                return Ok(());
            } else {
                bail!("unknown request {m}");
            }
        }
    }
}

impl Drop for Group {
    fn drop(&mut self) {
        // TRASHCAN_INKLING_PHASES=1: this machine's totals per forward, to compare the machines of a group.
        if std::env::var_os("TRASHCAN_INKLING_PHASES").is_some() {
            let (p, f) = (self.seconds, self.seconds.forwards.max(1) as f64);
            let st = self.mesh.as_ref().map(|m| (m.stats.sent, m.stats.first, m.stats.wire)).unwrap_or_default();
            eprintln!(
                "inkling phases: forwards {} ms per forward: gpu {:.3} exchange {:.3} (sent {:.3} first {:.3} wire {:.3}) host_math {:.3} host_sum {:.3} host {:.3}",
                p.forwards, p.gpu / f * 1e3, p.exchange / f * 1e3, st.0 / f * 1e3, st.1 / f * 1e3, st.2 / f * 1e3, p.host_in / f * 1e3, p.host_sum / f * 1e3, p.host / f * 1e3
            );
        }
        let _ = self.tell(json!({"bye": true}));
    }
}

fn score_name(s: Score) -> &'static str {
    match s {
        Score::Last => "last",
        Score::All => "all",
        Score::None => "none",
    }
}

fn parse_score(s: &str) -> Result<Score> {
    Ok(match s {
        "last" => Score::Last,
        "all" => Score::All,
        "none" => Score::None,
        other => bail!("unknown score {other}"),
    })
}

/// Host threads for batches of rows (prefill), and for adding the local ranks' partials.
const HOST_THREADS: usize = 12;
const SUM_THREADS: usize = 4;
/// Row threshold for prefill tasks and for parallel addition of rank partials.
const POOL_MIN: usize = 16;

/// A pointer several threads use for disjoint rows.
#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    fn at(self, i: usize) -> *mut f32 {
        // SAFETY: the callers stay inside the allocation.
        unsafe { self.0.add(i) }
    }
}

/// One pass over the rows of a flow (`Group::advance`), every row on its own, so that a row stays in the first-level
/// cache from the all-reduced output to the next segment's inputs: the output's convolution, the residual update,
/// the loop's pass boundary, then the residual mixes (and the loop messages) of the next segment, written into
/// `hin` of every local rank. The arithmetic is the reference's, in a fixed order per element, so every machine
/// keeps the same bits.
struct Pass<'a> {
    d: usize,
    n: usize,
    eps: f32,
    /// The segment whose all-reduced output is `sum` (plus `remote`, when it is still to be added).
    done: Option<Seg>,
    sum: &'a [f32],
    remote: Option<&'a [f32]>,
    /// The three outputs before the batch (oldest first) and the convolution's weights by tap.
    hist: &'a [f32],
    taps: &'a [f32],
    /// The output starts a block: the residual becomes the newest block state (the last of `normed`).
    push: bool,
    /// The next segment starts a pass of the loop core: false keeps the residual as the prelude's output; true
    /// makes the residual the bank's output and restarts from the prelude.
    pass_start: Option<bool>,
    /// The segment to prepare, its physical layer's weights, and whether that layer has one residual mix.
    next: Option<Seg>,
    layer: Option<&'a HostLayer>,
    swa: bool,
    out_query: &'a [f32],
    out_norm: &'a [f32],
    cur: SendPtr,
    /// The RMS-normed block states, the prelude's output and the bank (RMS-normed block and output).
    normed: Vec<SendPtr>,
    prelude: SendPtr,
    bank: [SendPtr; 2],
    /// Every local rank's input rows.
    hin: Vec<SendPtr>,
    /// The input rows are write-combining memory (prefill inputs in VRAM): every thread ends its rows with a store
    /// fence, so nothing it wrote is still in a write-combining buffer when the segment is submitted.
    fence: bool,
}

impl Pass<'_> {
    /// Output row `t` before the convolution (a negative `t` reads the history).
    #[inline(always)]
    fn load_y(&self, t: isize, out: &mut [f32]) {
        let d = self.d;
        if t < 0 {
            out.copy_from_slice(&self.hist[(3 + t) as usize * d..][..d]);
            return;
        }
        y_row(self.sum, self.remote, t as usize, d, out)
    }

    /// Write `row` as input row `at` of every local rank.
    #[inline(always)]
    unsafe fn write(&self, at: usize, row: &[f32]) {
        self.write_floats(at * self.d, &row[..self.d])
    }

    /// Write `values` from float `at` of every local rank's inputs.
    #[inline(always)]
    unsafe fn write_floats(&self, at: usize, values: &[f32]) {
        for h in &self.hin {
            std::ptr::copy_nonoverlapping(values.as_ptr(), h.at(at), values.len());
        }
    }

    unsafe fn rows(&self, t0: usize, t1: usize) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx") {
            return self.rows_avx(t0, t1);
        }
        self.rows_any(t0, t1)
    }

    /// `rows_any` compiled for AVX (the Mac Pros' Xeons; the build targets plain x86-64).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn rows_avx(&self, t0: usize, t1: usize) {
        self.rows_any(t0, t1)
    }

    /// Rows `t0..t1`. SAFETY: no other thread touches these rows of the flow's state or of `hin`.
    #[inline(always)]
    unsafe fn rows_any(&self, t0: usize, t1: usize) {
        let (d, n) = (self.d, self.n);
        if t0 >= t1 {
            return;
        }
        let row = |p: SendPtr, t: usize| std::slice::from_raw_parts_mut(p.at(t * d), d);
        // This thread's scratch rows (kept between passes): a ring of four outputs before the convolution (row t of
        // a batch at `(t - t0 + 3) % 4`), the normed residual, and two rows for the head's input.
        let base = SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            if s.len() < 7 * d {
                s.resize(7 * d, 0.);
            }
            s.as_mut_ptr()
        });
        let slot = |k: usize| std::slice::from_raw_parts_mut(base.add(k * d), d);
        let (cur_n, u, out) = (slot(4), slot(5), slot(6));
        // A row's three outputs before it come from the history, or from the batch: those of rows before `t0` are
        // loaded here, the others kept as their rows pass.
        let batch = n > 1;
        if self.done.is_some() {
            for k in 0..3 {
                let t = t0 as isize - 3 + k as isize;
                if t >= 0 {
                    self.load_y(t, slot(k));
                }
            }
        }
        for t in t0..t1 {
            let cur = row(self.cur, t);
            if let Some(seg) = self.done {
                let before = |k: usize| -> &[f32] {
                    let p = t as isize - 3 + k as isize;
                    if p < 0 {
                        &self.hist[(3 + p) as usize * d..][..d]
                    } else {
                        slot((p - t0 as isize + 3) as usize % 4)
                    }
                };
                let prev = [before(0), before(1), before(2)];
                let y = &self.sum[t * d..][..d];
                let addend = self.remote.map(|r| &r[t * d..][..d]);
                let keep = batch.then(|| slot((t - t0 + 3) % 4));
                match seg {
                    Seg::Attn { .. } if self.push => {
                        let inv = rms_inv(cur, 1e-6);
                        conv_blocks(y, addend, prev, self.taps, cur, Some((row(*self.normed.last().unwrap(), t), inv)), keep);
                    }
                    Seg::Attn { .. } | Seg::Moe { .. } => conv_blocks(y, addend, prev, self.taps, cur, None, keep),
                    Seg::Head { .. } => {}
                }
            }
            match self.pass_start {
                Some(false) => row(self.prelude, t).copy_from_slice(cur),
                Some(true) => {
                    rms_into(cur, None, 1e-6, row(self.bank[1], t));
                    cur.copy_from_slice(row(self.prelude, t));
                }
                None => {}
            }
            let Some(next) = self.next else { continue };
            if matches!(next, Seg::Head { rows } if t + rows < n) {
                continue;
            }
            // The mixes take the block states and the residual, all RMS-normed.
            rms_into(cur, None, 1e-6, cur_n);
            let mut states: [&[f32]; MIX_STATES] = [&[]; MIX_STATES];
            let count = self.normed.len() + 1;
            for (s, p) in states.iter_mut().zip(&self.normed) {
                *s = row(*p, t);
            }
            states[count - 1] = cur_n;
            let states = &states[..count];
            // The loop messages of a controlled segment mix the bank's two states.
            let bank = || -> [&[f32]; 2] { [row(self.bank[0], t), row(self.bank[1], t)] };
            match next {
                Seg::Attn { controlled, .. } => {
                    let l = self.layer.unwrap();
                    // One mix on sliding-window layers (their q, k and v inputs differ only by the messages), three
                    // on global ones.
                    let mixes = if self.swa { 1 } else { 3 };
                    let mut iu = [0f32; 3];
                    match &l.attn_query {
                        Some(q) => {
                            let qs: [&[f32]; 3] = [&q[0], &q[1.min(mixes - 1)], &q[2.min(mixes - 1)]];
                            iu = mix_blocks(states, &qs[..mixes], |k, at, v| self.write_floats((k * n + t) * d + at, v));
                        }
                        None => self.write(t, cur),
                    }
                    if controlled {
                        // The q, k and v messages, then every controller's inverse RMS of its input and message.
                        let q = l.msg.as_ref().unwrap();
                        let im = mix_blocks(&bank(), &[&q[0], &q[1], &q[2]], |k, at, v| self.write_floats(((mixes + k) * n + t) * d + at, v));
                        for k in 0..3 {
                            self.write_floats((mixes + 3) * n * d + (k * n + t) * 2, &[iu[k.min(mixes - 1)], im[k]]);
                        }
                    }
                }
                Seg::Moe { controlled, .. } => {
                    let l = self.layer.unwrap();
                    let iu = mix_blocks(states, &[&l.ffn_query], |_, at, v| self.write_floats(t * d + at, v));
                    if controlled {
                        let im = mix_blocks(&bank(), &[&l.msg.as_ref().unwrap()[3]], |_, at, v| self.write_floats((n + t) * d + at, v));
                        self.write_floats(2 * n * d + t * 2, &[iu[0], im[0]]);
                    }
                }
                Seg::Head { rows } => {
                    mix_blocks(states, &[self.out_query], |_, at, v| u[at..at + v.len()].copy_from_slice(v));
                    rms_into(u, Some(self.out_norm), self.eps, out);
                    self.write(t + rows - n, out);
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        if self.fence {
            std::arch::x86_64::_mm_sfence();
        }
    }
}

thread_local! {
    /// A thread's scratch rows for `Pass::rows`.
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Row `t` of an all-reduced output: `sum`, plus `remote` when that addend is still to be added.
#[inline(always)]
fn y_row(sum: &[f32], remote: Option<&[f32]>, t: usize, d: usize, out: &mut [f32]) {
    let s = &sum[t * d..][..d];
    match remote {
        Some(r) => out.iter_mut().zip(s).zip(&r[t * d..][..d]).for_each(|((o, a), b)| *o = a + b),
        None => out.copy_from_slice(s),
    }
}

/// Worker threads for the host's passes over batches of rows. They sleep between passes, except after a pass run
/// with `spin` (verification batches, whose passes follow each other within a segment's time): then as many
/// workers as it had tasks for keep polling for the next pass for `SPIN`, because waking a sleeping thread takes
/// longer than a row of such a pass.
struct Pool {
    shared: Arc<PoolShared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

struct PoolShared {
    state: Mutex<PoolState>,
    wake: Condvar,
    done: Condvar,
    /// Counts the passes posted (polling workers watch it) and the unfinished tasks of the pass in progress.
    posted: std::sync::atomic::AtomicU64,
    left: std::sync::atomic::AtomicUsize,
}

struct PoolState {
    /// The pass in progress: its task function, the next task and the number of tasks.
    job: Option<&'static (dyn Fn(usize) + Sync)>,
    next: usize,
    tasks: usize,
    /// Tasks of the pass that have not finished.
    pending: usize,
    /// Workers may poll until then, up to `poll_want` of them; `polling` do.
    poll_until: Instant,
    poll_want: usize,
    polling: usize,
    quit: bool,
}

/// How long workers poll for the next pass after one run with `spin`.
const SPIN: Duration = Duration::from_millis(2);

/// Whether verification batches keep their workers polling (`TRASHCAN_INKLING_ROWSPIN=0`: they sleep).
fn row_spin() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TRASHCAN_INKLING_ROWSPIN").as_deref() != Ok("0"))
}

impl Pool {
    fn new(threads: usize) -> Self {
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        let state = PoolState { job: None, next: 0, tasks: 0, pending: 0, poll_until: Instant::now(), poll_want: 0, polling: 0, quit: false };
        let shared = Arc::new(PoolShared { state: Mutex::new(state), wake: Condvar::new(), done: Condvar::new(), posted: AtomicU64::new(0), left: AtomicUsize::new(0) });
        let workers = (0..threads)
            .map(|_| {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    let mut st = shared.state.lock().unwrap();
                    loop {
                        if st.quit {
                            return;
                        }
                        match st.job {
                            Some(f) if st.next < st.tasks => {
                                let i = st.next;
                                st.next += 1;
                                drop(st);
                                f(i);
                                st = shared.state.lock().unwrap();
                                st.pending -= 1;
                                shared.left.store(st.pending, Ordering::Release);
                                if st.pending == 0 {
                                    shared.done.notify_all();
                                }
                            }
                            _ if st.polling < st.poll_want && Instant::now() < st.poll_until => {
                                // Poll for the next pass without the lock.
                                st.polling += 1;
                                let (seen, until) = (shared.posted.load(Ordering::Acquire), st.poll_until);
                                drop(st);
                                let mut turns = 0u32;
                                while shared.posted.load(Ordering::Acquire) == seen {
                                    std::hint::spin_loop();
                                    turns += 1;
                                    if turns % 1024 == 0 && Instant::now() >= until {
                                        break;
                                    }
                                }
                                st = shared.state.lock().unwrap();
                                st.polling -= 1;
                            }
                            _ => st = shared.wake.wait(st).unwrap(),
                        }
                    }
                })
            })
            .collect();
        Self { shared, workers }
    }

    /// Run `f(0) .. f(tasks - 1)` on the workers and the calling thread, and return when all have finished. With
    /// `spin` the calling thread polls for the workers' tasks, and they for the next pass.
    fn run_with<F: Fn(usize) + Sync>(&self, tasks: usize, spin: bool, f: F) {
        use std::sync::atomic::Ordering;
        if tasks <= 1 {
            return (0..tasks).for_each(f);
        }
        let f: &(dyn Fn(usize) + Sync) = &f;
        // SAFETY: the workers call `f` only for tasks counted in `pending`, and this function returns only once
        // `pending` is zero and the job is cleared, so the borrow outlives every use.
        let f: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let mut st = self.shared.state.lock().unwrap();
        (st.job, st.next, st.tasks, st.pending) = (Some(f), 0, tasks, tasks);
        self.shared.left.store(tasks, Ordering::Release);
        if spin {
            (st.poll_until, st.poll_want) = (Instant::now() + SPIN, (tasks - 1).min(self.workers.len()));
        } else {
            st.poll_want = 0;
        }
        self.shared.posted.fetch_add(1, Ordering::AcqRel);
        // The polling workers have seen the pass; the sleeping ones are woken for the tasks beyond them.
        let asleep = (tasks - 1).saturating_sub(st.polling);
        if asleep >= self.workers.len() {
            self.shared.wake.notify_all();
        } else {
            (0..asleep).for_each(|_| self.shared.wake.notify_one());
        }
        loop {
            if st.next < st.tasks {
                let i = st.next;
                st.next += 1;
                drop(st);
                f(i);
                st = self.shared.state.lock().unwrap();
                st.pending -= 1;
                self.shared.left.store(st.pending, Ordering::Release);
            } else if st.pending > 0 {
                if spin {
                    drop(st);
                    while self.shared.left.load(Ordering::Acquire) > 0 {
                        std::hint::spin_loop();
                    }
                    st = self.shared.state.lock().unwrap();
                } else {
                    st = self.shared.done.wait(st).unwrap();
                }
            } else {
                break;
            }
        }
        st.job = None;
    }

    fn run<F: Fn(usize) + Sync>(&self, tasks: usize, f: F) {
        self.run_with(tasks, false, f)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().quit = true;
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// The TOP_K highest of `row` (the logits of ids `v0..`), ties to the lower id, unordered: one scan that keeps what
/// reaches the K-th best so far, instead of a selection over the whole row.
fn top_k(row: &[f32], v0: usize) -> Candidates {
    let order = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    let mut keep: Candidates = Vec::with_capacity(4 * TOP_K);
    // The lowest value under total_cmp, including negative NaNs.
    let mut floor = f32::from_bits(u32::MAX);
    for (i, &x) in row.iter().enumerate() {
        if x.total_cmp(&floor).is_ge() {
            keep.push((x, (v0 + i) as u32));
            if keep.len() == 4 * TOP_K {
                keep.select_nth_unstable_by(TOP_K - 1, order);
                keep.truncate(TOP_K);
                floor = keep[TOP_K - 1].0;
            }
        }
    }
    if keep.len() > TOP_K {
        keep.select_nth_unstable_by(TOP_K - 1, order);
        keep.truncate(TOP_K);
    }
    keep
}

fn best(c: &mut Candidates) {
    c.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    c.truncate(TOP_K);
}

#[cfg(test)]
mod candidate_tests {
    use super::{best, top_k, Candidates, TOP_K};

    #[test]
    fn threshold_selection_matches_full_sort() {
        let values = [
            f32::from_bits(u32::MAX), f32::NEG_INFINITY, -1., -0., 0., 1.,
            f32::INFINITY, f32::NAN, f32::from_bits(f32::NAN.to_bits() + 1),
        ];
        let logits: Vec<_> = (0..1024).map(|i| values[i % values.len()]).collect();
        let base = 100_512;
        for n in [0, 1, TOP_K, logits.len()] {
            let row = &logits[..n];
            let mut want: Candidates = row.iter().enumerate().map(|(i, &x)| (x, (base + i) as u32)).collect();
            let mut got = top_k(row, base);
            best(&mut want);
            best(&mut got);
            let as_bits = |c: Candidates| c.into_iter().map(|(x, id)| (x.to_bits(), id)).collect::<Vec<_>>();
            assert_eq!(as_bits(got), as_bits(want));
        }
    }
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

/// Serve the ranks of `machine` (1 of 2, or with `peers` a larger group) to drivers, one at a time.
pub fn serve(model: &Path, gpus: &[usize], listen: &str, machine: usize, peers: &[String]) -> Result<()> {
    let peers = tp::parse_peers(peers)?;
    ensure!(machine > 0, "machine 0 drives the group");
    let listener = TcpListener::bind(listen).with_context(|| format!("listening on {listen}"))?;
    eprintln!("inkling node: {} on {listen}, machine {machine}, GPUs {gpus:?}", model.display());
    loop {
        let mesh = if peers.is_empty() {
            Lanes::accept(&listener).map(|(l, _)| Mesh::new(1, vec![(0, l)]))
        } else {
            Mesh::open(machine, Some(&listener), &peers)
        };
        match mesh.and_then(|m| session(model, gpus, m)) {
            Ok(()) => eprintln!("inkling node: session ended"),
            Err(e) => eprintln!("inkling node: session ended: {e:#}"),
        }
    }
}

fn session(model: &Path, gpus: &[usize], mut mesh: Mesh) -> Result<()> {
    let me = mesh.machine;
    loop {
        let bytes = mesh.lanes(tp::parent(me))?.recv_message()?;
        let m: Value = serde_json::from_slice(&bytes)?;
        if m.get("probe").is_some() {
            mesh.tell_children(&bytes)?;
            let mut counts = BTreeMap::from([(me, gpus.len())]);
            let mut cabled = BTreeMap::from([(me, mesh.peers())]);
            for reply in mesh.children_replies(None)? {
                tp::merge_counts(&mut counts, &reply)?;
                tp::merge_cabled(&mut cabled, &reply);
            }
            mesh.tell_parent(json!({ "gpus": counts, "cabled": cabled }))?;
            continue;
        }
        let s = m.get("setup").with_context(|| format!("unexpected request {m}"))?;
        mesh.tell_children(&bytes)?;
        let num = |k: &str| s[k].as_u64().with_context(|| format!("setup {k}")).map(|v| v as usize);
        let first = s["firsts"][me].as_u64().context("setup firsts")? as usize;
        let loaded = (|| {
            mesh.full = s["full"].as_bool().unwrap_or(false);
            mesh.join(num("machines")?)?;
            mesh.wire = Wire::parse(s["wire"].as_str().unwrap_or("f32"))?;
            load_shards(model, gpus, first, num("ranks")?, num("context")?, num("batch")?)
        })();
        let (gguf, c, shards) = match loaded {
            Ok(v) => v,
            Err(e) => {
                mesh.tell_parent(json!({"error": format!("machine {me}: {e:#}")}))?;
                return Err(e);
            }
        };
        for reply in mesh.children_replies(Some(Duration::from_secs(10 * 60)))? {
            if reply.get("ok").is_none() {
                mesh.tell_parent(reply.clone())?;
                bail!("another machine failed: {}", reply["error"]);
            }
        }
        mesh.tell_parent(json!({"ok": true, "hosts": {me.to_string(): host_settings()}}))?;
        let ranks = num("ranks")?;
        let mut group = Group::assemble(gguf, c, shards, Some(mesh), num("context")?, num("batch")?, ranks)?;
        return group.follow();
    }
}

