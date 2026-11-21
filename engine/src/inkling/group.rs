//! The host side of a tensor-parallel Inkling group: the residual, the block states and the loop passes, the
//! sublayer-output convolutions and the loop controllers, and the all-reduces that join the ranks' partial outputs
//! (this machine's GPUs, then the other machines over `qwen::tp::Mesh`). Every machine runs the same host math on
//! the same totals in the same order, so the residuals stay bitwise identical; the driver (machine 0) sends each
//! operation down the mesh first.
use super::{
    embed,
    gpu::{Seg, Shard, Split, GATED_MAX},
    mix, rms_into, App, Config, HostWeights,
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
    time::{Duration, Instant},
};

pub type Candidates = Vec<(f32, u32)>;
pub const TOP_K: usize = 64;

/// What a forward scores: the last row, or every row of a batch up to 16 (verification).
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
    pub max_batch: usize,
    attn_hist: Vec<Vec<f32>>,
    ffn_hist: Vec<Vec<f32>>,
    sum: Vec<f32>,
    remote: Vec<f32>,
    pub seconds: Phases,
    /// GPUs on every machine, for reports.
    pub ranks: usize,
    /// The forward in progress: its segments, the next one's index, its tokens, and whether it is event-gated.
    plan: Vec<Seg>,
    next: usize,
    n: usize,
    gated: bool,
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
    pub fn new(model: &Path, gpus: &[usize], mut mesh: Option<Mesh>, context: usize, max_batch: usize) -> Result<Self> {
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
            // Validation needs exact sums; the compact payload is the opt-in (TRASHCAN_TP_WIRE=f16).
            m.wire = if std::env::var("TRASHCAN_TP_WIRE").as_deref() == Ok("f16") { Wire::F16 } else { Wire::F32 };
        }
        let firsts: Vec<usize> = counts.values().scan(0, |at, &n| { *at += n; Some(*at - n) }).collect();
        let ranks: usize = counts.values().sum();
        if let Some(m) = mesh.as_mut() {
            let setup = json!({"firsts": firsts, "machines": counts.len(), "ranks": ranks, "context": context, "batch": max_batch, "wire": m.wire.name(), "full": m.full});
            m.tell_children(&message(json!({ "setup": setup }))?)?;
        }
        eprintln!("inkling: {ranks} ranks over {} machine(s), GPUs per machine {:?}", counts.len(), counts.values().collect::<Vec<_>>());
        let (gguf, c, shards) = load_shards(model, gpus, 0, ranks, context, max_batch)?;
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
            plan: Vec::new(),
            next: 0,
            n: 0,
            gated: false,
            c,
        })
    }

    /// GPU milliseconds per kernel of the first local rank since the group started (TRASHCAN_INKLING_PROFILE=1).
    pub fn profile(&self) -> Vec<(String, u64, f64)> {
        let mut v: Vec<_> = self.shards[0].profile.iter().map(|(k, (n, ms))| (k.clone(), *n, *ms)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
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
        for h in self.attn_hist.iter_mut().chain(self.ffn_hist.iter_mut()) {
            h.fill(0.);
        }
        self.shards.iter_mut().try_for_each(Shard::reset)
    }

    /// Forward `tokens` at the current position (which then advances), in batches of at most `max_batch`; the
    /// candidates of the scored rows of the last batch.
    pub fn forward(&mut self, tokens: &[u32], score: Score, mut reference: Option<&mut Reference>) -> Result<Vec<Candidates>> {
        let mut out = Vec::new();
        for (i, chunk) in tokens.chunks(self.max_batch).enumerate() {
            let last = (i + 1) * self.max_batch >= tokens.len();
            let s = if last { score } else { Score::None };
            let force = reference.as_ref().map(|r| r.dir.display().to_string());
            self.tell(json!({"forward": {"tokens": chunk, "score": score_name(s), "force": force}}))?;
            let x = embed(&self.gguf, &self.c, &self.hw, chunk)?;
            out = self.run(x, chunk.len(), s, reference.as_deref_mut())?;
        }
        Ok(out)
    }

    /// One batch of `n` embedded rows `x` at `self.pos`.
    fn run(&mut self, x: Vec<f32>, n: usize, score: Score, mut reference: Option<&mut Reference>) -> Result<Vec<Candidates>> {
        let start = Instant::now();
        let c = self.c.clone();
        ensure!(self.pos + n <= self.context, "context of {} tokens exceeded", self.context);
        for s in &self.shards {
            s.set_params(self.pos, n)?;
        }
        self.plan.clear();
        for a in &self.sched {
            let controlled = a.pass > 0;
            let kinds = if controlled || !c.swa(a.phys) { 3 } else { 1 };
            self.plan.push(Seg::Attn { slot: a.slot, phys: a.phys, kinds, controlled });
            self.plan.push(Seg::Moe { phys: a.phys, controlled });
        }
        match score {
            Score::Last => self.plan.push(Seg::Head { rows: 1 }),
            Score::All => self.plan.push(Seg::Head { rows: n }),
            Score::None => {}
        }
        self.next = 0;
        self.n = n;
        self.gated = n <= GATED_MAX;
        let result = self.run_plan(x, n, score, reference.as_deref_mut());
        if result.is_err() || self.gated {
            for s in self.shards.iter_mut() {
                s.retire()?;
            }
        }
        let result = result?;
        self.pos += n;
        self.seconds.forwards += 1;
        self.seconds.tokens += n as u64;
        self.seconds.host += start.elapsed().as_secs_f64();
        Ok(result)
    }

    fn run_plan(&mut self, x: Vec<f32>, n: usize, score: Score, mut reference: Option<&mut Reference>) -> Result<Vec<Candidates>> {
        let (c, d) = (self.c.clone(), self.c.d);
        let mut cur = x;
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut normed: Vec<Vec<f32>> = Vec::new();
        let mut prelude = (Vec::new(), 0usize);
        let mut messages: Option<Vec<[Vec<f32>; 4]>> = None;
        for app in self.sched.clone() {
            if app.phys == c.core_start {
                if app.pass == 0 {
                    prelude = (cur.clone(), blocks.len());
                } else {
                    // The bank of the previous pass: its blocks after the first core block, and its output.
                    let bank: Vec<&Vec<f32>> = normed[prelude.1 + 1..].iter().collect();
                    let out_n = rows_rms(&cur, d);
                    let mut msgs = Vec::new();
                    for t in 0..n {
                        let states: Vec<&[f32]> = bank.iter().map(|b| &b[t * d..(t + 1) * d]).chain([&out_n[t * d..(t + 1) * d]]).collect();
                        let per_layer: Vec<[Vec<f32>; 4]> = (c.core_start..c.core_end)
                            .map(|p| {
                                let q = self.hw.layers[p].msg.as_ref().unwrap();
                                std::array::from_fn(|k| {
                                    let mut o = vec![0.; d];
                                    mix(&states, &q[k], &mut o);
                                    o
                                })
                            })
                            .collect();
                        msgs.push(per_layer);
                    }
                    // messages[layer][kind] holds n rows.
                    let mut by_layer = Vec::new();
                    for li in 0..c.core() {
                        by_layer.push(std::array::from_fn(|k| msgs.iter().flat_map(|m| m[li][k].iter().copied()).collect::<Vec<f32>>()));
                    }
                    messages = Some(by_layer);
                    cur = prelude.0.clone();
                    blocks.truncate(prelude.1);
                    normed.truncate(prelude.1);
                }
            }
            let msg = if app.pass > 0 { messages.as_ref().map(|m| &m[app.phys - c.core_start]) } else { None };
            self.apply(app, n, &mut cur, &mut blocks, &mut normed, msg, reference.as_deref_mut())?;
            if app.phys == c.core_end - 1 && app.pass + 1 == c.passes {
                messages = None;
            }
        }
        let result = if score == Score::None {
            Vec::new()
        } else {
            let rows: Vec<usize> = match score {
                Score::Last => vec![n - 1],
                _ => (0..n).collect(),
            };
            ensure!(rows.len() <= 16, "at most 16 scored rows");
            let cur_n = rows_rms(&cur, d);
            let mut h = vec![0f32; rows.len() * d];
            for (i, &t) in rows.iter().enumerate() {
                let states: Vec<&[f32]> = normed.iter().map(|b| &b[t * d..(t + 1) * d]).chain([&cur_n[t * d..(t + 1) * d]]).collect();
                let mut m = vec![0.; d];
                mix(&states, &self.hw.out_query, &mut m);
                rms_into(&m, Some(&self.hw.out_norm), c.eps, &mut h[i * d..(i + 1) * d]);
            }
            if let Some(r) = reference.as_deref_mut() {
                if rows.len() == 1 && r.dir.join("result_norm.f32").exists() {
                    r.check("result_norm", &mut h, false)?;
                }
            }
            for s in self.shards.iter_mut() {
                s.hin(h.len()).copy_from_slice(&h);
            }
            self.segment()?;
            self.candidates(rows.len())?
        };
        Ok(result)
    }

    /// Run the next segment of the plan (its inputs are in every local rank's `hin`) and wait for it.
    fn segment(&mut self) -> Result<()> {
        let j = self.next;
        let seg = *self.plan.get(j).context("segment beyond the plan")?;
        self.next += 1;
        let g = Instant::now();
        for s in self.shards.iter_mut() {
            if !self.gated {
                s.submit(seg, self.n)?;
            } else if j == 0 {
                s.start_gated(&self.plan, self.n)?;
            } else {
                s.release()?;
            }
        }
        for s in self.shards.iter_mut() {
            s.wait_segment()?;
        }
        let dt = g.elapsed().as_secs_f64();
        self.seconds.gpu += dt;
        match seg {
            Seg::Attn { .. } => self.seconds.gpu_attn += dt,
            Seg::Moe { .. } => self.seconds.gpu_moe += dt,
            Seg::Head { .. } => {}
        }
        Ok(())
    }

    /// One layer application to the `n` rows of `cur`.
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &mut self,
        app: App,
        n: usize,
        cur: &mut Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        normed: &mut Vec<Vec<f32>>,
        msg: Option<&[Vec<f32>; 4]>,
        mut reference: Option<&mut Reference>,
    ) -> Result<()> {
        let c = self.c.clone();
        let d = c.d;
        let len = n * d;
        let inp_sa = cur.clone();
        let cur_n = rows_rms(cur, d);
        // Attention inputs: the residual mixes (one, or q/k/v on global layers), then with messages the q/k/v
        // messages; the GPUs apply the controllers and the norms.
        let controlled = msg.is_some();
        let kinds = {
            let l = &self.hw.layers[app.phys];
            let mixes = if c.swa(app.phys) { 1 } else { 3 };
            let kinds = if controlled { 3 } else { mixes };
            let hin = self.shards[0].hin(2 * 3 * len);
            for j in 0..kinds {
                let o = &mut hin[j * len..(j + 1) * len];
                match &l.attn_query {
                    Some(q) => {
                        let jq = if mixes == 1 { 0 } else { j };
                        for t in 0..n {
                            let states: Vec<&[f32]> = normed.iter().map(|b| &b[t * d..(t + 1) * d]).chain([&cur_n[t * d..(t + 1) * d]]).collect();
                            mix(&states, &q[jq], &mut o[t * d..(t + 1) * d]);
                        }
                    }
                    None => o.copy_from_slice(cur),
                }
            }
            if let Some(m) = msg {
                for j in 0..3 {
                    hin[(3 + j) * len..(4 + j) * len].copy_from_slice(&m[j]);
                }
            }
            let used = if controlled { 6 * len } else { kinds * len };
            let first = hin[..used].to_vec();
            for s in &self.shards[1..] {
                s.hin(used).copy_from_slice(&first);
            }
            kinds
        };
        debug_assert!(matches!(self.plan[self.next], Seg::Attn { kinds: k, .. } if k == kinds));
        self.segment()?;
        self.allreduce(len)?;
        let mut attn = vec![0f32; len];
        {
            let l = &self.hw.layers[app.phys];
            let hist = &mut self.attn_hist[app.slot];
            for t in 0..n {
                super::conv_step(&self.sum[t * d..(t + 1) * d], hist, &l.attn_post_conv, &mut attn[t * d..(t + 1) * d]);
            }
        }
        if app.phys % c.block == 0 {
            normed.push(rows_rms(&inp_sa, d));
            blocks.push(inp_sa);
            *cur = attn;
        } else {
            for (x, a) in cur.iter_mut().zip(&attn) {
                *x += a;
            }
        }
        let ffn_inp = cur.clone();
        {
            let l = &self.hw.layers[app.phys];
            let cur_n = rows_rms(cur, d);
            let mut h = vec![0f32; len];
            for t in 0..n {
                let states: Vec<&[f32]> = normed.iter().map(|b| &b[t * d..(t + 1) * d]).chain([&cur_n[t * d..(t + 1) * d]]).collect();
                mix(&states, &l.ffn_query, &mut h[t * d..(t + 1) * d]);
            }
            let hin = self.shards[0].hin(2 * len);
            hin[..len].copy_from_slice(&h);
            if let Some(m) = msg {
                hin[len..2 * len].copy_from_slice(&m[3]);
            }
            let used = if controlled { 2 * len } else { len };
            let first = hin[..used].to_vec();
            for s in &self.shards[1..] {
                s.hin(used).copy_from_slice(&first);
            }
        }
        self.segment()?;
        self.allreduce(len)?;
        {
            let l = &self.hw.layers[app.phys];
            let hist = &mut self.ffn_hist[app.slot];
            let mut out = vec![0f32; d];
            for t in 0..n {
                super::conv_step(&self.sum[t * d..(t + 1) * d], hist, &l.ffn_post_conv, &mut out);
                for (x, (a, o)) in cur[t * d..(t + 1) * d].iter_mut().zip(ffn_inp[t * d..(t + 1) * d].iter().zip(&out)) {
                    *x = a + o;
                }
            }
        }
        if let Some(r) = reference.as_deref_mut() {
            r.check(&format!("l_out-{}", app.slot), cur, true)?;
        }
        Ok(())
    }

    /// Add the local ranks' partials (rank order), all-reduce with the other machines; the total in `self.sum`.
    fn allreduce(&mut self, len: usize) -> Result<()> {
        let sum = &mut self.sum[..len];
        sum.copy_from_slice(self.shards[0].part(len));
        for s in &self.shards[1..] {
            for (a, b) in sum.iter_mut().zip(s.part(len)) {
                *a += b;
            }
        }
        if let Some(m) = self.mesh.as_mut() {
            let e = Instant::now();
            let last = m.all_reduce(sum, &mut self.remote[..len], Duration::from_secs(5), 1, self.c.d)?;
            if last {
                for (a, b) in sum.iter_mut().zip(&self.remote[..len]) {
                    *a += b;
                }
            }
            self.seconds.exchange += e.elapsed().as_secs_f64();
        }
        Ok(())
    }

    /// The best TOP_K candidates of each of `rows` scored rows: softcapped logits of the real vocabulary from every
    /// local rank, merged, then merged with the other machines' in the all-reduce rounds.
    fn candidates(&mut self, rows: usize) -> Result<Vec<Candidates>> {
        let cap = self.c.softcap;
        let vocab = self.c.vocab;
        let mut cols: Vec<Candidates> = vec![Vec::new(); rows];
        for s in &self.shards {
            let (v0, vn) = s.vocab_rows();
            let logits = s.logits(rows);
            for (r, col) in cols.iter_mut().enumerate() {
                let row = &logits[r * vn..(r + 1) * vn];
                let mut top: Candidates = Vec::with_capacity(vn);
                for (i, &x) in row.iter().enumerate() {
                    let id = v0 + i;
                    if id < vocab {
                        top.push((if cap > 0. { cap * (x / cap).tanh() } else { x }, id as u32));
                    }
                }
                let k = TOP_K.min(top.len());
                top.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                top.truncate(k);
                col.extend(top);
            }
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
            if let Some(f) = m.get("forward") {
                let tokens: Vec<u32> = serde_json::from_value(f["tokens"].clone())?;
                let score = parse_score(f["score"].as_str().unwrap_or("last"))?;
                let mut reference = f["force"].as_str().map(|p| Reference::new(Path::new(p)));
                let x = embed(&self.gguf, &self.c, &self.hw, &tokens)?;
                self.run(x, tokens.len(), score, reference.as_mut())?;
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

/// RMS-normed copies (no weight) of each `d`-row of `x`, as the residual mixes read them.
fn rows_rms(x: &[f32], d: usize) -> Vec<f32> {
    let mut o = vec![0f32; x.len()];
    for (a, b) in x.chunks_exact(d).zip(o.chunks_exact_mut(d)) {
        rms_into(a, None, 1e-6, b);
    }
    o
}

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

