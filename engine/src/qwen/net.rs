//! A chain of machines, each running a contiguous range of layers. The driver embeds tokens and runs the
//! first layers; every machine sends its output hidden state to the next one over TCP (Thunderbolt
//! networking or Ethernet), and the last machine returns the top candidates back along the chain.
//! Machines only talk to their neighbours, so a daisy chain of cables is enough.
//!
//! Frames are `[u32 header length][u64 blob length][JSON header][blob]`. Requests:
//! - `{"probe": [addresses after the receiver]}` -> `{"devices": [stage count per machine]}`
//! - `{"setup": {"plans": [[[first, last], ...] per machine], "context", "batch", "verify", "mtp"}}` ->
//!   `{"ok": true}`. `verify` sizes the rollback snapshots; with `mtp` the last machine loads its MTP layer.
//! - `{"forward": {"pos", "n", "reply", "dump", "verify", "tokens"}}` + f32 hidden rows -> when `reply`,
//!   `{"columns": [[[score, id], ...] per column], "captures": [[name, len], ...]}` + concatenated f32
//!   captures. Columns hold the last token, or every token with `verify`, which also keeps per-token
//!   recurrent states. `tokens` (the batch's ids) feed the MTP layer.
//! - `{"rollback": column}`: keep tokens `0..=column` of the latest verification batch.
//! - `{"save": slot}` / `{"restore": slot}`: copy the recurrent state into a checkpoint slot or back.
//! - `{"draft": {"token", "pos", "k", "vocab", "temperature", "top_k", "top_p", "seed", "stop"}}` ->
//!   `{"drafts": [[id, [[score, id], ...]], ...]}` from the MTP layer on the last machine.
//!
//! Requests without a reply stream on immediately, so prompt chunks pipeline across machines; a later
//! reply implies every earlier request finished, because each machine handles requests in order.
use super::{
    cpu::Captures,
    embed,
    gpu::Output,
    pipeline::{plan, Devices, Draft, Local},
    Config,
};
use crate::{generation::Sampler, gguf::Gguf};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    io::{BufReader, BufWriter, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
};

pub type Candidates = Vec<(f32, u32)>;

struct Conn {
    r: BufReader<TcpStream>,
    w: BufWriter<TcpStream>,
}

impl Conn {
    fn new(s: TcpStream) -> Result<Self> {
        s.set_nodelay(true)?;
        Ok(Self {
            r: BufReader::with_capacity(1 << 20, s.try_clone()?),
            w: BufWriter::with_capacity(1 << 20, s),
        })
    }
    fn connect(addr: &str) -> Result<Self> {
        Self::new(TcpStream::connect(addr).with_context(|| format!("connecting to {addr}"))?)
    }
    fn send(&mut self, header: &Value, blob: &[u8]) -> Result<()> {
        let h = serde_json::to_vec(header)?;
        self.w.write_all(&(h.len() as u32).to_le_bytes())?;
        self.w.write_all(&(blob.len() as u64).to_le_bytes())?;
        self.w.write_all(&h)?;
        self.w.write_all(blob)?;
        self.w.flush()?;
        Ok(())
    }
    fn recv(&mut self) -> Result<(Value, Vec<u8>)> {
        let mut len = [0u8; 12];
        self.r.read_exact(&mut len)?;
        let h = u32::from_le_bytes(len[..4].try_into()?) as usize;
        let b = u64::from_le_bytes(len[4..].try_into()?) as usize;
        let mut header = vec![0u8; h];
        self.r.read_exact(&mut header)?;
        let mut blob = vec![0u8; b];
        self.r.read_exact(&mut blob)?;
        let header: Value = serde_json::from_slice(&header)?;
        if let Some(e) = header.get("error") {
            bail!("remote: {}", e.as_str().unwrap_or("error"));
        }
        Ok((header, blob))
    }
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .collect()
}

fn ranges(v: &Value) -> Result<Vec<(usize, usize)>> {
    v.as_array()
        .context("invalid plan")?
        .iter()
        .map(|r| Ok((r[0].as_u64().context("plan")? as usize, r[1].as_u64().context("plan")? as usize)))
        .collect()
}

fn candidates_json(top: &[(f32, u32)]) -> Value {
    json!(top.iter().map(|(s, id)| json!([s, id])).collect::<Vec<_>>())
}

fn candidates(v: &Value) -> Result<Candidates> {
    v.as_array()
        .context("missing candidates")?
        .iter()
        .map(|p| Ok((p[0].as_f64().context("score")? as f32, p[1].as_u64().context("id")? as u32)))
        .collect()
}

/// Encode a result for the upstream machine.
fn result(columns: &[Candidates], caps: &Captures) -> (Value, Vec<u8>) {
    let names: Vec<Value> = caps.iter().map(|(n, v)| json!([n, v.len()])).collect();
    let blob: Vec<u8> = caps.values().flat_map(|v| bytemuck::cast_slice::<f32, u8>(v).to_vec()).collect();
    let columns: Vec<Value> = columns.iter().map(|c| candidates_json(c)).collect();
    (json!({"columns": columns, "captures": names}), blob)
}

fn decode_result(header: &Value, blob: &[u8], caps: Option<&mut Captures>) -> Result<Vec<Candidates>> {
    let columns = header["columns"]
        .as_array()
        .context("missing candidate columns")?
        .iter()
        .map(candidates)
        .collect::<Result<Vec<_>>>()?;
    if let Some(m) = caps {
        let mut at = 0;
        for c in header["captures"].as_array().into_iter().flatten() {
            let len = c[1].as_u64().context("capture length")? as usize;
            m.insert(c[0].as_str().context("capture name")?.into(), floats(&blob[at..at + len * 4]));
            at += len * 4;
        }
    }
    Ok(columns)
}

fn decode_drafts(header: &Value) -> Result<Vec<Draft>> {
    header["drafts"]
        .as_array()
        .context("missing drafts")?
        .iter()
        .map(|d| Ok((d[0].as_u64().context("draft id")? as u32, candidates(&d[1])?)))
        .collect()
}

/// Run a draft request on this machine, which ends the model.
fn draft_here(l: &mut Local, r: &Value) -> Result<Vec<Draft>> {
    let num = |k: &str| r[k].as_f64().with_context(|| format!("draft {k}"));
    let mut sampler = Sampler::new(num("temperature")? as f32, num("top_k")? as usize, num("top_p")? as f32, r["seed"].as_u64().context("seed")?)?;
    let stop: Vec<u32> = serde_json::from_value(r["stop"].clone())?;
    let vocab = r["vocab"].as_u64().unwrap_or(0) as usize;
    l.draft(num("token")? as u32, num("pos")? as usize, num("k")? as usize, vocab, &mut sampler, &stop)
}

fn drafts_json(drafts: &[Draft]) -> Value {
    json!({"drafts": drafts.iter().map(|(id, top)| json!([id, candidates_json(top)])).collect::<Vec<_>>()})
}

/// Serve layers to an upstream machine, one session at a time. `mtp` is this machine's MTP layer GGUF,
/// used when it ends the model and the driver asks for drafts.
pub fn serve(model: &Path, mtp: Option<PathBuf>, listen: &str, devices: Devices) -> Result<()> {
    let listener = TcpListener::bind(listen).with_context(|| format!("listening on {listen}"))?;
    eprintln!("node: {} on {listen}, devices {devices:?}", model.display());
    for stream in listener.incoming() {
        let stream = stream?;
        eprintln!("node: session from {}", stream.peer_addr()?);
        let mut up = Conn::new(stream)?;
        if let Err(e) = session(model, mtp.as_deref(), &devices, &mut up) {
            eprintln!("node: session ended: {e:#}");
            let _ = up.send(&json!({"error": format!("{e:#}")}), &[]);
        }
    }
    Ok(())
}

fn session(model: &Path, mtp: Option<&Path>, devices: &Devices, up: &mut Conn) -> Result<()> {
    let mut down: Option<Conn> = None;
    let mut local: Option<Local> = None;
    loop {
        let (header, blob) = match up.recv() {
            Ok(m) => m,
            Err(e) if e.downcast_ref::<std::io::Error>().is_some() => return Ok(()),
            Err(e) => return Err(e),
        };
        if let Some(chain) = header.get("probe") {
            let chain: Vec<String> = serde_json::from_value(chain.clone())?;
            let mut counts = vec![devices.count()];
            if let Some((next, rest)) = chain.split_first() {
                let mut d = Conn::connect(next)?;
                d.send(&json!({"probe": rest}), &[])?;
                let (reply, _) = d.recv()?;
                counts.extend(serde_json::from_value::<Vec<usize>>(reply["devices"].clone())?);
                down = Some(d);
            }
            up.send(&json!({"devices": counts}), &[])?;
        } else if let Some(s) = header.get("setup") {
            let plans = s["plans"].as_array().context("missing plans")?;
            let context = s["context"].as_u64().context("context")? as usize;
            let batch = s["batch"].as_u64().context("batch")? as usize;
            let verify = s["verify"].as_u64().unwrap_or(0) as usize;
            let want_mtp = s["mtp"].as_bool().unwrap_or(false);
            let mine = ranges(&plans[0])?;
            let last = plans.len() == 1;
            ensure!(last != down.is_some(), "plan does not match the probed chain");
            ensure!(!(last && want_mtp && mtp.is_none()), "the last machine needs --mtp for drafts");
            drop(local.take());
            let m = mtp.filter(|_| want_mtp);
            local = Some(Local::new(model, devices, &mine, last, context, batch, verify, m)?);
            if let Some(d) = down.as_mut() {
                let mut s = s.clone();
                s["plans"] = json!(&plans[1..]);
                d.send(&json!({"setup": s}), &[])?;
                d.recv()?;
            }
            up.send(&json!({"ok": true}), &[])?;
        } else if let Some(f) = header.get("forward") {
            let l = local.as_mut().context("forward before setup")?;
            let pos = f["pos"].as_u64().context("pos")? as usize;
            let n = f["n"].as_u64().context("n")? as usize;
            let reply = f["reply"].as_bool().unwrap_or(true);
            let dump = f["dump"].as_bool().unwrap_or(false);
            let verify = f["verify"].as_bool().unwrap_or(false);
            let tokens: Vec<u32> = match f.get("tokens") {
                Some(t) => serde_json::from_value(t.clone())?,
                None => vec![0; n],
            };
            ensure!(tokens.len() == n, "token count mismatch");
            let mut caps = Captures::new();
            let out = l.forward(floats(&blob), pos, &tokens, verify, dump.then_some(&mut caps))?;
            match (out, down.as_mut()) {
                (Output::Top(columns), _) => {
                    if reply {
                        let (h, b) = result(&columns, &caps);
                        up.send(&h, &b)?;
                    }
                }
                (Output::Hidden(x), Some(d)) => {
                    d.send(&json!({"forward": f}), bytemuck::cast_slice(&x))?;
                    if reply {
                        let (h, b) = d.recv()?;
                        let columns = decode_result(&h, &b, Some(&mut caps))?;
                        let (h, b) = result(&columns, &caps);
                        up.send(&h, &b)?;
                    }
                }
                (Output::Hidden(_), None) => bail!("the last machine has no LM head"),
            }
        } else if let Some(r) = header.get("draft") {
            let l = local.as_mut().context("draft before setup")?;
            match down.as_mut() {
                Some(d) => {
                    d.send(&header, &[])?;
                    let (h, _) = d.recv()?;
                    up.send(&h, &[])?;
                }
                None => up.send(&drafts_json(&draft_here(l, r)?), &[])?,
            }
        } else {
            let l = local.as_mut().context("request before setup")?;
            let arg = |k: &str| header[k].as_u64().map(|v| v as usize);
            if let Some(column) = arg("rollback") {
                l.rollback(column)?;
            } else if let Some(slot) = arg("save") {
                l.save(slot)?;
            } else if let Some(slot) = arg("restore") {
                l.restore(slot)?;
            } else {
                bail!("unknown request {header}");
            }
            if let Some(d) = down.as_mut() {
                d.send(&header, &[])?;
            }
        }
    }
}

/// The driving machine: tokens in, top candidates out, with the rest of the chain behind `next`.
pub struct Chain {
    pub local: Local,
    next: Option<Conn>,
    pub pos: usize,
    pub max_batch: usize,
    pub context: usize,
    /// Largest verification batch that can be rolled back (0: none).
    pub verify: usize,
    pub mtp: bool,
    /// Layer ranges per machine, the driver first.
    pub plans: Vec<Vec<(usize, usize)>>,
    /// Columns of the latest verification batch.
    verified: usize,
    /// Tokens processed at each position: the live state covers `history[..pos]`; later entries stay
    /// valid in the KV caches (and for checkpoints) until a forward rewrites their positions.
    pub history: Vec<u32>,
    checkpoints: Vec<Checkpoint>,
    /// Checkpoint slots kept for prefix reuse (least recently used are replaced); 0 disables reuse.
    pub max_checkpoints: usize,
    clock: u64,
}

/// Prompt positions between checkpoints in long prefills.
const CHECKPOINT_INTERVAL: usize = 4096;

/// Where a prefill of positions `start..end` stops for a checkpoint: the `marks` inside it, every
/// CHECKPOINT_INTERVAL positions (so an edit deep in a long history still reuses most of it) and `end`.
fn checkpoint_positions(marks: &[usize], start: usize, end: usize) -> Vec<usize> {
    let every = (start / CHECKPOINT_INTERVAL + 1..).map(|i| i * CHECKPOINT_INTERVAL).take_while(|&p| p < end);
    let mut cuts: Vec<usize> = marks.iter().copied().filter(|&m| m > start && m < end).chain(every).collect();
    cuts.sort_unstable();
    cuts.dedup();
    cuts.push(end);
    cuts
}

/// Recurrent state of every machine after `history[..pos]`.
struct Checkpoint {
    slot: usize,
    pos: usize,
    used: u64,
}

impl Chain {
    /// `cuts` overrides the byte-balanced placement: the first layer of every stage after the first.
    /// `verify` sizes the rollback snapshots for speculative decoding; `mtp` is the driver's MTP layer GGUF,
    /// used when it also ends the model (a remote last machine uses its own `--mtp`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &Path,
        devices: Devices,
        nodes: &[String],
        cuts: &[usize],
        context: usize,
        max_batch: usize,
        verify: usize,
        mtp: Option<&Path>,
        draft: bool,
    ) -> Result<Self> {
        ensure!(context > 0 && max_batch > 0, "context and batch must be positive");
        ensure!(verify <= 8 && verify <= max_batch, "verification batches are limited to eight columns");
        let g = Gguf::open(model)?;
        let c = Config::new(&g)?;
        let mut counts = vec![devices.count()];
        let mut next = None;
        if let Some((first, rest)) = nodes.split_first() {
            let mut conn = Conn::connect(first)?;
            conn.send(&json!({"probe": rest}), &[])?;
            let (reply, _) = conn.recv()?;
            counts.extend(serde_json::from_value::<Vec<usize>>(reply["devices"].clone())?);
            next = Some(conn);
        }
        let stages: usize = counts.iter().sum();
        let all = if cuts.is_empty() {
            plan(&g, &c, stages, draft)
        } else {
            ensure!(cuts.len() + 1 == stages, "--cuts needs one entry per stage after the first ({stages} stages)");
            let mut edges = vec![0];
            edges.extend(cuts);
            edges.push(c.layers);
            ensure!(edges.windows(2).all(|w| w[0] < w[1]), "cuts must increase within 1..{}", c.layers);
            edges.windows(2).map(|w| (w[0], w[1])).collect()
        };
        let mut plans = Vec::new();
        let mut it = all.into_iter();
        for count in &counts {
            plans.push(it.by_ref().take(*count).collect::<Vec<_>>());
        }
        ensure!(plans.iter().all(|p| !p.is_empty()), "more machines than layers");
        eprintln!("plan: {plans:?}");
        drop(g);
        ensure!(!draft || next.is_some() || mtp.is_some(), "drafting needs --mtp");
        if let Some(conn) = next.as_mut() {
            let remote: Vec<Value> = plans[1..]
                .iter()
                .map(|p| json!(p.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>()))
                .collect();
            conn.send(
                &json!({"setup": {"plans": remote, "context": context, "batch": max_batch, "verify": verify, "mtp": draft}}),
                &[],
            )?;
            conn.recv()?;
        }
        let head = next.is_none();
        let local = Local::new(model, &devices, &plans[0], head, context, max_batch, verify, mtp.filter(|_| draft))?;
        Ok(Self {
            local,
            next,
            pos: 0,
            max_batch,
            context,
            verify,
            mtp: draft,
            plans,
            verified: 0,
            history: Vec::new(),
            checkpoints: Vec::new(),
            max_checkpoints: 0,
            clock: 0,
        })
    }

    /// Tensor parallelism instead of a chain: every layer split across this machine's `gpus` and, with
    /// `peer`, the other machine's (`tp`).
    #[allow(clippy::too_many_arguments)]
    pub fn tensor_parallel(model: &Path, gpus: &[usize], peer: Option<crate::link::Lanes>, context: usize, max_batch: usize, verify: usize, mtp: Option<&Path>) -> Result<Self> {
        ensure!(context > 0 && max_batch > 0, "context and batch must be positive");
        ensure!(verify <= 8 && verify <= max_batch, "verification batches are limited to eight columns");
        let local = Local::tensor_parallel(model, gpus, peer, context, max_batch, verify, mtp)?;
        let layers = local.c.layers;
        Ok(Self {
            local,
            next: None,
            pos: 0,
            max_batch,
            context,
            verify,
            mtp: mtp.is_some(),
            plans: vec![vec![(0, layers)]],
            verified: 0,
            history: Vec::new(),
            checkpoints: Vec::new(),
            max_checkpoints: 0,
            clock: 0,
        })
    }

    fn run(&mut self, tokens: &[u32], verify: bool, reply: bool, mut dump: Option<&mut Captures>) -> Result<Option<Vec<Candidates>>> {
        let n = tokens.len();
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(self.pos + n <= self.context, "context exhausted");
        let x = embed(&self.local.gguf, &self.local.c, tokens)?;
        let pos = self.pos;
        // Rewriting these positions invalidates later history and the checkpoints that depend on it.
        self.history.truncate(pos);
        self.checkpoints.retain(|c| c.pos <= pos);
        self.history.extend_from_slice(tokens);
        let out = self.local.forward(x, pos, tokens, verify, dump.as_deref_mut())?;
        self.pos += n;
        self.verified = if verify { n } else { 0 };
        match (out, self.next.as_mut()) {
            (Output::Top(columns), _) => Ok(Some(columns)),
            (Output::Hidden(x), Some(conn)) => {
                let mut f = json!({"pos": pos, "n": n, "reply": reply, "dump": dump.is_some(), "verify": verify});
                if self.mtp {
                    f["tokens"] = json!(tokens);
                }
                conn.send(&json!({"forward": f}), bytemuck::cast_slice(&x))?;
                if !reply {
                    return Ok(None);
                }
                let (h, b) = conn.recv()?;
                Ok(Some(decode_result(&h, &b, dump)?))
            }
            (Output::Hidden(_), None) => bail!("no machine holds the LM head"),
        }
    }

    /// Process `tokens` at the current position and return the last token's candidates. Without `reply`
    /// the result of a remote chain is not awaited (prompt chunks stream through) and `None` is returned.
    pub fn forward(&mut self, tokens: &[u32], reply: bool, dump: Option<&mut Captures>) -> Result<Option<Candidates>> {
        Ok(self.run(tokens, false, reply, dump)?.map(|mut c| c.pop().unwrap()))
    }

    /// Process a verification batch: candidates for every token, recurrent states kept for `rollback`.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Candidates>> {
        ensure!(tokens.len() <= self.verify.max(1), "verification batch exceeds the configured {} columns", self.verify);
        self.run(tokens, true, true, None)?.context("missing candidates")
    }

    /// Keep the first `keep` tokens of the latest verification batch.
    pub fn rollback(&mut self, keep: usize) -> Result<()> {
        ensure!(keep > 0 && keep <= self.verified, "rollback outside the latest verification batch");
        if keep < self.verified {
            self.local.rollback(keep - 1)?;
            if let Some(conn) = self.next.as_mut() {
                conn.send(&json!({"rollback": keep - 1}), &[])?;
            }
            self.pos -= self.verified - keep;
            self.history.truncate(self.pos);
        }
        self.verified = 0;
        Ok(())
    }

    /// Checkpoint the recurrent state of every machine at the current position.
    pub fn save(&mut self, slot: usize) -> Result<()> {
        self.local.save(slot)?;
        if let Some(conn) = self.next.as_mut() {
            conn.send(&json!({"save": slot}), &[])?;
        }
        Ok(())
    }

    /// Return every machine to checkpoint `slot`, taken at `pos`. Attention KV below `pos` must be unchanged
    /// since then (the caller tracks which positions were rewritten).
    pub fn restore(&mut self, slot: usize, pos: usize) -> Result<()> {
        self.local.restore(slot)?;
        if let Some(conn) = self.next.as_mut() {
            conn.send(&json!({"restore": slot}), &[])?;
        }
        self.pos = pos;
        self.verified = 0;
        Ok(())
    }

    /// Up to `k` MTP drafts following the sampled, not yet processed `token` at the current position, from
    /// the first `vocab` tokens of the vocabulary (0: all).
    pub fn draft(&mut self, token: u32, k: usize, vocab: usize, sampler: &Sampler, seed: u64, stop: &[u32]) -> Result<Vec<Draft>> {
        ensure!(self.mtp, "drafting needs --mtp");
        let r = json!({"token": token, "pos": self.pos, "k": k, "vocab": vocab, "temperature": sampler.temperature,
            "top_k": sampler.top_k, "top_p": sampler.top_p, "seed": seed, "stop": stop});
        match self.next.as_mut() {
            Some(conn) => {
                conn.send(&json!({"draft": r}), &[])?;
                decode_drafts(&conn.recv()?.0)
            }
            None => draft_here(&mut self.local, &r),
        }
    }

    /// Prefill a prompt from position 0 in chunks; only the final chunk waits for its result.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<Candidates> {
        self.pos = 0;
        self.extend(ids)
    }

    /// Drop the cached sequence and checkpoints, e.g. after a failed request.
    pub fn forget(&mut self) {
        self.history.clear();
        self.checkpoints.clear();
        self.pos = 0;
        self.verified = 0;
    }

    /// Checkpoint the current position for prefix reuse, replacing the least recently used slot.
    pub fn checkpoint(&mut self) -> Result<()> {
        if self.max_checkpoints == 0 || self.pos == 0 {
            return Ok(());
        }
        self.clock += 1;
        if let Some(c) = self.checkpoints.iter_mut().find(|c| c.pos == self.pos) {
            c.used = self.clock;
            return Ok(());
        }
        let slot = if self.checkpoints.len() < self.max_checkpoints {
            (0..).find(|s| !self.checkpoints.iter().any(|c| c.slot == *s)).unwrap()
        } else {
            let oldest = (0..self.checkpoints.len()).min_by_key(|&i| self.checkpoints[i].used).unwrap();
            self.checkpoints.remove(oldest).slot
        };
        self.save(slot)?;
        self.checkpoints.push(Checkpoint { slot, pos: self.pos, used: self.clock });
        Ok(())
    }

    /// Prefill `ids` from the longest cached prefix: the live state or a checkpoint at or before the
    /// common prefix with `history` (at least the last token is recomputed for its candidates).
    /// Checkpoints are taken at `marks`, every CHECKPOINT_INTERVAL positions and at the end of the prompt. Returns the last token's candidates
    /// and the number of reused positions.
    pub fn prefill_cached(&mut self, ids: &[u32], marks: &[usize]) -> Result<(Candidates, usize)> {
        ensure!(!ids.is_empty(), "empty prompt");
        let common = self.history.iter().zip(ids).take_while(|(a, b)| a == b).count();
        let target = if self.max_checkpoints == 0 { 0 } else { common.min(ids.len() - 1) };
        let live = if self.pos <= target { self.pos } else { 0 };
        let best = self.checkpoints.iter().filter(|c| c.pos <= target).max_by_key(|c| c.pos);
        match best {
            Some(c) if c.pos > live => {
                let (slot, pos) = (c.slot, c.pos);
                self.clock += 1;
                self.checkpoints.iter_mut().find(|c| c.slot == slot).unwrap().used = self.clock;
                self.restore(slot, pos)?;
            }
            _ => self.pos = live,
        }
        let start = self.pos;
        let cuts = checkpoint_positions(marks, start, ids.len());
        let mut top = None;
        let mut at = start;
        for (i, &end) in cuts.iter().enumerate() {
            let pieces: Vec<_> = ids[at..end].chunks(self.max_batch).collect();
            for (j, chunk) in pieces.iter().enumerate() {
                top = self.forward(chunk, i + 1 == cuts.len() && j + 1 == pieces.len(), None)?;
            }
            self.checkpoint()?;
            at = end;
        }
        Ok((top.context("missing candidates")?, start))
    }

    /// Process `ids` at the current position in chunks; only the final chunk waits for its result.
    pub fn extend(&mut self, ids: &[u32]) -> Result<Candidates> {
        let chunks: Vec<_> = ids.chunks(self.max_batch).collect();
        let mut top = None;
        for (i, chunk) in chunks.iter().enumerate() {
            top = self.forward(chunk, i + 1 == chunks.len(), None)?;
        }
        top.context("empty prompt")
    }
}

#[cfg(test)]
mod tests {
    use super::checkpoint_positions;

    #[test]
    fn checkpoints_at_marks_intervals_and_the_end() {
        assert_eq!(checkpoint_positions(&[30], 0, 40), vec![30, 40]);
        assert_eq!(checkpoint_positions(&[30, 0, 40, 50], 10, 40), vec![30, 40]);
        assert_eq!(checkpoint_positions(&[5000], 100, 9000), vec![4096, 5000, 8192, 9000]);
        assert_eq!(checkpoint_positions(&[], 4096, 8192), vec![8192]);
        assert_eq!(checkpoint_positions(&[], 4095, 4097), vec![4096, 4097]);
    }
}
