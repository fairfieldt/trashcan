//! The qwen4exp machine chain, with the protocol of `qwen::net` (frames `[u32 header length][u64 blob
//! length][JSON header][blob]`; probe, setup and forward requests). The payload between machines is the
//! hyper-connection residual, `hc * d` floats per token. The driver embeds tokens, gathers the n-gram rows and
//! owns the PLE layer.
use super::{
    cpu::Captures,
    embed,
    gpu::{merge_top, KvCache, Output},
    ngram,
    pipeline::{assemble_logits, head_rows, plan, Done, Local},
    Config,
};
use crate::{gguf::Gguf, qwen::pipeline::Devices};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    io::{BufReader, BufWriter, Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    path::Path,
    sync::mpsc,
};

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
        read_frame(&mut self.r)
    }
}

fn read_frame(r: &mut impl Read) -> Result<(Value, Vec<u8>)> {
    let mut len = [0u8; 12];
    r.read_exact(&mut len)?;
    let h = u32::from_le_bytes(len[..4].try_into()?) as usize;
    let b = u64::from_le_bytes(len[4..].try_into()?) as usize;
    let mut header = vec![0u8; h];
    r.read_exact(&mut header)?;
    let mut blob = vec![0u8; b];
    r.read_exact(&mut blob)?;
    let header: Value = serde_json::from_slice(&header)?;
    if let Some(e) = header.get("error") {
        bail!("remote: {}", e.as_str().unwrap_or("error"));
    }
    Ok((header, blob))
}

type Requests = mpsc::Receiver<Result<(Value, Vec<u8>)>>;

/// Read the previous machine's requests on a thread of their own, so that it can send the next prompt chunk (MiBs,
/// more than the socket buffers take) while this machine's GPUs work.
fn requests(s: &TcpStream) -> Result<Requests> {
    let mut r = BufReader::with_capacity(1 << 20, s.try_clone()?);
    let (tx, rx) = mpsc::sync_channel(4);
    std::thread::spawn(move || loop {
        let m = read_frame(&mut r);
        let end = m.is_err();
        if tx.send(m).is_err() || end {
            break;
        }
    });
    Ok(rx)
}

/// Send a prompt chunk that left this machine's stages on to the next machine. On the machine that ends the model it
/// is done: nobody waits for its candidates.
fn pass_on(next: Option<&mut Conn>, d: Done) -> Result<()> {
    match (d.out, next) {
        (Output::Hidden(x), Some(conn)) => conn.send(
            &json!({"forward": {"pos": d.pos, "n": d.n, "reply": false, "dump": false}}),
            bytemuck::cast_slice(&x),
        ),
        (Output::Hidden(_), None) => bail!("no machine holds the LM head"),
        (Output::Final(_), _) => Ok(()),
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
        .map(|r| {
            Ok((
                r[0].as_u64().context("plan")? as usize,
                r[1].as_u64().context("plan")? as usize,
            ))
        })
        .collect()
}

// A forward reply is two frames. The final hidden state goes first and every machine passes it up before scoring
// its head rows, so all machines score at the same time; the merged candidates of the machine and those after it
// follow (`result`).

fn send_hidden(up: &mut Conn, h: &[f32]) -> Result<()> {
    up.send(&json!({"h": h.len()}), bytemuck::cast_slice(h))
}

fn recv_hidden(down: &mut Conn) -> Result<Vec<f32>> {
    let (header, blob) = down.recv()?;
    let n = header["h"].as_u64().context("missing hidden state")? as usize;
    ensure!(blob.len() == n * 4, "hidden state length mismatch");
    Ok(floats(&blob))
}

/// The second frame of a forward reply: the candidates so far and any captures.
fn result(top: &[(f32, u32)], caps: &Captures) -> (Value, Vec<u8>) {
    let names: Vec<Value> = caps.iter().map(|(n, v)| json!([n, v.len()])).collect();
    let blob: Vec<u8> = caps.values().flat_map(|v| bytemuck::cast_slice::<f32, u8>(v).to_vec()).collect();
    (
        json!({"top": top.iter().map(|(s, id)| json!([s, id])).collect::<Vec<_>>(), "captures": names}),
        blob,
    )
}

/// Head rows per stage for each machine, as sent in a setup request.
fn heads_json(heads: &[Vec<Option<(usize, usize)>>]) -> Value {
    json!(heads
        .iter()
        .map(|m| m.iter().map(|r| r.map(|(a, b)| json!([a, b]))).collect::<Vec<_>>())
        .collect::<Vec<_>>())
}

fn heads_of(v: &Value) -> Result<Vec<Option<(usize, usize)>>> {
    v.as_array()
        .context("invalid head rows")?
        .iter()
        .map(|r| {
            Ok(if r.is_null() {
                None
            } else {
                Some((
                    r[0].as_u64().context("head rows")? as usize,
                    r[1].as_u64().context("head rows")? as usize,
                ))
            })
        })
        .collect()
}

fn decode_result(header: &Value, blob: &[u8], caps: Option<&mut Captures>) -> Result<Vec<(f32, u32)>> {
    let top = header["top"]
        .as_array()
        .context("missing top candidates")?
        .iter()
        .map(|p| {
            Ok((
                p[0].as_f64().context("score")? as f32,
                p[1].as_u64().context("id")? as u32,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if let Some(m) = caps {
        let mut at = 0;
        for c in header["captures"].as_array().into_iter().flatten() {
            let len = c[1].as_u64().context("capture length")? as usize;
            m.insert(
                c[0].as_str().context("capture name")?.into(),
                floats(&blob[at..at + len * 4]),
            );
            at += len * 4;
        }
    }
    Ok(top)
}

/// Serve layers to an upstream machine, one session at a time.
pub fn serve(model: &Path, listen: &str, devices: Devices) -> Result<()> {
    let listener = TcpListener::bind(listen).with_context(|| format!("listening on {listen}"))?;
    eprintln!("qwen4 node: {} on {listen}, devices {devices:?}", model.display());
    for stream in listener.incoming() {
        let stream = stream?;
        eprintln!("qwen4 node: session from {}", stream.peer_addr()?);
        let requests = requests(&stream)?;
        let mut up = Conn::new(stream)?;
        if let Err(e) = session(model, &devices, &mut up, &requests) {
            eprintln!("qwen4 node: session ended: {e:#}");
            let _ = up.send(&json!({"error": format!("{e:#}")}), &[]);
        }
        // The reader thread holds the socket too: close it as dropping `up` alone did.
        let _ = up.w.get_ref().shutdown(Shutdown::Both);
    }
    Ok(())
}

fn session(model: &Path, devices: &Devices, up: &mut Conn, requests: &Requests) -> Result<()> {
    let mut down: Option<Conn> = None;
    let mut local: Option<Local> = None;
    loop {
        let (header, blob) = match requests.recv() {
            Ok(Ok(m)) => m,
            Ok(Err(e)) if e.downcast_ref::<std::io::Error>().is_none() => return Err(e),
            _ => return Ok(()),
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
            let heads = s["heads"].as_array().context("missing head rows")?;
            let context = s["context"].as_u64().context("context")? as usize;
            let batch = s["batch"].as_u64().context("batch")? as usize;
            let kv = KvCache::parse(s["kv"].as_str().unwrap_or("f16"))?;
            let mine = ranges(&plans[0])?;
            let my_heads = heads_of(&heads[0])?;
            ensure!(
                (plans.len() > 1) == down.is_some(),
                "plan does not match the probed chain"
            );
            drop(local.take());
            local = Some(Local::new(model, devices, &mine, &my_heads, context, batch, kv)?);
            if let Some(d) = down.as_mut() {
                d.send(
                    &json!({"setup": {"plans": &plans[1..], "heads": &heads[1..], "context": context,
                        "batch": batch, "kv": kv.name()}}),
                    &[],
                )?;
                d.recv()?;
            }
            up.send(&json!({"ok": true}), &[])?;
        } else if let Some(max) = header.get("enable_verify") {
            let l = local.as_mut().context("enable_verify before setup")?;
            l.enable_verify(max.as_u64().context("verification size")? as usize)?;
            if let Some(d) = down.as_mut() {
                d.send(&header, &[])?;
                d.recv()?;
            }
            up.send(&json!({"ok": true}), &[])?;
        } else if let Some(keep) = header.get("rollback") {
            let l = local.as_mut().context("rollback before setup")?;
            l.rollback(keep.as_u64().context("rollback")? as usize)?;
            if let Some(d) = down.as_mut() {
                d.send(&header, &[])?;
            }
        } else if let Some(f) = header.get("forward").filter(|f| f.get("keep").is_some()) {
            let l = local.as_mut().context("forward before setup")?;
            kept_node(l, down.as_mut(), up, f, &blob)?;
        } else if let Some(f) = header.get("forward") {
            let l = local.as_mut().context("forward before setup")?;
            let pos = f["pos"].as_u64().context("pos")? as usize;
            let n = f["n"].as_u64().context("n")? as usize;
            let reply = f["reply"].as_bool().unwrap_or(true);
            let dump = f["dump"].as_bool().unwrap_or(false);
            let mut caps = Captures::new();
            let x = floats(&blob);
            // Prompt chunks without a reply stream through this machine's stages, which work on consecutive chunks at
            // the same time; a chunk with a reply waits for all of them.
            let out = if dump {
                l.drain(|d| pass_on(down.as_mut(), d))?;
                l.forward(x, None, pos, n, Some(&mut caps))?
            } else if !reply {
                if let Some(d) = l.push(x, None, pos, n)? {
                    pass_on(down.as_mut(), d)?;
                }
                continue;
            } else {
                l.finish(x, None, pos, n, |d| pass_on(down.as_mut(), d))?
            };
            if let Output::Hidden(x) = &out {
                let d = down.as_mut().context("the last machine has no LM head")?;
                d.send(&json!({"forward": f}), bytemuck::cast_slice(x))?;
            }
            if !reply {
                continue;
            }
            // The final hidden state goes on up first; then this machine's stages score their head rows, and the
            // candidates follow, merged with those of the machines after it.
            let (h, rest) = match out {
                Output::Final(h) => (h, None),
                Output::Hidden(_) => {
                    let d = down.as_mut().context("the last machine has no LM head")?;
                    (recv_hidden(d)?, Some(d))
                }
            };
            send_hidden(up, &h)?;
            let mut top = l.head_slices(&h, dump.then_some(&mut caps))?;
            if let Some(d) = rest {
                let (hd, b) = d.recv()?;
                top = merge_top(top, decode_result(&hd, &b, Some(&mut caps))?);
            }
            let (hd, b) = result(&top, &caps);
            up.send(&hd, &b)?;
        } else {
            bail!("unknown request {header}");
        }
    }
}

// A batch that keeps rows (`"keep": "verify"` or `"rows"`): every machine passes the final hidden states up (all
// rows of a verification batch, the last row otherwise) with the batch's residual rows from the machine that ends the
// model (the driver's MTP block reads them while the head rows are scored), then the candidates of every row follow.

fn send_kept_hidden(up: &mut Conn, h: &[f32], rows: &[f32]) -> Result<()> {
    let mut blob = bytemuck::cast_slice::<f32, u8>(h).to_vec();
    blob.extend_from_slice(bytemuck::cast_slice(rows));
    up.send(&json!({"h": h.len(), "rows": rows.len()}), &blob)
}

fn recv_kept_hidden(down: &mut Conn) -> Result<(Vec<f32>, Vec<f32>)> {
    let (header, blob) = down.recv()?;
    let n = header["h"].as_u64().context("missing hidden state")? as usize;
    let r = header["rows"].as_u64().context("missing rows")? as usize;
    ensure!(blob.len() == (n + r) * 4, "hidden state length mismatch");
    let all = floats(&blob);
    Ok((all[..n].to_vec(), all[n..].to_vec()))
}

fn kept_result(tops: &[Vec<(f32, u32)>]) -> (Value, Vec<u8>) {
    let tops: Vec<Value> = tops.iter().map(|t| json!(t.iter().map(|(s, id)| json!([s, id])).collect::<Vec<_>>())).collect();
    (json!({"tops": tops}), Vec::new())
}

fn decode_kept(header: &Value) -> Result<Vec<Vec<(f32, u32)>>> {
    header["tops"]
        .as_array()
        .context("missing candidates")?
        .iter()
        .map(|t| {
            t.as_array()
                .context("candidates")?
                .iter()
                .map(|p| Ok((p[0].as_f64().context("score")? as f32, p[1].as_u64().context("id")? as u32)))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()
}

fn merge_rows(a: Vec<Vec<(f32, u32)>>, b: Vec<Vec<(f32, u32)>>) -> Result<Vec<Vec<(f32, u32)>>> {
    ensure!(a.len() == b.len(), "candidate rows mismatch");
    Ok(a.into_iter().zip(b).map(|(x, y)| merge_top(x, y)).collect())
}

fn kept_node(l: &mut Local, down: Option<&mut Conn>, up: &mut Conn, f: &Value, blob: &[u8]) -> Result<()> {
    let pos = f["pos"].as_u64().context("pos")? as usize;
    let n = f["n"].as_u64().context("n")? as usize;
    let verify = f["keep"].as_str() == Some("verify");
    let mut down = down;
    // Prompt chunks still in flight go on first.
    l.drain(|d| pass_on(down.as_deref_mut(), d))?;
    let out = l.forward_kept(floats(blob), None, pos, n, verify)?;
    if let Output::Hidden(x) = &out {
        let d = down.as_mut().context("the last machine has no LM head")?;
        d.send(&json!({"forward": f}), bytemuck::cast_slice(x))?;
    }
    let (h, rows) = match out {
        Output::Final(h) => {
            down = None;
            (h, std::mem::take(&mut l.rows))
        }
        Output::Hidden(_) => recv_kept_hidden(down.as_mut().context("the last machine has no LM head")?)?,
    };
    send_kept_hidden(up, &h, &rows)?;
    let mut tops = l.head_rows(&h)?;
    if let Some(d) = down {
        let (hd, _) = d.recv()?;
        tops = merge_rows(tops, decode_kept(&hd)?)?;
    }
    let (hd, b) = kept_result(&tops);
    up.send(&hd, &b)
}

/// Where the driver reads n-gram rows: through the GGUF map, or a `ngram::Table` (resident or from disk).
pub enum Rows {
    Map,
    Table(ngram::Table),
}

impl Rows {
    /// `mode`: map (the GGUF mapping), ram-q4 or ram-q8 (a resident copy), disk (pread through the page cache) or
    /// direct (O_DIRECT page reads).
    pub fn open(model: &Path, mode: &str) -> Result<Self> {
        if mode == "map" {
            return Ok(Rows::Map);
        }
        let g = Gguf::open(model)?;
        let t = g.tensor("per_layer_token_embd.weight").context("no n-gram table")?;
        let table = ngram::Table::open(&g.paths[t.file], mode == "direct")?;
        Ok(Rows::Table(match mode {
            "ram-q4" => table.to_ram(crate::gguf::GgmlType::Q4_0, 16)?,
            "ram-q8" => table.to_ram(crate::gguf::GgmlType::Q8_0, 16)?,
            "disk" | "direct" => table,
            other => bail!("unknown --ngram {other}"),
        }))
    }
}

/// The driving machine: tokens in, top candidates out.
pub struct Chain {
    pub local: Local,
    next: Option<Conn>,
    pub pos: usize,
    pub max_batch: usize,
    pub context: usize,
    pub plans: Vec<Vec<(usize, usize)>>,
    /// Every token so far (the n-gram hash reads the two before each new one).
    pub history: Vec<u32>,
    rows: Rows,
    pub threads: usize,
    /// Tokens of the latest verification batch (for `rollback`).
    verified: usize,
}

impl Chain {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &Path,
        devices: Devices,
        nodes: &[String],
        cuts: &[usize],
        head_split: usize,
        context: usize,
        max_batch: usize,
        kv: KvCache,
        rows: Rows,
    ) -> Result<Self> {
        ensure!(context > 0 && max_batch > 0, "context and batch must be positive");
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
            plan(&g, &c, stages, head_split)
        } else {
            ensure!(
                cuts.len() + 1 == stages,
                "--cuts needs one entry per stage after the first ({stages} stages)"
            );
            let mut edges = vec![0];
            edges.extend(cuts);
            edges.push(c.layers);
            ensure!(
                edges.windows(2).all(|w| w[0] < w[1]),
                "cuts must increase within 1..{}",
                c.layers
            );
            edges.windows(2).map(|w| (w[0], w[1])).collect()
        };
        let mut plans = Vec::new();
        let mut it = all.into_iter();
        for count in &counts {
            plans.push(it.by_ref().take(*count).collect::<Vec<_>>());
        }
        ensure!(plans.iter().all(|p| !p.is_empty()), "more machines than layers");
        if let Some(p) = &c.ngram {
            ensure!(
                plans[0].iter().any(|(a, b)| (*a..*b).contains(&p.layer)),
                "the driver must own the n-gram layer {}",
                p.layer
            );
        }
        eprintln!("plan: {plans:?}");
        let mut all_heads = head_rows(c.vocab, stages, head_split).into_iter();
        let heads: Vec<Vec<Option<(usize, usize)>>> =
            counts.iter().map(|count| all_heads.by_ref().take(*count).collect()).collect();
        ensure!(
            heads.last().and_then(|m| m.last()).is_some_and(|r| r.is_some()),
            "the last stage needs head rows"
        );
        if head_split > 1 {
            eprintln!("head rows: {heads:?}");
        }
        drop(g);
        if let Some(conn) = next.as_mut() {
            let remote: Vec<Value> = plans[1..]
                .iter()
                .map(|p| json!(p.iter().map(|(a, b)| json!([a, b])).collect::<Vec<_>>()))
                .collect();
            conn.send(
                &json!({"setup": {"plans": remote, "heads": heads_json(&heads[1..]), "context": context,
                    "batch": max_batch, "kv": kv.name()}}),
                &[],
            )?;
            conn.recv()?;
        }
        let local = Local::new(model, &devices, &plans[0], &heads[0], context, max_batch, kv)?;
        Ok(Self {
            local,
            next,
            pos: 0,
            max_batch,
            context,
            plans,
            history: Vec::new(),
            rows,
            threads: 16,
            verified: 0,
        })
    }

    /// The n-gram rows of `tokens` following the history.
    fn ple(&self, tokens: &[u32]) -> Result<Option<Vec<f32>>> {
        let Some(p) = &self.local.c.ngram else { return Ok(None) };
        match &self.rows {
            Rows::Map => super::ple_rows(&self.local.gguf, &self.local.c, &self.history, tokens),
            Rows::Table(t) => {
                let rows = p.rows(&self.history, tokens);
                let mut out = vec![0f32; rows.len() * t.dim];
                t.gather(&rows, &mut out, self.threads)?;
                Ok(Some(out))
            }
        }
    }

    /// Process `tokens` at the current position. Without `reply` the tokens are only started (prompt chunks stream
    /// through the stages of every machine): no candidates are computed or awaited.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        reply: bool,
        mut dump: Option<&mut Captures>,
    ) -> Result<Option<Vec<(f32, u32)>>> {
        let n = tokens.len();
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(self.pos + n <= self.context, "context exhausted");
        if self.pos == 0 {
            self.history.clear();
        }
        let x = embed(&self.local.gguf, &self.local.c, tokens)?;
        let ple = self.ple(tokens)?;
        self.history.extend_from_slice(tokens);
        let pos = self.pos;
        let out = if dump.is_some() {
            self.local.drain(|d| pass_on(self.next.as_mut(), d))?;
            self.local.forward(x, ple.as_deref(), pos, n, dump.as_deref_mut())?
        } else if !reply {
            if let Some(d) = self.local.push(x, ple, pos, n)? {
                pass_on(self.next.as_mut(), d)?;
            }
            self.pos += n;
            return Ok(None);
        } else {
            self.local.finish(x, ple, pos, n, |d| pass_on(self.next.as_mut(), d))?
        };
        self.pos += n;
        if let Output::Hidden(x) = &out {
            let conn = self.next.as_mut().context("no machine holds the LM head")?;
            conn.send(
                &json!({"forward": {"pos": pos, "n": n, "reply": reply, "dump": dump.is_some()}}),
                bytemuck::cast_slice(x),
            )?;
        }
        if !reply {
            return Ok(None);
        }
        // The final hidden state arrives first: this machine scores its head rows while the candidates of the
        // others come up the chain.
        let (h, rest) = match out {
            Output::Final(h) => (h, None),
            Output::Hidden(_) => {
                let conn = self.next.as_mut().context("no machine holds the LM head")?;
                (recv_hidden(conn)?, Some(conn))
            }
        };
        let mut top = self.local.head_slices(&h, dump.as_deref_mut())?;
        if let Some(conn) = rest {
            let (hd, b) = conn.recv()?;
            top = merge_top(top, decode_result(&hd, &b, dump.as_deref_mut())?);
        }
        if let Some(m) = dump {
            assemble_logits(m, self.local.c.vocab);
        }
        Ok(Some(top))
    }

    /// Prefill a prompt from position 0 in chunks; only the final chunk waits for its result.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<Vec<(f32, u32)>> {
        self.pos = 0;
        self.extend(ids)
    }

    /// Prefill `ids`, continuing from the live state when everything processed so far is a prefix of them (the
    /// recurrent DeltaNet and n-gram states cannot go back, so any other prompt starts again from position 0).
    /// Returns the last token's candidates and the number of reused positions.
    pub fn prefill_cached(&mut self, ids: &[u32]) -> Result<(Vec<(f32, u32)>, usize)> {
        ensure!(!ids.is_empty(), "empty prompt");
        let live = self.pos > 0 && self.pos < ids.len() && self.history.len() == self.pos && ids.starts_with(&self.history);
        if !live {
            self.pos = 0;
        }
        let start = self.pos;
        Ok((self.extend(&ids[start..])?, start))
    }

    /// Prefill chunk size for `n` tokens. The last chunk's result is back after (chunks + stages - 1) chunk times and
    /// a chunk costs about as much as 16 more tokens (results/qwen4/prefill-chunks-20261012), so the best size is about
    /// sqrt(16 n / (stages - 1)): the nearest power of two from 16 up to the configured batch.
    pub fn chunk_size(&self, n: usize) -> usize {
        let stages: usize = self.plans.iter().map(Vec::len).sum();
        let best = (16. * n as f64 / stages.saturating_sub(1).max(1) as f64).sqrt();
        let pow = 1usize << best.max(1.).log2().round() as u32;
        pow.max(16).min(self.max_batch)
    }

    /// Process `ids` at the current position in chunks; only the final chunk waits for its result.
    fn extend(&mut self, ids: &[u32]) -> Result<Vec<(f32, u32)>> {
        let chunks: Vec<_> = ids.chunks(self.chunk_size(ids.len())).collect();
        let mut top = None;
        for (i, chunk) in chunks.iter().enumerate() {
            top = self.forward(chunk, i + 1 == chunks.len(), None)?;
        }
        top.context("empty prompt")
    }

    /// Discard the live state: the next prompt starts from position 0.
    pub fn forget(&mut self) {
        // Chunks still in flight after an error are dropped with the rest of the state.
        let _ = self.local.drain(|_| Ok(()));
        self.pos = 0;
        self.history.clear();
    }

    /// Make every stage of the chain ready for verification batches of up to `max` tokens.
    pub fn enable_verify(&mut self, max: usize) -> Result<()> {
        self.local.enable_verify(max)?;
        if let Some(conn) = self.next.as_mut() {
            conn.send(&json!({"enable_verify": max}), &[])?;
            conn.recv()?;
        }
        Ok(())
    }

    /// Run `tokens` as one batch that keeps rows: a verification batch (`verify`: rollback possible, candidates of
    /// every row) or the end of a prompt (candidates of the last row). Returns the candidates per row and the
    /// residual rows of the batch at the end of the model (`n x hc x d`), the MTP block's input; `early` gets the
    /// rows as soon as they arrive, while the head rows are still being scored.
    pub fn forward_kept(
        &mut self,
        tokens: &[u32],
        verify: bool,
        early: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<(Vec<Vec<(f32, u32)>>, Vec<f32>)> {
        let n = tokens.len();
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(self.pos + n <= self.context, "context exhausted");
        if self.pos == 0 {
            self.history.clear();
        }
        ensure!(!verify || self.pos > 0, "a verification batch continues a sequence");
        let x = embed(&self.local.gguf, &self.local.c, tokens)?;
        let ple = self.ple(tokens)?;
        self.history.extend_from_slice(tokens);
        let pos = self.pos;
        self.local.drain(|d| pass_on(self.next.as_mut(), d))?;
        let out = self.local.forward_kept(x, ple.as_deref(), pos, n, verify)?;
        self.pos += n;
        self.verified = if verify { n } else { 0 };
        if let Output::Hidden(x) = &out {
            let conn = self.next.as_mut().context("no machine holds the LM head")?;
            let keep = if verify { "verify" } else { "rows" };
            conn.send(&json!({"forward": {"pos": pos, "n": n, "reply": true, "keep": keep}}), bytemuck::cast_slice(x))?;
        }
        let (h, rows, remote) = match out {
            Output::Final(h) => (h, std::mem::take(&mut self.local.rows), false),
            Output::Hidden(_) => {
                let (h, rows) = recv_kept_hidden(self.next.as_mut().unwrap())?;
                (h, rows, true)
            }
        };
        let started = self.local.head_rows_submit(&h)?;
        early(&rows)?;
        let mut tops = self.local.head_rows_collect(&h, &started)?;
        if remote {
            let (hd, _) = self.next.as_mut().unwrap().recv()?;
            tops = merge_rows(tops, decode_kept(&hd)?)?;
        }
        Ok((tops, rows))
    }

    /// Verify `tokens` (the pending token and its drafts): candidates of every row and the residual rows.
    pub fn verify(
        &mut self,
        tokens: &[u32],
        early: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<(Vec<Vec<(f32, u32)>>, Vec<f32>)> {
        self.forward_kept(tokens, true, early)
    }

    /// Keep tokens `0..keep` of the latest verification batch on every machine.
    pub fn rollback(&mut self, keep: usize) -> Result<()> {
        let n = self.verified;
        ensure!(keep >= 1 && keep <= n, "rollback to {keep} of a {n}-token verification batch");
        if keep < n {
            self.local.rollback(keep)?;
            if let Some(conn) = self.next.as_mut() {
                conn.send(&json!({"rollback": keep}), &[])?;
            }
            self.pos -= n - keep;
            self.history.truncate(self.history.len() - (n - keep));
        }
        self.verified = keep;
        Ok(())
    }

    /// `prefill_cached` whose last chunk (up to `max_batch` tokens: the prompt is cut from its end) also returns
    /// its residual rows. Returns the last token's candidates, the reused positions and the rows; `early` gets the
    /// chunk's tokens, the position after it and the rows while the head rows are scored.
    pub fn prefill_rows(
        &mut self,
        ids: &[u32],
        early: &mut dyn FnMut(&[u32], usize, &[f32]) -> Result<()>,
    ) -> Result<(Vec<(f32, u32)>, usize, Vec<f32>)> {
        ensure!(!ids.is_empty(), "empty prompt");
        let live = self.pos > 0 && self.pos < ids.len() && self.history.len() == self.pos && ids.starts_with(&self.history);
        if !live {
            self.pos = 0;
        }
        let start = self.pos;
        let rest = &ids[start..];
        let chunks: Vec<_> = rest.rchunks(self.max_batch).rev().collect();
        for chunk in &chunks[..chunks.len() - 1] {
            self.forward(chunk, false, None)?;
        }
        let last = chunks[chunks.len() - 1];
        let end = self.pos + last.len();
        let (mut tops, rows) = self.forward_kept(last, false, &mut |rows| early(last, end, rows))?;
        Ok((tops.pop().context("missing candidates")?, start, rows))
    }
}
