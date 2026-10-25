//! The qwen4exp machine chain, with the protocol of `qwen::net` (frames `[u32 header length][u64 blob
//! length][JSON header][blob]`; probe, setup and forward requests) over links (`crate::link`: the cable's USB4STREAM
//! when it has one, else TCP). The payload between machines is the hyper-connection residual, `hc * d` floats per
//! token. The driver embeds tokens, gathers the n-gram rows and owns the PLE layer.
use super::{
    cpu::Captures,
    embed,
    gpu::{merge_top, KvCache, Output, POLLED_MAX},
    ngram,
    pipeline::{assemble_logits, head_rows, plan, tl, Done, Local},
    trace, Config,
};
use crate::{
    gguf::Gguf,
    link::{Accepted, Link, Options},
    qwen::pipeline::Devices,
};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    io::{BufReader, ErrorKind, IoSlice, Read, Write},
    net::TcpListener,
    path::Path,
    sync::mpsc,
};

struct Conn {
    r: BufReader<Link>,
    w: Link,
    /// Toward the driver (trace marks only).
    up: bool,
}

impl Conn {
    fn new(link: Link, up: bool) -> Self {
        Self { r: BufReader::with_capacity(1 << 20, link.clone()), w: link, up }
    }
    /// Connect to the next machine (the link retries for up to a minute while it is still starting).
    fn connect(addr: &str) -> Result<Self> {
        Ok(Self::new(Link::connect(addr, "chain")?, false))
    }
    fn send(&mut self, header: &Value, blob: &[u8]) -> Result<()> {
        trace::mark(if self.up { "tx_up0" } else { "tx_dn0" }, blob.len() as u64);
        write_frame(&mut self.w, header, blob)?;
        trace::mark(if self.up { "tx_up1" } else { "tx_dn1" }, blob.len() as u64);
        Ok(())
    }
    fn recv(&mut self) -> Result<(Value, Vec<f32>)> {
        read_frame(&mut self.r, !self.up).map(|(h, b, _)| (h, b))
    }
    /// `recv` for a frame expected shortly (a decode step's reply): watch for it before sleeping.
    fn recv_hot(&mut self) -> Result<(Value, Vec<f32>)> {
        if self.r.buffer().is_empty() {
            self.r.get_ref().spin_readable(HOT);
        }
        self.recv()
    }
}

/// Write a frame: the lengths and the header from one buffer, the blob straight from the caller's memory, in one
/// vectored write when the transport takes it whole.
fn write_frame(w: &mut impl Write, header: &Value, blob: &[u8]) -> Result<()> {
    let h = serde_json::to_vec(header)?;
    let mut head = Vec::with_capacity(12 + h.len());
    head.extend_from_slice(&(h.len() as u32).to_le_bytes());
    head.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    head.extend_from_slice(&h);
    let (mut a, mut b) = (&head[..], blob);
    while !a.is_empty() || !b.is_empty() {
        match w.write_vectored(&[IoSlice::new(a), IoSlice::new(b)]) {
            Ok(0) => return Err(std::io::Error::from(ErrorKind::WriteZero).into()),
            Ok(k) => {
                let first = k.min(a.len());
                (a, b) = (&a[first..], &b[k - first..]);
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// A frame (its blob as 4-byte words: floats, or candidate pairs) and the host time it had arrived completely.
/// `down`: from the next machine (trace marks only).
fn read_frame(r: &mut impl Read, down: bool) -> Result<(Value, Vec<f32>, u64)> {
    let mut len = [0u8; 12];
    r.read_exact(&mut len)?;
    let t0 = trace::now();
    let h = u32::from_le_bytes(len[..4].try_into()?) as usize;
    let b = u64::from_le_bytes(len[4..].try_into()?) as usize;
    ensure!(b % 4 == 0, "frame of {b} bytes");
    let mut header = vec![0u8; h];
    r.read_exact(&mut header)?;
    let mut blob = vec![0f32; b / 4];
    r.read_exact(bytemuck::cast_slice_mut(&mut blob))?;
    let t1 = trace::now();
    trace::mark_at(if down { "rx_dn0" } else { "rx_up0" }, b as u64, t0);
    trace::mark_at(if down { "rx_dn1" } else { "rx_up1" }, b as u64, t1);
    let header: Value = serde_json::from_slice(&header)?;
    if let Some(e) = header.get("error") {
        bail!("remote: {}", e.as_str().unwrap_or("error"));
    }
    Ok((header, blob, t1))
}

type Requests = mpsc::Receiver<Result<(Value, Vec<f32>, u64)>>;

/// Read the previous machine's requests on a thread of their own, so that it can send the next prompt chunk (MiBs)
/// while this machine's GPUs work.
fn requests(link: &Link) -> Requests {
    let mut r = BufReader::with_capacity(1 << 20, link.clone());
    let (tx, rx) = mpsc::sync_channel(4);
    std::thread::spawn(move || loop {
        let m = read_frame(&mut r, false);
        let end = m.is_err();
        // After a decode step the next one is due shortly: watch for it instead of sleeping.
        let step = matches!(&m, Ok((h, _, _)) if decode_step(h));
        if tx.send(m).is_err() || end {
            break;
        }
        if step && r.buffer().is_empty() {
            r.get_ref().spin_readable(HOT);
        }
    });
    rx
}

/// Whether a request belongs to decoding (a decode step's or a verification batch's forward, a rollback after one):
/// the next request follows within a step.
fn decode_step(header: &Value) -> bool {
    header.get("rollback").is_some()
        || header.get("forward").and_then(|f| f["n"].as_u64()).is_some_and(|n| n as usize <= POLLED_MAX)
}

/// How long a thread expecting a decode step's next frame spins before it sleeps: the frames of a decode step follow
/// each other within tens of milliseconds, and a sleeping thread wakes tens of microseconds after its data arrived.
const HOT: std::time::Duration = std::time::Duration::from_millis(100);

/// The next request; a frame expected shortly (`hot`) is watched for before sleeping.
fn next_request(requests: &Requests, hot: bool) -> std::result::Result<Result<(Value, Vec<f32>, u64)>, mpsc::RecvError> {
    if hot {
        let start = std::time::Instant::now();
        while start.elapsed() < HOT {
            match requests.try_recv() {
                Ok(m) => return Ok(m),
                Err(mpsc::TryRecvError::Disconnected) => return Err(mpsc::RecvError),
                Err(mpsc::TryRecvError::Empty) => std::hint::spin_loop(),
            }
        }
    }
    requests.recv()
}

/// Offset of the next machine's clock from this machine's (ns) and the round trip, from the fastest of `k`
/// ping-pongs (`TRASHCAN_Q4_TRACE`).
fn ping(d: &mut Conn, k: usize) -> Result<(i64, i64)> {
    let mut best = (0i64, i64::MAX);
    for _ in 0..k {
        let t1 = trace::now() as i64;
        d.send(&json!({"ping": true}), &[])?;
        let (pong, _) = d.recv()?;
        let t4 = trace::now() as i64;
        let t = |i: usize| pong["pong"][i].as_u64().map(|v| v as i64).context("pong");
        let (t2, t3) = (t(0)?, t(1)?);
        let rtt = (t4 - t1) - (t3 - t2);
        if rtt < best.1 {
            best = (((t2 - t1) + (t3 - t4)) / 2, rtt);
        }
    }
    Ok(best)
}

/// Clock offsets of the machines after this one (relative to this machine's clock) and their ping round trips.
fn clocks(down: Option<&mut Conn>, k: usize) -> Result<Vec<(i64, i64)>> {
    let Some(d) = down else { return Ok(Vec::new()) };
    let (off, rtt) = ping(d, k)?;
    d.send(&json!({"clocks": k}), &[])?;
    let (reply, _) = d.recv()?;
    let mut all = vec![(off, rtt)];
    for o in reply["offsets"].as_array().context("missing offsets")? {
        all.push((off + o[0].as_i64().context("offset")?, o[1].as_i64().context("round trip")?));
    }
    Ok(all)
}

/// How each link from this machine on carries its data: this machine's link to the next (`Link::report`, with
/// `from`), then those after it.
fn links_down(down: Option<&mut Conn>) -> Result<Vec<Value>> {
    let Some(d) = down else { return Ok(Vec::new()) };
    let mut mine = d.w.report();
    mine["from"] = json!(crate::link::hostname());
    d.send(&json!({"links": true}), &[])?;
    let (reply, _) = d.recv()?;
    let mut all = vec![mine];
    all.extend(reply["links"].as_array().context("missing links")?.iter().cloned());
    Ok(all)
}

/// The links' transports by name (`{"mac0->mac1": "stream", ...}`), also written to stderr in one line.
pub fn link_summary(links: &[Value]) -> Value {
    let mut map = serde_json::Map::new();
    let mut line = Vec::new();
    for l in links {
        let name = format!("{}->{}", l["from"].as_str().unwrap_or("?"), l["to"].as_str().unwrap_or("?"));
        let transport = l["transport"].as_str().unwrap_or("?");
        let detail = match l["first_up_ms"].as_f64() {
            Some(ms) if transport == "stream" => format!("up in {ms:.0} ms"),
            Some(ms) => format!("up in {ms:.0} ms after {} failed epochs and {} fallbacks", l["failed_epochs"], l["fallbacks"]),
            None if l["stream_at_both_ends"] == json!(false) => "no stream at both ends".into(),
            None => format!("{} failed epochs", l["failed_epochs"]),
        };
        line.push(format!("{name} {transport} ({detail})"));
        map.insert(name, json!(transport));
    }
    eprintln!("links: {}", line.join(", "));
    Value::Object(map)
}

/// Send a prompt chunk that left this machine's stages on to the next machine. On the machine that ends the model it
/// is done: nobody waits for its candidates.
fn pass_on(next: Option<&mut Conn>, d: Done) -> Result<()> {
    match (d.out, next) {
        (Output::Hidden(x), Some(conn)) => {
            tl("send", 0, d.pos, 0., 0.);
            conn.send(
                &json!({"forward": {"pos": d.pos, "n": d.n, "reply": false, "dump": false}}),
                bytemuck::cast_slice(&x),
            )?;
            tl("sent", 0, d.pos, 0., 0.);
            Ok(())
        }
        (Output::Hidden(_), None) => bail!("no machine holds the LM head"),
        (Output::Final(_), _) => Ok(()),
    }
}

/// Candidates as (score, id) pairs of little-endian words, as frames carry them.
fn put_top(blob: &mut Vec<u8>, top: &[(f32, u32)]) {
    for (s, id) in top {
        blob.extend_from_slice(&s.to_le_bytes());
        blob.extend_from_slice(&id.to_le_bytes());
    }
}

fn get_top(words: &[f32]) -> Vec<(f32, u32)> {
    bytemuck::cast_slice::<f32, u32>(words)
        .chunks_exact(2)
        .map(|p| (f32::from_bits(p[0]), p[1]))
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
    let (header, blob) = down.recv_hot()?;
    let n = header["h"].as_u64().context("missing hidden state")? as usize;
    ensure!(blob.len() == n, "hidden state length mismatch");
    Ok(blob)
}

/// The second frame of a forward reply: the candidates so far (binary pairs) and any captures.
fn result(top: &[(f32, u32)], caps: &Captures) -> (Value, Vec<u8>) {
    let names: Vec<Value> = caps.iter().map(|(n, v)| json!([n, v.len()])).collect();
    let mut blob = Vec::with_capacity(top.len() * 8 + caps.values().map(|v| v.len() * 4).sum::<usize>());
    put_top(&mut blob, top);
    for v in caps.values() {
        blob.extend_from_slice(bytemuck::cast_slice(v));
    }
    (json!({"top": top.len(), "captures": names}), blob)
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

fn decode_result(header: &Value, blob: &[f32], caps: Option<&mut Captures>) -> Result<Vec<(f32, u32)>> {
    let k = header["top"].as_u64().context("missing top candidates")? as usize;
    ensure!(blob.len() >= 2 * k, "candidates length mismatch");
    let top = get_top(&blob[..2 * k]);
    if let Some(m) = caps {
        let mut at = 2 * k;
        for c in header["captures"].as_array().into_iter().flatten() {
            let len = c[1].as_u64().context("capture length")? as usize;
            ensure!(at + len <= blob.len(), "capture length mismatch");
            m.insert(c[0].as_str().context("capture name")?.into(), blob[at..at + len].to_vec());
            at += len;
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
        let peer = stream.peer_addr()?;
        let link = match Link::accept_with(stream, &Options::default()) {
            Ok(Accepted::Link(link, _)) => link,
            // A second path of a link over several cables (a chain link has one).
            Ok(Accepted::Joined(_)) => continue,
            // E.g. a runner's check that the port is open.
            Err(e) => {
                eprintln!("qwen4 node: no session from {peer}: {e:#}");
                continue;
            }
        };
        eprintln!("qwen4 node: session from {peer}");
        let requests = requests(&link);
        let mut up = Conn::new(link, true);
        let mut traced = false;
        if let Err(e) = session(model, &devices, &mut up, &requests, &mut traced) {
            eprintln!("qwen4 node: session ended: {e:#}");
            let _ = up.send(&json!({"error": format!("{e:#}")}), &[]);
        }
        // The reader thread holds the link too: end it for both.
        up.w.shutdown();
        if !traced {
            trace::write("node", Value::Null);
        }
    }
    Ok(())
}

/// `traced`: the driver had this machine write its trace (`Chain::write_traces`).
fn session(model: &Path, devices: &Devices, up: &mut Conn, requests: &Requests, traced: &mut bool) -> Result<()> {
    let mut down: Option<Conn> = None;
    let mut local: Option<Local> = None;
    let mut hot = false;
    loop {
        let waited = std::time::Instant::now();
        let (header, blob, arrived) = match next_request(requests, std::mem::take(&mut hot)) {
            Ok(Ok(m)) => m,
            Ok(Err(e)) if e.downcast_ref::<std::io::Error>().is_none() => return Err(e),
            _ => return Ok(()),
        };
        if let Some(f) = header.get("forward") {
            let pos = f["pos"].as_u64().unwrap_or(0) as usize;
            tl("recv", 0, pos, waited.elapsed().as_secs_f64() * 1e6, 0.);
        }
        // The next decode step follows this one shortly.
        hot = decode_step(&header);
        if header.get("ping").is_some() {
            up.send(&json!({"pong": [arrived, trace::now()]}), &[])?;
        } else if let Some(k) = header.get("clocks") {
            let all = clocks(down.as_mut(), k.as_u64().context("clocks")? as usize)?;
            up.send(&json!({"offsets": all.iter().map(|(o, r)| json!([o, r])).collect::<Vec<_>>()}), &[])?;
        } else if header.get("links").is_some() {
            let all = links_down(down.as_mut())?;
            up.send(&json!({"links": all}), &[])?;
        } else if header.get("trace").is_some() {
            // Write the trace now: the runner may stop the nodes as soon as the driver ends.
            trace::write("node", Value::Null);
            *traced = true;
            if let Some(d) = down.as_mut() {
                d.send(&header, &[])?;
                d.recv()?;
            }
            up.send(&json!({"ok": true}), &[])?;
        } else if let Some(chain) = header.get("probe") {
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
            // The machines after this one load their stages while this one loads its own.
            if let Some(d) = down.as_mut() {
                d.send(
                    &json!({"setup": {"plans": &plans[1..], "heads": &heads[1..], "context": context,
                        "batch": batch, "kv": kv.name()}}),
                    &[],
                )?;
            }
            local = Some(Local::new(model, devices, &mine, &my_heads, context, batch, kv)?);
            if let Some(d) = down.as_mut() {
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
        } else if let Some(m) = header.get("tail") {
            let l = local.as_mut().context("tail before setup")?;
            let rows = if l.ends() {
                l.tail(m.as_u64().context("tail rows")? as usize)?
            } else {
                let d = down.as_mut().context("no machine ends the model")?;
                d.send(&header, &[])?;
                d.recv()?.1
            };
            up.send(&json!({"rows": rows.len()}), bytemuck::cast_slice(&rows))?;
        } else if let Some(f) = header.get("forward").filter(|f| f.get("verify").is_some()) {
            let l = local.as_mut().context("forward before setup")?;
            verify_node(l, down.as_mut(), up, f, blob)?;
        } else if let Some(f) = header.get("forward") {
            let l = local.as_mut().context("forward before setup")?;
            let pos = f["pos"].as_u64().context("pos")? as usize;
            let n = f["n"].as_u64().context("n")? as usize;
            let reply = f["reply"].as_bool().unwrap_or(true);
            let dump = f["dump"].as_bool().unwrap_or(false);
            trace::mark("fwd", pos as u64);
            let mut caps = Captures::new();
            let x = blob;
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
                let (hd, b) = d.recv_hot()?;
                top = merge_top(top, decode_result(&hd, &b, Some(&mut caps))?);
            }
            let (hd, b) = result(&top, &caps);
            up.send(&hd, &b)?;
        } else {
            bail!("unknown request {header}");
        }
    }
}

// A verification batch (`"verify": true`): every machine passes the final hidden states of all rows up and starts
// scoring them with its head rows, then passes up the batch's residual rows from the machine that ends the model (the
// driver's MTP block reads them; 40 KiB a row, so they travel while the head slices score, off the critical path),
// and the candidates of every row follow.

fn send_rows(up: &mut Conn, rows: &[f32]) -> Result<()> {
    up.send(&json!({"rows": rows.len()}), bytemuck::cast_slice(rows))
}

fn recv_rows(down: &mut Conn) -> Result<Vec<f32>> {
    let (header, blob) = down.recv_hot()?;
    let n = header["rows"].as_u64().context("missing residual rows")? as usize;
    ensure!(blob.len() == n, "residual rows length mismatch");
    Ok(blob)
}

/// The candidates of every row of a verification batch (binary pairs, row after row).
fn kept_result(tops: &[Vec<(f32, u32)>]) -> (Value, Vec<u8>) {
    let mut blob = Vec::with_capacity(tops.iter().map(|t| t.len() * 8).sum());
    for t in tops {
        put_top(&mut blob, t);
    }
    (json!({"tops": tops.iter().map(Vec::len).collect::<Vec<_>>()}), blob)
}

fn decode_kept(header: &Value, blob: &[f32]) -> Result<Vec<Vec<(f32, u32)>>> {
    let mut at = 0;
    header["tops"]
        .as_array()
        .context("missing candidates")?
        .iter()
        .map(|k| {
            let k = k.as_u64().context("candidates")? as usize;
            ensure!(at + 2 * k <= blob.len(), "candidates length mismatch");
            at += 2 * k;
            Ok(get_top(&blob[at - 2 * k..at]))
        })
        .collect::<Result<Vec<_>>>()
}

fn merge_rows(a: Vec<Vec<(f32, u32)>>, b: Vec<Vec<(f32, u32)>>) -> Result<Vec<Vec<(f32, u32)>>> {
    ensure!(a.len() == b.len(), "candidate rows mismatch");
    Ok(a.into_iter().zip(b).map(|(x, y)| merge_top(x, y)).collect())
}

fn verify_node(l: &mut Local, down: Option<&mut Conn>, up: &mut Conn, f: &Value, blob: Vec<f32>) -> Result<()> {
    let pos = f["pos"].as_u64().context("pos")? as usize;
    let n = f["n"].as_u64().context("n")? as usize;
    trace::mark("vfy", pos as u64);
    let mut down = down;
    // Prompt chunks still in flight go on first.
    l.drain(|d| pass_on(down.as_deref_mut(), d))?;
    let out = l.verify(blob, None, pos, n)?;
    if let Output::Hidden(x) = &out {
        let d = down.as_mut().context("the last machine has no LM head")?;
        d.send(&json!({"forward": f}), bytemuck::cast_slice(x))?;
    }
    let h = match out {
        Output::Final(h) => {
            down = None;
            h
        }
        Output::Hidden(_) => recv_hidden(down.as_mut().context("the last machine has no LM head")?)?,
    };
    send_hidden(up, &h)?;
    let started = l.head_rows_submit(&h)?;
    let rows = match down.as_mut() {
        None => std::mem::take(&mut l.rows),
        Some(d) => recv_rows(d)?,
    };
    send_rows(up, &rows)?;
    let mut tops = l.head_rows_collect(&h, &started)?;
    if let Some(d) = down {
        let (hd, b) = d.recv_hot()?;
        tops = merge_rows(tops, decode_kept(&hd, &b)?)?;
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
        }
        // Every machine loads its stages at the same time; wait for the others after loading this one's.
        let local = Local::new(model, &devices, &plans[0], &heads[0], context, max_batch, kv)?;
        if let Some(conn) = next.as_mut() {
            conn.recv()?;
        }
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
    /// Every other machine's clock offset from this one's (ns) and the ping round trip, measured along the chain with
    /// `k` ping-pongs per link (`TRASHCAN_Q4_TRACE`).
    pub fn sync_clocks(&mut self, k: usize) -> Result<Vec<(i64, i64)>> {
        clocks(self.next.as_mut(), k)
    }

    /// How every link of the chain carries its data, from this machine's on (see `links_down`).
    pub fn links(&mut self) -> Result<Vec<Value>> {
        links_down(self.next.as_mut())
    }

    /// Have every other machine write its trace now (`TRASHCAN_Q4_TRACE`).
    pub fn write_traces(&mut self) -> Result<()> {
        if let Some(conn) = self.next.as_mut() {
            conn.send(&json!({"trace": true}), &[])?;
            conn.recv()?;
        }
        Ok(())
    }

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
        tl("drv", 0, self.pos, 0., 0.);
        trace::mark("fwd", self.pos as u64);
        let x = embed(&self.local.gguf, &self.local.c, tokens)?;
        let ple = self.ple(tokens)?;
        tl("ple", 0, self.pos, 0., 0.);
        trace::mark("ple", self.pos as u64);
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
            let (hd, b) = conn.recv_hot()?;
            top = merge_top(top, decode_result(&hd, &b, dump.as_deref_mut())?);
        }
        if let Some(m) = dump {
            assemble_logits(m, self.local.c.vocab);
        }
        trace::mark("done", pos as u64);
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

    /// Prefill chunk size for `n` tokens. The last chunk's result is back after (chunks + stages - 1) chunk times, so
    /// short prompts take small chunks and long ones large chunks. Measured on six Macs (12 stages,
    /// results/qwen4/prefill-chunks-20261012): 32-token chunks were best at 256 and 512 tokens, 64 from 1024 to 4096
    /// tokens and 128 at 6144 and 8192; the bands are in tokens per stage after the first, up to the configured batch.
    pub fn chunk_size(&self, n: usize) -> usize {
        let stages: usize = self.plans.iter().map(Vec::len).sum();
        let chunk = match n / stages.saturating_sub(1).max(1) {
            0..12 => 16,
            12..70 => 32,
            70..450 => 64,
            _ => 128,
        };
        chunk.min(self.max_batch)
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

    /// Verify `tokens` (the pending token and its drafts) as one batch that can be rolled back: the candidates of every
    /// row and the batch's residual rows at the end of the model (`n x hc x d`), the MTP block's input.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<(Vec<Vec<(f32, u32)>>, Vec<f32>)> {
        self.verify_with(tokens, || {})
    }

    /// `verify`, running `during` once this machine's stages are done, while the rest of the chain works.
    pub fn verify_with(&mut self, tokens: &[u32], during: impl FnOnce()) -> Result<(Vec<Vec<(f32, u32)>>, Vec<f32>)> {
        let n = tokens.len();
        ensure!(n > 0 && n <= self.max_batch, "batch exceeds configured capacity");
        ensure!(self.pos + n <= self.context, "context exhausted");
        if self.pos == 0 {
            self.history.clear();
        }
        ensure!(self.pos > 0, "a verification batch continues a sequence");
        trace::mark("vfy", self.pos as u64);
        let x = embed(&self.local.gguf, &self.local.c, tokens)?;
        let ple = self.ple(tokens)?;
        trace::mark("ple", self.pos as u64);
        self.history.extend_from_slice(tokens);
        let pos = self.pos;
        self.local.drain(|d| pass_on(self.next.as_mut(), d))?;
        let out = self.local.verify(x, ple.as_deref(), pos, n)?;
        self.pos += n;
        self.verified = n;
        if let Output::Hidden(x) = &out {
            let conn = self.next.as_mut().context("no machine holds the LM head")?;
            conn.send(&json!({"forward": {"pos": pos, "n": n, "reply": true, "verify": true}}), bytemuck::cast_slice(x))?;
        }
        during();
        let remote = matches!(out, Output::Hidden(_));
        let h = match out {
            Output::Final(h) => h,
            Output::Hidden(_) => recv_hidden(self.next.as_mut().unwrap())?,
        };
        let started = self.local.head_rows_submit(&h)?;
        let rows = if remote { recv_rows(self.next.as_mut().unwrap())? } else { std::mem::take(&mut self.local.rows) };
        let mut tops = self.local.head_rows_collect(&h, &started)?;
        if remote {
            let (hd, b) = self.next.as_mut().unwrap().recv_hot()?;
            tops = merge_rows(tops, decode_kept(&hd, &b)?)?;
        }
        trace::mark("done", pos as u64);
        Ok((tops, rows))
    }

    /// The last `m` final residual rows of the latest batch (from the machine that ends the model).
    pub fn tail(&mut self, m: usize) -> Result<Vec<f32>> {
        if self.local.ends() {
            return self.local.tail(m);
        }
        let conn = self.next.as_mut().context("no machine ends the model")?;
        conn.send(&json!({"tail": m}), &[])?;
        Ok(conn.recv()?.1)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> (Conn, Conn) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let accepted = std::thread::spawn(move || match Link::accept_with(listener.accept().unwrap().0, &Options::default()) {
            Ok(Accepted::Link(link, _)) => link,
            _ => panic!("no link"),
        });
        let a = Conn::connect(&address).unwrap();
        let b = Conn::new(accepted.join().unwrap(), true);
        (a, b)
    }

    fn bits(top: &[(f32, u32)]) -> Vec<(u32, u32)> {
        top.iter().map(|(s, id)| (s.to_bits(), *id)).collect()
    }

    #[test]
    fn frames_carry_headers_floats_and_candidates() {
        let (mut a, mut b) = pair();
        let x: Vec<f32> = (0..40960 / 4).map(|i| i as f32 * 0.5 - 7.25).collect();
        a.send(&json!({"forward": {"pos": 3}}), bytemuck::cast_slice(&x)).unwrap();
        let (h, blob) = b.recv().unwrap();
        assert_eq!(h["forward"]["pos"], 3);
        assert_eq!(blob, x);
        // Candidates keep their scores' bits (-0, values JSON would round) and every id.
        let top = vec![(f32::MAX, 0), (1.0e-38, 7), (-0.0, 248319), (0.1, u32::MAX)];
        let mut caps = Captures::new();
        caps.insert("result_output@0".into(), vec![1.5, -2.5]);
        let (hd, blob) = result(&top, &caps);
        b.send(&hd, &blob).unwrap();
        let (hd, blob) = a.recv().unwrap();
        let mut got = Captures::new();
        assert_eq!(bits(&decode_result(&hd, &blob, Some(&mut got)).unwrap()), bits(&top));
        assert_eq!(got["result_output@0"], vec![1.5, -2.5]);
        let rows = vec![top.clone(), Vec::new(), top[1..].to_vec()];
        let (hd, blob) = kept_result(&rows);
        a.send(&hd, &blob).unwrap();
        let (hd, blob) = b.recv().unwrap();
        let back = decode_kept(&hd, &blob).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(bits(&back[0]), bits(&top));
        assert!(back[1].is_empty());
        assert_eq!(bits(&back[2]), bits(&top[1..]));
    }

    #[test]
    fn a_large_frame_survives_partial_writes() {
        // More than the socket buffers hold at once: the vectored send resumes mid-blob.
        let (mut a, mut b) = pair();
        let x: Vec<f32> = (0..(16 << 20) / 4).map(|i| (i % 1000) as f32).collect();
        let reader = std::thread::spawn(move || b.recv().unwrap());
        a.send(&json!({"forward": {"n": 128}}), bytemuck::cast_slice(&x)).unwrap();
        let (h, blob) = reader.join().unwrap();
        assert_eq!(h["forward"]["n"], 128);
        assert_eq!(blob, x);
    }

    #[test]
    fn frames_must_hold_whole_words() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        bytes.extend_from_slice(b"{}abc");
        assert!(read_frame(&mut &bytes[..], false).is_err());
    }
}
