//! The MTP (next-token prediction) block of Qwen3.8-Flash-Next, on the CPU reference path.
//!
//! The block reads the trunk's hyper-connection residual (all `hc` streams, before the final mixer) at position
//! p - 1 and the embedding of the token at p: `x_s = W_eh [RMS(e) enorm | RMS(h_s) hnorm_s]` for every stream s
//! (`W_eh` is HF's `fc_embedding | fc_hidden`), then one QSA layer with its own 512 experts and both mixers, its own
//! final mixer (`nextn.hc_head`) and the shared LM head, as llama.cpp's `graph_mtp` (b8e9ce8). A chained draft
//! feeds the block its own output residual as `h`. The tensors come from a separate GGUF holding `blk.48.*`
//! (docs/qwen4.md, MTP).
use super::{
    cpu::{matmul, rms_norm, Captures, CpuStage},
    Config,
};
use crate::{gguf::Gguf, qwen::vector};
use anyhow::{ensure, Context, Result};

pub struct CpuMtp {
    c: Config,
    /// The block's layer index (`c.layers`).
    pub il: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    kraw: Vec<f32>,
    pooled: Vec<f32>,
    /// The first position the block has keys for (it attends to nothing before).
    pub first: usize,
}

impl CpuMtp {
    pub fn new(c: &Config, context: usize) -> Self {
        let kv = context * c.kv_heads * c.head_dim;
        Self {
            c: c.clone(),
            il: c.layers,
            k: vec![0.; kv],
            v: vec![0.; kv],
            kraw: vec![0.; context * c.idx_dim],
            pooled: vec![0.; context.div_ceil(c.kpool) * c.idx_dim],
            first: 0,
        }
    }

    /// Run the block over `tokens` at positions `pos..`, token i paired with row i of `h` (the hc-wide residual of
    /// the position before it: the trunk's, zeros before position 0, or the block's own output when chaining).
    /// `emb` holds the token embedding (the trunk GGUF). Returns the block's output residual, `n x hc x d`.
    pub fn forward(
        &mut self,
        mg: &Gguf,
        emb: &Gguf,
        tokens: &[u32],
        h: &[f32],
        pos: usize,
        cap: &mut Option<&mut Captures>,
    ) -> Result<Vec<f32>> {
        let c = self.c.clone();
        let (d, hc, w, il, n) = (c.d, c.hc, c.width(), self.il, tokens.len());
        ensure!(h.len() == n * w, "MTP hidden rows length mismatch");
        let mut record = |name: String, rows: &[f32], width: usize| {
            if let Some(m) = cap.as_mut() {
                m.insert(name, rows[rows.len() - width..].to_vec());
            }
        };
        let t = |name: &str| c.layer_tensor(mg, il, name);
        let enorm = vector(mg, t("nextn.enorm.weight")?)?;
        let hnorm = vector(mg, t("nextn.hnorm.weight")?)?;
        let et = emb.tensor("token_embd.weight").context("missing token_embd")?;
        // [token][stream][e_norm | h_norm_s]
        let mut cat = vec![0f32; n * hc * 2 * d];
        let mut e = vec![0f32; d];
        for (i, &id) in tokens.iter().enumerate() {
            ensure!((id as u64) < et.dims[1], "invalid token {id}");
            crate::qwen::dequant_row(emb, et, id as usize, &mut e)?;
            rms_norm(&mut e, Some(&enorm), c.eps);
            for s in 0..hc {
                let row = &mut cat[(i * hc + s) * 2 * d..(i * hc + s + 1) * 2 * d];
                row[..d].copy_from_slice(&e);
                row[d..].copy_from_slice(&h[(i * hc + s) * d..(i * hc + s + 1) * d]);
                rms_norm(&mut row[d..], Some(&hnorm[s * d..(s + 1) * d]), c.eps);
            }
        }
        record(format!("mtp_enorm-{il}"), &cat[(n * hc - 1) * 2 * d..][..d], d);
        let hn: Vec<f32> = cat[(n - 1) * hc * 2 * d..]
            .chunks_exact(2 * d)
            .flat_map(|r| r[d..].to_vec())
            .collect();
        record(format!("mtp_hnorm-{il}"), &hn, w);
        let mut x = matmul(mg, t("nextn.eh_proj.weight")?, &cat, n * hc);
        record(format!("mtp_eh_proj-{il}"), &x, w);
        let m = CpuStage::hc_mix(mg, &c, &format!("blk.{il}.hc_attn"), &x, n, &format!("-{il}"), &mut record)?;
        let y = CpuStage::attention(
            mg,
            &c,
            il,
            &m.input,
            pos,
            n,
            &mut self.k,
            &mut self.v,
            &mut self.kraw,
            &mut self.pooled,
            self.first,
            &mut record,
        )?;
        CpuStage::combine(&c, &mut x, &y, m.write.as_deref().unwrap(), n);
        record(format!("hc_combine-{il}"), &x, w);
        let m = CpuStage::hc_mix(mg, &c, &format!("blk.{il}.hc_ffn"), &x, n, &format!("-{il}#2"), &mut record)?;
        let y = CpuStage::moe(mg, &c, il, &m.input, n, &mut record)?;
        record(format!("ffn_out-{il}"), &y, d);
        CpuStage::combine(&c, &mut x, &y, m.write.as_deref().unwrap(), n);
        record(format!("h_nextn-{il}"), &x, w);
        Ok(x)
    }

    /// The block's final mixer on the last row of its output residual `x`: the head's input (`d` floats).
    pub fn hidden(&self, mg: &Gguf, x: &[f32], cap: &mut Option<&mut Captures>) -> Result<Vec<f32>> {
        let w = self.c.width();
        let mut record = |name: String, rows: &[f32], width: usize| {
            if let Some(m) = cap.as_mut() {
                m.insert(name, rows[rows.len() - width..].to_vec());
            }
        };
        let m = CpuStage::hc_mix(
            mg,
            &self.c,
            &format!("blk.{}.nextn.hc_head", self.il),
            &x[x.len() - w..],
            1,
            "",
            &mut record,
        )?;
        record("result_norm".into(), &m.input, self.c.d);
        Ok(m.input)
    }

    /// Draft logits of the last row of `x` with the LM head of `head` (rows `0..rows`).
    pub fn logits(&self, mg: &Gguf, head: &Gguf, x: &[f32], cap: &mut Option<&mut Captures>) -> Result<Vec<f32>> {
        let hn = self.hidden(mg, x, cap)?;
        Ok(matmul(head, head.tensor("output.weight").context("missing output.weight")?, &hn, 1))
    }
}

fn argmax(l: &[f32]) -> u32 {
    (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b }) as u32
}

fn read_ids(path: &std::path::Path) -> Result<Vec<u32>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    s.split_whitespace().map(|t| Ok(t.parse()?)).collect()
}

/// The trunk on the CPU reference over a prompt: its residual after the last layer for every token (`n x hc x d`)
/// and the last token's logits.
fn trunk_prompt(g: &Gguf, c: &Config, stage: &mut CpuStage, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut x = super::embed(g, c, tokens)?;
    let ple = super::ple_rows(g, c, &[], tokens)?;
    stage.forward(g, &mut x, ple.as_deref(), 0, tokens.len(), &mut None)?;
    let logits = stage.logits(g, &x, &mut None)?;
    Ok((x, logits))
}

/// The block's input rows for a prompt: zeros, then the trunk residual of every token but the last.
fn shifted(x: &[f32], w: usize) -> Vec<f32> {
    let mut h = vec![0f32; x.len()];
    h[w..].copy_from_slice(&x[..x.len() - w]);
    h
}

#[derive(clap::Subcommand)]
pub enum Cmd {
    /// CPU reference: run the trunk and the MTP block over a prompt, dump the block's last-row intermediates and
    /// draft logits (`tools/dump-activations` format), then `chain` chained draft steps (draft{i}_logits.f32).
    Ref {
        model: std::path::PathBuf,
        mtp: std::path::PathBuf,
        out: std::path::PathBuf,
        #[arg(long)]
        tokens_file: Option<std::path::PathBuf>,
        #[arg(long, default_value_t = 2)]
        chain: usize,
        /// Score drafts with the MTP GGUF's own copy of the LM head instead of the trunk's.
        #[arg(long)]
        head_from_mtp: bool,
    },
    /// Compare the fast CPU draft (`draft::Draft`) with the reference block on the same inputs: the prompt, then
    /// `chain` chained steps; prints the relative error and cosine of the draft logits (head rows below `vocab`).
    DraftCheck {
        model: std::path::PathBuf,
        mtp: std::path::PathBuf,
        #[arg(long)]
        tokens_file: Option<std::path::PathBuf>,
        #[arg(long, default_value_t = 3)]
        chain: usize,
        #[arg(long, default_value_t = 65536)]
        vocab: usize,
        #[arg(long, default_value_t = 0)]
        threads: usize,
    },
    /// Time draft steps of the fast CPU draft after `start` positions (random tokens and residuals): batches of
    /// each size in `batch` with the head on the last row, `steps` times each.
    DraftBench {
        model: std::path::PathBuf,
        mtp: std::path::PathBuf,
        #[arg(long, default_value_t = 4096)]
        context: usize,
        #[arg(long, default_value_t = 512)]
        start: usize,
        #[arg(long, default_value_t = 16)]
        steps: usize,
        #[arg(long, value_delimiter = ',', default_value = "1,2,3,4")]
        batch: Vec<usize>,
        #[arg(long, default_value_t = 65536)]
        vocab: usize,
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// Leave the first head rows to the GPUs (`Speculator::use_local_head`): time the CPU's share only.
        #[arg(long, default_value_t = 0)]
        skip_head_rows: usize,
    },
    /// Speculative decoding on the chain: for every prompt, greedy decoding without and with MTP drafts, `runs` times
    /// alternating; the outputs must be identical. Prints one JSON line per run.
    Spec {
        model: std::path::PathBuf,
        #[command(flatten)]
        run: super::cli::Run,
        /// The MTP GGUF (blk.{layers}); the driver drafts on its CPU.
        #[arg(long)]
        mtp: std::path::PathBuf,
        /// Drafts per round at most (each round drafts as many as the recent acceptance pays for).
        #[arg(long, default_value_t = 3)]
        draft_tokens: usize,
        /// Draft head rows: tokens below this id.
        #[arg(long, default_value_t = 65536)]
        draft_vocab: usize,
        #[arg(long, default_value_t = 0)]
        draft_threads: usize,
        #[arg(long, default_value_t = 128)]
        tg: usize,
        #[arg(long, default_value_t = 1)]
        runs: usize,
        /// Prompts (token id files); default the 14-token reference prompt.
        prompts: Vec<std::path::PathBuf>,
    },
    /// Draft again along a greedy run saved by `accept` (its tokens and trunk residual rows), with the fast CPU
    /// draft (`--vocab` head rows, `--window` prompt tokens) or the reference block; writes the same JSON.
    Replay {
        model: std::path::PathBuf,
        mtp: std::path::PathBuf,
        run: std::path::PathBuf,
        out: std::path::PathBuf,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value_t = 128)]
        window: usize,
        #[arg(long, default_value_t = 65536)]
        vocab: usize,
        #[arg(long)]
        reference: bool,
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// Simulate speculative rounds of `depth` drafts (fast draft) in which the block's rows at accepted drafts
        /// keep the chained drafts' values (no rewrite with the trunk's rows); prints tokens per round.
        #[arg(long)]
        rounds: bool,
    },
    /// Greedy decoding on the CPU reference, drafting `depth` tokens with the MTP block before every step; writes
    /// the generated ids and each step's drafts as JSON (acceptance of greedy drafts).
    Accept {
        model: std::path::PathBuf,
        mtp: std::path::PathBuf,
        out: std::path::PathBuf,
        #[arg(long)]
        tokens_file: std::path::PathBuf,
        #[arg(long, default_value_t = 256)]
        steps: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        /// Run the block over only the last `window` prompt tokens (it attends to nothing before them).
        #[arg(long)]
        window: Option<usize>,
    },
}

/// Prompt rows `accept` saves before the generated ones (enough for a `replay` window of 128 tokens).
const SAVED_ROWS: usize = 129;

fn threads(n: usize) -> usize {
    if n > 0 {
        n
    } else {
        std::thread::available_parallelism().map_or(8, |n| n.get())
    }
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Spec {
            model,
            run,
            mtp,
            draft_tokens,
            draft_vocab,
            draft_threads,
            tg,
            runs,
            prompts,
        } => {
            let mut chain = run.chain(&model)?;
            chain.enable_verify(draft_tokens + 1)?;
            let t0 = std::time::Instant::now();
            let draft = super::draft::Draft::new(&model, &mtp, chain.context, draft_vocab, threads(draft_threads))?;
            eprintln!("draft block loaded in {:.1} s", t0.elapsed().as_secs_f64());
            let mut spec = super::spec::Speculator::new(draft, draft_tokens);
            spec.use_local_head(&chain);
            let prompts: Vec<(String, Vec<u32>)> = if prompts.is_empty() {
                vec![("reference".into(), super::cli::REFERENCE.to_vec())]
            } else {
                prompts
                    .iter()
                    .map(|p| Ok((p.file_name().unwrap_or_default().to_string_lossy().into_owned(), read_ids(p)?)))
                    .collect::<Result<_>>()?
            };
            let mut failed = false;
            // Experiment: extra passes with session 4's pooled acceptance and with fixed rounds.
            let ab = std::env::var("TRASHCAN_MTP_AB").unwrap_or_default();
            for i in 0..runs {
                for (name, ids) in &prompts {
                    let (plain, pre0, dec0) = super::spec::greedy(&mut chain, None, ids, tg)?;
                    let mut row = serde_json::json!({"run": i, "prompt": name, "prompt_tokens": ids.len(), "tg": tg,
                        "plain_tok_s": (tg - 1) as f64 / dec0, "plain_prefill_s": pre0,
                        "draft_tokens": draft_tokens, "draft_vocab": spec.draft.vocab()});
                    for label in ["mtp", "pooled", "fixed"] {
                        if label != "mtp" && !ab.contains(label) {
                            continue;
                        }
                        spec.set_pooled(label == "pooled");
                        spec.fixed = label == "fixed";
                        let (fast, pre1, dec1) = super::spec::greedy(&mut chain, Some(&mut spec), ids, tg)?;
                        let same = plain == fast;
                        failed |= !same;
                        let st = &spec.stats;
                        let r = st.rounds.max(1) as f64;
                        row[label] = serde_json::json!({"identical": same,
                            "first_difference": plain.iter().zip(&fast).position(|(a, b)| a != b),
                            "tok_s": (tg - 1) as f64 / dec1, "prefill_s": pre1, "stats": st,
                            "tokens_per_round": (tg - 1) as f64 / r,
                            "accepted_per_draft": st.accepted as f64 / st.drafted.max(1) as f64,
                            "shadow_accepted": st.shadow_hits as f64 / st.by_k[0].max(1) as f64,
                            "ms_per_round": {"draft": st.draft_ms / r, "verify": st.verify_ms / r,
                                "rollback": st.rollback_ms / r},
                            "ids": fast});
                    }
                    spec.set_pooled(false);
                    spec.fixed = false;
                    row["plain_ids"] = serde_json::json!(plain);
                    println!("{row}");
                }
            }
            ensure!(!failed, "speculative output differs from greedy decoding");
            Ok(())
        }
        Cmd::Replay {
            model,
            mtp,
            run,
            out,
            depth,
            window,
            vocab,
            reference,
            threads: th,
            rounds,
        } => {
            let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&run)?)?;
            let ids = |v: &serde_json::Value| -> Vec<u32> {
                v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default()
            };
            let (prompt, gen) = (ids(&r["prompt"]), ids(&r["tokens"]));
            let from = r["rows_from"].as_u64().context("run without rows")? as usize;
            let g = Gguf::open(&model)?;
            let mg = Gguf::open(&mtp)?;
            let c = Config::new(&g)?;
            let w = c.width();
            let rows: Vec<f32> = std::fs::read(run.with_extension("rows.f32"))?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let n = prompt.len();
            let row = |p: usize| &rows[(p - from) * w..(p - from + 1) * w];
            let first = n - window.clamp(2, n.min(SAVED_ROWS)) + 1;
            let ctx = n + gen.len() + depth + 2;
            let mut fast = if reference {
                None
            } else {
                Some(super::draft::Draft::new(&model, &mtp, ctx, vocab, threads(th))?)
            };
            let mut block = CpuMtp::new(&c, ctx);
            block.first = first;
            let hin: Vec<f32> = (first..n).flat_map(|p| row(p - 1).to_vec()).collect();
            match fast.as_mut() {
                Some(f) => {
                    f.first = first;
                    f.forward(&prompt[first..], &hin, first, false)?;
                }
                None => {
                    block.forward(&mg, &g, &prompt[first..], &hin, first, &mut None)?;
                }
            }
            if rounds {
                let f = fast.as_mut().context("rounds need the fast draft")?;
                let (mut i, mut count) = (0, 0);
                while i + 1 < gen.len() {
                    let pos = n + i;
                    let (mut h, mut tok) = (row(pos - 1).to_vec(), gen[i]);
                    let mut a = 0;
                    let mut ok = true;
                    for j in 0..depth {
                        let (y, b) = f.forward(&[tok], &h, pos + j, true)?;
                        tok = b.context("head")?.0;
                        ok = ok && gen.get(i + 1 + j) == Some(&tok);
                        a += ok as usize;
                        h = y;
                    }
                    count += 1;
                    i += (a + 1).min(gen.len() - 1 - i);
                }
                let o = serde_json::json!({"depth": depth, "vocab": vocab, "window": window, "rounds": count,
                    "tokens": gen.len() - 1, "tokens_per_round": (gen.len() - 1) as f64 / count as f64});
                std::fs::write(&out, o.to_string())?;
                println!("{o}");
                return Ok(());
            }
            let t0 = std::time::Instant::now();
            let mut drafts = Vec::new();
            for i in 0..gen.len() - 1 {
                let pos = n + i;
                let (mut h, mut tok, mut ds): (Vec<f32>, u32, Vec<u32>) = (row(pos - 1).to_vec(), gen[i], Vec::new());
                for j in 0..depth {
                    let (y, best) = match fast.as_mut() {
                        Some(f) => {
                            let (y, b) = f.forward(&[tok], &h, pos + j, true)?;
                            (y, b.context("head")?.0)
                        }
                        None => {
                            let y = block.forward(&mg, &g, &[tok], &h, pos + j, &mut None)?;
                            let l = block.logits(&mg, &g, &y, &mut None)?;
                            let best = argmax(&l[..vocab.min(l.len())]);
                            (y, best)
                        }
                    };
                    ds.push(best);
                    tok = best;
                    h = y;
                }
                drafts.push(ds);
            }
            let draft_s = t0.elapsed().as_secs_f64();
            let o = serde_json::json!({"prompt_tokens": n, "depth": depth, "window": window, "vocab": vocab,
                "reference": reference, "tokens": gen, "drafts": drafts, "draft_s": draft_s});
            std::fs::write(&out, serde_json::to_string(&o)?)?;
            println!("{} steps replayed, {:.3} s per draft step", drafts.len(), draft_s / (drafts.len() * depth).max(1) as f64);
            Ok(())
        }
        Cmd::DraftCheck {
            model,
            mtp,
            tokens_file,
            chain,
            vocab,
            threads: th,
        } => {
            let g = Gguf::open(&model)?;
            let mg = Gguf::open(&mtp)?;
            let c = Config::new(&g)?;
            let w = c.width();
            let tokens = match tokens_file {
                Some(f) => read_ids(&f)?,
                None => super::cli::REFERENCE.to_vec(),
            };
            let n = tokens.len();
            let ctx = n + chain + 2;
            let mut stage = CpuStage::new(&c, 0, c.layers, ctx, true);
            let (x, logits) = trunk_prompt(&g, &c, &mut stage, &tokens)?;
            let mut block = CpuMtp::new(&c, ctx);
            let mut fast = super::draft::Draft::new(&model, &mtp, ctx, vocab, threads(th))?;
            let vocab = fast.vocab();
            let report = |what: &str, r: &[f32], f: &[f32]| {
                let (rel, cos) = super::cli::metrics(f, &r[..vocab]);
                let (ar, af) = (argmax(&r[..vocab]), argmax(f));
                println!("{what}: draft logits rel {rel:.3e} cos {cos:.6}, argmax {ar} / {af}");
            };
            let hin = shifted(&x, w);
            let y = block.forward(&mg, &g, &tokens, &hin, 0, &mut None)?;
            let (yf, _) = fast.forward(&tokens, &hin, 0, false)?;
            let (rr, _) = super::cli::metrics(&yf, &y);
            println!("prompt ({n} tokens): output residual rel {rr:.3e}");
            report("prompt", &block.logits(&mg, &g, &y, &mut None)?, &fast.logits(&yf));
            let mut h = x[(n - 1) * w..].to_vec();
            let mut tok = argmax(&logits);
            for i in 0..chain {
                let y = block.forward(&mg, &g, &[tok], &h, n + i, &mut None)?;
                let (yf, best) = fast.forward(&[tok], &h, n + i, true)?;
                let r = block.logits(&mg, &g, &y, &mut None)?;
                report(&format!("step {i}"), &r, &fast.logits(&yf));
                ensure!(best.map(|b| b.0) == Some(argmax(&fast.logits(&yf))), "head argmax mismatch");
                tok = argmax(&r);
                h = y;
            }
            Ok(())
        }
        Cmd::DraftBench {
            model,
            mtp,
            context,
            start,
            steps,
            batch,
            vocab,
            threads: th,
            skip_head_rows,
        } => {
            let t0 = std::time::Instant::now();
            let mut fast = super::draft::Draft::new(&model, &mtp, context, vocab, threads(th))?;
            fast.skip_head_rows(skip_head_rows);
            let load_s = t0.elapsed().as_secs_f64();
            let g = Gguf::open(&model)?;
            let c = Config::new(&g)?;
            let w = c.width();
            let mut seed = 12345u64;
            let mut rnd = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.
            };
            let toks = |p: usize| (1000 + (p as u32 * 7919) % 50000) as u32;
            let mut pos = 0;
            let t0 = std::time::Instant::now();
            while pos < start {
                let m = 64.min(start - pos);
                let ids: Vec<u32> = (pos..pos + m).map(toks).collect();
                let h: Vec<f32> = (0..m * w).map(|_| rnd()).collect();
                fast.forward(&ids, &h, pos, false)?;
                pos += m;
            }
            let prefill_s = t0.elapsed().as_secs_f64();
            let mut out = Vec::new();
            for &b in &batch {
                let mut times = Vec::new();
                fast.prof = [0.; 8];
                for s in 0..steps + 1 {
                    if s == 1 {
                        fast.prof = [0.; 8];
                    }
                    let ids: Vec<u32> = (pos..pos + b).map(toks).collect();
                    let h: Vec<f32> = (0..b * w).map(|_| rnd()).collect();
                    let t1 = std::time::Instant::now();
                    fast.forward(&ids, &h, pos, true)?;
                    if s > 0 {
                        times.push(t1.elapsed().as_secs_f64() * 1e3);
                    }
                }
                times.sort_by(f64::total_cmp);
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                let phases: Vec<f64> = fast.prof.iter().map(|v| v / steps as f64).collect();
                out.push(serde_json::json!({"batch": b, "mean_ms": mean, "p50_ms": times[times.len() / 2],
                    "min_ms": times[0], "max_ms": times[times.len() - 1],
                    "phases_ms": {"input": phases[0], "mixer1": phases[1], "attention": phases[2], "mixer2": phases[3],
                        "experts": phases[4], "head": phases[5], "attn_proj": phases[6], "attn_until_o": phases[7]}}));
            }
            let (step_bytes, all_bytes) = fast.bytes();
            println!(
                "{}",
                serde_json::json!({"threads": fast.pool.threads(), "vocab": fast.vocab(), "context": context,
                    "start": start, "load_s": load_s, "prefill_s": prefill_s, "prefill_tok_s": start as f64 / prefill_s,
                    "step_mib": step_bytes as f64 / 1048576., "weights_mib": all_bytes as f64 / 1048576.,
                    "steps": out})
            );
            Ok(())
        }
        Cmd::Ref {
            model,
            mtp,
            out,
            tokens_file,
            chain,
            head_from_mtp,
        } => {
            let g = Gguf::open(&model)?;
            let mg = Gguf::open(&mtp)?;
            let c = Config::new(&g)?;
            let tokens = match tokens_file {
                Some(f) => read_ids(&f)?,
                None => super::cli::REFERENCE.to_vec(),
            };
            let n = tokens.len();
            let ctx = n + chain + 2;
            let mut stage = CpuStage::new(&c, 0, c.layers, ctx, true);
            let t0 = std::time::Instant::now();
            let (x, logits) = trunk_prompt(&g, &c, &mut stage, &tokens)?;
            let next = argmax(&logits);
            eprintln!("trunk: {n} tokens, {:.2} s, next token {next}", t0.elapsed().as_secs_f64());
            let head = if head_from_mtp { &mg } else { &g };
            let mut block = CpuMtp::new(&c, ctx);
            let mut caps = Captures::new();
            let t0 = std::time::Instant::now();
            let mut y = block.forward(&mg, &g, &tokens, &shifted(&x, c.width()), 0, &mut Some(&mut caps))?;
            let draft = block.logits(&mg, head, &y, &mut Some(&mut caps))?;
            eprintln!("MTP over the prompt: {:.2} s", t0.elapsed().as_secs_f64());
            caps.insert("result_output".into(), draft.clone());
            std::fs::create_dir_all(&out)?;
            let mut index = String::new();
            for (name, v) in &caps {
                std::fs::write(out.join(format!("{name}.f32")), bytemuck::cast_slice(v))?;
                index.push_str(&format!("{name}\t{},1,1,1\t{}\n", v.len(), v.len()));
            }
            std::fs::write(out.join("index.tsv"), index)?;
            // The draft for the token after `next`: the block at position n on (next, trunk residual n - 1), then
            // chained steps on the block's own output.
            let w = c.width();
            let mut h = x[(n - 1) * w..].to_vec();
            let mut tok = next;
            let mut drafts = Vec::new();
            for i in 0..chain {
                y = block.forward(&mg, &g, &[tok], &h, n + i, &mut None)?;
                let l = block.logits(&mg, head, &y, &mut None)?;
                std::fs::write(out.join(format!("draft{i}_logits.f32")), bytemuck::cast_slice(&l))?;
                tok = argmax(&l);
                drafts.push(tok);
                h = y;
            }
            std::fs::write(
                out.join("drafts.json"),
                serde_json::json!({"prompt_next": next, "prompt_draft": argmax(&draft), "chain": drafts}).to_string(),
            )?;
            println!("next {next}, draft over the prompt {}, chained drafts {drafts:?}", argmax(&draft));
            Ok(())
        }
        Cmd::Accept {
            model,
            mtp,
            out,
            tokens_file,
            steps,
            depth,
            window,
        } => {
            let g = Gguf::open(&model)?;
            let mg = Gguf::open(&mtp)?;
            let c = Config::new(&g)?;
            let w = c.width();
            let tokens = read_ids(&tokens_file)?;
            let n = tokens.len();
            let ctx = n + steps + depth + 2;
            let mut stage = CpuStage::new(&c, 0, c.layers, ctx, true);
            let mut block = CpuMtp::new(&c, ctx);
            let t0 = std::time::Instant::now();
            let (x, logits) = trunk_prompt(&g, &c, &mut stage, &tokens)?;
            let first = n - window.unwrap_or(n).clamp(1, n);
            block.first = first;
            block.forward(&mg, &g, &tokens[first..], &shifted(&x, w)[first * w..], first, &mut None)?;
            let prefill_s = t0.elapsed().as_secs_f64();
            eprintln!("prompt: {n} tokens, {prefill_s:.1} s");
            let mut h = x[(n - 1) * w..].to_vec();
            let mut tok = argmax(&logits);
            let mut history = tokens.clone();
            // The trunk's residual rows from position n - SAVED_ROWS on, for `replay`.
            let from = n.saturating_sub(SAVED_ROWS);
            let mut rows = x[from * w..].to_vec();
            let (mut gen, mut drafts) = (Vec::new(), Vec::new());
            let (mut target_s, mut draft_s) = (0f64, 0f64);
            let eos = [248044u32, 248046];
            for step in 0..steps {
                let pos = n + step;
                let t1 = std::time::Instant::now();
                let (mut hh, mut dt, mut ds) = (h.clone(), tok, Vec::new());
                for j in 0..depth {
                    let y = block.forward(&mg, &g, &[dt], &hh, pos + j, &mut None)?;
                    dt = argmax(&block.logits(&mg, &g, &y, &mut None)?);
                    ds.push(dt);
                    hh = y;
                }
                draft_s += t1.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                let mut x1 = super::embed(&g, &c, &[tok])?;
                let ple = super::ple_rows(&g, &c, &history, &[tok])?;
                stage.forward(&g, &mut x1, ple.as_deref(), pos, 1, &mut None)?;
                let next = argmax(&stage.logits(&g, &x1, &mut None)?);
                target_s += t1.elapsed().as_secs_f64();
                gen.push(tok);
                drafts.push(ds);
                history.push(tok);
                rows.extend_from_slice(&x1);
                h = x1;
                tok = next;
                if eos.contains(&gen[gen.len() - 1]) {
                    break;
                }
            }
            gen.push(tok);
            let r = serde_json::json!({"prompt_tokens": n, "depth": depth, "window": window, "tokens": gen, "drafts": drafts,
                "prefill_s": prefill_s, "target_s": target_s, "draft_s": draft_s, "prompt": tokens, "rows_from": from});
            std::fs::write(&out, serde_json::to_string(&r)?)?;
            std::fs::write(out.with_extension("rows.f32"), bytemuck::cast_slice(&rows))?;
            let ok = (0..drafts.len()).filter(|&i| gen.get(i + 1) == Some(&drafts[i][0])).count();
            println!(
                "{} steps, first draft right {ok}/{} ({:.3}); {:.2} s per target step, {:.3} s per draft step",
                drafts.len(),
                drafts.len(),
                ok as f64 / drafts.len().max(1) as f64,
                target_s / drafts.len().max(1) as f64,
                draft_s / (drafts.len() * depth).max(1) as f64
            );
            Ok(())
        }
    }
}
