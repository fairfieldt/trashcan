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
    },
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
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
            block.forward(&mg, &g, &tokens, &shifted(&x, w), 0, &mut None)?;
            let prefill_s = t0.elapsed().as_secs_f64();
            eprintln!("prompt: {n} tokens, {prefill_s:.1} s");
            let mut h = x[(n - 1) * w..].to_vec();
            let mut tok = argmax(&logits);
            let mut history = tokens.clone();
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
                h = x1;
                tok = next;
                if eos.contains(&gen[gen.len() - 1]) {
                    break;
                }
            }
            gen.push(tok);
            let r = serde_json::json!({"prompt_tokens": n, "depth": depth, "tokens": gen, "drafts": drafts,
                "prefill_s": prefill_s, "target_s": target_s, "draft_s": draft_s});
            std::fs::write(&out, serde_json::to_string(&r)?)?;
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
