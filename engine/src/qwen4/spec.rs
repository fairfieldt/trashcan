//! Speculative decoding with the MTP block (docs/qwen4.md, MTP): the driver drafts tokens on its CPU
//! (`draft::Draft`), the chain verifies the pending token and the drafts in one batch, the longest matching prefix is
//! kept and every stage rolls its recurrent state back to it. Greedy speculative output equals greedy decoding.
use super::{
    gpu::{merge_top, Output},
    net::Chain,
    pipeline::{Local, Stage},
};
use anyhow::{ensure, Context, Result};

impl Local {
    /// Allocate the state copies verification batches of up to `max` tokens need.
    pub fn enable_verify(&mut self, max: usize) -> Result<()> {
        for s in &mut self.stages {
            if let Stage::Gpu(g) = s {
                g.enable_verify(max)?;
            }
        }
        Ok(())
    }

    /// Run a batch through the local stages, as `forward`; a verification batch (`verify`) keeps what `rollback`
    /// needs, scores every row at the head and keeps the residual rows on the stage that ends the model (in
    /// `rows`); otherwise only the residual rows are kept (the end of a prompt, for the MTP block).
    pub fn forward_kept(
        &mut self,
        mut x: Vec<f32>,
        ple: Option<&[f32]>,
        pos: usize,
        n: usize,
        verify: bool,
    ) -> Result<Output> {
        if let Some(c) = &self.clocks {
            c.busy();
        }
        let g = &self.gguf;
        for stage in self.stages.iter_mut() {
            let out = match stage {
                Stage::Gpu(s) => {
                    let out = if verify { s.verify(&x, ple, pos, n)? } else { s.forward_rows(&x, ple, pos, n)? };
                    if let Output::Final(_) = out {
                        self.rows = s.residual(n)?;
                    }
                    out
                }
                Stage::Cpu(s) => {
                    if verify {
                        s.verify(g, &mut x, ple, pos, n)?;
                    } else {
                        s.forward(g, &mut x, ple, pos, n, &mut None)?;
                    }
                    if s.head {
                        self.rows = x.clone();
                        let w = self.c.width();
                        Output::Final(if verify { s.final_rows(g, &x)? } else { s.final_rows(g, &x[(n - 1) * w..])? })
                    } else {
                        Output::Hidden(std::mem::take(&mut x))
                    }
                }
            };
            match out {
                Output::Hidden(h) => x = h,
                fin => return Ok(fin),
            }
        }
        Ok(Output::Hidden(x))
    }

    /// Keep tokens `0..keep` of the latest verification batch on every local stage.
    pub fn rollback(&mut self, keep: usize) -> Result<()> {
        let g = &self.gguf;
        for s in self.stages.iter_mut() {
            match s {
                Stage::Gpu(s) => s.rollback(keep)?,
                Stage::Cpu(s) => s.rollback(g, keep)?,
            }
        }
        let w = self.c.width();
        self.rows.truncate(keep * w);
        Ok(())
    }

    /// Score final hidden states (`rows x d`) with the head rows of every local stage: merged candidates per row.
    pub fn head_rows(&mut self, h: &[f32]) -> Result<Vec<Vec<(f32, u32)>>> {
        if let Some(c) = &self.clocks {
            c.busy();
        }
        let d = self.c.d;
        let n = h.len() / d;
        let mut tops = vec![Vec::new(); n];
        let mut started = vec![false; self.stages.len()];
        for (stage, s) in self.stages.iter_mut().zip(started.iter_mut()) {
            if let Stage::Gpu(g) = stage {
                *s = g.head_submit(h, false)?;
            }
        }
        for ((stage, rows), started) in self.stages.iter_mut().zip(&self.heads).zip(started) {
            let mine: Vec<Vec<(f32, u32)>> = match stage {
                Stage::Gpu(g) if started => g.head_collect_rows()?,
                Stage::Gpu(_) => continue,
                Stage::Cpu(_) => match rows {
                    Some((a, b)) => h
                        .chunks_exact(d)
                        .map(|row| {
                            let logits = super::cpu::CpuStage::head_logits(&self.gguf, row, *a, *b);
                            crate::qwen::pipeline::top_k(&logits)
                                .into_iter()
                                .map(|(s, id)| (s, id + *a as u32))
                                .collect()
                        })
                        .collect(),
                    None => continue,
                },
            };
            for (t, m) in tops.iter_mut().zip(mine) {
                *t = merge_top(std::mem::take(t), m);
            }
        }
        Ok(tops)
    }
}

/// Counters of a speculative generation.
#[derive(Default, Debug, Clone, serde::Serialize)]
pub struct Stats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub rollback_ms: f64,
}

/// The driver's drafting state over a chain: the MTP block on the CPU and the pending token's inputs.
pub struct Speculator {
    pub draft: super::draft::Draft,
    /// Drafts per round.
    pub k: usize,
    pub stats: Stats,
    /// The residual row of the token before the pending one (the block's `h` for the pending token).
    last: Vec<f32>,
    /// Accepted drafts of the latest round and their trunk residual rows: the next first draft step rewrites the
    /// block's rows there.
    redo_tokens: Vec<u32>,
    redo_rows: Vec<f32>,
}

impl Speculator {
    pub fn new(draft: super::draft::Draft, k: usize) -> Self {
        Self { draft, k, stats: Stats::default(), last: Vec::new(), redo_tokens: Vec::new(), redo_rows: Vec::new() }
    }

    /// Prefill `ids` on the chain (continuing from the live state when it is a prefix, as `Chain::prefill_cached`),
    /// keeping the residual rows of the last chunk, and run the MTP block over that chunk. Returns the last token's
    /// candidates and the reused positions.
    pub fn prefill(&mut self, chain: &mut Chain, ids: &[u32]) -> Result<(Vec<(f32, u32)>, usize)> {
        let (top, start, rows) = chain.prefill_rows(ids)?;
        // The block reads token p with the residual of p - 1: over the chunk's tokens after its first.
        let w = chain.local.c.width();
        let m = rows.len() / w;
        let end = chain.pos;
        let first = end - m + 1;
        self.draft.first = first;
        if m > 1 {
            let toks = &chain.history[first..end];
            self.draft.forward(toks, &rows[..(m - 1) * w], first, false)?;
        }
        self.last = rows[(m - 1) * w..].to_vec();
        self.redo_tokens.clear();
        self.redo_rows.clear();
        Ok((top, start))
    }

    /// One round after the chain has processed every token but `pending`: draft up to `k` tokens, verify them with
    /// `pending` in one batch, keep the matching prefix. Returns the tokens the round produced: the accepted drafts
    /// and the next pending token (greedy), and the candidates of the next pending token.
    pub fn round(&mut self, chain: &mut Chain, pending: u32, k: usize) -> Result<(Vec<u32>, Vec<(f32, u32)>)> {
        let w = chain.local.c.width();
        let pos = chain.pos;
        let k = k.min(chain.context.saturating_sub(pos + 1)).min(super::gpu::VERIFY_MAX - 1);
        let t0 = std::time::Instant::now();
        let mut drafts = Vec::with_capacity(k);
        if k > 0 {
            // The first step also rewrites the block's rows of the previous round's accepted tokens (pending rows).
            let mut toks = std::mem::take(&mut self.redo_tokens);
            toks.push(pending);
            let mut h = std::mem::take(&mut self.redo_rows);
            h.extend_from_slice(&self.last);
            let p0 = pos + 1 - toks.len();
            let (mut y, best) = self.draft.forward(&toks, &h, p0, true)?;
            drafts.push(best.context("draft head")?.0);
            while drafts.len() < k {
                let (y2, best) = self.draft.forward(&[*drafts.last().unwrap()], &y[y.len() - w..], pos + drafts.len(), true)?;
                drafts.push(best.context("draft head")?.0);
                y = y2;
            }
        }
        self.stats.draft_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = std::time::Instant::now();
        let mut batch = vec![pending];
        batch.extend_from_slice(&drafts);
        let (tops, rows) = chain.verify(&batch)?;
        self.stats.verify_ms += t0.elapsed().as_secs_f64() * 1e3;
        ensure!(tops.len() == batch.len() && rows.len() == batch.len() * w, "verification reply size");
        let mut a = 0;
        while a < drafts.len() && tops[a].first().map(|t| t.1) == Some(drafts[a]) {
            a += 1;
        }
        let t0 = std::time::Instant::now();
        if a < drafts.len() {
            chain.rollback(a + 1)?;
        }
        self.stats.rollback_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.stats.rounds += 1;
        self.stats.drafted += drafts.len();
        self.stats.accepted += a;
        let next = tops[a].first().context("missing candidates")?.1;
        // The accepted drafts sit at pos + 1..=pos + a: the next round's first draft step rewrites the block's rows
        // there with the trunk's residuals (rows 0..a), then reads the next pending token with row a.
        self.redo_tokens = drafts[..a].to_vec();
        self.redo_rows = rows[..a * w].to_vec();
        self.last = rows[a * w..(a + 1) * w].to_vec();
        let mut out = drafts[..a].to_vec();
        out.push(next);
        Ok((out, tops[a].clone()))
    }
}

/// Greedy decoding of `n` tokens after `prompt` from position 0, plain (`spec` None) or speculative. Returns the
/// generated ids, the prefill and decode seconds.
pub fn greedy(chain: &mut Chain, spec: Option<&mut Speculator>, prompt: &[u32], n: usize) -> Result<(Vec<u32>, f64, f64)> {
    chain.forget();
    let t0 = std::time::Instant::now();
    let mut out = Vec::with_capacity(n + 4);
    match spec {
        None => {
            let mut top = chain.prefill(prompt)?;
            let pre = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            out.push(top[0].1);
            while out.len() < n {
                top = chain.forward(&[*out.last().unwrap()], true, None)?.context("missing candidates")?;
                out.push(top[0].1);
            }
            Ok((out, pre, t0.elapsed().as_secs_f64()))
        }
        Some(s) => {
            s.stats = Stats::default();
            let (top, _) = s.prefill(chain, prompt)?;
            let pre = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            out.push(top[0].1);
            while out.len() < n {
                let k = s.k.min(n - out.len());
                let (toks, _) = s.round(chain, *out.last().unwrap(), k)?;
                out.extend(toks);
            }
            out.truncate(n);
            Ok((out, pre, t0.elapsed().as_secs_f64()))
        }
    }
}
