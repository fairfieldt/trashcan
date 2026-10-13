//! Speculative decoding with the MTP block (docs/qwen4.md, MTP): the driver drafts tokens on its CPU
//! (`draft::Draft`), the chain verifies the pending token and the drafts in one batch, the longest matching prefix is
//! kept and every stage rolls its recurrent state back to it. Tokens are picked with `Draws`, so speculative output
//! equals plain decoding's, greedy or sampled.
use super::{
    gpu::{merge_top, Output},
    net::Chain,
    pipeline::{Local, Stage},
};
use anyhow::{ensure, Context, Result};

/// How the chain picks tokens, plainly or speculatively: greedily at temperature 0, otherwise a sample of
/// `Sampler::probabilities` (temperature, top-k, top-p over the candidates) by the Gumbel-max trick, with the noise of
/// each token id keyed by the seed and the position being sampled. A draft is picked from the drafter's candidates with
/// the same position's noise and kept when the target's pick equals it, so speculative decoding returns plain
/// decoding's tokens for every seed whatever the drafts (the draft and the sample agree with probability
/// sum_v 1 / sum_w max(p_w / p_v, q_w / q_v), against p(argmax q) for a greedy draft; results/qwen4/mtp-rounds-20261013).
#[derive(Clone, Copy, Debug)]
pub struct Draws {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Draws {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Result<Self> {
        crate::generation::Sampler::new(temperature, top_k, top_p, seed)?;
        Ok(Self { temperature, top_k, top_p, seed })
    }

    /// The token at position `pos` from its candidates (best first).
    pub fn pick(&self, cands: &[(f32, u32)], pos: usize) -> u32 {
        if self.temperature == 0. {
            return cands[0].1;
        }
        let p = crate::generation::Sampler::new(self.temperature, self.top_k, self.top_p, 0)
            .expect("validated sampling parameters")
            .probabilities(cands);
        let mut best = (f64::NEG_INFINITY, p[0].1);
        for &(prob, id) in &p {
            let v = prob.ln() + gumbel(self.seed, pos, id);
            if v > best.0 {
                best = (v, id);
            }
        }
        best.1
    }
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Standard Gumbel noise for token `id` at position `pos` under `seed`.
fn gumbel(seed: u64, pos: usize, id: u32) -> f64 {
    let z = mix(mix(mix(seed.wrapping_add(0x9e3779b97f4a7c15)) ^ pos as u64) ^ id as u64);
    let u = ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    -(-u.ln()).ln()
}

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

    /// Run a verification batch through the local stages, as `forward`: every stage keeps what `rollback` needs, and the
    /// stage that ends the model returns the final hidden state of every row and keeps the residual rows (`rows`).
    pub fn verify(&mut self, mut x: Vec<f32>, ple: Option<&[f32]>, pos: usize, n: usize) -> Result<Output> {
        if let Some(c) = &self.clocks {
            c.busy();
        }
        if self.direct(n) {
            return self.forward_direct(x, ple, pos, n, super::gpu::Mode::Verify);
        }
        let g = &self.gguf;
        for stage in self.stages.iter_mut() {
            let out = match stage {
                Stage::Gpu(s) => {
                    let out = s.verify(&x, ple, pos, n)?;
                    if let Output::Final(_) = out {
                        self.rows = s.residual(n)?;
                    }
                    out
                }
                Stage::Cpu(s) => {
                    s.verify(g, &mut x, ple, pos, n)?;
                    if s.head {
                        self.rows = x.clone();
                        Output::Final(s.final_rows(g, &x)?)
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

    /// Whether a local stage ends the model.
    pub fn ends(&self) -> bool {
        self.stages.iter().any(|s| match s {
            Stage::Gpu(g) => g.last == self.c.layers,
            Stage::Cpu(c) => c.head,
        })
    }

    /// The last `m` final residual rows of the latest batch, from the local stage that ends the model.
    pub fn tail(&mut self, m: usize) -> Result<Vec<f32>> {
        let layers = self.c.layers;
        for s in self.stages.iter_mut() {
            match s {
                Stage::Gpu(g) if g.last == layers => return g.tail(m),
                Stage::Cpu(c) if c.head => return c.tail(m),
                _ => {}
            }
        }
        anyhow::bail!("no local stage ends the model")
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

    /// Start scoring final hidden states (`rows x d`) with the head rows of the GPU stages; which stages started.
    pub fn head_rows_submit(&mut self, h: &[f32]) -> Result<Vec<bool>> {
        if let Some(c) = &self.clocks {
            c.busy();
        }
        let mut started = vec![false; self.stages.len()];
        for (stage, s) in self.stages.iter_mut().zip(started.iter_mut()) {
            if let Stage::Gpu(g) = stage {
                *s = g.head_submit(h, false)?;
            }
        }
        Ok(started)
    }

    /// Finish `head_rows_submit` (CPU stages score here): merged candidates per row.
    pub fn head_rows_collect(&mut self, h: &[f32], started: &[bool]) -> Result<Vec<Vec<(f32, u32)>>> {
        let d = self.c.d;
        let n = h.len() / d;
        let mut tops = vec![Vec::new(); n];
        for ((stage, rows), &started) in self.stages.iter_mut().zip(&self.heads).zip(started) {
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

/// Prompt tokens the MTP block reads (the last ones): with 16 or 128 of a 3K-token document it gave 2.39 and 2.52
/// tokens per round (k = 3) against 2.83 with the whole prompt; each costs ~3 ms on the driver's CPU before the first
/// draft (results/qwen4/mtp-20261012).
const WINDOW: usize = 64;

/// Counters of a speculative generation.
#[derive(Default, Debug, Clone, serde::Serialize)]
pub struct Stats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub rollback_ms: f64,
    /// Rounds by the number of drafts (index 0: plain steps with a shadow draft) and their verification and
    /// rollback milliseconds.
    pub by_k: [usize; super::gpu::VERIFY_MAX],
    pub verify_ms_by_k: [f64; super::gpu::VERIFY_MAX],
    /// Shadow drafts that matched the next token.
    pub shadow_hits: usize,
}

/// Weight older drafts keep at each new one: the acceptance of about the last 16 drafts.
const ACCEPT_DECAY: f64 = 0.94;
/// Weight of the newest round in the running costs.
const COST_WEIGHT: f64 = 0.2;
/// Rounds after which a batch size's cost is measured again (one round of that size), so that sizes the choice
/// avoids keep a current cost (e.g. a shorter context in the next request).
const REFRESH: usize = 256;

/// How many drafts the next round makes: the count with the most expected tokens per millisecond, from the recent
/// acceptance of drafts (`a`, all depths pooled: a round of k drafts yields 1 + a + ... + a^k tokens on average)
/// and the measured running costs of a draft step and of each verification batch size (each size is run once
/// before the first choice, largest first, and again when its cost is `REFRESH` rounds old). With no drafts the
/// round is a plain step (a 1-token verification batch) while the block drafts the next token on the CPU anyway (a
/// shadow draft), so the acceptance stays measured at no cost to the chain and drafting resumes once it pays again.
#[derive(Clone, Debug)]
struct Policy {
    hits: f64,
    tests: f64,
    draft_ms: Option<f64>,
    /// Running cost of each verification batch size (1 to `VERIFY_MAX` tokens) and the rounds since it was measured.
    verify_ms: [Option<f64>; super::gpu::VERIFY_MAX],
    age: [usize; super::gpu::VERIFY_MAX],
}

impl Policy {
    fn new() -> Self {
        const N: usize = super::gpu::VERIFY_MAX;
        Self { hits: 0., tests: 0., draft_ms: None, verify_ms: [None; N], age: [0; N] }.restart()
    }

    /// A new sequence: acceptance back to its prior (0.8, worth 2 drafts); the costs stay.
    fn restart(mut self) -> Self {
        (self.hits, self.tests) = (1.6, 2.);
        self
    }

    fn tested(&mut self, hit: bool) {
        self.hits = self.hits * ACCEPT_DECAY + hit as u8 as f64;
        self.tests = self.tests * ACCEPT_DECAY + 1.;
    }

    fn update(v: &mut Option<f64>, ms: f64) {
        *v = Some(v.map_or(ms, |v| v + COST_WEIGHT * (ms - v)));
    }

    /// A round of `k` drafts took `draft_ms` drafting and `verify_ms` verifying (and rolling back).
    fn measured(&mut self, k: usize, draft_ms: f64, verify_ms: f64) {
        if k > 0 {
            Self::update(&mut self.draft_ms, draft_ms / k as f64);
        }
        Self::update(&mut self.verify_ms[k], verify_ms);
        for a in &mut self.age {
            *a += 1;
        }
        self.age[k] = 0;
    }

    /// Drafts for the next round, at most `max`.
    fn choose(&self, max: usize) -> usize {
        if let Some(k) = (0..=max).rev().find(|&k| self.verify_ms[k].is_none() || self.age[k] > REFRESH) {
            return k;
        }
        let (a, d) = (self.hits / self.tests, self.draft_ms.unwrap_or(0.));
        let (mut best, mut rate, mut tokens, mut p) = (0, f64::MIN, 0., 1.);
        for k in 0..=max {
            tokens += p;
            p *= a;
            let r = tokens / (k as f64 * d + self.verify_ms[k].unwrap_or(f64::MAX));
            if r > rate {
                (best, rate) = (k, r);
            }
        }
        best
    }
}

/// The driver's drafting state over a chain: the MTP block on the CPU and the pending token's inputs.
pub struct Speculator {
    pub draft: super::draft::Draft,
    /// Drafts per round at most.
    pub k: usize,
    pub stats: Stats,
    policy: Policy,
    /// The residual row of the token before the pending one (the block's `h` for the pending token).
    last: Vec<f32>,
    /// Head rows `0..gpu_rows` are scored by the driver's GPU head slices (at the same time as the CPU scores the
    /// rest of the draft vocabulary).
    gpu_rows: usize,
}

impl Speculator {
    pub fn new(draft: super::draft::Draft, k: usize) -> Self {
        Self { draft, k, stats: Stats::default(), policy: Policy::new(), last: Vec::new(), gpu_rows: 0 }
    }

    /// Score the first draft head rows with the driver's GPU head slices when they hold the vocabulary's first rows:
    /// the CPU then ranks only the rows after them.
    pub fn use_local_head(&mut self, chain: &Chain) {
        let mut end = 0;
        let mut ranges: Vec<(usize, usize)> = chain
            .local
            .stages
            .iter()
            .zip(&chain.local.heads)
            .filter_map(|(s, r)| matches!(s, Stage::Gpu(_)).then_some(*r).flatten())
            .collect();
        ranges.sort();
        for (a, b) in ranges {
            if a == end {
                end = b;
            }
        }
        let end = end.min(self.draft.vocab());
        self.draft.skip_head_rows(end);
        self.gpu_rows = end;
    }

    /// One draft step: the block on `tok` at `pos` with input row `h`; the draft for position `pos + 1` (`draws`'
    /// pick from the drafter's candidates) and the output residual.
    fn step(&mut self, chain: &mut Chain, tok: u32, h: &[f32], pos: usize, draws: &Draws) -> Result<(u32, Vec<f32>)> {
        let greedy = draws.temperature == 0. || ab_greedy_drafts();
        if self.gpu_rows == 0 {
            let (y, best) = self.draft.forward(&[tok], h, pos, greedy)?;
            if greedy {
                return Ok((best.context("draft head")?.0, y));
            }
            let top = self.draft.head_top(&self.draft.final_hidden(&y));
            ensure!(!top.is_empty(), "draft head");
            return Ok((draws.pick(&top, pos + 1), y));
        }
        let (y, _) = self.draft.forward(&[tok], h, pos, false)?;
        let hn = self.draft.final_hidden(&y);
        let started = chain.local.head_rows_submit(&hn)?;
        if !greedy {
            let cpu = self.draft.head_top(&hn);
            let gpu = chain.local.head_rows_collect(&hn, &started)?;
            let top = merge_top(cpu, gpu.into_iter().next().unwrap_or_default());
            ensure!(!top.is_empty(), "draft head");
            return Ok((draws.pick(&top, pos + 1), y));
        }
        let cpu = self.draft.head_best(&hn);
        let gpu = chain.local.head_rows_collect(&hn, &started)?;
        let gpu = gpu.first().and_then(|t| t.first()).map(|(s, id)| (*id, *s));
        let best = match (cpu, gpu) {
            (Some(c), Some(g)) => if g.1 >= c.1 { g } else { c },
            (c, g) => c.or(g).context("draft head")?,
        };
        Ok((best.0, y))
    }

    /// Prefill `ids` on the chain (`Chain::prefill_cached`), fetch the final residual rows of its last `WINDOW` tokens
    /// (at most the last chunk's) and run the MTP block over them. Returns the last token's candidates and the reused
    /// positions.
    pub fn prefill(&mut self, chain: &mut Chain, ids: &[u32]) -> Result<(Vec<(f32, u32)>, usize)> {
        let w = chain.local.c.width();
        let (top, start) = chain.prefill_cached(ids)?;
        let end = chain.pos;
        let rows = chain.tail(WINDOW.min(end - start))?;
        let m = rows.len() / w;
        // The block reads token p with the residual of p - 1: the window's tokens after its first.
        let first = end - m + 1;
        self.draft.first = first;
        if m > 1 {
            self.draft.forward(&chain.history[first..end], &rows[..(m - 1) * w], first, false)?;
        }
        self.last = rows[(m - 1) * w..].to_vec();
        self.policy = self.policy.clone().restart();
        Ok((top, start))
    }

    /// One round after the chain has processed every token but `pending` (at position `chain.pos`): draft up to `max`
    /// tokens (as many as `Policy` expects to pay, up to `self.k`), verify them with `pending` in one batch, keep the
    /// drafts that equal `draws`' pick at their row (the picks are plain decoding's, see `Draws`); a pick in `stop`
    /// ends the round. Returns the tokens the round produced (the kept drafts and the next pending token) and the
    /// candidates of the next pending token.
    ///
    /// The block's rows at accepted drafts keep the chained drafts' values: rewriting them with the trunk's rows
    /// raised acceptance by about 3% at k = 3 but cost a 3-token step per round (results/qwen4/mtp-20261012).
    pub fn round(
        &mut self,
        chain: &mut Chain,
        pending: u32,
        max: usize,
        draws: &Draws,
        stop: &[u32],
    ) -> Result<(Vec<u32>, Vec<(f32, u32)>)> {
        let w = chain.local.c.width();
        let pos = chain.pos;
        let k = self.k.min(max).min(chain.context.saturating_sub(pos + 1)).min(super::gpu::VERIFY_MAX - 1);
        let k = self.policy.choose(k);
        if k == 0 {
            return self.shadow_round(chain, pending, draws);
        }
        let t0 = std::time::Instant::now();
        let mut drafts: Vec<u32> = Vec::with_capacity(k);
        let mut h = std::mem::take(&mut self.last);
        while drafts.len() < k {
            let tok = drafts.last().copied().unwrap_or(pending);
            let (draft, y) = self.step(chain, tok, &h, pos + drafts.len(), draws)?;
            drafts.push(draft);
            h = y;
        }
        let draft_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.stats.draft_ms += draft_ms;
        let t0 = std::time::Instant::now();
        let mut batch = vec![pending];
        batch.extend_from_slice(&drafts);
        let (tops, rows) = chain.verify(&batch)?;
        let verify_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.stats.verify_ms += verify_ms;
        ensure!(tops.len() == batch.len() && rows.len() == batch.len() * w, "verification reply size");
        let mut a = 0;
        // Row r predicts position pos + 1 + r.
        let mut next = draws.pick(&tops[0], pos + 1);
        // A stop token is never kept as a draft: it ends the round as the next pending token, unprocessed as in
        // decoding one token at a time.
        while a < drafts.len() && next == drafts[a] && !stop.contains(&next) {
            a += 1;
            next = draws.pick(&tops[a], pos + 1 + a);
        }
        let t0 = std::time::Instant::now();
        if a < drafts.len() {
            chain.rollback(a + 1)?;
        }
        let rollback_ms = t0.elapsed().as_secs_f64() * 1e3;
        self.stats.rollback_ms += rollback_ms;
        self.stats.rounds += 1;
        self.stats.by_k[k] += 1;
        self.stats.drafted += drafts.len();
        self.stats.accepted += a;
        for _ in 0..a {
            self.policy.tested(true);
        }
        if a < drafts.len() {
            // A draft equal to a stop token was right; the round only ended there.
            self.policy.tested(next == drafts[a]);
        }
        self.policy.measured(k, draft_ms, verify_ms + rollback_ms);
        self.stats.verify_ms_by_k[k] += verify_ms + rollback_ms;
        self.last = rows[a * w..(a + 1) * w].to_vec();
        let mut out = drafts[..a].to_vec();
        out.push(next);
        Ok((out, tops[a].clone()))
    }

    /// A round without drafts: `pending` alone through the chain while the block drafts the token after it on the
    /// CPU once this machine's stages are done (its whole draft vocabulary there), which keeps the block's keys
    /// complete and measures the acceptance.
    fn shadow_round(&mut self, chain: &mut Chain, pending: u32, draws: &Draws) -> Result<(Vec<u32>, Vec<(f32, u32)>)> {
        let w = chain.local.c.width();
        let pos = chain.pos;
        let t0 = std::time::Instant::now();
        let (h, draft) = (std::mem::take(&mut self.last), &mut self.draft);
        let mut shadow = Ok(None);
        let (tops, rows) = chain.verify_with(&[pending], || {
            shadow = draft.forward(&[pending], &h, pos, false).map(|(y, _)| {
                let hn = draft.final_hidden(&y);
                if draws.temperature == 0. || ab_greedy_drafts() {
                    draft.best_all(&hn).map(|b| b.0)
                } else {
                    Some(draft.top_all(&hn)).filter(|t| !t.is_empty()).map(|t| draws.pick(&t, pos + 1))
                }
            });
        })?;
        let shadow = shadow?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        ensure!(tops.len() == 1 && rows.len() == w, "verification reply size");
        self.stats.verify_ms += ms;
        self.stats.rounds += 1;
        self.stats.by_k[0] += 1;
        self.policy.measured(0, 0., ms);
        self.stats.verify_ms_by_k[0] += ms;
        let next = draws.pick(&tops[0], pos + 1);
        if let Some(d) = shadow {
            self.policy.tested(d == next);
            self.stats.shadow_hits += (d == next) as usize;
        }
        self.last = rows;
        Ok((vec![next], tops[0].clone()))
    }
}

/// TEMPORARY session A/B switch (removed after the measurement): TRASHCAN_MTP_AB=greedy-drafts drafts the argmax at
/// every temperature, as main did.
fn ab_greedy_drafts() -> bool {
    std::env::var("TRASHCAN_MTP_AB").is_ok_and(|v| v == "greedy-drafts")
}

/// Decoding of `n` tokens after `prompt` from position 0 with `draws`, plain (`spec` None) or speculative. Returns
/// the generated ids, the prefill and decode seconds.
pub fn decode(
    chain: &mut Chain,
    spec: Option<&mut Speculator>,
    prompt: &[u32],
    n: usize,
    draws: &Draws,
) -> Result<(Vec<u32>, f64, f64)> {
    chain.forget();
    let t0 = std::time::Instant::now();
    let mut out = Vec::with_capacity(n + 4);
    match spec {
        None => {
            let mut top = chain.prefill(prompt)?;
            let pre = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            out.push(draws.pick(&top, chain.pos));
            while out.len() < n {
                top = chain.forward(&[*out.last().unwrap()], true, None)?.context("missing candidates")?;
                out.push(draws.pick(&top, chain.pos));
            }
            Ok((out, pre, t0.elapsed().as_secs_f64()))
        }
        Some(s) => {
            s.stats = Stats::default();
            let (top, _) = s.prefill(chain, prompt)?;
            let pre = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            out.push(draws.pick(&top, chain.pos));
            while out.len() < n {
                let (toks, _) = s.round(chain, *out.last().unwrap(), n - out.len(), draws, &[])?;
                out.extend(toks);
            }
            out.truncate(n);
            Ok((out, pre, t0.elapsed().as_secs_f64()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::Sampler;

    /// Candidates (best first) with logits `l`, ids 100, 101, ...
    fn cands(l: &[f32]) -> Vec<(f32, u32)> {
        l.iter().enumerate().map(|(i, v)| (*v, 100 + i as u32)).collect()
    }

    /// Pearson's chi-square of `counts` (by id) against `p`, and its 0.9999 quantile (Wilson-Hilferty).
    fn chi_square(counts: &std::collections::HashMap<u32, usize>, p: &[(f64, u32)], n: usize) -> (f64, f64) {
        assert!(counts.keys().all(|id| p.iter().any(|q| q.1 == *id)), "a pick outside the support");
        let x: f64 = p
            .iter()
            .map(|&(pv, id)| {
                let e = pv * n as f64;
                let o = *counts.get(&id).unwrap_or(&0) as f64;
                (o - e) * (o - e) / e
            })
            .sum();
        let df = (p.len() - 1).max(1) as f64;
        let z = 3.719; // standard normal 0.9999
        let q = df * (1. - 2. / (9. * df) + z * (2. / (9. * df)).sqrt()).powi(3);
        (x, q)
    }

    #[test]
    fn greedy_picks_the_top_candidate() {
        let c = cands(&[3., 2.9, 1.]);
        for pos in 0..10 {
            assert_eq!(Draws::new(0., 64, 0.95, 1).unwrap().pick(&c, pos), 100);
        }
    }

    #[test]
    fn keyed_picks_follow_the_sampler_distribution() {
        let n = 200_000;
        let shapes: [&[f32]; 3] = [
            &[5., 4.5, 4., 3., 1., 0.5, 0., -1.],
            &[10., 2., 1.9, 1.8, 1.7],
            &[1.; 40],
        ];
        for (shape, (t, k, top_p)) in shapes.iter().zip([(1., 64, 0.95), (0.6, 20, 0.95), (0.8, 8, 1.)]) {
            let c = cands(shape);
            let draws = Draws::new(t, k, top_p, 7).unwrap();
            let p = Sampler::new(t, k, top_p, 0).unwrap().probabilities(&c);
            let mut counts = std::collections::HashMap::new();
            for pos in 0..n {
                *counts.entry(draws.pick(&c, pos)).or_insert(0) += 1;
            }
            let (x, q) = chi_square(&counts, &p, n);
            assert!(x < q, "chi-square {x:.1} above {q:.1} for {shape:?}");
        }
    }

    #[test]
    fn positions_and_seeds_draw_independently() {
        // Pairs of consecutive positions: the joint distribution is the product of the marginals.
        let c = cands(&[2., 1.5, 1., 0.]);
        let p = Sampler::new(1., 64, 1., 0).unwrap().probabilities(&c);
        let pairs: Vec<(f64, u32)> =
            p.iter().flat_map(|a| p.iter().map(move |b| (a.0 * b.0, a.1 * 1000 + b.1))).collect();
        let n = 100_000;
        for seed in [1, 2] {
            let draws = Draws::new(1., 64, 1., seed).unwrap();
            let mut counts = std::collections::HashMap::new();
            for i in 0..n {
                *counts.entry(draws.pick(&c, 2 * i) * 1000 + draws.pick(&c, 2 * i + 1)).or_insert(0) += 1;
            }
            let (x, q) = chi_square(&counts, &pairs, n);
            assert!(x < q, "chi-square {x:.1} above {q:.1}");
        }
        // Another seed gives another sequence.
        let (a, b) = (Draws::new(1., 64, 1., 1).unwrap(), Draws::new(1., 64, 1., 2).unwrap());
        assert!((0..64).any(|pos| a.pick(&c, pos) != b.pick(&c, pos)));
    }

    #[test]
    fn a_draft_with_the_same_noise_agrees_as_coupled() {
        // Target p and draft q: the share of positions where both picks agree is
        // sum_v 1 / sum_w max(p_w / p_v, q_w / q_v), above p(argmax q) (a greedy draft) here.
        let (target, draft) = (cands(&[3., 2.6, 2., 1., 0.]), cands(&[2.5, 2.7, 1.5, 1.2, 0.]));
        let draws = Draws::new(1., 64, 1., 3).unwrap();
        let s = Sampler::new(1., 64, 1., 0).unwrap();
        let (p, q) = (s.probabilities(&target), s.probabilities(&draft));
        let prob = |d: &[(f64, u32)], id: u32| d.iter().find(|x| x.1 == id).map_or(0., |x| x.0);
        let expected: f64 = p
            .iter()
            .map(|&(pv, v)| {
                let qv = prob(&q, v);
                1. / p.iter().map(|&(pw, w)| (pw / pv).max(prob(&q, w) / qv)).sum::<f64>()
            })
            .sum();
        let n = 200_000;
        let agree = (0..n).filter(|&pos| draws.pick(&target, pos) == draws.pick(&draft, pos)).count() as f64 / n as f64;
        assert!((agree - expected).abs() < 0.005, "agreement {agree:.4}, expected {expected:.4}");
        assert!(agree > prob(&p, 101) + 0.05, "agreement {agree:.4} against a greedy draft's {:.4}", prob(&p, 101));
    }
}
