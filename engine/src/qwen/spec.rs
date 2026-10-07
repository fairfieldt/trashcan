//! Speculative decoding with the model's own MTP layer. Each round drafts up to `k` tokens on the last
//! machine (chained MTP steps), verifies the pending token and the drafts in one batch through every
//! stage, accepts the matching prefix (rejection sampling above temperature 0, so the output follows the
//! target distribution) and rolls every stage's recurrent state back to the last accepted token.
use super::net::Chain;
use crate::{assistant::verify_prefix, generation::Sampler};
use anyhow::{ensure, Result};
use std::time::Instant;

#[derive(Default)]
pub struct Stats {
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub draft_s: f64,
    pub verify_s: f64,
    pub rollback_s: f64,
    /// Rounds that drafted at least i + 1 tokens, and that accepted at least i + 1.
    pub offered_at: Vec<usize>,
    pub accepted_at: Vec<usize>,
    /// Serial target steps taken by the controller or at the generation/context boundary.
    pub plain_steps: usize,
    pub plain_s: f64,
}

impl Stats {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"rounds": self.rounds, "proposed": self.proposed, "accepted": self.accepted,
            "acceptance": self.accepted as f64 / self.proposed.max(1) as f64,
            "tokens_per_round": (self.rounds + self.accepted) as f64 / self.rounds.max(1) as f64,
            "draft_ms_per_round": self.draft_s * 1e3 / self.rounds.max(1) as f64,
            "verify_ms_per_round": self.verify_s * 1e3 / self.rounds.max(1) as f64,
            "rollback_ms_per_round": self.rollback_s * 1e3 / self.rounds.max(1) as f64,
            "offered_at": self.offered_at, "accepted_at": self.accepted_at,
            "plain_steps": self.plain_steps,
            "plain_ms_per_step": self.plain_s * 1e3 / self.plain_steps.max(1) as f64})
    }
}

/// The caller can emit the final sampled lookahead without processing it. Reserve that token as well
/// as `pending`, so a short remaining budget never drafts or verifies an unused continuation.
fn draft_budget(k: usize, context_left: usize, output_left: usize) -> usize {
    k.min(context_left.saturating_sub(1)).min(output_left.saturating_sub(2))
}

#[derive(Default)]
struct Cost {
    rounds: usize,
    seconds: f64,
    tokens: f64,
}

impl Cost {
    fn observe(&mut self, seconds: f64, tokens: usize) {
        self.rounds += 1;
        let weight = 1. / self.rounds.min(8) as f64;
        self.seconds += weight * (seconds - self.seconds);
        self.tokens += weight * (tokens as f64 - self.tokens);
    }

    fn per_token(&self) -> f64 {
        self.seconds / self.tokens
    }
}

/// Optional, driver-only controller. Decisions depend only on completed rounds, so choosing serial
/// decoding or a draft round cannot condition the current proposal distribution or acceptance test.
/// Compare measured time per committed token, probe both modes, and require a 5% advantage to switch.
/// Bounded by the configured draft count; no extra GPU kernels or peer protocol are needed.
struct Adaptive {
    cost: [Cost; 2],
    preferred: usize,
    until_probe: usize,
    probe: Option<(usize, usize)>,
}

impl Default for Adaptive {
    fn default() -> Self {
        Self { cost: Default::default(), preferred: 1, until_probe: 32, probe: None }
    }
}

impl Adaptive {
    const WARMUP: [usize; 2] = [4, 8];

    fn ready(&self) -> bool {
        self.cost.iter().zip(Self::WARMUP).all(|(c, n)| c.rounds >= n)
    }

    fn choose(&mut self) -> usize {
        for (mode, rounds) in Self::WARMUP.into_iter().enumerate() {
            if self.cost[mode].rounds < rounds {
                return mode;
            }
        }
        if self.until_probe == 0 && self.probe.is_none() {
            let mode = 1 - self.preferred;
            self.probe = Some((mode, Self::WARMUP[mode]));
        }
        self.probe.map_or(self.preferred, |p| p.0)
    }

    fn select(&mut self) {
        let other = 1 - self.preferred;
        if self.cost[other].per_token() < self.cost[self.preferred].per_token() * 0.95 {
            self.preferred = other;
        }
        self.until_probe = 32;
    }

    fn observe(&mut self, mode: usize, seconds: f64, tokens: usize) {
        let ready = self.ready();
        self.cost[mode].observe(seconds, tokens);
        if !ready {
            if self.ready() { self.select(); }
        } else if let Some((probe, left)) = self.probe {
            debug_assert_eq!(mode, probe);
            self.probe = (left > 1).then_some((probe, left - 1));
            if self.probe.is_none() { self.select(); }
        } else {
            self.until_probe = self.until_probe.saturating_sub(1);
        }
    }
}

pub struct Spec {
    /// Drafts per round.
    pub k: usize,
    /// Drafts come from the first `vocab` token ids (0: the whole vocabulary).
    pub vocab: usize,
    pub stats: Stats,
    adaptive: Option<Adaptive>,
}

impl Spec {
    pub fn new(chain: &Chain, k: usize) -> Result<Self> {
        ensure!(chain.mtp, "speculative decoding needs --mtp");
        ensure!(k >= 1 && k < chain.verify, "--draft-tokens must be 1..{}", chain.verify.max(2) - 1);
        let adaptive = (std::env::var("TRASHCAN_QWEN_ADAPTIVE_MTP").as_deref() == Ok("1")).then(Adaptive::default);
        Ok(Self { k, vocab: 0, stats: Stats::default(), adaptive })
    }

    /// Fresh timing observations per request/corpus case; do not carry a previous prompt's acceptance
    /// or thermal conditions into a new request. Cumulative public statistics are unchanged.
    pub fn begin_generation(&mut self) {
        if let Some(a) = self.adaptive.as_mut() { *a = Adaptive::default(); }
    }

    /// One round from `pending`, sampled but not yet processed. Returns the tokens now processed (`pending`
    /// and the accepted drafts) and the next pending token, sampled from the target.
    pub fn step(&mut self, chain: &mut Chain, pending: u32, sampler: &mut Sampler, stop: &[u32], output_left: usize) -> Result<(Vec<u32>, u32)> {
        ensure!(output_left > 1, "a speculative step needs room for its sampled lookahead");
        let cap = draft_budget(self.k, chain.context.saturating_sub(chain.pos), output_left);
        // Short final rounds use another verification width and do not provide comparable costs.
        let adaptive_round = cap == self.k;
        let k = if adaptive_round && self.adaptive.as_mut().is_some_and(|a| a.choose() == 0) { 0 } else { cap };
        let start = Instant::now();
        if k == 0 {
            let top = chain.forward(&[pending], true, None)?.unwrap();
            let next = sampler.sample(&top);
            let seconds = start.elapsed().as_secs_f64();
            self.stats.plain_steps += 1;
            self.stats.plain_s += seconds;
            if adaptive_round {
                if let Some(a) = self.adaptive.as_mut() { a.observe(0, seconds, 1); }
            }
            return Ok((vec![pending], next));
        }
        let seed = if sampler.temperature == 0. { 0 } else { (sampler.uniform() * 9007199254740992.) as u64 };
        let drafts = chain.draft(pending, k, self.vocab, sampler, seed, stop)?;
        let verify = Instant::now();
        let mut tokens = vec![pending];
        tokens.extend(drafts.iter().map(|d| d.0));
        let q: Vec<_> = drafts.iter().map(|d| sampler.probabilities(&d.1)).collect();
        let columns = chain.verify(&tokens)?;
        let rollback = Instant::now();
        let (accepted, next) = verify_prefix(&tokens, &columns, &q, sampler);
        chain.rollback(accepted + 1)?;
        let s = &mut self.stats;
        s.rounds += 1;
        s.proposed += drafts.len();
        s.accepted += accepted;
        s.offered_at.resize(self.k, 0);
        s.accepted_at.resize(self.k, 0);
        for i in 0..drafts.len() {
            s.offered_at[i] += 1;
            s.accepted_at[i] += usize::from(i < accepted);
        }
        s.draft_s += (verify - start).as_secs_f64();
        s.verify_s += (rollback - verify).as_secs_f64();
        s.rollback_s += rollback.elapsed().as_secs_f64();
        tokens.truncate(accepted + 1);
        // An early proposed stop is part of the configured drafting policy. Observe it even when
        // rejected; otherwise repeated false stop proposals could keep a probe pending forever.
        if adaptive_round {
            if let Some(a) = self.adaptive.as_mut() { a.observe(1, start.elapsed().as_secs_f64(), tokens.len()); }
        }
        Ok((tokens, next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_lookahead_needs_no_draft_or_target_evaluation() {
        assert_eq!(draft_budget(2, 100, 2), 0);
        assert_eq!(draft_budget(2, 100, 3), 1);
        assert_eq!(draft_budget(2, 100, 4), 2);
        for k in 1..=7 {
            for context in 1..=16 {
                for output in 2..=16 {
                    let drafts = draft_budget(k, context, output);
                    assert!(drafts <= k);
                    assert!(drafts < context);
                    assert!(drafts + 1 < output);
                }
            }
        }
    }

    #[test]
    fn draft_budget_is_maximal_within_context_and_output_boundaries() {
        for k in 1..=7 {
            for context_left in 0..=10 {
                for output_left in 0..=10 {
                    let cap = draft_budget(k, context_left, output_left);
                    // Independent oracle: enumerate widths that fit the pending target input and
                    // leave a sampled lookahead in the output budget, rather than repeating min/sub.
                    let feasible = (0..=k).filter(|drafts| *drafts < context_left && drafts + 2 <= output_left).max();
                    assert_eq!(cap, feasible.unwrap_or(0), "k={k}, context={context_left}, output={output_left}");
                    if context_left == 0 || output_left <= 1 {
                        assert_eq!(cap, 0); // The caller must not enter Spec::step in these cases.
                    } else {
                        assert!(cap < context_left);
                        assert!(cap + 1 < output_left);
                    }
                }
            }
        }
    }

    #[test]
    fn budgeted_rounds_bound_verification_rollback_and_history_at_context_end() {
        // Model a target that returns input+1, using the real prefix verifier for every possible
        // rejection position. This checks token/state bookkeeping, not GPU snapshot contents.
        const PROMPT: [u32; 3] = [101, 102, 103];
        for k in 1..=7 {
            for context_left in 0..=10 {
                for output_left in 0..=10 {
                    for clip_to_context in [false, true] {
                        for reject_at in 0..=k {
                            let context = PROMPT.len() + context_left;
                            let budget = if clip_to_context { output_left.min(context_left) } else { output_left };
                            let mut sampler = Sampler::new(0., 1, 1., 1).unwrap();
                            let (mut history, mut emitted, mut pending) = (PROMPT.to_vec(), Vec::new(), 1u32);
                            let mut pos = PROMPT.len();
                            // chat::generate checks live position; the server additionally clips the
                            // request's output budget to its initial available context.
                            while emitted.len() < budget && pos < context {
                                let remaining = budget - emitted.len();
                                if remaining == 1 {
                                    emitted.push(pending); // Final lookahead is emitted without a forward.
                                    break;
                                }
                                let cap = draft_budget(k, context - pos, remaining);
                                let mut tokens = vec![pending];
                                for i in 0..cap {
                                    tokens.push(if i == reject_at { 1000 } else { tokens.last().unwrap() + 1 });
                                }
                                let start = pos;
                                pos += tokens.len(); // Target verification temporarily processes every proposal.
                                history.extend_from_slice(&tokens);
                                assert!(pos <= context, "verification exceeded context");
                                assert_eq!(history.len(), pos);
                                let (keep, next) = if cap == 0 {
                                    (1, pending + 1) // Spec::step takes one ordinary target step.
                                } else {
                                    let columns: Vec<_> = tokens.iter().map(|id| vec![(1., id + 1)]).collect();
                                    let q: Vec<_> = tokens[1..].iter().map(|id| vec![(1., *id)]).collect();
                                    let (accepted, next) = verify_prefix(&tokens, &columns, &q, &mut sampler);
                                    (accepted + 1, next)
                                };
                                assert!(keep > 0 && keep <= tokens.len());
                                pos = start + keep;
                                history.truncate(pos); // Chain::rollback discards the unaccepted suffix.
                                emitted.extend_from_slice(&tokens[..keep]);
                                pending = next;
                                assert!(emitted.len() < budget, "a processed chunk consumed the final lookahead");
                                assert_eq!(&history[..PROMPT.len()], &PROMPT);
                                assert_eq!(&history[PROMPT.len()..], emitted);
                                assert_eq!(pending, emitted.len() as u32 + 1);
                            }
                            let count = output_left.min(context_left);
                            assert_eq!(emitted, (1..=count as u32).collect::<Vec<_>>());
                            assert_eq!(history.len(), pos);
                            assert!(pos <= context);
                            if count == 0 {
                                assert_eq!(history, PROMPT);
                                assert_eq!(pending, 1);
                            } else if clip_to_context || output_left <= context_left {
                                assert_eq!(&history[PROMPT.len()..], &emitted[..count - 1]);
                                assert_eq!(pending, *emitted.last().unwrap());
                                assert!(pos < context); // The final emitted token has no KV/history entry.
                            } else {
                                assert_eq!(&history[PROMPT.len()..], emitted);
                                assert_eq!(pos, context);
                                assert_eq!(pending, count as u32 + 1); // Context-clipped lookahead is not emitted.
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn budgeted_rounds_keep_greedy_tokens_and_only_committed_history() {
        // A tiny target whose next token is the input + 1. Reject each possible draft position,
        // including accepting all proposals, across every supported depth and short output budget.
        for k in 1..=7 {
            for budget in 2..=24 {
                for reject_at in 0..=k {
                    let mut sampler = Sampler::new(0., 1, 1., 1).unwrap();
                    let mut emitted = Vec::new();
                    let mut history = Vec::new();
                    let mut pending = 1;
                    while emitted.len() + 1 < budget {
                        let drafts = draft_budget(k, 128 - history.len(), budget - emitted.len());
                        let mut tokens = vec![pending];
                        for i in 0..drafts {
                            tokens.push(if i == reject_at { 1000 } else { tokens.last().unwrap() + 1 });
                        }
                        let columns: Vec<_> = tokens.iter().map(|id| vec![(1., id + 1)]).collect();
                        let q: Vec<_> = tokens[1..].iter().map(|id| vec![(1., *id)]).collect();
                        let (accepted, next) = verify_prefix(&tokens, &columns, &q, &mut sampler);
                        history.extend_from_slice(&tokens[..accepted + 1]);
                        emitted.extend_from_slice(&tokens[..accepted + 1]);
                        pending = next;
                        assert!(history.len() < budget);
                    }
                    emitted.push(pending);
                    assert_eq!(emitted, (1..=budget as u32).collect::<Vec<_>>());
                    assert_eq!(history, emitted[..budget - 1]);
                }
            }
        }
    }

    fn replay(rounds: usize, mut cost: impl FnMut(usize, usize) -> (f64, usize)) -> Vec<usize> {
        let mut controller = Adaptive::default();
        (0..rounds).map(|i| {
            let mode = controller.choose();
            let (seconds, tokens) = cost(i, mode);
            controller.observe(mode, seconds, tokens);
            mode
        }).collect()
    }

    #[test]
    fn adaptive_uses_committed_tokens_and_rechecks_both_modes() {
        let modes = replay(100, |_, mode| if mode == 0 { (0.035, 1) } else { (0.064, 3) });
        assert_eq!(&modes[..4], &[0; 4]);
        assert_eq!(&modes[4..12], &[1; 8]);
        assert!(modes[12..].iter().filter(|&&m| m == 1).count() > 70);

        let modes = replay(100, |_, mode| if mode == 0 { (0.035, 1) } else { (0.064, 1) });
        assert!(modes[12..].iter().filter(|&&m| m == 0).count() > 60);
        assert!(modes[44..].contains(&1), "serial mode must periodically check MTP again");
    }

    #[test]
    fn adaptive_recovers_when_acceptance_changes() {
        let modes = replay(300, |i, mode| if mode == 0 { (0.035, 1) } else { (0.064, if i < 100 { 1 } else { 3 }) });
        assert!(modes[200..].iter().filter(|&&m| m == 1).count() > 80);
    }

    #[test]
    fn adaptive_hysteresis_ignores_small_timing_differences() {
        let modes = replay(100, |_, mode| if mode == 0 { (0.035, 1) } else { (0.071, 2) });
        assert!(modes[12..].iter().filter(|&&m| m == 1).count() > 70);
    }

    #[test]
    fn adaptive_rounds_preserve_sampled_target_mass() {
        let p = vec![(0.6f32.ln(), 1), (0.4f32.ln(), 2)];
        let q = vec![(0.2, 1), (0.8, 2)];
        let columns = vec![p.clone(); 3];
        let mut sampler = Sampler::new(1., 2, 1., 91).unwrap();
        let mut proposer = Sampler::new(1., 2, 1., 72).unwrap();
        let mut controller = Adaptive::default();
        let mut pending = sampler.sample(&p);
        let (mut first, mut count, mut modes) = (0, 0, [0; 2]);
        for i in 0..50000 {
            let mode = controller.choose();
            modes[mode] += 1;
            let (tokens, next) = if mode == 0 {
                (vec![pending], sampler.sample(&p))
            } else {
                let mut tokens = vec![pending, proposer.draw(&q), proposer.draw(&q)];
                let (accepted, next) = verify_prefix(&tokens, &columns, &[q.clone(), q.clone()], &mut sampler);
                tokens.truncate(accepted + 1);
                (tokens, next)
            };
            // Force a sustained cost change as well as acceptance-dependent selection.
            controller.observe(mode, if mode == 0 { 0.035 } else if i < 25000 { 0.080 } else { 0.020 }, tokens.len());
            first += tokens.iter().filter(|&&t| t == 1).count();
            count += tokens.len();
            pending = next;
        }
        assert!(modes.iter().all(|&n| n > 10000), "both modes must contribute substantial output");
        assert!((first as f64 / count as f64 - 0.6).abs() < 0.01);
    }
}
