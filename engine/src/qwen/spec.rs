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
}

impl Stats {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"rounds": self.rounds, "proposed": self.proposed, "accepted": self.accepted,
            "acceptance": self.accepted as f64 / self.proposed.max(1) as f64,
            "tokens_per_round": (self.rounds + self.accepted) as f64 / self.rounds.max(1) as f64,
            "draft_ms_per_round": self.draft_s * 1e3 / self.rounds.max(1) as f64,
            "verify_ms_per_round": self.verify_s * 1e3 / self.rounds.max(1) as f64,
            "rollback_ms_per_round": self.rollback_s * 1e3 / self.rounds.max(1) as f64,
            "offered_at": self.offered_at, "accepted_at": self.accepted_at})
    }
}

pub struct Spec {
    /// Drafts per round.
    pub k: usize,
    /// Drafts come from the first `vocab` token ids (0: the whole vocabulary).
    pub vocab: usize,
    pub stats: Stats,
}

impl Spec {
    pub fn new(chain: &Chain, k: usize) -> Result<Self> {
        ensure!(chain.mtp, "speculative decoding needs --mtp");
        ensure!(k >= 1 && k < chain.verify, "--draft-tokens must be 1..{}", chain.verify.max(2) - 1);
        Ok(Self { k, vocab: 0, stats: Stats::default() })
    }

    /// One round from `pending`, sampled but not yet processed. Returns the tokens now processed (`pending`
    /// and the accepted drafts) and the next pending token, sampled from the target.
    pub fn step(&mut self, chain: &mut Chain, pending: u32, sampler: &mut Sampler, stop: &[u32]) -> Result<(Vec<u32>, u32)> {
        let k = self.k.min((chain.context - chain.pos).saturating_sub(1));
        if k == 0 {
            let top = chain.forward(&[pending], true, None)?.unwrap();
            return Ok((vec![pending], sampler.sample(&top)));
        }
        let start = Instant::now();
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
        Ok((tokens, next))
    }
}
