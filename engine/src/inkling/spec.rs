//! Speculative decoding with the DSpark drafter: draft a block, verify it in one forward, keep the accepted prefix.
use super::group::Group;
use crate::generation::Sampler;
use anyhow::Result;

/// One round after `pending` (sampled, not yet in the model): draft up to `k` tokens, verify `[pending, drafts...]`
/// and sample the target's token at each row in order, accepting drafts while they equal it. Returns the tokens now
/// committed (`pending` and the accepted drafts) and the next pending token (the target's sample after them). Every
/// emitted token is the target's own sample, so the output follows the target's distribution at any temperature.
pub fn step(g: &mut Group, sampler: &mut Sampler, pending: u32, k: usize) -> Result<(Vec<u32>, u32)> {
    let k = k.min(g.context.saturating_sub(g.pos + 2)).min(super::draft::BLOCK - 1);
    if k == 0 {
        let top = g.forward(&[pending], super::group::Score::Last, None)?;
        return Ok((vec![pending], sampler.sample(&top[0])));
    }
    let drafts = g.draft(pending, k)?;
    let mut tokens = vec![pending];
    tokens.extend_from_slice(&drafts);
    let cands = g.verify(&tokens)?;
    let mut out = vec![pending];
    let mut m = 0;
    let next = loop {
        let t = sampler.sample(&cands[m]);
        if m < drafts.len() && t == drafts[m] {
            out.push(t);
            m += 1;
        } else {
            break t;
        }
    };
    g.rollback(1 + m)?;
    Ok((out, next))
}
