//! Layer placement and the stages of one machine. The last stage of the last machine owns the final
//! norm, LM head and optional MTP block; every other boundary passes the `n x d` hidden state.
use super::{
    cpu::{Captures, CpuMtp, CpuStage},
    embed,
    gpu::{GpuStage, Output, TOP_K},
    tp::TpGroup,
    Config,
};
use crate::{generation::Sampler, gguf::Gguf, gpu::Instance, link::Lanes, vulkan::Device};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{path::Path, rc::Rc};

/// Where a machine runs its layers: the CPU reference or a list of local GPU indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Devices {
    Cpu,
    Gpus(Vec<usize>),
}
impl Devices {
    pub fn count(&self) -> usize {
        match self {
            Devices::Cpu => 1,
            Devices::Gpus(g) => g.len(),
        }
    }
}

/// Bytes a layer occupies once uploaded: Q4_0 repacks 18-byte blocks into 20 bytes, others stay as stored.
fn layer_bytes(g: &Gguf, il: usize) -> u64 {
    let prefix = format!("blk.{il}.");
    g.tensors
        .iter()
        .filter(|t| t.name.starts_with(&prefix))
        .map(|t| match t.ty {
            crate::gguf::GgmlType::Q4_0 => t.size / 18 * 20,
            _ => t.size,
        })
        .sum()
}

/// Contiguous layer ranges for up to `devices` stages in order, balancing uploaded bytes. The final stage
/// also holds the LM head (Q6_K repacked to 1.25 bytes per weight) and, with `mtp`, the MTP layer: one
/// gated-attention block, an FFN and eh_proj in Q4_0 (0.625 bytes per weight once repacked).
pub fn plan(g: &Gguf, c: &Config, devices: usize, mtp: bool) -> Vec<(usize, usize)> {
    let costs: Vec<f64> = (0..c.layers).map(|il| layer_bytes(g, il) as f64).collect();
    let stages = devices.clamp(1, c.layers);
    let (d, qd, kvd) = (c.d, c.heads * c.head_dim, c.kv_heads * c.head_dim);
    let mtp_weights = d * 2 * qd + 2 * d * kvd + qd * d + 3 * d * c.ff + 2 * d * d;
    let head = (c.vocab * c.d) as f64 * 1.25 + if mtp { mtp_weights as f64 * 0.625 } else { 0. };
    let target = (costs.iter().sum::<f64>() + head) / stages as f64;
    split(&costs, stages, target)
}

/// Cut before a layer when that keeps the running total closest to the next multiple of `target`, or
/// when every remaining layer is needed to give each remaining stage one.
fn split(costs: &[f64], stages: usize, target: f64) -> Vec<(usize, usize)> {
    let mut cuts = vec![0];
    let mut acc = 0.;
    for (il, cost) in costs.iter().enumerate() {
        let next = cuts.len();
        if next < stages && il > *cuts.last().unwrap() {
            let boundary = next as f64 * target;
            let closer = acc + cost > boundary && boundary - acc <= acc + cost - boundary;
            if closer || costs.len() - il == stages - next {
                cuts.push(il);
            }
        }
        acc += cost;
    }
    cuts.push(costs.len());
    cuts.windows(2).map(|w| (w[0], w[1])).collect()
}

pub enum Stage {
    Cpu(CpuStage),
    Gpu(GpuStage),
    /// Every layer split across GPUs and machines; the only stage when present.
    Tp(Box<TpGroup>),
}

/// A drafted token and the MTP's top candidates it was sampled from.
pub type Draft = (u32, Vec<(f32, u32)>);

/// The stages of this machine, run in order; with `head`, the last one ends the model.
pub struct Local {
    pub gguf: Gguf,
    pub c: Config,
    pub stages: Vec<Stage>,
    mtp: bool,
}

impl Local {
    /// `verify` is the largest verification batch that can be rolled back; `mtp` is a GGUF holding the
    /// MTP layer, loaded when this machine ends the model.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        path: &Path,
        devices: &Devices,
        ranges: &[(usize, usize)],
        head: bool,
        context: usize,
        max_batch: usize,
        verify: usize,
        mtp: Option<&Path>,
    ) -> Result<Self> {
        let gguf = Gguf::open(path)?;
        let c = Config::new(&gguf)?;
        ensure!(ranges.len() <= devices.count(), "more stages than devices");
        let mtp = match mtp.filter(|_| head) {
            Some(p) => Some(Gguf::open(p).with_context(|| format!("opening {}", p.display()))?),
            None => None,
        };
        let has_mtp = mtp.is_some();
        let mut stages = Vec::new();
        match devices {
            Devices::Cpu => {
                ensure!(ranges.len() == 1, "the CPU runs a single stage");
                let (first, last) = ranges[0];
                let mut s = CpuStage::new(&c, first, last, context, head);
                if let Some(m) = mtp {
                    s.mtp = Some(CpuMtp::new(m, &c, context)?);
                }
                stages.push(Stage::Cpu(s));
            }
            Devices::Gpus(indices) => {
                let inst = Rc::new(Instance::new()?);
                let all = inst.amd_gpus()?;
                for (i, &(first, last)) in ranges.iter().enumerate() {
                    let pd = *all
                        .get(indices[i])
                        .ok_or_else(|| anyhow::anyhow!("GPU {} not found", indices[i]))?;
                    let is_head = head && i + 1 == ranges.len();
                    let extra = if !is_head {
                        ""
                    } else if has_mtp {
                        " + head + MTP"
                    } else {
                        " + head"
                    };
                    eprintln!("GPU{}: layers {first}..{last}{extra}", indices[i]);
                    let gpu = Device::new(inst.clone(), pd)?;
                    let m = mtp.as_ref().filter(|_| is_head);
                    let stage = GpuStage::new(gpu, &gguf, &c, first, last, is_head, context, max_batch, verify, m)?;
                    eprintln!("GPU{}: {:.1} MiB allocated", indices[i], stage.device_mib());
                    stages.push(Stage::Gpu(stage));
                }
            }
        }
        Ok(Self { gguf, c, stages, mtp: has_mtp })
    }

    /// Tensor parallelism over this machine's `gpus` and, with `peer`, the other machine's (`tp`), with the
    /// MTP block in `mtp` split the same way.
    pub fn tensor_parallel(path: &Path, gpus: &[usize], peer: Option<Lanes>, context: usize, max_batch: usize, verify: usize, mtp: Option<&Path>) -> Result<Self> {
        let group = TpGroup::new(path, gpus, peer, context, max_batch, verify, mtp)?;
        let (gguf, c) = (Gguf::open(path)?, group.c.clone());
        Ok(Self { gguf, c, stages: vec![Stage::Tp(Box::new(group))], mtp: mtp.is_some() })
    }

    /// Run all local stages over the hidden rows of `tokens` (the token ids feed the MTP block). Returns the
    /// hidden state, or the top candidates when this machine ends the model: of the last column, or of every
    /// column with `verify`, which also keeps the per-column states for `rollback`.
    pub fn forward(
        &mut self,
        mut x: Vec<f32>,
        pos: usize,
        tokens: &[u32],
        verify: bool,
        mut dump: Option<&mut Captures>,
    ) -> Result<Output> {
        let n = tokens.len();
        // A tensor-parallel group reads the MTP block's embeddings from its input rows.
        let tp = matches!(self.stages.first(), Some(Stage::Tp(_)));
        let emb = if self.mtp && !tp { Some(embed(&self.gguf, &self.c, tokens)?) } else { None };
        for stage in &mut self.stages {
            let out = match stage {
                Stage::Tp(t) => Output::Top(t.forward(&x, pos, tokens, verify, dump.as_deref_mut())?),
                Stage::Gpu(s) => s.forward(&x, pos, n, verify, emb.as_deref(), dump.as_deref_mut())?,
                Stage::Cpu(s) => {
                    let mut cap = dump.as_deref_mut();
                    s.forward(&self.gguf, &mut x, pos, n, verify, &mut cap)?;
                    if s.head {
                        let cols = if verify { n } else { 1 };
                        let (rows, logits) = s.head(&self.gguf, &x, cols, &mut cap)?;
                        if let Some(m) = cap {
                            m.insert("result_output".into(), logits[cols - 1].clone());
                        }
                        if let (Some(m), Some(e)) = (s.mtp.as_mut(), emb.as_deref()) {
                            m.process(&self.c, e, &rows, pos, n)?;
                        }
                        Output::Top(logits.iter().map(|l| top_k(l)).collect())
                    } else {
                        Output::Hidden(std::mem::take(&mut x))
                    }
                }
            };
            match out {
                Output::Hidden(h) => x = h,
                top => return Ok(top),
            }
        }
        Ok(Output::Hidden(x))
    }

    /// Keep columns `0..=column` of the latest verification batch on every stage.
    pub fn rollback(&mut self, column: usize) -> Result<()> {
        for stage in &mut self.stages {
            match stage {
                Stage::Gpu(s) => s.rollback(column)?,
                Stage::Cpu(s) => s.rollback(column)?,
                Stage::Tp(t) => t.rollback(column)?,
            }
        }
        Ok(())
    }

    pub fn save(&mut self, slot: usize) -> Result<()> {
        for stage in &mut self.stages {
            match stage {
                Stage::Gpu(s) => s.save(slot)?,
                Stage::Cpu(s) => s.save(slot),
                Stage::Tp(t) => t.save(slot)?,
            }
        }
        Ok(())
    }

    pub fn restore(&mut self, slot: usize) -> Result<()> {
        for stage in &mut self.stages {
            match stage {
                Stage::Gpu(s) => s.restore(slot)?,
                Stage::Cpu(s) => s.restore(slot)?,
                Stage::Tp(t) => t.restore(slot)?,
            }
        }
        Ok(())
    }

    /// Chain up to `k` MTP drafts from `token` at `pos`, sampling each from the MTP's candidates among the
    /// first `vocab` tokens (0: all). Stops after a token in `stop`.
    pub fn draft(&mut self, token: u32, pos: usize, k: usize, vocab: usize, sampler: &mut Sampler, stop: &[u32]) -> Result<Vec<Draft>> {
        ensure!(self.mtp, "this machine has no MTP block");
        if let Some(Stage::Tp(t)) = self.stages.last_mut() {
            return t.draft(token, pos, k, vocab, sampler, stop);
        }
        let (g, c) = (&self.gguf, &self.c);
        let mut drafts = Vec::new();
        let mut token = token;
        let mut hidden: Option<Vec<f32>> = None;
        for j in 0..k {
            let emb = embed(g, c, &[token])?;
            let top = match self.stages.last_mut().context("no stages")? {
                Stage::Tp(_) => unreachable!("tensor-parallel drafts return above"),
                Stage::Gpu(s) => s.draft(&emb, pos + j, j == 0, vocab)?,
                Stage::Cpu(s) => {
                    let m = s.mtp.as_mut().context("no MTP block")?;
                    let (logits, h) = m.step(g, c, &emb, hidden.as_deref(), pos + j)?;
                    hidden = Some(h);
                    top_k(&logits[..if vocab == 0 { logits.len() } else { vocab.min(logits.len()) }])
                }
            };
            token = sampler.sample(&top);
            drafts.push((token, top));
            if stop.contains(&token) {
                break;
            }
        }
        Ok(drafts)
    }
}

/// Best TOP_K logits, highest first, ties by lower id (as the GPU merge orders them).
pub fn top_k(logits: &[f32]) -> Vec<(f32, u32)> {
    let mut all: Vec<(f32, u32)> = logits.iter().enumerate().map(|(i, v)| (*v, i as u32)).collect();
    let k = TOP_K.min(all.len());
    all.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    all.truncate(k);
    all.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    all
}

#[cfg(test)]
mod tests {
    #[test]
    fn split_balances_and_leaves_room_for_the_head() {
        // 64 equal layers on four stages, the head worth two layers on the last one.
        let ranges = super::split(&[1.; 64], 4, 66. / 4.);
        assert_eq!(ranges, vec![(0, 16), (16, 33), (33, 49), (49, 64)]);
        assert_eq!(super::split(&[1.; 3], 3, 1.), vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(super::split(&[1.; 8], 1, 8.), vec![(0, 8)]);
    }
    #[test]
    fn top_k_orders_and_breaks_ties_by_id() {
        let mut l = vec![0f32; 100];
        l[7] = 3.;
        l[3] = 3.;
        l[50] = 5.;
        let t = super::top_k(&l);
        assert_eq!(&t[..3], &[(5., 50), (3., 3), (3., 7)]);
        assert_eq!(t.len(), 64);
    }
}
