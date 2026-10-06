//! Layer placement and the stages of one machine. The last stage of the last machine owns the final
//! norm and LM head; every other boundary passes the `n x d` hidden state.
use super::{
    cpu::{Captures, CpuStage},
    gpu::{GpuStage, Output, TOP_K},
    Config,
};
use crate::{gguf::Gguf, gpu::Instance, vulkan::Device};
use anyhow::{ensure, Result};
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
/// also holds the LM head (Q6_K repacked to 1.25 bytes per weight).
pub fn plan(g: &Gguf, c: &Config, devices: usize) -> Vec<(usize, usize)> {
    let costs: Vec<f64> = (0..c.layers).map(|il| layer_bytes(g, il) as f64).collect();
    let stages = devices.clamp(1, c.layers);
    let head = (c.vocab * c.d) as f64 * 1.25;
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
}

/// The stages of this machine, run in order; with `head`, the last one ends the model.
pub struct Local {
    pub gguf: Gguf,
    pub c: Config,
    pub stages: Vec<Stage>,
}

impl Local {
    pub fn new(
        path: &Path,
        devices: &Devices,
        ranges: &[(usize, usize)],
        head: bool,
        context: usize,
        max_batch: usize,
    ) -> Result<Self> {
        let gguf = Gguf::open(path)?;
        let c = Config::new(&gguf)?;
        ensure!(ranges.len() <= devices.count(), "more stages than devices");
        let mut stages = Vec::new();
        match devices {
            Devices::Cpu => {
                ensure!(ranges.len() == 1, "the CPU runs a single stage");
                let (first, last) = ranges[0];
                stages.push(Stage::Cpu(CpuStage::new(&c, first, last, context, head)));
            }
            Devices::Gpus(indices) => {
                let inst = Rc::new(Instance::new()?);
                let all = inst.amd_gpus()?;
                for (i, &(first, last)) in ranges.iter().enumerate() {
                    let pd = *all
                        .get(indices[i])
                        .ok_or_else(|| anyhow::anyhow!("GPU {} not found", indices[i]))?;
                    let is_head = head && i + 1 == ranges.len();
                    eprintln!("GPU{}: layers {first}..{last}{}", indices[i], if is_head { " + head" } else { "" });
                    let gpu = Device::new(inst.clone(), pd)?;
                    let stage = GpuStage::new(gpu, &gguf, &c, first, last, is_head, context, max_batch)?;
                    eprintln!("GPU{}: {:.1} MiB allocated", indices[i], stage.device_mib());
                    stages.push(Stage::Gpu(stage));
                }
            }
        }
        Ok(Self { gguf, c, stages })
    }

    /// Run all local stages. Returns the hidden state, or the top candidates when this machine ends the model.
    pub fn forward(
        &mut self,
        mut x: Vec<f32>,
        pos: usize,
        n: usize,
        mut dump: Option<&mut Captures>,
    ) -> Result<Output> {
        for stage in &mut self.stages {
            let out = match stage {
                Stage::Gpu(s) => s.forward(&x, pos, n, dump.as_deref_mut())?,
                Stage::Cpu(s) => {
                    let mut cap = dump.as_deref_mut();
                    s.forward(&self.gguf, &mut x, pos, n, &mut cap)?;
                    if s.head {
                        let logits = s.logits(&self.gguf, &x, &mut cap)?;
                        if let Some(m) = cap {
                            m.insert("result_output".into(), logits.clone());
                        }
                        Output::Top(top_k(&logits))
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
