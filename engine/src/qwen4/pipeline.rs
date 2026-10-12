//! Layer placement and the stages of one machine for qwen4exp. Stages pass the whole hyper-connection residual
//! (`n x hc x d`). The n-gram rows are gathered on the driver, so the stage owning the PLE layer must be the
//! driver's; the last stage of the last machine owns the final mixer and the LM head.
use super::{
    cpu::{Captures, CpuStage},
    gpu::{merge_top, GpuStage, KvCache, Output},
    Config,
};
use crate::{gguf::Gguf, gpu::Instance, qwen::pipeline::Devices, vulkan::Device};
use anyhow::{ensure, Context, Result};
use std::{path::Path, rc::Rc};

/// Bytes a layer occupies on a GPU: Q4_0 repacks 18-byte blocks into 20 bytes, the f32 matrices stay f32.
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

/// Contiguous layer ranges for `devices` stages in order, balancing uploaded bytes; the head is split over the last
/// `head_split` stages. The first stage keeps at least the layers through the PLE layer.
pub fn plan(g: &Gguf, c: &Config, devices: usize, head_split: usize) -> Vec<(usize, usize)> {
    let costs: Vec<f64> = (0..c.layers).map(|il| layer_bytes(g, il) as f64).collect();
    let min_first = c.ngram.as_ref().map_or(1, |p| p.layer + 1);
    split_layers(&costs, (c.vocab * c.d) as f64 * 1.25, devices, head_split, min_first)
}

/// `plan` on per-layer costs and the head's cost.
fn split_layers(costs: &[f64], head: f64, devices: usize, head_split: usize, min_first: usize) -> Vec<(usize, usize)> {
    let stages = devices.clamp(1, costs.len());
    let split = head_split.clamp(1, stages);
    let target = (costs.iter().sum::<f64>() + head) / stages as f64;
    // Layer bytes before the cut that starts stage k: each earlier stage's target minus its head share.
    let share = |j: usize| if j + split >= stages { head / split as f64 } else { 0. };
    let bound = |k: usize| (0..k).map(|j| target - share(j)).sum::<f64>();
    let mut cuts = vec![0];
    let mut acc = 0.;
    for (il, cost) in costs.iter().enumerate() {
        let next = cuts.len();
        if next < stages && il >= min_first && il > *cuts.last().unwrap() {
            let boundary = bound(next);
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

/// Size, budget and usage in MiB of a GPU's largest device-local heap (VK_EXT_memory_budget). RADV places
/// allocations that do not fit elsewhere, so usage near the size means a stage does not fit.
pub fn heap_mib(inst: &Instance, pd: ash::vk::PhysicalDevice) -> (f64, f64, f64) {
    let mut b = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut p = ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut b);
    unsafe { inst.instance.get_physical_device_memory_properties2(pd, &mut p) };
    let mem = p.memory_properties;
    let n = mem.memory_heap_count as usize;
    let h = (0..n)
        .filter(|&h| mem.memory_heaps[h].flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
        .max_by_key(|&h| mem.memory_heaps[h].size)
        .unwrap_or(0);
    let mib = |v: u64| v as f64 / 1048576.;
    (mib(mem.memory_heaps[h].size), mib(b.heap_budget[h]), mib(b.heap_usage[h]))
}

/// Head rows of each of `stages` stages (in chain order) when the LM head is split by vocabulary rows over the
/// last `split` of them: equal parts in multiples of 1024 rows, the last stage (which ends the model) taking the rest.
pub fn head_rows(vocab: usize, stages: usize, split: usize) -> Vec<Option<(usize, usize)>> {
    let split = split.clamp(1, stages);
    let per = vocab.div_ceil(split).next_multiple_of(1024);
    (0..stages)
        .map(|i| {
            let k = (i + split).checked_sub(stages)?;
            let (a, b) = ((k * per).min(vocab), if k + 1 == split { vocab } else { ((k + 1) * per).min(vocab) });
            (a < b).then_some((a, b))
        })
        .collect()
}

pub enum Stage {
    Cpu(CpuStage),
    Gpu(GpuStage),
}

pub struct Local {
    pub gguf: Gguf,
    pub c: Config,
    pub stages: Vec<Stage>,
    /// The LM head rows each stage scores (the stage that ends the model always has some).
    pub heads: Vec<Option<(usize, usize)>>,
    /// Holds the GPUs' clocks up while work arrives (GPU stages only).
    clocks: Option<crate::clocks::Clocks>,
}

impl Local {
    pub fn new(
        path: &Path,
        devices: &Devices,
        ranges: &[(usize, usize)],
        heads: &[Option<(usize, usize)>],
        context: usize,
        max_batch: usize,
        kv: KvCache,
    ) -> Result<Self> {
        let gguf = Gguf::open(path)?;
        super::cpu::set_kv_cache(kv);
        let c = Config::new(&gguf)?;
        ensure!(ranges.len() <= devices.count(), "more stages than devices");
        ensure!(heads.len() == ranges.len(), "one head entry per stage");
        let mut stages = Vec::new();
        let mut clocks = None;
        match devices {
            Devices::Cpu => {
                ensure!(ranges.len() == 1, "the CPU runs a single stage");
                let (first, last) = ranges[0];
                stages.push(Stage::Cpu(CpuStage::new(&c, first, last, context, last == c.layers)));
            }
            Devices::Gpus(indices) => {
                clocks = crate::clocks::Clocks::new();
                let inst = Rc::new(Instance::new()?);
                let all = inst.amd_gpus()?;
                for (i, &(first, last)) in ranges.iter().enumerate() {
                    let pd = *all
                        .get(indices[i])
                        .ok_or_else(|| anyhow::anyhow!("GPU {} not found", indices[i]))?;
                    eprintln!(
                        "GPU{}: layers {first}..{last}{}",
                        indices[i],
                        heads[i].map_or(String::new(), |(a, b)| format!(" + head rows {a}..{b}"))
                    );
                    let gpu = Device::new(inst.clone(), pd)?;
                    let stage = GpuStage::new(gpu, &gguf, &c, first, last, heads[i], context, max_batch, kv)?;
                    let (size, budget, usage) = heap_mib(&inst, pd);
                    eprintln!(
                        "GPU{}: {:.1} MiB allocated; device-local heap {usage:.0} of {size:.0} MiB used, budget {budget:.0}",
                        indices[i],
                        stage.device_mib()
                    );
                    // RADV silently places what does not fit in system memory, where the stage runs far slower.
                    ensure!(
                        usage <= budget,
                        "GPU{}: layers {first}..{last} need {usage:.0} MiB of device-local memory but the budget is \
                         {budget:.0} MiB (heap {size:.0}); give the GPU fewer layers or lower --batch or --context",
                        indices[i]
                    );
                    stages.push(Stage::Gpu(stage));
                }
            }
        }
        Ok(Self {
            gguf,
            c,
            stages,
            heads: heads.to_vec(),
            clocks,
        })
    }

    /// Score a final hidden state with the head rows of the stages that do not end the model; merged candidates.
    /// GPU stages score at the same time.
    pub fn head_slices(&mut self, h: &[f32], mut dump: Option<&mut Captures>) -> Result<Vec<(f32, u32)>> {
        if let Some(c) = &self.clocks {
            c.busy();
        }
        let mut top = Vec::new();
        let mut started = vec![false; self.stages.len()];
        for (stage, s) in self.stages.iter_mut().zip(started.iter_mut()) {
            if let Stage::Gpu(g) = stage {
                *s = g.head_submit(h, dump.is_some())?;
            }
        }
        for ((stage, rows), started) in self.stages.iter_mut().zip(&self.heads).zip(started) {
            let mine = match stage {
                Stage::Gpu(g) if started => g.head_collect(dump.as_deref_mut())?,
                Stage::Gpu(_) => None,
                Stage::Cpu(s) => match rows {
                    Some((a, b)) if !s.head => {
                        let logits = CpuStage::head_logits(&self.gguf, h, *a, *b);
                        if let Some(m) = dump.as_deref_mut() {
                            m.insert(format!("result_output@{a}"), logits.clone());
                        }
                        Some(offset(crate::qwen::pipeline::top_k(&logits), *a))
                    }
                    _ => None,
                },
            };
            if let Some(m) = mine {
                top = merge_top(top, m);
            }
        }
        Ok(top)
    }

    /// Run all local stages. Returns the residual, or the top candidates when this machine ends the model.
    pub fn forward(
        &mut self,
        mut x: Vec<f32>,
        ple: Option<&[f32]>,
        pos: usize,
        n: usize,
        mut dump: Option<&mut Captures>,
    ) -> Result<Output> {
        if let Some(c) = &self.clocks {
            c.busy();
        }
        let mut end = None;
        for (stage, rows) in self.stages.iter_mut().zip(&self.heads) {
            let out = match stage {
                Stage::Gpu(s) => s.forward(&x, ple, pos, n, dump.as_deref_mut())?,
                Stage::Cpu(s) => {
                    let mut cap = dump.as_deref_mut();
                    s.forward(&self.gguf, &mut x, ple, pos, n, &mut cap)?;
                    if s.head {
                        let (a, b) = rows.context("the stage that ends the model needs head rows")?;
                        let h = s.final_hidden(&self.gguf, &x, &mut cap)?;
                        let logits = CpuStage::head_logits(&self.gguf, &h, a, b);
                        if let Some(m) = cap {
                            let name = if b - a == self.c.vocab {
                                "result_output".to_string()
                            } else {
                                format!("result_output@{a}")
                            };
                            m.insert(name, logits.clone());
                        }
                        Output::Top(offset(crate::qwen::pipeline::top_k(&logits), a), h)
                    } else {
                        Output::Hidden(std::mem::take(&mut x))
                    }
                }
            };
            match out {
                Output::Hidden(h) => x = h,
                top => {
                    end = Some(top);
                    break;
                }
            }
        }
        // This machine ends the model: its other stages score their head rows on the final hidden state.
        match end {
            Some(Output::Top(top, h)) => {
                let mine = self.head_slices(&h, dump)?;
                Ok(Output::Top(merge_top(top, mine), h))
            }
            _ => Ok(Output::Hidden(x)),
        }
    }
}

/// Candidates of a head slice with vocabulary ids.
fn offset(top: Vec<(f32, u32)>, row0: usize) -> Vec<(f32, u32)> {
    top.into_iter().map(|(s, id)| (s, id + row0 as u32)).collect()
}

/// Assemble the full logits from head-slice captures (`result_output@ROW0`), when every row is present.
pub fn assemble_logits(caps: &mut Captures, vocab: usize) {
    let parts: Vec<String> = caps.keys().filter(|k| k.starts_with("result_output@")).cloned().collect();
    if parts.is_empty() {
        return;
    }
    let mut full = vec![f32::NAN; vocab];
    let mut covered = 0;
    for k in &parts {
        let row0: usize = k["result_output@".len()..].parse().unwrap_or(usize::MAX);
        let v = caps.remove(k).unwrap();
        if row0 + v.len() <= vocab {
            full[row0..row0 + v.len()].copy_from_slice(&v);
            covered += v.len();
        }
    }
    if covered == vocab {
        caps.insert("result_output".into(), full);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_split_head_spreads_layers_evenly() {
        let costs = vec![1.4e9; 48];
        let head = 0.8e9;
        let even: Vec<_> = (0..12).map(|i| (4 * i, 4 * i + 4)).collect();
        assert_eq!(split_layers(&costs, head, 12, 12, 2), even);
        // With the whole head on the last stage, that stage gets fewer layers.
        let one = split_layers(&costs, head, 12, 1, 2);
        assert!(one.last().map(|(a, b)| b - a).unwrap() < 4, "{one:?}");
        let rows = head_rows(248320, 12, 12);
        assert_eq!(rows[0], Some((0, 21504)));
        assert_eq!(rows[11], Some((11 * 21504, 248320)));
        assert!(head_rows(248320, 3, 1)[..2].iter().all(|r| r.is_none()));
    }
}
