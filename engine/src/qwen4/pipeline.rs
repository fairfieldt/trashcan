//! Layer placement and the stages of one machine for qwen4exp. Stages pass the whole hyper-connection residual
//! (`n x hc x d`). The n-gram rows are gathered on the driver, so the stage owning the PLE layer must be the
//! driver's; the last stage of the last machine owns the final mixer and the LM head.
use super::{
    cpu::{Captures, CpuStage},
    gpu::{GpuStage, Output},
    Config,
};
use crate::{gguf::Gguf, gpu::Instance, qwen::pipeline::Devices, vulkan::Device};
use anyhow::{ensure, Result};
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

/// Contiguous layer ranges for `devices` stages in order, balancing uploaded bytes; the last stage also holds
/// the head. The first stage keeps at least the layers through the PLE layer.
pub fn plan(g: &Gguf, c: &Config, devices: usize) -> Vec<(usize, usize)> {
    let costs: Vec<f64> = (0..c.layers).map(|il| layer_bytes(g, il) as f64).collect();
    let stages = devices.clamp(1, c.layers);
    let head = (c.vocab * c.d) as f64 * 1.25;
    let target = (costs.iter().sum::<f64>() + head) / stages as f64;
    let mut cuts = vec![0];
    let mut acc = 0.;
    let min_first = c.ngram.as_ref().map_or(1, |p| p.layer + 1);
    for (il, cost) in costs.iter().enumerate() {
        let next = cuts.len();
        if next < stages && il >= min_first && il > *cuts.last().unwrap() {
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

pub enum Stage {
    Cpu(CpuStage),
    Gpu(GpuStage),
}

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
                    eprintln!(
                        "GPU{}: layers {first}..{last}{}",
                        indices[i],
                        if is_head { " + head" } else { "" }
                    );
                    let gpu = Device::new(inst.clone(), pd)?;
                    let stage = GpuStage::new(gpu, &gguf, &c, first, last, is_head, context, max_batch)?;
                    let (size, budget, usage) = heap_mib(&inst, pd);
                    eprintln!(
                        "GPU{}: {:.1} MiB allocated; device-local heap {usage:.0} of {size:.0} MiB used, budget {budget:.0}",
                        indices[i],
                        stage.device_mib()
                    );
                    stages.push(Stage::Gpu(stage));
                }
            }
        }
        Ok(Self { gguf, c, stages })
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
        for stage in &mut self.stages {
            let out = match stage {
                Stage::Gpu(s) => s.forward(&x, ple, pos, n, dump.as_deref_mut())?,
                Stage::Cpu(s) => {
                    let mut cap = dump.as_deref_mut();
                    s.forward(&self.gguf, &mut x, ple, pos, n, &mut cap)?;
                    if s.head {
                        let logits = s.logits(&self.gguf, &x, &mut cap)?;
                        if let Some(m) = cap {
                            m.insert("result_output".into(), logits.clone());
                        }
                        Output::Top(crate::qwen::pipeline::top_k(&logits))
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
