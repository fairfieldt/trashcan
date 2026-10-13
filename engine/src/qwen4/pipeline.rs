//! Layer placement and the stages of one machine for qwen4exp. Stages pass the whole hyper-connection residual
//! (`n x hc x d`). The n-gram rows are gathered on the driver, so the stage owning the PLE layer must be the
//! driver's; the last stage of the last machine owns the final mixer and the LM head.
use super::{
    cpu::{Captures, CpuStage},
    gpu::{merge_top, GpuStage, KvCache, Mode, Output, POLLED_MAX},
    Config,
};
use crate::{gguf::Gguf, gpu::Instance, qwen::pipeline::Devices, vulkan::Device};
use anyhow::{bail, ensure, Result};
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

/// `TRASHCAN_Q4_TIMELINE=1` (diagnostics): a stderr line per prompt-chunk event, `tl HOST_US EVENT STAGE POS A B`
/// with CLOCK_MONOTONIC microseconds; with `TRASHCAN_TP_PROFILE=1` too, `done` carries the chunk's GPU start and end
/// in the same clock.
pub fn tl(ev: &str, stage: usize, pos: usize, a: f64, b: f64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("TRASHCAN_Q4_TIMELINE").is_some()) {
        eprintln!("tl {} {ev} {stage} {pos} {a:.0} {b:.0}", crate::vulkan::host_ns() / 1000);
    }
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

/// A prompt chunk that left this machine's stages (`Local::push`): its residual, or on the machine that ends the
/// model its final hidden state.
pub struct Done {
    pub pos: usize,
    pub n: usize,
    pub out: Output,
}

/// A prompt chunk a GPU stage works on, with its n-gram rows for the stage that owns the PLE layer.
struct Flight {
    pos: usize,
    n: usize,
    ple: Option<Vec<f32>>,
}

pub struct Local {
    pub gguf: Gguf,
    pub c: Config,
    pub stages: Vec<Stage>,
    /// The LM head rows each stage scores (the stage that ends the model always has some).
    pub heads: Vec<Option<(usize, usize)>>,
    /// Holds the GPUs' clocks up while work arrives (GPU stages only).
    pub(crate) clocks: Option<crate::clocks::Clocks>,
    /// The chunks each stage is working on, oldest first: the running one and one queued behind it (`push`).
    flight: Vec<std::collections::VecDeque<Flight>>,
    /// The residual rows the stage that ends the model kept from the latest verification or `forward_rows` batch
    /// (speculative decoding, `spec.rs`).
    pub rows: Vec<f32>,
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
                ensure!(last < c.layers || heads[0].is_some(), "the stage that ends the model needs head rows");
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
                    let start = std::time::Instant::now();
                    let gpu = Device::new(inst.clone(), pd)?;
                    let stage = GpuStage::new(gpu, &gguf, &c, first, last, heads[i], context, max_batch, kv)?;
                    let (size, budget, usage) = heap_mib(&inst, pd);
                    eprintln!(
                        "GPU{}: {:.1} MiB allocated in {:.2} s; device-local heap {usage:.0} of {size:.0} MiB used, budget \
                         {budget:.0}",
                        indices[i],
                        stage.device_mib(),
                        start.elapsed().as_secs_f64()
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
            flight: stages.iter().map(|_| std::collections::VecDeque::new()).collect(),
            stages,
            heads: heads.to_vec(),
            clocks,
            rows: Vec::new(),
        })
    }

    /// Start a prompt chunk on the first stage without waiting for it. The chunks in flight move on by one stage: each
    /// stage queues the incoming chunk behind the one it is running and then waits for that one, so the GPUs of a
    /// machine work on consecutive chunks at the same time and a GPU never waits for the host between chunks.
    /// Returns the chunk that left the last stage, if one did. `forward` and `head_slices` need the stages idle:
    /// `drain` or `finish` first.
    pub fn push(&mut self, x: Vec<f32>, ple: Option<Vec<f32>>, pos: usize, n: usize) -> Result<Option<Done>> {
        if self.stages.iter().any(|s| matches!(s, Stage::Cpu(_))) {
            // A CPU machine runs a single stage, synchronously.
            let out = self.forward(x, ple.as_deref(), pos, n, None)?;
            return Ok(Some(Done { pos, n, out }));
        }
        self.step(Some((Flight { pos, n, ple }, x)))
    }

    /// Wait for the chunks in flight, handing each to `pass` as it leaves the last stage.
    pub fn drain(&mut self, mut pass: impl FnMut(Done) -> Result<()>) -> Result<()> {
        while self.flight.iter().any(|f| !f.is_empty()) {
            if let Some(d) = self.step(None)? {
                pass(d)?;
            }
        }
        Ok(())
    }

    /// Run a chunk after those in flight and wait for all of them: the earlier chunks go to `pass` as they leave (see
    /// `push`), and this chunk's output is returned as `forward` returns it.
    pub fn finish(
        &mut self,
        x: Vec<f32>,
        ple: Option<Vec<f32>>,
        pos: usize,
        n: usize,
        mut pass: impl FnMut(Done) -> Result<()>,
    ) -> Result<Output> {
        if self.flight.iter().all(|f| f.is_empty()) {
            return self.forward(x, ple.as_deref(), pos, n, None);
        }
        if let Some(d) = self.push(x, ple, pos, n)? {
            pass(d)?;
        }
        // The chunk leaves last.
        loop {
            let d = self.step(None)?;
            let idle = self.flight.iter().all(|f| f.is_empty());
            match d {
                Some(d) if idle => return Ok(d.out),
                Some(d) => pass(d)?,
                None => ensure!(!idle, "the chunk did not leave the stages"),
            }
        }
    }

    /// Move every chunk in flight on by one stage and start `input` on the first; the chunk that left the last stage.
    fn step(&mut self, mut input: Option<(Flight, Vec<f32>)>) -> Result<Option<Done>> {
        if let Some(c) = &self.clocks {
            c.busy();
        }
        let last = self.stages.len() - 1;
        let mut left = None;
        for (i, (stage, flight)) in self.stages.iter_mut().zip(&mut self.flight).enumerate() {
            let Stage::Gpu(s) = stage else { bail!("only GPU stages work on chunks in flight") };
            // Queue the incoming chunk behind the running one, then wait for the running one (with nothing coming in,
            // for whatever this stage holds).
            let incoming = input.is_some();
            if let Some((f, x)) = input.take() {
                s.submit(&x, f.ple.as_deref(), f.pos, f.n, false)?;
                tl("sub", i, f.pos, 0., 0.);
                flight.push_back(f);
            }
            let prev = if flight.len() > usize::from(incoming) {
                let f = flight.pop_front().unwrap();
                tl("wait", i, f.pos, 0., 0.);
                s.wait()?;
                let out = s.output(None)?;
                let (a, b) = s.window.unwrap_or((0., 0.));
                tl("done", i, f.pos, a / 1e3, b / 1e3);
                Some((out, f))
            } else {
                None
            };
            match prev {
                Some((Output::Hidden(x), f)) if i < last => input = Some((f, x)),
                Some((out, f)) => left = Some(Done { pos: f.pos, n: f.n, out }),
                None => {}
            }
        }
        Ok(left)
    }

    /// Score a final hidden state with the head rows of every stage; merged candidates. GPU stages score at the
    /// same time.
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
                Stage::Cpu(_) => match rows {
                    Some((a, b)) => {
                        let logits = CpuStage::head_logits(&self.gguf, h, *a, *b);
                        if let Some(m) = dump.as_deref_mut() {
                            let name = if b - a == self.c.vocab {
                                "result_output".to_string()
                            } else {
                                format!("result_output@{a}")
                            };
                            m.insert(name, logits.clone());
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
        super::trace::mark("top", 0);
        Ok(top)
    }

    /// Run all local stages. Returns the residual, or the final hidden state when this machine ends the model
    /// (`head_slices` scores it).
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
        if dump.is_none() && self.direct(n) {
            return self.forward_direct(x, ple, pos, n, Mode::Plain);
        }
        let mut end = None;
        for stage in self.stages.iter_mut() {
            let out = match stage {
                Stage::Gpu(s) => s.forward(&x, ple, pos, n, dump.as_deref_mut())?,
                Stage::Cpu(s) => {
                    let mut cap = dump.as_deref_mut();
                    s.forward(&self.gguf, &mut x, ple, pos, n, &mut cap)?;
                    if s.head {
                        Output::Final(s.final_hidden(&self.gguf, &x, &mut cap)?)
                    } else {
                        Output::Hidden(std::mem::take(&mut x))
                    }
                }
            };
            match out {
                Output::Hidden(h) => x = h,
                fin => {
                    end = Some(fin);
                    break;
                }
            }
        }
        Ok(end.unwrap_or(Output::Hidden(x)))
    }

    /// Whether an `n`-token forward runs as `forward_direct`: a decode step or verification batch on GPU stages only,
    /// with no prompt chunks in flight.
    pub(crate) fn direct(&self, n: usize) -> bool {
        n <= POLLED_MAX
            && self.stages.iter().all(|s| matches!(s, Stage::Gpu(_)))
            && self.flight.iter().all(|f| f.is_empty())
    }

    /// Run a forward of up to `POLLED_MAX` tokens on GPU stages, each starting on the previous one's output copied
    /// straight from that GPU's host buffer into its own. In verification mode the stage that ends the model also
    /// leaves the batch's residual rows in `rows`.
    pub(crate) fn forward_direct(&mut self, x: Vec<f32>, ple: Option<&[f32]>, pos: usize, n: usize, mode: Mode) -> Result<Output> {
        fn gpu(s: &mut Stage) -> Result<&mut GpuStage> {
            match s {
                Stage::Gpu(g) => Ok(g),
                Stage::Cpu(_) => bail!("direct forwards run on GPU stages only"),
            }
        }
        let k = self.stages.len();
        gpu(&mut self.stages[0])?.submit_mode(&x, ple, pos, n, false, mode)?;
        for i in 0..k - 1 {
            let (a, b) = self.stages.split_at_mut(i + 1);
            gpu(&mut a[i])?.collect_into(gpu(&mut b[0])?, pos, mode)?;
        }
        let g = gpu(&mut self.stages[k - 1])?;
        let out = g.collect(None)?;
        if let (Mode::Verify, Output::Final(_)) = (mode, &out) {
            self.rows = g.residual(n)?;
        }
        Ok(out)
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
