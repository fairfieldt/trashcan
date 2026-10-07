//! `trashcan qwen4 ...` commands.
use anyhow::{ensure, Context, Result};
use clap::Subcommand;
use std::path::{Path, PathBuf};

/// The 14-token ChatML reference prompt ("Hello there", thinking off), as for the qwen35 models.
pub const REFERENCE: [u32; 14] = [
    248045, 846, 198, 9419, 1017, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];

#[derive(Subcommand)]
pub enum Cmd {
    /// Run the CPU reference over a prompt, dump activations in the reference format and decode greedily.
    Ref {
        /// The model GGUF (the first shard of a split model).
        model: PathBuf,
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        greedy: usize,
        /// Prompt token ids, whitespace separated (default: the 14-token reference prompt).
        #[arg(long)]
        tokens_file: Option<PathBuf>,
        tokens: Vec<u32>,
        /// Round like ggml's CPU backend (see `cpu::EMULATE_GGML`).
        #[arg(long)]
        emulate_ggml: bool,
    },
    /// Teacher-forced per-layer check: run each layer on the reference's input residual (every prompt token, from a
    /// dump made with DUMP_FULL=l_last,hc_init,ple_embd) and compare with the reference's output, so errors do not
    /// accumulate across layers.
    LayerCheck {
        model: PathBuf,
        reference: PathBuf,
        #[arg(long, default_value_t = 0)]
        first: usize,
        #[arg(long)]
        last: Option<usize>,
        #[arg(long)]
        emulate_ggml: bool,
        /// Write this run's last-token intermediates here.
        #[arg(long)]
        dump: Option<PathBuf>,
    },
    /// Render a text file as one user turn (thinking off) and print the token ids.
    TokenizeChat { model: PathBuf, file: PathBuf },
    /// Measure n-gram row reads: a resident table (ram-q4, ram-q8) or rows read from disk (buffered,
    /// buffered-cold, direct). Prints one JSON line.
    NgramBench {
        /// A GGUF with the model metadata (the first shard is enough).
        meta: PathBuf,
        /// The GGUF file holding per_layer_token_embd (its shard of a split model is enough).
        table: PathBuf,
        #[arg(long)]
        mode: String,
        #[arg(long, default_value_t = 16)]
        threads: usize,
        /// Token ids to hash (whitespace separated, cycled); default uniformly random tokens.
        #[arg(long)]
        ids: Option<PathBuf>,
        #[arg(long, default_value_t = 8192)]
        tokens: usize,
        #[arg(long, default_value_t = 512)]
        batch: usize,
    },
    /// Time GPU stages alone (no chain): layer ranges on local GPUs, fed token embeddings, decode then prefill
    /// chunks back to back; prints per-layer and per-phase GPU milliseconds.
    StageBench {
        model: PathBuf,
        /// Layer ranges, one per GPU in order, e.g. 0:4,4:8.
        #[arg(long, value_delimiter = ',')]
        ranges: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        gpus: Option<Vec<usize>>,
        /// Give the last stage the final mixer and the LM head.
        #[arg(long)]
        head: bool,
        /// LM head rows (a:b) for the last stages, one entry per stage; stages that do not end the model are timed
        /// scoring a hidden state with them, alone and all at once.
        #[arg(long, value_delimiter = ',')]
        head_rows: Vec<String>,
        #[arg(long, default_value_t = 2048)]
        context: usize,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        /// Decode steps timed, after a prefill of `start` tokens.
        #[arg(long, default_value_t = 32)]
        decode: usize,
        #[arg(long, default_value_t = 64)]
        start: usize,
        /// Prefill chunks of `batch` tokens timed.
        #[arg(long, default_value_t = 3)]
        prefill: usize,
        /// Instead of timing, run this prompt (token ids) and dump the last chunk's activations into `--out`.
        #[arg(long)]
        tokens_file: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Allocate device-local memory in `chunk_mib` pieces on each GPU until allocation fails; report the totals.
    VramProbe {
        #[arg(long, default_value_t = 64)]
        chunk_mib: usize,
    },
    /// Serve layers to the previous machine in a chain.
    Node {
        model: PathBuf,
        #[arg(long, default_value = "0.0.0.0:7071")]
        listen: String,
        #[arg(long)]
        cpu: bool,
        #[arg(long, value_delimiter = ',')]
        gpus: Option<Vec<usize>>,
    },
    /// Run a prompt through the chain, dump the last token's activations in the reference format, decode greedily.
    Validate {
        model: PathBuf,
        out: PathBuf,
        #[command(flatten)]
        run: Run,
        #[arg(long, default_value_t = 10)]
        greedy: usize,
        #[arg(long)]
        tokens_file: Option<PathBuf>,
    },
    /// Measure prefill and greedy decode.
    Bench {
        model: PathBuf,
        #[command(flatten)]
        run: Run,
        #[arg(long, default_value_t = 512)]
        pp: usize,
        #[arg(long, default_value_t = 64)]
        tg: usize,
    },
}

/// This machine's devices and the rest of the chain.
#[derive(clap::Args)]
pub struct Run {
    /// Further machines, in chain order (host:port of `trashcan qwen4 node`).
    #[arg(long, value_delimiter = ',')]
    nodes: Vec<String>,
    #[arg(long)]
    cpu: bool,
    #[arg(long, value_delimiter = ',')]
    gpus: Option<Vec<usize>>,
    /// First layer of each stage after the first, in chain order (default: balance weight bytes).
    #[arg(long, value_delimiter = ',')]
    cuts: Vec<usize>,
    #[arg(long, default_value_t = 2048)]
    context: usize,
    #[arg(long, default_value_t = 256)]
    batch: usize,
    /// Split the LM head by vocabulary rows over the last N stages; each scores its rows as the result returns.
    #[arg(long, default_value_t = 1)]
    head_split: usize,
    /// N-gram rows: map (the GGUF mapping), ram-q4 or ram-q8 (a resident copy), disk (pread through the page
    /// cache) or direct (O_DIRECT page reads).
    #[arg(long, default_value = "map")]
    ngram: String,
}

impl Run {
    fn chain(self, model: &Path) -> Result<super::net::Chain> {
        let devices = crate::qwen::chat::devices(self.cpu, self.gpus)?;
        let rows = match self.ngram.as_str() {
            "map" => super::net::Rows::Map,
            mode => {
                let g = crate::gguf::Gguf::open(model)?;
                let t = g.tensor("per_layer_token_embd.weight").context("no n-gram table")?;
                let table = super::ngram::Table::open(&g.paths[t.file], mode == "direct")?;
                super::net::Rows::Table(match mode {
                    "ram-q4" => table.to_ram(crate::gguf::GgmlType::Q4_0, 16)?,
                    "ram-q8" => table.to_ram(crate::gguf::GgmlType::Q8_0, 16)?,
                    "disk" | "direct" => table,
                    other => anyhow::bail!("unknown --ngram {other}"),
                })
            }
        };
        super::net::Chain::new(
            model,
            devices,
            &self.nodes,
            &self.cuts,
            self.head_split,
            self.context,
            self.batch,
            rows,
        )
    }
}

fn prompt(file: Option<&Path>, tokens: &[u32]) -> Result<Vec<u32>> {
    if let Some(f) = file {
        let s = std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?;
        return s.split_whitespace().map(|t| Ok(t.parse()?)).collect();
    }
    Ok(if tokens.is_empty() {
        REFERENCE.to_vec()
    } else {
        tokens.to_vec()
    })
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Ref {
            model,
            out,
            greedy,
            tokens_file,
            tokens,
            emulate_ggml,
        } => {
            super::cpu::EMULATE_GGML.store(emulate_ggml, std::sync::atomic::Ordering::Relaxed);
            let ids = super::reference(&model, &out, greedy, &prompt(tokens_file.as_deref(), &tokens)?)?;
            println!("greedy: {ids:?}");
            Ok(())
        }
        Cmd::LayerCheck {
            model,
            reference,
            first,
            last,
            emulate_ggml,
            dump,
        } => {
            super::cpu::EMULATE_GGML.store(emulate_ggml, std::sync::atomic::Ordering::Relaxed);
            layer_check(&model, &reference, first, last, dump.as_deref())
        }
        Cmd::NgramBench {
            meta,
            table,
            mode,
            threads,
            ids,
            tokens,
            batch,
        } => {
            let r = super::ngram::bench(&meta, &table, &mode, threads, ids.as_deref(), tokens, batch)?;
            println!("{r}");
            Ok(())
        }
        Cmd::StageBench {
            model,
            ranges,
            gpus,
            head,
            head_rows,
            context,
            batch,
            decode,
            start,
            prefill,
            tokens_file,
            out,
        } => {
            let ranges: Vec<(usize, usize)> = ranges
                .iter()
                .map(|r| {
                    let (a, b) = r.split_once(':').context("ranges look like 0:4")?;
                    Ok((a.parse()?, b.parse()?))
                })
                .collect::<Result<_>>()?;
            let gpus = gpus.unwrap_or_else(|| (0..ranges.len()).collect());
            let devices = crate::qwen::pipeline::Devices::Gpus(gpus);
            let mut heads = vec![None; ranges.len()];
            if head {
                let g = crate::gguf::Gguf::open(&model)?;
                heads[ranges.len() - 1] = Some((0, super::Config::new(&g)?.vocab));
            }
            ensure!(head_rows.len() <= ranges.len(), "more head row ranges than stages");
            for (k, r) in head_rows.iter().enumerate() {
                let (a, b) = r.split_once(':').context("head rows look like 0:82944")?;
                heads[ranges.len() - head_rows.len() + k] = Some((a.parse()?, b.parse()?));
            }
            let mut local = super::pipeline::Local::new(&model, &devices, &ranges, &heads, context, batch)?;
            if let (Some(tf), Some(out)) = (tokens_file, out) {
                let tokens = prompt(Some(&tf), &[])?;
                return super::bench::stage_dump(&mut local, &out, &tokens, batch);
            }
            println!(
                "{}",
                super::bench::stage_bench(&mut local, decode, start, prefill, batch)?
            );
            Ok(())
        }
        Cmd::VramProbe { chunk_mib } => {
            // RADV over-commits device-local memory into system RAM, so never allocate past the device-local heap:
            // allocate chunks while the driver's budget (VK_EXT_memory_budget) still covers the heap's usage.
            let inst = std::rc::Rc::new(crate::gpu::Instance::new()?);
            for (i, pd) in inst.amd_gpus()?.into_iter().enumerate() {
                let gpu = crate::vulkan::Device::new(inst.clone(), pd)?;
                let budget = || {
                    let mut b = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
                    let mut p = ash::vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut b);
                    unsafe { inst.instance.get_physical_device_memory_properties2(pd, &mut p) };
                    (b.heap_budget, b.heap_usage)
                };
                let mem = gpu.mem;
                let n = mem.memory_heap_count as usize;
                let local = (0..n)
                    .filter(|&h| {
                        mem.memory_heaps[h]
                            .flags
                            .contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL)
                    })
                    .max_by_key(|&h| mem.memory_heaps[h].size)
                    .context("no device-local heap")?;
                let (b0, u0) = budget();
                let mut held = Vec::new();
                loop {
                    let (b, u) = budget();
                    if u[local] + ((chunk_mib as u64) << 20) > b[local].min(mem.memory_heaps[local].size) {
                        break;
                    }
                    match gpu.buffer(chunk_mib << 20, false, false, None) {
                        Ok(buf) => held.push(buf),
                        Err(_) => break,
                    }
                }
                let (b1, u1) = budget();
                let mib = |v: u64| v as f64 / 1048576.;
                println!(
                    "{}",
                    serde_json::json!({"gpu": i,
                        "heaps_mib": (0..n).map(|h| mib(mem.memory_heaps[h].size)).collect::<Vec<_>>(),
                        "device_local_heap": local, "budget_mib_before": mib(b0[local]), "usage_mib_before": mib(u0[local]),
                        "allocated_mib": held.len() * chunk_mib, "budget_mib_after": mib(b1[local]),
                        "usage_mib_after": mib(u1[local]), "chunk_mib": chunk_mib})
                );
            }
            Ok(())
        }
        Cmd::Node {
            model,
            listen,
            cpu,
            gpus,
        } => super::net::serve(&model, &listen, crate::qwen::chat::devices(cpu, gpus)?),
        Cmd::Validate {
            model,
            out,
            run,
            greedy,
            tokens_file,
        } => {
            let tokens = prompt(tokens_file.as_deref(), &[])?;
            let mut chain = run.chain(&model)?;
            let ids = validate(&mut chain, &out, &tokens, greedy)?;
            println!("greedy: {ids:?}");
            Ok(())
        }
        Cmd::Bench { model, run, pp, tg } => {
            let mut chain = run.chain(&model)?;
            println!("{}", bench(&mut chain, pp, tg)?);
            Ok(())
        }
        Cmd::TokenizeChat { model, file } => {
            let g = crate::gguf::Gguf::open(&model)?;
            let t = crate::tokenizer::Tokenizer::new(&g)?;
            let text = std::fs::read_to_string(&file)?;
            let m = vec![serde_json::json!({"role": "user", "content": text})];
            let ids = t.encode(&t.render(&m, false)?, true, true);
            println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
            Ok(())
        }
    }
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(b.chunks_exact(4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
        .collect())
}

/// Relative L2 error and cosine of `a` against the reference `b`.
pub fn metrics(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut dot, mut na) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
        dot += x * y;
        na += x * x;
    }
    ((num / den).sqrt(), dot / (na * den).sqrt())
}

fn layer_check(model: &Path, dir: &Path, first: usize, last: Option<usize>, dump: Option<&Path>) -> Result<()> {
    let g = crate::gguf::Gguf::open(model)?;
    let c = super::Config::new(&g)?;
    let w = c.width();
    let ple = read_f32(&dir.join("ple_embd.f32"))?;
    let n = ple.len() / c.d;
    let last = last.unwrap_or(c.layers - 1);
    println!("layer\tupdate_rel\tupdate_cos\tresidual_rel\tlast_token_update_rel");
    for il in first..=last {
        let input = if il == 0 {
            read_f32(&dir.join("hc_init.f32"))?
        } else {
            read_f32(&dir.join(format!("l_last-{}.f32", il - 1)))?
        };
        let want = read_f32(&dir.join(format!("l_last-{il}.f32")))?;
        anyhow::ensure!(input.len() == n * w, "layer {il}: expected {n} input rows");
        let mut x = input.clone();
        let mut stage = super::cpu::CpuStage::new(&c, il, il + 1, n + 1, false);
        let mut caps = super::cpu::Captures::new();
        stage.forward(&g, &mut x, Some(&ple), 0, n, &mut Some(&mut caps))?;
        // Intermediates the reference also dumped (last token), for layers it dumped in detail.
        for (name, v) in &caps {
            if let Some(out) = dump {
                std::fs::create_dir_all(out)?;
                std::fs::write(out.join(format!("{name}.f32")), bytemuck::cast_slice(v))?;
            }
            let path = dir.join(format!("{name}.f32"));
            if let Ok(r) = read_f32(&path) {
                if r.len() == v.len() {
                    let (rel, cos) = metrics(v, &r);
                    println!("  {name}\trel {rel:.3e}\tcos {cos:.6}");
                }
            }
        }
        // The last layer's reference keeps only the rows gathered for the output.
        let rows = want.len() / w;
        let skip = n - rows;
        let delta = |a: &[f32]| -> Vec<f32> { a.iter().zip(&input[skip * w..]).map(|(a, b)| a - b).collect() };
        let (ours, theirs) = (delta(&x[skip * w..]), delta(&want));
        let (ur, uc) = metrics(&ours, &theirs);
        let (rr, _) = metrics(&x[skip * w..], &want);
        let (lr, _) = metrics(&ours[(rows - 1) * w..], &theirs[(rows - 1) * w..]);
        println!("{il}\t{ur:.3e}\t{uc:.6}\t{rr:.3e}\t{lr:.3e}");
    }
    Ok(())
}

/// Run a prompt (chunked by the batch size), dump the last token's activations and logits in the reference format,
/// then decode greedily.
fn validate(e: &mut super::net::Chain, out: &Path, tokens: &[u32], greedy: usize) -> Result<Vec<u32>> {
    let mut caps = super::cpu::Captures::new();
    e.pos = 0;
    let chunks: Vec<_> = tokens.chunks(e.max_batch).collect();
    let mut top = None;
    for (i, chunk) in chunks.iter().enumerate() {
        let last = i + 1 == chunks.len();
        top = e.forward(chunk, last, if last { Some(&mut caps) } else { None })?;
    }
    let mut top = top.context("empty prompt")?;
    std::fs::create_dir_all(out)?;
    let mut index = String::new();
    for (name, v) in &caps {
        std::fs::write(out.join(format!("{name}.f32")), bytemuck::cast_slice(v))?;
        index.push_str(&format!("{name}\t{},1,1,1\t{}\n", v.len(), v.len()));
    }
    std::fs::write(out.join("index.tsv"), index)?;
    if let Some(l) = caps.get("result_output") {
        std::fs::write(out.join("logits.f32"), bytemuck::cast_slice(l))?;
    }
    let mut ids = Vec::new();
    while ids.len() < greedy {
        ids.push(top[0].1);
        if ids.len() < greedy {
            top = e.forward(&[top[0].1], true, None)?.context("missing candidates")?;
        }
    }
    std::fs::write(
        out.join("greedy.txt"),
        ids.iter().map(|x| format!("{x}\n")).collect::<String>(),
    )?;
    Ok(ids)
}

/// Prefill `pp` copies of one token, then decode `tg` tokens greedily.
fn bench(e: &mut super::net::Chain, pp: usize, tg: usize) -> Result<serde_json::Value> {
    let hello = 9419;
    let start = std::time::Instant::now();
    let mut top = e.prefill(&vec![hello; pp])?;
    let pp_s = start.elapsed().as_secs_f64();
    // Per-layer GPU milliseconds of the local stages: the last prefill chunk, then the mean over decode tokens.
    let profile = |e: &super::net::Chain| -> Vec<Vec<f64>> {
        e.local
            .stages
            .iter()
            .filter_map(|s| match s {
                super::pipeline::Stage::Gpu(g) => Some(g.profile.clone()),
                _ => None,
            })
            .collect()
    };
    let prefill_ms = profile(e);
    let mut decode_ms: Vec<Vec<f64>> = prefill_ms.iter().map(|p| vec![0.; p.len()]).collect();
    let start = std::time::Instant::now();
    for _ in 0..tg {
        top = e.forward(&[top[0].1], true, None)?.context("missing candidates")?;
        for (sum, p) in decode_ms.iter_mut().zip(profile(e)) {
            for (a, b) in sum.iter_mut().zip(p) {
                *a += b / tg as f64;
            }
        }
    }
    let tg_s = start.elapsed().as_secs_f64();
    let mib: Vec<f64> = e
        .local
        .stages
        .iter()
        .filter_map(|s| match s {
            super::pipeline::Stage::Gpu(g) => Some(g.device_mib()),
            _ => None,
        })
        .collect();
    Ok(
        serde_json::json!({"pp": pp, "pp_seconds": pp_s, "pp_tok_s": pp as f64 / pp_s, "tg": tg, "tg_seconds": tg_s,
        "tg_tok_s": tg as f64 / tg_s, "plans": e.plans, "local_device_mib": mib,
        "prefill_last_chunk_layer_ms": prefill_ms, "decode_layer_ms": decode_ms}),
    )
}
