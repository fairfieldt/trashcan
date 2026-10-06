mod assistant;
mod generation;
mod gguf;
mod gpu;
mod micro;
mod model;
mod prefill_micro;
mod quant;
mod qwen;
mod responses;
mod server;
mod tokenizer;
mod visibility;
mod vulkan;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(about = "Gemma 4 12B inference on two FirePro D700s")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve the loaded model through the OpenAI Responses API.
    #[command(alias = "server")]
    Serve(server::Options),
    /// Show both GPUs and check every Vulkan feature and extension the engine depends on.
    Devices,
    /// Measure a repacked matrix kernel; check small batches against the original GPU kernel and CPU rows.
    MatvecBench {
        model: PathBuf,
        #[arg(long, default_value_t = 100)]
        iterations: u32,
        #[arg(long, default_value_t = 1)]
        batch: usize,
        #[arg(long)]
        kernel: Option<String>,
        /// Q4 projection or Q6 token embedding/head, with the engine's tensor-parallel shard.
        #[arg(long, default_value = "blk.0.ffn_gate.weight")]
        tensor: String,
        /// Sustain GPU load before collecting timestamps (D700 clocks ramp from idle).
        #[arg(long, default_value_t = 500)]
        warmup_ms: u64,
    },
    /// Measure GPU candidate selection and check ranks/ties against a CPU sort.
    TopkBench {
        #[arg(long, default_value_t = 20)]
        iterations: u32,
        #[arg(long, default_value_t = 1)]
        batch: usize,
    },
    /// Check fused RMS norm against the generic GPU kernel and a CPU reference.
    NormBench {
        #[arg(long, default_value_t = 32)]
        iterations: u32,
    },
    /// Sweep decode attention partitions and compare outputs with a CPU softmax.
    AttentionBench {
        #[arg(long, default_value_t = 100)]
        iterations: u32,
        #[arg(long, default_value_t = 768)]
        tokens: usize,
        #[arg(long)]
        partitions: Option<u32>,
        /// Compare scalar KV loads with the production shape-based kernel choice.
        #[arg(long, default_value = "selected", value_parser = ["scalar", "selected"])]
        kernel: String,
    },
    /// Compare bounded tiled prefill with scalar attention and an f64 CPU softmax.
    PrefillAttentionBench {
        #[arg(long, default_value_t = 4096)]
        position: usize,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 2)]
        iterations: u32,
        #[arg(long)]
        sliding: bool,
    },
    /// Stress P2P receiver cache visibility on both GPUs.
    P2pTest {
        #[arg(long, default_value_t = 10000)]
        rounds: u32,
        #[arg(long, default_value_t = 3840)]
        elements: usize,
    },
    /// Generate chat responses. Omit PROMPT for interactive mode (/reset, /quit).
    Chat {
        model: PathBuf,
        prompt: Option<String>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        thinking: bool,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 8192)]
        context: usize,
        #[arg(long, default_value_t = 512)]
        batch: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        #[arg(long, default_value_t = 64)]
        top_k: usize,
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long)]
        cpu_reduce: bool,
        #[arg(long)]
        profile: Option<PathBuf>,
        #[arg(long)]
        draft: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        draft_tokens: usize,
    },
    /// Compare chunked execution and speculative output with sequential execution, including ring wrap.
    CheckBatches {
        model: PathBuf,
        #[arg(long, default_value_t = 1700)]
        tokens: usize,
        #[arg(long, default_value_t = 512)]
        batch: usize,
        #[arg(long)]
        draft: Option<PathBuf>,
    },
    /// Measure prefill and decode without stopping on end-of-turn tokens.
    Bench {
        model: PathBuf,
        #[arg(long, default_value_t = 512)]
        pp: usize,
        #[arg(long, default_value_t = 128)]
        tg: usize,
        #[arg(long, default_value_t = 512)]
        batch: usize,
        #[arg(long)]
        profile: Option<PathBuf>,
        #[arg(long)]
        draft: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        draft_tokens: usize,
        /// Report each prefill chunk and ordinary decode step, checking finite top scores.
        #[arg(long)]
        progress: bool,
    },
    /// Render the model's embedded Jinja chat template.
    Render {
        model: PathBuf,
        #[arg(required_unless_present = "fixture", conflicts_with = "fixture")]
        prompt: Option<String>,
        #[arg(long)]
        fixture: Option<PathBuf>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        thinking: bool,
    },
    /// Validate or inspect the independent tokenizer.
    Tokenize {
        model: PathBuf,
        text: Option<String>,
        #[arg(long)]
        fixture: Option<PathBuf>,
    },
    /// Run the reference prompt and dump activations and greedy tokens.
    Validate {
        model: PathBuf,
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        greedy: usize,
        #[arg(long)]
        cpu_reduce: bool,
        #[arg(long, default_value_t = 1)]
        batch: usize,
        #[arg(long)]
        draft: Option<PathBuf>,
        #[arg(long, default_value_t = 3)]
        draft_tokens: usize,
    },
    /// Chat with a qwen35 model (Qwen3.8-27B, Qwen3.5-9B), optionally across a chain of machines.
    QwenChat {
        model: PathBuf,
        prompt: Option<String>,
        #[command(flatten)]
        run: QwenRun,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        thinking: bool,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 20)]
        top_k: usize,
        #[arg(long, default_value_t = 0.8)]
        top_p: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Measure qwen35 prefill and greedy decode.
    QwenBench {
        model: PathBuf,
        #[command(flatten)]
        run: QwenRun,
        #[arg(long, default_value_t = 512)]
        pp: usize,
        #[arg(long, default_value_t = 128)]
        tg: usize,
    },
    /// Check checkpoint restore, verification rollback and speculative decoding against greedy decode.
    QwenCheck {
        model: PathBuf,
        #[command(flatten)]
        run: QwenRun,
        /// Prompt length.
        #[arg(long, default_value_t = 600)]
        tokens: usize,
        /// Greedy continuation length.
        #[arg(long, default_value_t = 64)]
        steps: usize,
    },
    /// Run the reference prompt through the qwen35 pipeline, dump activations and greedy tokens.
    QwenValidate {
        model: PathBuf,
        out: PathBuf,
        #[command(flatten)]
        run: QwenRun,
        #[arg(long, default_value_t = 10)]
        greedy: usize,
    },
    /// Serve qwen35 layers to the previous machine in a chain.
    QwenNode {
        model: PathBuf,
        #[arg(long, default_value = "0.0.0.0:7070")]
        listen: String,
        #[arg(long)]
        cpu: bool,
        #[arg(long, value_delimiter = ',')]
        gpus: Option<Vec<usize>>,
        /// MTP layer GGUF, used when this machine ends the chain and the driver drafts.
        #[arg(long)]
        mtp: Option<PathBuf>,
    },
    /// Run a qwen35 model on the CPU reference path and dump activations in the reference format.
    QwenRef {
        model: PathBuf,
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        greedy: usize,
        /// Prompt token ids.
        #[arg(required = true)]
        tokens: Vec<u32>,
    },
    /// Dump a GGUF file's metadata and tensor inventory.
    Info {
        model: PathBuf,
        /// Also list the tensors of these layers (all non-layer tensors are always listed).
        #[arg(long, value_delimiter = ',', default_value = "0")]
        layers: Vec<u32>,
    },
    /// List the special (non-normal) tokens, skipping <unusedN> and <0xXX> byte tokens.
    Vocab { model: PathBuf },
    /// Print an F32 tensor's stats and values. A `*` in NAME matches the layer number, e.g. blk.*.layer_output_scale.weight.
    Tensor {
        model: PathBuf,
        name: String,
        /// How many leading values to print.
        #[arg(long, default_value_t = 8)]
        n: usize,
    },
}

/// Devices of this machine and the rest of the chain, shared by the qwen35 commands.
#[derive(clap::Args)]
struct QwenRun {
    /// Further machines, in chain order (host:port of `trashcan qwen-node`).
    #[arg(long, value_delimiter = ',')]
    nodes: Vec<String>,
    /// Run this machine's layers on the CPU reference path.
    #[arg(long)]
    cpu: bool,
    /// Local GPU indices (default: all).
    #[arg(long, value_delimiter = ',')]
    gpus: Option<Vec<usize>>,
    /// First layer of each stage after the first, in chain order (default: balance weight bytes).
    #[arg(long, value_delimiter = ',')]
    cuts: Vec<usize>,
    #[arg(long, default_value_t = 8192)]
    context: usize,
    #[arg(long, default_value_t = 512)]
    batch: usize,
    /// MTP layer GGUF (the model itself if it has one); enables speculative decoding.
    #[arg(long)]
    mtp: Option<PathBuf>,
    /// Drafted tokens per speculative round.
    #[arg(long, default_value_t = 3)]
    draft_tokens: usize,
    /// Largest rollback batch (default: draft tokens + 1 with --mtp, else 0).
    #[arg(long)]
    verify: Option<usize>,
}
impl QwenRun {
    fn chain(&self, model: &Path) -> Result<qwen::net::Chain> {
        let devices = qwen::chat::devices(self.cpu, self.gpus.clone())?;
        let draft = self.mtp.is_some();
        let verify = self.verify.unwrap_or(if draft { self.draft_tokens + 1 } else { 0 });
        qwen::net::Chain::new(model, devices, &self.nodes, &self.cuts, self.context, self.batch, verify, self.mtp.as_deref(), draft)
    }
    fn spec(&self, chain: &qwen::net::Chain) -> Result<Option<qwen::spec::Spec>> {
        self.mtp
            .as_ref()
            .map(|_| qwen::spec::Spec::new(chain, self.draft_tokens))
            .transpose()
    }
}

/// The 14-token ChatML reference prompt ("Hello there", thinking off) used for activation dumps.
const QWEN_REFERENCE: [u32; 14] = [
    248045, 846, 198, 9419, 1017, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
];

fn info(path: &Path, layers: &[u32]) -> Result<()> {
    let g = gguf::Gguf::open(path)?;
    println!(
        "{}: GGUF v{}, {} tensors, {:.2} GiB",
        path.display(),
        g.version,
        g.tensors.len(),
        g.file_size() as f64 / (1u64 << 30) as f64
    );

    println!("\nmetadata:");
    for (k, v) in &g.meta {
        println!("  {k} = {v}");
    }

    let mut by_type: BTreeMap<gguf::GgmlType, (usize, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let e = by_type.entry(t.ty).or_default();
        e.0 += 1;
        e.1 += t.size;
    }
    println!("\nby type:");
    for (ty, (n, bytes)) in by_type {
        println!(
            "  {ty:>6}: {n:4} tensors, {:8.1} MiB",
            bytes as f64 / (1u64 << 20) as f64
        );
    }

    println!("\ntensors (ggml dims: row length first):");
    for t in &g.tensors {
        let layer = t
            .name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|n| n.parse::<u32>().ok());
        if layer.is_none_or(|l| layers.contains(&l)) {
            println!(
                "  {:40} {:>6} {:>20} {:9.2} MiB",
                t.name,
                t.ty.to_string(),
                format!("{:?}", t.dims),
                t.size as f64 / (1u64 << 20) as f64
            );
        }
    }
    Ok(())
}

fn tensor(path: &Path, pattern: &str, n: usize) -> Result<()> {
    let g = gguf::Gguf::open(path)?;
    let (pre, post) = pattern.split_once('*').unwrap_or((pattern, ""));
    for t in &g.tensors {
        let matches = if pattern.contains('*') {
            t.name.starts_with(pre)
                && t.name.ends_with(post)
                && t.name.len() > pre.len() + post.len()
        } else {
            t.name == pattern
        };
        if !matches {
            continue;
        }
        anyhow::ensure!(
            t.ty == gguf::GgmlType::F32,
            "{} is {}, only F32 is supported",
            t.name,
            t.ty
        );
        let v: Vec<f32> = g
            .data(t)
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let (mut lo, mut hi, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
        for &x in &v {
            lo = lo.min(x);
            hi = hi.max(x);
            sum += x as f64;
        }
        let head: Vec<String> = v.iter().take(n).map(|x| format!("{x:.6}")).collect();
        println!(
            "{} {:?}: min {lo:.6} max {hi:.6} mean {:.6} | {}",
            t.name,
            t.dims,
            sum / v.len() as f64,
            head.join(" ")
        );
    }
    Ok(())
}

fn vocab(path: &Path) -> Result<()> {
    let g = gguf::Gguf::open(path)?;
    let tokens = g.get("tokenizer.ggml.tokens")?.as_arr().unwrap_or_default();
    let types = g
        .get("tokenizer.ggml.token_type")?
        .as_arr()
        .unwrap_or_default();
    let (mut unused, mut bytes) = (0, 0);
    for (id, (t, ty)) in tokens.iter().zip(types).enumerate() {
        let (t, ty) = (t.as_str().unwrap_or(""), ty.as_u64().unwrap_or(0));
        if ty == 1 {
            continue;
        }
        if t.starts_with("<unused") {
            unused += 1;
        } else if t.starts_with("<0x") && t.len() == 6 {
            bytes += 1;
        } else {
            println!("{id:7} type {ty} {t:?}");
        }
    }
    println!("(skipped {unused} <unusedN> and {bytes} byte tokens; types: 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte)");
    Ok(())
}

fn validate_draft(path: Option<&std::path::PathBuf>, k: usize, batch: usize) -> Result<()> {
    if path.is_some() {
        anyhow::ensure!((1..=7).contains(&k), "draft-tokens must be 1..7");
        anyhow::ensure!(batch > k, "batch must be at least draft-tokens + 1");
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve(options) => server::serve(options),
        Cmd::Devices => gpu::report(),
        Cmd::MatvecBench {
            model,
            iterations,
            batch,
            kernel,
            tensor,
            warmup_ms,
        } => micro::bench(
            &model,
            iterations,
            batch,
            kernel.as_deref(),
            &tensor,
            warmup_ms,
        ),
        Cmd::TopkBench { iterations, batch } => micro::selection(iterations, batch),
        Cmd::NormBench { iterations } => micro::norm(iterations),
        Cmd::AttentionBench {
            iterations,
            tokens,
            partitions,
            kernel,
        } => micro::attention(iterations, tokens, partitions, &kernel),
        Cmd::PrefillAttentionBench {
            position,
            batch,
            iterations,
            sliding,
        } => prefill_micro::bench(position, batch, iterations, sliding),
        Cmd::P2pTest { rounds, elements } => visibility::test(rounds, elements),
        Cmd::Render {
            model,
            prompt,
            fixture,
            system,
            thinking,
        } => {
            let g = gguf::Gguf::open(&model)?;
            let t = tokenizer::Tokenizer::new(&g)?;
            if let Some(path) = fixture {
                return t.chat_fixture(&path);
            }
            let mut m = Vec::new();
            if let Some(s) = system {
                m.push(serde_json::json!({"role":"system","content":s}));
            }
            m.push(serde_json::json!({"role":"user","content":prompt.unwrap()}));
            let s = t.render(&m, thinking)?;
            println!("{s:?}");
            println!("{:?}", t.encode(&s, true, true));
            Ok(())
        }
        Cmd::Chat {
            model,
            prompt,
            system,
            thinking,
            max_tokens,
            context,
            batch,
            temperature,
            top_k,
            top_p,
            seed,
            cpu_reduce,
            profile,
            draft,
            draft_tokens,
        } => {
            validate_draft(draft.as_ref(), draft_tokens, batch)?;
            let mut sampler = generation::Sampler::new(temperature, top_k, top_p, seed)?;
            let mut e = model::Engine::new(&model, context, batch, cpu_reduce)?;
            e.set_top_k(if temperature == 0. { 1 } else { top_k })?;
            let t = tokenizer::Tokenizer::new(&e.gguf)?;
            let mut drafter = draft
                .as_ref()
                .map(|p| {
                    assistant::Assistant::new(p, &e, if temperature == 0. { 1 } else { top_k })
                })
                .transpose()?;
            generation::chat(
                &mut e,
                &t,
                prompt,
                system,
                thinking,
                max_tokens,
                batch,
                &mut sampler,
                drafter.as_mut(),
                draft_tokens,
            )?;
            if let Some(d) = drafter {
                eprintln!(
                    "MTP: accepted {}/{} proposals in {} rounds; draft {:.2} ms/proposal, verify {:.2} ms/round",
                    d.accepted, d.proposed, d.rounds,d.draft_ms/d.proposed.max(1) as f64,d.verify_ms/d.rounds.max(1) as f64
                );
            }
            if let Some(p) = profile {
                generation::profile(&e, &p)?;
            }
            Ok(())
        }
        Cmd::CheckBatches {
            model,
            tokens,
            batch,
            draft,
        } => {
            anyhow::ensure!(
                tokens > 0 && batch >= 8,
                "check-batches needs tokens > 0 and batch >= 8"
            );
            let mut e = model::Engine::new(&model, tokens + 128, batch, false)?;
            let tokenizer = tokenizer::Tokenizer::new(&e.gguf)?;
            let body = tokenizer.encode("The sun heats the surface of the Earth. Water evaporates, condenses into clouds, and falls as rain. ", false, false);
            let tail = tokenizer.encode("\nExplain the water cycle in detail, with examples, in English.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>", false, true);
            let mut ids = vec![2, 105, 2364, 107];
            anyhow::ensure!(
                tokens > ids.len() + tail.len(),
                "token count too small for a rendered chat"
            );
            ids.extend((0..tokens - ids.len() - tail.len()).map(|i| body[i % body.len()]));
            ids.extend(tail);
            let serial = generation::prefill(&mut e, &ids, 1)?;
            let mut serial_continuation = Vec::new();
            let mut pending = serial[0].1;
            for _ in 0..64 {
                serial_continuation.push(pending);
                pending = e.forward(&[pending], None, false)?.remove(0)[0].1;
            }
            e.pos = 0;
            let parallel = generation::prefill(&mut e, &ids, batch)?;
            let a: std::collections::HashMap<_, _> =
                serial.iter().map(|(v, id)| (*id, *v)).collect();
            let mut compared = 0;
            let mut max = 0f32;
            let mut diff = 0f64;
            let mut energy = 0f64;
            for (v, id) in &parallel {
                if let Some(w) = a.get(id) {
                    max = max.max((v - w).abs());
                    compared += 1;
                    diff += (v - w).powi(2) as f64;
                    energy += (*w as f64).powi(2);
                }
            }
            anyhow::ensure!(
                serial[0].1 == parallel[0].1 && compared >= 60 && (diff / energy).sqrt() < 0.01,
                "batch mismatch: top1 {} vs {}, overlap {compared}, max error {max}",
                serial[0].1,
                parallel[0].1
            );
            // Isolate verification/rollback from prefill's accumulation-order differences.
            let mut expected = Vec::new();
            let mut pending = parallel[0].1;
            for _ in 0..64 {
                expected.push(pending);
                pending = e.forward(&[pending], None, false)?.remove(0)[0].1;
            }
            let serial_prefix = expected
                .iter()
                .zip(&serial_continuation)
                .take_while(|(a, b)| a == b)
                .count();
            check_continuation(&mut e, &ids, batch, &expected, None)?;
            let mut d = draft
                .as_ref()
                .map(|p| assistant::Assistant::new(p, &e, 1))
                .transpose()?;
            if let Some(d) = d.as_mut() {
                check_continuation(&mut e, &ids, batch, &expected, Some(d))?;
            }
            println!(
                "{}",
                serde_json::json!({"tokens":tokens,"batch":batch,"top64_overlap":compared,"max_logit_error":max,"relative_logit_error":(diff/energy).sqrt(),"greedy_tokens":64,"serial_chunked_greedy_common_prefix":serial_prefix,"draft_proposed":d.as_ref().map(|d|d.proposed),"draft_accepted":d.as_ref().map(|d|d.accepted),"pass":true})
            );
            Ok(())
        }
        Cmd::Bench {
            model,
            pp,
            tg,
            batch,
            profile,
            draft,
            draft_tokens,
            progress,
        } => {
            anyhow::ensure!(pp > 0 && tg > 0, "pp and tg must be positive");
            validate_draft(draft.as_ref(), draft_tokens, batch)?;
            let mut e = model::Engine::new(&model, (pp + tg + 1).max(8192), batch, false)?;
            e.set_top_k(1)?;
            let mut drafter = draft
                .as_ref()
                .map(|p| assistant::Assistant::new(p, &e, 1))
                .transpose()?;
            let ids = vec![9259; pp];
            let start = std::time::Instant::now();
            let mut logits = if progress {
                let mut logits = Vec::new();
                for chunk in ids.chunks(batch) {
                    let chunk_start = std::time::Instant::now();
                    logits = e.forward(chunk, None, false)?.remove(0);
                    anyhow::ensure!(logits[0].0.is_finite(), "nonfinite prefill top score");
                    eprintln!(
                        "{}",
                        serde_json::json!({"phase":"prefill","position":e.pos,"chunk_tokens":chunk.len(),"chunk_seconds":chunk_start.elapsed().as_secs_f64(),"elapsed_seconds":start.elapsed().as_secs_f64(),"top1_score":logits[0].0,"top1_id":logits[0].1})
                    );
                }
                logits
            } else {
                generation::prefill(&mut e, &ids, batch)?
            };
            let pp_time = start.elapsed().as_secs_f64();
            if let Some(p) = profile.as_ref() {
                generation::profile(&e, &p.with_extension("prefill.tsv"))?;
            }
            let start = std::time::Instant::now();
            if let Some(d) = drafter.as_mut() {
                let mut sampler = generation::Sampler::new(0., 64, 0.95, 1)?;
                let mut pending = logits[0].1;
                let mut done = 0;
                while done < tg {
                    let k = draft_tokens.min(tg - done - 1);
                    if k > 0 {
                        let (ids, next) = d.step(&mut e, pending, &mut sampler, k)?;
                        done += ids.len();
                        pending = next;
                    } else {
                        pending = e.forward(&[pending], None, false)?.remove(0)[0].1;
                        done += 1;
                    }
                }
            } else {
                for step in 0..tg {
                    logits = e.forward(&[logits[0].1], None, false)?.remove(0);
                    if progress {
                        anyhow::ensure!(logits[0].0.is_finite(), "nonfinite decode top score");
                        eprintln!(
                            "{}",
                            serde_json::json!({"phase":"decode","position":e.pos,"tokens":step+1,"elapsed_seconds":start.elapsed().as_secs_f64(),"top1_score":logits[0].0,"top1_id":logits[0].1})
                        );
                    }
                }
            }
            let tg_time = start.elapsed().as_secs_f64();
            println!(
                "{}",
                serde_json::json!({"pp":pp,"pp_seconds":pp_time,"pp_tok_s":pp as f64/pp_time,"tg":tg,"tg_seconds":tg_time,"tg_tok_s":tg as f64/tg_time,"last_forward_submit_wait_read_ms":e.timing,"device_allocations_mib":e.cards.iter().map(|c|c.gpu.device_bytes.get()as f64/1048576.).collect::<Vec<_>>(),"draft_proposed":drafter.as_ref().map(|d|d.proposed),"draft_accepted":drafter.as_ref().map(|d|d.accepted)})
            );
            if let Some(p) = profile {
                generation::profile(&e, &p)?;
            }
            Ok(())
        }
        Cmd::Tokenize {
            model,
            text,
            fixture,
        } => {
            let g = gguf::Gguf::open(&model)?;
            let t = tokenizer::Tokenizer::new(&g)?;
            if let Some(f) = fixture {
                t.fixture(&f)?;
            }
            if let Some(s) = text {
                println!("{:?}", t.encode(&s, true, true));
            }
            Ok(())
        }
        Cmd::Validate {
            model,
            out,
            greedy,
            cpu_reduce,
            batch,
            draft,
            draft_tokens,
        } => {
            validate_draft(draft.as_ref(), draft_tokens, batch)?;
            let mut e = model::Engine::new(&model, 8192, batch, cpu_reduce)?;
            let prompt = [
                2, 105, 2364, 107, 9259, 993, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
            ];
            let mut logits = Vec::new();
            let chunks: Vec<_> = prompt.chunks(batch).collect();
            for (i, chunk) in chunks.iter().enumerate() {
                logits = e
                    .forward(
                        chunk,
                        if i + 1 == chunks.len() {
                            Some(&out)
                        } else {
                            None
                        },
                        false,
                    )?
                    .remove(0);
            }
            let mut ids = Vec::new();
            if let Some(p) = draft {
                let mut d = assistant::Assistant::new(&p, &e, 1)?;
                let mut sampler = generation::Sampler::new(0., 64, 0.95, 1)?;
                let mut pending = logits[0].1;
                while ids.len() < greedy {
                    let k = draft_tokens.min(greedy - ids.len() - 1);
                    if k > 0 {
                        let (got, next) = d.step(&mut e, pending, &mut sampler, k)?;
                        ids.extend(got);
                        pending = next;
                    } else {
                        ids.push(pending);
                        pending = e.forward(&[pending], None, false)?.remove(0)[0].1;
                    }
                }
                println!("draft acceptance {}/{}", d.accepted, d.proposed);
            } else {
                for _ in 0..greedy {
                    let id = logits[0].1;
                    ids.push(id);
                    logits = e.forward(&[id], None, false)?.remove(0);
                }
            }
            std::fs::write(
                out.join("greedy.txt"),
                ids.iter().map(|x| format!("{x}\n")).collect::<String>(),
            )?;
            println!("greedy: {ids:?}");
            let mut prof = String::from("segment\tgpu0_ms\tgpu1_ms\n");
            for (i, t) in e.profile.iter().enumerate() {
                prof.push_str(&format!("{i}\t{:.6}\t{:.6}\n", t[0], t[1]));
            }
            std::fs::write(out.join("profile.tsv"), prof)?;
            Ok(())
        }
        Cmd::QwenChat {
            model,
            prompt,
            run,
            system,
            thinking,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
        } => {
            let mut sampler = generation::Sampler::new(temperature, top_k, top_p, seed)?;
            let mut chain = run.chain(&model)?;
            let mut spec = run.spec(&chain)?;
            let t = tokenizer::Tokenizer::new(&chain.local.gguf)?;
            qwen::chat::chat(&mut chain, &t, prompt, system, thinking, max_tokens, &mut sampler, spec.as_mut())
        }
        Cmd::QwenBench { model, run, pp, tg } => {
            let mut chain = run.chain(&model)?;
            let mut spec = run.spec(&chain)?;
            let t = tokenizer::Tokenizer::new(&chain.local.gguf)?;
            println!("{}", qwen::chat::bench(&mut chain, &t, pp, tg, spec.as_mut())?);
            Ok(())
        }
        Cmd::QwenCheck {
            model,
            run,
            tokens,
            steps,
        } => {
            let mut chain = run.chain(&model)?;
            let mut spec = run.spec(&chain)?;
            let t = tokenizer::Tokenizer::new(&chain.local.gguf)?;
            let report = qwen::chat::check(&mut chain, &t, tokens, steps, spec.as_mut())?;
            println!("{report}");
            anyhow::ensure!(report["pass"] == true, "qwen check failed");
            Ok(())
        }
        Cmd::QwenValidate {
            model,
            out,
            run,
            greedy,
        } => {
            let mut chain = run.chain(&model)?;
            let ids = qwen::chat::validate(&mut chain, &out, &QWEN_REFERENCE, greedy)?;
            println!("greedy: {ids:?}");
            Ok(())
        }
        Cmd::QwenNode {
            model,
            listen,
            cpu,
            gpus,
            mtp,
        } => qwen::net::serve(&model, mtp, &listen, qwen::chat::devices(cpu, gpus)?),
        Cmd::QwenRef {
            model,
            out,
            greedy,
            tokens,
        } => qwen_ref(&model, &out, greedy, &tokens),
        Cmd::Info { model, layers } => info(&model, &layers),
        Cmd::Tensor { model, name, n } => tensor(&model, &name, n),
        Cmd::Vocab { model } => vocab(&model),
    }
}

fn write_dump(out: &Path, caps: &qwen::cpu::Captures) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut index = String::new();
    for (name, v) in caps {
        std::fs::write(out.join(format!("{name}.f32")), bytemuck::cast_slice(v))?;
        index.push_str(&format!("{name}\t{},1,1,1\t{}\n", v.len(), v.len()));
    }
    std::fs::write(out.join("index.tsv"), index)?;
    Ok(())
}

fn qwen_ref(model: &Path, out: &Path, greedy: usize, tokens: &[u32]) -> Result<()> {
    let g = gguf::Gguf::open(model)?;
    let c = qwen::Config::new(&g)?;
    let mut stage = qwen::cpu::CpuStage::new(&c, 0, c.layers, tokens.len() + greedy + 1, true);
    let mut caps = qwen::cpu::Captures::new();
    let mut x = qwen::embed(&g, &c, tokens)?;
    let start = std::time::Instant::now();
    stage.forward(&g, &mut x, 0, tokens.len(), false, &mut Some(&mut caps))?;
    let logits = stage.head(&g, &x, 1, &mut Some(&mut caps))?.1.remove(0);
    caps.insert("result_output".into(), logits.clone());
    write_dump(out, &caps)?;
    std::fs::write(out.join("logits.f32"), bytemuck::cast_slice(&logits))?;
    eprintln!("prompt: {:.2} s", start.elapsed().as_secs_f64());
    let argmax = |l: &[f32]| (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b }) as u32;
    let mut ids = vec![argmax(&logits)];
    let mut pos = tokens.len();
    while ids.len() < greedy {
        let mut x = qwen::embed(&g, &c, &ids[ids.len() - 1..])?;
        stage.forward(&g, &mut x, pos, 1, false, &mut None)?;
        pos += 1;
        ids.push(argmax(&stage.head(&g, &x, 1, &mut None)?.1[0]));
    }
    std::fs::write(
        out.join("greedy.txt"),
        ids.iter().map(|x| format!("{x}\n")).collect::<String>(),
    )?;
    println!("greedy: {ids:?}");
    Ok(())
}

fn check_continuation(
    e: &mut model::Engine,
    prompt: &[u32],
    batch: usize,
    expected: &[u32],
    mut d: Option<&mut assistant::Assistant>,
) -> Result<()> {
    e.pos = 0;
    let mut pending = generation::prefill(e, prompt, batch)?[0].1;
    let mut got = Vec::new();
    let mut sampler = generation::Sampler::new(0., 64, 0.95, 1)?;
    while got.len() < 64 {
        let n = (64 - got.len()).min(8);
        if let Some(d) = d.as_mut().filter(|_| n > 1) {
            let (chunk, next) = d.step(e, pending, &mut sampler, n - 1)?;
            got.extend(chunk);
            pending = next;
        } else {
            let mut chunk = vec![pending];
            for _ in 1..n {
                chunk.push(expected[got.len() + chunk.len()]);
            }
            let logits = e.forward(&chunk, None, true)?;
            for (i, column) in logits.iter().enumerate() {
                let expect = if got.len() + i + 1 < 64 {
                    Some(expected[got.len() + i + 1])
                } else {
                    None
                };
                if let Some(expect) = expect {
                    anyhow::ensure!(
                        column[0].1 == expect,
                        "verification column mismatch at {}",
                        got.len() + i
                    );
                }
            }
            pending = logits.last().unwrap()[0].1;
            got.extend(chunk);
        }
    }
    anyhow::ensure!(
        got == expected,
        "verification/rollback continuation differs at {:?}",
        got.iter().zip(expected).position(|(a, b)| a != b)
    );
    Ok(())
}
