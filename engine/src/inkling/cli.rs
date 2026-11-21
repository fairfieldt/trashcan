//! `trashcan inkling ...` commands.
use super::{
    chat::{self, Event, Output, Specials},
    group::{Group, Reference, Score},
};
use crate::{generation::Sampler, qwen::tp, tokenizer::Tokenizer};
use anyhow::{ensure, Context, Result};
use clap::{Args, Subcommand};
use serde_json::{json, Value};
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

/// The group a command runs on: this machine's GPUs, and the other machines of a tensor-parallel group.
#[derive(Args, Clone)]
pub struct Run {
    /// Local GPU indices (default: all).
    #[arg(long, value_delimiter = ',')]
    pub gpus: Option<Vec<usize>>,
    /// The other machine of a two-machine group, one HOST:PORT per cable (it runs `inkling node`).
    #[arg(long, value_delimiter = ',')]
    pub tp_connect: Vec<String>,
    /// The other machines of a larger group: MACHINE=HOST:PORT[,HOST:PORT...] per cabled machine.
    #[arg(long)]
    pub tp_peer: Vec<String>,
    #[arg(long, default_value_t = 32768)]
    pub context: usize,
    /// Tokens per prefill batch.
    #[arg(long, default_value_t = 256)]
    pub batch: usize,
}

impl Run {
    pub fn group(&self, model: &Path) -> Result<Group> {
        let gpus = match &self.gpus {
            Some(g) => g.clone(),
            None => (0..crate::gpu::Instance::new()?.amd_gpus()?.len()).collect(),
        };
        let mesh = tp::mesh(&self.tp_connect, &self.tp_peer)?;
        Group::new(model, &gpus, mesh, self.context, self.batch)
    }
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Teacher-forced check against a reference dump (tools/dump-activations with DUMP_FULL=l_out): after every layer
    /// application the residual is replaced by the reference's, so each error is that application's own. Then a
    /// free run over the same prompt compares the greedy tokens with the reference's.
    Validate {
        model: PathBuf,
        reference: PathBuf,
        #[command(flatten)]
        run: Run,
        /// Prompt token ids (whitespace separated); default REFERENCE/../../../fixtures/NAME.ids is not assumed:
        /// give the file.
        #[arg(long)]
        ids: PathBuf,
        #[arg(long, default_value_t = 16)]
        greedy: usize,
        /// Skip the teacher-forced pass (a reference of last-token rows only); prompts may span several batches.
        #[arg(long)]
        free: bool,
    },
    /// Chat (interactive without a prompt).
    Chat {
        model: PathBuf,
        prompt: Option<String>,
        #[command(flatten)]
        run: Run,
        #[arg(long)]
        system: Option<String>,
        /// Thinking effort: none, minimal, low, medium, high (default), xhigh, or a number in [0, 0.99].
        #[arg(long)]
        effort: Option<String>,
        #[arg(long, default_value_t = 1024)]
        max_tokens: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
        #[arg(long, default_value_t = 64)]
        top_k: usize,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Measure prefill and greedy decode.
    Bench {
        model: PathBuf,
        #[command(flatten)]
        run: Run,
        #[arg(long, default_value_t = 512)]
        pp: usize,
        #[arg(long, default_value_t = 128)]
        tg: usize,
        #[arg(long, default_value_t = 1)]
        reps: usize,
    },
    /// Serve this machine's ranks of a tensor-parallel group (every machine but the driver).
    Node {
        model: PathBuf,
        #[arg(long, default_value = "0.0.0.0:7073")]
        listen: String,
        #[arg(long, default_value_t = 1)]
        machine: usize,
        #[arg(long)]
        peer: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        gpus: Option<Vec<usize>>,
    },
    /// Render one user message and print its token ids.
    Tokenize {
        model: PathBuf,
        text: String,
        #[arg(long)]
        effort: Option<String>,
    },
}

fn effort_value(e: &Option<String>) -> Option<Value> {
    e.as_ref().map(|s| s.parse::<f64>().map(|n| json!(n)).unwrap_or_else(|_| json!(s)))
}

fn read_ids(path: &Path) -> Result<Vec<u32>> {
    std::fs::read_to_string(path)?.split_whitespace().map(|s| s.parse().context("token id")).collect()
}

pub fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Validate { model, reference, run, ids, greedy, free } => validate(&model, &reference, &run, &ids, greedy, free),
        Cmd::Chat { model, prompt, run, system, effort, max_tokens, temperature, top_k, top_p, seed } => {
            let mut group = run.group(&model)?;
            let t = Tokenizer::new(&group.gguf)?;
            let mut sampler = Sampler::new(temperature, top_k, top_p, seed)?;
            chat_loop(&mut group, &t, prompt, system, effort_value(&effort), max_tokens, &mut sampler)
        }
        Cmd::Bench { model, run, pp, tg, reps } => {
            let mut group = run.group(&model)?;
            for rep in 0..reps {
                let row = bench(&mut group, pp, tg)?;
                println!("{}", json!({"rep": rep, "pp": pp, "tg": tg, "mib": group.device_mib(), "ranks": group.ranks, "result": row}));
                for (k, n, ms) in group.profile() {
                    println!("profile\t{k}\t{n}\t{ms:.3}");
                }
            }
            Ok(())
        }
        Cmd::Node { model, listen, machine, peer, gpus } => {
            let gpus = match gpus {
                Some(g) => g,
                None => (0..crate::gpu::Instance::new()?.amd_gpus()?.len()).collect(),
            };
            super::group::serve(&model, &gpus, &listen, machine, &peer)
        }
        Cmd::Tokenize { model, text, effort } => {
            let g = crate::gguf::Gguf::open(&model)?;
            let t = Tokenizer::new(&g)?;
            let r = chat::render(&[json!({"role": "user", "content": text})], &[], effort_value(&effort).as_ref())?;
            println!("{r}");
            println!("{:?}", t.encode(&r, false, true));
            Ok(())
        }
    }
}

fn validate(model: &Path, reference: &Path, run: &Run, ids: &Path, greedy: usize, free: bool) -> Result<()> {
    let tokens = read_ids(ids)?;
    let mut group = run.group(model)?;
    if !free {
    ensure!(tokens.len() <= group.max_batch, "the prompt must fit one batch (--batch)");
    let mut r = Reference::new(reference);
    let t = Instant::now();
    let top = group.forward(&tokens, Score::Last, Some(&mut r))?;
    eprintln!("teacher-forced forward of {} tokens in {:.2} s", tokens.len(), t.elapsed().as_secs_f64());
    let mut worst = (String::new(), 0f64);
    for (name, e) in &r.errors {
        println!("{name}\t{e:.3e}");
        if *e > worst.1 {
            worst = (name.clone(), *e);
        }
    }
    println!("worst\t{}\t{:.3e}", worst.0, worst.1);
    let logits = r.read("result_output").ok();
    if let (Some(l), Some(c)) = (logits, top.first()) {
        let best = l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        println!("forced: top {:?} (logit {:.3}); reference top {} (logit {:.3})", c.first(), c.first().map_or(0., |x| x.0), best.0, best.1);
    }
    }
    // Free run: the whole prompt without forcing, then greedy tokens.
    group.reset()?;
    let t = Instant::now();
    let mut top = group.forward(&tokens, Score::Last, None)?;
    eprintln!("free prefill of {} tokens in {:.2} s", tokens.len(), t.elapsed().as_secs_f64());
    if let Ok(l) = Reference::new(reference).read("logits") {
        let c = &top[0];
        let best = l.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
        println!("free: top {:?}; reference top {} (logit {:.3})", c.first(), best.0, best.1);
    }
    let mut ours = Vec::new();
    for _ in 0..greedy {
        let tok = top[0][0].1;
        ours.push(tok);
        top = group.forward(&[tok], Score::Last, None)?;
    }
    let want: Vec<u32> = std::fs::read_to_string(reference.join("greedy.txt")).unwrap_or_default().split_whitespace().filter_map(|s| s.parse().ok()).collect();
    let same = ours.iter().zip(&want).take_while(|(a, b)| a == b).count();
    println!("greedy ours      {ours:?}");
    println!("greedy reference {:?}", &want[..want.len().min(greedy)]);
    println!("greedy agree for {same} of {}", greedy.min(want.len()));
    Ok(())
}

/// Prefill `pp` tokens, then decode `tg` greedily; tokens per second of each.
pub fn bench(group: &mut Group, pp: usize, tg: usize) -> Result<Value> {
    group.reset()?;
    let prompt: Vec<u32> = (0..pp).map(|i| 1000 + (i as u32 * 7919) % 150000).collect();
    let t = Instant::now();
    let top = group.forward(&prompt, Score::Last, None)?;
    let pp_s = t.elapsed().as_secs_f64();
    let mut tok = top[0][0].1;
    let before = group.seconds;
    let t = Instant::now();
    for _ in 0..tg {
        tok = group.forward(&[tok], Score::Last, None)?[0][0].1;
    }
    let tg_s = t.elapsed().as_secs_f64();
    let after = group.seconds;
    let per = |a: f64, b: f64| (a - b) / tg as f64 * 1e3;
    Ok(json!({
        "pp_tok_s": pp as f64 / pp_s,
        "tg_tok_s": tg as f64 / tg_s,
        "tg_ms": tg_s / tg as f64 * 1e3,
        "tg_gpu_wait_ms": per(after.gpu, before.gpu),
        "tg_exchange_ms": per(after.exchange, before.exchange),
        "tg_attn_wait_ms": per(after.gpu_attn, before.gpu_attn),
        "tg_moe_wait_ms": per(after.gpu_moe, before.gpu_moe),
        "tg_host_ms": per(after.host, before.host) - per(after.gpu, before.gpu) - per(after.exchange, before.exchange),
    }))
}

fn chat_loop(group: &mut Group, t: &Tokenizer, prompt: Option<String>, system: Option<String>, effort: Option<Value>, max_tokens: usize, sampler: &mut Sampler) -> Result<()> {
    let sp = Specials::new(t)?;
    let mut messages: Vec<Value> = system.into_iter().map(|s| json!({"role": "system", "content": s})).collect();
    let interactive = prompt.is_none();
    let mut next = prompt;
    loop {
        let user = match next.take() {
            Some(p) => p,
            None if interactive => {
                print!("> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line)? == 0 {
                    return Ok(());
                }
                let line = line.trim().to_owned();
                if line == "/quit" {
                    return Ok(());
                }
                if line == "/reset" {
                    messages.retain(|m| m["role"] == "system");
                    continue;
                }
                line
            }
            None => return Ok(()),
        };
        messages.push(json!({"role": "user", "content": user}));
        let rendered = chat::render(&messages, &[], effort.as_ref())?;
        let ids = t.encode(&rendered, false, true);
        group.reset()?;
        let start = Instant::now();
        let mut top = group.forward(&ids, Score::Last, None)?;
        let prefill = start.elapsed().as_secs_f64();
        let mut out = Output::new(sp);
        let start = Instant::now();
        let mut n = 0;
        let mut in_thinking = false;
        while n < max_tokens {
            let tok = sampler.sample(&top[0]);
            n += 1;
            match out.token(tok, &t.decode_bytes(tok, false)) {
                Some(Event::Thinking(s)) => {
                    if !in_thinking {
                        print!("\x1b[2m");
                        in_thinking = true;
                    }
                    print!("{s}");
                }
                Some(Event::Text(s)) => {
                    if in_thinking {
                        print!("\x1b[0m\n");
                        in_thinking = false;
                    }
                    print!("{s}");
                }
                Some(Event::Tool(name, args)) => print!("\n[tool call {name} {args}]\n"),
                None => {}
            }
            std::io::stdout().flush()?;
            if out.done || group.pos + 1 >= group.context {
                break;
            }
            top = group.forward(&[tok], Score::Last, None)?;
        }
        let dt = start.elapsed().as_secs_f64();
        println!("\x1b[0m");
        eprintln!("[{} prompt tokens in {:.2} s, {n} tokens at {:.1} tok/s]", ids.len(), prefill, n as f64 / dt);
        let mut reply = json!({"role": "assistant", "content": out.text});
        if !out.thinking.is_empty() {
            reply["reasoning_content"] = json!(out.thinking);
        }
        messages.push(reply);
        if !interactive {
            return Ok(());
        }
    }
}
