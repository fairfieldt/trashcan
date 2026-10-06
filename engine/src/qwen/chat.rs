//! Chat, benchmark and validation commands over a machine chain.
use super::{cpu::Captures, net::Chain, pipeline::Devices};
use crate::{generation::Sampler, tokenizer::Tokenizer};
use anyhow::{Context, Result};
use std::{
    io::{self, Write},
    path::Path,
    time::Instant,
};

/// All local AMD GPUs unless a CPU stage or an explicit list is requested.
pub fn devices(cpu: bool, gpus: Option<Vec<usize>>) -> Result<Devices> {
    if cpu {
        return Ok(Devices::Cpu);
    }
    if let Some(g) = gpus {
        return Ok(Devices::Gpus(g));
    }
    let inst = crate::gpu::Instance::new()?;
    Ok(Devices::Gpus((0..inst.amd_gpus()?.len()).collect()))
}

struct Specials {
    stop: Vec<u32>,
    think: Option<u32>,
    end_think: Option<u32>,
}
impl Specials {
    fn new(t: &Tokenizer) -> Self {
        Self {
            stop: ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| t.id(s)).collect(),
            think: t.id("<think>"),
            end_think: t.id("</think>"),
        }
    }
}

#[allow(clippy::too_many_arguments)] // CLI fields are passed through together.
pub fn chat(
    e: &mut Chain,
    t: &Tokenizer,
    prompt: Option<String>,
    system: Option<String>,
    thinking: bool,
    max_tokens: usize,
    sampler: &mut Sampler,
) -> Result<()> {
    let sp = Specials::new(t);
    let interactive = prompt.is_none();
    let mut pending_prompt = prompt;
    let mut messages = Vec::new();
    if let Some(s) = system {
        messages.push(serde_json::json!({"role":"system","content":s}));
    }
    loop {
        let user = if let Some(s) = pending_prompt.take() {
            s
        } else {
            print!("\n> ");
            io::stdout().flush()?;
            let mut line = String::new();
            if io::stdin().read_line(&mut line)? == 0 {
                return Ok(());
            }
            let s = line.trim_end().to_owned();
            if s == "/quit" {
                return Ok(());
            }
            if s == "/reset" {
                messages.retain(|x| x["role"] == "system");
                continue;
            }
            s
        };
        messages.push(serde_json::json!({"role":"user","content":user}));
        let ids = t.encode(&t.render(&messages, thinking)?, true, true);
        let start = Instant::now();
        let mut top = e.prefill(&ids)?;
        let pp = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut generated = Vec::new();
        let mut in_thought = false;
        let mut bytes = Vec::new();
        while generated.len() < max_tokens && e.pos < e.context {
            let id = sampler.sample(&top);
            if sp.stop.contains(&id) {
                break;
            }
            generated.push(id);
            if Some(id) == sp.think {
                in_thought = true;
            } else if Some(id) == sp.end_think {
                in_thought = false;
            } else if !in_thought {
                bytes.extend(t.decode_bytes(id, false));
                let valid = match std::str::from_utf8(&bytes) {
                    Ok(_) => bytes.len(),
                    Err(err) => err.valid_up_to(),
                };
                io::stdout().write_all(&bytes[..valid])?;
                io::stdout().flush()?;
                bytes.drain(..valid);
            }
            if generated.len() < max_tokens {
                top = e.forward(&[id], true, None)?.context("missing candidates")?;
            }
        }
        println!();
        eprintln!(
            "prefill: {} tokens, {:.2} tok/s; decode: {} tokens, {:.2} tok/s",
            ids.len(),
            ids.len() as f64 / pp,
            generated.len(),
            generated.len() as f64 / start.elapsed().as_secs_f64()
        );
        messages.push(serde_json::json!({"role":"assistant","content":t.decode(&generated, false)}));
        if !interactive {
            return Ok(());
        }
    }
}

pub fn bench(e: &mut Chain, pp: usize, tg: usize) -> Result<serde_json::Value> {
    let hello = 9419;
    let start = Instant::now();
    let mut top = e.prefill(&vec![hello; pp])?;
    let pp_s = start.elapsed().as_secs_f64();
    let start = Instant::now();
    for _ in 0..tg {
        top = e.forward(&[top[0].1], true, None)?.context("missing candidates")?;
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
    Ok(serde_json::json!({"pp":pp,"pp_seconds":pp_s,"pp_tok_s":pp as f64/pp_s,"tg":tg,"tg_seconds":tg_s,"tg_tok_s":tg as f64/tg_s,"plans":e.plans,"local_device_mib":mib}))
}

/// Run a prompt, dump the last token's activations in the reference format and decode greedily.
pub fn validate(e: &mut Chain, out: &Path, tokens: &[u32], greedy: usize) -> Result<Vec<u32>> {
    let mut caps = Captures::new();
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
    std::fs::write(out.join("greedy.txt"), ids.iter().map(|x| format!("{x}\n")).collect::<String>())?;
    Ok(ids)
}
