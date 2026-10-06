//! Chat, benchmark and validation commands over a machine chain.
use super::{cpu::Captures, net::Chain, pipeline::Devices, spec::Spec};
use crate::{generation::Sampler, tokenizer::Tokenizer};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
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

pub struct Specials {
    pub stop: Vec<u32>,
    think: Option<u32>,
    end_think: Option<u32>,
}
impl Specials {
    pub fn new(t: &Tokenizer) -> Self {
        Self {
            stop: ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| t.id(s)).collect(),
            think: t.id("<think>"),
            end_think: t.id("</think>"),
        }
    }
}

/// Generate from the sampled, not yet processed `pending` token until a stop token or `max_tokens`,
/// passing each generated token to `emit`. Returns the generated tokens.
pub fn generate(
    e: &mut Chain,
    mut pending: u32,
    max_tokens: usize,
    sampler: &mut Sampler,
    stop: &[u32],
    mut spec: Option<&mut Spec>,
    mut emit: impl FnMut(u32) -> Result<()>,
) -> Result<Vec<u32>> {
    let mut generated = Vec::new();
    while generated.len() < max_tokens && !stop.contains(&pending) && e.pos < e.context {
        let (chunk, next) = match spec.as_deref_mut() {
            Some(s) if max_tokens - generated.len() > 1 => s.step(e, pending, sampler, stop)?,
            _ => {
                generated.push(pending);
                emit(pending)?;
                if generated.len() == max_tokens {
                    break;
                }
                let top = e.forward(&[pending], true, None)?.context("missing candidates")?;
                pending = sampler.sample(&top);
                continue;
            }
        };
        for id in chunk {
            if generated.len() == max_tokens || stop.contains(&id) {
                return Ok(generated);
            }
            generated.push(id);
            emit(id)?;
        }
        pending = next;
    }
    Ok(generated)
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
    mut spec: Option<&mut Spec>,
) -> Result<()> {
    let sp = Specials::new(t);
    let interactive = prompt.is_none();
    let mut pending_prompt = prompt;
    let mut messages = Vec::new();
    if let Some(s) = system {
        messages.push(json!({"role":"system","content":s}));
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
        messages.push(json!({"role":"user","content":user}));
        let ids = t.encode(&t.render(&messages, thinking)?, true, true);
        let start = Instant::now();
        let top = e.prefill(&ids)?;
        let pp = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut in_thought = false;
        let mut bytes = Vec::new();
        let pending = sampler.sample(&top);
        let generated = generate(e, pending, max_tokens, sampler, &sp.stop, spec.as_deref_mut(), |id| {
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
            Ok(())
        })?;
        println!();
        eprintln!(
            "prefill: {} tokens, {:.2} tok/s; decode: {} tokens, {:.2} tok/s",
            ids.len(),
            ids.len() as f64 / pp,
            generated.len(),
            generated.len() as f64 / start.elapsed().as_secs_f64()
        );
        if let Some(s) = spec.as_deref() {
            eprintln!("MTP: {}", s.stats.json());
        }
        messages.push(json!({"role":"assistant","content":t.decode(&generated, false)}));
        if !interactive {
            return Ok(());
        }
    }
}

/// The long consistency prompt: a repeated paragraph in one ChatML user turn, thinking off.
fn check_prompt(t: &Tokenizer, tokens: usize) -> Result<Vec<u32>> {
    let body = t.encode(
        "The sun heats the surface of the Earth. Water evaporates, condenses into clouds, and falls as rain. ",
        false,
        false,
    );
    let head = t.encode("<|im_start|>user\n", false, true);
    let tail = t.encode(
        "\nExplain the water cycle in detail, with examples.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        false,
        true,
    );
    ensure!(tokens > head.len() + tail.len(), "token count too small for a rendered chat");
    let mut ids = head.clone();
    ids.extend((0..tokens - head.len() - tail.len()).map(|i| body[i % body.len()]));
    ids.extend(tail);
    Ok(ids)
}

pub fn bench(e: &mut Chain, t: &Tokenizer, pp: usize, tg: usize, spec: Option<&mut Spec>) -> Result<Value> {
    let ids = check_prompt(t, pp)?;
    let start = Instant::now();
    let top = e.prefill(&ids)?;
    let pp_s = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut sampler = Sampler::new(0., 1, 1., 1)?;
    let speculative = spec.is_some();
    let mut spec = spec;
    let out = generate(e, top[0].1, tg, &mut sampler, &[], spec.as_deref_mut(), |_| Ok(()))?;
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
    Ok(json!({"pp":pp,"pp_seconds":pp_s,"pp_tok_s":pp as f64/pp_s,"tg":out.len(),"tg_seconds":tg_s,
        "tg_tok_s":out.len() as f64/tg_s,"plans":e.plans,"local_device_mib":mib,
        "mtp":spec.filter(|_| speculative).map(|s| s.stats.json()),"text":t.decode(&out, false)}))
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

/// Where two greedy continuations first differ, with the reference's top-two margin at that step.
fn divergence(got: &[u32], want: &[u32], margins: &[f32]) -> Value {
    match got.iter().zip(want).position(|(a, b)| a != b) {
        Some(i) => json!({"index": i, "got": got[i], "want": want[i], "reference_top2_margin": margins.get(i)}),
        None if got.len() < want.len() => json!({"index": got.len(), "short": true}),
        None => Value::Null,
    }
}

/// Greedy consistency of the rollback paths against ordinary decoding: after one prefill and a checkpoint,
/// (1) serial decode, (2) the same after restoring the checkpoint, (3) verification batches of 2..=verify
/// tokens of that continuation with a rollback after every batch, checking each column's top token, and
/// (4) speculative decoding with the MTP layer.
pub fn check(e: &mut Chain, t: &Tokenizer, tokens: usize, steps: usize, spec: Option<&mut Spec>) -> Result<Value> {
    ensure!(e.verify >= 2, "check needs --verify of at least 2");
    let ids = check_prompt(t, tokens)?;
    let start = Instant::now();
    let top = e.prefill(&ids)?;
    let prefill_s = start.elapsed().as_secs_f64();
    let p0 = e.pos;
    e.save(0)?;
    let serial = |e: &mut Chain, first: &[(f32, u32)]| -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        let (mut ids, mut scores, mut margins) = (Vec::new(), Vec::new(), Vec::new());
        let mut top = first.to_vec();
        for i in 0..steps {
            ids.push(top[0].1);
            scores.push(top[0].0);
            margins.push(top[0].0 - top[1].0);
            if i + 1 < steps {
                top = e.forward(&[top[0].1], true, None)?.context("missing candidates")?;
            }
        }
        Ok((ids, scores, margins))
    };
    let start = Instant::now();
    let (expected, scores, margins) = serial(e, &top)?;
    let decode_s = start.elapsed().as_secs_f64();
    e.restore(0, p0)?;
    let (again, again_scores, _) = serial(e, &top)?;
    let restore_exact = again == expected && again_scores == scores;

    // Verification batches over the expected continuation, rolling back to a varying prefix each time.
    e.restore(0, p0)?;
    let mut column_mismatches = Vec::new();
    let mut max_score_error = 0f32;
    let (mut i, mut round) = (0, 0);
    while i + 1 < steps {
        let m = (2 + round % (e.verify - 1)).min(steps - i);
        let cols = e.verify(&expected[i..i + m])?;
        for (j, col) in cols.iter().enumerate().take((steps - 1 - i).min(m)) {
            if col[0].1 != expected[i + j + 1] {
                column_mismatches.push(json!({"index": i + j + 1, "got": col[0].1, "want": expected[i + j + 1], "reference_top2_margin": margins[i + j + 1]}));
            } else {
                max_score_error = max_score_error.max((col[0].0 - scores[i + j + 1]).abs());
            }
        }
        let keep = 1 + round % m;
        e.rollback(keep)?;
        i += keep;
        round += 1;
    }

    let speculative = match spec {
        Some(s) => {
            e.restore(0, p0)?;
            let mut sampler = Sampler::new(0., 1, 1., 1)?;
            let start = Instant::now();
            let got = generate(e, expected[0], steps, &mut sampler, &[], Some(&mut *s), |_| Ok(()))?;
            let s_time = start.elapsed().as_secs_f64();
            json!({"identical": got == expected, "divergence": divergence(&got, &expected, &margins),
                "tok_s": got.len() as f64 / s_time, "stats": s.stats.json()})
        }
        None => Value::Null,
    };
    let pass = restore_exact && column_mismatches.is_empty() && speculative.get("identical").is_none_or(|v| v == true);
    Ok(json!({"prompt_tokens": ids.len(), "steps": steps, "prefill_tok_s": ids.len() as f64 / prefill_s,
        "decode_tok_s": steps as f64 / decode_s, "restore_exact": restore_exact,
        "verify_rounds": round, "column_mismatches": column_mismatches, "max_column_score_error": max_score_error,
        "speculative": speculative, "text": t.decode(&expected, false), "pass": pass}))
}
