use crate::{model::Engine, tokenizer::Tokenizer};
use anyhow::{ensure, Result};
use std::{
    io::{self, Write},
    path::Path,
    time::Instant,
};

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    state: u64,
}
impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Result<Self> {
        ensure!(
            temperature.is_finite()
                && temperature >= 0.
                && top_k > 0
                && top_k <= 64
                && top_p > 0.
                && top_p <= 1.,
            "invalid sampling parameters: temperature >=0, top-k 1..64, top-p (0,1]"
        );
        Ok(Self {
            temperature,
            top_k,
            top_p,
            state: seed,
        })
    }
    pub(crate) fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn probabilities(&self, logits: &[(f32, u32)]) -> Vec<(f64, u32)> {
        if self.temperature == 0. {
            return vec![(1., logits[0].1)];
        }
        let k = self.top_k.min(logits.len());
        let max = logits[0].0;
        let mut probabilities: Vec<_> = logits[..k]
            .iter()
            .map(|(v, id)| (((v - max) / self.temperature).exp() as f64, *id))
            .collect();
        let sum: f64 = probabilities.iter().map(|p| p.0).sum();
        let mut total = 0.;
        let mut keep = k;
        for (i, p) in probabilities.iter().enumerate() {
            total += p.0 / sum;
            if total >= self.top_p as f64 {
                keep = i + 1;
                break;
            }
        }
        probabilities.truncate(keep);
        probabilities.retain(|p| p.0 > 0.);
        let sum: f64 = probabilities.iter().map(|p| p.0).sum();
        for p in &mut probabilities {
            p.0 /= sum;
        }
        probabilities
    }
    pub fn draw(&mut self, probabilities: &[(f64, u32)]) -> u32 {
        if self.temperature == 0. {
            return probabilities[0].1;
        }
        let mut r = self.uniform();
        for &(p, id) in probabilities {
            r -= p;
            if r <= 0. {
                return id;
            }
        }
        probabilities.last().unwrap().1
    }
    pub fn sample(&mut self, logits: &[(f32, u32)]) -> u32 {
        self.draw(&self.probabilities(logits))
    }
}
pub fn prefill(e: &mut Engine, ids: &[u32], batch: usize) -> Result<Vec<(f32, u32)>> {
    let mut logits = Vec::new();
    for chunk in ids.chunks(batch) {
        logits = e.forward(chunk, None, false)?.remove(0);
    }
    Ok(logits)
}
pub fn profile(e: &Engine, path: &Path) -> Result<()> {
    let mut s = String::from("segment\tgpu0_ms\tgpu1_ms\tgap0_ms\tgap1_ms\n");
    for (i, t) in e.profile.iter().enumerate() {
        s.push_str(&format!(
            "{i}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\n",
            t[0], t[1], e.gaps[i][0], e.gaps[i][1]
        ));
    }
    std::fs::write(path, s)?;
    Ok(())
}
#[allow(clippy::too_many_arguments)] // CLI fields, sampler and optional drafter are supplied together.
pub fn chat(
    e: &mut Engine,
    t: &Tokenizer,
    prompt: Option<String>,
    system: Option<String>,
    thinking: bool,
    max_tokens: usize,
    batch: usize,
    sampler: &mut Sampler,
    mut drafter: Option<&mut crate::assistant::Assistant>,
    draft_tokens: usize,
) -> Result<()> {
    let interactive = prompt.is_none();
    let mut pending = prompt;
    let mut messages = Vec::new();
    if let Some(s) = system {
        messages.push(serde_json::json!({"role":"system","content":s}));
    }
    loop {
        let user = if let Some(s) = pending.take() {
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
        let rendered = t.render(&messages, thinking)?;
        let ids = t.encode(&rendered, true, true);
        e.pos = 0;
        if let Some(d) = drafter.as_mut() {
            d.reset();
        }
        let start = Instant::now();
        let mut logits = prefill(e, &ids, batch)?;
        let pp = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut generated = Vec::new();
        let mut in_thought = false;
        let mut bytes = Vec::new();
        let mut pending = sampler.sample(&logits);
        'generate: while generated.len() < max_tokens {
            if matches!(pending, 1 | 106 | 50) {
                break;
            }
            let k = draft_tokens
                .min(max_tokens - generated.len() - 1)
                .min(e.remaining().saturating_sub(1));
            let chunk = if let Some(d) = drafter.as_mut().filter(|_| k > 0) {
                let (ids, next) = d.step(e, pending, sampler, k)?;
                pending = next;
                ids
            } else {
                let id = pending;
                logits = e.forward(&[id], None, false)?.remove(0);
                pending = sampler.sample(&logits);
                vec![id]
            };
            for id in chunk {
                if matches!(id, 1 | 106 | 50) {
                    break 'generate;
                }
                generated.push(id);
                if id == 100 {
                    in_thought = true;
                } else if id == 101 {
                    in_thought = false;
                } else if !in_thought {
                    bytes.extend(t.decode_bytes(id, false));
                    let valid = match std::str::from_utf8(&bytes) {
                        Ok(_) => bytes.len(),
                        Err(err) => err.valid_up_to(),
                    };
                    if valid > 0 {
                        io::stdout().write_all(&bytes[..valid])?;
                        io::stdout().flush()?;
                        bytes.drain(..valid);
                    }
                }
            }
        }
        if !bytes.is_empty() {
            print!("{}", String::from_utf8_lossy(&bytes));
        }
        println!();
        eprintln!(
            "prefill: {} tokens, {:.2} tok/s; decode: {} tokens, {:.2} tok/s",
            ids.len(),
            ids.len() as f64 / pp,
            generated.len(),
            generated.len() as f64 / start.elapsed().as_secs_f64()
        );
        messages.push(serde_json::json!({"role":"assistant","content":t.decode(&generated,true)}));
        if !interactive {
            return Ok(());
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_and_seed() {
        let l = [(4., 2), (3., 5), (1., 9)];
        let mut a = Sampler::new(0., 64, 0.95, 1).unwrap();
        assert_eq!(a.sample(&l), 2);
        let mut a = Sampler::new(1., 3, 0.95, 7).unwrap();
        let mut b = Sampler::new(1., 3, 0.95, 7).unwrap();
        for _ in 0..100 {
            assert_eq!(a.sample(&l), b.sample(&l));
        }
    }
}
