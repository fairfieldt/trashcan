//! `trashcan qwen4 ...` commands.
use anyhow::{Context, Result};
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
