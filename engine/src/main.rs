mod gguf;
mod gpu;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Gemma 4 12B inference on two FirePro D700s")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show both GPUs and check every Vulkan feature and extension the engine depends on.
    Devices,
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

fn info(path: &PathBuf, layers: &[u32]) -> Result<()> {
    let g = gguf::Gguf::open(path)?;
    println!("{}: GGUF v{}, {} tensors, {:.2} GiB", path.display(), g.version, g.tensors.len(), g.file_size() as f64 / (1u64 << 30) as f64);

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
        println!("  {ty:>6}: {n:4} tensors, {:8.1} MiB", bytes as f64 / (1u64 << 20) as f64);
    }

    println!("\ntensors (ggml dims: row length first):");
    for t in &g.tensors {
        let layer = t.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|n| n.parse::<u32>().ok());
        if layer.map_or(true, |l| layers.contains(&l)) {
            println!("  {:40} {:>6} {:>20} {:9.2} MiB", t.name, t.ty.to_string(), format!("{:?}", t.dims), t.size as f64 / (1u64 << 20) as f64);
        }
    }
    Ok(())
}

fn tensor(path: &PathBuf, pattern: &str, n: usize) -> Result<()> {
    let g = gguf::Gguf::open(path)?;
    let (pre, post) = pattern.split_once('*').unwrap_or((pattern, ""));
    for t in &g.tensors {
        let matches = if pattern.contains('*') {
            t.name.starts_with(pre) && t.name.ends_with(post) && t.name.len() > pre.len() + post.len()
        } else {
            t.name == pattern
        };
        if !matches {
            continue;
        }
        anyhow::ensure!(t.ty == gguf::GgmlType::F32, "{} is {}, only F32 is supported", t.name, t.ty);
        let v: Vec<f32> = g.data(t).chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let (mut lo, mut hi, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
        for &x in &v {
            lo = lo.min(x);
            hi = hi.max(x);
            sum += x as f64;
        }
        let head: Vec<String> = v.iter().take(n).map(|x| format!("{x:.6}")).collect();
        println!("{} {:?}: min {lo:.6} max {hi:.6} mean {:.6} | {}", t.name, t.dims, sum / v.len() as f64, head.join(" "));
    }
    Ok(())
}

fn vocab(path: &PathBuf) -> Result<()> {
    let g = gguf::Gguf::open(path)?;
    let tokens = g.get("tokenizer.ggml.tokens")?.as_arr().unwrap_or_default();
    let types = g.get("tokenizer.ggml.token_type")?.as_arr().unwrap_or_default();
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

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Devices => gpu::report(),
        Cmd::Info { model, layers } => info(&model, &layers),
        Cmd::Tensor { model, name, n } => tensor(&model, &name, n),
        Cmd::Vocab { model } => vocab(&model),
    }
}
