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

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Devices => gpu::report(),
        Cmd::Info { model, layers } => info(&model, &layers),
    }
}
