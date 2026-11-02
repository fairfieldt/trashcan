//! gpud: a userspace driver server for the Mac Pro 6,1's FirePro D700s. So far it takes ownership of one GPU through
//! VFIO and probes it (`gpud probe`); see driver/README.md for the plan.

// Off Linux only the tests use the probe.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

mod dma_smoke;
mod probe;
mod si;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Userspace driver server for the Mac Pro 6,1's FirePro D700s")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Open a GPU bound to vfio-pci and check config space, VBIOS, registers, VRAM, DMA mappings and MSI.
    Probe {
        /// PCI address, e.g. 0000:02:00.0
        address: String,
        /// The ACPI VFCT table holding the VBIOS images.
        #[arg(long, default_value = "/sys/firmware/acpi/tables/VFCT")]
        vfct: PathBuf,
        /// Write the GPU's VBIOS image from VFCT here.
        #[arg(long)]
        dump_vbios: Option<PathBuf>,
        /// Only read: no VRAM test, no MSI route.
        #[arg(long)]
        read_only: bool,
    },
    /// Have the GPU's DMA engine (no microcode needed) write a dword to VRAM and to an IOMMU-mapped host buffer.
    DmaSmoke {
        /// PCI address, e.g. 0000:02:00.0
        address: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Probe { address, vfct, dump_vbios, read_only } => {
            let failed = run_probe(&address, &vfct, dump_vbios, read_only)?;
            if failed > 0 {
                eprintln!("{failed} checks failed");
                std::process::exit(1);
            }
            Ok(())
        }
        Command::DmaSmoke { address } => {
            let failed = run_dma_smoke(&address)?;
            if failed > 0 {
                eprintln!("{failed} checks failed");
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "linux")]
fn run_dma_smoke(address: &str) -> Result<u32> {
    let (container, dev) = gpud_platform::vfio::open(address.parse()?, 1 << 20..1 << 40)?;
    dma_smoke::run(&mut std::io::stdout().lock(), &dev, &container)
}

#[cfg(not(target_os = "linux"))]
fn run_dma_smoke(_: &str) -> Result<u32> {
    anyhow::bail!("gpud dma-smoke needs Linux (VFIO)")
}

#[cfg(target_os = "linux")]
fn run_probe(address: &str, vfct: &PathBuf, dump_vbios: Option<PathBuf>, read_only: bool) -> Result<u32> {
    use gpud_platform::{PciAddress, PciDevice, vfio};
    use std::io::Write;

    let addr: PciAddress = address.parse()?;
    let mut out = std::io::stdout().lock();
    let group = vfio::iommu_group(addr)?;
    writeln!(out, "driver     {}", vfio::driver(addr).as_deref().unwrap_or("none"))?;
    for (a, d) in vfio::group_devices(group)? {
        writeln!(out, "group {group:<4} {a} {}", d.as_deref().unwrap_or("(no driver)"))?;
    }
    // SI's GART and page tables hold 40-bit addresses; IOVA 0 up to 1 MiB stays unmapped to catch stray zeros.
    let (container, dev) = vfio::open(addr, 1 << 20..1 << 40)?;
    let iommu = container.iommu_info().unwrap_or_default();
    writeln!(
        out,
        "vfio       {} regions, {} irq indices, MSI vectors {}, INTx {}, reset {:?}, DMA mappings available {}",
        dev.num_regions(),
        dev.num_irqs(),
        dev.irq_count(1)?,
        dev.irq_count(0)?,
        dev.reset_capability(),
        iommu.dma_mappings_available.map_or("?".into(), |n| n.to_string())
    )?;
    match dev.hot_reset_dependents() {
        Ok(deps) => {
            for (g, a) in deps {
                writeln!(out, "bus reset  would also reset {a} (group {g})")?;
            }
        }
        Err(e) => writeln!(out, "bus reset  unavailable ({e})")?,
    }
    let opts = probe::Options { vfct: std::fs::read(vfct).ok(), dump_vbios, read_only };
    probe::probe(&mut out, &dev, &container, &opts)
}

#[cfg(not(target_os = "linux"))]
fn run_probe(_: &str, _: &PathBuf, _: Option<PathBuf>, _: bool) -> Result<u32> {
    anyhow::bail!("gpud probe needs Linux (VFIO)")
}
