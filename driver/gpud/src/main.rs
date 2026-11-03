//! gpud: a userspace driver server for the Mac Pro 6,1's FirePro D700s. So far it takes ownership of one GPU through
//! VFIO and probes it (`gpud probe`); see driver/README.md for the plan.

// Off Linux only the tests use the probe.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

mod dma_smoke;
#[cfg(target_os = "linux")]
mod hw;
mod probe;
#[cfg(target_os = "linux")]
mod proto;
#[cfg(target_os = "linux")]
mod serve;
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
    /// Bring the GPU up and serve clients (RADV through libgpud_drm.so) on a Unix socket.
    Serve {
        /// PCI address, e.g. 0000:02:00.0
        address: String,
        #[arg(long, default_value = "/lib/firmware")]
        firmware_dir: String,
        #[arg(long, default_value = "/run/gpud/gpud.sock")]
        socket: String,
    },
    /// Bring the GPU up with Linux's SI code (memory controller, GART, interrupts, GFX and DMA rings) and run the
    /// ring and IB tests.
    Init {
        /// PCI address, e.g. 0000:02:00.0
        address: String,
        /// Directory holding amdgpu/tahiti_*.bin.
        #[arg(long, default_value = "/lib/firmware")]
        firmware_dir: String,
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
        Command::Init { address, firmware_dir } => run_init(&address, &firmware_dir),
        Command::Serve { address, firmware_dir, socket } => run_serve(&address, &firmware_dir, &socket),
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
fn run_init(address: &str, firmware_dir: &str) -> Result<()> {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let (container, dev) = gpud_platform::vfio::open(address.parse()?, 1 << 20..1 << 40)?;
    hw::log("debug", "opened");
    let device = hw::Device::create(Arc::new(dev), Arc::new(container), firmware_dir)?;
    hw::log("debug", "created");
    let t0 = Instant::now();
    device.init()?;
    println!("init: {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
    let failed = device.ib_tests(Duration::from_secs(1));
    for r in device.rings() {
        println!("ring {:<12} {}", r.name, if r.ready { "ready" } else { "not ready" });
    }
    std::thread::sleep(Duration::from_millis(100));
    println!("interrupts: {}", device.interrupts.load(std::sync::atomic::Ordering::Relaxed));
    drop(device);
    hw::log("debug", "closed");
    if failed > 0 {
        anyhow::bail!("{failed} IB tests failed");
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn run_serve(address: &str, firmware_dir: &str, socket: &str) -> Result<()> {
    use std::sync::Arc;

    let (container, dev) = gpud_platform::vfio::open(address.parse()?, 1 << 20..1 << 40)?;
    serve::run(Arc::new(dev), Arc::new(container), firmware_dir, socket)
}

#[cfg(not(target_os = "linux"))]
fn run_serve(_: &str, _: &str, _: &str) -> Result<()> {
    anyhow::bail!("gpud serve needs Linux (VFIO)")
}

#[cfg(not(target_os = "linux"))]
fn run_init(_: &str, _: &str) -> Result<()> {
    anyhow::bail!("gpud init needs Linux (VFIO)")
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
