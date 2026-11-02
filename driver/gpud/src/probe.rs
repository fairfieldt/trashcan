//! `gpud probe`: what an exclusively owned D700 looks like through the platform interface. Identity, regions and
//! capabilities from config space; the VBIOS from VFCT and the ROM BAR; registers through the BAR2 mapping; VRAM
//! through BAR0 checked against the indirect MM_INDEX/MM_DATA path; DMA mappings in the IOMMU; the MSI route.
//!
//! None of it starts a GPU engine, so device-initiated DMA is not tested here.

use crate::si;
use gpud_platform::dma::iova_align;
use gpud_platform::pci::{
    CAP_ID_EXPRESS, CONFIG_BAR0, CONFIG_COMMAND, CONFIG_DEVICE_ID, CONFIG_REVISION, CONFIG_SUBSYSTEM_ID,
    CONFIG_SUBSYSTEM_VENDOR_ID, CONFIG_VENDOR_ID, capabilities,
};
use gpud_platform::vbios::{atom_info, vfct_images, vfct_select};
use gpud_platform::{DmaAccess, DmaSpace, Error, Mmio, PciDevice, Region};
use std::fmt::Write as _;
use std::io::Write;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::time::Duration;

pub struct Options {
    /// The raw ACPI VFCT table, if the platform has one.
    pub vfct: Option<Vec<u8>>,
    pub dump_vbios: Option<PathBuf>,
    /// Skip everything that writes: VRAM, MM_INDEX, the MSI route.
    pub read_only: bool,
}

/// Prints the probe and returns how many checks failed.
pub fn probe(out: &mut dyn Write, dev: &dyn PciDevice, dma: &dyn DmaSpace, opts: &Options) -> anyhow::Result<u32> {
    let mut failed = 0;
    let vendor = dev.config_u16(CONFIG_VENDOR_ID)?;
    let device = dev.config_u16(CONFIG_DEVICE_ID)?;
    let class = dev.config_u32(CONFIG_REVISION)? >> 8;
    writeln!(
        out,
        "device     {} {vendor:04x}:{device:04x} subsystem {:04x}:{:04x} rev {:02x} class {class:06x}",
        dev.address(),
        dev.config_u16(CONFIG_SUBSYSTEM_VENDOR_ID)?,
        dev.config_u16(CONFIG_SUBSYSTEM_ID)?,
        dev.config_u8(CONFIG_REVISION)?,
    )?;
    failed += check(out, vendor == si::VENDOR_ATI && device == si::DEVICE_TAHITI_XT, "Tahiti XT")?;
    if vendor != si::VENDOR_ATI {
        return Ok(failed);
    }
    let command = dev.config_u16(CONFIG_COMMAND)?;
    writeln!(out, "command    {command:#06x} (memory {}, bus master {})", on(command & 2), on(command & 4))?;

    for bar in 0..6u8 {
        let raw = dev.config_u32(CONFIG_BAR0 + 4 * bar as u16)?;
        let info = dev.region_info(Region::Bar(bar))?;
        if info.size > 0 {
            writeln!(out, "BAR{bar}       {raw:#010x} size {} {}", size(info.size), access(&info))?;
        }
    }
    for region in [Region::Rom, Region::Config] {
        let info = dev.region_info(region)?;
        writeln!(out, "{region:<10} size {} {}", size(info.size), access(&info))?;
    }

    let caps = capabilities(dev)?;
    let mut line = String::new();
    for (id, off) in &caps.standard {
        write!(line, " {}@{off:#x}", cap_name(*id))?;
    }
    writeln!(out, "caps      {line}")?;
    line.clear();
    for (id, off) in &caps.extended {
        write!(line, " {id:#x}@{off:#x}")?;
    }
    writeln!(out, "ext caps  {line}")?;
    if let Some(exp) = caps.find(CAP_ID_EXPRESS) {
        let lnk = dev.config_u16(exp + 0x12)?;
        writeln!(out, "link       gen {} x{}", lnk & 0xf, (lnk >> 4) & 0x3f)?;
    }

    // VBIOS: the VFCT image for exactly this function; then whatever the ROM BAR holds.
    if let Some(table) = &opts.vfct {
        let images = vfct_images(table)?;
        for i in &images {
            let pn = atom_info(i.image).map(|a| a.part_number).unwrap_or_else(|e| format!("({e})"));
            writeln!(
                out,
                "VFCT       bus {:02x} dev {:02x}.{} {:04x}:{:04x} {:04x}:{:04x} {} KiB {pn}",
                i.bus,
                i.device,
                i.function,
                i.vendor_id,
                i.device_id,
                i.subsystem_vendor_id,
                i.subsystem_id,
                i.image.len() >> 10
            )?;
        }
        match vfct_select(&images, dev.address(), vendor, device) {
            Ok(img) => {
                let atom = atom_info(img.image);
                failed += check(
                    out,
                    atom.is_ok(),
                    &format!("VBIOS from VFCT: {}", atom.as_ref().map_or("", |a| &a.part_number)),
                )?;
                if let Some(path) = &opts.dump_vbios {
                    std::fs::write(path, img.image)?;
                    writeln!(out, "           written to {}", path.display())?;
                }
            }
            Err(e) => failed += check(out, false, &format!("VBIOS from VFCT: {e}"))?,
        }
    }
    let rom = dev.region_info(Region::Rom)?;
    if rom.size > 0 && rom.readable {
        let mut head = vec![0u8; rom.size.min(64 << 10) as usize];
        let what = match dev.read(Region::Rom, 0, &mut head) {
            Ok(()) => match atom_info(&head) {
                Ok(a) => format!("ATOM {} ({} KiB)", a.part_number, a.rom_size >> 10),
                Err(_) => format!("no ATOM image (starts {:02x} {:02x})", head[0], head[1]),
            },
            Err(e) => format!("unreadable ({e})"),
        };
        writeln!(out, "ROM BAR    {what}")?;
    }

    // Registers.
    let regs = dev.map(Region::Bar(si::REGISTER_BAR))?;
    let r = |reg: u32| regs.read32(reg as usize * 4);
    for &(name, reg) in si::PROBE_REGISTERS {
        writeln!(out, "{name:<32} {reg:#06X} {:#010x}", r(reg))?;
    }
    let memsize_mib = r(si::CONFIG_MEMSIZE) as u64;
    let (fb_base, fb_size) = si::fb_location(r(si::MC_VM_FB_LOCATION));
    writeln!(out, "VRAM       {memsize_mib} MiB at MC {fb_base:#x}")?;
    failed += check(out, memsize_mib == 6144, "CONFIG_MEMSIZE is 6 GiB (the VBIOS has set up the memory controller)")?;
    failed += check(out, fb_size == memsize_mib << 20, "MC_VM_FB_LOCATION spans CONFIG_MEMSIZE")?;

    if !opts.read_only {
        failed += vram_test(out, dev, &regs, memsize_mib << 20)?;
    }
    failed += dma_test(out, dev, dma)?;
    if !opts.read_only {
        let irq = dev.enable_msi();
        let fired = irq.as_ref().ok().map(|e| e.wait(Some(Duration::from_millis(20))));
        let disabled = dev.disable_irqs();
        let detail = match &fired {
            Some(Ok(n)) => format!("{n} interrupts in 20 ms"),
            Some(Err(Error::Timeout)) => "no interrupts in 20 ms".into(),
            Some(Err(e)) => format!("{e}"),
            None => format!("{}", irq.err().unwrap()),
        };
        failed += check(out, fired.is_some() && disabled.is_ok(), &format!("MSI routed and removed ({detail})"))?;
    }
    Ok(failed)
}

/// Writes a pattern into the last MiB of the BAR0 aperture, reads it back through BAR0 and through MM_INDEX/
/// MM_DATA (which must see the same VRAM offset), checks the last page of VRAM, past the aperture, through
/// MM_INDEX/MM_DATA, and restores what was there.
fn vram_test(out: &mut dyn Write, dev: &dyn PciDevice, regs: &Mmio, vram_size: u64) -> anyhow::Result<u32> {
    let mut failed = 0;
    let fb_en = regs.read32(si::BIF_FB_EN as usize * 4);
    if fb_en & 3 != 3 {
        writeln!(out, "FAIL BIF_FB_EN {fb_en:#x}: host VRAM access is off")?;
        return Ok(1);
    }
    let vram = dev.map(Region::Bar(si::VRAM_BAR))?;
    let off = vram.len() - (1 << 20);
    let mut saved = vec![0u32; 1024];
    vram.read_dwords(off, &mut saved);
    let pattern: Vec<u32> = (0..1024u32).map(|i| i.wrapping_mul(0x9E37_79B9) ^ 0x5A5A_0000 ^ i).collect();
    vram.write_dwords(off, &pattern);
    let mut back = vec![0u32; 1024];
    vram.read_dwords(off, &mut back);
    let through_bar = back == pattern;
    let mm: Vec<u32> = (0..16).map(|i| mm_read(regs, (off + i * 4) as u64)).collect();
    let through_mm = mm[..] == pattern[..16];
    vram.write_dwords(off, &saved);
    writeln!(
        out,
        "{} VRAM through BAR0 at {:#x}: 4 KiB written and read back{}",
        if through_bar { "ok  " } else { "FAIL" },
        off,
        if through_bar { String::new() } else { format!(" (first word {:#x}, wrote {:#x})", back[0], pattern[0]) }
    )?;
    writeln!(
        out,
        "{} MM_INDEX/MM_DATA sees the same VRAM offset{}",
        if through_mm { "ok  " } else { "FAIL" },
        if through_mm { String::new() } else { format!(" (read {:#x}, expected {:#x})", mm[0], pattern[0]) }
    )?;
    failed += !through_bar as u32 + !through_mm as u32;

    let top = vram_size - 4096;
    let old: Vec<u32> = (0..16).map(|i| mm_read(regs, top + i * 4)).collect();
    for i in 0..16u64 {
        mm_write(regs, top + i * 4, pattern[i as usize] ^ 0xffff_ffff);
    }
    let ok = (0..16u64).all(|i| mm_read(regs, top + i * 4) == pattern[i as usize] ^ 0xffff_ffff);
    for i in 0..16u64 {
        mm_write(regs, top + i * 4, old[i as usize]);
    }
    regs.write32(si::MM_INDEX_HI as usize * 4, 0);
    writeln!(
        out,
        "{} VRAM past the aperture (offset {top:#x}) through MM_INDEX/MM_DATA",
        if ok { "ok  " } else { "FAIL" }
    )?;
    failed += !ok as u32;
    Ok(failed)
}

/// VRAM access through the register window, as `amdgpu_device_mm_access` does it.
fn mm_select(regs: &Mmio, pos: u64) {
    regs.write32(si::MM_INDEX as usize * 4, pos as u32 | 0x8000_0000);
    regs.write32(si::MM_INDEX_HI as usize * 4, (pos >> 31) as u32);
}

fn mm_read(regs: &Mmio, pos: u64) -> u32 {
    mm_select(regs, pos);
    regs.read32(si::MM_DATA as usize * 4)
}

fn mm_write(regs: &Mmio, pos: u64, v: u32) {
    mm_select(regs, pos);
    regs.write32(si::MM_DATA as usize * 4, v);
}

/// Maps host buffers of a few sizes into the device's address space and checks where they land; then maps BAR0
/// as a peer would. The device does not touch them.
fn dma_test(out: &mut dyn Write, dev: &dyn PciDevice, dma: &dyn DmaSpace) -> anyhow::Result<u32> {
    let mut failed = 0;
    let ranges = dma.iova_ranges();
    let mut line = String::new();
    for r in &ranges {
        write!(line, " {:#x}-{:#x}", r.start, r.end - 1)?;
    }
    writeln!(out, "IOVA      {line} (page sizes {:#x})", dma.page_sizes())?;
    let mut first = None;
    for size in [4usize << 10, 2 << 20, 64 << 20] {
        let b = match dma.alloc(size, DmaAccess::ReadWrite) {
            Ok(b) => b,
            Err(e) => {
                writeln!(out, "FAIL DMA buffer of {}: {e}", self::size(size as u64))?;
                failed += 1;
                continue;
            }
        };
        first.get_or_insert(b.iova());
        let iova_ok = ranges.iter().any(|r| r.start <= b.iova() && b.iova() + b.len() as u64 <= r.end)
            && b.iova() % iova_align(size) == 0;
        for i in (0..b.len()).step_by(4096) {
            b.write32(i, i as u32 ^ 0xC0DE_0000);
        }
        let cpu_ok = (0..b.len()).step_by(4096).all(|i| b.read32(i) == i as u32 ^ 0xC0DE_0000);
        writeln!(
            out,
            "{} DMA buffer of {} mapped at IOVA {:#x}",
            if iova_ok && cpu_ok { "ok  " } else { "FAIL" },
            self::size(size as u64),
            b.iova()
        )?;
        failed += !(iova_ok && cpu_ok) as u32;
    }
    let again = dma.alloc(4096, DmaAccess::DeviceReads)?;
    let reused = Some(again.iova()) == first;
    writeln!(out, "{} unmapped IOVA reused ({:#x})", if reused { "ok  " } else { "FAIL" }, again.iova())?;
    failed += !reused as u32;
    drop(again);

    // Peer-to-peer later maps the other GPU's BAR0 into the same container; mapping our own shows whether the
    // IOMMU layer accepts device memory at all. Informational: nothing here depends on it yet.
    let vram = dev.map(Region::Bar(si::VRAM_BAR))?;
    // SAFETY: the BAR mapping outlives the DMA mapping (dropped first below).
    let peer = unsafe { dma.map_external(NonNull::new(vram.as_ptr()).unwrap(), 2 << 20, DmaAccess::ReadWrite) };
    match &peer {
        Ok(p) => writeln!(out, "info BAR0 (2 MiB) mapped for peer DMA at IOVA {:#x}", p.iova())?,
        Err(e) => writeln!(out, "info BAR0 not mappable for peer DMA: {e}")?,
    }
    drop(peer);
    drop(vram);
    Ok(failed)
}

fn check(out: &mut dyn Write, ok: bool, what: &str) -> std::io::Result<u32> {
    writeln!(out, "{} {what}", if ok { "ok  " } else { "FAIL" })?;
    Ok(!ok as u32)
}

fn on(bit: u16) -> &'static str {
    if bit != 0 { "on" } else { "off" }
}

fn size(n: u64) -> String {
    match n {
        n if n >= 1 << 30 && n % (1 << 30) == 0 => format!("{} GiB", n >> 30),
        n if n >= 1 << 20 && n % (1 << 20) == 0 => format!("{} MiB", n >> 20),
        n if n >= 1 << 10 && n % (1 << 10) == 0 => format!("{} KiB", n >> 10),
        n => format!("{n} B"),
    }
}

fn access(info: &gpud_platform::RegionInfo) -> String {
    format!(
        "{}{}{}",
        if info.readable { "r" } else { "-" },
        if info.writable { "w" } else { "-" },
        if info.mappable { "m" } else { "-" }
    )
}

fn cap_name(id: u8) -> String {
    match id {
        0x01 => "PM".into(),
        0x05 => "MSI".into(),
        0x09 => "vendor".into(),
        0x10 => "PCIe".into(),
        0x11 => "MSI-X".into(),
        id => format!("{id:#x}"),
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;
    use gpud_platform::mock::{MockDevice, MockDma};
    use gpud_platform::pci::{CAP_ID_MSI, CAP_ID_PM};

    /// A D700 as VFIO shows it after the VBIOS has run, with its registers and config space faked.
    fn d700() -> MockDevice {
        let d = MockDevice::new("0000:02:00.0".parse().unwrap(), 0x1002, 0x6798, &[(0, 4 << 20), (2, 256 << 10)]);
        d.poke_config(0x08, &[0x00, 0x00, 0x00, 0x03]);
        d.poke_config(0x2c, &[0x6b, 0x10, 0x28, 0x01]);
        d.add_capability(CAP_ID_PM, &[3, 0, 0, 0, 0, 0]);
        d.add_capability(CAP_ID_EXPRESS, &{
            let mut b = [0u8; 58];
            b[0x10] = 0x03; // link status at cap+0x12: gen 3
            b[0x11] = 0x01; // x16
            b
        });
        d.add_capability(CAP_ID_MSI, &[0; 12]);
        let regs = d.map(Region::Bar(2)).unwrap();
        regs.write32(si::CONFIG_MEMSIZE as usize * 4, 6144);
        regs.write32(si::MC_VM_FB_LOCATION as usize * 4, 0xf57f_f400);
        d
    }

    #[test]
    fn probe_passes_on_a_posted_d700() {
        let d = d700();
        let dma = MockDma::new(&[0x10_0000..1 << 40]);
        let mut out = Vec::new();
        let opts = Options { vfct: None, dump_vbios: None, read_only: true };
        let failed = probe(&mut out, &d, &dma, &opts).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(failed, 0, "{text}");
        assert!(text.contains("subsystem 106b:0128"), "{text}");
        assert!(text.contains("link       gen 3 x16"), "{text}");
        assert!(text.contains("VRAM       6144 MiB at MC 0xf400000000"), "{text}");
        assert!(text.contains("DMA buffer of 64 MiB mapped at IOVA 0x"), "{text}");
        assert!(dma.mappings().is_empty(), "every DMA buffer was unmapped");
    }

    #[test]
    fn probe_flags_an_unposted_gpu() {
        let d = d700();
        d.map(Region::Bar(2)).unwrap().write32(si::CONFIG_MEMSIZE as usize * 4, 0);
        let dma = MockDma::new(&[0x10_0000..1 << 40]);
        let mut out = Vec::new();
        let opts = Options { vfct: None, dump_vbios: None, read_only: true };
        assert_eq!(probe(&mut out, &d, &dma, &opts).unwrap(), 2);
    }
}
