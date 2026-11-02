//! `gpud dma-smoke`: does the GPU's own DMA reach host memory through the IOMMU?
//!
//! SI's async DMA engine runs without microcode, so it can write one dword before any of the driver exists. The
//! ring sits in VRAM (written through BAR0); the engine writes first to VRAM, which shows it runs, then to a host
//! buffer mapped in the IOMMU. The memory controller reaches host memory through its AGP aperture here: MC
//! addresses [0, 4 GiB) pass through untranslated as bus addresses, which the IOMMU translates as IOVAs. A write
//! that goes astray shows up in the kernel log as a DMAR fault with the address the GPU emitted.
//!
//! Register sequence after `si_dma_start` (amdgpu/si_dma.c, DMA0 only, no read pointer write-back) and packets
//! from `amdgpu/sid.h`. This is a check on the platform, not the driver's ring code: milestone 4 brings si_dma.c.

use crate::si;
use gpud_platform::pci::{COMMAND_BUS_MASTER, CONFIG_COMMAND};
use gpud_platform::{DmaAccess, DmaSpace, Mmio, PciDevice, Region};
use std::io::Write;
use std::time::{Duration, Instant};

// oss_1_0_d.h, DMA0 (DMA1 is 0x200 dwords further)
const DMA_GFX_RB_CNTL: u32 = 0x3400;
const DMA_GFX_RB_BASE: u32 = 0x3401;
const DMA_GFX_RB_RPTR: u32 = 0x3402;
const DMA_GFX_RB_WPTR: u32 = 0x3403;
const DMA_GFX_IB_CNTL: u32 = 0x3409;
const DMA_STATUS_REG: u32 = 0x340D;
const DMA_SEM_INCOMPLETE_TIMER_CNTL: u32 = 0x3411;
const DMA_SEM_WAIT_FAIL_TIMER_CNTL: u32 = 0x3412;
const RB_ENABLE: u32 = 1;
const IB_ENABLE: u32 = 1;
const CMD_VMID_FORCE: u32 = 1 << 31;
// gmc_6_0_d.h, bif_3_0_d.h
const MC_VM_AGP_TOP: u32 = 0x080A;
const MC_VM_AGP_BOT: u32 = 0x080B;
const MC_VM_AGP_BASE: u32 = 0x080C;
const MC_VM_MX_L1_TLB_CNTL: u32 = 0x0819;
const HDP_MEM_COHERENCY_FLUSH_CNTL: u32 = 0x1520;
// gmc_v6_0_gart_enable's L1 TLB setting, without enabling any VM context
const L1_TLB_ENABLED: u32 = (0xA << 7) | 1 | 2 | (3 << 3) | (1 << 6);

const fn dma_packet(cmd: u32, n: u32) -> u32 {
    (cmd & 0xf) << 28 | (n & 0xfffff)
}
const DMA_PACKET_WRITE: u32 = 0x2;
const DMA_NOP: u32 = dma_packet(0xf, 0);

const RING_DWORDS: usize = 1024; // RB_SIZE = log2 of this
const RING_OFFSET: usize = 0x0FE0_0000; // in the BAR0 window: VRAM offset 254 MiB, away from the probe's 255 MiB
const VRAM_TARGET: usize = RING_OFFSET + 0x1_0000;
/// Inside the AGP window, below the MSI range, never handed out here.
const UNMAPPED_IOVA: u64 = 0xF000_0000;

struct Dma<'a> {
    regs: &'a Mmio,
    vram: &'a Mmio,
    fb_base: u64,
    wptr: usize, // dwords
}

impl Dma<'_> {
    fn r(&self, reg: u32) -> u32 {
        self.regs.read32(reg as usize * 4)
    }

    fn w(&self, reg: u32, v: u32) {
        self.regs.write32(reg as usize * 4, v)
    }

    fn start(&mut self) {
        self.w(DMA_GFX_RB_CNTL, 0);
        self.w(DMA_SEM_INCOMPLETE_TIMER_CNTL, 0);
        self.w(DMA_SEM_WAIT_FAIL_TIMER_CNTL, 0);
        let rb_cntl = RING_DWORDS.trailing_zeros() << 1;
        self.w(DMA_GFX_RB_CNTL, rb_cntl);
        self.w(DMA_GFX_RB_RPTR, 0);
        self.w(DMA_GFX_RB_WPTR, 0);
        self.w(DMA_GFX_RB_BASE, ((self.fb_base + RING_OFFSET as u64) >> 8) as u32);
        self.w(DMA_GFX_IB_CNTL, IB_ENABLE | CMD_VMID_FORCE);
        self.vram.write_dwords(RING_OFFSET, &[DMA_NOP; RING_DWORDS]);
        self.wptr = 0;
        self.w(DMA_GFX_RB_CNTL, rb_cntl | RB_ENABLE);
    }

    fn stop(&self) {
        self.w(DMA_GFX_RB_CNTL, self.r(DMA_GFX_RB_CNTL) & !RB_ENABLE);
    }

    /// Writes one dword at an MC address and waits for the engine to consume the packet.
    fn write_dword(&mut self, mc: u64, value: u32) -> bool {
        let mut p = vec![dma_packet(DMA_PACKET_WRITE, 1), mc as u32 & !3, (mc >> 32) as u32 & 0xff, value];
        p.resize(16, DMA_NOP); // the ring advances in 16-dword steps (si_dma align_mask 0xf)
        self.vram.write_dwords(RING_OFFSET + self.wptr * 4, &p);
        self.wptr = (self.wptr + p.len()) % RING_DWORDS;
        // The ring was written through the HDP: flush it before the engine reads.
        self.w(HDP_MEM_COHERENCY_FLUSH_CNTL, 1);
        self.r(HDP_MEM_COHERENCY_FLUSH_CNTL);
        self.w(DMA_GFX_RB_WPTR, (self.wptr as u32 * 4) & 0x3fffc);
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            if (self.r(DMA_GFX_RB_RPTR) & 0x3fffc) >> 2 == self.wptr as u32 {
                return true;
            }
            std::hint::spin_loop();
        }
        false
    }
}

fn wait_for(f: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_millis(100);
    while Instant::now() < deadline {
        if f() {
            return true;
        }
    }
    f()
}

/// Returns how many checks failed.
pub fn run(out: &mut dyn Write, dev: &dyn PciDevice, dma: &dyn DmaSpace) -> anyhow::Result<u32> {
    let regs = dev.map(Region::Bar(si::REGISTER_BAR))?;
    let vram = dev.map(Region::Bar(si::VRAM_BAR))?;
    let (fb_base, _) = si::fb_location(regs.read32(si::MC_VM_FB_LOCATION as usize * 4));
    let host = dma.alloc(4096, DmaAccess::DeviceWrites)?;
    host.write32(0, 0xCAFE_DEAD);
    anyhow::ensure!(host.iova() + 4096 <= 4 << 30, "host buffer above the 4 GiB AGP window");

    let r = |reg: u32| regs.read32(reg as usize * 4);
    let saved = [MC_VM_AGP_TOP, MC_VM_AGP_BOT, MC_VM_AGP_BASE, MC_VM_MX_L1_TLB_CNTL].map(|reg| (reg, r(reg)));
    writeln!(
        out,
        "before     DMA_STATUS {:#010x} RB_CNTL {:#010x} L1_TLB_CNTL {:#010x} AGP bot/top/base {:#x}/{:#x}/{:#x}",
        r(DMA_STATUS_REG),
        r(DMA_GFX_RB_CNTL),
        saved[3].1,
        saved[1].1,
        saved[0].1,
        saved[2].1
    )?;
    let command = dev.config_u16(CONFIG_COMMAND)?;
    dev.set_config_u16(CONFIG_COMMAND, command | COMMAND_BUS_MASTER)?;

    let mut failed = 0;
    let mut e = Dma { regs: &regs, vram: &vram, fb_base, wptr: 0 };
    e.start();
    vram.write32(VRAM_TARGET, 0);
    let consumed = e.write_dword(fb_base + VRAM_TARGET as u64, 0xDEAD_BEEF);
    let landed = wait_for(|| vram.read32(VRAM_TARGET) == 0xDEAD_BEEF);
    writeln!(
        out,
        "{} DMA engine wrote VRAM (packet consumed: {consumed}, value {:#x})",
        if landed { "ok  " } else { "FAIL" },
        vram.read32(VRAM_TARGET)
    )?;
    failed += !landed as u32;

    // MC [0, 4 GiB) -> bus addresses [0, 4 GiB): AGP_BOT/TOP in 4 MiB units, AGP_BASE the bus address >> 22.
    let w = |reg: u32, v: u32| regs.write32(reg as usize * 4, v);
    w(MC_VM_AGP_BASE, 0);
    w(MC_VM_AGP_BOT, 0);
    w(MC_VM_AGP_TOP, ((4u64 << 30) >> 22) as u32 - 1);
    let mut host_ok = false;
    for (label, tlb) in [("L1 TLB as left", saved[3].1), ("L1 TLB enabled", L1_TLB_ENABLED)] {
        w(MC_VM_MX_L1_TLB_CNTL, tlb);
        let value = 0x600D_0000 | (tlb & 0xffff);
        let consumed = e.write_dword(host.iova(), value);
        host_ok = wait_for(|| host.read32(0) == value);
        writeln!(
            out,
            "{} DMA engine wrote host memory at IOVA {:#x} through the AGP aperture ({label}: {tlb:#x}; packet consumed: {consumed}, value {:#x})",
            if host_ok { "ok  " } else { "miss" },
            host.iova(),
            host.read32(0)
        )?;
        if host_ok {
            break;
        }
    }
    failed += !host_ok as u32;
    if host_ok {
        // A second buffer at another IOVA, written at its last page.
        let big = dma.alloc(8 << 20, DmaAccess::DeviceWrites)?;
        let at = big.len() - 4096;
        let consumed = e.write_dword(big.iova() + at as u64, 0x5EC0_0D00);
        let ok = wait_for(|| big.read32(at) == 0x5EC0_0D00);
        writeln!(
            out,
            "{} DMA engine wrote host memory at IOVA {:#x} (packet consumed: {consumed}, value {:#x})",
            if ok { "ok  " } else { "FAIL" },
            big.iova() + at as u64,
            big.read32(at)
        )?;
        failed += !ok as u32;
        // Negative controls: nothing is mapped at these, so the IOMMU must drop the writes and log DMAR faults.
        for iova in [host.iova() + 4096, UNMAPPED_IOVA, 0x4000_0000] {
            let consumed = e.write_dword(iova, 0xBAD0_BAD0);
            writeln!(out, "sent       a write to unmapped IOVA {iova:#x} (packet consumed: {consumed})")?;
        }
        let unchanged = host.read32(0) != 0xBAD0_BAD0 && big.read32(at) != 0xBAD0_BAD0;
        writeln!(out, "{} mapped buffers untouched by those writes", if unchanged { "ok  " } else { "FAIL" })?;
        failed += !unchanged as u32;
    }

    e.stop();
    for (reg, v) in saved {
        w(reg, v);
    }
    dev.set_config_u16(CONFIG_COMMAND, command)?;
    writeln!(
        out,
        "after      DMA_STATUS {:#010x} RB_CNTL {:#010x} (stopped, MC registers restored)",
        r(DMA_STATUS_REG),
        r(DMA_GFX_RB_CNTL)
    )?;
    Ok(failed)
}
