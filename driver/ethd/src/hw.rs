//! How the driver reaches the chip: its registers (BAR0), its PCI config space, and time. The real backend is any
//! [`PciDevice`] from gpud-platform (VFIO on Linux); tests use the simulated chip in `sim.rs`.

use crate::regs::*;
use anyhow::{Result, bail};
use gpud_platform::{Mmio, PciDevice, Region};
use std::time::{Duration, Instant};

pub trait Hw: Send + Sync {
    fn read32(&self, off: u32) -> u32;
    fn write32(&self, off: u32, value: u32);
    fn config_read32(&self, off: u16) -> Result<u32>;
    fn config_write32(&self, off: u16, value: u32) -> Result<()>;
    fn config_read16(&self, off: u16) -> Result<u16>;
    fn config_write16(&self, off: u16, value: u16) -> Result<()>;
    /// Waits at least `us` microseconds (the simulation doesn't).
    fn delay_us(&self, us: u64) {
        let until = Instant::now() + Duration::from_micros(us);
        if us >= 1000 {
            std::thread::sleep(Duration::from_micros(us));
        } else {
            while Instant::now() < until {
                std::hint::spin_loop();
            }
        }
    }
}

/// Register helpers in tg3's terms.
pub trait HwExt: Hw {
    /// tw32_f: a write, then a read back so it has reached the chip.
    fn write_flush(&self, off: u32, value: u32) {
        self.write32(off, value);
        self.read32(off);
    }

    fn set_bits(&self, off: u32, bits: u32) {
        self.write32(off, self.read32(off) | bits);
    }

    fn clear_bits(&self, off: u32, bits: u32) {
        self.write32(off, self.read32(off) & !bits);
    }

    /// A dword of NIC SRAM through the memory window (tg3_read_mem).
    fn read_mem(&self, off: u32) -> u32 {
        self.write_flush(MEM_WIN_BASE_ADDR, off);
        let v = self.read32(MEM_WIN_DATA);
        self.write_flush(MEM_WIN_BASE_ADDR, 0);
        v
    }

    fn write_mem(&self, off: u32, value: u32) {
        self.write_flush(MEM_WIN_BASE_ADDR, off);
        self.write_flush(MEM_WIN_DATA, value);
        self.write_flush(MEM_WIN_BASE_ADDR, 0);
    }

    /// Polls until `(read32(off) & mask) == want`, every 10 µs, for at most `limit`.
    fn poll(&self, off: u32, mask: u32, want: u32, limit: Duration) -> bool {
        let start = Instant::now();
        loop {
            if self.read32(off) & mask == want {
                return true;
            }
            if start.elapsed() > limit {
                return false;
            }
            self.delay_us(10);
        }
    }
}

impl<T: Hw + ?Sized> HwExt for T {}

/// A function from gpud-platform with BAR0 mapped.
pub struct PciHw {
    pub dev: Box<dyn PciDevice>,
    mmio: Mmio,
}

impl PciHw {
    pub fn new(dev: Box<dyn PciDevice>) -> Result<PciHw> {
        let mmio = dev.map(Region::Bar(0))?;
        if mmio.len() < 0x10000 {
            bail!("BAR0 is {:#x} bytes, expected 64 KiB", mmio.len());
        }
        Ok(PciHw { dev, mmio })
    }
}

impl Hw for PciHw {
    fn read32(&self, off: u32) -> u32 {
        self.mmio.read32(off as usize)
    }

    fn write32(&self, off: u32, value: u32) {
        self.mmio.write32(off as usize, value)
    }

    fn config_read32(&self, off: u16) -> Result<u32> {
        Ok(self.dev.config_u32(off)?)
    }

    fn config_write32(&self, off: u16, value: u32) -> Result<()> {
        Ok(self.dev.write(Region::Config, off as u64, &value.to_le_bytes())?)
    }

    fn config_read16(&self, off: u16) -> Result<u16> {
        Ok(self.dev.config_u16(off)?)
    }

    fn config_write16(&self, off: u16, value: u16) -> Result<()> {
        Ok(self.dev.set_config_u16(off, value)?)
    }
}
