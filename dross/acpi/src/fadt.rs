//! FADT ("FACP"): power management blocks, the reset register and the DSDT.

use crate::{u8_at, u16_at, u32_at, u64_at, Error, Gas, Sdt};

#[derive(Clone, Copy, Debug)]
pub struct Fadt {
    pub dsdt: u64,
    pub sci_irq: u16,
    pub pm1a_event: u32,
    pub pm1a_control: u32,
    pub pm1b_control: u32,
    pub pm_timer: u32,
    /// IA-PC boot architecture flags (bit 3: MSI not supported, bit 4: ASPM controls not supported).
    pub iapc_boot_arch: u16,
    pub flags: u32,
    /// `None` unless the FADT says the reset register is supported (flags bit 10).
    pub reset: Option<(Gas, u8)>,
}

impl Fadt {
    pub const RESET_REG_SUP: u32 = 1 << 10;
    pub const IAPC_MSI_NOT_SUPPORTED: u16 = 1 << 3;

    pub fn parse(t: &Sdt<'_>) -> Result<Self, Error> {
        if &t.header.signature != b"FACP" {
            return Err(Error::BadSignature);
        }
        let b = t.bytes;
        let flags = u32_at(b, 112)?;
        let x_dsdt = if b.len() >= 148 { u64_at(b, 140)? } else { 0 };
        let reset = if flags & Self::RESET_REG_SUP != 0 && b.len() >= 129 { Some((Gas::parse(b, 116)?, u8_at(b, 128)?)) } else { None };
        Ok(Fadt {
            dsdt: if x_dsdt != 0 { x_dsdt } else { u32_at(b, 40)? as u64 },
            sci_irq: u16_at(b, 46)?,
            pm1a_event: u32_at(b, 56)?,
            pm1a_control: u32_at(b, 64)?,
            pm1b_control: u32_at(b, 68)?,
            pm_timer: u32_at(b, 76)?,
            iapc_boot_arch: if b.len() >= 111 { u16_at(b, 109)? } else { 0 },
            flags,
            reset,
        })
    }
}
