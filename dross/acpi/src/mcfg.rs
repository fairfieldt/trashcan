//! MCFG: where PCI Express configuration space (ECAM) is.

use crate::{u8_at, u16_at, u64_at, Error, Sdt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ecam {
    pub base: u64,
    pub segment: u16,
    pub start_bus: u8,
    pub end_bus: u8,
}

impl Ecam {
    /// The physical address of a function's 4 KiB of config space.
    pub fn function(&self, bus: u8, dev: u8, func: u8) -> Option<u64> {
        if bus < self.start_bus || bus > self.end_bus || dev > 31 || func > 7 {
            return None;
        }
        Some(self.base + (((bus - self.start_bus) as u64) << 20 | (dev as u64) << 15 | (func as u64) << 12))
    }
}

pub fn ranges<'a>(t: &Sdt<'a>) -> Result<impl Iterator<Item = Ecam> + 'a, Error> {
    if &t.header.signature != b"MCFG" {
        return Err(Error::BadSignature);
    }
    let body = t.bytes.get(44..).ok_or(Error::Truncated)?;
    Ok(body.chunks_exact(16).filter_map(|c| {
        Some(Ecam { base: u64_at(c, 0).ok()?, segment: u16_at(c, 8).ok()?, start_bus: u8_at(c, 10).ok()?, end_bus: u8_at(c, 11).ok()? })
    }))
}
