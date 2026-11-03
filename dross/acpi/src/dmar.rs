//! DMAR: the VT-d remapping hardware units and the memory the firmware keeps DMA going to (RMRRs).

use crate::{u8_at, u16_at, u64_at, Error, Sdt};

pub struct Dmar<'a> {
    /// Physical address bits DMA can use (the field plus one).
    pub host_address_width: u8,
    pub flags: u8,
    structures: &'a [u8],
}

/// One device in a scope: a PCI path from a start bus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scope<'a> {
    /// 1 PCI endpoint, 2 PCI sub-hierarchy, 3 IOAPIC, 4 HPET, 5 ACPI namespace device.
    pub kind: u8,
    pub enumeration_id: u8,
    pub start_bus: u8,
    /// (device, function) pairs, one per bridge on the way down.
    pub path: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Remapping<'a> {
    /// A DMA remapping hardware unit. `include_all`: it covers every device of the segment not claimed by another.
    Unit { segment: u16, base: u64, include_all: bool, scopes: &'a [u8] },
    /// Memory that devices in `scopes` DMA to on their own (USB legacy emulation, typically); must stay identity-mapped.
    Reserved { segment: u16, base: u64, limit: u64, scopes: &'a [u8] },
    /// Root ports that support Address Translation Services.
    Ats { segment: u16, all_ports: bool, scopes: &'a [u8] },
    Other { kind: u16 },
}

impl<'a> Dmar<'a> {
    /// Interrupt remapping is supported.
    pub const INTR_REMAP: u8 = 1;

    pub fn parse(t: &Sdt<'a>) -> Result<Self, Error> {
        if &t.header.signature != b"DMAR" {
            return Err(Error::BadSignature);
        }
        let b = t.bytes;
        Ok(Dmar { host_address_width: u8_at(b, 36)? + 1, flags: u8_at(b, 37)?, structures: b.get(48..).ok_or(Error::Truncated)? })
    }

    pub fn interrupt_remapping(&self) -> bool {
        self.flags & Self::INTR_REMAP != 0
    }

    pub fn structures(&self) -> impl Iterator<Item = Result<Remapping<'a>, Error>> + 'a {
        let mut rest = self.structures;
        core::iter::from_fn(move || {
            if rest.is_empty() {
                return None;
            }
            let r = (|| {
                let kind = u16_at(rest, 0)?;
                let len = u16_at(rest, 2)? as usize;
                if len < 4 {
                    return Err(Error::Truncated);
                }
                let s = rest.get(..len).ok_or(Error::Truncated)?;
                let tail = |at: usize| s.get(at..).ok_or(Error::Truncated);
                let out = match kind {
                    0 => Remapping::Unit { include_all: u8_at(s, 4)? & 1 != 0, segment: u16_at(s, 6)?, base: u64_at(s, 8)?, scopes: tail(16)? },
                    1 => Remapping::Reserved { segment: u16_at(s, 6)?, base: u64_at(s, 8)?, limit: u64_at(s, 16)?, scopes: tail(24)? },
                    2 => Remapping::Ats { all_ports: u8_at(s, 4)? & 1 != 0, segment: u16_at(s, 6)?, scopes: tail(8)? },
                    kind => Remapping::Other { kind },
                };
                rest = &rest[len..];
                Ok(out)
            })();
            if r.is_err() {
                rest = &[];
            }
            Some(r)
        })
    }
}

/// The device scopes of a unit, reserved region or ATS entry.
pub fn scopes(mut b: &[u8]) -> impl Iterator<Item = Scope<'_>> {
    core::iter::from_fn(move || {
        let len = *b.get(1)? as usize;
        let s = b.get(..len).filter(|_| len >= 6)?;
        b = &b[len..];
        Some(Scope { kind: s[0], enumeration_id: s[4], start_bus: s[5], path: &s[6..] })
    })
}
