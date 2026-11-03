//! MADT ("APIC"): the local APICs, I/O APICs and interrupt overrides.

use crate::{u8_at, u16_at, u32_at, u64_at, Error, Sdt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    LocalApic { processor_uid: u32, apic_id: u32, flags: u32 },
    IoApic { id: u8, address: u32, gsi_base: u32 },
    /// An ISA IRQ wired to a different GSI or with other polarity/trigger (MPS INTI flags).
    Override { bus: u8, source: u8, gsi: u32, flags: u16 },
    /// LINT pin of a processor (0xff: all) that is NMI.
    LocalApicNmi { processor_uid: u32, flags: u16, lint: u8 },
    LocalApicAddress { address: u64 },
    Other { kind: u8 },
}

impl Entry {
    /// Local APIC flags: the processor can be used.
    pub const ENABLED: u32 = 1;
    /// Local APIC flags: can be brought online later.
    pub const ONLINE_CAPABLE: u32 = 2;
}

pub struct Madt<'a> {
    pub local_apic_address: u32,
    /// Bit 0: the machine also has 8259 PICs (to be masked).
    pub flags: u32,
    entries: &'a [u8],
}

impl<'a> Madt<'a> {
    pub fn parse(t: &Sdt<'a>) -> Result<Self, Error> {
        if &t.header.signature != b"APIC" {
            return Err(Error::BadSignature);
        }
        let b = t.bytes;
        Ok(Madt { local_apic_address: u32_at(b, 36)?, flags: u32_at(b, 40)?, entries: b.get(44..).ok_or(Error::Truncated)? })
    }

    pub fn entries(&self) -> impl Iterator<Item = Result<Entry, Error>> + 'a {
        let mut rest = self.entries;
        core::iter::from_fn(move || {
            if rest.is_empty() {
                return None;
            }
            let r = (|| {
                let kind = u8_at(rest, 0)?;
                let len = u8_at(rest, 1)? as usize;
                let e = rest.get(..len.max(2)).ok_or(Error::Truncated)?;
                if len < 2 {
                    return Err(Error::Truncated);
                }
                let entry = match kind {
                    0 => Entry::LocalApic { processor_uid: u8_at(e, 2)? as u32, apic_id: u8_at(e, 3)? as u32, flags: u32_at(e, 4)? },
                    1 => Entry::IoApic { id: u8_at(e, 2)?, address: u32_at(e, 4)?, gsi_base: u32_at(e, 8)? },
                    2 => Entry::Override { bus: u8_at(e, 2)?, source: u8_at(e, 3)?, gsi: u32_at(e, 4)?, flags: u16_at(e, 8)? },
                    4 => Entry::LocalApicNmi { processor_uid: u8_at(e, 2)? as u32, flags: u16_at(e, 3)?, lint: u8_at(e, 5)? },
                    5 => Entry::LocalApicAddress { address: u64_at(e, 4)? },
                    9 => Entry::LocalApic { apic_id: u32_at(e, 4)?, flags: u32_at(e, 8)?, processor_uid: u32_at(e, 12)? },
                    0x0a => Entry::LocalApicNmi { flags: u16_at(e, 2)?, processor_uid: u32_at(e, 4)?, lint: u8_at(e, 8)? },
                    kind => Entry::Other { kind },
                };
                rest = &rest[len..];
                Ok(entry)
            })();
            if r.is_err() {
                rest = &[];
            }
            Some(r)
        })
    }

    /// The APIC IDs of the processors that are enabled, in table order (the first is normally the BSP).
    pub fn cpus(&self) -> impl Iterator<Item = u32> + 'a {
        self.entries().filter_map(|e| match e {
            Ok(Entry::LocalApic { apic_id, flags, .. }) if flags & Entry::ENABLED != 0 => Some(apic_id),
            _ => None,
        })
    }
}
