//! The static ACPI tables the kernel needs, parsed from byte slices: no AML interpreter, no allocation.
//!
//! The kernel reads tables through its direct map and hands each one here as a slice of its full length;
//! the tests run the same code on the Mac Pro 6,1's own tables (`tests/macpro61`).
#![no_std]

pub mod dmar;
pub mod fadt;
pub mod hpet;
pub mod madt;
pub mod mcfg;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Shorter than its header or its own length field says.
    Truncated,
    BadSignature,
    BadChecksum,
}

pub(crate) fn u8_at(b: &[u8], off: usize) -> Result<u8, Error> {
    b.get(off).copied().ok_or(Error::Truncated)
}

pub(crate) fn u16_at(b: &[u8], off: usize) -> Result<u16, Error> {
    Ok(u16::from_le_bytes(b.get(off..off + 2).ok_or(Error::Truncated)?.try_into().unwrap()))
}

pub(crate) fn u32_at(b: &[u8], off: usize) -> Result<u32, Error> {
    Ok(u32::from_le_bytes(b.get(off..off + 4).ok_or(Error::Truncated)?.try_into().unwrap()))
}

pub(crate) fn u64_at(b: &[u8], off: usize) -> Result<u64, Error> {
    Ok(u64::from_le_bytes(b.get(off..off + 8).ok_or(Error::Truncated)?.try_into().unwrap()))
}

fn checksum_ok(b: &[u8]) -> bool {
    b.iter().fold(0u8, |a, &x| a.wrapping_add(x)) == 0
}

/// The Root System Description Pointer (ACPI 2.0+ form).
#[derive(Clone, Copy, Debug)]
pub struct Rsdp {
    pub revision: u8,
    pub oem_id: [u8; 6],
    pub rsdt: u32,
    /// 0 for an ACPI 1.0 RSDP.
    pub xsdt: u64,
}

impl Rsdp {
    pub const LEN_V1: usize = 20;
    pub const LEN_V2: usize = 36;

    /// `b` must hold at least 20 bytes; 36 if the RSDP is revision 2 or later.
    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.get(..8) != Some(b"RSD PTR ") {
            return Err(Error::BadSignature);
        }
        if !checksum_ok(b.get(..Self::LEN_V1).ok_or(Error::Truncated)?) {
            return Err(Error::BadChecksum);
        }
        let revision = u8_at(b, 15)?;
        let mut xsdt = 0;
        if revision >= 2 {
            let len = u32_at(b, 20)? as usize;
            if !checksum_ok(b.get(..len.max(Self::LEN_V2)).ok_or(Error::Truncated)?) {
                return Err(Error::BadChecksum);
            }
            xsdt = u64_at(b, 24)?;
        }
        Ok(Rsdp { revision, oem_id: b[9..15].try_into().unwrap(), rsdt: u32_at(b, 16)?, xsdt })
    }
}

/// The 36-byte header every system description table starts with.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub signature: [u8; 4],
    pub length: u32,
    pub revision: u8,
    pub oem_id: [u8; 6],
    pub oem_table_id: [u8; 8],
}

impl Header {
    pub const LEN: usize = 36;

    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN {
            return Err(Error::Truncated);
        }
        Ok(Header {
            signature: b[0..4].try_into().unwrap(),
            length: u32_at(b, 4)?,
            revision: b[8],
            oem_id: b[10..16].try_into().unwrap(),
            oem_table_id: b[16..24].try_into().unwrap(),
        })
    }
}

/// A whole table: header checked, `bytes` cut to the header's length.
#[derive(Clone, Copy, Debug)]
pub struct Sdt<'a> {
    pub header: Header,
    pub bytes: &'a [u8],
    /// Tables with a bad checksum are still returned (Linux uses them too, with a warning); the kernel logs it.
    pub checksum_ok: bool,
}

impl<'a> Sdt<'a> {
    pub fn parse(b: &'a [u8]) -> Result<Self, Error> {
        let header = Header::parse(b)?;
        let len = header.length as usize;
        if len < Header::LEN {
            return Err(Error::Truncated);
        }
        let bytes = b.get(..len).ok_or(Error::Truncated)?;
        Ok(Sdt { header, bytes, checksum_ok: checksum_ok(bytes) })
    }

    pub fn expect(b: &'a [u8], signature: &[u8; 4]) -> Result<Self, Error> {
        let t = Self::parse(b)?;
        if &t.header.signature != signature {
            return Err(Error::BadSignature);
        }
        Ok(t)
    }

    pub fn body(&self) -> &'a [u8] {
        &self.bytes[Header::LEN..]
    }
}

/// The physical addresses of the tables the XSDT (8-byte entries) or RSDT (4-byte entries) lists.
pub fn root_entries<'a>(root: &Sdt<'a>) -> impl Iterator<Item = u64> + 'a {
    let wide = &root.header.signature == b"XSDT";
    let step = if wide { 8 } else { 4 };
    root.body()
        .chunks_exact(step)
        .map(move |c| if wide { u64::from_le_bytes(c.try_into().unwrap()) } else { u32::from_le_bytes(c.try_into().unwrap()) as u64 })
}

/// A Generic Address Structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gas {
    /// 0 system memory, 1 system I/O, 2 PCI config.
    pub space: u8,
    pub bit_width: u8,
    pub bit_offset: u8,
    pub access_size: u8,
    pub address: u64,
}

impl Gas {
    pub const MEMORY: u8 = 0;
    pub const IO: u8 = 1;

    pub(crate) fn parse(b: &[u8], off: usize) -> Result<Self, Error> {
        Ok(Gas {
            space: u8_at(b, off)?,
            bit_width: u8_at(b, off + 1)?,
            bit_offset: u8_at(b, off + 2)?,
            access_size: u8_at(b, off + 3)?,
            address: u64_at(b, off + 4)?,
        })
    }
}

/// The sleep type values of `\_S5` (soft off) from the DSDT, without an AML interpreter: the usual pattern
/// `NameOp "_S5_" PackageOp PkgLength NumElements (BytePrefix n | ZeroOp | OneOp)...`.
/// Returns (SLP_TYPa, SLP_TYPb).
pub fn s5_sleep_type(dsdt: &[u8]) -> Option<(u8, u8)> {
    let at = dsdt.windows(4).position(|w| w == b"_S5_")?;
    let mut i = at + 4;
    if *dsdt.get(i)? != 0x12 {
        return None;
    }
    i += 1;
    // PkgLength: bits 6-7 of the lead byte count the bytes that follow.
    i += 1 + (*dsdt.get(i)? >> 6) as usize;
    i += 1; // NumElements
    let mut value = || -> Option<u8> {
        let op = *dsdt.get(i)?;
        let (v, n) = match op {
            0x00 => (0, 1),
            0x01 => (1, 1),
            0x0a => (*dsdt.get(i + 1)?, 2),
            _ => return None,
        };
        i += n;
        Some(v)
    };
    let a = value()?;
    let b = value()?;
    Some((a, b))
}
