//! Finding a GPU's VBIOS: the ACPI VFCT table, which the Mac's firmware fills with one ATOM image per GPU, and
//! checks on an ATOM image.
//!
//! The layouts are those of `VFCT_IMAGE_HEADER` and `UEFI_ACPI_VFCT` in Linux's
//! `drivers/gpu/drm/amd/include/atombios.h`; the checks follow `amdgpu_acpi_vfct_bios`, `check_atom_bios`
//! (`amdgpu/amdgpu_bios.c`) and `atom_get_vbios_pn` (`amdgpu/atom.c`). Unlike amdgpu, an image is only taken for
//! an exact bus/device/function match: on a Mac Pro 6,1 the two D700s carry different images (113-C3861XA and
//! 113-C3861XB), and handing both cards the first image is the bug kernel patch 0001 fixes.

#[allow(unused_imports)]
use crate::prelude::*;
use crate::{Error, PciAddress, Result};

/// One image in the VFCT table.
#[derive(Clone, Debug)]
pub struct VfctImage<'a> {
    pub bus: u32,
    pub device: u32,
    pub function: u32,
    pub vendor_id: u16,
    pub device_id: u16,
    pub subsystem_vendor_id: u16,
    pub subsystem_id: u16,
    pub revision: u32,
    pub image: &'a [u8],
}

const VFCT_LEN: usize = 0x4c; // UEFI_ACPI_VFCT: ACPI header, TableUUID, VBIOSImageOffset, Lib1ImageOffset, Reserved[4]
const IMAGE_HEADER_LEN: usize = 0x1c; // VFCT_IMAGE_HEADER

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

/// Every image in a VFCT table (the raw ACPI table, header included).
pub fn vfct_images(table: &[u8]) -> Result<Vec<VfctImage<'_>>> {
    if table.len() < VFCT_LEN || &table[0..4] != b"VFCT" {
        return Err(Error::Invalid("not a VFCT table".into()));
    }
    let length = u32_at(table, 4) as usize;
    if length < VFCT_LEN || length > table.len() {
        return Err(Error::Invalid(format!("VFCT length {length} for {} bytes", table.len())));
    }
    let table = &table[..length];
    if table.iter().fold(0u8, |s, &b| s.wrapping_add(b)) != 0 {
        return Err(Error::Invalid("VFCT checksum".into()));
    }
    let mut images = Vec::new();
    let mut off = u32_at(table, 0x34) as usize;
    while off < length {
        let h = table
            .get(off..off + IMAGE_HEADER_LEN)
            .ok_or_else(|| Error::Invalid(format!("VFCT image header at {off:#x} truncated")))?;
        let image_len = u32_at(h, 0x18) as usize;
        let start = off + IMAGE_HEADER_LEN;
        let image = table
            .get(start..start + image_len)
            .ok_or_else(|| Error::Invalid(format!("VFCT image at {start:#x} truncated")))?;
        images.push(VfctImage {
            bus: u32_at(h, 0),
            device: u32_at(h, 4),
            function: u32_at(h, 8),
            vendor_id: u16_at(h, 0x0c),
            device_id: u16_at(h, 0x0e),
            subsystem_vendor_id: u16_at(h, 0x10),
            subsystem_id: u16_at(h, 0x12),
            revision: u32_at(h, 0x14),
            image,
        });
        off = start + image_len;
        if image_len == 0 {
            break;
        }
    }
    Ok(images)
}

/// The image for exactly this function and vendor/device ID.
pub fn vfct_select<'a>(
    images: &[VfctImage<'a>],
    addr: PciAddress,
    vendor_id: u16,
    device_id: u16,
) -> Result<VfctImage<'a>> {
    images
        .iter()
        .find(|i| {
            !i.image.is_empty()
                && i.bus == addr.bus as u32
                && i.device == addr.device as u32
                && i.function == addr.function as u32
                && i.vendor_id == vendor_id
                && i.device_id == device_id
        })
        .cloned()
        .ok_or_else(|| Error::NotFound(format!("VFCT image for {addr} ({vendor_id:04x}:{device_id:04x})")))
}

/// What an ATOM image says about itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomInfo {
    /// The ROM's own size field (512-byte units at offset 2).
    pub rom_size: usize,
    /// Offset of the ATOM ROM header (the pointer at 0x48).
    pub header_offset: usize,
    /// The VBIOS part number amdgpu logs as "ATOM BIOS: ...".
    pub part_number: String,
}

const BIOS_STRING_LENGTH: usize = 43; // include/atomfirmware.h
const OFFSET_TO_GET_ATOMBIOS_NUMBER_OF_STRINGS: usize = 0x2f;
const OFFSET_TO_GET_ATOMBIOS_STRING_START: usize = 0x6e;
const OFFSET_TO_VBIOS_PART_NUMBER: usize = 0x80;

pub fn atom_info(rom: &[u8]) -> Result<AtomInfo> {
    if rom.len() < 0x4a || rom[0] != 0x55 || rom[1] != 0xaa {
        return Err(Error::Invalid("no 55 AA ROM signature".into()));
    }
    let header_offset = u16_at(rom, 0x48) as usize;
    let magic = rom.get(header_offset + 4..header_offset + 8).unwrap_or_default();
    if header_offset == 0 || (magic != b"ATOM" && magic != b"MOTA") {
        return Err(Error::Invalid("no ATOM ROM header".into()));
    }
    let mut s = if rom.get(OFFSET_TO_GET_ATOMBIOS_NUMBER_OF_STRINGS).copied().unwrap_or(0) != 0 {
        rom.get(OFFSET_TO_GET_ATOMBIOS_STRING_START..OFFSET_TO_GET_ATOMBIOS_STRING_START + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
            .unwrap_or(0)
    } else {
        OFFSET_TO_VBIOS_PART_NUMBER
    };
    if rom.get(s) == Some(&0) {
        s += 1;
    }
    let part_number = rom
        .get(s..)
        .unwrap_or_default()
        .iter()
        .take(BIOS_STRING_LENGTH)
        .take_while(|&&c| (b' '..=b'z').contains(&c))
        .map(|&c| c as char)
        .collect::<String>()
        .trim_end()
        .to_string();
    Ok(AtomInfo { rom_size: rom[2] as usize * 512, header_offset, part_number })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A minimal ATOM image: signature, size, ROM header pointer, "ATOM", part number string.
    pub(crate) fn atom_image(part_number: &str) -> Vec<u8> {
        let mut rom = vec![0u8; 1024];
        rom[0] = 0x55;
        rom[1] = 0xaa;
        rom[2] = 2; // 1 KiB
        rom[0x48..0x4a].copy_from_slice(&0x100u16.to_le_bytes());
        rom[0x104..0x108].copy_from_slice(b"ATOM");
        rom[OFFSET_TO_GET_ATOMBIOS_NUMBER_OF_STRINGS] = 1;
        rom[OFFSET_TO_GET_ATOMBIOS_STRING_START..OFFSET_TO_GET_ATOMBIOS_STRING_START + 2]
            .copy_from_slice(&0x200u16.to_le_bytes());
        rom[0x200..0x200 + part_number.len()].copy_from_slice(part_number.as_bytes());
        rom
    }

    pub(crate) fn vfct(images: &[(u32, u16, Vec<u8>)]) -> Vec<u8> {
        let mut t = vec![0u8; VFCT_LEN];
        t[0..4].copy_from_slice(b"VFCT");
        t[0x34..0x38].copy_from_slice(&(VFCT_LEN as u32).to_le_bytes());
        for (bus, ssid, img) in images {
            let mut h = [0u8; IMAGE_HEADER_LEN];
            h[0..4].copy_from_slice(&bus.to_le_bytes());
            h[0x0c..0x0e].copy_from_slice(&0x1002u16.to_le_bytes());
            h[0x0e..0x10].copy_from_slice(&0x6798u16.to_le_bytes());
            h[0x10..0x12].copy_from_slice(&0x106bu16.to_le_bytes());
            h[0x12..0x14].copy_from_slice(&ssid.to_le_bytes());
            h[0x18..0x1c].copy_from_slice(&(img.len() as u32).to_le_bytes());
            t.extend_from_slice(&h);
            t.extend_from_slice(img);
        }
        let len = t.len() as u32;
        t[4..8].copy_from_slice(&len.to_le_bytes());
        let sum = t.iter().fold(0u8, |s, &b| s.wrapping_add(b));
        t[9] = 0u8.wrapping_sub(sum);
        t
    }

    #[test]
    fn selects_the_image_for_the_exact_bus() {
        let t = vfct(&[(2, 0x0128, atom_image("113-C3861XA-028")), (6, 0x0127, atom_image("113-C3861XB-028"))]);
        let images = vfct_images(&t).unwrap();
        assert_eq!(images.len(), 2);
        let addr = |bus| PciAddress { domain: 0, bus, device: 0, function: 0 };
        let a = vfct_select(&images, addr(2), 0x1002, 0x6798).unwrap();
        let b = vfct_select(&images, addr(6), 0x1002, 0x6798).unwrap();
        assert_eq!(a.subsystem_id, 0x0128);
        assert_eq!(atom_info(a.image).unwrap().part_number, "113-C3861XA-028");
        assert_eq!(atom_info(b.image).unwrap().part_number, "113-C3861XB-028");
        // No fallback to another bus's image.
        assert!(vfct_select(&images, addr(3), 0x1002, 0x6798).is_err());
        assert!(vfct_select(&images, addr(2), 0x1002, 0x679a).is_err());
    }

    #[test]
    fn rejects_corrupt_tables_and_images() {
        let mut t = vfct(&[(2, 0x0128, atom_image("X"))]);
        t[100] ^= 1;
        assert!(vfct_images(&t).is_err(), "checksum");
        let t = vfct(&[(2, 0x0128, atom_image("X"))]);
        assert!(vfct_images(&t[..t.len() - 1]).is_err(), "truncated");
        let mut rom = atom_image("X");
        rom[0x104] = b'B';
        assert!(atom_info(&rom).is_err());
        assert!(atom_info(&[0x55, 0xaa]).is_err());
    }
}
