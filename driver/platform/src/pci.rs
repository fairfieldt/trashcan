#[allow(unused_imports)]
use crate::prelude::*;
use crate::{Error, IrqEvent, Mmio, Result};
use core::fmt;
use core::str::FromStr;

/// Segment (domain), bus, device and function of a PCI function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PciAddress {
    pub domain: u16,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl FromStr for PciAddress {
    type Err = Error;

    /// `0000:02:00.0`, or `02:00.0` for domain 0.
    fn from_str(s: &str) -> Result<Self> {
        let bad = || Error::Invalid(format!("PCI address {s:?} (expected DDDD:BB:DD.F)"));
        let (rest, function) = s.rsplit_once('.').ok_or_else(bad)?;
        let parts: Vec<&str> = rest.split(':').collect();
        let (domain, bus, device) = match parts[..] {
            [d, b, dev] => (d, b, dev),
            [b, dev] => ("0", b, dev),
            _ => return Err(bad()),
        };
        let addr = PciAddress {
            domain: u16::from_str_radix(domain, 16).map_err(|_| bad())?,
            bus: u8::from_str_radix(bus, 16).map_err(|_| bad())?,
            device: u8::from_str_radix(device, 16).map_err(|_| bad())?,
            function: u8::from_str_radix(function, 16).map_err(|_| bad())?,
        };
        if addr.device > 31 || addr.function > 7 {
            return Err(bad());
        }
        Ok(addr)
    }
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}:{:02x}:{:02x}.{:x}", self.domain, self.bus, self.device, self.function)
    }
}

/// A PCI function's address spaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    Bar(u8),
    Rom,
    Config,
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Region::Bar(n) => write!(f, "BAR{n}"),
            Region::Rom => write!(f, "ROM"),
            Region::Config => write!(f, "config"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionInfo {
    pub region: Region,
    /// 0 when the function doesn't implement the region.
    pub size: u64,
    pub readable: bool,
    pub writable: bool,
    /// [`PciDevice::map`] works; otherwise only [`PciDevice::read`]/[`PciDevice::write`].
    pub mappable: bool,
}

/// How the platform can reset the function, if at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetCapability {
    /// A reset of this function alone (FLR, PM or a bus it has to itself).
    Function,
    /// Only a reset of the whole bus or slot, which takes every function on it.
    Bus,
    None,
}

/// One PCI function owned exclusively by this process.
///
/// Region offsets are byte offsets within the region. The owner keeps the function in D0; a power transition or a
/// reset that the driver didn't ask for loses the device state the driver set up.
pub trait PciDevice: Send + Sync {
    fn address(&self) -> PciAddress;

    fn region_info(&self, region: Region) -> Result<RegionInfo>;

    /// Reads through the platform: config space, the expansion ROM, or a BAR that can't be mapped.
    fn read(&self, region: Region, offset: u64, buf: &mut [u8]) -> Result<()>;

    fn write(&self, region: Region, offset: u64, data: &[u8]) -> Result<()>;

    /// Maps a memory BAR into this address space.
    fn map(&self, region: Region) -> Result<Mmio>;

    /// Routes the function's MSI (vector 0) to a notification.
    fn enable_msi(&self) -> Result<IrqEvent>;

    /// Routes MSI-X vectors 0..count, each to a notification of its own.
    fn enable_msix(&self, count: u32) -> Result<Vec<IrqEvent>> {
        let _ = count;
        Err(Error::Unsupported(format!("{}: MSI-X", self.address())))
    }

    /// Stops delivering the function's interrupts.
    fn disable_irqs(&self) -> Result<()>;

    fn reset_capability(&self) -> ResetCapability;

    fn config_u8(&self, offset: u16) -> Result<u8> {
        let mut b = [0u8; 1];
        self.read(Region::Config, offset as u64, &mut b)?;
        Ok(b[0])
    }

    fn config_u16(&self, offset: u16) -> Result<u16> {
        let mut b = [0u8; 2];
        self.read(Region::Config, offset as u64, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn config_u32(&self, offset: u16) -> Result<u32> {
        let mut b = [0u8; 4];
        self.read(Region::Config, offset as u64, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn set_config_u16(&self, offset: u16, value: u16) -> Result<()> {
        self.write(Region::Config, offset as u64, &value.to_le_bytes())
    }
}

pub const CONFIG_VENDOR_ID: u16 = 0x00;
pub const CONFIG_DEVICE_ID: u16 = 0x02;
pub const CONFIG_COMMAND: u16 = 0x04;
pub const CONFIG_STATUS: u16 = 0x06;
pub const CONFIG_REVISION: u16 = 0x08;
pub const CONFIG_CLASS: u16 = 0x09;
pub const CONFIG_BAR0: u16 = 0x10;
pub const CONFIG_SUBSYSTEM_VENDOR_ID: u16 = 0x2c;
pub const CONFIG_SUBSYSTEM_ID: u16 = 0x2e;
pub const CONFIG_CAPABILITIES: u16 = 0x34;

pub const COMMAND_MEMORY: u16 = 1 << 1;
pub const COMMAND_BUS_MASTER: u16 = 1 << 2;
const STATUS_CAPABILITIES: u16 = 1 << 4;

pub const CAP_ID_PM: u8 = 0x01;
pub const CAP_ID_MSI: u8 = 0x05;
pub const CAP_ID_EXPRESS: u8 = 0x10;
pub const CAP_ID_MSIX: u8 = 0x11;

/// A function's capability lists, as (ID, config space offset) pairs.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    pub standard: Vec<(u8, u16)>,
    /// PCI Express extended capabilities (from 0x100).
    pub extended: Vec<(u16, u16)>,
}

impl Capabilities {
    pub fn find(&self, id: u8) -> Option<u16> {
        self.standard.iter().find(|c| c.0 == id).map(|c| c.1)
    }
}

pub fn capabilities(dev: &dyn PciDevice) -> Result<Capabilities> {
    let mut standard = Vec::new();
    if dev.config_u16(CONFIG_STATUS)? & STATUS_CAPABILITIES != 0 {
        let mut ptr = dev.config_u8(CONFIG_CAPABILITIES)? & 0xfc;
        // The list lives in the first 256 bytes, at most 48 entries; a longer chain is a loop.
        for _ in 0..48 {
            if ptr < 0x40 {
                break;
            }
            let id = dev.config_u8(ptr as u16)?;
            standard.push((id, ptr as u16));
            ptr = dev.config_u8(ptr as u16 + 1)? & 0xfc;
        }
    }
    let mut extended = Vec::new();
    let is_express = standard.iter().any(|&(id, _)| id == CAP_ID_EXPRESS);
    if is_express && dev.region_info(Region::Config)?.size >= 4096 {
        let mut off = 0x100u16;
        for _ in 0..(4096 - 256) / 4 {
            let header = dev.config_u32(off)?;
            if header == 0 || header == u32::MAX {
                break;
            }
            extended.push(((header & 0xffff) as u16, off));
            off = ((header >> 20) & 0xffc) as u16;
            if off < 0x100 {
                break;
            }
        }
    }
    Ok(Capabilities { standard, extended })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_prints_addresses() {
        let a: PciAddress = "0000:02:00.0".parse().unwrap();
        assert_eq!(a, PciAddress { domain: 0, bus: 2, device: 0, function: 0 });
        assert_eq!(a.to_string(), "0000:02:00.0");
        let b: PciAddress = "a3:1f.7".parse().unwrap();
        assert_eq!(b.to_string(), "0000:a3:1f.7");
        for bad in ["", "02:00", "0000:02:00.8", "0000:02:20.0", "x:02:00.0", "1:2:3:4.0"] {
            assert!(bad.parse::<PciAddress>().is_err(), "{bad}");
        }
    }
}
