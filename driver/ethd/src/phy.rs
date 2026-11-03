//! The 57766's internal copper PHY over the MAC's MII interface: register access (tg3's __tg3_readphy and
//! __tg3_writephy), the link result (tg3_setup_copper_phy without a PHY reset), autonegotiation and pause resolution.

use crate::hw::{Hw, HwExt};
use crate::regs::*;
use anyhow::{Result, bail};
use std::fmt;

/// What tg3 advertises by default: every 10/100/1000 mode, symmetric pause.
pub const ADVERTISE: u32 =
    ADVERTISE_CSMA | ADVERTISE_10HALF | ADVERTISE_10FULL | ADVERTISE_100HALF | ADVERTISE_100FULL | ADVERTISE_PAUSE_CAP;
pub const CTRL1000: u32 = ADVERTISE_1000HALF | ADVERTISE_1000FULL;

pub fn read(hw: &dyn Hw, reg: u32) -> Result<u16> {
    let frame = ((PHY_ADDR << MI_COM_PHY_ADDR_SHIFT) & MI_COM_PHY_ADDR_MASK)
        | ((reg << MI_COM_REG_ADDR_SHIFT) & MI_COM_REG_ADDR_MASK)
        | MI_COM_CMD_READ
        | MI_COM_START;
    let v = transact(hw, frame)?;
    if v & MI_COM_READ_FAILED != 0 {
        bail!("PHY register {reg:#x}: read failed");
    }
    Ok((v & MI_COM_DATA_MASK) as u16)
}

pub fn write(hw: &dyn Hw, reg: u32, value: u16) -> Result<()> {
    let frame = ((PHY_ADDR << MI_COM_PHY_ADDR_SHIFT) & MI_COM_PHY_ADDR_MASK)
        | ((reg << MI_COM_REG_ADDR_SHIFT) & MI_COM_REG_ADDR_MASK)
        | (value as u32 & MI_COM_DATA_MASK)
        | MI_COM_CMD_WRITE
        | MI_COM_START;
    transact(hw, frame).map(|_| ())
}

fn transact(hw: &dyn Hw, frame: u32) -> Result<u32> {
    hw.write_flush(MAC_MI_COM, frame);
    // tg3: PHY_BUSY_LOOPS (5000) polls 10 µs apart.
    for _ in 0..5000 {
        hw.delay_us(10);
        if hw.read32(MAC_MI_COM) & MI_COM_BUSY == 0 {
            hw.delay_us(5);
            return Ok(hw.read32(MAC_MI_COM));
        }
    }
    bail!("PHY access {frame:#010x} timed out")
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Link {
    pub up: bool,
    /// Mb/s; 0 when down.
    pub speed: u32,
    pub full_duplex: bool,
    /// We may send PAUSE frames / we honour received ones.
    pub tx_pause: bool,
    pub rx_pause: bool,
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.up {
            return write!(f, "down");
        }
        write!(f, "up, {} Mb/s, {} duplex", self.speed, if self.full_duplex { "full" } else { "half" })?;
        match (self.tx_pause, self.rx_pause) {
            (true, true) => write!(f, ", flow control on"),
            (true, false) => write!(f, ", flow control TX only"),
            (false, true) => write!(f, ", flow control RX only"),
            (false, false) => write!(f, ", flow control off"),
        }
    }
}

/// Speed and duplex from the Broadcom auxiliary status register (tg3_aux_stat_to_speed_duplex).
pub fn aux_stat_speed(aux: u32) -> Option<(u32, bool)> {
    Some(match aux & MII_TG3_AUX_STAT_SPDMASK {
        0x0100 => (10, false),
        0x0200 => (10, true),
        0x0300 => (100, false),
        0x0500 => (100, true),
        0x0600 => (1000, false),
        0x0700 => (1000, true),
        _ => return None,
    })
}

/// Pause resolution of a full-duplex link from both sides' advertisements (mii_resolve_flowctrl_fdx): (tx, rx).
pub fn resolve_pause(local: u32, remote: u32) -> (bool, bool) {
    if local & remote & ADVERTISE_PAUSE_CAP != 0 {
        (true, true)
    } else if local & remote & ADVERTISE_PAUSE_ASYM != 0 {
        if local & ADVERTISE_PAUSE_CAP != 0 {
            (false, true)
        } else if remote & ADVERTISE_PAUSE_CAP != 0 {
            (true, false)
        } else {
            (false, false)
        }
    } else {
        (false, false)
    }
}

/// The link as the PHY reports it. A link counts as up only if it was negotiated with our advertisement
/// (tg3_phy_copper_an_config_ok); otherwise `Ok(None)`: the caller restarts autonegotiation.
pub fn read_link(hw: &dyn Hw) -> Result<Option<Link>> {
    // The link bit latches low: the first read clears a past drop.
    read(hw, MII_BMSR)?;
    let bmsr = read(hw, MII_BMSR)? as u32;
    if bmsr & BMSR_LSTATUS == 0 {
        return Ok(Some(Link::default()));
    }
    let mut aux = 0;
    for _ in 0..2000 {
        aux = read(hw, MII_TG3_AUX_STAT)? as u32;
        if aux != 0 {
            break;
        }
        hw.delay_us(10);
    }
    let Some((speed, full_duplex)) = aux_stat_speed(aux) else {
        return Ok(Some(Link::default()));
    };
    let bmcr = read(hw, MII_BMCR)? as u32;
    let adv = read(hw, MII_ADVERTISE)? as u32;
    let ctrl1000 = read(hw, MII_CTRL1000)? as u32;
    let modes = ADVERTISE & !ADVERTISE_CSMA;
    let adv_mask = modes | ADVERTISE_PAUSE_ASYM;
    if bmcr & BMCR_ANENABLE == 0 || adv & adv_mask != modes || ctrl1000 & CTRL1000 != CTRL1000 {
        return Ok(None);
    }
    let lpa = read(hw, MII_LPA)? as u32;
    let (tx_pause, rx_pause) = if full_duplex { resolve_pause(adv, lpa) } else { (false, false) };
    Ok(Some(Link { up: true, speed, full_duplex, tx_pause, rx_pause }))
}

/// Advertises every mode and restarts autonegotiation (tg3_phy_autoneg_cfg and tg3_phy_copper_begin).
pub fn restart_autoneg(hw: &dyn Hw) -> Result<()> {
    write(hw, MII_ADVERTISE, ADVERTISE as u16)?;
    write(hw, MII_CTRL1000, CTRL1000 as u16)?;
    write(hw, MII_BMCR, (BMCR_ANENABLE | BMCR_ANRESTART) as u16)
}

const MII_TG3_EXT_CTRL: u32 = 0x10;
const MII_TG3_EXT_CTRL_FIFO_ELASTIC: u16 = 0x0001;
const MII_TG3_AUX_CTRL: u32 = 0x18;
const AUXCTL_SHDWSEL_AUXCTL: u16 = 0x0000;
const AUXCTL_ACTL_EXTPKTLEN: u16 = 0x4000;
const AUXCTL_SHDWSEL_MISC: u16 = 0x0007;
const AUXCTL_MISC_WIRESPD_EN: u16 = 0x0010;
const AUXCTL_MISC_FORCE_AMDIX: u16 = 0x0200;
const AUXCTL_MISC_RDSEL_SHIFT: u16 = 12;
const AUXCTL_MISC_WREN: u16 = 0x8000;
const BMCR_RESET: u16 = 0x8000;

/// A shadow register behind MII_TG3_AUX_CTRL (tg3_phy_auxctl_read).
fn auxctl_read(hw: &dyn Hw, shadow: u16) -> Result<u16> {
    write(hw, MII_TG3_AUX_CTRL, (shadow << AUXCTL_MISC_RDSEL_SHIFT) | AUXCTL_SHDWSEL_MISC)?;
    read(hw, MII_TG3_AUX_CTRL)
}

/// tg3_phy_auxctl_write
pub fn auxctl_write(hw: &dyn Hw, shadow: u16, value: u16) -> Result<()> {
    let value = if shadow == AUXCTL_SHDWSEL_MISC { value | AUXCTL_MISC_WREN } else { value };
    write(hw, MII_TG3_AUX_CTRL, value | shadow)
}

/// tg3_phy_reset for the 57766's PHY: a BMCR reset (which also powers it up after tg3 powered it down), then
/// what tg3 sets afterwards on this chip: long packets and the elastic FIFO (it is jumbo capable), automatic MDI
/// crossover and Ethernet@WireSpeed. The link renegotiates.
pub fn reset(hw: &dyn Hw) -> Result<()> {
    read(hw, MII_BMSR)?;
    read(hw, MII_BMSR)?;
    write(hw, MII_BMCR, BMCR_RESET)?;
    let mut done = false;
    for _ in 0..5000 {
        if read(hw, MII_BMCR)? & BMCR_RESET == 0 {
            hw.delay_us(40);
            done = true;
            break;
        }
        hw.delay_us(10);
    }
    if !done {
        bail!("PHY reset didn't finish");
    }
    let v = auxctl_read(hw, AUXCTL_SHDWSEL_AUXCTL)?;
    auxctl_write(hw, AUXCTL_SHDWSEL_AUXCTL, v | AUXCTL_ACTL_EXTPKTLEN)?;
    let v = read(hw, MII_TG3_EXT_CTRL)?;
    write(hw, MII_TG3_EXT_CTRL, v | MII_TG3_EXT_CTRL_FIFO_ELASTIC)?;
    let v = auxctl_read(hw, AUXCTL_SHDWSEL_MISC)?;
    auxctl_write(hw, AUXCTL_SHDWSEL_MISC, v | AUXCTL_MISC_FORCE_AMDIX)?;
    let v = auxctl_read(hw, AUXCTL_SHDWSEL_MISC)?;
    auxctl_write(hw, AUXCTL_SHDWSEL_MISC, v | AUXCTL_MISC_WIRESPD_EN)
}

/// The PHY's OUI-based identifier (PHYSID1:PHYSID2).
pub fn id(hw: &dyn Hw) -> Result<u32> {
    Ok((read(hw, MII_PHYSID1)? as u32) << 16 | read(hw, MII_PHYSID2)? as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_aux_status() {
        assert_eq!(aux_stat_speed(0x0700), Some((1000, true)));
        assert_eq!(aux_stat_speed(0x0f00), Some((1000, true)));
        assert_eq!(aux_stat_speed(0x0300), Some((100, false)));
        assert_eq!(aux_stat_speed(0x0200), Some((10, true)));
        assert_eq!(aux_stat_speed(0x0400), None);
        assert_eq!(aux_stat_speed(0), None);
    }

    #[test]
    fn resolves_pause_like_linux() {
        let (cap, asym) = (ADVERTISE_PAUSE_CAP, ADVERTISE_PAUSE_ASYM);
        assert_eq!(resolve_pause(cap, cap), (true, true));
        assert_eq!(resolve_pause(cap, 0), (false, false));
        assert_eq!(resolve_pause(cap | asym, asym), (false, true));
        assert_eq!(resolve_pause(asym, cap | asym), (true, false));
        assert_eq!(resolve_pause(asym, asym), (false, false));
    }
}
