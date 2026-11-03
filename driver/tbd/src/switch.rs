//! The host router (the controller's switch, route 0): its config space through the control channel, its ports,
//! and the hop entries that route DMA paths between the NHI and a lane adapter.
//!
//! Register layouts are Linux's tb_regs.h; the sequences are from switch.c, cap.c, lc.c and path.c (7.3), cut down
//! to a Thunderbolt 2 host router talking to other hosts.

use crate::ctl::{CfgError, Ctl, SPACE_HOPS, SPACE_PORT, SPACE_SWITCH};
use crate::uuid::Uuid;
use anyhow::{Context, Result, bail};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const TYPE_INACTIVE: u32 = 0;
pub const TYPE_PORT: u32 = 1;
pub const TYPE_NHI: u32 = 2;

const CAP_PHY: u32 = 0x01;
const SWITCH_CAP_TMU: u32 = 0x03;
const SWITCH_CAP_VSE: u32 = 0x05;
const VSE_CAP_PLUG_EVENTS: u32 = 0x01;
const VSE_CAP_LINK_CONTROLLER: u32 = 0x06;

pub const LANE_ADP_CS_0: u32 = 0;
pub const LANE_ADP_CS_1: u32 = 1;
const LANE_CS_1_TARGET_WIDTH_SHIFT: u32 = 4;
const LANE_CS_1_TARGET_WIDTH_MASK: u32 = 3 << 4;
pub const WIDTH_SINGLE: u32 = 1;
pub const WIDTH_DUAL: u32 = 3;
const LANE_CS_1_LD: u32 = 1 << 14;
const LANE_CS_1_LB: u32 = 1 << 15;

const LC_DESC: u32 = 0x02;
const LC_FUSE: u32 = 0x03;
const LC_SX_CTRL: u32 = 0x96;
const LC_SX_CTRL_L1D: u32 = 1 << 17;
const LC_SX_CTRL_L2D: u32 = 1 << 21;
const PLUG_EVENTS_USB_DISABLE: u32 = 1 << 2;

/// Port states in the PHY capability (`enum tb_port_state`).
pub const STATE_DISABLED: u32 = 0;
pub const STATE_UNPLUGGED: u32 = 7;

/// Credits of the control path on a pre-USB4 lane adapter.
const CTL_CREDITS: u32 = 2;
/// DMA path credits (`TB_DMA_CREDITS`).
pub const DMA_CREDITS: u32 = 14;
const DMA_PRIORITY: u32 = 5;
const DMA_WEIGHT: u32 = 1;

#[derive(Clone, Debug)]
pub struct Port {
    pub number: u32,
    pub kind: u32,
    pub max_in_hop: u32,
    pub max_out_hop: u32,
    pub total_credits: u32,
    pub nfc_credits: u32,
    pub cap_phy: u32,
    pub disabled: bool,
}

impl Port {
    pub fn is_lane(&self) -> bool {
        self.kind == TYPE_PORT && !self.disabled
    }

    /// The second lane of a connector: Falcon Ridge pairs lane adapters 1/2 and 3/4.
    pub fn dual(&self) -> u32 {
        if self.number % 2 == 1 { self.number + 1 } else { self.number - 1 }
    }

    /// Lane 0 of its connector (the one Linux scans and routes over).
    pub fn primary(&self) -> bool {
        self.number % 2 == 1
    }
}

/// One router reached at `route`.
pub struct Router {
    pub ctl: Arc<Ctl>,
    pub route: u64,
    pub vendor: u16,
    pub device: u16,
    pub upstream_port: u32,
    pub max_port: u32,
    pub ports: Vec<Port>,
    pub cap_plug_events: u32,
    pub cap_lc: u32,
}

fn cfg<T>(r: Result<T, CfgError>, what: impl FnOnce() -> String) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("{}: {e}", what()))
}

impl Router {
    pub fn read(&self, port: u32, space: u32, offset: u32, len: u32) -> Result<Vec<u32>> {
        cfg(self.ctl.read(self.route, port, space, offset, len).map(|v| v.0), || {
            format!("read {:x}:{port} space {space} at {offset:#x}", self.route)
        })
    }

    pub fn write(&self, port: u32, space: u32, offset: u32, data: &[u32]) -> Result<()> {
        cfg(self.ctl.write(self.route, port, space, offset, data), || {
            format!("write {:x}:{port} space {space} at {offset:#x}", self.route)
        })
    }

    pub fn read1(&self, port: u32, space: u32, offset: u32) -> Result<u32> {
        Ok(self.read(port, space, offset, 1)?[0])
    }

    /// Reads the switch header, caps and ports (`tb_switch_alloc` and `tb_init_port`).
    pub fn open(ctl: Arc<Ctl>, route: u64) -> Result<Router> {
        let (hdr, upstream) =
            cfg(ctl.read(route, 0, SPACE_SWITCH, 0, 5), || format!("router {route:#x} switch header"))?;
        let mut r = Router {
            ctl,
            route,
            vendor: hdr[0] as u16,
            device: (hdr[0] >> 16) as u16,
            upstream_port: upstream,
            max_port: (hdr[1] >> 14) & 0x3f,
            ports: Vec::new(),
            cap_plug_events: 0,
            cap_lc: 0,
        };
        r.cap_plug_events = r.find_vse_cap(hdr[1] & 0xff, VSE_CAP_PLUG_EVENTS)?.unwrap_or(0);
        r.cap_lc = r.find_vse_cap(hdr[1] & 0xff, VSE_CAP_LINK_CONTROLLER)?.unwrap_or(0);
        for n in 1..=r.max_port {
            let port = match r.ctl.read(route, n, SPACE_PORT, 0, 8) {
                Ok((p, _)) => {
                    let kind = p[2] & 0xff_ffff;
                    let cap_phy = if kind == TYPE_PORT { r.find_port_cap(n, p[1] & 0xff, CAP_PHY)?.unwrap_or(0) } else { 0 };
                    Port {
                        number: n,
                        kind,
                        max_in_hop: p[5] & 0x7ff,
                        max_out_hop: (p[5] >> 11) & 0x7ff,
                        total_credits: (p[4] >> 20) & 0x3ff,
                        nfc_credits: p[4],
                        cap_phy,
                        disabled: false,
                    }
                }
                Err(CfgError::Router(crate::ctl::ERROR_INVALID_CONFIG_SPACE, _)) => {
                    Port { number: n, kind: TYPE_INACTIVE, max_in_hop: 0, max_out_hop: 0, total_credits: 0, nfc_credits: 0, cap_phy: 0, disabled: true }
                }
                Err(e) => bail!("router {route:#x} port {n}: {e}"),
            };
            r.ports.push(port);
        }
        Ok(r)
    }

    pub fn port(&self, n: u32) -> &Port {
        &self.ports[n as usize - 1]
    }

    pub fn nhi_port(&self) -> Option<&Port> {
        self.ports.iter().find(|p| p.kind == TYPE_NHI)
    }

    fn find_port_cap(&self, port: u32, first: u32, cap: u32) -> Result<Option<u32>> {
        let mut off = first;
        for _ in 0..64 {
            if off == 0 || off > 0xff {
                return Ok(None);
            }
            let h = self.read1(port, SPACE_PORT, off)?;
            if (h >> 8) & 0xff == cap {
                return Ok(Some(off));
            }
            off = h & 0xff;
        }
        Ok(None)
    }

    /// `tb_switch_find_vse_cap`: TMU caps have a basic header, VSE caps a short or long one.
    fn find_vse_cap(&self, first: u32, vsec: u32) -> Result<Option<u32>> {
        let mut off = first;
        for _ in 0..64 {
            if off == 0 || off >= 0xffff {
                return Ok(None);
            }
            let h = self.read(0, SPACE_SWITCH, off, 2)?;
            let cap = (h[0] >> 8) & 0xff;
            if cap == SWITCH_CAP_VSE && (h[0] >> 16) & 0xff == vsec {
                return Ok(Some(off));
            }
            off = match cap {
                SWITCH_CAP_TMU => h[0] & 0xff,
                SWITCH_CAP_VSE if h[0] >> 24 == 0 => h[1] & 0xffff,
                SWITCH_CAP_VSE => h[0] & 0xff,
                _ => return Ok(None),
            };
        }
        Ok(None)
    }

    /// Enumerates the router (`tb_switch_configure` for a pre-USB4 router): route, depth, upstream port,
    /// enabled, a 255 ms notification timeout, and plug events on.
    pub fn configure(&self) -> Result<()> {
        let mut h = self.read(0, SPACE_SWITCH, 1, 4)?;
        // ROUTER_CS_1: upstream port (bits 8-13) and depth (20-22).
        let depth = 0u32;
        h[0] = (h[0] & !(0x3f << 8) & !(7 << 20)) | (self.upstream_port & 0x3f) << 8 | depth << 20;
        h[1] = self.route as u32;
        h[2] = ((self.route >> 32) as u32 & 0x7fff_ffff) | 1 << 31;
        h[3] = (h[3] & !0xff) | 0xff;
        self.write(0, SPACE_SWITCH, 1, &h)?;
        if self.cap_plug_events == 0 {
            bail!("router {:#x} has no plug events capability", self.route);
        }
        let v = self.read1(0, SPACE_SWITCH, self.cap_plug_events + 1)?;
        self.write(0, SPACE_SWITCH, self.cap_plug_events + 1, &[(v & 0xffff_ff83) | PLUG_EVENTS_USB_DISABLE])
    }

    /// The UUID fused into the link controller (`tb_lc_read_uuid`), which Linux uses as the host's UUID.
    pub fn uuid(&self) -> Result<Uuid> {
        if self.cap_lc == 0 {
            bail!("router {:#x} has no link controller capability", self.route);
        }
        Ok(Uuid::from_words(&self.read(0, SPACE_SWITCH, self.cap_lc + LC_FUSE, 4)?))
    }

    pub fn port_state(&self, port: &Port) -> Result<u32> {
        Ok((self.read(port.number, SPACE_PORT, port.cap_phy, 2)?[1] >> 26) & 0xf)
    }

    /// `tb_wait_for_port`: 1 if the link is up, 0 if disabled or unplugged or it never came up.
    pub fn wait_for_port(&self, port: &Port) -> Result<bool> {
        for _ in 0..10 {
            match self.port_state(port)? {
                STATE_DISABLED | STATE_UNPLUGGED => return Ok(false),
                2..=6 => return Ok(true),
                _ => std::thread::sleep(Duration::from_millis(100)),
            }
        }
        Ok(false)
    }

    pub fn lane_cs(&self, port: &Port) -> Result<[u32; 2]> {
        let v = self.read(port.number, SPACE_PORT, port.cap_phy + LANE_ADP_CS_0, 2)?;
        Ok([v[0], v[1]])
    }

    fn set_lane_cs1(&self, port: &Port, f: impl FnOnce(u32) -> u32) -> Result<()> {
        let v = self.read1(port.number, SPACE_PORT, port.cap_phy + LANE_ADP_CS_1)?;
        self.write(port.number, SPACE_PORT, port.cap_phy + LANE_ADP_CS_1, &[f(v)])
    }

    /// `tb_port_enable`/`tb_port_disable`: the lane disable bit.
    pub fn set_lane_enabled(&self, port: &Port, on: bool) -> Result<()> {
        self.set_lane_cs1(port, |v| if on { v & !LANE_CS_1_LD } else { v | LANE_CS_1_LD })
    }

    pub fn link_width(&self, port: &Port) -> Result<u32> {
        Ok((self.lane_cs(port)?[1] >> 20) & 0x3f)
    }

    pub fn set_target_width(&self, port: &Port, width: u32) -> Result<()> {
        self.set_lane_cs1(port, |v| (v & !LANE_CS_1_TARGET_WIDTH_MASK) | width << LANE_CS_1_TARGET_WIDTH_SHIFT)
    }

    pub fn set_lane_bonding(&self, port: &Port, on: bool) -> Result<()> {
        self.set_lane_cs1(port, |v| if on { v | LANE_CS_1_LB } else { v & !LANE_CS_1_LB })
    }

    /// `tb_port_wait_for_link_width`.
    pub fn wait_for_width(&self, port: &Port, mask: u32, timeout: Duration) -> Result<bool> {
        let t = Instant::now() + timeout;
        loop {
            match self.link_width(port) {
                Ok(w) if w & mask != 0 => return Ok(true),
                Ok(_) => {}
                Err(e) if e.to_string().contains("router error 15") => {}
                Err(e) => return Err(e),
            }
            if Instant::now() > t {
                return Ok(false);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Re-reads a lane adapter's credits (`tb_port_update_credits`): bonding changes them.
    pub fn update_credits(&mut self, n: u32) -> Result<()> {
        let v = self.read1(n, SPACE_PORT, 4)?;
        let p = &mut self.ports[n as usize - 1];
        p.nfc_credits = v;
        p.total_credits = (v >> 20) & 0x3ff;
        Ok(())
    }

    fn port_lc(&self, port: &Port) -> Result<u32> {
        if self.cap_lc == 0 {
            bail!("router {:#x} has no link controller", self.route);
        }
        let desc = self.read1(0, SPACE_SWITCH, self.cap_lc + LC_DESC)?;
        let start = (desc >> 8) & 0xff;
        let size = (desc >> 16) & 0xfff;
        let phys = (port.number - 1) / 2;
        Ok(self.cap_lc + start + phys * size)
    }

    /// `tb_lc_configure_xdomain`: tells the link controller its lane is connected to another domain.
    pub fn set_xdomain(&self, port: &Port, on: bool) -> Result<()> {
        let lc = self.port_lc(port)?;
        let v = self.read1(0, SPACE_SWITCH, lc + LC_SX_CTRL)?;
        let bit = if port.number % 2 == 1 { LC_SX_CTRL_L1D } else { LC_SX_CTRL_L2D };
        self.write(0, SPACE_SWITCH, lc + LC_SX_CTRL, &[if on { v | bit } else { v & !bit }])
    }

    pub fn read_hop(&self, port: u32, hop: u32) -> Result<Hop> {
        let v = self.read(port, SPACE_HOPS, 2 * hop, 2)?;
        Ok(Hop::from_words(v[0], v[1]))
    }

    pub fn write_hop(&self, port: u32, hop: u32, h: &Hop) -> Result<()> {
        self.write(port, SPACE_HOPS, 2 * hop, &h.words())
    }

    /// `__tb_path_deactivate_hop` with clear_fc: disable, wait for it to drain, clear flow control.
    pub fn deactivate_hop(&self, port: u32, hop: u32) -> Result<()> {
        let mut h = self.read_hop(port, hop)?;
        if !h.enable {
            return Ok(());
        }
        h.enable = false;
        self.write_hop(port, hop, &h)?;
        let t = Instant::now() + Duration::from_millis(500);
        loop {
            let mut h = self.read_hop(port, hop)?;
            if !h.pending {
                h.ingress_fc = false;
                h.ingress_shared_buffer = false;
                h.egress_fc = false;
                h.egress_shared_buffer = false;
                return self.write_hop(port, hop, &h);
            }
            if Instant::now() > t {
                bail!("hop {hop} on port {port} did not drain");
            }
            std::thread::sleep(Duration::from_micros(20));
        }
    }

    /// Activates a one-hop path through this router (`tb_path_activate` for a path of length 1: the source and
    /// destination flow control masks both apply to the single hop).
    pub fn activate_hop(&self, in_port: u32, in_hop: u32, out_port: u32, out_hop: u32, p: &PathConfig) -> Result<()> {
        self.deactivate_hop(in_port, in_hop).context("deactivating the hop first")?;
        let mut h = self.read_hop(in_port, in_hop)?;
        h.next_hop = out_hop;
        h.out_port = out_port;
        h.initial_credits = p.credits;
        h.pmps = false;
        h.unknown1 = 0;
        h.enable = true;
        h.weight = DMA_WEIGHT;
        h.unknown2 = 0;
        h.priority = DMA_PRIORITY;
        h.drop_packages = false;
        h.counter = 0x7ff;
        h.counter_enable = false;
        h.ingress_fc = p.ingress_fc;
        h.egress_fc = p.egress_fc;
        h.ingress_shared_buffer = false;
        h.egress_shared_buffer = false;
        h.unknown3 = 0;
        self.write_hop(in_port, in_hop, &h)
    }

    /// Credits a DMA RX path may take on a lane adapter (`tb_dma_init_rx_path`, first hop): all but the
    /// control path's.
    pub fn rx_credits(&self, lane: &Port) -> u32 {
        (lane.total_credits.saturating_sub(CTL_CREDITS)).min(DMA_CREDITS)
    }

    /// Credits of the NHI's TX hop (`tb_dma_reserve_credits` on a non-lane port).
    pub fn tx_credits(&self, nhi: &Port) -> u32 {
        nhi.total_credits.min(DMA_CREDITS)
    }
}

/// How a DMA path's hop is programmed.
pub struct PathConfig {
    pub credits: u32,
    pub ingress_fc: bool,
    pub egress_fc: bool,
}

/// `struct tb_regs_hop`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Hop {
    pub next_hop: u32,
    pub out_port: u32,
    pub initial_credits: u32,
    pub pmps: bool,
    pub unknown1: u32,
    pub enable: bool,
    pub weight: u32,
    pub unknown2: u32,
    pub priority: u32,
    pub drop_packages: bool,
    pub counter: u32,
    pub counter_enable: bool,
    pub ingress_fc: bool,
    pub egress_fc: bool,
    pub ingress_shared_buffer: bool,
    pub egress_shared_buffer: bool,
    pub pending: bool,
    pub unknown3: u32,
}

impl Hop {
    pub fn from_words(a: u32, b: u32) -> Hop {
        let bit = |v: u32, n: u32| v >> n & 1 != 0;
        Hop {
            next_hop: a & 0x7ff,
            out_port: (a >> 11) & 0x3f,
            initial_credits: (a >> 17) & 0x7f,
            pmps: bit(a, 24),
            unknown1: (a >> 25) & 0x3f,
            enable: bit(a, 31),
            weight: b & 0xf,
            unknown2: (b >> 4) & 0xf,
            priority: (b >> 8) & 7,
            drop_packages: bit(b, 11),
            counter: (b >> 12) & 0x7ff,
            counter_enable: bit(b, 23),
            ingress_fc: bit(b, 24),
            egress_fc: bit(b, 25),
            ingress_shared_buffer: bit(b, 26),
            egress_shared_buffer: bit(b, 27),
            pending: bit(b, 28),
            unknown3: (b >> 29) & 7,
        }
    }

    pub fn words(&self) -> [u32; 2] {
        let a = (self.next_hop & 0x7ff)
            | (self.out_port & 0x3f) << 11
            | (self.initial_credits & 0x7f) << 17
            | (self.pmps as u32) << 24
            | (self.unknown1 & 0x3f) << 25
            | (self.enable as u32) << 31;
        let b = (self.weight & 0xf)
            | (self.unknown2 & 0xf) << 4
            | (self.priority & 7) << 8
            | (self.drop_packages as u32) << 11
            | (self.counter & 0x7ff) << 12
            | (self.counter_enable as u32) << 23
            | (self.ingress_fc as u32) << 24
            | (self.egress_fc as u32) << 25
            | (self.ingress_shared_buffer as u32) << 26
            | (self.egress_shared_buffer as u32) << 27
            | (self.pending as u32) << 28
            | (self.unknown3 & 7) << 29;
        [a, b]
    }
}

#[cfg(test)]
mod tests {
    use super::Hop;

    #[test]
    fn hop_round_trips() {
        // A DMA RX hop as Linux programs it on a host router lane (0:1): in HopID 8 => out port 5, HopID 2, 14 credits, priority 5, weight 1,
        // counter 2047 off, ingress flow control.
        let h = Hop {
            next_hop: 2,
            out_port: 5,
            initial_credits: 14,
            enable: true,
            weight: 1,
            priority: 5,
            counter: 0x7ff,
            ingress_fc: true,
            ..Default::default()
        };
        let [a, b] = h.words();
        assert_eq!(a, 2 | 5 << 11 | 14 << 17 | 1 << 31);
        assert_eq!(b, 1 | 5 << 8 | 0x7ff << 12 | 1 << 24);
        assert_eq!(Hop::from_words(a, b), h);
    }
}
