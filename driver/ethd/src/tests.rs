//! The driver against the simulated chip (sim.rs): bring-up, the register values tg3 leaves on the Mac Pro's
//! port, frames through the rings, filtering, flow control, link changes and teardown.

use crate::hw::Hw;
use crate::nic::{Nic, Options, fmt_mac};
use crate::regs::*;
use crate::sim::{SIM_MAC, SimChip, SimDma};
use std::sync::Arc;
use std::time::Duration;

fn open(opts: Options) -> (Arc<SimChip>, SimDma, Nic) {
    let opts = Options { link_wait: Duration::from_secs(1), ..opts };
    let dma = SimDma::new();
    let chip = SimChip::new(&dma);
    let hw: Arc<dyn Hw> = chip.clone();
    let nic = Nic::open(hw, &dma, Some(SIM_MAC), opts).expect("bring-up");
    (chip, dma, nic)
}

fn frame(dst: [u8; 6], src: [u8; 6], n: u32, len: usize) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&dst);
    f.extend_from_slice(&src);
    f.extend_from_slice(&[0x88, 0xb5]);
    f.extend_from_slice(&n.to_be_bytes());
    while f.len() < len {
        f.push(f.len() as u8 ^ n as u8);
    }
    f
}

fn drain(nic: &Nic) -> Vec<Vec<u8>> {
    let mut got = Vec::new();
    nic.service(&mut |f: &[u8]| got.push(f.to_vec())).unwrap();
    got
}

#[test]
fn brings_up_with_the_registers_tg3_leaves() {
    let (chip, _dma, nic) = open(Options::default());
    assert_eq!(nic.mac, SIM_MAC);
    assert_eq!(nic.id.chip_rev_id, 0x5776_6000);
    assert_eq!(chip.core_resets(), 1);
    assert_eq!(chip.phy_resets(), 1);
    assert!(nic.link().up);
    assert_eq!(nic.link().speed, 1000);
    // Captured from Linux 7.3's tg3 on incite's enp12s0 (ETHTOOL_GREGS) where the configuration is the same.
    // tg3 there runs MSI-X with RSS, so the RSS bits of RX_MODE and the per-vector registers differ. The backoff
    // seed reads back other than written (0x329 for a written 0x2a8) and isn't compared.
    let mut bad = Vec::new();
    for (reg, want) in [
        (MAC_MODE, 0x00e0_4808),
        (MAC_ADDR_0_HIGH, 0x0000_003e),
        (MAC_ADDR_0_LOW, 0xe1ce_a417),
        (MAC_RX_MTU_SIZE, 0x0000_05f2),
        (MAC_TX_MODE, 0x0000_0112),
        (MAC_TX_LENGTHS, 0x0000_2620),
        (MAC_MI_MODE, 0x0000_8000),
        (MAC_MI_STAT, 0x0000_0001),
        (MAC_EVENT, 0x0000_1000),
        (MAC_RCV_RULE_0, 0x4200_0000),
        (MAC_RCV_VALUE_0, 0x7fff_ffff),
        (MAC_RCV_RULE_1, 0x0600_0004),
        (MAC_RCV_RULE_CFG, 0x0000_0008),
        (MAC_LOW_WMARK_MAX_RX_FRAME, 1),
        (SNDDATAI_MODE, 0x0000_000a),
        (SNDDATAI_STATSCTRL, 0x0000_0003),
        (SNDDATAC_MODE, 0x0000_0002),
        (SNDBDS_MODE, 0x0000_0006),
        (SNDBDC_MODE, 0x0000_0006), // what tg3 writes; the chip reads back 0x2
        (RCVLPC_MODE, 0x0000_0002),
        (RCVLPC_CONFIG, 0x0000_0181),
        (RCVLPC_STATSCTRL, 0x0000_0001),
        (RCVDBDI_MODE, 0x0000_0012),
        (RCVDBDI_STD_BD + BDINFO_MAXLEN_FLAGS, 0x0200_1800),
        (RCVDCC_MODE, 0x0000_0006),
        (RCVBDI_MODE, 0x0000_0006),
        (RCVBDI_STD_THRESH, 0x0000_0019),
        (RCVBDI_JUMBO_THRESH, 0x0000_000c),
        (RCVCC_MODE, 0x0000_0006),
        (HOSTCC_TXCOL_TICKS, 0x48),
        (HOSTCC_TXMAX_FRAMES, 0x35),
        (HOSTCC_TXCOAL_MAXF_INT, 0x05),
        (MEMARB_MODE, 0x0000_0002),
        (BUFMGR_MODE, 0x0000_0006),
        (BUFMGR_MB_MACRX_LOW_WATER, 0x2a),
        (BUFMGR_MB_HIGH_WATER, 0xa0),
        (RDMAC_MODE, 0x1883_03fe),
        (WDMAC_MODE, 0x2000_03fe),
        (GRC_MODE, 0x0413_0034),
    ] {
        if chip.reg(reg) != want {
            bad.push(format!("{reg:#06x}: {:#x}, tg3 {want:#x}", chip.reg(reg)));
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
    assert_eq!(chip.reg(MAC_RX_MODE) & !0x00ff_0000, 0x0100_0006);
    assert_eq!(chip.reg(DMA_RW_CTRL), 0x761b_000f);
    assert_eq!(chip.reg(GRC_LOCAL_CTRL) & 0x0101_f808, 0x0100_9008);
}

#[test]
fn points_the_chip_at_its_rings() {
    let (chip, dma, nic) = open(Options::default());
    let map = dma.map.lock().unwrap().clone();
    let iova_of = |len: usize| map.iter().find(|m| m.2 == len).unwrap().0;
    let sb = (chip.reg(HOSTCC_STATUS_BLK_HOST_ADDR) as u64) << 32 | chip.reg(HOSTCC_STATUS_BLK_HOST_ADDR + 4) as u64;
    assert_eq!(sb, iova_of(4096));
    let send = (chip.sram(SRAM_SEND_RCB) as u64) << 32 | chip.sram(SRAM_SEND_RCB + 4) as u64;
    assert_eq!(send, iova_of(TX_RING_SIZE * 16));
    assert_eq!(chip.sram(SRAM_SEND_RCB + BDINFO_SIZE + BDINFO_MAXLEN_FLAGS), BDINFO_FLAGS_DISABLED);
    let ret = (chip.sram(SRAM_RCV_RET_RCB) as u64) << 32 | chip.sram(SRAM_RCV_RET_RCB + 4) as u64;
    assert_eq!(ret, iova_of(RX_RET_RING_SIZE * 32));
    for i in 1..4 {
        assert_eq!(chip.sram(SRAM_RCV_RET_RCB + i * BDINFO_SIZE + BDINFO_MAXLEN_FLAGS), BDINFO_FLAGS_DISABLED);
    }
    let std = (chip.reg(RCVDBDI_STD_BD) as u64) << 32 | chip.reg(RCVDBDI_STD_BD + 4) as u64;
    assert_eq!(std, iova_of(RX_STD_RING_SIZE * 32));
    assert_eq!(chip.reg(MBOX_RCV_STD_PROD_IDX), RX_PENDING as u32);
    // Device addresses above 4 GiB exercise the high halves.
    assert!(sb >> 32 != 0);
    drop(nic);
}

#[test]
fn loops_frames_back_intact_through_wrapping_rings() {
    let (chip, _dma, nic) = open(Options::default());
    let mut n = 0u32;
    // Several times around every ring, with varying batch sizes and frame lengths.
    for batch in [1usize, 7, 64, 199, 3, 150].iter().cycle().take(60) {
        let sent: Vec<_> = (0..*batch)
            .map(|_| {
                n += 1;
                frame(SIM_MAC, [2, 0, 0, 0, 0, 1], n, 60 + (n as usize * 37) % 1455)
            })
            .collect();
        for f in &sent {
            assert!(nic.send(f).unwrap());
        }
        let got = drain(&nic);
        assert_eq!(got.len(), sent.len());
        for (g, s) in got.iter().zip(&sent) {
            assert_eq!(g, s);
        }
        assert_eq!(chip.wire().len(), sent.len());
    }
    let s = nic.stats();
    assert_eq!((s.tx_frames, s.rx_frames, s.rx_errors), (n as u64, n as u64, 0));
    assert!(n as usize > 3 * RX_RET_RING_SIZE);
    assert_eq!(chip.dropped(), 0);
}

#[test]
fn pads_short_frames_and_refuses_oversize() {
    let (chip, _dma, nic) = open(Options::default());
    let short = frame([2, 0, 0, 0, 0, 9], SIM_MAC, 1, 20);
    assert!(nic.send(&short).unwrap());
    let wire = chip.wire();
    assert_eq!(wire[0].len(), 60);
    assert_eq!(&wire[0][..20], &short[..]);
    assert!(wire[0][20..].iter().all(|&b| b == 0));
    assert!(nic.send(&vec![0u8; 1515]).is_err());
    assert!(nic.send(&[0u8; 13]).is_err());
}

#[test]
fn reports_a_full_send_ring_until_the_chip_catches_up() {
    let (chip, _dma, nic) = open(Options::default());
    // The simulated chip only sends once the data path is up; take TX down to keep frames queued.
    chip.write32(MAC_TX_MODE, 0);
    let f = frame([2, 0, 0, 0, 0, 9], SIM_MAC, 0, 100);
    for _ in 0..TX_RING_SIZE - 1 {
        assert!(nic.send(&f).unwrap());
    }
    assert!(!nic.send(&f).unwrap());
    assert_eq!(nic.stats().tx_ring_full, 1);
    assert_eq!(nic.tx_pending(), TX_RING_SIZE - 1);
}

#[test]
fn filters_by_address() {
    let (chip, _dma, nic) = open(Options::default());
    chip.inject(&frame([2, 9, 9, 9, 9, 9], [2, 0, 0, 0, 0, 1], 1, 64)); // someone else's
    chip.inject(&frame([0x33, 0x33, 0, 0, 0, 1], [2, 0, 0, 0, 0, 1], 2, 64)); // multicast
    chip.inject(&frame([0xff; 6], [2, 0, 0, 0, 0, 1], 3, 64));
    chip.inject(&frame(SIM_MAC, [2, 0, 0, 0, 0, 1], 4, 64));
    let got = drain(&nic);
    assert_eq!(got.iter().map(|f| f[17]).collect::<Vec<_>>(), [3, 4]);

    let (chip, _dma, nic) = open(Options { all_multi: true, ..Default::default() });
    chip.inject(&frame([0x33, 0x33, 0, 0, 0, 1], [2, 0, 0, 0, 0, 1], 2, 64));
    assert_eq!(drain(&nic).len(), 1);
}

#[test]
fn interrupts_follow_the_status_tag() {
    let (chip, _dma, nic) = open(Options::default());
    let irq = chip.irq_event();
    // Bring-up asked for a status update with interrupts unmasked.
    irq.wait(Some(Duration::from_millis(100))).unwrap();
    drain(&nic);
    assert!(irq.wait(Some(Duration::from_millis(10))).is_err(), "nothing new, no interrupt");
    chip.inject(&frame(SIM_MAC, [2, 0, 0, 0, 0, 1], 1, 64));
    irq.wait(Some(Duration::from_millis(100))).unwrap();
    assert_eq!(drain(&nic).len(), 1);
}

#[test]
fn follows_link_changes_and_flow_control() {
    let (chip, _dma, nic) = open(Options::default());
    assert!(nic.link().tx_pause && nic.link().rx_pause);
    assert_ne!(chip.reg(MAC_RX_MODE) & RX_MODE_FLOW_CTRL_ENABLE, 0);
    // The partner drops to 100 Mb/s full duplex without pause.
    chip.set_phy(MII_TG3_AUX_STAT, 0x0500);
    chip.set_phy(MII_LPA, 0x41e1 & !(ADVERTISE_PAUSE_CAP as u16));
    let link = nic.check_link().unwrap();
    assert_eq!((link.speed, link.full_duplex, link.tx_pause, link.rx_pause), (100, true, false, false));
    assert_eq!(chip.reg(MAC_MODE) & MAC_MODE_PORT_MODE_MASK, MAC_MODE_PORT_MODE_MII);
    assert_eq!(chip.reg(MAC_RX_MODE) & RX_MODE_FLOW_CTRL_ENABLE, 0);
    assert_eq!(chip.reg(MAC_TX_MODE) & TX_MODE_FLOW_CTRL_ENABLE, 0);
    chip.set_phy(MII_BMSR, 0x7969);
    assert!(!nic.check_link().unwrap().up);
    assert_eq!(chip.reg(MAC_EVENT), MAC_EVENT_LNKSTATE_CHANGED, "link changes still interrupt");
    assert_eq!(chip.reg(MAC_MODE) & MAC_MODE_PORT_MODE_MASK, MAC_MODE_PORT_MODE_GMII);
    assert_eq!(nic.stats().link_changes, 3);
}

#[test]
fn resets_the_phy_and_advertises_every_mode_with_pause() {
    let (chip, _dma, nic) = open(Options::default());
    let mii_writes: Vec<_> = chip
        .writes()
        .iter()
        .filter(|(o, v)| *o == MAC_MI_COM && v & MI_COM_CMD_WRITE != 0)
        .map(|(_, v)| ((v >> 16) & 0x1f, v & 0xffff))
        .collect();
    let at = |w: (u32, u32)| mii_writes.iter().position(|&x| x == w).unwrap_or_else(|| panic!("no MII write {w:x?}"));
    let reset = at((MII_BMCR, 0x8000));
    assert!(at((MII_ADVERTISE, crate::phy::ADVERTISE)) > reset);
    assert!(at((MII_CTRL1000, crate::phy::CTRL1000)) > reset);
    assert!(at((MII_BMCR, BMCR_ANENABLE | BMCR_ANRESTART)) > reset);
    assert!(nic.link().up);
}

#[test]
fn refuses_another_port_before_resetting_it() {
    let dma = SimDma::new();
    let chip = SimChip::new(&dma);
    let hw: Arc<dyn Hw> = chip.clone();
    let other = [0x00, 0x3e, 0xe1, 0xce, 0xa4, 0x16];
    let err = Nic::open(hw, &dma, Some(other), Options::default()).err().expect("must refuse");
    assert!(err.to_string().contains(&fmt_mac(&other)), "{err}");
    assert_eq!(chip.core_resets(), 0);
    assert!(dma.map.lock().unwrap().is_empty(), "no memory given to the chip");
    // Only the memory window and arbiter were touched to read the address.
    assert!(chip.writes().iter().all(|(o, _)| [MEM_WIN_BASE_ADDR, MEMARB_MODE, GRC_MODE, REG_BASE_ADDR].contains(o)));
}

#[test]
fn halts_before_its_memory_is_freed() {
    let (chip, _dma, nic) = open(Options::default());
    assert!(nic.send(&frame(SIM_MAC, [2, 0, 0, 0, 0, 1], 1, 64)).unwrap());
    drop(nic);
    assert_eq!(chip.core_resets(), 2);
    for reg in [MAC_TX_MODE, MAC_RX_MODE, RDMAC_MODE, WDMAC_MODE, HOSTCC_MODE, SNDBDI_MODE, RCVDBDI_MODE] {
        assert_eq!(chip.reg(reg) & MODE_ENABLE, 0, "{reg:#x} still enabled");
    }
    // The station address stays in the MAC for the next driver, as tg3 leaves it.
    assert_eq!(chip.reg(MAC_ADDR_0_LOW), 0xe1ce_a417);
}
