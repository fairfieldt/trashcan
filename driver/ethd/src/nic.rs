//! The BCM57766 MAC: identification, bring-up, the send/receive rings and teardown, after Linux 7.3's tg3
//! (tg3_get_invariants, tg3_chip_reset, tg3_reset_hw, tg3_abort_hw, tg3_rx, tg3_tx), for this one chip in the
//! configuration tg3 uses when it gets a single MSI: one send ring, the standard receive ring, one return ring,
//! tagged status, no jumbo frames, no offloads, no RSS.
//!
//! Frames are copied: every ring entry owns a 2 KiB buffer for its lifetime, so the device only ever sees memory
//! the driver allocated at start, and a frame handed to the host is a copy it can keep.

use crate::hw::{Hw, HwExt};
use crate::phy::{self, Link};
use crate::regs::*;
use anyhow::{Context, Result, bail, ensure};
use gpud_platform::{DmaAccess, DmaBuffer, DmaSpace};
use std::sync::atomic::{Ordering, fence};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const BUF_SIZE: usize = 2048;
/// The largest frame sent or received, without FCS (MTU 1500).
pub const MAX_FRAME: usize = 1514;
const MIN_FRAME: usize = 60;
const TX_BD: usize = 16;
const RX_BD: usize = 32;
const PCI_EXP_DEVCTL: u16 = 0x08;
const PCI_EXP_DEVSTA: u16 = 0x0a;
const PCI_EXP_DEVCTL_RELAX_EN: u16 = 0x0010;
const PCI_EXP_DEVCTL_NOSNOOP_EN: u16 = 0x0800;
const PCI_COMMAND: u16 = 0x04;
const PCI_COMMAND_MEMORY: u16 = 0x2;
const PCI_COMMAND_MASTER: u16 = 0x4;

/// Fixed parts of the mode registers (tg3_get_invariants for a 57766).
const MISC_HOST_CTRL_BASE: u32 = MISC_HOST_CTRL_MASK_PCI_INT
    | MISC_HOST_CTRL_WORD_SWAP
    | MISC_HOST_CTRL_INDIR_ACCESS
    | MISC_HOST_CTRL_PCISTATE_RW
    | MISC_HOST_CTRL_TAGGED_STATUS;
const GRC_MODE_BASE: u32 = GRC_MODE_WSWAP_DATA | GRC_MODE_BSWAP_DATA | GRC_MODE_WSWAP_NONFRM_DATA;
const GRC_MODE_RUN: u32 = GRC_MODE_BASE | GRC_MODE_HOST_SENDBDS | GRC_MODE_NO_TX_PHDR_CSUM;
const COALESCE_MODE: u32 = HOSTCC_MODE_32BYTE | HOSTCC_MODE_CLRTICK_RXBD | HOSTCC_MODE_CLRTICK_TXBD;
const MAC_MODE_RUN: u32 =
    MAC_MODE_TXSTAT_ENABLE | MAC_MODE_RXSTAT_ENABLE | MAC_MODE_TDE_ENABLE | MAC_MODE_RDE_ENABLE | MAC_MODE_FHDE_ENABLE;
const TX_MODE_RUN: u32 = TX_MODE_ENABLE | TX_MODE_MBUF_LOCKUP_FIX;
const RX_MODE_RUN: u32 = RX_MODE_ENABLE | RX_MODE_IPV6_CSUM_ENABLE;
/// GPIO 0-2 output enables and levels, kept as found; GPIO 1 driven high is the EEPROM write protect tg3 sets on
/// LOMs (and the value tg3 leaves on the Mac Pro).
const GPIO_MASK: u32 = 0x0001_f800;
const GPIO_EEPROM_WP: u32 = 0x0000_1000 | 0x0000_8000;
const MI_MODE: u32 = MAC_MI_MODE_500KHZ_CONST;

/// What the chip is, read from config space before anything is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub vendor: u16,
    pub device: u16,
    /// tg3's pci_chip_rev_id (0x57766000 for the Mac Pro's BCM57762).
    pub chip_rev_id: u32,
}

pub fn identify(hw: &dyn Hw) -> Result<Identity> {
    let vendor = hw.config_read16(0x00)?;
    let device = hw.config_read16(0x02)?;
    ensure!(
        vendor == PCI_VENDOR_BROADCOM && (device == PCI_DEVICE_57762 || device == PCI_DEVICE_57766),
        "{vendor:04x}:{device:04x} is not a BCM57762/57766"
    );
    let misc = hw.config_read32(MISC_HOST_CTRL as u16)?;
    let mut chip_rev_id = misc >> 16;
    if chip_rev_id >> 12 == ASIC_REV_USE_PROD_ID_REG {
        chip_rev_id = hw.config_read32(GEN15_PRODID_ASICREV)?;
    }
    ensure!(chip_rev_id >> 12 == ASIC_REV_57766, "chip revision {chip_rev_id:#x} is not a 57766");
    ensure!(chip_rev_id >> 8 != CHIPREV_57765_AX, "57765 A steppings are not supported");
    Ok(Identity { vendor, device, chip_rev_id })
}

/// The station address the boot code left in SRAM (or, failing that, in the MAC; tg3_get_device_address).
pub fn station_address(hw: &dyn Hw) -> Result<[u8; 6]> {
    let hi = hw.read_mem(SRAM_MAC_ADDR_HIGH_MBOX);
    let mut mac = [0u8; 6];
    if hi >> 16 == 0x484b {
        let lo = hw.read_mem(SRAM_MAC_ADDR_LOW_MBOX);
        mac[..2].copy_from_slice(&(hi as u16).to_be_bytes());
        mac[2..].copy_from_slice(&lo.to_be_bytes());
    }
    if !valid_unicast(&mac) {
        let (hi, lo) = (hw.read32(MAC_ADDR_0_HIGH), hw.read32(MAC_ADDR_0_LOW));
        mac[..2].copy_from_slice(&(hi as u16).to_be_bytes());
        mac[2..].copy_from_slice(&lo.to_be_bytes());
    }
    ensure!(valid_unicast(&mac), "no valid station address (found {})", fmt_mac(&mac));
    Ok(mac)
}

pub fn valid_unicast(mac: &[u8; 6]) -> bool {
    mac[0] & 1 == 0 && mac.iter().any(|&b| b != 0)
}

pub fn fmt_mac(mac: &[u8; 6]) -> String {
    mac.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(":")
}

pub fn parse_mac(s: &str) -> Result<[u8; 6]> {
    let parts: Vec<_> = s.split(':').collect();
    ensure!(parts.len() == 6, "MAC address {s:?}: expected six colon-separated bytes");
    let mut mac = [0u8; 6];
    for (m, p) in mac.iter_mut().zip(parts) {
        *m = u8::from_str_radix(p, 16).with_context(|| format!("MAC address {s:?}"))?;
    }
    Ok(mac)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Accept every multicast frame (needed for IPv6 neighbour discovery on a TAP); off: broadcast and our
    /// address only.
    pub all_multi: bool,
    /// Wait this long for the link at start.
    pub link_wait: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub tx_frames: u64,
    pub tx_bytes: u64,
    pub tx_ring_full: u64,
    pub rx_frames: u64,
    pub rx_bytes: u64,
    pub rx_errors: u64,
    pub irq_services: u64,
    pub link_changes: u64,
}

/// What one pass over the status block did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Serviced {
    pub rx: usize,
    pub tx_done: usize,
    pub link: Option<Link>,
}

struct Rings {
    status: DmaBuffer,
    tx_ring: DmaBuffer,
    tx_bufs: DmaBuffer,
    rx_std: DmaBuffer,
    rx_ret: DmaBuffer,
    rx_bufs: DmaBuffer,
}

struct State {
    tx_prod: usize,
    tx_cons: usize,
    rx_ret_cons: usize,
    rx_std_prod: usize,
    last_tag: u32,
    link: Link,
    mac_mode: u32,
    tx_mode: u32,
    rx_mode: u32,
    stats: Stats,
    halted: bool,
    autoneg_at: Option<Instant>,
}

pub struct Nic {
    hw: Arc<dyn Hw>,
    pub id: Identity,
    pub mac: [u8; 6],
    opts: Options,
    st: Mutex<State>,
    // Last: freed only after Drop has halted the chip.
    rings: Rings,
}

impl Nic {
    /// Resets and starts the chip: rings posted, MAC and PHY configured, interrupts unmasked. `expect_mac`, when
    /// given, must match the station address, which is read before the chip is reset or given any memory.
    pub fn open(hw: Arc<dyn Hw>, dma: &dyn DmaSpace, expect_mac: Option<[u8; 6]>, opts: Options) -> Result<Nic> {
        let id = identify(&*hw)?;
        let cmd = hw.config_read16(PCI_COMMAND)?;
        hw.config_write16(PCI_COMMAND, cmd | PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER)?;
        let misc = MISC_HOST_CTRL_BASE | (hw.config_read32(MISC_HOST_CTRL as u16)? & MISC_HOST_CTRL_CHIPREV);
        hw.config_write32(MISC_HOST_CTRL as u16, misc)?;
        ensure!(hw.read32(MISC_HOST_CTRL) != u32::MAX, "BAR0 reads all ones: the device is not answering");
        hw.set_bits(MEMARB_MODE, MODE_ENABLE);
        hw.write32(GRC_MODE, (hw.read32(GRC_MODE) & GRC_MODE_HOST_STACKUP) | GRC_MODE_BASE);
        hw.write32(MEM_WIN_BASE_ADDR, 0);
        hw.write32(REG_BASE_ADDR, 0);
        let mac = station_address(&*hw)?;
        if let Some(want) = expect_mac {
            ensure!(mac == want, "station address is {}, not the expected {}", fmt_mac(&mac), fmt_mac(&want));
        }
        check_no_asf(&*hw)?;

        let rings = Rings {
            status: dma.alloc(4096, DmaAccess::ReadWrite)?,
            tx_ring: dma.alloc(TX_RING_SIZE * TX_BD, DmaAccess::DeviceReads)?,
            tx_bufs: dma.alloc(TX_RING_SIZE * BUF_SIZE, DmaAccess::DeviceReads)?,
            rx_std: dma.alloc(RX_STD_RING_SIZE * RX_BD, DmaAccess::DeviceReads)?,
            rx_ret: dma.alloc(RX_RET_RING_SIZE * RX_BD, DmaAccess::ReadWrite)?,
            rx_bufs: dma.alloc(RX_STD_RING_SIZE * BUF_SIZE, DmaAccess::DeviceWrites)?,
        };
        let st = State {
            tx_prod: 0,
            tx_cons: 0,
            rx_ret_cons: 0,
            rx_std_prod: RX_PENDING,
            last_tag: 0,
            link: Link::default(),
            mac_mode: MAC_MODE_RUN,
            tx_mode: TX_MODE_RUN,
            rx_mode: RX_MODE_RUN,
            stats: Stats::default(),
            halted: true,
            autoneg_at: None,
        };
        let nic = Nic { hw, id, mac, opts, st: Mutex::new(st), rings };
        nic.reset_hw(misc)?;
        nic.enable_ints(misc);
        Ok(nic)
    }

    fn hw(&self) -> &dyn Hw {
        &*self.hw
    }

    /// tg3_disable_ints
    fn disable_ints(&self, misc: u32) {
        self.hw().write32(MISC_HOST_CTRL, misc | MISC_HOST_CTRL_MASK_PCI_INT);
        self.hw().write_flush(MBOX_INTERRUPT_0, 1);
    }

    /// tg3_enable_ints, with the status block updated at once (HOSTCC_MODE_NOW).
    fn enable_ints(&self, misc: u32) {
        let hw = self.hw();
        let tag = self.st.lock().unwrap().last_tag;
        fence(Ordering::SeqCst);
        hw.write32(MISC_HOST_CTRL, misc & !MISC_HOST_CTRL_MASK_PCI_INT);
        // One-shot MSI: written twice, as tg3 does.
        hw.write_flush(MBOX_INTERRUPT_0, tag << 24);
        hw.write_flush(MBOX_INTERRUPT_0, tag << 24);
        hw.write32(HOSTCC_MODE, COALESCE_MODE | MODE_ENABLE | HOSTCC_MODE_NOW);
    }

    /// tg3_abort_hw: stops the receive and send paths, the DMA engines and host coalescing. Returns the blocks
    /// that didn't stop.
    fn abort_hw(&self, st: &mut State) -> Vec<u32> {
        let hw = self.hw();
        let mut stuck = Vec::new();
        st.rx_mode &= !RX_MODE_ENABLE;
        hw.write_flush(MAC_RX_MODE, st.rx_mode);
        hw.delay_us(10);
        for reg in [
            RCVBDI_MODE,
            RCVLPC_MODE,
            RCVDBDI_MODE,
            RCVDCC_MODE,
            RCVCC_MODE,
            SNDBDS_MODE,
            SNDBDI_MODE,
            SNDDATAI_MODE,
            RDMAC_MODE,
            SNDDATAC_MODE,
            SNDBDC_MODE,
        ] {
            if !stop_block(hw, reg) {
                stuck.push(reg);
            }
        }
        st.mac_mode &= !MAC_MODE_TDE_ENABLE;
        hw.write_flush(MAC_MODE, st.mac_mode);
        hw.delay_us(40);
        st.tx_mode &= !TX_MODE_ENABLE;
        hw.write_flush(MAC_TX_MODE, st.tx_mode);
        if !hw.poll(MAC_TX_MODE, TX_MODE_ENABLE, 0, Duration::from_millis(100)) {
            stuck.push(MAC_TX_MODE);
        }
        for reg in [HOSTCC_MODE, WDMAC_MODE] {
            if !stop_block(hw, reg) {
                stuck.push(reg);
            }
        }
        hw.write32(FTQ_RESET, 0xffff_ffff);
        hw.write32(FTQ_RESET, 0);
        stuck
    }

    /// tg3_chip_reset: a core clock reset (the PCIe core keeps its state), then the boot code's handshake.
    fn chip_reset(&self, misc: u32) -> Result<()> {
        let hw = self.hw();
        hw.write32(GRC_FASTBOOT_PC, 0);
        self.clear_status();
        let cmd = hw.config_read16(PCI_COMMAND)?;
        hw.write32(GRC_MISC_CFG, GRC_MISC_CFG_PCIE_KEEP);
        hw.write32(GRC_MISC_CFG, GRC_MISC_CFG_CORECLK_RESET | GRC_MISC_CFG_PCIE_KEEP);
        hw.delay_us(120);
        hw.config_read16(PCI_COMMAND)?;
        hw.delay_us(120);
        let exp = self.express_cap()?;
        let devctl = hw.config_read16(exp + PCI_EXP_DEVCTL)?;
        hw.config_write16(exp + PCI_EXP_DEVCTL, devctl & !(PCI_EXP_DEVCTL_RELAX_EN | PCI_EXP_DEVCTL_NOSNOOP_EN))?;
        hw.config_write16(exp + PCI_EXP_DEVSTA, 0x000f)?;
        // tg3_restore_pci_state
        hw.config_write32(MISC_HOST_CTRL as u16, misc)?;
        hw.config_write32(PCISTATE as u16, PCISTATE_ROM_ENABLE | PCISTATE_ROM_RETRY_ENABLE)?;
        hw.config_write16(PCI_COMMAND, cmd)?;
        hw.write32(MEMARB_MODE, MODE_ENABLE);
        // tg3_poll_fw: the boot code writes the complement of the magic when it is done (up to 1 s).
        let mut fw = false;
        for _ in 0..100_000 {
            if hw.read_mem(SRAM_FIRMWARE_MBOX) == !SRAM_FIRMWARE_MBOX_MAGIC1 {
                fw = true;
                break;
            }
            ensure!(hw.read32(MISC_HOST_CTRL) != u32::MAX, "device stopped answering during reset");
            hw.delay_us(10);
        }
        if !fw {
            eprintln!("ethd: no firmware running (boot code didn't answer the reset handshake)");
        }
        hw.write32(GRC_MODE, GRC_MODE_BASE);
        hw.write_flush(MAC_MODE, 0);
        hw.delay_us(40);
        // tg3_mdio_start
        hw.write_flush(MAC_MI_MODE, MI_MODE);
        hw.delay_us(80);
        check_no_asf(hw)
    }

    fn express_cap(&self) -> Result<u16> {
        let hw = self.hw();
        let mut ptr = hw.config_read16(0x34)? & 0xfc;
        for _ in 0..48 {
            if ptr < 0x40 {
                break;
            }
            let hdr = hw.config_read16(ptr)?;
            if hdr & 0xff == 0x10 {
                return Ok(ptr);
            }
            ptr = (hdr >> 8) & 0xfc;
        }
        bail!("no PCI Express capability")
    }

    /// tg3_reset_hw for a 57766 with one vector: everything from the halt to the PHY.
    fn reset_hw(&self, misc: u32) -> Result<()> {
        let hw = self.hw();
        let mut st = self.st.lock().unwrap();
        self.disable_ints(misc);
        hw.write_mem(SRAM_FIRMWARE_MBOX, SRAM_FIRMWARE_MBOX_MAGIC1);
        let stuck = self.abort_hw(&mut st);
        if !stuck.is_empty() {
            eprintln!("ethd: blocks that didn't stop before the reset: {stuck:#x?}");
        }
        // As tg3_open (reset_phy): this also powers the PHY up after tg3 powered it down.
        hw.write_flush(MAC_MI_MODE, MI_MODE);
        hw.delay_us(80);
        phy::reset(hw)?;
        self.chip_reset(misc)?;
        // From here on the chip may reach host memory: Drop halts it before the rings are freed.
        st.halted = false;

        // 57765 class, not an A stepping
        hw.set_bits(CPMU_PADRNG_CTL, CPMU_PADRNG_CTL_RDIV2);
        let grc = hw.read32(GRC_MODE);
        hw.write32(GRC_MODE, (grc & !GRC_MODE_PCIE_PORT_MASK) | GRC_MODE_PCIE_DL_SEL);
        let fts = hw.read32(PCIE_TLDLPL_PORT + PCIE_DL_LO_FTSMAX) & !PCIE_DL_LO_FTSMAX_MSK;
        hw.write32(PCIE_TLDLPL_PORT + PCIE_DL_LO_FTSMAX, fts | PCIE_DL_LO_FTSMAX_VAL);
        hw.write32(GRC_MODE, grc);
        let clk = hw.read32(CPMU_LSPD_10MB_CLK) & !CPMU_LSPD_10MB_MACCLK_MASK;
        hw.write32(CPMU_LSPD_10MB_CLK, clk | CPMU_LSPD_10MB_MACCLK_6_25);

        self.init_rings(&mut st);

        // tg3_calc_dma_bndry gives DIS_CACHE_ALIGNMENT for the 57765 family on x86.
        hw.write32(DMA_RW_CTRL, hw.read32(DMA_RW_CTRL) | DMA_RWCTRL_DIS_CACHE_ALIGNMENT);
        hw.write32(GRC_MODE, GRC_MODE_RUN | GRC_MODE_IRQ_ON_MAC_ATTN | GRC_MODE_HOST_STACKUP);
        hw.write32(GRC_MISC_CFG, (hw.read32(GRC_MISC_CFG) & !0xff) | (65 << GRC_MISC_CFG_PRESCALAR_SHIFT));

        hw.write32(BUFMGR_MB_RDMA_LOW_WATER, MB_RDMA_LOW_WATER_5705);
        hw.write32(BUFMGR_MB_MACRX_LOW_WATER, MB_MACRX_LOW_WATER_57765);
        hw.write32(BUFMGR_MB_HIGH_WATER, MB_HIGH_WATER_57765);
        hw.write32(BUFMGR_DMA_LOW_WATER, DMA_LOW_WATER);
        hw.write32(BUFMGR_DMA_HIGH_WATER, DMA_HIGH_WATER);
        hw.write32(BUFMGR_MODE, MODE_ENABLE | MODE_ATTN_ENABLE);
        ensure!(
            hw.poll(BUFMGR_MODE, MODE_ENABLE, MODE_ENABLE, Duration::from_millis(20)),
            "buffer manager won't start"
        );

        // tg3_setup_rxbd_thresholds
        hw.write32(RCVBDI_STD_THRESH, (RX_STD_BDCACHE_SIZE / 2).min((RX_PENDING as u32 / 8).max(1)));
        hw.write32(STD_REPLENISH_LWM, RX_STD_BDCACHE_SIZE);
        hw.write32(RCVBDI_JUMBO_THRESH, (RX_JMB_BDCACHE_SIZE / 2).min(100 / 8));
        hw.write32(JMB_REPLENISH_LWM, RX_JMB_BDCACHE_SIZE);

        let std = self.rings.rx_std.iova();
        hw.write32(RCVDBDI_STD_BD + BDINFO_HOST_ADDR_HIGH, (std >> 32) as u32);
        hw.write32(RCVDBDI_STD_BD + BDINFO_HOST_ADDR_LOW, std as u32);
        hw.write32(RCVDBDI_STD_BD + 0xc, 0x6000); // NIC_SRAM_RX_BUFFER_DESC
        hw.write32(RCVDBDI_JUMBO_BD + BDINFO_MAXLEN_FLAGS, BDINFO_FLAGS_DISABLED);
        hw.write32(
            RCVDBDI_STD_BD + BDINFO_MAXLEN_FLAGS,
            ((RX_STD_RING_SIZE as u32) << BDINFO_FLAGS_MAXLEN_SHIFT) | (RX_STD_DMA_SZ << 2),
        );
        hw.write32(MBOX_RCV_STD_PROD_IDX, st.rx_std_prod as u32);
        hw.write32(MBOX_RCV_JUMBO_PROD_IDX, 0);

        self.rings_reset();
        self.set_mac_addr();
        hw.write32(MAC_RX_MTU_SIZE, 1500 + 14 + 4 + 4);
        hw.write32(MAC_TX_LENGTHS, (2 << 12) | (6 << 8) | 32);
        hw.write32(MAC_RCV_RULE_CFG, RCV_RULE_CFG_DEFAULT_CLASS);
        hw.write32(RCVLPC_CONFIG, 0x0181);

        let rdmac = MODE_ENABLE
            | RDMAC_MODE_TGTABORT_ENAB
            | RDMAC_MODE_MSTABORT_ENAB
            | RDMAC_MODE_PARITYERR_ENAB
            | RDMAC_MODE_ADDROFLOW_ENAB
            | RDMAC_MODE_FIFOOFLOW_ENAB
            | RDMAC_MODE_FIFOURUN_ENAB
            | RDMAC_MODE_FIFOOREAD_ENAB
            | RDMAC_MODE_LNGREAD_ENAB
            | RDMAC_MODE_FIFO_LONG_BURST
            | RDMAC_MODE_JMB_2K_MMRR
            | RDMAC_MODE_IPV4_LSO_EN
            | RDMAC_MODE_IPV6_LSO_EN;
        hw.set_bits(RDMA_RSRVCTRL, RDMA_RSRVCTRL_FIFO_OFLW_FIX);
        hw.clear_bits(RCVLPC_STATS_ENABLE, RCVLPC_STATSENAB_DACK_FIX);
        hw.write32(RCVLPC_STATSCTRL, RCVLPC_STATSCTRL_ENABLE);
        hw.write32(SNDDATAI_STATSENAB, 0xff_ffff);
        hw.write32(SNDDATAI_STATSCTRL, SNDDATAI_SCTRL_ENABLE | SNDDATAI_SCTRL_FASTUPD);

        hw.write32(HOSTCC_MODE, 0);
        hw.poll(HOSTCC_MODE, MODE_ENABLE, 0, Duration::from_millis(20));
        // __tg3_set_coalesce: vector 0 only, vectors 1-4 zeroed
        hw.write32(HOSTCC_TXCOL_TICKS, TXCOL_TICKS);
        hw.write32(HOSTCC_TXMAX_FRAMES, TXMAX_FRAMES);
        hw.write32(HOSTCC_TXCOAL_MAXF_INT, TXCOAL_MAXF_INT);
        hw.write32(HOSTCC_RXCOL_TICKS, RXCOL_TICKS);
        hw.write32(HOSTCC_RXMAX_FRAMES, RXMAX_FRAMES);
        hw.write32(HOSTCC_RXCOAL_MAXF_INT, RXCOAL_MAXF_INT);
        for v in 0..4 {
            for r in 0..6 {
                hw.write32(HOSTCC_VEC1 + v * 0x18 + r * 4, 0);
            }
        }
        hw.write32(HOSTCC_MODE, MODE_ENABLE | COALESCE_MODE);
        hw.write32(RCVCC_MODE, MODE_ENABLE | MODE_ATTN_ENABLE);
        hw.write32(RCVLPC_MODE, MODE_ENABLE);

        st.mac_mode = MAC_MODE_RUN;
        hw.write_flush(MAC_MODE, st.mac_mode | MAC_MODE_RXSTAT_CLEAR | MAC_MODE_TXSTAT_CLEAR);
        hw.delay_us(40);
        let gpio = (hw.read32(GRC_LOCAL_CTRL) & GPIO_MASK) | GPIO_EEPROM_WP;
        hw.write_flush(GRC_LOCAL_CTRL, GRC_LCLCTRL_INT_ON_ATTN | GRC_LCLCTRL_AUTO_SEEPROM | gpio);
        hw.delay_us(100);
        // tg3_ints_init with MSI: one-shot mode stays on.
        hw.write32(MSGINT_MODE, (hw.read32(MSGINT_MODE) & !MSGINT_MODE_ONE_SHOT_DISABLE) | MSGINT_MODE_ENABLE);

        hw.write_flush(WDMAC_MODE, MODE_ENABLE | DMAC_ERROR_ENABLES | WDMAC_MODE_STATUS_TAG_FIX);
        hw.delay_us(40);
        hw.write_flush(RDMAC_MODE, rdmac);
        hw.delay_us(40);
        hw.write32(RCVDCC_MODE, MODE_ENABLE | MODE_ATTN_ENABLE);
        hw.write32(SNDDATAC_MODE, MODE_ENABLE);
        hw.write32(SNDBDC_MODE, MODE_ENABLE | MODE_ATTN_ENABLE);
        hw.write32(RCVBDI_MODE, MODE_ENABLE | MODE_ATTN_ENABLE);
        hw.write32(RCVDBDI_MODE, MODE_ENABLE | RCVDBDI_MODE_INV_RING_SZ);
        hw.write32(SNDDATAI_MODE, MODE_ENABLE | 0x8); // hardware TSO 3 path enabled, as tg3 does
        hw.write32(SNDBDI_MODE, MODE_ENABLE | MODE_ATTN_ENABLE);
        hw.write32(SNDBDS_MODE, MODE_ENABLE | MODE_ATTN_ENABLE);

        st.tx_mode = TX_MODE_RUN;
        hw.write_flush(MAC_TX_MODE, st.tx_mode);
        hw.delay_us(100);
        st.rx_mode = RX_MODE_RUN;
        hw.write_flush(MAC_RX_MODE, st.rx_mode);
        hw.delay_us(10);
        hw.write32(MAC_LED_CTRL, LED_CTRL_MODE_PHY_1);
        hw.write32(MAC_MI_STAT, MAC_MI_STAT_LNKSTAT_ATTN_ENAB);
        hw.write_flush(MAC_RX_MODE, st.rx_mode);
        hw.delay_us(10);
        hw.write_flush(MAC_LOW_WMARK_MAX_RX_FRAME, 1);

        // tg3_setup_copper_phy, without resetting the PHY: a link negotiated with our advertisement is kept.
        clear_mac_status(hw);
        phy::auxctl_write(hw, 0x0002, 0)?; // the power control shadow register cleared
        phy::read(hw, MII_TG3_ISTAT)?;
        phy::read(hw, MII_TG3_ISTAT)?;
        phy::write(hw, MII_TG3_IMASK, 0xffff)?;
        // A link that isn't up, or wasn't negotiated with our advertisement, is negotiated again
        // (tg3_phy_copper_begin).
        let link = match phy::read_link(hw)? {
            Some(link) if link.up => link,
            _ => {
                phy::restart_autoneg(hw)?;
                st.autoneg_at = Some(Instant::now());
                Link::default()
            }
        };
        self.apply_link(&mut st, link);
        hw.write_flush(MAC_EVENT, MAC_EVENT_LNKSTATE_CHANGED);
        hw.delay_us(40);

        // __tg3_set_rx_mode: no promiscuous mode, VLAN tags stripped, multicast all or nothing.
        let hash = if self.opts.all_multi { u32::MAX } else { 0 };
        for i in 0..4 {
            hw.write32(MAC_HASH_REG_0 + i * 4, hash);
        }
        // tg3's two default receive rules, with their enable bit masked off (RCV_RULE_DISABLE_MASK)
        hw.write32(MAC_RCV_RULE_0, 0xc200_0000 & RCV_RULE_DISABLE_MASK);
        hw.write32(MAC_RCV_VALUE_0, RCV_RULE_DISABLE_MASK);
        hw.write32(MAC_RCV_RULE_1, 0x8600_0004 & RCV_RULE_DISABLE_MASK);
        hw.write32(MAC_RCV_VALUE_1, RCV_RULE_DISABLE_MASK);
        for i in 4..8 {
            hw.write32(MAC_RCV_RULE_0 + i * 8, 0);
            hw.write32(MAC_RCV_VALUE_0 + i * 8, 0);
        }
        drop(st);

        if !link.up && !self.opts.link_wait.is_zero() {
            let until = Instant::now() + self.opts.link_wait;
            while Instant::now() < until {
                std::thread::sleep(Duration::from_millis(100));
                if self.check_link()?.up {
                    break;
                }
            }
        }
        Ok(())
    }

    /// tg3_init_rings: every standard ring entry points at its own buffer; the first RX_PENDING are posted.
    fn init_rings(&self, st: &mut State) {
        let r = &self.rings;
        for i in 0..RX_STD_RING_SIZE {
            let iova = r.rx_bufs.iova() + (i * BUF_SIZE) as u64;
            let o = i * RX_BD;
            r.rx_std.write32(o, (iova >> 32) as u32);
            r.rx_std.write32(o + 4, iova as u32);
            r.rx_std.write32(o + 8, RX_STD_DMA_SZ);
            r.rx_std.write32(o + 12, RXD_FLAG_END);
            for f in [16, 20, 24] {
                r.rx_std.write32(o + f, 0);
            }
            r.rx_std.write32(o + 28, RXD_OPAQUE_RING_STD | i as u32);
        }
        for o in (0..r.rx_ret.len()).step_by(4) {
            r.rx_ret.write32(o, 0);
        }
        for o in (0..r.tx_ring.len()).step_by(4) {
            r.tx_ring.write32(o, 0);
        }
        st.tx_prod = 0;
        st.tx_cons = 0;
        st.rx_ret_cons = 0;
        st.rx_std_prod = RX_PENDING;
        st.last_tag = 0;
    }

    /// tg3_rings_reset: the unused send and return rings disabled, vectors 1-4 masked, our rings' addresses
    /// in the NIC's ring control blocks.
    fn rings_reset(&self) {
        let hw = self.hw();
        hw.write_mem(SRAM_SEND_RCB + BDINFO_SIZE + BDINFO_MAXLEN_FLAGS, BDINFO_FLAGS_DISABLED);
        for i in 1..4 {
            hw.write_mem(SRAM_RCV_RET_RCB + i * BDINFO_SIZE + BDINFO_MAXLEN_FLAGS, BDINFO_FLAGS_DISABLED);
        }
        hw.write_flush(MBOX_INTERRUPT_0, 1);
        for v in 1..5 {
            hw.write32(MBOX_RCVRET_CON_IDX_0 + v * 8, 0);
            hw.write_flush(MBOX_INTERRUPT_0 + v * 8, 1);
        }
        hw.write32(MBOX_SNDHOST_PROD_IDX_0, 0);
        hw.write32(MBOX_RCVRET_CON_IDX_0, 0);
        self.clear_status();
        let sb = self.rings.status.iova();
        hw.write32(HOSTCC_STATUS_BLK_HOST_ADDR, (sb >> 32) as u32);
        hw.write32(HOSTCC_STATUS_BLK_HOST_ADDR + 4, sb as u32);
        set_bdinfo(hw, SRAM_SEND_RCB, self.rings.tx_ring.iova(), (TX_RING_SIZE as u32) << BDINFO_FLAGS_MAXLEN_SHIFT);
        set_bdinfo(
            hw,
            SRAM_RCV_RET_RCB,
            self.rings.rx_ret.iova(),
            (RX_RET_RING_SIZE as u32) << BDINFO_FLAGS_MAXLEN_SHIFT,
        );
    }

    fn clear_status(&self) {
        for o in (0..STATUS_BLOCK_SIZE).step_by(4) {
            self.rings.status.write32(o, 0);
        }
    }

    /// __tg3_set_mac_addr: the four perfect-match slots and the backoff seed.
    fn set_mac_addr(&self) {
        let hw = self.hw();
        let m = self.mac;
        let hi = u16::from_be_bytes([m[0], m[1]]) as u32;
        let lo = u32::from_be_bytes([m[2], m[3], m[4], m[5]]);
        for i in 0..4 {
            hw.write32(MAC_ADDR_0_HIGH + i * 8, hi);
            hw.write32(MAC_ADDR_0_LOW + i * 8, lo);
        }
        hw.write32(MAC_TX_BACKOFF_SEED, m.iter().map(|&b| b as u32).sum::<u32>() & TX_BACKOFF_SEED_MASK);
    }

    /// Port mode, duplex and flow control for a link result (the end of tg3_setup_copper_phy).
    fn apply_link(&self, st: &mut State, link: Link) {
        let hw = self.hw();
        st.mac_mode &= !(MAC_MODE_PORT_MODE_MASK | MAC_MODE_HALF_DUPLEX);
        st.mac_mode |= if link.up && link.speed < 1000 { MAC_MODE_PORT_MODE_MII } else { MAC_MODE_PORT_MODE_GMII };
        if link.up && !link.full_duplex {
            st.mac_mode |= MAC_MODE_HALF_DUPLEX;
        }
        hw.write_flush(MAC_MODE, st.mac_mode);
        hw.delay_us(40);
        let rx =
            if link.rx_pause { st.rx_mode | RX_MODE_FLOW_CTRL_ENABLE } else { st.rx_mode & !RX_MODE_FLOW_CTRL_ENABLE };
        if rx != st.rx_mode {
            st.rx_mode = rx;
            hw.write_flush(MAC_RX_MODE, rx);
        }
        let tx =
            if link.tx_pause { st.tx_mode | TX_MODE_FLOW_CTRL_ENABLE } else { st.tx_mode & !TX_MODE_FLOW_CTRL_ENABLE };
        if tx != st.tx_mode {
            st.tx_mode = tx;
            hw.write_flush(MAC_TX_MODE, tx);
        }
        if link != st.link {
            st.stats.link_changes += 1;
        }
        st.link = link;
    }

    /// Reads the PHY and follows a link change.
    pub fn check_link(&self) -> Result<Link> {
        let mut st = self.st.lock().unwrap();
        ensure!(!st.halted, "halted");
        let hw = self.hw();
        clear_mac_status(hw);
        let link = match phy::read_link(hw)? {
            Some(link) => link,
            None => {
                // Up, but not with our advertisement: negotiate again, at most every 5 s.
                if st.autoneg_at.is_none_or(|t| t.elapsed() > Duration::from_secs(5)) {
                    phy::restart_autoneg(hw)?;
                    st.autoneg_at = Some(Instant::now());
                }
                Link::default()
            }
        };
        if link != st.link {
            self.apply_link(&mut st, link);
        }
        // Link changes interrupt again (the end of tg3_setup_copper_phy).
        hw.write_flush(MAC_EVENT, MAC_EVENT_LNKSTATE_CHANGED);
        Ok(link)
    }

    pub fn link(&self) -> Link {
        self.st.lock().unwrap().link
    }

    pub fn stats(&self) -> Stats {
        self.st.lock().unwrap().stats
    }

    /// Queues one frame (destination address first, no FCS). `Ok(false)`: the send ring is full.
    pub fn send(&self, frame: &[u8]) -> Result<bool> {
        ensure!((14..=MAX_FRAME).contains(&frame.len()), "frame of {} bytes", frame.len());
        let mut st = self.st.lock().unwrap();
        ensure!(!st.halted, "halted");
        self.reclaim_tx(&mut st);
        if (st.tx_prod + TX_RING_SIZE - st.tx_cons) % TX_RING_SIZE == TX_RING_SIZE - 1 {
            st.stats.tx_ring_full += 1;
            return Ok(false);
        }
        let r = &self.rings;
        let slot = st.tx_prod;
        let len = frame.len().max(MIN_FRAME);
        // SAFETY: the slot's buffer is ours until its descriptor is posted below (the ring isn't full, so the
        // device has finished with it), and the copy stays inside it.
        unsafe {
            let dst = r.tx_bufs.as_ptr().add(slot * BUF_SIZE);
            std::ptr::copy_nonoverlapping(frame.as_ptr(), dst, frame.len());
            std::ptr::write_bytes(dst.add(frame.len()), 0, len - frame.len());
        }
        let iova = r.tx_bufs.iova() + (slot * BUF_SIZE) as u64;
        let o = slot * TX_BD;
        r.tx_ring.write32(o, (iova >> 32) as u32);
        r.tx_ring.write32(o + 4, iova as u32);
        r.tx_ring.write32(o + 8, ((len as u32) << TXD_LEN_SHIFT) | TXD_FLAG_END);
        r.tx_ring.write32(o + 12, 0);
        st.tx_prod = (slot + 1) % TX_RING_SIZE;
        fence(Ordering::SeqCst);
        self.hw().write32(MBOX_SNDHOST_PROD_IDX_0, st.tx_prod as u32);
        st.stats.tx_frames += 1;
        st.stats.tx_bytes += frame.len() as u64;
        Ok(true)
    }

    fn reclaim_tx(&self, st: &mut State) -> usize {
        let hw_cons = (self.rings.status.read32(16) >> 16) as usize % TX_RING_SIZE;
        let done = (hw_cons + TX_RING_SIZE - st.tx_cons) % TX_RING_SIZE;
        st.tx_cons = hw_cons;
        done
    }

    /// Frames in the send ring the chip hasn't finished.
    pub fn tx_pending(&self) -> usize {
        let mut st = self.st.lock().unwrap();
        self.reclaim_tx(&mut st);
        (st.tx_prod + TX_RING_SIZE - st.tx_cons) % TX_RING_SIZE
    }

    /// One pass over the status block (tg3_poll_work): received frames go to `on_rx`, finished sends are
    /// reclaimed, a link change is followed. Then the interrupt is acknowledged with the status tag, so the
    /// chip raises it again for anything newer.
    pub fn service(&self, on_rx: &mut dyn FnMut(&[u8])) -> Result<Serviced> {
        let mut out = Serviced::default();
        let mut link_chg = false;
        {
            let mut st = self.st.lock().unwrap();
            ensure!(!st.halted, "halted");
            let r = &self.rings;
            let tag = r.status.read32(4);
            fence(Ordering::Acquire);
            let status = r.status.read32(0);
            if status & SD_STATUS_LINK_CHG != 0 {
                r.status.write32(0, SD_STATUS_UPDATED | (status & !SD_STATUS_LINK_CHG));
                link_chg = true;
            }
            out.tx_done = self.reclaim_tx(&mut st);
            let rx_prod = (r.status.read32(16) & 0xffff) as usize % RX_RET_RING_SIZE;
            fence(Ordering::Acquire);
            let mut frame = [0u8; BUF_SIZE];
            let mut consumed = 0;
            while st.rx_ret_cons != rx_prod {
                consumed += 1;
                let o = st.rx_ret_cons * RX_BD;
                let idx_len = r.rx_ret.read32(o + 8);
                let err = r.rx_ret.read32(o + 20);
                let opaque = r.rx_ret.read32(o + 28);
                let entry = (opaque & RXD_OPAQUE_INDEX_MASK) as usize;
                let len = ((idx_len & RXD_LEN_MASK) as usize).saturating_sub(ETH_FCS_LEN);
                let bad_err = err & RXD_ERR_MASK != 0 && err & RXD_ERR_ODD_NIBBLE_RCVD_MII == 0;
                if opaque & RXD_OPAQUE_RING_MASK != RXD_OPAQUE_RING_STD
                    || entry >= RX_STD_RING_SIZE
                    || bad_err
                    || !(14..=MAX_FRAME + 4).contains(&len)
                {
                    st.stats.rx_errors += 1;
                } else {
                    // SAFETY: the entry was returned, so the device has finished writing its buffer and won't
                    // touch it until it is posted again below.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            r.rx_bufs.as_ptr().add(entry * BUF_SIZE),
                            frame.as_mut_ptr(),
                            len,
                        );
                    }
                    st.stats.rx_frames += 1;
                    st.stats.rx_bytes += len as u64;
                    out.rx += 1;
                    on_rx(&frame[..len]);
                }
                st.rx_ret_cons = (st.rx_ret_cons + 1) % RX_RET_RING_SIZE;
                st.rx_std_prod = (st.rx_std_prod + 1) % RX_STD_RING_SIZE;
            }
            let hw = self.hw();
            if consumed > 0 {
                hw.write32(MBOX_RCVRET_CON_IDX_0, st.rx_ret_cons as u32);
                hw.write32(MBOX_RCV_STD_PROD_IDX, st.rx_std_prod as u32);
            }
            st.last_tag = tag;
            st.stats.irq_services += 1;
            fence(Ordering::SeqCst);
            hw.write32(MBOX_INTERRUPT_0, tag << 24);
        }
        if link_chg {
            out.link = Some(self.check_link()?);
        }
        Ok(out)
    }

    /// tg3_halt: stops every block, resets the chip and leaves the station address in the MAC, as tg3 does
    /// when the interface goes down. Nothing reaches host memory afterwards. Returns the blocks that didn't stop.
    pub fn halt(&self) -> Result<Vec<u32>> {
        let mut st = self.st.lock().unwrap();
        if st.halted {
            return Ok(Vec::new());
        }
        let misc = MISC_HOST_CTRL_BASE | (self.hw().config_read32(MISC_HOST_CTRL as u16)? & MISC_HOST_CTRL_CHIPREV);
        self.disable_ints(misc);
        self.hw().write_mem(SRAM_FIRMWARE_MBOX, SRAM_FIRMWARE_MBOX_MAGIC1);
        let stuck = self.abort_hw(&mut st);
        st.halted = true;
        self.chip_reset(misc)?;
        self.set_mac_addr();
        Ok(stuck)
    }

    /// The registers tg3 reports for `ethtool -d`, without the statistics counters (clear on read).
    pub fn dump_regs(&self) -> Vec<(u32, u32)> {
        dump_regs(self.hw())
    }
}

impl Drop for Nic {
    fn drop(&mut self) {
        match self.halt() {
            Ok(stuck) if stuck.is_empty() => {}
            Ok(stuck) => eprintln!("ethd: halt: blocks that didn't stop: {stuck:#x?}"),
            Err(e) => eprintln!("ethd: halt: {e:#}"),
        }
    }
}

/// tg3_stop_block: clears a block's enable bit and waits up to 100 ms for it to stop.
fn stop_block(hw: &dyn Hw, reg: u32) -> bool {
    hw.write_flush(reg, hw.read32(reg) & !MODE_ENABLE);
    hw.poll(reg, MODE_ENABLE, 0, Duration::from_millis(100))
}

fn set_bdinfo(hw: &dyn Hw, rcb: u32, iova: u64, maxlen_flags: u32) {
    hw.write_mem(rcb + BDINFO_HOST_ADDR_HIGH, (iova >> 32) as u32);
    hw.write_mem(rcb + BDINFO_HOST_ADDR_LOW, iova as u32);
    hw.write_mem(rcb + BDINFO_MAXLEN_FLAGS, maxlen_flags);
}

/// tg3_clear_mac_status
fn clear_mac_status(hw: &dyn Hw) {
    hw.write32(MAC_EVENT, 0);
    hw.write_flush(
        MAC_STATUS,
        MAC_STATUS_SYNC_CHANGED | MAC_STATUS_CFG_CHANGED | MAC_STATUS_MI_COMPLETION | MAC_STATUS_LNKSTATE_CHANGED,
    );
    hw.delay_us(40);
}

/// ASF/IPMI firmware sharing the port needs a pause/resume handshake this driver doesn't implement.
fn check_no_asf(hw: &dyn Hw) -> Result<()> {
    if hw.read_mem(SRAM_DATA_SIG) == SRAM_DATA_SIG_MAGIC && hw.read_mem(SRAM_DATA_CFG) & SRAM_DATA_CFG_ASF_ENABLE != 0 {
        bail!("the NIC's ASF management firmware is enabled; not supported");
    }
    Ok(())
}

/// tg3_dump_legacy_regs' blocks (and the CPMU), skipping the clear-on-read statistics at 0x800-0x8ff.
pub const DUMP_BLOCKS: &[(u32, u32)] = &[
    (0x0000, 0xb0),
    (0x0200, 0x200),
    (0x0400, 0x4f0),
    (0x0c00, 0xe0),
    (0x1000, 0x04),
    (0x1400, 0x80),
    (0x1800, 0x48),
    (0x1c00, 0x04),
    (0x2000, 0x20),
    (0x2100, 0x15c),
    (0x2400, 0x0c),
    (0x2440, 0x3c),
    (0x2500, 0x44),
    (0x2800, 0x04),
    (0x2c00, 0x20),
    (0x3000, 0x14),
    (0x3400, 0x08),
    (0x3800, 0x08),
    (0x3c00, 0x100),
    (0x3d80, 0x180),
    (0x4000, 0x10),
    (0x4400, 0x58),
    (0x4800, 0x08),
    (0x4c00, 0x08),
    (0x5000, 0x04),
    (0x5004, 0x04),
    (0x501c, 0x04),
    (0x5034, 0x04),
    (0x5800, 0x110),
    (0x5c00, 0x120),
    (0x6000, 0x0c),
    (0x6400, 0x04),
    (0x6800, 0x4c),
];

pub fn dump_regs(hw: &dyn Hw) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for &(base, len) in DUMP_BLOCKS {
        for off in (base..base + len).step_by(4) {
            if (0x800..0x900).contains(&off) {
                continue;
            }
            out.push((off, hw.read32(off)));
        }
    }
    out
}
