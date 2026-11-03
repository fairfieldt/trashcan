//! BCM57766 registers and bits, by their names in Linux 7.3's drivers/net/ethernet/broadcom/tg3.h. Offsets are into
//! BAR0; 0x00-0xff of BAR0 shadows PCI config space.

// PCI config registers (also reachable at the same BAR0 offsets).
pub const MISC_HOST_CTRL: u32 = 0x68;
pub const DMA_RW_CTRL: u32 = 0x6c;
pub const PCISTATE: u32 = 0x70;
pub const REG_BASE_ADDR: u32 = 0x78;
pub const MEM_WIN_BASE_ADDR: u32 = 0x7c;
pub const MEM_WIN_DATA: u32 = 0x84;
pub const GEN15_PRODID_ASICREV: u16 = 0xfc;

pub const MISC_HOST_CTRL_MASK_PCI_INT: u32 = 0x0000_0002;
pub const MISC_HOST_CTRL_WORD_SWAP: u32 = 0x0000_0008;
pub const MISC_HOST_CTRL_PCISTATE_RW: u32 = 0x0000_0010;
pub const MISC_HOST_CTRL_INDIR_ACCESS: u32 = 0x0000_0080;
pub const MISC_HOST_CTRL_TAGGED_STATUS: u32 = 0x0000_0200;
pub const MISC_HOST_CTRL_CHIPREV: u32 = 0xffff_0000;
pub const DMA_RWCTRL_DIS_CACHE_ALIGNMENT: u32 = 0x0000_0001;
pub const PCISTATE_ROM_ENABLE: u32 = 0x0000_0020;
pub const PCISTATE_ROM_RETRY_ENABLE: u32 = 0x0000_0040;

/// The chip revision field of MISC_HOST_CTRL that sends the driver to the product ID register.
pub const ASIC_REV_USE_PROD_ID_REG: u32 = 0x0f;
pub const ASIC_REV_57766: u32 = 0x57766;
/// tg3_chip_rev() of the 57765 A steppings, which need extra workarounds this driver doesn't carry.
pub const CHIPREV_57765_AX: u32 = 0x577650;
pub const PCI_VENDOR_BROADCOM: u16 = 0x14e4;
/// The PCI device IDs tg3 reads the 57766 family's product ID for (BCM57762 on the Mac Pro 6,1; BCM57766).
pub const PCI_DEVICE_57762: u16 = 0x1682;
pub const PCI_DEVICE_57766: u16 = 0x1686;

// Mailboxes (the low dword of each 64-bit mailbox).
pub const MBOX_INTERRUPT_0: u32 = 0x200 + 4;
pub const MBOX_RCV_STD_PROD_IDX: u32 = 0x268 + 4;
pub const MBOX_RCV_JUMBO_PROD_IDX: u32 = 0x270 + 4;
pub const MBOX_RCVRET_CON_IDX_0: u32 = 0x280 + 4;
pub const MBOX_SNDHOST_PROD_IDX_0: u32 = 0x300 + 4;

// MAC
pub const MAC_MODE: u32 = 0x400;
pub const MAC_MODE_HALF_DUPLEX: u32 = 0x0000_0002;
pub const MAC_MODE_PORT_MODE_MII: u32 = 0x0000_0004;
pub const MAC_MODE_PORT_MODE_GMII: u32 = 0x0000_0008;
pub const MAC_MODE_PORT_MODE_MASK: u32 = 0x0000_000c;
pub const MAC_MODE_RXSTAT_ENABLE: u32 = 0x0000_0800;
pub const MAC_MODE_RXSTAT_CLEAR: u32 = 0x0000_1000;
pub const MAC_MODE_TXSTAT_ENABLE: u32 = 0x0000_4000;
pub const MAC_MODE_TXSTAT_CLEAR: u32 = 0x0000_8000;
pub const MAC_MODE_TDE_ENABLE: u32 = 0x0020_0000;
pub const MAC_MODE_RDE_ENABLE: u32 = 0x0040_0000;
pub const MAC_MODE_FHDE_ENABLE: u32 = 0x0080_0000;
pub const MAC_STATUS: u32 = 0x404;
pub const MAC_STATUS_CFG_CHANGED: u32 = 0x0000_0008;
pub const MAC_STATUS_SYNC_CHANGED: u32 = 0x0000_0010;
pub const MAC_STATUS_LNKSTATE_CHANGED: u32 = 0x0000_1000;
pub const MAC_STATUS_MI_COMPLETION: u32 = 0x0040_0000;
pub const MAC_EVENT: u32 = 0x408;
pub const MAC_EVENT_LNKSTATE_CHANGED: u32 = 0x0000_1000;
pub const MAC_LED_CTRL: u32 = 0x40c;
pub const LED_CTRL_MODE_PHY_1: u32 = 0x0000_0800;
pub const MAC_ADDR_0_HIGH: u32 = 0x410;
pub const MAC_ADDR_0_LOW: u32 = 0x414;
pub const MAC_TX_BACKOFF_SEED: u32 = 0x438;
pub const TX_BACKOFF_SEED_MASK: u32 = 0x3ff;
pub const MAC_RX_MTU_SIZE: u32 = 0x43c;
pub const MAC_MI_COM: u32 = 0x44c;
pub const MI_COM_CMD_WRITE: u32 = 0x0400_0000;
pub const MI_COM_CMD_READ: u32 = 0x0800_0000;
pub const MI_COM_READ_FAILED: u32 = 0x1000_0000;
pub const MI_COM_START: u32 = 0x2000_0000;
pub const MI_COM_BUSY: u32 = 0x2000_0000;
pub const MI_COM_PHY_ADDR_SHIFT: u32 = 21;
pub const MI_COM_PHY_ADDR_MASK: u32 = 0x03e0_0000;
pub const MI_COM_REG_ADDR_SHIFT: u32 = 16;
pub const MI_COM_REG_ADDR_MASK: u32 = 0x001f_0000;
pub const MI_COM_DATA_MASK: u32 = 0x0000_ffff;
pub const MAC_MI_STAT: u32 = 0x450;
pub const MAC_MI_STAT_LNKSTAT_ATTN_ENAB: u32 = 0x0000_0001;
pub const MAC_MI_MODE: u32 = 0x454;
pub const MAC_MI_MODE_AUTO_POLL: u32 = 0x0000_0010;
pub const MAC_MI_MODE_500KHZ_CONST: u32 = 0x0000_8000;
pub const MAC_TX_MODE: u32 = 0x45c;
pub const TX_MODE_ENABLE: u32 = 0x0000_0002;
pub const TX_MODE_FLOW_CTRL_ENABLE: u32 = 0x0000_0010;
pub const TX_MODE_MBUF_LOCKUP_FIX: u32 = 0x0000_0100;
pub const MAC_TX_LENGTHS: u32 = 0x464;
pub const MAC_RX_MODE: u32 = 0x468;
pub const RX_MODE_ENABLE: u32 = 0x0000_0002;
pub const RX_MODE_FLOW_CTRL_ENABLE: u32 = 0x0000_0004;
pub const RX_MODE_IPV6_CSUM_ENABLE: u32 = 0x0100_0000;
pub const MAC_HASH_REG_0: u32 = 0x470;
pub const MAC_RCV_RULE_0: u32 = 0x480;
pub const MAC_RCV_VALUE_0: u32 = 0x484;
pub const MAC_RCV_RULE_1: u32 = 0x488;
pub const MAC_RCV_VALUE_1: u32 = 0x48c;
pub const RCV_RULE_DISABLE_MASK: u32 = 0x7fff_ffff;
pub const MAC_RCV_RULE_CFG: u32 = 0x500;
pub const RCV_RULE_CFG_DEFAULT_CLASS: u32 = 0x0000_0008;
pub const MAC_LOW_WMARK_MAX_RX_FRAME: u32 = 0x504;

// Send and receive blocks: (mode register, enable bit, attention bit if any).
pub const SNDDATAI_MODE: u32 = 0xc00;
pub const SNDDATAI_STATSCTRL: u32 = 0xc08;
pub const SNDDATAI_STATSENAB: u32 = 0xc0c;
pub const SNDDATAI_SCTRL_ENABLE: u32 = 0x1;
pub const SNDDATAI_SCTRL_FASTUPD: u32 = 0x2;
pub const SNDDATAC_MODE: u32 = 0x1000;
pub const SNDBDS_MODE: u32 = 0x1400;
pub const SNDBDI_MODE: u32 = 0x1800;
pub const SNDBDC_MODE: u32 = 0x1c00;
pub const RCVLPC_MODE: u32 = 0x2000;
pub const RCVLPC_CONFIG: u32 = 0x2010;
pub const RCVLPC_STATSCTRL: u32 = 0x2014;
pub const RCVLPC_STATSCTRL_ENABLE: u32 = 0x1;
pub const RCVLPC_STATS_ENABLE: u32 = 0x2018;
pub const RCVLPC_STATSENAB_DACK_FIX: u32 = 0x0004_0000;
pub const RCVDBDI_MODE: u32 = 0x2400;
pub const RCVDBDI_MODE_INV_RING_SZ: u32 = 0x10;
pub const RCVDBDI_JUMBO_BD: u32 = 0x2440;
pub const RCVDBDI_STD_BD: u32 = 0x2450;
pub const RCVDCC_MODE: u32 = 0x2800;
pub const RCVBDI_MODE: u32 = 0x2c00;
pub const RCVBDI_STD_THRESH: u32 = 0x2c18;
pub const RCVBDI_JUMBO_THRESH: u32 = 0x2c1c;
pub const STD_REPLENISH_LWM: u32 = 0x2d00;
pub const JMB_REPLENISH_LWM: u32 = 0x2d04;
pub const RCVCC_MODE: u32 = 0x3000;
/// The enable bit of every block mode register above, and of the DMA engines and managers below.
pub const MODE_ENABLE: u32 = 0x2;
/// Their attention-enable bit (where the block has one).
pub const MODE_ATTN_ENABLE: u32 = 0x4;

// BDINFO blocks (in registers or NIC SRAM)
pub const BDINFO_HOST_ADDR_HIGH: u32 = 0x0;
pub const BDINFO_HOST_ADDR_LOW: u32 = 0x4;
pub const BDINFO_MAXLEN_FLAGS: u32 = 0x8;
pub const BDINFO_SIZE: u32 = 0x10;
pub const BDINFO_FLAGS_DISABLED: u32 = 0x2;
pub const BDINFO_FLAGS_MAXLEN_SHIFT: u32 = 16;

// Host coalescing
pub const HOSTCC_MODE: u32 = 0x3c00;
pub const HOSTCC_MODE_NOW: u32 = 0x0000_0008;
pub const HOSTCC_MODE_32BYTE: u32 = 0x0000_0100;
pub const HOSTCC_MODE_CLRTICK_RXBD: u32 = 0x0000_0200;
pub const HOSTCC_MODE_CLRTICK_TXBD: u32 = 0x0000_0400;
pub const HOSTCC_RXCOL_TICKS: u32 = 0x3c08;
pub const HOSTCC_TXCOL_TICKS: u32 = 0x3c0c;
pub const HOSTCC_RXMAX_FRAMES: u32 = 0x3c10;
pub const HOSTCC_TXMAX_FRAMES: u32 = 0x3c14;
pub const HOSTCC_RXCOAL_MAXF_INT: u32 = 0x3c20;
pub const HOSTCC_TXCOAL_MAXF_INT: u32 = 0x3c24;
pub const HOSTCC_STATUS_BLK_HOST_ADDR: u32 = 0x3c38;
/// Per-vector coalescing for vectors 1-4 (0x18 apart); unused with one vector, so zeroed.
pub const HOSTCC_VEC1: u32 = 0x3d80;

// CPMU and PCIe
pub const CPMU_LSPD_10MB_CLK: u32 = 0x3604;
pub const CPMU_LSPD_10MB_MACCLK_MASK: u32 = 0x001f_0000;
pub const CPMU_LSPD_10MB_MACCLK_6_25: u32 = 0x0013_0000;
pub const CPMU_PADRNG_CTL: u32 = 0x3668;
pub const CPMU_PADRNG_CTL_RDIV2: u32 = 0x0004_0000;
pub const PCIE_TLDLPL_PORT: u32 = 0x7c00;
pub const PCIE_DL_LO_FTSMAX: u32 = 0x0c;
pub const PCIE_DL_LO_FTSMAX_MSK: u32 = 0xff;
pub const PCIE_DL_LO_FTSMAX_VAL: u32 = 0x2c;

// Memory arbiter, buffer manager, DMA engines
pub const MEMARB_MODE: u32 = 0x4000;
pub const BUFMGR_MODE: u32 = 0x4400;
pub const BUFMGR_MB_RDMA_LOW_WATER: u32 = 0x4410;
pub const BUFMGR_MB_MACRX_LOW_WATER: u32 = 0x4414;
pub const BUFMGR_MB_HIGH_WATER: u32 = 0x4418;
pub const BUFMGR_DMA_LOW_WATER: u32 = 0x4434;
pub const BUFMGR_DMA_HIGH_WATER: u32 = 0x4438;
pub const RDMAC_MODE: u32 = 0x4800;
pub const RDMAC_MODE_TGTABORT_ENAB: u32 = 0x0000_0004;
pub const RDMAC_MODE_MSTABORT_ENAB: u32 = 0x0000_0008;
pub const RDMAC_MODE_PARITYERR_ENAB: u32 = 0x0000_0010;
pub const RDMAC_MODE_ADDROFLOW_ENAB: u32 = 0x0000_0020;
pub const RDMAC_MODE_FIFOOFLOW_ENAB: u32 = 0x0000_0040;
pub const RDMAC_MODE_FIFOURUN_ENAB: u32 = 0x0000_0080;
pub const RDMAC_MODE_FIFOOREAD_ENAB: u32 = 0x0000_0100;
pub const RDMAC_MODE_LNGREAD_ENAB: u32 = 0x0000_0200;
pub const RDMAC_MODE_FIFO_LONG_BURST: u32 = 0x0003_0000;
pub const RDMAC_MODE_JMB_2K_MMRR: u32 = 0x0080_0000;
pub const RDMAC_MODE_IPV4_LSO_EN: u32 = 0x0800_0000;
pub const RDMAC_MODE_IPV6_LSO_EN: u32 = 0x1000_0000;
pub const RDMA_RSRVCTRL: u32 = 0x4900;
pub const RDMA_RSRVCTRL_FIFO_OFLW_FIX: u32 = 0x4;
pub const WDMAC_MODE: u32 = 0x4c00;
pub const WDMAC_MODE_STATUS_TAG_FIX: u32 = 0x2000_0000;
/// The abort and error enables shared by RDMAC_MODE and WDMAC_MODE (bits 2-9).
pub const DMAC_ERROR_ENABLES: u32 = 0x3fc;

pub const FTQ_RESET: u32 = 0x5c00;
pub const MSGINT_MODE: u32 = 0x6000;
pub const MSGINT_MODE_ENABLE: u32 = 0x2;
pub const MSGINT_MODE_ONE_SHOT_DISABLE: u32 = 0x20;

// GRC
pub const GRC_MODE: u32 = 0x6800;
pub const GRC_MODE_WSWAP_NONFRM_DATA: u32 = 0x0000_0004;
pub const GRC_MODE_BSWAP_DATA: u32 = 0x0000_0010;
pub const GRC_MODE_WSWAP_DATA: u32 = 0x0000_0020;
pub const GRC_MODE_HOST_STACKUP: u32 = 0x0001_0000;
pub const GRC_MODE_HOST_SENDBDS: u32 = 0x0002_0000;
pub const GRC_MODE_NO_TX_PHDR_CSUM: u32 = 0x0010_0000;
pub const GRC_MODE_IRQ_ON_MAC_ATTN: u32 = 0x0400_0000;
pub const GRC_MODE_PCIE_DL_SEL: u32 = 0x2000_0000;
/// GRC_MODE_PCIE_TL_SEL (0) | GRC_MODE_PCIE_PL_SEL | GRC_MODE_PCIE_DL_SEL | GRC_MODE_PCIE_HI_1K_EN
pub const GRC_MODE_PCIE_PORT_MASK: u32 = 0x0040_0000 | 0x2000_0000 | 0x8000_0000;
pub const GRC_MISC_CFG: u32 = 0x6804;
pub const GRC_MISC_CFG_CORECLK_RESET: u32 = 0x1;
/// tg3 sets bit 29 on PCIe chips around the core reset, so the PCIe core keeps its state.
pub const GRC_MISC_CFG_PCIE_KEEP: u32 = 1 << 29;
pub const GRC_MISC_CFG_PRESCALAR_SHIFT: u32 = 1;
pub const GRC_LOCAL_CTRL: u32 = 0x6808;
pub const GRC_LCLCTRL_INT_ON_ATTN: u32 = 0x0000_0008;
pub const GRC_LCLCTRL_AUTO_SEEPROM: u32 = 0x0100_0000;
pub const GRC_FASTBOOT_PC: u32 = 0x6894;

// NIC SRAM (through the memory window)
pub const SRAM_SEND_RCB: u32 = 0x100;
pub const SRAM_RCV_RET_RCB: u32 = 0x200;
pub const SRAM_FIRMWARE_MBOX: u32 = 0xb50;
pub const SRAM_FIRMWARE_MBOX_MAGIC1: u32 = 0x4b65_7654;
pub const SRAM_DATA_SIG: u32 = 0xb54;
pub const SRAM_DATA_SIG_MAGIC: u32 = 0x4b65_7654;
pub const SRAM_DATA_CFG: u32 = 0xb58;
pub const SRAM_DATA_CFG_ASF_ENABLE: u32 = 0x80;
pub const SRAM_MAC_ADDR_HIGH_MBOX: u32 = 0xc14;
pub const SRAM_MAC_ADDR_LOW_MBOX: u32 = 0xc18;

// Defaults tg3 uses for this chip
pub const MB_RDMA_LOW_WATER_5705: u32 = 0x00;
pub const MB_MACRX_LOW_WATER_57765: u32 = 0x2a;
pub const MB_HIGH_WATER_57765: u32 = 0xa0;
pub const DMA_LOW_WATER: u32 = 0x05;
pub const DMA_HIGH_WATER: u32 = 0x0a;
pub const RX_STD_BDCACHE_SIZE: u32 = 128;
pub const RX_JMB_BDCACHE_SIZE: u32 = 64;
// Coalescing (tg3_init_coal with tagged status: the CLRTCKS tick values; the _irq ticks are 0 on 5705+)
pub const RXCOL_TICKS: u32 = 0x14;
pub const TXCOL_TICKS: u32 = 0x48;
pub const RXMAX_FRAMES: u32 = 0x05;
pub const TXMAX_FRAMES: u32 = 0x35;
pub const RXCOAL_MAXF_INT: u32 = 0x05;
pub const TXCOAL_MAXF_INT: u32 = 0x05;

// PHY registers: IEEE (include/uapi/linux/mii.h) and Broadcom (tg3.h)
pub const MII_BMCR: u32 = 0x00;
pub const MII_BMSR: u32 = 0x01;
pub const MII_PHYSID1: u32 = 0x02;
pub const MII_PHYSID2: u32 = 0x03;
pub const MII_ADVERTISE: u32 = 0x04;
pub const MII_LPA: u32 = 0x05;
pub const MII_CTRL1000: u32 = 0x09;
pub const MII_TG3_AUX_STAT: u32 = 0x19;
pub const MII_TG3_ISTAT: u32 = 0x1a;
pub const MII_TG3_IMASK: u32 = 0x1b;
pub const BMCR_ANRESTART: u32 = 0x0200;
pub const BMCR_ANENABLE: u32 = 0x1000;
pub const BMSR_LSTATUS: u32 = 0x0004;
pub const ADVERTISE_CSMA: u32 = 0x0001;
pub const ADVERTISE_10HALF: u32 = 0x0020;
pub const ADVERTISE_10FULL: u32 = 0x0040;
pub const ADVERTISE_100HALF: u32 = 0x0080;
pub const ADVERTISE_100FULL: u32 = 0x0100;
pub const ADVERTISE_PAUSE_CAP: u32 = 0x0400;
pub const ADVERTISE_PAUSE_ASYM: u32 = 0x0800;
pub const ADVERTISE_1000HALF: u32 = 0x0100;
pub const ADVERTISE_1000FULL: u32 = 0x0200;
pub const MII_TG3_AUX_STAT_SPDMASK: u32 = 0x0700;
pub const PHY_ADDR: u32 = 1;

/// Ring geometry (tg3 on a 57766: standard producer ring 512, return ring 1024, send ring 512).
pub const RX_STD_RING_SIZE: usize = 512;
pub const RX_RET_RING_SIZE: usize = 1024;
pub const TX_RING_SIZE: usize = 512;
/// Standard receive buffers posted (TG3_DEF_RX_RING_PENDING).
pub const RX_PENDING: usize = 200;
/// Receive buffer size the chip may fill (TG3_RX_STD_DMA_SZ).
pub const RX_STD_DMA_SZ: u32 = 1536;
pub const STATUS_BLOCK_SIZE: usize = 0x50;

// Descriptor fields
pub const TXD_FLAG_END: u32 = 0x0004;
pub const TXD_LEN_SHIFT: u32 = 16;
pub const RXD_FLAG_END: u32 = 0x0004;
pub const RXD_LEN_MASK: u32 = 0x0000_ffff;
pub const RXD_OPAQUE_INDEX_MASK: u32 = 0x0000_ffff;
pub const RXD_OPAQUE_RING_STD: u32 = 0x0001_0000;
pub const RXD_OPAQUE_RING_MASK: u32 = 0x0007_0000;
pub const RXD_ERR_ODD_NIBBLE_RCVD_MII: u32 = 0x0010_0000;
/// BAD_CRC | COLLISION | LINK_LOST | PHY_DECODE | MAC_ABRT | TOO_SMALL | NO_RESOURCES | HUGE_FRAME
pub const RXD_ERR_MASK: u32 =
    0x0001_0000 | 0x0002_0000 | 0x0004_0000 | 0x0008_0000 | 0x0020_0000 | 0x0040_0000 | 0x0080_0000 | 0x0100_0000;
pub const SD_STATUS_UPDATED: u32 = 0x1;
pub const SD_STATUS_LINK_CHG: u32 = 0x2;
pub const ETH_FCS_LEN: usize = 4;
