//! A simulated BCM57766 for tests: registers, NIC SRAM behind the memory window, the boot code's reset handshake,
//! an MII PHY with a negotiated gigabit link, and a data path that follows what the driver programmed (ring
//! control blocks, status block address, mode enables, address filter). Frames sent to the chip's own address
//! loop back into its receive rings; everything sent is recorded as "the wire".

#![allow(clippy::single_range_in_vec_init)]

use crate::hw::Hw;
use crate::regs::*;
use anyhow::Result;
use gpud_platform::irq::IrqWait;
use gpud_platform::mock::MockDma;
use gpud_platform::{DmaAccess, DmaBuffer, DmaSpace, IrqEvent};
use std::collections::HashMap;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub const SIM_MAC: [u8; 6] = [0x00, 0x3e, 0xe1, 0xce, 0xa4, 0x17];

/// DMA through MockDma, remembering where each device address lives so the simulated chip can reach it.
pub struct SimDma {
    inner: MockDma,
    pub map: Arc<Mutex<Vec<(u64, usize, usize)>>>,
}

impl SimDma {
    pub fn new() -> SimDma {
        SimDma { inner: MockDma::new(&[1 << 32..1 << 40]), map: Arc::default() }
    }
}

impl DmaSpace for SimDma {
    fn alloc(&self, size: usize, access: DmaAccess) -> gpud_platform::Result<DmaBuffer> {
        let b = self.inner.alloc(size, access)?;
        self.map.lock().unwrap().push((b.iova(), b.as_ptr() as usize, b.len()));
        Ok(b)
    }

    unsafe fn map_external(
        &self,
        ptr: NonNull<u8>,
        size: usize,
        access: DmaAccess,
    ) -> gpud_platform::Result<DmaBuffer> {
        unsafe { self.inner.map_external(ptr, size, access) }
    }

    fn iova_ranges(&self) -> Vec<Range<u64>> {
        self.inner.iova_ranges()
    }

    fn page_sizes(&self) -> u64 {
        self.inner.page_sizes()
    }
}

#[derive(Default)]
pub struct SimIrq {
    count: Mutex<u64>,
    cv: Condvar,
}

impl SimIrq {
    fn raise(&self) {
        *self.count.lock().unwrap() += 1;
        self.cv.notify_all();
    }
}

struct IrqHandle(Arc<SimIrq>);

impl IrqWait for IrqHandle {
    fn wait(&self, timeout: Option<Duration>) -> gpud_platform::Result<u64> {
        let c = self.0.count.lock().unwrap();
        let limit = timeout.unwrap_or(Duration::from_secs(3600));
        let (mut c, _) = self.0.cv.wait_timeout_while(c, limit, |n| *n == 0).unwrap();
        if *c == 0 {
            return Err(gpud_platform::Error::Timeout);
        }
        Ok(std::mem::take(&mut *c))
    }

    fn take(&self) -> gpud_platform::Result<u64> {
        let mut c = self.0.count.lock().unwrap();
        Ok(std::mem::take(&mut *c))
    }
}

#[derive(Default)]
struct State {
    config: HashMap<u16, u32>,
    regs: HashMap<u32, u32>,
    sram: HashMap<u32, u32>,
    phy: HashMap<u32, u16>,
    core_resets: u32,
    phy_resets: u32,
    tx_cons: usize,
    std_cons: usize,
    std_prod: usize,
    ret_prod: usize,
    tag: u32,
    irq_armed: bool,
    wire: Vec<Vec<u8>>,
    dropped: u64,
    /// Every register write, in order (off, value).
    writes: Vec<(u32, u32)>,
}

pub struct SimChip {
    st: Mutex<State>,
    dma: Arc<Mutex<Vec<(u64, usize, usize)>>>,
    pub irq: Arc<SimIrq>,
}

impl SimChip {
    pub fn new(dma: &SimDma) -> Arc<SimChip> {
        let mut st = State::default();
        st.config.insert(0x00, (PCI_DEVICE_57762 as u32) << 16 | PCI_VENDOR_BROADCOM as u32);
        st.config.insert(0x04, 0);
        st.config.insert(0x34, 0xac);
        st.config.insert(0xac, 0x0002_0010); // PCI Express capability, last in the list
        st.config.insert(MISC_HOST_CTRL as u16, 0xf000_0000);
        st.config.insert(GEN15_PRODID_ASICREV, 0x5776_6000);
        let [a, b, c, d, e, f] = SIM_MAC;
        st.sram.insert(SRAM_MAC_ADDR_HIGH_MBOX, 0x484b_0000 | u32::from_be_bytes([0, 0, a, b]));
        st.sram.insert(SRAM_MAC_ADDR_LOW_MBOX, u32::from_be_bytes([c, d, e, f]));
        for (r, v) in [
            (MII_BMCR, 0x1140),
            (MII_BMSR, 0x796d),
            (MII_PHYSID1, 0x5c0d),
            (MII_PHYSID2, 0x8a40),
            (MII_ADVERTISE, 0x05e1),
            (MII_LPA, 0xc5e1),
            (MII_CTRL1000, 0x0300),
            (MII_TG3_AUX_STAT, 0x0700),
        ] {
            st.phy.insert(r, v);
        }
        Arc::new(SimChip { st: Mutex::new(st), dma: dma.map.clone(), irq: Arc::default() })
    }

    pub fn irq_event(&self) -> IrqEvent {
        IrqEvent::new(Box::new(IrqHandle(self.irq.clone())))
    }

    pub fn reg(&self, off: u32) -> u32 {
        self.read32(off)
    }

    pub fn sram(&self, off: u32) -> u32 {
        self.st.lock().unwrap().sram.get(&off).copied().unwrap_or(0)
    }

    pub fn phy_resets(&self) -> u32 {
        self.st.lock().unwrap().phy_resets
    }

    pub fn core_resets(&self) -> u32 {
        self.st.lock().unwrap().core_resets
    }

    pub fn wire(&self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.st.lock().unwrap().wire)
    }

    pub fn dropped(&self) -> u64 {
        self.st.lock().unwrap().dropped
    }

    pub fn writes(&self) -> Vec<(u32, u32)> {
        self.st.lock().unwrap().writes.clone()
    }

    pub fn set_phy(&self, reg: u32, value: u16) {
        self.st.lock().unwrap().phy.insert(reg, value);
    }

    /// A frame arriving from the cable.
    pub fn inject(&self, frame: &[u8]) {
        let mut st = self.st.lock().unwrap();
        self.receive(&mut st, frame);
        self.update_status(&mut st);
    }

    fn mem(&self, iova: u64, len: usize) -> *mut u8 {
        for &(base, ptr, size) in self.dma.lock().unwrap().iter() {
            if iova >= base && iova + len as u64 <= base + size as u64 {
                return (ptr + (iova - base) as usize) as *mut u8;
            }
        }
        panic!("device access to unmapped memory {iova:#x}+{len:#x}");
    }

    fn rd(&self, iova: u64) -> u32 {
        // SAFETY: inside a live mapping (mem checks); tests keep the buffers alive while the chip runs.
        unsafe { self.mem(iova, 4).cast::<u32>().read_volatile() }
    }

    fn wr(&self, iova: u64, v: u32) {
        // SAFETY: as in rd.
        unsafe { self.mem(iova, 4).cast::<u32>().write_volatile(v) }
    }

    fn running(st: &State, reg: u32) -> bool {
        st.regs.get(&reg).copied().unwrap_or(0) & MODE_ENABLE != 0
    }

    fn data_path_up(st: &State) -> bool {
        [MAC_TX_MODE, MAC_RX_MODE, RDMAC_MODE, WDMAC_MODE, BUFMGR_MODE, HOSTCC_MODE, SNDBDI_MODE, RCVDBDI_MODE]
            .iter()
            .all(|&r| Self::running(st, r))
            && st.regs.get(&MAC_MODE).copied().unwrap_or(0) & (MAC_MODE_TDE_ENABLE | MAC_MODE_RDE_ENABLE)
                == MAC_MODE_TDE_ENABLE | MAC_MODE_RDE_ENABLE
    }

    fn reg64(st: &State, off: u32) -> u64 {
        (st.regs.get(&off).copied().unwrap_or(0) as u64) << 32 | st.regs.get(&(off + 4)).copied().unwrap_or(0) as u64
    }

    fn sram64(st: &State, off: u32) -> u64 {
        (st.sram.get(&off).copied().unwrap_or(0) as u64) << 32 | st.sram.get(&(off + 4)).copied().unwrap_or(0) as u64
    }

    fn accepts(st: &State, dst: &[u8]) -> bool {
        let hi = st.regs.get(&MAC_ADDR_0_HIGH).copied().unwrap_or(0);
        let lo = st.regs.get(&MAC_ADDR_0_LOW).copied().unwrap_or(0);
        let mut ours = [0u8; 6];
        ours[..2].copy_from_slice(&(hi as u16).to_be_bytes());
        ours[2..].copy_from_slice(&lo.to_be_bytes());
        if dst == ours || dst == [0xff; 6] {
            return true;
        }
        dst[0] & 1 == 1 && st.regs.get(&MAC_HASH_REG_0).copied().unwrap_or(0) == u32::MAX
    }

    fn transmit(&self, st: &mut State, prod: usize) {
        if !Self::data_path_up(st) {
            return;
        }
        let ring = Self::sram64(st, SRAM_SEND_RCB);
        let size = (st.sram.get(&(SRAM_SEND_RCB + BDINFO_MAXLEN_FLAGS)).copied().unwrap_or(0) >> 16) as usize;
        assert_eq!(size, TX_RING_SIZE, "send ring size");
        while st.tx_cons != prod {
            let bd = ring + (st.tx_cons * 16) as u64;
            let addr = (self.rd(bd) as u64) << 32 | self.rd(bd + 4) as u64;
            let len_flags = self.rd(bd + 8);
            assert!(len_flags & TXD_FLAG_END != 0, "single-descriptor frames");
            let len = (len_flags >> 16) as usize;
            // SAFETY: a mapped buffer of at least len bytes (mem checks).
            let frame = unsafe { std::slice::from_raw_parts(self.mem(addr, len), len) }.to_vec();
            if Self::accepts(st, &frame[..6]) {
                self.receive(st, &frame);
            }
            st.wire.push(frame);
            st.tx_cons = (st.tx_cons + 1) % TX_RING_SIZE;
        }
        self.update_status(st);
    }

    fn receive(&self, st: &mut State, frame: &[u8]) {
        if !Self::data_path_up(st) || !Self::accepts(st, &frame[..6]) || st.std_cons == st.std_prod {
            st.dropped += 1;
            return;
        }
        let std = Self::reg64(st, RCVDBDI_STD_BD);
        let flags = st.regs.get(&(RCVDBDI_STD_BD + BDINFO_MAXLEN_FLAGS)).copied().unwrap_or(0);
        assert_eq!(flags >> 16, RX_STD_RING_SIZE as u32, "standard ring size");
        let bd = std + (st.std_cons * 32) as u64;
        let addr = (self.rd(bd) as u64) << 32 | self.rd(bd + 4) as u64;
        let room = self.rd(bd + 8) & 0xffff;
        let opaque = self.rd(bd + 28);
        let len = frame.len() + ETH_FCS_LEN;
        assert!(len as u32 <= room, "frame larger than the posted buffer");
        // SAFETY: a mapped buffer of at least `room` bytes.
        unsafe {
            let p = self.mem(addr, len);
            std::ptr::copy_nonoverlapping(frame.as_ptr(), p, frame.len());
            std::ptr::write_bytes(p.add(frame.len()), 0xcc, ETH_FCS_LEN);
        }
        let ret = Self::sram64(st, SRAM_RCV_RET_RCB);
        let ret_size = (st.sram.get(&(SRAM_RCV_RET_RCB + BDINFO_MAXLEN_FLAGS)).copied().unwrap_or(0) >> 16) as usize;
        assert_eq!(ret_size, RX_RET_RING_SIZE, "return ring size");
        let rb = ret + (st.ret_prod * 32) as u64;
        for (o, v) in [(0, addr >> 32), (4, addr & 0xffff_ffff), (8, len as u64), (12, RXD_FLAG_END as u64)] {
            self.wr(rb + o, v as u32);
        }
        self.wr(rb + 20, 0);
        self.wr(rb + 28, opaque);
        st.ret_prod = (st.ret_prod + 1) % RX_RET_RING_SIZE;
        st.std_cons = (st.std_cons + 1) % RX_STD_RING_SIZE;
    }

    fn update_status(&self, st: &mut State) {
        let sb = Self::reg64(st, HOSTCC_STATUS_BLK_HOST_ADDR);
        if sb == 0 || !Self::running(st, HOSTCC_MODE) {
            return;
        }
        st.tag = (st.tag + 1) & 0xff;
        self.wr(sb, self.rd(sb) | SD_STATUS_UPDATED);
        self.wr(sb + 4, st.tag);
        self.wr(sb + 16, (st.tx_cons as u32) << 16 | st.ret_prod as u32);
        if st.irq_armed {
            st.irq_armed = false;
            self.irq.raise();
        }
    }

    fn core_reset(st: &mut State) {
        st.core_resets += 1;
        let keep: HashMap<u32, u32> = st
            .regs
            .iter()
            .filter(|(k, _)| **k == MAC_ADDR_0_HIGH || **k == MAC_ADDR_0_LOW)
            .map(|(k, v)| (*k, *v))
            .collect();
        st.regs = keep;
        if st.sram.get(&SRAM_FIRMWARE_MBOX) == Some(&SRAM_FIRMWARE_MBOX_MAGIC1) {
            st.sram.insert(SRAM_FIRMWARE_MBOX, !SRAM_FIRMWARE_MBOX_MAGIC1);
        }
        st.config.insert(DMA_RW_CTRL as u16, 0x761b_000e);
        (st.tx_cons, st.std_cons, st.std_prod, st.ret_prod, st.tag, st.irq_armed) = (0, 0, 0, 0, 0, false);
    }

    fn mii(st: &mut State, frame: u32) -> u32 {
        let reg = (frame & MI_COM_REG_ADDR_MASK) >> MI_COM_REG_ADDR_SHIFT;
        assert_eq!((frame & MI_COM_PHY_ADDR_MASK) >> MI_COM_PHY_ADDR_SHIFT, PHY_ADDR);
        if frame & MI_COM_CMD_WRITE != 0 && reg == MII_BMCR && frame & 0x8000 != 0 {
            // A PHY reset: autonegotiation on, the PHY's default advertisement (no pause).
            st.phy.insert(MII_BMCR, 0x1140);
            st.phy.insert(MII_ADVERTISE, 0x01e1);
            st.phy_resets += 1;
            frame & !MI_COM_BUSY
        } else if frame & MI_COM_CMD_WRITE != 0 {
            st.phy.insert(reg, frame as u16);
            frame & !MI_COM_BUSY
        } else {
            (frame & !(MI_COM_BUSY | MI_COM_DATA_MASK)) | st.phy.get(&reg).copied().unwrap_or(0) as u32
        }
    }
}

impl Hw for SimChip {
    fn read32(&self, off: u32) -> u32 {
        let st = self.st.lock().unwrap();
        match off {
            MEM_WIN_DATA => {
                let base = st.config.get(&(MEM_WIN_BASE_ADDR as u16)).copied().unwrap_or(0);
                st.sram.get(&base).copied().unwrap_or(0)
            }
            o if o < 0x100 => st.config.get(&(o as u16)).copied().unwrap_or(0),
            o => st.regs.get(&o).copied().unwrap_or(0),
        }
    }

    fn write32(&self, off: u32, v: u32) {
        let mut st = self.st.lock().unwrap();
        st.writes.push((off, v));
        match off {
            MEM_WIN_DATA => {
                let base = st.config.get(&(MEM_WIN_BASE_ADDR as u16)).copied().unwrap_or(0);
                st.sram.insert(base, v);
            }
            o if o < 0x100 => {
                st.config.insert(o as u16, v);
            }
            GRC_MISC_CFG if v & GRC_MISC_CFG_CORECLK_RESET != 0 => Self::core_reset(&mut st),
            MAC_MI_COM => {
                let r = Self::mii(&mut st, v);
                st.regs.insert(off, r);
            }
            MBOX_SNDHOST_PROD_IDX_0 => {
                st.regs.insert(off, v);
                self.transmit(&mut st, v as usize % TX_RING_SIZE);
            }
            MBOX_RCV_STD_PROD_IDX => {
                st.regs.insert(off, v);
                st.std_prod = v as usize % RX_STD_RING_SIZE;
            }
            MBOX_INTERRUPT_0 => {
                st.regs.insert(off, v);
                // Tagged status: unmasked with an old tag means there is news, so interrupt at once.
                if v != 1 {
                    st.irq_armed = true;
                    if v >> 24 != st.tag {
                        st.irq_armed = false;
                        self.irq.raise();
                    }
                }
            }
            HOSTCC_MODE => {
                st.regs.insert(off, v & !HOSTCC_MODE_NOW);
                if v & HOSTCC_MODE_NOW != 0 {
                    self.update_status(&mut st);
                }
            }
            o => {
                st.regs.insert(o, v);
            }
        }
    }

    fn config_read32(&self, off: u16) -> Result<u32> {
        let st = self.st.lock().unwrap();
        let dw = st.config.get(&(off & !3)).copied().unwrap_or(0);
        Ok(dw >> ((off & 3) * 8))
    }

    fn config_write32(&self, off: u16, value: u32) -> Result<()> {
        self.st.lock().unwrap().config.insert(off, value);
        Ok(())
    }

    fn config_read16(&self, off: u16) -> Result<u16> {
        Ok(self.config_read32(off)? as u16)
    }

    fn config_write16(&self, off: u16, value: u16) -> Result<()> {
        let mut st = self.st.lock().unwrap();
        let sh = (off & 3) * 8;
        let dw = st.config.get(&(off & !3)).copied().unwrap_or(0);
        st.config.insert(off & !3, (dw & !(0xffff << sh)) | (value as u32) << sh);
        Ok(())
    }

    fn delay_us(&self, _: u64) {}
}
