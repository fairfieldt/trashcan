//! The NHI (native host interface): the PCI function of a Thunderbolt controller that moves frames between host
//! memory and the controller's HopIDs through descriptor rings, one TX and one RX ring per HopID.
//!
//! Register layout, ring programming and the Falcon Ridge quirks follow Linux's drivers/thunderbolt/nhi.c and
//! nhi_regs.h (7.3). Rings here keep one fixed buffer per descriptor slot, posted and completed in order, which is
//! all the control channel, ThunderboltIP and USB4STREAM need.

use anyhow::{Context, Result, bail, ensure};
use gpud_platform::{DmaAccess, DmaBuffer, DmaSpace, IrqEvent, Mmio};
use std::sync::atomic::{Ordering, fence};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub const REG_TX_RING_BASE: usize = 0x00000;
pub const REG_RX_RING_BASE: usize = 0x08000;
pub const REG_TX_OPTIONS_BASE: usize = 0x19800;
pub const REG_RX_OPTIONS_BASE: usize = 0x29800;
pub const REG_RING_NOTIFY_BASE: usize = 0x37800;
pub const REG_RING_INTERRUPT_BASE: usize = 0x38200;
pub const REG_CAPS: usize = 0x39640;

const RING_FLAG_E2E_FLOW_CONTROL: u32 = 1 << 28;
const RING_FLAG_RAW: u32 = 1 << 30;
const RING_FLAG_ENABLE: u32 = 1 << 31;
const RX_OPTIONS_E2E_HOP_SHIFT: u32 = 12;
const RX_OPTIONS_E2E_HOP_MASK: u32 = 0x7ff << 12;

pub const DESC_CRC_ERROR: u32 = 0x1;
pub const DESC_COMPLETED: u32 = 0x2;
pub const DESC_POSTED: u32 = 0x4;
pub const DESC_BUFFER_OVERRUN: u32 = 0x4;
pub const DESC_INTERRUPT: u32 = 0x8;

/// Control frames are 256 bytes; data frames up to 4 KiB (a length of 0 in a descriptor means 4096).
pub const FRAME_SIZE: usize = 256;
pub const MAX_FRAME_SIZE: usize = 4096;

/// On Falcon Ridge, an RX ring with end-to-end flow control sends its credits to this unused TX HopID, and
/// automatic HopIDs start after it (Linux's QUIRK_E2E: otherwise RX packets get lost).
pub const E2E_RESERVED_HOPID: u32 = 1;

/// A ring's state that only its owner touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Tx,
    Rx,
}

/// How a ring is set up.
#[derive(Clone, Copy, Debug)]
pub struct RingConfig {
    /// `None` takes the first free HopID.
    pub hop: Option<u32>,
    pub size: u32,
    /// Frame mode (data rings): the NHI checks SOF/EOF PDFs and frames are up to 4 KiB. Without it the ring is raw
    /// (the control channel): 256-byte frames, CRC included, any PDF.
    pub frame: bool,
    /// RX only: end-to-end flow control.
    pub e2e: bool,
    /// RX only: PDF masks of the frames' first and last descriptors.
    pub sof_mask: u16,
    pub eof_mask: u16,
    /// Raise the ring's interrupt on completions (otherwise the owner polls).
    pub interrupt: bool,
}

struct Waiter {
    count: Mutex<u64>,
    cv: Condvar,
}

pub struct Nhi {
    mmio: Mmio,
    dma: Arc<dyn DmaSpace>,
    pub hop_count: u32,
    quirk_e2e: bool,
    /// Serializes read-modify-write of the interrupt enable registers, and the ring allocation tables.
    lock: Mutex<Alloc>,
    /// Per ring: interrupts seen (TX 0..hop_count, then RX).
    waiters: Vec<Waiter>,
}

struct Alloc {
    tx: Vec<bool>,
    rx: Vec<bool>,
    /// What we last wrote to the interrupt enable registers (no MMIO read needed to change one bit).
    int_enable: Vec<u32>,
}

impl Nhi {
    /// Takes over the NHI: interrupts off and their status cleared, no ring enabled.
    pub fn new(mmio: Mmio, dma: Arc<dyn DmaSpace>, falcon_ridge: bool) -> Result<Arc<Self>> {
        let caps = mmio.read32(REG_CAPS);
        ensure!(caps != u32::MAX, "NHI registers read all ones (device gone or powered off)");
        let hop_count = caps & 0x3ff;
        ensure!((2..=64).contains(&hop_count), "NHI reports {hop_count} paths (REG_CAPS {caps:#x})");
        let n = hop_count as usize;
        let nhi = Arc::new(Nhi {
            mmio,
            dma,
            hop_count,
            quirk_e2e: falcon_ridge,
            lock: Mutex::new(Alloc { tx: vec![false; n], rx: vec![false; n], int_enable: vec![0; (31 + 2 * n) / 32] }),
            waiters: (0..2 * n).map(|_| Waiter { count: Mutex::new(0), cv: Condvar::new() }).collect(),
        });
        for i in 0..n as u32 {
            nhi.stop_hw(Kind::Tx, i);
            nhi.stop_hw(Kind::Rx, i);
        }
        nhi.disable_interrupts();
        Ok(nhi)
    }

    pub fn mmio(&self) -> &Mmio {
        &self.mmio
    }

    /// Stops every ring and masks every interrupt, so the NHI no longer touches host memory (before exit).
    pub fn quiesce(&self) {
        for i in 0..self.hop_count {
            self.stop_hw(Kind::Tx, i);
            self.stop_hw(Kind::Rx, i);
        }
        self.disable_interrupts();
    }

    fn interrupt_regs(&self) -> usize {
        (31 + 2 * self.hop_count as usize) / 32
    }

    fn notify_regs(&self) -> usize {
        (31 + 3 * self.hop_count as usize) / 32
    }

    /// Intel NHIs clear the interrupt status on read (QUIRK_AUTO_CLEAR_INT): masking is a read-modify-write.
    pub fn disable_interrupts(&self) {
        self.lock.lock().unwrap().int_enable.fill(0);
        for i in 0..self.interrupt_regs() {
            self.mmio.write32(REG_RING_INTERRUPT_BASE + 4 * i, 0);
        }
        for i in 0..self.notify_regs() {
            self.mmio.read32(REG_RING_NOTIFY_BASE + 4 * i);
        }
    }

    fn interrupt_index(&self, kind: Kind, hop: u32) -> usize {
        hop as usize + if kind == Kind::Rx { self.hop_count as usize } else { 0 }
    }

    fn set_interrupt(&self, kind: Kind, hop: u32, on: bool) {
        let mut a = self.lock.lock().unwrap();
        let idx = self.interrupt_index(kind, hop);
        let v = &mut a.int_enable[idx / 32];
        let new = if on { *v | 1 << (idx % 32) } else { *v & !(1 << (idx % 32)) };
        if new != *v {
            *v = new;
            self.mmio.write32(REG_RING_INTERRUPT_BASE + idx / 32 * 4, new);
        }
    }

    /// Reads (and so clears) the notify registers and wakes the rings that completed something. Called by the
    /// interrupt thread after the MSI fires, or by anyone polling.
    pub fn service_interrupts(&self) {
        let n = self.hop_count as usize;
        let mut value = 0u32;
        for bit in 0..3 * n {
            if bit % 32 == 0 {
                value = self.mmio.read32(REG_RING_NOTIFY_BASE + 4 * (bit / 32));
            }
            if value & (1 << (bit % 32)) == 0 {
                continue;
            }
            if bit >= 2 * n {
                eprintln!("tbd: RX overflow on ring {}", bit - 2 * n);
                continue;
            }
            let w = &self.waiters[bit];
            *w.count.lock().unwrap() += 1;
            w.cv.notify_all();
        }
    }

    /// Runs the interrupt thread: every MSI services the notify registers.
    pub fn spawn_irq_thread(self: &Arc<Self>, irq: IrqEvent) {
        let nhi = self.clone();
        std::thread::Builder::new()
            .name("tbd-irq".into())
            .spawn(move || {
                loop {
                    match irq.wait(Some(Duration::from_millis(100))) {
                        Ok(_) => nhi.service_interrupts(),
                        Err(gpud_platform::Error::Timeout) => {}
                        Err(e) => {
                            eprintln!("tbd: interrupt wait failed: {e}");
                            return;
                        }
                    }
                }
            })
            .expect("spawn tbd-irq");
    }

    fn alloc_hop(&self, kind: Kind, hop: Option<u32>) -> Result<u32> {
        let mut a = self.lock.lock().unwrap();
        let table = if kind == Kind::Tx { &mut a.tx } else { &mut a.rx };
        let first = if self.quirk_e2e { E2E_RESERVED_HOPID + 1 } else { 1 };
        let hop = match hop {
            Some(h) => h,
            None => (first..self.hop_count).find(|&h| !table[h as usize]).context("no free HopID on the NHI")?,
        };
        ensure!(hop < self.hop_count, "HopID {hop} past the NHI's {}", self.hop_count);
        ensure!(!table[hop as usize], "{kind:?} ring {hop} already in use");
        table[hop as usize] = true;
        Ok(hop)
    }

    fn free_hop(&self, kind: Kind, hop: u32) {
        let mut a = self.lock.lock().unwrap();
        let table = if kind == Kind::Tx { &mut a.tx } else { &mut a.rx };
        table[hop as usize] = false;
    }

    fn desc_base(kind: Kind, hop: u32) -> usize {
        (if kind == Kind::Tx { REG_TX_RING_BASE } else { REG_RX_RING_BASE }) + hop as usize * 16
    }

    fn options_base(kind: Kind, hop: u32) -> usize {
        (if kind == Kind::Tx { REG_TX_OPTIONS_BASE } else { REG_RX_OPTIONS_BASE }) + hop as usize * 32
    }

    fn stop_hw(&self, kind: Kind, hop: u32) {
        let d = Self::desc_base(kind, hop);
        self.mmio.write32(Self::options_base(kind, hop), 0);
        self.mmio.write32(d, 0);
        self.mmio.write32(d + 4, 0);
        self.mmio.write32(d + 8, 0);
        self.mmio.write32(d + 12, 0);
    }

    /// Allocates and starts a ring. RX rings come back with every slot posted.
    pub fn ring(self: &Arc<Self>, kind: Kind, cfg: RingConfig) -> Result<Ring> {
        ensure!(cfg.size >= 2 && cfg.size <= 4096, "ring size {}", cfg.size);
        let hop = self.alloc_hop(kind, cfg.hop)?;
        let frame_bytes = if cfg.frame { MAX_FRAME_SIZE } else { FRAME_SIZE };
        let size = cfg.size as usize;
        let alloc = || -> Result<(DmaBuffer, DmaBuffer)> {
            let desc = self.dma.alloc(size * 16, DmaAccess::ReadWrite).context("ring descriptors")?;
            let access = if kind == Kind::Tx { DmaAccess::DeviceReads } else { DmaAccess::DeviceWrites };
            let bufs = self.dma.alloc(size * frame_bytes, access).context("ring buffers")?;
            Ok((desc, bufs))
        };
        let (desc, bufs) = match alloc() {
            Ok(v) => v,
            Err(e) => {
                self.free_hop(kind, hop);
                return Err(e);
            }
        };
        let mut ring = Ring {
            nhi: self.clone(),
            kind,
            hop,
            size: cfg.size,
            frame_bytes,
            desc,
            bufs,
            head: 0,
            tail: 0,
            interrupt: cfg.interrupt,
            seen: 0,
            e2e_tx_hop: 0,
        };
        // tb_ring_start.
        let d = Self::desc_base(kind, hop);
        let o = Self::options_base(kind, hop);
        let iova = ring.desc.iova();
        let mut flags = RING_FLAG_ENABLE | if cfg.frame { 0 } else { RING_FLAG_RAW };
        self.mmio.write32(d, iova as u32);
        self.mmio.write32(d + 4, (iova >> 32) as u32);
        match kind {
            Kind::Tx => {
                self.mmio.write32(d + 12, cfg.size);
                self.mmio.write32(o + 4, 0);
                self.mmio.write32(o, flags);
            }
            Kind::Rx => {
                let frame_size = if cfg.frame { 0 } else { FRAME_SIZE as u32 };
                self.mmio.write32(d + 12, frame_size << 16 | cfg.size);
                self.mmio.write32(o + 4, (cfg.sof_mask as u32) << 16 | cfg.eof_mask as u32);
                self.mmio.write32(o, flags);
                if cfg.e2e {
                    // Falcon Ridge: credits go to the reserved HopID (see E2E_RESERVED_HOPID).
                    let tx_hop = if self.quirk_e2e { E2E_RESERVED_HOPID } else { hop };
                    ring.e2e_tx_hop = tx_hop;
                    flags |= (tx_hop << RX_OPTIONS_E2E_HOP_SHIFT) & RX_OPTIONS_E2E_HOP_MASK;
                    flags |= RING_FLAG_E2E_FLOW_CONTROL;
                    self.mmio.write32(o, flags);
                }
            }
        }
        if kind == Kind::Rx {
            for _ in 0..cfg.size - 1 {
                ring.post_rx();
            }
        }
        Ok(ring)
    }

    fn waiter(&self, kind: Kind, hop: u32) -> &Waiter {
        &self.waiters[self.interrupt_index(kind, hop)]
    }
}

/// One TX or RX ring with its descriptors and frame buffers. Slot `i` always uses buffer `i`.
pub struct Ring {
    nhi: Arc<Nhi>,
    pub kind: Kind,
    pub hop: u32,
    pub size: u32,
    frame_bytes: usize,
    desc: DmaBuffer,
    bufs: DmaBuffer,
    /// Next slot software posts.
    head: u32,
    /// Next slot the NHI completes.
    tail: u32,
    interrupt: bool,
    seen: u64,
    /// RX with E2E: the TX HopID its credits go to.
    pub e2e_tx_hop: u32,
}

/// A completed RX frame: where its data is and what the NHI reported.
#[derive(Clone, Copy, Debug)]
pub struct RxFrame {
    pub slot: u32,
    pub len: usize,
    #[allow(dead_code)]
    pub sof: u8,
    pub eof: u8,
    pub flags: u32,
}

impl Ring {
    fn desc_write(&self, slot: u32, phys: u64, word2: u32) {
        let o = slot as usize * 16;
        self.desc.write32(o, phys as u32);
        self.desc.write32(o + 4, (phys >> 32) as u32);
        self.desc.write32(o + 12, 0);
        self.desc.write32(o + 8, word2);
    }

    fn desc_word2(&self, slot: u32) -> u32 {
        self.desc.read32(slot as usize * 16 + 8)
    }

    fn buf_iova(&self, slot: u32) -> u64 {
        self.bufs.iova() + slot as u64 * self.frame_bytes as u64
    }

    pub fn full(&self) -> bool {
        (self.head + 1) % self.size == self.tail
    }

    pub fn empty(&self) -> bool {
        self.head == self.tail
    }

    /// Slots posted to the NHI and not completed yet.
    pub fn in_flight(&self) -> u32 {
        (self.head + self.size - self.tail) % self.size
    }

    fn desc_flags(&self) -> u32 {
        DESC_POSTED | if self.interrupt { DESC_INTERRUPT } else { 0 }
    }

    fn doorbell(&self) {
        fence(Ordering::SeqCst);
        let r = Nhi::desc_base(self.kind, self.hop) + 8;
        match self.kind {
            Kind::Tx => self.nhi.mmio.write32(r, self.head << 16),
            Kind::Rx => self.nhi.mmio.write32(r, self.head),
        }
    }

    fn post_rx(&mut self) {
        debug_assert!(!self.full());
        self.desc_write(self.head, self.buf_iova(self.head), self.desc_flags() << 20);
        self.head = (self.head + 1) % self.size;
        self.doorbell();
    }

    /// TX: the buffer of the next free slot, or `None` while the ring is full.
    pub fn tx_slot(&mut self) -> Option<&mut [u8]> {
        if self.full() {
            return None;
        }
        let o = self.head as usize * self.frame_bytes;
        // SAFETY: the NHI only reads slots between tail and head; head is not posted yet.
        Some(unsafe { &mut self.bufs.as_mut_slice()[o..o + self.frame_bytes] })
    }

    /// TX: free slots.
    pub fn tx_room(&self) -> u32 {
        self.size - 1 - self.in_flight()
    }

    /// TX: posts the slot filled through `tx_slot` with `len` bytes.
    pub fn tx_submit(&mut self, len: usize, sof: u8, eof: u8) {
        assert!(!self.full() && len > 0 && len <= self.frame_bytes, "tx_submit of {len} bytes");
        let word2 = (len as u32 & 0xfff) | (eof as u32 & 0xf) << 12 | (sof as u32 & 0xf) << 16 | self.desc_flags() << 20;
        self.desc_write(self.head, self.buf_iova(self.head), word2);
        self.head = (self.head + 1) % self.size;
        self.doorbell();
    }

    /// TX: frees the slots the NHI has sent; returns how many.
    pub fn tx_reap(&mut self) -> u32 {
        let mut n = 0;
        while !self.empty() && self.desc_word2(self.tail) >> 20 & DESC_COMPLETED != 0 {
            self.tail = (self.tail + 1) % self.size;
            n += 1;
        }
        fence(Ordering::Acquire);
        n
    }

    /// RX: the next completed frame, if any. Its data stays valid until `rx_release`.
    pub fn rx_peek(&self) -> Option<RxFrame> {
        self.rx_peek_nth(0)
    }

    /// RX: the `k`-th completed frame after the next one (frames complete in order).
    pub fn rx_peek_nth(&self, k: u32) -> Option<RxFrame> {
        if k >= self.in_flight() {
            return None;
        }
        let slot = (self.tail + k) % self.size;
        let w = self.desc_word2(slot);
        let flags = w >> 20;
        if flags & DESC_COMPLETED == 0 {
            return None;
        }
        fence(Ordering::Acquire);
        let len = match w & 0xfff {
            0 => MAX_FRAME_SIZE,
            l => l as usize,
        };
        Some(RxFrame { slot, len, eof: (w >> 12 & 0xf) as u8, sof: (w >> 16 & 0xf) as u8, flags })
    }

    pub fn rx_data(&self, f: &RxFrame) -> &[u8] {
        let o = f.slot as usize * self.frame_bytes;
        // SAFETY: the slot is completed and not reposted until rx_release.
        unsafe { &self.bufs.as_slice()[o..o + f.len.min(self.frame_bytes)] }
    }

    /// RX: gives the frame from `rx_peek` back to the NHI.
    pub fn rx_release(&mut self) {
        debug_assert!(!self.empty());
        self.tail = (self.tail + 1) % self.size;
        self.post_rx();
    }

    /// The descriptor `k` slots past the tail is done (an RX frame to take, or a TX slot to reap).
    pub fn has_work(&self, k: u32) -> bool {
        k < self.in_flight() && self.desc_word2((self.tail + k) % self.size) >> 20 & DESC_COMPLETED != 0
    }

    /// Sleeps until the NHI completes a descriptor on this ring, or `timeout`. The ring's interrupt is unmasked
    /// only while sleeping (as NAPI does): a busy ring raises none.
    pub fn wait(&mut self, timeout: Duration) {
        self.wait_nth(0, timeout)
    }

    /// `wait` for the descriptor `k` slots past the tail (RX frames before it are held, not yet released).
    pub fn wait_nth(&mut self, k: u32, timeout: Duration) {
        if !self.interrupt {
            // Busy polling: give the core back briefly, nothing more.
            std::thread::yield_now();
            return;
        }
        let w = self.nhi.waiter(self.kind, self.hop);
        let g = w.count.lock().unwrap();
        self.seen = *g;
        drop(g);
        self.nhi.set_interrupt(self.kind, self.hop, true);
        // A completion that landed before the unmask raised nothing.
        if !self.has_work(k) {
            let g = w.count.lock().unwrap();
            let _ = w.cv.wait_timeout_while(g, timeout, |c| *c == self.seen).unwrap();
        }
        self.nhi.set_interrupt(self.kind, self.hop, false);
    }
}

impl Drop for Ring {
    /// tb_ring_stop, then the buffers go once the NHI no longer points at them.
    fn drop(&mut self) {
        if self.interrupt {
            self.nhi.set_interrupt(self.kind, self.hop, false);
        }
        self.nhi.stop_hw(self.kind, self.hop);
        self.nhi.free_hop(self.kind, self.hop);
    }
}

/// Falcon Ridge 2C and 4C NHIs.
pub fn is_falcon_ridge(vendor: u16, device: u16) -> bool {
    vendor == 0x8086 && matches!(device, 0x156a | 0x156c)
}

pub fn check_supported(vendor: u16, device: u16) -> Result<()> {
    if !is_falcon_ridge(vendor, device) {
        bail!("{vendor:04x}:{device:04x} is not a Falcon Ridge NHI (8086:156a/156c); tbd supports only those so far");
    }
    Ok(())
}
