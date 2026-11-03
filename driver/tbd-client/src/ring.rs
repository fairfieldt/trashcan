//! One NHI descriptor ring (Linux's drivers/thunderbolt/nhi.c `ring_write_descriptors`, `ring_work`): descriptors
//! and frame buffers in memory the NHI reaches, and the ring's index register in the NHI's BAR. Slot `i` always
//! uses buffer `i`; slots are posted and completed in order.
//!
//! tbd runs its own rings on this, and a client process runs a stream's two rings on it once tbd has handed them
//! over (the same memory mapped in both, the BAR mapped in both).

use std::ptr::NonNull;
use std::sync::atomic::{Ordering, fence};

pub const DESC_CRC_ERROR: u32 = 0x1;
pub const DESC_COMPLETED: u32 = 0x2;
pub const DESC_POSTED: u32 = 0x4;
pub const DESC_BUFFER_OVERRUN: u32 = 0x4;
pub const DESC_INTERRUPT: u32 = 0x8;

/// Data frames are up to 4 KiB (a length of 0 in a descriptor means 4096).
pub const MAX_FRAME_SIZE: usize = 4096;

/// A completed RX frame: where its data is and what the NHI reported.
#[derive(Clone, Copy, Debug)]
pub struct RxFrame {
    pub slot: u32,
    pub len: usize,
    pub sof: u8,
    pub eof: u8,
    pub flags: u32,
}

pub struct RingCore {
    pub tx: bool,
    pub size: u32,
    frame_bytes: usize,
    desc: NonNull<u8>,
    bufs: NonNull<u8>,
    bufs_iova: u64,
    /// The ring's index register (descriptor base + 8): TX producer in bits 16-31, RX head in bits 0-15.
    index_reg: NonNull<u32>,
    /// Next slot software posts.
    pub head: u32,
    /// Next slot the NHI completes.
    pub tail: u32,
    /// Flags for posted descriptors: POSTED, and INTERRUPT if completions should raise the ring's interrupt.
    desc_flags: u32,
}

// SAFETY: the ring is plain memory and a register; one owner drives it at a time.
unsafe impl Send for RingCore {}

impl RingCore {
    /// # Safety
    /// `desc` must hold `size` 16-byte descriptors and `bufs` `size` frames of `frame_bytes`, both mapped for the
    /// NHI (`bufs` at `bufs_iova`) and for this process while the ring lives; `index_reg` is the ring's register.
    pub unsafe fn new(
        tx: bool,
        size: u32,
        frame_bytes: usize,
        desc: NonNull<u8>,
        bufs: NonNull<u8>,
        bufs_iova: u64,
        index_reg: NonNull<u32>,
        interrupt: bool,
    ) -> Self {
        let desc_flags = DESC_POSTED | if interrupt { DESC_INTERRUPT } else { 0 };
        RingCore { tx, size, frame_bytes, desc, bufs, bufs_iova, index_reg, head: 0, tail: 0, desc_flags }
    }

    fn desc_ptr(&self, slot: u32, word: usize) -> *mut u32 {
        // SAFETY: slot < size, word < 4: inside the descriptor array.
        unsafe { self.desc.as_ptr().add(slot as usize * 16 + word * 4).cast::<u32>() }
    }

    fn desc_write(&self, slot: u32, phys: u64, word2: u32) {
        // SAFETY: in-bounds, aligned volatile stores to descriptor memory.
        unsafe {
            self.desc_ptr(slot, 0).write_volatile(phys as u32);
            self.desc_ptr(slot, 1).write_volatile((phys >> 32) as u32);
            self.desc_ptr(slot, 3).write_volatile(0);
            self.desc_ptr(slot, 2).write_volatile(word2);
        }
    }

    fn desc_word2(&self, slot: u32) -> u32 {
        // SAFETY: an in-bounds, aligned volatile load the NHI may have written.
        unsafe { self.desc_ptr(slot, 2).read_volatile() }
    }

    fn buf_iova(&self, slot: u32) -> u64 {
        self.bufs_iova + slot as u64 * self.frame_bytes as u64
    }

    fn buf(&self, slot: u32) -> *mut u8 {
        // SAFETY: slot < size: inside the buffer area.
        unsafe { self.bufs.as_ptr().add(slot as usize * self.frame_bytes) }
    }

    pub fn full(&self) -> bool {
        (self.head + 1) % self.size == self.tail
    }

    pub fn empty(&self) -> bool {
        self.head == self.tail
    }

    /// Slots posted to the NHI and not completed (or, on RX, not released) yet.
    pub fn in_flight(&self) -> u32 {
        (self.head + self.size - self.tail) % self.size
    }

    /// TX: free slots.
    pub fn tx_room(&self) -> u32 {
        self.size - 1 - self.in_flight()
    }

    fn doorbell(&self) {
        fence(Ordering::SeqCst);
        let v = if self.tx { self.head << 16 } else { self.head };
        // SAFETY: the ring's index register in the mapped BAR.
        unsafe { self.index_reg.as_ptr().write_volatile(v) };
    }

    /// RX: posts the slot at the head (its buffer) to the NHI.
    pub fn post_rx(&mut self) {
        debug_assert!(!self.tx && !self.full());
        self.desc_write(self.head, self.buf_iova(self.head), self.desc_flags << 20);
        self.head = (self.head + 1) % self.size;
        self.doorbell();
    }

    /// RX: posts every free slot (a new ring).
    pub fn post_all_rx(&mut self) {
        while !self.full() {
            self.desc_write(self.head, self.buf_iova(self.head), self.desc_flags << 20);
            self.head = (self.head + 1) % self.size;
        }
        self.doorbell();
    }

    /// TX: the buffer of the next free slot, or `None` while the ring is full.
    pub fn tx_slot(&mut self) -> Option<&mut [u8]> {
        if self.full() {
            return None;
        }
        // SAFETY: the NHI only reads slots between tail and head; head is not posted yet.
        Some(unsafe { std::slice::from_raw_parts_mut(self.buf(self.head), self.frame_bytes) })
    }

    /// TX: queues the slot filled through `tx_slot` with `len` bytes; the NHI sees it at the next `tx_flush`.
    /// Ring the doorbell for every frame (`tx_submit`): with several frames behind one doorbell, Falcon Ridge's
    /// TX engine was seen to send some and stop until the next doorbell.
    pub fn tx_queue(&mut self, len: usize, sof: u8, eof: u8) {
        assert!(!self.full() && len > 0 && len <= self.frame_bytes, "tx of {len} bytes");
        let word2 = (len as u32 & 0xfff) | (eof as u32 & 0xf) << 12 | (sof as u32 & 0xf) << 16 | self.desc_flags << 20;
        self.desc_write(self.head, self.buf_iova(self.head), word2);
        self.head = (self.head + 1) % self.size;
    }

    pub fn tx_flush(&self) {
        self.doorbell();
    }

    /// TX: writes the producer index again, unchanged.
    pub fn tx_kick(&self) {
        self.doorbell();
    }

    /// TX: queues and posts one slot.
    pub fn tx_submit(&mut self, len: usize, sof: u8, eof: u8) {
        self.tx_queue(len, sof, eof);
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

    /// The descriptor `k` slots past the tail is done.
    pub fn has_work(&self, k: u32) -> bool {
        k < self.in_flight() && self.desc_word2((self.tail + k) % self.size) >> 20 & DESC_COMPLETED != 0
    }

    /// RX: the `k`-th completed frame from the tail (frames complete in order). Its data stays valid until released.
    pub fn rx_peek_nth(&self, k: u32) -> Option<RxFrame> {
        if !self.has_work(k) {
            return None;
        }
        let slot = (self.tail + k) % self.size;
        let w = self.desc_word2(slot);
        fence(Ordering::Acquire);
        let len = match w & 0xfff {
            0 => MAX_FRAME_SIZE,
            l => l as usize,
        };
        Some(RxFrame { slot, len: len.min(self.frame_bytes), eof: (w >> 12 & 0xf) as u8, sof: (w >> 16 & 0xf) as u8, flags: w >> 20 })
    }

    pub fn rx_peek(&self) -> Option<RxFrame> {
        self.rx_peek_nth(0)
    }

    pub fn rx_data(&self, f: &RxFrame) -> &[u8] {
        // SAFETY: the slot is completed and not reposted until released.
        unsafe { std::slice::from_raw_parts(self.buf(f.slot), f.len) }
    }

    /// RX: gives the frame at the tail back to the NHI.
    pub fn rx_release(&mut self) {
        debug_assert!(!self.empty());
        self.tail = (self.tail + 1) % self.size;
        self.post_rx();
    }

    /// TX: takes over a ring someone else drove: the NHI's consumer index and the last producer index written,
    /// from the ring's register.
    pub fn resync_tx(&mut self) {
        // SAFETY: a read of the ring's register.
        let v = unsafe { self.index_reg.as_ptr().read_volatile() };
        self.tail = (v & 0xffff) % self.size;
        self.head = (v >> 16) % self.size;
    }

    /// The descriptors from the tail on (word 2: length, PDFs, flags), and the index register.
    pub fn debug_descs(&self, n: u32) -> String {
        // SAFETY: a read of the ring's register.
        let reg = unsafe { self.index_reg.as_ptr().read_volatile() };
        let d: Vec<String> = (0..n).map(|k| format!("{:08x}", self.desc_word2((self.tail + k) % self.size))).collect();
        format!("reg {reg:08x} descs {}", d.join(" "))
    }

    /// TX: the NHI's consumer index caught up with the producer.
    pub fn tx_hw_idle(&self) -> bool {
        // SAFETY: a read of the ring's register.
        let v = unsafe { self.index_reg.as_ptr().read_volatile() };
        v & 0xffff == v >> 16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ring over plain memory, with the NHI's part played by the test.
    struct Fake {
        mem: Vec<u64>,
        reg: Box<u32>,
    }

    fn ring(f: &mut Fake, tx: bool, size: u32) -> RingCore {
        let desc = NonNull::new(f.mem.as_mut_ptr().cast::<u8>()).unwrap();
        // SAFETY: the buffers follow the descriptors in f.mem (size * 16 + size * 64 bytes).
        let bufs = unsafe { NonNull::new_unchecked(desc.as_ptr().add(size as usize * 16)) };
        let reg = NonNull::from(&mut *f.reg);
        // SAFETY: as documented, with 64-byte frames.
        unsafe { RingCore::new(tx, size, 64, desc, bufs, 0x1000, reg, false) }
    }

    fn complete(r: &RingCore, slot: u32, len: u32, eof: u32) {
        let w = r.desc_word2(slot);
        // SAFETY: test memory.
        unsafe { r.desc_ptr(slot, 2).write_volatile((w & !0xffff) | len | eof << 12 | DESC_COMPLETED << 20) };
    }

    #[test]
    fn tx_and_rx_in_order() {
        let mut f = Fake { mem: vec![0; 4 * (16 + 64) / 8 * 2], reg: Box::new(0) };
        let mut tx = ring(&mut f, true, 4);
        assert_eq!(tx.tx_room(), 3);
        tx.tx_slot().unwrap()[..3].copy_from_slice(b"abc");
        tx.tx_submit(3, 1, 2);
        assert_eq!(*f.reg, 1 << 16);
        assert_eq!(tx.desc_word2(0) & 0xffff, 3 | 2 << 12);
        assert_eq!(tx.tx_reap(), 0);
        complete(&tx, 0, 3, 2);
        assert_eq!(tx.tx_reap(), 1);
        assert!(tx.empty());

        let mut f = Fake { mem: vec![0; 4 * (16 + 64) / 8 * 2], reg: Box::new(0) };
        let mut rx = ring(&mut f, false, 4);
        rx.post_all_rx();
        assert_eq!(*f.reg, 3);
        assert!(rx.rx_peek().is_none());
        complete(&rx, 0, 10, 2);
        complete(&rx, 1, 0, 3);
        let a = rx.rx_peek().unwrap();
        assert_eq!((a.slot, a.len, a.eof), (0, 10, 2));
        assert_eq!(rx.rx_peek_nth(1).unwrap().len, 64);
        assert!(rx.rx_peek_nth(2).is_none());
        rx.rx_release();
        assert_eq!(*f.reg, 0);
        assert_eq!(rx.rx_peek().unwrap().eof, 3);
    }
}
