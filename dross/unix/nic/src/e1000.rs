//! QEMU's 82540EM test transport. PCI ownership and all DMA come from pcid/VT-d.
//! Register/descriptor layout: Intel 8254x Software Developer's Manual, chapters 3/13.
use alloc::{vec, vec::Vec};
use core::sync::atomic::{Ordering, fence};
use dross_rt::{
    pci::{Device as PciDevice, DmaBuffer, DmaSpace, Mmio, Pci},
    sys,
};

const COUNT: usize = 32;
const BUFFER: usize = 2048;
pub struct Nic {
    mmio: Mmio,
    rx: DmaBuffer,
    tx: DmaBuffer,
    rx_data: DmaBuffer,
    tx_data: DmaBuffer,
    rx_head: usize,
    tx_tail: usize,
    pub mac: [u8; 6],
    _device: PciDevice,
    _space: DmaSpace,
}
impl Nic {
    pub fn new(pci: Pci) -> Self {
        let bdf = pci.find(0x8086, 0x100e, 0).expect("QEMU e1000");
        let device = pci.open(bdf).expect("claim e1000");
        let mmio = device.map_bar(0).expect("e1000 BAR");
        let mut space = device.dma_space().expect("e1000 DMA domain");
        device.enable().expect("bus master");
        mmio.write32(0xd8, u32::MAX);
        mmio.write32(0x100, 0);
        mmio.write32(0x400, 0);
        mmio.write32(0, mmio.read32(0) | (1 << 26));
        let deadline = sys::after_ms(100);
        while mmio.read32(0) & (1 << 26) != 0 {
            assert!(sys::clock() < deadline, "e1000 reset deadline");
            sys::yield_now();
        }
        mmio.write32(0xd8, u32::MAX);
        mmio.write32(0, mmio.read32(0) | (1 << 6));
        let low = mmio.read32(0x5400).to_le_bytes();
        let high = mmio.read32(0x5404).to_le_bytes();
        let mac = [low[0], low[1], low[2], low[3], high[0], high[1]];
        let rx = space.alloc(COUNT * 16).unwrap();
        let tx = space.alloc(COUNT * 16).unwrap();
        let rx_data = space.alloc(COUNT * BUFFER).unwrap();
        let tx_data = space.alloc(COUNT * BUFFER).unwrap();
        for i in 0..COUNT {
            let ra = rx_data.iova + (i * BUFFER) as u64;
            let ta = tx_data.iova + (i * BUFFER) as u64;
            rx.write32(i * 16, ra as u32);
            rx.write32(i * 16 + 4, (ra >> 32) as u32);
            tx.write32(i * 16, ta as u32);
            tx.write32(i * 16 + 4, (ta >> 32) as u32);
            tx.write32(i * 16 + 12, 1);
        }
        for (base, ring) in [(0x2800, &rx), (0x3800, &tx)] {
            mmio.write32(base, ring.iova as u32);
            mmio.write32(base + 4, (ring.iova >> 32) as u32);
            mmio.write32(base + 8, (COUNT * 16) as u32);
            mmio.write32(base + 16, 0);
        }
        for i in 0..128 {
            mmio.write32(0x5200 + i * 4, 0);
        }
        mmio.write32(0x2818, (COUNT - 1) as u32);
        mmio.write32(0x3818, 0);
        mmio.write32(0x410, 10 | 8 << 10 | 6 << 20);
        fence(Ordering::SeqCst);
        mmio.write32(0x400, 2 | 8 | 0x10 << 4 | 0x40 << 12);
        mmio.write32(0x100, 2 | (1 << 15) | (1 << 26));
        dross_rt::println!("e1000 ready {:02x?} status {:#x}", mac, mmio.read32(8));
        Self {
            mmio,
            rx,
            tx,
            rx_data,
            tx_data,
            rx_head: 0,
            tx_tail: 0,
            mac,
            _device: device,
            _space: space,
        }
    }
    pub fn available(&self) -> bool {
        self.tx.read32(self.tx_tail * 16 + 12) & 1 != 0
    }
    pub fn send(&mut self, bytes: &[u8]) {
        assert!(bytes.len() <= BUFFER);
        let at = self.tx_tail * BUFFER;
        // SAFETY: this TX slot's DD bit transferred ownership to software.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.tx_data.ptr + at as u64) as *mut u8,
                bytes.len(),
            );
        }
        self.tx
            .write32(self.tx_tail * 16 + 8, bytes.len() as u32 | (0x0b << 24));
        self.tx.write32(self.tx_tail * 16 + 12, 0);
        fence(Ordering::SeqCst);
        self.tx_tail = (self.tx_tail + 1) % COUNT;
        self.mmio.write32(0x3818, self.tx_tail as u32);
    }
    pub fn link(&self) -> oscar_ethernet::Link {
        oscar_ethernet::Link {
            up: self.mmio.read32(8) & 2 != 0,
            mbps: 1000,
        }
    }
    pub fn read_frame(&mut self) -> Option<Vec<u8>> {
        let status = self.rx.read32(self.rx_head * 16 + 12);
        if status & 1 == 0 {
            return None;
        }
        fence(Ordering::SeqCst);
        let len = (self.rx.read32(self.rx_head * 16 + 8) & 0xffff) as usize;
        let valid = status & 2 != 0 && status & 0xff00 == 0 && (14..=1514).contains(&len);
        let mut bytes = vec![0; if valid { len } else { 0 }];
        // SAFETY: DD transfers this bounded descriptor from the NIC to software.
        unsafe {
            core::ptr::copy_nonoverlapping(
                (self.rx_data.ptr + (self.rx_head * BUFFER) as u64) as *const u8,
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        self.rx.write32(self.rx_head * 16 + 12, 0);
        fence(Ordering::SeqCst);
        self.mmio.write32(0x2818, self.rx_head as u32);
        self.rx_head = (self.rx_head + 1) % COUNT;
        if valid { Some(bytes) } else { None }
    }
}

impl Drop for Nic {
    fn drop(&mut self) {
        self.mmio.write32(0x100, 0);
        self.mmio.write32(0x400, 0);
        let _ = self
            ._device
            .config_write(4, 2, self._device.config_read(4, 2).unwrap_or(0) & !4);
    }
}
