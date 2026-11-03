//! A driver for QEMU's `edu` test device (1234:11e8), to check the driver path end to end: a BAR through
//! pcid, MSIs as notifications, and DMA both ways through the device's IOMMU domain, plus one DMA to an address
//! that isn't mapped, which the IOMMU must stop (the kernel logs the fault).
#![no_std]
#![no_main]

extern crate alloc;

use dross_rt::bootstrap::Bootstrap;
use dross_rt::pci::{Mmio, Pci};
use dross_rt::{println, sys, Handle};

dross_rt::main!(main);

const IDENT: usize = 0x00;
const LIVENESS: usize = 0x04;
const FACTORIAL: usize = 0x08;
const STATUS: usize = 0x20;
const IRQ_STATUS: usize = 0x24;
const IRQ_RAISE: usize = 0x60;
const IRQ_ACK: usize = 0x64;
const DMA_SRC: usize = 0x80;
const DMA_DST: usize = 0x88;
const DMA_COUNT: usize = 0x90;
const DMA_CMD: usize = 0x98;
/// The device's own 4 KiB buffer, at this address on its side of DMA.
const EDU_BUFFER: u64 = 0x40000;
const DMA_IRQ: u32 = 0x100;

fn dma(m: &Mmio, irq: &dross_rt::pci::IrqEvent, src: u64, dst: u64, to_ram: bool) -> bool {
    m.write64(DMA_SRC, src);
    m.write64(DMA_DST, dst);
    m.write64(DMA_COUNT, 4096);
    m.write64(DMA_CMD, 1 | (to_ram as u64) << 1 | 4);
    let ok = irq.wait(sys::after_ms(5000)).is_ok() && m.read32(IRQ_STATUS) & DMA_IRQ != 0;
    m.write32(IRQ_ACK, DMA_IRQ);
    ok
}

fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("bootstrap");
    let pci = Pci::new(b.handle("pci").expect("pci"));
    let Ok(bdf) = pci.find(0x1234, 0x11e8, 0) else {
        println!("no edu device");
        return 2;
    };
    let dev = pci.open(bdf).expect("open");
    dev.enable().expect("enable");
    let m = dev.map_bar(0).expect("bar 0");
    let mut failed = 0;
    let mut check = |name: &str, ok: bool| {
        println!("{} {name}", if ok { "ok  " } else { "FAIL" });
        failed += !ok as i32;
    };

    let id = m.read32(IDENT);
    check("identification", id & 0xff == 0xed);
    m.write32(LIVENESS, 0x1234_5678);
    check("liveness", m.read32(LIVENESS) == !0x1234_5678);
    m.write32(FACTORIAL, 10);
    while m.read32(STATUS) & 1 != 0 {}
    check("factorial", m.read32(FACTORIAL) == 3_628_800);

    let n = sys::notify_create().unwrap();
    let irq = dev.enable_msi(n, 1).expect("msi");
    let t0 = sys::clock();
    m.write32(IRQ_RAISE, 0x5a);
    let got = irq.wait(sys::after_ms(5000));
    let us = (sys::clock() - t0) / 1000;
    check("msi", got == Ok(1) && m.read32(IRQ_STATUS) == 0x5a);
    println!("msi arrived after {us} us");
    m.write32(IRQ_ACK, 0x5a);

    let mut space = dev.dma_space().expect("dma space");
    let src = space.alloc(4096).expect("dma buffer");
    let dst = space.alloc(4096).expect("dma buffer");
    for (i, v) in src.slice().iter_mut().enumerate() {
        *v = (i * 7 + 3) as u8;
    }
    let to_dev = dma(&m, &irq, src.iova, EDU_BUFFER, false);
    let to_ram = dma(&m, &irq, EDU_BUFFER, dst.iova, true);
    check("dma to device and back", to_dev && to_ram && src.slice() == dst.slice());
    println!("dma buffers at iova {:#x} and {:#x}", src.iova, dst.iova);

    // Nothing is mapped here: the IOMMU must block the write (the device still reports completion).
    let unmapped = dst.iova + 64 * 4096;
    dma(&m, &irq, EDU_BUFFER, unmapped, true);
    println!("dma to unmapped iova {unmapped:#x} issued");

    if failed == 0 { 0 } else { 1 }
}
