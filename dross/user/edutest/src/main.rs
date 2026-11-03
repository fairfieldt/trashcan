//! A driver for QEMU's `edu` test device (1234:11e8), written against gpud-platform (`driver/platform`) as gpud
//! is, so it checks the dross backend end to end: config space and a BAR through pcid, an MSI as a
//! notification, DMA both ways through the function's IOMMU domain, and one DMA to an address that isn't mapped,
//! which the IOMMU must stop (the kernel logs the fault).
#![no_std]
#![no_main]

extern crate alloc;

use core::time::Duration;
use dross_rt::bootstrap::Bootstrap;
use dross_rt::pci::Pci;
use dross_rt::{println, Handle};
use gpud_platform::{DmaAccess, DmaSpace, IrqEvent, Mmio, PciAddress, PciDevice, Region};

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

fn dma(m: &Mmio, irq: &IrqEvent, src: u64, dst: u64, to_ram: bool) -> bool {
    // Device addresses stay below 4 GiB, so the low halves of the 64-bit registers are enough.
    m.write32(DMA_SRC, src as u32);
    m.write32(DMA_DST, dst as u32);
    m.write32(DMA_COUNT, 4096);
    m.write32(DMA_CMD, 1 | (to_ram as u32) << 1 | 4);
    let ok = irq.wait(Some(Duration::from_secs(5))).is_ok() && m.read32(IRQ_STATUS) & DMA_IRQ != 0;
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
    let addr = PciAddress { domain: 0, bus: (bdf >> 8) as u8, device: (bdf >> 3 & 0x1f) as u8, function: (bdf & 7) as u8 };
    let (dev, space) = gpud_platform::dross::open(&pci, addr).expect("open");
    let mut failed = 0;
    let mut check = |name: &str, ok: bool| {
        println!("{} {name}", if ok { "ok  " } else { "FAIL" });
        failed += !ok as i32;
    };

    check("config space", dev.config_u16(0).ok() == Some(0x1234) && dev.config_u16(2).ok() == Some(0x11e8));
    let caps = gpud_platform::pci::capabilities(&dev).expect("capabilities");
    check("capabilities", caps.find(gpud_platform::pci::CAP_ID_MSI).is_some());
    // Bus master on; a write to the BARs is pcid's to refuse.
    let cmd = dev.config_u16(4).unwrap();
    dev.set_config_u16(4, cmd | 0x6).unwrap();
    check("filtered config write", dev.write(Region::Config, 0x10, &[0; 4]).is_err());

    let m = dev.map(Region::Bar(0)).expect("BAR 0");
    check("identification", m.read32(IDENT) & 0xff == 0xed);
    m.write32(LIVENESS, 0x1234_5678);
    check("liveness", m.read32(LIVENESS) == !0x1234_5678);
    m.write32(FACTORIAL, 10);
    while m.read32(STATUS) & 1 != 0 {}
    check("factorial", m.read32(FACTORIAL) == 3_628_800);

    let irq = dev.enable_msi().expect("msi");
    m.write32(IRQ_RAISE, 0x5a);
    let got = irq.wait(Some(Duration::from_secs(5)));
    check("msi", got.is_ok() && m.read32(IRQ_STATUS) == 0x5a);
    m.write32(IRQ_ACK, 0x5a);
    check("no interrupt pending", irq.take().ok() == Some(0));

    let mut src = space.alloc(4096, DmaAccess::DeviceReads).expect("dma buffer");
    let dst = space.alloc(4096, DmaAccess::DeviceWrites).expect("dma buffer");
    // SAFETY: the device doesn't use the buffer yet.
    for (i, v) in unsafe { src.as_mut_slice() }.iter_mut().enumerate() {
        *v = (i * 7 + 3) as u8;
    }
    let to_dev = dma(&m, &irq, src.iova(), EDU_BUFFER, false);
    let to_ram = dma(&m, &irq, EDU_BUFFER, dst.iova(), true);
    // SAFETY: the device is done with both.
    let same = unsafe { src.as_slice() == dst.as_slice() };
    check("dma to device and back", to_dev && to_ram && same);
    println!("dma buffers at iova {:#x} and {:#x}", src.iova(), dst.iova());

    // Nothing is mapped here: the IOMMU must block the write (the device still reports completion).
    let unmapped = dst.iova() + 64 * 4096;
    dma(&m, &irq, EDU_BUFFER, unmapped, true);
    println!("dma to unmapped iova {unmapped:#x} issued");
    dev.disable_irqs().expect("disable interrupts");

    if failed == 0 { 0 } else { 1 }
}
