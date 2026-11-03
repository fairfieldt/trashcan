//! usbd: an xHCI driver for USB keyboards and mice (HID boot protocol), for local debugging at the machine.
//!
//! It takes the controller from the firmware, sets up the command and event rings, enumerates the devices on
//! the root ports, configures the HID boot interfaces it finds and logs keys and mouse movements. No hubs yet
//! (devices must be on a root port), no other device classes. The Mac Pro 6,1's rear ports are on a Fresco
//! Logic FL1100 xHCI (1b73:1100).
#![no_std]
#![no_main]

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{compiler_fence, Ordering};
use dross_rt::bootstrap::Bootstrap;
use dross_rt::pci::{DmaBuffer, DmaSpace, IrqEvent, Mmio, Pci};
use dross_rt::{println, sys, Handle};

dross_rt::main!(main);

// Operational registers (from CAPLENGTH).
const USBCMD: usize = 0x00;
const USBSTS: usize = 0x04;
const CRCR: usize = 0x18;
const DCBAAP: usize = 0x30;
const CONFIG: usize = 0x38;
const PORTSC: usize = 0x400;

const CMD_RUN: u32 = 1;
const CMD_RESET: u32 = 1 << 1;
const CMD_INTE: u32 = 1 << 2;
const STS_HALTED: u32 = 1;
const STS_NOT_READY: u32 = 1 << 11;

const PORT_CONNECTED: u32 = 1;
const PORT_ENABLED: u32 = 1 << 1;
const PORT_RESET: u32 = 1 << 4;
const PORT_RESET_CHANGE: u32 = 1 << 21;
/// PORTSC bits that read back what they write (the rest are write-1-to-clear or self-clearing).
const PORT_KEEP: u32 = 0x4e00_fde9 & !PORT_ENABLED;
const PORT_CHANGES: u32 = 0x00fe_0000;

const TRB_NORMAL: u32 = 1;
const TRB_SETUP: u32 = 2;
const TRB_DATA: u32 = 3;
const TRB_STATUS: u32 = 4;
const TRB_LINK: u32 = 6;
const TRB_ENABLE_SLOT: u32 = 9;
const TRB_ADDRESS_DEVICE: u32 = 11;
const TRB_CONFIGURE_EP: u32 = 12;
const TRB_EVALUATE: u32 = 13;
const EV_TRANSFER: u32 = 32;
const EV_COMMAND: u32 = 33;
const EV_PORT: u32 = 34;

const RING_TRBS: usize = 256;

type Trb = [u32; 4];

fn trb_type(t: &Trb) -> u32 {
    t[3] >> 10 & 0x3f
}

fn code(t: &Trb) -> u32 {
    t[2] >> 24
}

/// A producer ring (commands, transfers), one page with a link TRB at the end.
struct Ring {
    buf: DmaBuffer,
    idx: usize,
    cycle: u32,
}

impl Ring {
    fn new(space: &mut DmaSpace) -> Ring {
        let buf = space.alloc(RING_TRBS * 16).expect("ring");
        let r = Ring { buf, idx: 0, cycle: 1 };
        let link = (RING_TRBS - 1) * 16;
        r.buf.write32(link, r.buf.iova as u32);
        r.buf.write32(link + 4, (r.buf.iova >> 32) as u32);
        r.buf.write32(link + 12, TRB_LINK << 10 | 1 << 1);
        r
    }

    /// Adds a TRB (the cycle bit is set here) and returns its device address.
    fn push(&mut self, t: Trb) -> u64 {
        let at = self.idx * 16;
        self.buf.write32(at, t[0]);
        self.buf.write32(at + 4, t[1]);
        self.buf.write32(at + 8, t[2]);
        compiler_fence(Ordering::Release);
        self.buf.write32(at + 12, t[3] & !1 | self.cycle);
        let addr = self.buf.iova + at as u64;
        self.idx += 1;
        if self.idx == RING_TRBS - 1 {
            let link = (RING_TRBS - 1) * 16;
            self.buf.write32(link + 12, TRB_LINK << 10 | 1 << 1 | self.cycle);
            self.cycle ^= 1;
            self.idx = 0;
        }
        addr
    }
}

struct Endpoint {
    slot: u8,
    dci: u8,
    ring: Ring,
    buf: DmaBuffer,
    len: u32,
    kind: Hid,
    last: [u8; 8],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hid {
    Keyboard,
    Mouse,
}

struct Xhci {
    m: Mmio,
    op: usize,
    rt: usize,
    db: usize,
    ctx_size: usize,
    space: DmaSpace,
    cmd: Ring,
    events: DmaBuffer,
    ev_idx: usize,
    ev_cycle: u32,
    dcbaa: DmaBuffer,
    irq: IrqEvent,
    pending: VecDeque<Trb>,
    ep0: Vec<Option<Ring>>,
    endpoints: Vec<Endpoint>,
    seen: (bool, bool, bool, bool),
}

impl Xhci {
    fn wait(&self, what: &str, mut done: impl FnMut() -> bool) {
        let end = sys::after_ms(1000);
        while !done() {
            if sys::clock() > end {
                panic!("timed out waiting for {what}");
            }
            sys::yield_now();
        }
    }

    fn opr(&self, r: usize) -> u32 {
        self.m.read32(self.op + r)
    }
    fn opw(&self, r: usize, v: u32) {
        self.m.write32(self.op + r, v)
    }
    fn opw64(&self, r: usize, v: u64) {
        self.m.write32(self.op + r, v as u32);
        self.m.write32(self.op + r + 4, (v >> 32) as u32);
    }
    fn portsc(&self, port: usize) -> u32 {
        self.opr(PORTSC + 0x10 * (port - 1))
    }
    fn set_portsc(&self, port: usize, v: u32) {
        self.opw(PORTSC + 0x10 * (port - 1), v)
    }
    fn doorbell(&self, slot: u8, target: u8) {
        self.m.write32(self.db + slot as usize * 4, target as u32);
    }

    /// The next event, if the controller has posted one.
    fn poll_event(&mut self) -> Option<Trb> {
        let at = self.ev_idx * 16;
        let c = self.events.read32(at + 12);
        if c & 1 != self.ev_cycle {
            return None;
        }
        compiler_fence(Ordering::Acquire);
        let t = [self.events.read32(at), self.events.read32(at + 4), self.events.read32(at + 8), c];
        self.ev_idx += 1;
        if self.ev_idx == RING_TRBS {
            self.ev_idx = 0;
            self.ev_cycle ^= 1;
        }
        // Dequeue pointer, clearing Event Handler Busy.
        let erdp = self.events.iova + (self.ev_idx * 16) as u64 | 1 << 3;
        self.m.write32(self.rt + 0x20 + 0x18, erdp as u32);
        self.m.write32(self.rt + 0x20 + 0x1c, (erdp >> 32) as u32);
        Some(t)
    }

    /// Waits for an event matching `want`, keeping the others for later.
    fn wait_event(&mut self, want: impl Fn(&Trb) -> bool) -> Trb {
        let end = sys::after_ms(2000);
        loop {
            while let Some(t) = self.poll_event() {
                if want(&t) {
                    return t;
                }
                self.pending.push_back(t);
            }
            if sys::clock() > end {
                panic!("timed out waiting for an event");
            }
            // Clear the interrupter's pending bit, then wait for the next MSI (or poll every 10 ms).
            self.m.write32(self.rt + 0x20, 0b11);
            let _ = self.irq.wait(sys::after_ms(10));
        }
    }

    fn command(&mut self, t: Trb) -> Trb {
        let addr = self.cmd.push(t);
        self.doorbell(0, 0);
        let ev = self.wait_event(|e| trb_type(e) == EV_COMMAND && (e[0] as u64 | (e[1] as u64) << 32) == addr);
        if code(&ev) != 1 {
            panic!("command type {} failed with code {}", trb_type(&t), code(&ev));
        }
        ev
    }

    fn init(pci: &Pci, bdf: u16) -> Xhci {
        let dev = pci.open(bdf).expect("open");
        dev.enable().expect("enable");
        let m = dev.map_bar(0).expect("bar 0");
        let caplen = m.read8(0) as usize;
        let hcs1 = m.read32(0x04);
        let hcs2 = m.read32(0x08);
        let hcc1 = m.read32(0x10);
        let (db, rt) = ((m.read32(0x14) & !3) as usize, (m.read32(0x18) & !0x1f) as usize);
        let max_slots = (hcs1 & 0xff) as usize;
        let ports = (hcs1 >> 24) as usize;
        let scratch = ((hcs2 >> 21 & 0x1f) << 5 | hcs2 >> 27) as usize;
        println!(
            "xhci {:04x}:{:04x} version {:#x}: {} slots, {} ports, {} scratchpad pages, {}-byte contexts",
            dev.info.vendor,
            dev.info.device,
            m.read32(0) >> 16,
            max_slots,
            ports,
            scratch,
            if hcc1 & 4 != 0 { 64 } else { 32 }
        );

        // Take the controller from the firmware (USB legacy support capability).
        let mut x = (hcc1 >> 16) as usize * 4;
        while x != 0 {
            let cap = m.read32(x);
            if cap & 0xff == 1 {
                m.write32(x, cap | 1 << 24);
                let end = sys::after_ms(1000);
                while m.read32(x) & 1 << 16 != 0 && sys::clock() < end {
                    sys::yield_now();
                }
                if m.read32(x) & 1 << 16 != 0 {
                    println!("firmware kept the controller; taking it anyway");
                }
                // SMIs off.
                m.write32(x + 4, m.read32(x + 4) & 0x000e_1fee);
            }
            let next = (cap >> 8 & 0xff) as usize * 4;
            x = if next == 0 { 0 } else { x + next };
        }

        let mut space = dev.dma_space().expect("dma space");
        let n = sys::notify_create().unwrap();
        let irq = dev.enable_msi(n, 1).expect("msi");
        let cmd = Ring::new(&mut space);
        let events = space.alloc(RING_TRBS * 16).unwrap();
        let erst = space.alloc(4096).unwrap();
        let dcbaa = space.alloc(4096).unwrap();
        let mut xh = Xhci {
            m,
            op: caplen,
            rt,
            db,
            ctx_size: if hcc1 & 4 != 0 { 64 } else { 32 },
            space,
            cmd,
            events,
            ev_idx: 0,
            ev_cycle: 1,
            dcbaa,
            irq,
            pending: VecDeque::new(),
            ep0: (0..=max_slots).map(|_| None).collect(),
            endpoints: Vec::new(),
            seen: (false, false, false, false),
        };

        // Stop, reset, wait until ready.
        xh.opw(USBCMD, xh.opr(USBCMD) & !CMD_RUN);
        xh.wait("halt", || xh.opr(USBSTS) & STS_HALTED != 0);
        xh.opw(USBCMD, CMD_RESET);
        xh.wait("reset", || xh.opr(USBCMD) & CMD_RESET == 0 && xh.opr(USBSTS) & STS_NOT_READY == 0);

        xh.opw(CONFIG, max_slots as u32);
        if scratch > 0 {
            let array = xh.space.alloc(4096).unwrap();
            for i in 0..scratch {
                let page = xh.space.alloc(4096).unwrap();
                array.write32(i * 8, page.iova as u32);
                array.write32(i * 8 + 4, (page.iova >> 32) as u32);
                core::mem::forget(page);
            }
            xh.dcbaa.write32(0, array.iova as u32);
            xh.dcbaa.write32(4, (array.iova >> 32) as u32);
            core::mem::forget(array);
        }
        xh.opw64(DCBAAP, xh.dcbaa.iova);
        xh.opw64(CRCR, xh.cmd.buf.iova | 1);

        // One event ring segment for interrupter 0.
        erst.write32(0, xh.events.iova as u32);
        erst.write32(4, (xh.events.iova >> 32) as u32);
        erst.write32(8, RING_TRBS as u32);
        let ir = xh.rt + 0x20;
        xh.m.write32(ir + 0x08, 1);
        xh.m.write32(ir + 0x18, xh.events.iova as u32);
        xh.m.write32(ir + 0x1c, (xh.events.iova >> 32) as u32);
        xh.m.write32(ir + 0x10, erst.iova as u32);
        xh.m.write32(ir + 0x14, (erst.iova >> 32) as u32);
        core::mem::forget(erst);
        // At most one interrupt per 250 us; interrupter on.
        xh.m.write32(ir + 0x04, 1000);
        xh.m.write32(ir, 0b11);
        xh.opw(USBCMD, CMD_RUN | CMD_INTE);
        xh.wait("run", || xh.opr(USBSTS) & STS_HALTED == 0);
        core::mem::forget(dev);
        xh
    }

    fn input_context(&mut self) -> DmaBuffer {
        self.space.alloc(4096).unwrap()
    }

    /// Dword `dw` of context `index` (0: input control, 1: slot, 2..: endpoint DCI-1) in an input context.
    fn ctx_w(&self, b: &DmaBuffer, index: usize, dw: usize, v: u32) {
        b.write32(index * self.ctx_size + dw * 4, v);
    }

    fn control(&mut self, slot: u8, setup: [u8; 8], data: Option<&DmaBuffer>) -> bool {
        let len = u16::from_le_bytes([setup[6], setup[7]]) as u32;
        let dir_in = setup[0] & 0x80 != 0;
        let trt = if len == 0 { 0 } else if dir_in { 3 } else { 2 };
        let ring = self.ep0[slot as usize].as_mut().unwrap();
        ring.push([u32::from_le_bytes(setup[..4].try_into().unwrap()), u32::from_le_bytes(setup[4..].try_into().unwrap()), 8, TRB_SETUP << 10 | 1 << 6 | trt << 16]);
        if let Some(b) = data.filter(|_| len > 0) {
            ring.push([b.iova as u32, (b.iova >> 32) as u32, len, TRB_DATA << 10 | (dir_in as u32) << 16]);
        }
        let status_in = len == 0 || !dir_in;
        let addr = ring.push([0, 0, 0, TRB_STATUS << 10 | 1 << 5 | (status_in as u32) << 16]);
        self.doorbell(slot, 1);
        let ev = self.wait_event(|e| trb_type(e) == EV_TRANSFER && (e[0] as u64 | (e[1] as u64) << 32) == addr);
        matches!(code(&ev), 1 | 13)
    }

    fn get_descriptor(&mut self, slot: u8, kind: u8, index: u8, len: u16, buf: &DmaBuffer) -> bool {
        let l = len.to_le_bytes();
        self.control(slot, [0x80, 6, index, kind, 0, 0, l[0], l[1]], Some(buf))
    }

    fn enumerate(&mut self, port: usize) {
        let mut sc = self.portsc(port);
        if sc & PORT_ENABLED == 0 {
            self.set_portsc(port, sc & PORT_KEEP | PORT_RESET);
            let end = sys::after_ms(500);
            while self.portsc(port) & PORT_RESET_CHANGE == 0 && sys::clock() < end {
                sys::yield_now();
            }
            sc = self.portsc(port);
            self.set_portsc(port, sc & PORT_KEEP | PORT_CHANGES);
            sc = self.portsc(port);
            if sc & PORT_ENABLED == 0 {
                println!("port {port}: not enabled after reset (portsc {sc:#x})");
                return;
            }
        }
        let speed = sc >> 10 & 0xf;
        let ev = self.command([0, 0, 0, TRB_ENABLE_SLOT << 10]);
        let slot = (ev[3] >> 24) as u8;

        // Address the device: slot context and the default control endpoint.
        let mps0: u32 = match speed {
            1 | 2 => 8,
            3 => 64,
            _ => 512,
        };
        let out = self.space.alloc(4096).unwrap();
        self.dcbaa.write32(slot as usize * 8, out.iova as u32);
        self.dcbaa.write32(slot as usize * 8 + 4, (out.iova >> 32) as u32);
        core::mem::forget(out);
        let ring = Ring::new(&mut self.space);
        let input = self.input_context();
        self.ctx_w(&input, 0, 1, 0b11);
        self.ctx_w(&input, 1, 0, speed << 20 | 1 << 27);
        self.ctx_w(&input, 1, 1, (port as u32) << 16);
        self.ctx_w(&input, 2, 1, 3 << 1 | 4 << 3 | mps0 << 16);
        self.ctx_w(&input, 2, 2, ring.buf.iova as u32 | 1);
        self.ctx_w(&input, 2, 3, (ring.buf.iova >> 32) as u32);
        self.ctx_w(&input, 2, 4, 8);
        self.ep0[slot as usize] = Some(ring);
        self.command([input.iova as u32, (input.iova >> 32) as u32, 0, TRB_ADDRESS_DEVICE << 10 | (slot as u32) << 24]);

        let buf = self.space.alloc(4096).unwrap();
        if !self.get_descriptor(slot, 1, 0, 8, &buf) {
            println!("port {port}: no device descriptor");
            return;
        }
        let mps = buf.slice()[7] as u32;
        let mps = if speed >= 4 { 1 << mps } else { mps };
        if mps != mps0 && mps != 0 {
            self.ctx_w(&input, 0, 1, 0b10);
            self.ctx_w(&input, 2, 1, 3 << 1 | 4 << 3 | mps << 16);
            self.command([input.iova as u32, (input.iova >> 32) as u32, 0, TRB_EVALUATE << 10 | (slot as u32) << 24]);
        }
        self.get_descriptor(slot, 1, 0, 18, &buf);
        let d = buf.slice()[..18].to_vec();
        let (vendor, product) = (u16::from_le_bytes([d[8], d[9]]), u16::from_le_bytes([d[10], d[11]]));
        if !self.get_descriptor(slot, 2, 0, 9, &buf) {
            return;
        }
        let total = u16::from_le_bytes([buf.slice()[2], buf.slice()[3]]).min(4096);
        self.get_descriptor(slot, 2, 0, total, &buf);
        let cfg = buf.slice()[..total as usize].to_vec();
        let config_value = cfg[5];

        // HID boot interfaces and their interrupt IN endpoints.
        let mut found: Vec<(u8, Hid, u8, u16, u8)> = Vec::new();
        let mut i = 0;
        let mut current: Option<(u8, Hid)> = None;
        while i + 2 <= cfg.len() && cfg[i] >= 2 {
            let (len, kind) = (cfg[i] as usize, cfg[i + 1]);
            if kind == 4 && i + 9 <= cfg.len() {
                current = match (cfg[i + 5], cfg[i + 6], cfg[i + 7]) {
                    (3, 1, 1) => Some((cfg[i + 2], Hid::Keyboard)),
                    (3, 1, 2) => Some((cfg[i + 2], Hid::Mouse)),
                    _ => None,
                };
            }
            if kind == 5 && i + 7 <= cfg.len() {
                if let Some((iface, hid)) = current.take() {
                    let (addr, attr) = (cfg[i + 2], cfg[i + 3]);
                    if addr & 0x80 != 0 && attr & 3 == 3 {
                        found.push((iface, hid, addr & 0xf, u16::from_le_bytes([cfg[i + 4], cfg[i + 5]]) & 0x7ff, cfg[i + 6]));
                    }
                }
            }
            i += len;
        }
        println!("port {port}: slot {slot}, {:04x}:{:04x}, speed {speed}, {} HID boot interface(s)", vendor, product, found.len());
        if found.is_empty() {
            return;
        }
        self.control(slot, [0, 9, config_value, 0, 0, 0, 0, 0], None);

        let input = self.input_context();
        let mut add = 1u32;
        let mut max_dci = 1;
        for &(iface, hid, ep, mps, interval) in &found {
            // Boot protocol, no idle repeat.
            self.control(slot, [0x21, 0x0b, 0, 0, iface, 0, 0, 0], None);
            self.control(slot, [0x21, 0x0a, 0, 0, iface, 0, 0, 0], None);
            let dci = ep * 2 + 1;
            add |= 1 << dci;
            max_dci = max_dci.max(dci);
            let ivl: u32 = if speed >= 3 {
                (interval.clamp(1, 16) - 1) as u32
            } else {
                (31 - ((interval.max(1) as u32) * 8).leading_zeros()).clamp(3, 10)
            };
            let ring = Ring::new(&mut self.space);
            let c = dci as usize + 1;
            self.ctx_w(&input, c, 0, ivl << 16);
            self.ctx_w(&input, c, 1, 3 << 1 | 7 << 3 | (mps as u32) << 16);
            self.ctx_w(&input, c, 2, ring.buf.iova as u32 | 1);
            self.ctx_w(&input, c, 3, (ring.buf.iova >> 32) as u32);
            self.ctx_w(&input, c, 4, (mps as u32) << 16 | mps as u32);
            let buf = self.space.alloc(64).unwrap();
            self.endpoints.push(Endpoint { slot, dci, ring, buf, len: mps.min(64) as u32, kind: hid, last: [0; 8] });
        }
        self.ctx_w(&input, 0, 1, add);
        self.ctx_w(&input, 1, 0, speed << 20 | (max_dci as u32) << 27);
        self.ctx_w(&input, 1, 1, (port as u32) << 16);
        self.command([input.iova as u32, (input.iova >> 32) as u32, 0, TRB_CONFIGURE_EP << 10 | (slot as u32) << 24]);
        for e in self.endpoints.iter_mut().filter(|e| e.slot == slot) {
            queue(e);
        }
        for e in self.endpoints.iter().filter(|e| e.slot == slot) {
            self.doorbell(e.slot, e.dci);
            println!("port {port}: {} ready", if e.kind == Hid::Keyboard { "keyboard" } else { "mouse" });
        }
    }

    fn report(&mut self, ev: &Trb) {
        let (slot, dci) = ((ev[3] >> 24) as u8, (ev[3] >> 16 & 0x1f) as u8);
        let Some(i) = self.endpoints.iter().position(|e| e.slot == slot && e.dci == dci) else { return };
        let e = &mut self.endpoints[i];
        if matches!(code(ev), 1 | 13) {
            let got = e.len - (ev[2] & 0xff_ffff);
            let mut r = [0u8; 8];
            let n = (got as usize).min(8);
            r[..n].copy_from_slice(&e.buf.slice()[..n]);
            match e.kind {
                Hid::Keyboard => {
                    for &k in &r[2..8] {
                        if k != 0 && !e.last[2..8].contains(&k) {
                            println!("key {}", key_name(k, r[0]));
                            if k == 0x04 {
                                self.seen.0 = true;
                            }
                            if k == 0x05 {
                                self.seen.1 = true;
                            }
                        }
                    }
                }
                Hid::Mouse => {
                    let (b, dx, dy) = (r[0], r[1] as i8, r[2] as i8);
                    println!("mouse dx {dx} dy {dy} buttons {b:#x}");
                    if dx != 0 || dy != 0 {
                        self.seen.2 = true;
                    }
                    if b & 1 != 0 {
                        self.seen.3 = true;
                    }
                }
            }
            e.last = r;
        } else {
            println!("transfer on slot {slot} ep {dci} ended with code {}", code(ev));
        }
        let e = &mut self.endpoints[i];
        queue(e);
        let (s, d) = (e.slot, e.dci);
        self.doorbell(s, d);
    }
}

fn queue(e: &mut Endpoint) {
    e.ring.push([e.buf.iova as u32, (e.buf.iova >> 32) as u32, e.len, TRB_NORMAL << 10 | 1 << 5 | 1 << 2]);
}

fn key_name(k: u8, mods: u8) -> String {
    let shift = mods & 0x22 != 0;
    let c = match k {
        0x04..=0x1d => (b'a' + k - 0x04) as char,
        0x1e..=0x26 => (b'1' + k - 0x1e) as char,
        0x27 => '0',
        0x28 => return String::from("enter"),
        0x29 => return String::from("escape"),
        0x2a => return String::from("backspace"),
        0x2b => return String::from("tab"),
        0x2c => return String::from("space"),
        _ => return alloc::format!("{k:#04x}"),
    };
    let mut s = String::new();
    s.push(if shift { c.to_ascii_uppercase() } else { c });
    s
}

fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("bootstrap");
    let pci = Pci::new(b.handle("pci").expect("pci"));
    let report = b.handle("report");
    let Ok(bdf) = pci.find_class(0x0c0330) else {
        println!("no xHCI controller");
        return 0;
    };
    let mut x = Xhci::init(&pci, bdf);
    let ports = (x.m.read32(0x04) >> 24) as usize;
    // Give devices time to connect after the reset.
    let _ = x.irq.wait(sys::after_ms(50));
    for port in 1..=ports {
        if x.portsc(port) & PORT_CONNECTED != 0 {
            x.enumerate(port);
        }
    }
    println!("input ready");
    let mut reported = false;
    loop {
        while let Some(t) = x.pending.pop_front().or_else(|| x.poll_event()) {
            match trb_type(&t) {
                EV_TRANSFER => x.report(&t),
                EV_PORT => {
                    let port = (t[0] >> 24) as usize;
                    let sc = x.portsc(port);
                    x.set_portsc(port, sc & PORT_KEEP | PORT_CHANGES);
                    if sc & PORT_CONNECTED != 0 && sc & PORT_ENABLED == 0 {
                        x.enumerate(port);
                    }
                }
                _ => {}
            }
        }
        if let (Some(r), false, (true, true, true, true)) = (report, reported, x.seen) {
            let _ = sys::channel_write(r, b"keyboard and mouse ok", &[]);
            reported = true;
        }
        x.m.write32(x.rt + 0x20, 0b11);
        let _ = x.irq.wait(sys::after_ms(100));
    }
}
