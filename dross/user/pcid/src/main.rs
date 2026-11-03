//! pcid: owns PCI configuration space (ECAM) and hands PCI functions to drivers.
//!
//! Each client (a channel from init) can open functions it then owns: their config space through filtered
//! reads and writes, their BARs as memory objects, a DMA domain with only the function attached, and MSIs or
//! MSI-X vectors programmed by pcid. (A driver maps the BAR that holds its MSI-X table, so it could rewrite it;
//! without interrupt remapping a device can write any MSI anyway, so this is no further exposure.) A driver can't move its BARs, retarget its MSIs or touch another function. When a client
//! goes away, pcid stops its functions' bus mastering and MSIs and drops their domains, which blocks their DMA.
#![no_std]
#![no_main]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use dross_abi::{rights, vm, Error};
use dross_rt::bootstrap::Bootstrap;
use dross_rt::pci::{self, op, Bar, Info, Request};
use dross_rt::{println, sys, Handle, FOREVER};

dross_rt::main!(main);

struct Function {
    info: Info,
    owner: Option<usize>,
    domain: Option<Handle>,
    irqs: Vec<Handle>,
    /// pcid's own mapping of the MSI-X table.
    msix_table: Option<u64>,
    msi_cap: Option<u16>,
    msix_cap: Option<u16>,
}

struct Pcid {
    ecam: u64,
    first_bus: u8,
    physical: Handle,
    irq: Handle,
    iommu: Handle,
    functions: Vec<Function>,
}

impl Pcid {
    fn cfg(&self, bdf: u16, off: u16) -> u64 {
        self.ecam + ((((bdf >> 8) as u64) - self.first_bus as u64) << 20 | ((bdf & 0xff) as u64) << 12) + off as u64
    }
    fn r32(&self, bdf: u16, off: u16) -> u32 {
        // SAFETY: inside the mapped ECAM window.
        unsafe { (self.cfg(bdf, off & !3) as *const u32).read_volatile() }
    }
    fn w32(&self, bdf: u16, off: u16, v: u32) {
        // SAFETY: as above.
        unsafe { (self.cfg(bdf, off & !3) as *mut u32).write_volatile(v) }
    }
    fn read(&self, bdf: u16, off: u16, size: u8) -> u32 {
        let v = self.r32(bdf, off) >> ((off & 3) * 8);
        match size {
            1 => v & 0xff,
            2 => v & 0xffff,
            _ => v,
        }
    }
    fn write(&self, bdf: u16, off: u16, size: u8, value: u32) {
        let shift = (off & 3) * 8;
        let mask = match size {
            1 => 0xffu32,
            2 => 0xffff,
            _ => u32::MAX,
        } << shift;
        let old = if size == 4 { 0 } else { self.r32(bdf, off) };
        self.w32(bdf, off, old & !mask | (value << shift) & mask);
    }

    fn capability(&self, bdf: u16, id: u8) -> Option<u16> {
        if self.read(bdf, 6, 2) & 0x10 == 0 {
            return None;
        }
        let mut p = self.read(bdf, 0x34, 1) as u16 & !3;
        for _ in 0..48 {
            if p < 0x40 {
                return None;
            }
            if self.read(bdf, p, 1) as u8 == id {
                return Some(p);
            }
            p = self.read(bdf, p + 1, 1) as u16 & !3;
        }
        None
    }

    /// Sizes the BARs of a type-0 function with decoding off.
    fn bars(&self, bdf: u16) -> [Bar; 6] {
        let mut bars = [Bar::default(); 6];
        let cmd = self.read(bdf, 4, 2);
        self.write(bdf, 4, 2, cmd & !3);
        let mut i = 0;
        while i < 6 {
            let off = 0x10 + i as u16 * 4;
            let orig = self.r32(bdf, off);
            self.w32(bdf, off, u32::MAX);
            let probe = self.r32(bdf, off);
            self.w32(bdf, off, orig);
            if probe == 0 {
                i += 1;
                continue;
            }
            if orig & 1 != 0 {
                let size = (!(probe & !3) & 0xffff) + 1;
                bars[i] = Bar { phys: (orig & !3) as u64, size: size as u64, flags: Bar::IO };
                i += 1;
                continue;
            }
            let is64 = orig >> 1 & 3 == 2;
            let mut phys = (orig & !0xf) as u64;
            let mut mask = (probe & !0xf) as u64 | 0xffff_ffff_0000_0000;
            if is64 && i < 5 {
                let hi = self.r32(bdf, off + 4);
                self.w32(bdf, off + 4, u32::MAX);
                let phi = self.r32(bdf, off + 4);
                self.w32(bdf, off + 4, hi);
                phys |= (hi as u64) << 32;
                mask = (phi as u64) << 32 | (probe & !0xf) as u64;
            }
            let size = (!mask).wrapping_add(1);
            let mut flags = if is64 { Bar::MEM64 } else { 0 };
            if orig & 8 != 0 {
                flags |= Bar::PREFETCH;
            }
            bars[i] = Bar { phys, size, flags };
            i += if is64 { 2 } else { 1 };
        }
        self.write(bdf, 4, 2, cmd);
        bars
    }

    fn enumerate(&mut self, last_bus: u8) {
        for bus in self.first_bus..=last_bus {
            for dev in 0..32u16 {
                for f in 0..8u16 {
                    let bdf = (bus as u16) << 8 | dev << 3 | f;
                    let id = self.r32(bdf, 0);
                    if id & 0xffff == 0xffff {
                        if f == 0 {
                            break;
                        }
                        continue;
                    }
                    let header = self.read(bdf, 0xe, 1);
                    let class = self.r32(bdf, 8) >> 8;
                    let bars = if header & 0x7f == 0 { self.bars(bdf) } else { [Bar::default(); 6] };
                    let info = Info { bdf, vendor: id as u16, device: (id >> 16) as u16, class, bars };
                    let (msi_cap, msix_cap) = (self.capability(bdf, 0x05), self.capability(bdf, 0x11));
                    self.functions.push(Function { info, owner: None, domain: None, irqs: Vec::new(), msix_table: None, msi_cap, msix_cap });
                    if f == 0 && header & 0x80 == 0 {
                        break;
                    }
                }
            }
        }
    }

    fn find(&self, bdf: u32) -> Result<usize, Error> {
        self.functions.iter().position(|f| f.info.bdf as u32 == bdf).ok_or(Error::NotFound)
    }

    fn owned(&self, bdf: u32, client: usize) -> Result<usize, Error> {
        let i = self.find(bdf)?;
        if self.functions[i].owner != Some(client) {
            return Err(Error::AccessDenied);
        }
        Ok(i)
    }

    /// Config writes a driver may make: not the IDs, BARs, ROM, or the MSI and MSI-X capabilities.
    fn write_allowed(&self, f: &Function, off: u16, size: u8) -> bool {
        let end = off + size as u16;
        let overlaps = |s: u16, e: u16| off < e && end > s;
        if overlaps(0, 4) || overlaps(0x10, 0x28) || overlaps(0x30, 0x34) || end > 0x1000 {
            return false;
        }
        if let Some(c) = f.msi_cap {
            if overlaps(c, c + 24) {
                return false;
            }
        }
        if let Some(c) = f.msix_cap {
            if overlaps(c, c + 12) {
                return false;
            }
        }
        true
    }

    fn handle(&mut self, client: usize, r: Request, sent: &[Handle]) -> (Vec<u8>, Vec<Handle>) {
        let mut out = Vec::new();
        let res = self.handle_inner(client, r, sent, &mut out);
        match res {
            Ok((v, payload)) => (pci::reply(Ok(v), &payload), out),
            Err(e) => (pci::reply(Err(e), &[]), out),
        }
    }

    fn handle_inner(&mut self, client: usize, r: Request, sent: &[Handle], out: &mut Vec<Handle>) -> Result<(u64, Vec<u8>), Error> {
        match r.op {
            op::LIST => Ok((self.functions.len() as u64, self.functions.iter().flat_map(|f| f.info.encode()).collect())),
            op::FIND => {
                let matches = |f: &&Function| {
                    if r.a >> 63 != 0 {
                        f.info.class == r.a as u32 & 0xff_ffff
                    } else {
                        (f.info.vendor as u64) << 16 | f.info.device as u64 == r.a
                    }
                };
                let f = self.functions.iter().filter(matches).nth(r.b as usize).ok_or(Error::NotFound)?;
                Ok((f.info.bdf as u64, Vec::new()))
            }
            op::OPEN => {
                let i = self.find(r.bdf)?;
                let f = &mut self.functions[i];
                match f.owner {
                    Some(o) if o != client => return Err(Error::Busy),
                    _ => f.owner = Some(client),
                }
                Ok((0, f.info.encode()))
            }
            op::CONFIG_READ => {
                let i = self.owned(r.bdf, client)?;
                if r.a + r.b > 0x1000 || ![1, 2, 4].contains(&r.b) || r.a % r.b != 0 {
                    return Err(Error::InvalidArgs);
                }
                Ok((self.read(self.functions[i].info.bdf, r.a as u16, r.b as u8) as u64, Vec::new()))
            }
            op::CONFIG_WRITE => {
                let i = self.owned(r.bdf, client)?;
                if ![1, 2, 4].contains(&r.b) || r.a % r.b != 0 {
                    return Err(Error::InvalidArgs);
                }
                if !self.write_allowed(&self.functions[i], r.a as u16, r.b as u8) {
                    return Err(Error::AccessDenied);
                }
                self.write(self.functions[i].info.bdf, r.a as u16, r.b as u8, r.c as u32);
                Ok((0, Vec::new()))
            }
            op::MAP_BAR => {
                let i = self.owned(r.bdf, client)?;
                let bar = *self.functions[i].info.bars.get(r.a as usize).ok_or(Error::InvalidArgs)?;
                if bar.size == 0 || bar.flags & Bar::IO != 0 {
                    return Err(Error::InvalidArgs);
                }
                let base = bar.phys & !0xfff;
                let size = (bar.phys + bar.size - base).next_multiple_of(4096);
                out.push(sys::mem_physical(self.physical, base, size)?);
                Ok((0, Vec::new()))
            }
            op::DOMAIN => {
                let i = self.owned(r.bdf, client)?;
                if self.functions[i].domain.is_none() {
                    let d = sys::dma_domain_create(self.iommu)?;
                    sys::dma_attach(d, self.iommu, self.functions[i].info.bdf)?;
                    self.functions[i].domain = Some(d);
                }
                out.push(sys::dup(self.functions[i].domain.unwrap(), rights::ALL)?);
                Ok((0, Vec::new()))
            }
            op::MSI => {
                // a: bits, b: CPU, c: MSI-X vector index.
                let i = self.owned(r.bdf, client)?;
                let notification = *sent.first().ok_or(Error::InvalidArgs)?;
                let bdf = self.functions[i].info.bdf;
                let (irq, addr, data) = sys::irq_msi(self.irq, notification, r.a, r.b)?;
                let _ = sys::close(notification);
                if let Some(cap) = self.functions[i].msi_cap {
                    for old in self.functions[i].irqs.drain(..) {
                        let _ = sys::close(old);
                    }
                    let ctl = self.read(bdf, cap + 2, 2);
                    // Address, upper address if 64-bit, data; one vector; enable.
                    self.write(bdf, cap + 2, 2, ctl & !(1 | 0x70));
                    self.w32(bdf, cap + 4, addr as u32);
                    let data_off = if ctl & 0x80 != 0 {
                        self.w32(bdf, cap + 8, (addr >> 32) as u32);
                        cap + 12
                    } else {
                        cap + 8
                    };
                    self.write(bdf, data_off, 2, data);
                    self.write(bdf, cap + 2, 2, ctl & !0x70 | 1);
                } else if let Some(cap) = self.functions[i].msix_cap {
                    let ctl = self.read(bdf, cap + 2, 2);
                    if r.c > (ctl & 0x7ff) as u64 {
                        let _ = sys::close(irq);
                        return Err(Error::OutOfRange);
                    }
                    let table = match self.functions[i].msix_table {
                        Some(t) => t,
                        None => {
                            let t = self.r32(bdf, cap + 4);
                            let bar = self.functions[i].info.bars[(t & 7) as usize];
                            let at = bar.phys + (t & !7) as u64;
                            let base = at & !0xfff;
                            let size = (at + ((ctl & 0x7ff) as u64 + 1) * 16 - base).next_multiple_of(4096);
                            let m = sys::mem_physical(self.physical, base, size)?;
                            let v = sys::vm_map(m, 0, size, 0, vm::READ | vm::WRITE)? + (at - base);
                            self.functions[i].msix_table = Some(v);
                            v
                        }
                    };
                    // Enabled with every vector masked, then this entry programmed and unmasked.
                    self.write(bdf, cap + 2, 2, ctl | 1 << 15 | 1 << 14);
                    let e = (table + r.c * 16) as *mut u32;
                    // SAFETY: entry r.c of the table pcid mapped above.
                    unsafe {
                        e.add(3).write_volatile(1);
                        e.write_volatile(addr as u32);
                        e.add(1).write_volatile((addr >> 32) as u32);
                        e.add(2).write_volatile(data);
                        e.add(3).write_volatile(0);
                    }
                    self.write(bdf, cap + 2, 2, (ctl | 1 << 15) & !(1 << 14));
                } else {
                    let _ = sys::close(irq);
                    return Err(Error::NotFound);
                }
                let cmd = self.read(bdf, 4, 2);
                self.write(bdf, 4, 2, cmd | 1 << 10);
                out.push(sys::dup(irq, rights::ALL)?);
                self.functions[i].irqs.push(irq);
                Ok((0, Vec::new()))
            }
            op::DISABLE_IRQS => {
                let i = self.owned(r.bdf, client)?;
                self.disable_irqs(i);
                Ok((0, Vec::new()))
            }
            _ => Err(Error::BadSyscall),
        }
    }

    fn disable_irqs(&mut self, i: usize) {
        let bdf = self.functions[i].info.bdf;
        if let Some(c) = self.functions[i].msi_cap {
            let ctl = self.read(bdf, c + 2, 2);
            self.write(bdf, c + 2, 2, ctl & !1);
        }
        if let Some(c) = self.functions[i].msix_cap {
            let ctl = self.read(bdf, c + 2, 2);
            self.write(bdf, c + 2, 2, ctl & !(1 << 15));
        }
        for h in self.functions[i].irqs.drain(..) {
            let _ = sys::close(h);
        }
    }

    /// The client is gone: quiesce its functions.
    fn release(&mut self, client: usize) {
        for i in 0..self.functions.len() {
            if self.functions[i].owner != Some(client) {
                continue;
            }
            let bdf = self.functions[i].info.bdf;
            let cmd = self.read(bdf, 4, 2);
            self.write(bdf, 4, 2, cmd & !4);
            self.disable_irqs(i);
            if let Some(d) = self.functions[i].domain.take() {
                let _ = sys::close(d);
            }
            self.functions[i].owner = None;
            println!("{} released", pci::bdf_string(bdf));
        }
    }
}

fn class_name(class: u32) -> &'static str {
    match class >> 8 {
        0x0106 => "SATA",
        0x0200 => "Ethernet",
        0x0280 => "network",
        0x0300 => "VGA",
        0x0403 => "audio",
        0x0600 => "host bridge",
        0x0601 => "ISA bridge",
        0x0604 => "PCI bridge",
        0x0805 => "SD host",
        0x0880 => "system",
        0x0c03 => "USB",
        0x0c05 => "SMBus",
        0x00ff => "unclassified",
        _ => "",
    }
}

fn main(boot: Handle) -> i32 {
    let b = Bootstrap::read(boot).expect("bootstrap");
    let ecam_res = b.handle("resource/ecam").expect("ecam");
    let base = b.value_u64("ecam").expect("ecam base");
    let buses: Vec<u8> = b.value("ecam-buses").unwrap_or("0 0").split(' ').filter_map(|v| v.parse().ok()).collect();
    let (first, last) = (buses[0], buses[1]);
    let size = ((last - first) as u64 + 1) << 20;
    let mem = sys::mem_physical(ecam_res, base, size).expect("ecam memory");
    let ecam = sys::vm_map(mem, 0, size, 0, vm::READ | vm::WRITE).expect("ecam map");
    let mut p = Pcid {
        ecam,
        first_bus: first,
        physical: b.handle("resource/physical").expect("physical"),
        irq: b.handle("resource/irq").expect("irq"),
        iommu: b.handle("resource/iommu").expect("iommu"),
        functions: Vec::new(),
    };
    let t0 = sys::clock();
    p.enumerate(last);
    println!("{} functions on buses {}-{} in {} us", p.functions.len(), first, last, (sys::clock() - t0) / 1000);
    for f in &p.functions {
        let mut bars = String::new();
        for (i, bar) in f.info.bars.iter().enumerate().filter(|(_, b)| b.size != 0) {
            bars.push_str(&alloc::format!(
                " bar{i} {}{:#x}+{:#x}",
                if bar.flags & Bar::IO != 0 { "io " } else { "" },
                bar.phys,
                bar.size
            ));
        }
        println!(
            "{} {:04x}:{:04x} class {:06x} {}{}{}{}",
            pci::bdf_string(f.info.bdf),
            f.info.vendor,
            f.info.device,
            f.info.class,
            class_name(f.info.class),
            if f.msi_cap.is_some() { " msi" } else { "" },
            if f.msix_cap.is_some() { " msi-x" } else { "" },
            bars
        );
    }

    let clients: Vec<Handle> = b.handles.iter().filter(|(n, _)| n.starts_with("client/")).map(|(_, h)| *h).collect();
    let n = sys::notify_create().unwrap();
    for (i, &c) in clients.iter().enumerate() {
        sys::channel_bind(c, n, 1 << i).unwrap();
    }
    let mut live: u64 = if clients.len() == 64 { u64::MAX } else { (1 << clients.len()) - 1 };
    let mut buf = alloc::vec![0u8; 256];
    let mut hs = [Handle(0); 4];
    while live != 0 {
        let bits = sys::notify_wait(n, FOREVER).unwrap();
        for (i, &c) in clients.iter().enumerate() {
            if bits & live & 1 << i == 0 {
                continue;
            }
            loop {
                match sys::channel_read(c, &mut buf, &mut hs, 0) {
                    Ok((nb, nh)) => {
                        let reply = match Request::decode(&buf[..nb]) {
                            Some(r) => p.handle(i, r, &hs[..nh]),
                            None => (pci::reply(Err(Error::InvalidArgs), &[]), Vec::new()),
                        };
                        let _ = sys::channel_write(c, &reply.0, &reply.1);
                    }
                    Err(Error::PeerClosed) => {
                        p.release(i);
                        live &= !(1 << i);
                        break;
                    }
                    Err(_) => break,
                }
            }
        }
    }
    0
}
