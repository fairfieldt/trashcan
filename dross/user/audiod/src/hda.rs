//! Intel HDA 1.0a controller and analogue output routing. No kernel codec policy.
//! Register/verb definitions: Intel HDA specification 1.0a, sections 3 and 7.
use alloc::vec::Vec;
use core::sync::atomic::{Ordering, compiler_fence};
use dross_rt::pci::{Device, DmaBuffer, DmaSpace, IrqEvent, Mmio, Pci};
use dross_rt::{Error, println, sys};

const PCM_BYTES: usize = (crate::MAX_FRAMES * crate::FRAME_BYTES) as usize;
// One second of silence after the submitted samples gives the service time to stop after BCIS.
// No client memory is ever mapped into this DMA domain.
const SILENCE_BYTES: usize = PCM_BYTES;
const FORMAT: u16 = 0x11; // 48 kHz, 16 bits, 2 channels.

pub struct Hda {
    dev: Device,
    mmio: Mmio,
    dma: DmaSpace,
    rings: DmaBuffer,
    bdl: DmaBuffer,
    samples: DmaBuffer,
    irq: IrqEvent,
    corb_wp: u16,
    rirb_rp: u16,
    stream: usize,
    stream_index: usize,
    codec: u8,
    fg: u8,
    pub vendor: u32,
    pub subsystem: u32,
    widgets: Vec<Widget>,
    routes: Vec<Vec<u8>>,
    selected: Vec<u8>,
    pub completions: u64,
    alive: bool,
    pub stopped: bool,
    staged: usize,
    operation_deadline: u64,
    pub failed: bool,
}
#[derive(Clone)]
struct Widget {
    nid: u8,
    caps: u32,
    pin: u32,
    config: u32,
    connections: Vec<u8>,
}
impl Widget {
    fn kind(&self) -> u32 {
        (self.caps >> 20) & 15
    }
}

// Mmio supplies a bounds-checked mapped BAR; HDA registers also need 16-bit and byte writes.
fn r16(m: &Mmio, off: usize) -> u16 {
    assert!(off + 2 <= m.len());
    unsafe { m.as_ptr().add(off).cast::<u16>().read_volatile() }
}
fn w16(m: &Mmio, off: usize, value: u16) {
    assert!(off + 2 <= m.len());
    unsafe { m.as_ptr().add(off).cast::<u16>().write_volatile(value) }
}
fn address(m: &Mmio, off: usize, value: u64) {
    m.write32(off, value as u32);
    m.write32(off + 4, (value >> 32) as u32);
}
fn w8(m: &Mmio, off: usize, value: u8) {
    assert!(off < m.len());
    unsafe { m.as_ptr().add(off).write_volatile(value) }
}
fn wait(mut condition: impl FnMut() -> bool, deadline: u64) -> sys::Result<()> {
    while !condition() {
        if sys::clock() >= deadline {
            return Err(Error::TimedOut);
        }
        sys::yield_now();
    }
    Ok(())
}
fn delay_us(us: u64) {
    let until = sys::clock().saturating_add(us * 1000);
    while sys::clock() < until {
        sys::yield_now();
    }
}

impl Hda {
    pub fn open(pci: &Pci) -> sys::Result<Self> {
        // Only Intel's onboard controller is supported. Do not claim either D700 HDMI function.
        let info = pci
            .list()?
            .into_iter()
            .find(|d| d.vendor == 0x8086 && d.class == 0x040300)
            .ok_or(Error::NotFound)?;
        let dev = pci.open(info.bdf)?;
        println!(
            "controller {:02x}:{:02x}.{} {:04x}:{:04x}",
            info.bdf >> 8,
            (info.bdf >> 3) & 31,
            info.bdf & 7,
            info.vendor,
            info.device
        );
        // Decode MMIO with bus mastering stopped until a new domain and buffers exist.
        let command = dev.config_read(4, 2)?;
        dev.config_write(4, 2, (command | 2) & !4)?;
        if info.device == 0x1d20 {
            let tcsel = dev.config_read(0x44, 1)?;
            dev.config_write(0x44, 1, tcsel & !7)?;
            // PCH DMA must snoop CPU caches; see PROVENANCE.md.
            let control = dev.config_read(0x78, 2)?;
            dev.config_write(0x78, 2, control & !(1 << 11))?;
            if dev.config_read(0x78, 2)? & (1 << 11) != 0 {
                return Err(Error::NoResources);
            }
        }
        let mmio = dev.map_bar(0)?;
        if mmio.len() < 0x100 {
            return Err(Error::InvalidArgs);
        }
        mmio.write32(0x20, 0);
        w8(&mmio, 0x4c, 0);
        w8(&mmio, 0x5c, 0);
        mmio.write32(0x08, mmio.read32(0x08) & !1);
        wait(|| mmio.read32(0x08) & 1 == 0, sys::after_ms(100))?;
        delay_us(100);
        mmio.write32(0x08, 1);
        wait(|| mmio.read32(0x08) & 1 != 0, sys::after_ms(100))?;
        delay_us(521); // Codec initialization needs at least 521 us after reset release.
        println!("link reset complete");
        let codecs = r16(&mmio, 0x0e) & 0x7fff;
        if codecs == 0 {
            return Err(Error::NotFound);
        }
        let gcap = r16(&mmio, 0);
        let inputs = ((gcap >> 8) & 15) as usize;
        if gcap >> 12 == 0 {
            return Err(Error::WrongType);
        }
        let stream = 0x80 + inputs * 0x20;
        if stream + 0x20 > mmio.len() {
            return Err(Error::InvalidArgs);
        }
        let mut dma = dev.dma_space()?;
        let rings = dma.alloc(4096)?;
        let bdl = dma.alloc_with(4096, dross_abi::dma::READ)?;
        let samples = dma.alloc_with(PCM_BYTES + SILENCE_BYTES, dross_abi::dma::READ)?;
        if gcap & 1 == 0 && samples.iova + samples.len as u64 > 1u64 << 32 {
            return Err(Error::WrongType);
        }
        if mmio.read8(0x4e) & 0x40 == 0 || mmio.read8(0x5e) & 0x40 == 0 {
            return Err(Error::WrongType);
        }
        w8(&mmio, 0x4e, 2);
        w8(&mmio, 0x5e, 2); // 256 entries.
        address(&mmio, 0x40, rings.iova);
        address(&mmio, 0x50, rings.iova + 2048);
        w16(&mmio, 0x4a, 0x8000);
        wait(|| r16(&mmio, 0x4a) & 0x8000 != 0, sys::after_ms(100))?;
        w16(&mmio, 0x4a, 0);
        wait(|| r16(&mmio, 0x4a) & 0x8000 == 0, sys::after_ms(100))?;
        w16(&mmio, 0x48, 0);
        w16(&mmio, 0x58, 0x8000);
        w16(&mmio, 0x5a, 1);
        w8(&mmio, 0x5d, 5);
        w8(&mmio, 0x4d, 1);
        dev.enable()?;
        w8(&mmio, 0x5c, 3); // DMA+response status; INTCTL.CIE stays masked (polled replies).
        w8(&mmio, 0x4c, 2);
        let notify = sys::notify_create()?;
        let irq = dev.enable_msi(notify, 1)?;
        println!("rings and MSI ready");
        let mut h = Self {
            dev,
            mmio,
            dma,
            rings,
            bdl,
            samples,
            irq,
            corb_wp: 0,
            rirb_rp: 0,
            stream,
            stream_index: inputs,
            codec: codecs.trailing_zeros() as u8,
            fg: 0,
            vendor: 0,
            subsystem: 0,
            widgets: Vec::new(),
            routes: Vec::new(),
            selected: Vec::new(),
            completions: 0,
            alive: true,
            stopped: true,
            staged: 0,
            operation_deadline: sys::after_ms(5000),
            failed: false,
        };
        h.discover()?;
        Ok(h)
    }
    fn verb(&mut self, nid: u8, verb: u16, payload: u16) -> sys::Result<u32> {
        if self.failed {
            return Err(Error::NoResources);
        }
        if sys::clock() >= self.operation_deadline {
            return Err(Error::TimedOut);
        }
        let command =
            (self.codec as u32) << 28 | (nid as u32) << 20 | (verb as u32) << 8 | payload as u32;
        let next = (self.corb_wp + 1) & 255;
        self.rings.write32(next as usize * 4, command);
        compiler_fence(Ordering::Release);
        w16(&self.mmio, 0x48, next);
        self.corb_wp = next;
        let deadline = sys::after_ms(100).min(self.operation_deadline);
        loop {
            if self.mmio.read8(0x5d) & 4 != 0 {
                self.failed = true;
                return Err(Error::NoResources);
            }
            let wp = r16(&self.mmio, 0x58) & 255;
            while self.rirb_rp != wp {
                self.rirb_rp = (self.rirb_rp + 1) & 255;
                compiler_fence(Ordering::Acquire);
                let at = 2048 + self.rirb_rp as usize * 8;
                let value = self.rings.read32(at);
                let ext = self.rings.read32(at + 4);
                if ext & 0x10 == 0 && ext & 15 == self.codec as u32 {
                    w8(&self.mmio, 0x5d, 1);
                    return Ok(value);
                }
            }
            if sys::clock() >= deadline {
                self.failed = true;
                println!(
                    "verb timeout cmd={command:#x} corbrp={:#x} corbwp={:#x} rirbwp={:#x} corbsts={:#x}",
                    r16(&self.mmio, 0x4a),
                    r16(&self.mmio, 0x48),
                    r16(&self.mmio, 0x58),
                    self.mmio.read8(0x4d)
                );
                return Err(Error::TimedOut);
            }
            sys::yield_now();
        }
    }
    fn parameter(&mut self, nid: u8, p: u16) -> sys::Result<u32> {
        self.verb(nid, 0xf00, p)
    }
    fn power_on(&mut self, nid: u8) -> sys::Result<()> {
        self.verb(nid, 0x705, 0)?;
        let deadline = sys::after_ms(100).min(self.operation_deadline);
        loop {
            let state = self.verb(nid, 0xf05, 0)?;
            if state & 0x100 != 0 {
                return Err(Error::NoResources);
            }
            if state & 0xf0 == 0 {
                return Ok(());
            }
            if sys::clock() >= deadline {
                return Err(Error::TimedOut);
            }
            sys::yield_now();
        }
    }
    fn discover(&mut self) -> sys::Result<()> {
        self.vendor = self.parameter(0, 0)?;
        let nodes = self.parameter(0, 4)?;
        for n in (nodes >> 16)..(nodes >> 16) + (nodes & 255) {
            if n > 255 {
                return Err(Error::InvalidArgs);
            }
            if self.parameter(n as u8, 5)? & 255 == 1 {
                self.fg = n as u8;
                break;
            }
        }
        if self.fg == 0 {
            return Err(Error::NotFound);
        }
        self.subsystem = self.verb(self.fg, 0xf20, 0)?;
        println!(
            "codec{} vendor={:#010x} subsystem={:#010x} afg={:#x}",
            self.codec, self.vendor, self.subsystem, self.fg
        );
        self.power_on(self.fg)?;
        if self.vendor == 0x10134208 {
            // CS4208 A1 input current-sense setup, documented in PROVENANCE.md.
            self.verb(0x24, 0x703, 1)?;
            self.verb(0x24, 0x500, 0x33)?;
            self.verb(0x24, 0x400, 1)?;
            self.verb(0x24, 0x500, 0x34)?;
            self.verb(0x24, 0x400, 0x1c01)?;
        }
        let nodes = self.parameter(self.fg, 4)?;
        for n in (nodes >> 16)..(nodes >> 16) + (nodes & 255) {
            if n > 255 {
                return Err(Error::InvalidArgs);
            }
            let nid = n as u8;
            let caps = self.parameter(nid, 9)?;
            let kind = (caps >> 20) & 15;
            let (pin, config) = if kind == 4 {
                (self.parameter(nid, 0x0c)?, self.verb(nid, 0xf1c, 0)?)
            } else {
                (0, 0)
            };
            let mut connections = Vec::new();
            if caps & (1 << 8) != 0 {
                let len = self.parameter(nid, 0x0e)?;
                let count = (len & 127) as usize;
                let long = len & 128 != 0;
                let mut raw = Vec::new();
                let per = if long { 2 } else { 4 };
                for i in (0..count).step_by(per) {
                    let v = self.verb(nid, 0xf02, i as u16)?;
                    for j in 0..per.min(count - i) {
                        raw.push(
                            ((v >> (j * if long { 16 } else { 8 }))
                                & if long { 65535 } else { 255 })
                                as u16,
                        );
                    }
                }
                connections = crate::connections(&raw, long).ok_or(Error::InvalidArgs)?;
            }
            self.widgets.push(Widget {
                nid,
                caps,
                pin,
                config,
                connections,
            });
        }
        // Prefer connected analogue speaker/line/headphone pins. Never enable digital output pins.
        let pins: Vec<_> = self
            .widgets
            .iter()
            .filter(|w| {
                w.kind() == 4
                    && w.pin & 16 != 0
                    && w.caps & (1 << 9) == 0
                    && w.config >> 30 != 1
                    && matches!((w.config >> 20) & 15, 0 | 1 | 2)
            })
            .map(|w| w.nid)
            .collect();
        for pin in pins {
            let mut path = Vec::new();
            if self.route(pin, &mut path) {
                println!("route {:?}", path);
                self.routes.push(path);
            }
        }
        if self.routes.is_empty() {
            return Err(Error::WrongType);
        }
        self.select_outputs()?;
        Ok(())
    }
    fn route(&self, nid: u8, path: &mut Vec<u8>) -> bool {
        if path.contains(&nid) || path.len() >= 16 {
            return false;
        }
        let Some(w) = self.widgets.iter().find(|w| w.nid == nid) else {
            return false;
        };
        path.push(nid);
        // Require a stereo analogue DAC advertising48kHz/16bit PCM (validated before programming).
        if w.kind() == 0 && w.caps & 1 != 0 && w.caps & (1 << 9) == 0 {
            return true;
        }
        for &next in &w.connections {
            if self.route(next, path) {
                return true;
            }
        }
        path.pop();
        false
    }
    fn select_outputs(&mut self) -> sys::Result<()> {
        let mac = self.vendor == 0x10134208 && self.subsystem == 0x106b7800;
        let mut headphones = false;
        for path in self.routes.clone() {
            let w = self.widgets.iter().find(|w| w.nid == path[0]).unwrap();
            if (w.config >> 20) & 15 == 2 && w.pin & 4 != 0 {
                let mut present = self.verb(path[0], 0xf09, 0)? & (1 << 31) != 0;
                if mac {
                    present = !present;
                } // Mac Pro6,1 jack sense is inverted.
                headphones |= present;
            }
        }
        self.selected.clear();
        for path in self.routes.clone() {
            let w = self
                .widgets
                .iter()
                .find(|w| w.nid == path[0])
                .unwrap()
                .clone();
            let hp = (w.config >> 20) & 15 == 2;
            let enable = if headphones { hp } else { !hp };
            self.verb(
                w.nid,
                0x707,
                if enable {
                    0x40 | if hp { 0x80 } else { 0 }
                } else {
                    0
                },
            )?;
            if !enable {
                continue;
            }
            for (i, &nid) in path.iter().enumerate() {
                let w = self.widgets.iter().find(|w| w.nid == nid).unwrap().clone();
                if w.caps & (1 << 10) != 0 {
                    self.power_on(nid)?;
                }
                if let Some(&next) = path.get(i + 1) {
                    let index = w.connections.iter().position(|&n| n == next).unwrap();
                    if w.kind() == 4 || w.kind() == 3 {
                        self.verb(nid, 0x701, index as u16)?;
                    }
                    if w.caps & 2 != 0 {
                        let caps = if w.caps & 8 != 0 {
                            self.parameter(nid, 0x0d)?
                        } else {
                            self.parameter(self.fg, 0x0d)?
                        };
                        if w.connections.len() > 16 {
                            return Err(Error::WrongType);
                        }
                        for j in 0..w.connections.len() {
                            self.verb(
                                nid,
                                0x300,
                                0x7000
                                    | (j as u16) << 8
                                    | if j == index {
                                        ((caps & 127).min((caps >> 8) & 127)) as u16
                                    } else {
                                        0x80
                                    },
                            )?;
                        }
                    }
                }
                if w.caps & 4 != 0 {
                    let caps = if w.caps & 8 != 0 {
                        self.parameter(nid, 0x12)?
                    } else {
                        self.parameter(self.fg, 0x12)?
                    };
                    // 0 dB or the nearest lower step. Test client uses low-amplitude samples.
                    self.verb(
                        nid,
                        0x300,
                        0xb000 | ((caps & 127).min((caps >> 8) & 127)) as u16,
                    )?;
                }
                if w.kind() == 0 {
                    let pcm = if w.caps & 16 != 0 {
                        self.parameter(nid, 0x0a)?
                    } else {
                        self.parameter(self.fg, 0x0a)?
                    };
                    let formats = if w.caps & 16 != 0 {
                        self.parameter(nid, 0x0b)?
                    } else {
                        self.parameter(self.fg, 0x0b)?
                    };
                    if pcm & (1 << 6) == 0 || pcm & (1 << 17) == 0 || formats & 1 == 0 {
                        return Err(Error::WrongType);
                    }
                    self.verb(nid, 0x200, FORMAT)?;
                    self.verb(nid, 0x706, 0x10)?; // Stream tag 1, channel 0.
                }
            }
            if w.pin & (1 << 16) != 0 {
                self.verb(w.nid, 0x70c, 2)?;
            }
            self.selected.push(path[0]);
        }
        if mac {
            // GPIO 0 is the speaker amplifier enable on this board; preserve unrelated bits.
            for verb in [0x716, 0x717] {
                let v = self.verb(self.fg, verb + 0x800, 0)?;
                self.verb(self.fg, verb, (v | 1) as u16)?;
            }
            let old = self.verb(self.fg, 0xf15, 0)?;
            self.verb(self.fg, 0x715, ((old & !1) | u32::from(!headphones)) as u16)?;
        }
        if self.selected.is_empty() {
            return Err(Error::NotFound);
        }
        println!("output pins {:?}, PCM 48000 Hz stereo16", self.selected);
        Ok(())
    }
    fn stop_stream(&mut self) -> sys::Result<()> {
        self.stopped = false;
        let s = self.stream;
        w8(&self.mmio, s, self.mmio.read8(s) & !2);
        wait(|| self.mmio.read8(s) & 2 == 0, sys::after_ms(100))?;
        w8(&self.mmio, s + 3, 0x1c);
        self.stopped = true;
        Ok(())
    }
    /// Copy a client snapshot while the stream is stopped. The caller can immediately
    /// unmap and close its input object; playback only refers to private DMA storage.
    ///
    /// # Safety
    /// `input` must remain readable for `length` bytes throughout this call. It may
    /// refer to externally mutable memory. No ordinary Rust reference is created to
    /// that memory; volatile byte reads make no promise of an atomic snapshot.
    pub unsafe fn snapshot(&mut self, input: *const u8, length: usize) -> sys::Result<()> {
        if !self.alive || !self.stopped || length == 0 || length > PCM_BYTES || length % 4 != 0 {
            return Err(Error::InvalidArgs);
        }
        let destination = self.samples.slice();
        destination.fill(0);
        for (index, byte) in destination[..length].iter_mut().enumerate() {
            // SAFETY: the caller retains the readable mapping; each access is in range.
            *byte = unsafe { input.add(index).read_volatile() };
        }
        self.staged = length;
        Ok(())
    }
    pub fn play(&mut self, frames: u32, deadline: u64) -> sys::Result<u32> {
        let length = frames as usize * 4;
        if !self.alive || length == 0 || self.staged != length {
            return Err(Error::InvalidArgs);
        }
        if deadline <= sys::clock() {
            return Err(Error::TimedOut);
        }
        self.operation_deadline = deadline;
        self.stop_stream()?;
        self.select_outputs()?;
        self.bdl.write32(0, self.samples.iova as u32);
        self.bdl.write32(4, (self.samples.iova >> 32) as u32);
        self.bdl.write32(8, length as u32);
        self.bdl.write32(12, 1);
        let silence = self.samples.iova + PCM_BYTES as u64;
        self.bdl.write32(16, silence as u32);
        self.bdl.write32(20, (silence >> 32) as u32);
        let drain = 19_200usize; // 100 ms silence drains the converter and host backend before stopping.
        self.bdl.write32(24, drain as u32);
        self.bdl.write32(28, 1);
        let guard = silence + drain as u64;
        self.bdl.write32(32, guard as u32);
        self.bdl.write32(36, (guard >> 32) as u32);
        self.bdl.write32(40, (SILENCE_BYTES - drain) as u32);
        self.bdl.write32(44, 0);
        let s = self.stream;
        w8(&self.mmio, s, 1);
        wait(|| self.mmio.read8(s) & 1 != 0, sys::after_ms(100))?;
        w8(&self.mmio, s, 0);
        wait(|| self.mmio.read8(s) & 1 == 0, sys::after_ms(100))?;
        self.mmio.write32(s + 8, (length + SILENCE_BYTES) as u32);
        w16(&self.mmio, s + 12, 2);
        w16(&self.mmio, s + 18, FORMAT);
        address(&self.mmio, s + 24, self.bdl.iova);
        w8(&self.mmio, s + 2, 0x10);
        w8(&self.mmio, s + 3, 0x1c);
        let _ = self.irq.wait(sys::clock()); // Drain stale notification bits before RUN.
        self.mmio
            .write32(0x20, 0x8000_0000 | 1u32 << self.stream_index);
        compiler_fence(Ordering::Release);
        if sys::clock() >= deadline {
            return Err(Error::TimedOut);
        }
        self.stopped = false;
        w8(&self.mmio, s, 0x1e); // RUN, IOCE, FEIE, DEIE.
        let started = sys::clock();
        let result = (|| {
            loop {
                let status = self.mmio.read8(s + 3);
                if status & 0x18 != 0 {
                    println!("stream error status={status:#x}");
                    return Err(Error::NoResources);
                }
                let position = self.mmio.read32(s + 4) as usize;
                let now = sys::clock();
                if status & 4 != 0 {
                    self.completions += 1;
                    w8(&self.mmio, s + 3, 4);
                    println!(
                        "stream IOC position={} target={} elapsed={}us",
                        position,
                        length + drain,
                        (now - started) / 1000
                    );
                }
                // HDA 1.0a 3.3.36: output BCIS marks descriptor data fetched
                // into the FIFO, not all data transmitted on the link. The
                // final IOC can precede this LPIB boundary. The silent guard
                // has no IOC; recheck on a bounded timer as well as interrupts.
                // Failure evidence: results/dross/enlight-audio-20261103.
                if now >= deadline {
                    return Err(Error::TimedOut);
                }
                if position > length + SILENCE_BYTES {
                    return Err(Error::NoResources);
                }
                if position >= length + drain {
                    println!(
                        "link drained position={} target={} bcis={} elapsed={}us",
                        position,
                        length + drain,
                        status & 4 != 0,
                        (now - started) / 1000
                    );
                    return Ok((length / 4) as u32);
                }
                let recheck = now.saturating_add(1_000_000).min(deadline);
                match self.irq.wait(recheck) {
                    Ok(_) | Err(Error::TimedOut) => {}
                    Err(error) => return Err(error),
                }
            }
        })();
        let stopped = self.stop_stream();
        println!(
            "playback {:?} elapsed={}us position={} irq_completions={}",
            result,
            (sys::clock() - started) / 1000,
            self.mmio.read32(s + 4),
            self.completions
        );
        // Never report completion while a controller can still read a client's samples.
        stopped?;
        result
    }
    /// Stop every DMA engine and mask MSI before releasing any DMA mapping.
    pub fn shutdown(&mut self) -> sys::Result<()> {
        if !self.alive {
            return Ok(());
        }
        self.operation_deadline = sys::after_ms(1000);
        self.stop_stream()?;
        for pin in self.selected.clone() {
            self.verb(pin, 0x707, 0)?;
        }
        if self.vendor == 0x10134208 && self.subsystem == 0x106b7800 {
            let data = self.verb(self.fg, 0xf15, 0)?;
            self.verb(self.fg, 0x715, (data & !1) as u16)?;
        }
        self.mmio.write32(0x20, 0);
        w8(&self.mmio, 0x4c, 0);
        w8(&self.mmio, 0x5c, 0);
        wait(
            || self.mmio.read8(0x4c) & 2 == 0 && self.mmio.read8(0x5c) & 2 == 0,
            sys::after_ms(100),
        )?;
        self.dev.disable_irqs()?;
        if self.failed {
            return Err(Error::NoResources);
        }
        if sys::clock() >= self.operation_deadline {
            return Err(Error::TimedOut);
        }
        let command = self.dev.config_read(4, 2)?;
        self.dev.config_write(4, 2, command & !4)?;
        if self.dev.config_read(4, 2)? & 4 != 0 {
            return Err(Error::NoResources);
        }
        for b in [&self.samples, &self.bdl, &self.rings] {
            sys::dma_unmap(b.domain, b.iova, b.len as u64)?;
            sys::vm_unmap(b.ptr, b.len as u64)?;
            sys::close(b.mem)?;
        }
        sys::close(self.irq.irq)?;
        sys::close(self.irq.notification)?;
        sys::close(self.dma.domain())?;
        self.alive = false;
        println!("stopped, DMA unmapped, interrupts disabled");
        Ok(())
    }
}
