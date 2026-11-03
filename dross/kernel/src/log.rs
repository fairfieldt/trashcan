//! The kernel log: COM1 at 115200 8N1 (QEMU's `-serial`) and the firmware framebuffer. A Mac Pro has no UART
//! behind 0x3f8; the writes go nowhere and the line-status read returns all ones, so nothing waits.

use core::fmt::{self, Write};
use spin::Mutex;
use x86_64::instructions::port::Port;

const COM1: u16 = 0x3f8;

struct Serial {
    fb: Option<crate::fbcon::Console>,
}

impl Serial {
    fn put(&mut self, b: u8) {
        // SAFETY: COM1's registers; bounded wait for the transmit holding register.
        unsafe {
            let mut lsr = Port::<u8>::new(COM1 + 5);
            for _ in 0..100_000 {
                if lsr.read() & 0x20 != 0 {
                    break;
                }
            }
            Port::<u8>::new(COM1).write(b);
        }
        if let Some(fb) = &mut self.fb {
            if !crate::fbcon::claimed() || EMERGENCY.load(core::sync::atomic::Ordering::Relaxed) {
                fb.put(b as char);
            }
        }
    }
}

impl Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            if b == b'\n' {
                self.put(b'\r');
            }
            self.put(b);
        }
        Ok(())
    }
}

static SERIAL: Mutex<Serial> = Mutex::new(Serial { fb: None });

pub fn init(fb: &bootinfo::Framebuffer) {
    SERIAL.lock().fb = crate::fbcon::Console::new(fb);
    // SAFETY: programming COM1: no interrupts, divisor 1 (115200), 8N1, FIFOs on.
    unsafe {
        for (reg, v) in [(1, 0x00u8), (3, 0x80), (0, 0x01), (1, 0x00), (3, 0x03), (2, 0xc7), (4, 0x03)] {
            Port::<u8>::new(COM1 + reg).write(v);
        }
    }
}

fn line(w: &mut Serial, args: fmt::Arguments) {
    let us = crate::time::now_ns() / 1000;
    let _ = write!(w, "[{:5}.{:06}] {}\n", us / 1_000_000, us % 1_000_000, args);
    if let Some(mut r) = RING.try_lock() {
        r.record(us, args);
    }
}

/// The last MiB of the log as records in Linux's /dev/kmsg format ("6,SEQ,USEC,-;MESSAGE\n"), for programs that
/// read /dev/kmsg (ethd's network log sends it to the LAN).
const RING_BYTES: usize = 1 << 20;

pub struct Ring {
    buf: [u8; RING_BYTES],
    /// Bytes ever written.
    written: u64,
    seq: u64,
}

static RING: Mutex<Ring> = Mutex::new(Ring { buf: [0; RING_BYTES], written: 0, seq: 0 });

impl Ring {
    fn put(&mut self, b: u8) {
        self.buf[(self.written % RING_BYTES as u64) as usize] = b;
        self.written += 1;
    }

    fn record(&mut self, us: u64, args: fmt::Arguments) {
        struct W<'a>(&'a mut Ring);
        impl Write for W<'_> {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                // One record is one line.
                for b in s.bytes() {
                    self.0.put(if b == b'\n' { b' ' } else { b });
                }
                Ok(())
            }
        }
        let seq = self.seq;
        self.seq += 1;
        let _ = write!(W(self), "6,{seq},{us},-;{args}");
        self.put(b'\n');
    }
}

/// The record at `cursor` (a byte position, 0 for the oldest kept), with the cursor moved past it.
pub fn read_record(cursor: &mut u64) -> Option<alloc::vec::Vec<u8>> {
    let r = RING.lock();
    let oldest = r.written.saturating_sub(RING_BYTES as u64);
    if *cursor < oldest {
        // Overwritten: resume at the first whole record still there.
        let mut at = oldest;
        while at < r.written && r.buf[(at % RING_BYTES as u64) as usize] != b'\n' {
            at += 1;
        }
        *cursor = at + 1;
    }
    let mut out = alloc::vec::Vec::new();
    let mut at = *cursor;
    while at < r.written {
        let b = r.buf[(at % RING_BYTES as u64) as usize];
        out.push(b);
        at += 1;
        if b == b'\n' {
            *cursor = at;
            return Some(out);
        }
    }
    None
}

pub fn print(args: fmt::Arguments) {
    // Interrupts off while holding the lock, so a handler on this CPU that logs can't deadlock on it.
    x86_64::instructions::interrupts::without_interrupts(|| line(&mut SERIAL.lock(), args));
}

/// For panics and fatal exceptions: does not wait for a lock another CPU may hold forever.
/// Set by a panic, a fatal fault or a reset: the log goes back on the screen even if a program has it.
static EMERGENCY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

pub fn emergency(args: fmt::Arguments) {
    EMERGENCY.store(true, core::sync::atomic::Ordering::Relaxed);
    match SERIAL.try_lock() {
        Some(mut s) => line(&mut s, args),
        None => line(&mut Serial { fb: None }, args),
    }
}

#[macro_export]
macro_rules! println {
    ($($t:tt)*) => { $crate::log::print(format_args!($($t)*)) };
}
