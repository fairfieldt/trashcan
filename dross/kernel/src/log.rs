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
            fb.put(b as char);
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
}

pub fn print(args: fmt::Arguments) {
    // Interrupts off while holding the lock, so a handler on this CPU that logs can't deadlock on it.
    x86_64::instructions::interrupts::without_interrupts(|| line(&mut SERIAL.lock(), args));
}

/// For panics and fatal exceptions: does not wait for a lock another CPU may hold forever.
pub fn emergency(args: fmt::Arguments) {
    match SERIAL.try_lock() {
        Some(mut s) => line(&mut s, args),
        None => line(&mut Serial { fb: None }, args),
    }
}

#[macro_export]
macro_rules! println {
    ($($t:tt)*) => { $crate::log::print(format_args!($($t)*)) };
}
