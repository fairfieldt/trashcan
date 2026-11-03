//! The log on the firmware's framebuffer (the Mac Pro has no serial port). Text wraps from the bottom back to
//! the top instead of scrolling, so a line costs a few KiB of framebuffer writes and no reads.

use bootinfo::{Framebuffer, PixelFormat};
use core::sync::atomic::{AtomicUsize, Ordering};

/// Mappings of the framebuffer by programs (a display server). While there are any, the log stays off the
/// screen, except for a panic or a reset.
static CLAIMS: AtomicUsize = AtomicUsize::new(0);
// Serialize the last release/clear with a new display owner. Logging only reads
// CLAIMS, so it never takes this lock while holding the serial console lock.
static OWNERSHIP: spin::Mutex<()> = spin::Mutex::new(());

pub struct Claim(());

impl Claim {
    pub fn new() -> Claim {
        let _ownership = OWNERSHIP.lock();
        if CLAIMS.fetch_add(1, Ordering::AcqRel) == 0 {
            crate::println!("display: taken by a program; the log continues off screen");
        }
        Claim(())
    }
}

impl Clone for Claim {
    fn clone(&self) -> Claim {
        let _ownership = OWNERSHIP.lock();
        CLAIMS.fetch_add(1, Ordering::AcqRel);
        Claim(())
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        let _ownership = OWNERSHIP.lock();
        if CLAIMS.load(Ordering::Acquire) == 1 {
            crate::log::resume_framebuffer();
        }
        CLAIMS.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn claimed() -> bool {
    CLAIMS.load(Ordering::Acquire) != 0
}
use noto_sans_mono_bitmap::{get_raster, get_raster_width, FontWeight, RasterHeight};

pub struct Console {
    base: *mut u32,
    width: usize,
    height: usize,
    stride: usize,
    format: PixelFormat,
    size: RasterHeight,
    char_w: usize,
    char_h: usize,
    col: usize,
    row: usize,
}

// SAFETY: the framebuffer is only touched under the log's lock.
unsafe impl Send for Console {}

impl Console {
    pub fn clear(&mut self) {
        self.col = 0;
        self.row = 0;
        for row in 0..=self.rows() { self.clear_row(row); }
    }
    pub fn new(fb: &Framebuffer) -> Option<Self> {
        if fb.base == 0 || fb.width == 0 {
            return None;
        }
        let size = if fb.width >= 2560 { RasterHeight::Size32 } else { RasterHeight::Size16 };
        let c = Console {
            base: crate::mem::virt(fb.base) as *mut u32,
            width: fb.width as usize,
            height: fb.height as usize,
            stride: fb.stride as usize,
            format: fb.format,
            size,
            char_w: get_raster_width(FontWeight::Regular, size),
            char_h: size.val(),
            col: 0,
            row: 0,
        };
        for row in 0..=c.rows() {
            c.clear_row(row);
        }
        Some(c)
    }

    fn rows(&self) -> usize {
        self.height / self.char_h
    }

    fn cols(&self) -> usize {
        self.width / self.char_w
    }

    fn pixel(&self, x: usize, y: usize, level: u8) {
        let l = level as u32;
        // A light grey on black.
        let (r, g, b) = (l * 0xd0 / 255, l * 0xd0 / 255, l * 0xd0 / 255);
        let v = match self.format {
            PixelFormat::Bgr => r << 16 | g << 8 | b,
            PixelFormat::Rgb => b << 16 | g << 8 | r,
        };
        // SAFETY: x < width, y < height, inside the framebuffer.
        unsafe { self.base.add(y * self.stride + x).write_volatile(v) };
    }

    fn clear_row(&self, row: usize) {
        for y in row * self.char_h..((row + 1) * self.char_h).min(self.height) {
            for x in 0..self.width {
                self.pixel(x, y, 0);
            }
        }
    }

    fn newline(&mut self) {
        self.col = 0;
        self.row = (self.row + 1) % self.rows();
        self.clear_row(self.row);
    }

    pub fn put(&mut self, c: char) {
        match c {
            '\n' => return self.newline(),
            '\r' => return,
            _ => {}
        }
        if self.col == self.cols() {
            self.newline();
        }
        let r = get_raster(c, FontWeight::Regular, self.size).or_else(|| get_raster('?', FontWeight::Regular, self.size));
        if let Some(r) = r {
            let (x0, y0) = (self.col * self.char_w, self.row * self.char_h);
            for (dy, line) in r.rows().enumerate() {
                for (dx, &level) in line.iter().enumerate() {
                    self.pixel(x0 + dx, y0 + dy, level);
                }
            }
        }
        self.col += 1;
    }
}
