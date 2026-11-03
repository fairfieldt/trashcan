//! Opaque CPU surfaces. All drawing clips; coordinates are signed, sizes are bounded.
use crate::protocol::Fault;
use alloc::{vec, vec::Vec};
use noto_sans_mono_bitmap::{FontWeight, RasterHeight, get_raster, get_raster_width};

pub const CHAR_H: i32 = 16;
pub fn char_w() -> i32 {
    get_raster_width(FontWeight::Regular, RasterHeight::Size16) as i32
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}
impl Rect {
    pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }
    pub fn contains(self, x: i32, y: i32) -> bool {
        x >= self.x
            && y >= self.y
            && (x as i64) < self.x as i64 + self.w as i64
            && (y as i64) < self.y as i64 + self.h as i64
    }
    pub fn intersect(self, other: Self) -> Self {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let r = (self.x as i64 + self.w as i64).min(other.x as i64 + other.w as i64);
        let b = (self.y as i64 + self.h as i64).min(other.y as i64 + other.h as i64);
        Self::new(
            x,
            y,
            (r - x as i64).max(0) as u32,
            (b - y as i64).max(0) as u32,
        )
    }
}

/// Rendering mechanism used by window policy. Coordinates are display pixels.
/// Implementations must clip every operation to bounds; source pixels are opaque XRGB8888.
pub trait Painter {
    fn bounds(&self) -> Rect;
    fn fill(&mut self, rect: Rect, color: u32);
    fn text(&mut self, x: i32, y: i32, text: &str, color: u32, clip: Rect);
    fn blit(&mut self, source: &Canvas, x: i32, y: i32);
    fn pixel(&mut self, x: i32, y: i32, color: u32) {
        self.fill(Rect::new(x, y, 1, 1), color);
    }
    fn desktop(&mut self, color: u32) {
        self.fill(self.bounds(), color);
        for y in (0..self.bounds().h as i32).step_by(4) {
            for x in ((y % 8 / 4 * 2)..self.bounds().w as i32).step_by(4) {
                self.pixel(x, y, 0x555d61);
            }
        }
    }
}

pub struct Canvas {
    width: u32,
    height: u32,
    pixels: Vec<u32>,
}
impl Canvas {
    pub fn new(width: u32, height: u32) -> Result<Self, Fault> {
        let n = width as u64 * height as u64;
        if width == 0 || height == 0 || width > 4096 || height > 4096 || n > 2 * 1024 * 1024 {
            return Err(Fault::Geometry);
        }
        Ok(Self {
            width,
            height,
            pixels: vec![0; n as usize],
        })
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }
    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }
    pub fn pixels_mut(&mut self) -> &mut [u32] {
        &mut self.pixels
    }
    pub fn fill(&mut self, rect: Rect, color: u32) {
        let r = self.bounds().intersect(rect);
        if r.w == 0 || r.h == 0 {
            return;
        }
        for y in r.y..r.y + r.h as i32 {
            let start = y as usize * self.width as usize + r.x as usize;
            self.pixels[start..start + r.w as usize].fill(color & 0xffffff);
        }
    }
    pub fn pixel(&mut self, x: i32, y: i32, color: u32) {
        if self.bounds().contains(x, y) {
            self.pixels[y as usize * self.width as usize + x as usize] = color & 0xffffff;
        }
    }
    pub fn text(&mut self, x: i32, y: i32, text: &str, color: u32, clip: Rect) {
        let clip = clip.intersect(self.bounds());
        let mut cx = x;
        for ch in text.chars() {
            if cx as i64 >= clip.x as i64 + clip.w as i64 {
                break;
            }
            if let Some(raster) = get_raster(ch, FontWeight::Regular, RasterHeight::Size16)
                .or_else(|| get_raster('?', FontWeight::Regular, RasterHeight::Size16))
            {
                for (dy, row) in raster.rows().enumerate() {
                    for (dx, &alpha) in row.iter().enumerate() {
                        let px = cx.saturating_add(dx as i32);
                        let py = y.saturating_add(dy as i32);
                        if alpha != 0 && clip.contains(px, py) {
                            let idx = py as usize * self.width as usize + px as usize;
                            self.pixels[idx] = blend(self.pixels[idx], color, alpha as u32);
                        }
                    }
                }
            }
            cx = cx.saturating_add(char_w());
        }
    }
    pub fn blit(&mut self, source: &Canvas, x: i32, y: i32) {
        let visible = self
            .bounds()
            .intersect(Rect::new(x, y, source.width, source.height));
        if visible.w == 0 || visible.h == 0 {
            return;
        }
        for dy in visible.y..visible.y + visible.h as i32 {
            let dst = dy as usize * self.width as usize + visible.x as usize;
            let src = (dy - y) as usize * source.width as usize + (visible.x - x) as usize;
            self.pixels[dst..dst + visible.w as usize]
                .copy_from_slice(&source.pixels[src..src + visible.w as usize]);
        }
    }
}
fn blend(bg: u32, fg: u32, a: u32) -> u32 {
    let mut out = 0;
    for shift in [0, 8, 16] {
        out |= ((((bg >> shift) & 255) * (255 - a) + ((fg >> shift) & 255) * a) / 255) << shift;
    }
    out
}

impl Painter for Canvas {
    fn bounds(&self) -> Rect {
        self.bounds()
    }
    fn fill(&mut self, r: Rect, color: u32) {
        self.fill(r, color);
    }
    fn text(&mut self, x: i32, y: i32, text: &str, color: u32, clip: Rect) {
        self.text(x, y, text, color, clip);
    }
    fn blit(&mut self, source: &Canvas, x: i32, y: i32) {
        self.blit(source, x, y);
    }
}
