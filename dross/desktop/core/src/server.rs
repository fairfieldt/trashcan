//! Single-owner compositor and window manager. A connection ID is assigned by the
//! transport, never accepted from a client packet. Surfaces belong to that ID.
use crate::{
    canvas::{Canvas, Rect},
    protocol::{Fault, Input, Key, MAX_DIMENSION, MAX_WINDOW_PIXELS, Message, Mods, Packet},
    ui::Theme,
};
use alloc::{string::String, vec::Vec};

pub type Client = u32;
pub const TITLE_H: u32 = 24;
pub const BORDER: u32 = 3;
pub const MAX_WINDOWS: usize = 16;
pub const TOTAL_PIXELS: usize = 512 * 1024;

pub struct Delivery {
    pub client: Client,
    pub packet: Packet,
}
struct Window {
    id: u32,
    owner: Client,
    title: String,
    x: i32,
    y: i32,
    front: Canvas,
    pending: Canvas,
}
impl Window {
    fn frame(&self) -> Rect {
        Rect::new(
            self.x,
            self.y,
            self.front.width() + BORDER * 2,
            self.front.height() + TITLE_H + BORDER,
        )
    }
    fn content(&self) -> Rect {
        Rect::new(
            self.x + BORDER as i32,
            self.y + TITLE_H as i32,
            self.front.width(),
            self.front.height(),
        )
    }
    fn close(&self) -> Rect {
        Rect::new(self.x + self.frame().w as i32 - 21, self.y + 4, 16, 16)
    }
}

pub struct Server {
    screen: Canvas,
    windows: Vec<Window>,
    next_id: u32,
    focus: Option<u32>,
    pointer: (i32, i32),
    down: bool,
    drag: Option<(u32, i32, i32)>,
    capture: Option<u32>,
    dirty: bool,
}
impl Server {
    pub fn new(width: u32, height: u32) -> Result<Self, Fault> {
        if width < 320 || height < 240 {
            return Err(Fault::Geometry);
        }
        Ok(Self {
            screen: Canvas::new(width, height)?,
            windows: Vec::new(),
            next_id: 1,
            focus: None,
            pointer: (0, 0),
            down: false,
            drag: None,
            capture: None,
            dirty: true,
        })
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.screen.width(), self.screen.height())
    }
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
    pub fn window_count(&self) -> usize {
        self.windows.len()
    }
    pub fn focused(&self) -> Option<u32> {
        self.focus
    }
    fn delivery(w: &Window, serial: u32, message: Message) -> Delivery {
        Delivery {
            client: w.owner,
            packet: Packet::new(serial, w.id, message),
        }
    }
    fn set_focus(&mut self, id: Option<u32>, out: &mut Vec<Delivery>) {
        if self.focus == id {
            return;
        }
        if let Some(w) = self.windows.iter().find(|w| Some(w.id) == self.focus) {
            out.push(Self::delivery(w, 0, Message::Focus(false)));
        }
        self.focus = id;
        if let Some(w) = self.windows.iter().find(|w| Some(w.id) == id) {
            out.push(Self::delivery(w, 0, Message::Focus(true)));
        }
        self.dirty = true;
    }
    pub fn request(&mut self, client: Client, packet: Packet) -> Vec<Delivery> {
        let mut out = Vec::new();
        let serial = packet.serial;
        let id = packet.window;
        if let Err(error) = self.apply(client, packet, &mut out) {
            out.push(Delivery {
                client,
                packet: Packet::new(serial, id, Message::Error(error)),
            });
        }
        out
    }
    fn apply(&mut self, client: Client, p: Packet, out: &mut Vec<Delivery>) -> Result<(), Fault> {
        if let Message::Create {
            width,
            height,
            title,
        } = p.message
        {
            if p.window != 0
                || title.len() > crate::protocol::MAX_TITLE
                || title.chars().any(char::is_control)
            {
                return Err(Fault::Malformed);
            }
            let n = width as u64 * height as u64;
            let allocated: usize = self.windows.iter().map(|w| w.front.pixels().len()).sum();
            if width < 120
                || height < 80
                || width > MAX_DIMENSION
                || height > MAX_DIMENSION
                || width + BORDER * 2 > self.screen.width()
                || height + TITLE_H + BORDER > self.screen.height()
            {
                return Err(Fault::Geometry);
            }
            if self.windows.len() == MAX_WINDOWS
                || n > MAX_WINDOW_PIXELS as u64
                || allocated as u64 + n > TOTAL_PIXELS as u64
                || self.next_id == u32::MAX
            {
                return Err(Fault::Limit);
            }
            let front = Canvas::new(width, height)?;
            let pending = Canvas::new(width, height)?;
            let id = self.next_id;
            self.next_id += 1;
            // Beside the open windows if it fits on the screen, else cascaded over them.
            let right = self.windows.iter().map(|w| w.x + (w.front.width() + BORDER * 2) as i32).max();
            let (x, y) = match right {
                Some(r) if r as u32 + 16 + width + BORDER * 2 <= self.screen.width() => (r + 16, 42),
                _ => {
                    let step = self.windows.len() as u32 * 26;
                    (
                        (78 + step).min(self.screen.width() - width - BORDER * 2) as i32,
                        (42 + step).min(self.screen.height() - height - TITLE_H - BORDER) as i32,
                    )
                }
            };
            self.windows.push(Window {
                id,
                owner: client,
                title,
                x,
                y,
                front,
                pending,
            });
            out.push(Self::delivery(
                self.windows.last().unwrap(),
                p.serial,
                Message::Created { width, height },
            ));
            self.set_focus(Some(id), out);
            self.dirty = true;
            return Ok(());
        }
        let index = self
            .windows
            .iter()
            .position(|w| w.id == p.window && w.owner == client)
            .ok_or(Fault::Ownership)?;
        let w = &mut self.windows[index];
        match p.message {
            Message::Upload { offset, pixels } => {
                let start = offset as usize;
                let end = start.checked_add(pixels.len()).ok_or(Fault::Geometry)?;
                let dest = w
                    .pending
                    .pixels_mut()
                    .get_mut(start..end)
                    .ok_or(Fault::Geometry)?;
                for (d, s) in dest.iter_mut().zip(pixels) {
                    *d = s & 0xffffff;
                }
            }
            Message::Present => {
                w.front.pixels_mut().copy_from_slice(w.pending.pixels());
                out.push(Self::delivery(w, p.serial, Message::FrameDone));
                self.dirty = true;
            }
            Message::Close => {
                out.push(Self::delivery(w, p.serial, Message::Closed));
                self.remove(index, out);
            }
            _ => return Err(Fault::Unknown),
        }
        Ok(())
    }
    fn remove(&mut self, index: usize, out: &mut Vec<Delivery>) {
        let w = self.windows.remove(index);
        if self.capture == Some(w.id) {
            self.capture = None;
        }
        if self.drag.is_some_and(|d| d.0 == w.id) {
            self.drag = None;
        }
        if self.focus == Some(w.id) {
            self.set_focus(self.windows.last().map(|w| w.id), out);
        }
        self.dirty = true;
    }
    pub fn disconnect(&mut self, client: Client) -> Vec<Delivery> {
        let mut out = Vec::new();
        while let Some(i) = self.windows.iter().position(|w| w.owner == client) {
            self.remove(i, &mut out);
        }
        out
    }
    /// Only the trusted seat backend calls this, never an application connection.
    pub fn input(&mut self, input: Input) -> Vec<Delivery> {
        let mut out = Vec::new();
        match input {
            Input::Key {
                key: Key::Tab,
                mods,
            } if mods.has(Mods::ALT) => {
                if self.windows.len() > 1 {
                    let w = self.windows.remove(0);
                    let id = w.id;
                    self.windows.push(w);
                    self.set_focus(Some(id), &mut out);
                }
            }
            Input::Key { .. } => {
                if let Some(w) = self.windows.iter().find(|w| Some(w.id) == self.focus) {
                    out.push(Self::delivery(w, 0, Message::Input(input)));
                }
            }
            Input::Pointer { x, y, down } => {
                let x = x.clamp(0, self.screen.width() as i32 - 1);
                let y = y.clamp(0, self.screen.height() as i32 - 1);
                if self.pointer != (x, y) {
                    self.dirty = true;
                }
                self.pointer = (x, y);
                let pressed = down && !self.down;
                if pressed {
                    if let Some(i) = self.windows.iter().rposition(|w| w.frame().contains(x, y)) {
                        let w = self.windows.remove(i);
                        let id = w.id;
                        self.windows.push(w);
                        self.set_focus(Some(id), &mut out);
                        self.dirty = true;
                        let w = self.windows.last().unwrap();
                        if w.close().contains(x, y) {
                            out.push(Self::delivery(w, 0, Message::CloseRequested));
                        } else if y < w.content().y {
                            self.drag = Some((id, x - w.x, y - w.y));
                        } else {
                            self.capture = Some(id);
                        }
                    } else {
                        self.set_focus(None, &mut out);
                    }
                }
                if down
                    && let Some((id, dx, dy)) = self.drag
                    && let Some(w) = self.windows.iter_mut().find(|w| w.id == id)
                {
                    w.x = (x - dx).clamp(0, self.screen.width() as i32 - w.frame().w as i32);
                    w.y = (y - dy).clamp(0, self.screen.height() as i32 - w.frame().h as i32);
                    self.dirty = true;
                }
                let target = self.capture.or_else(|| {
                    self.windows
                        .iter()
                        .rfind(|w| w.content().contains(x, y))
                        .map(|w| w.id)
                });
                if self.drag.is_none()
                    && let Some(w) = self.windows.iter().find(|w| Some(w.id) == target)
                {
                    out.push(Self::delivery(
                        w,
                        0,
                        Message::Input(Input::Pointer {
                            x: x - w.content().x,
                            y: y - w.content().y,
                            down,
                        }),
                    ));
                }
                if !down {
                    self.drag = None;
                    self.capture = None;
                }
                self.down = down;
            }
        }
        out
    }
    pub fn render(&mut self) -> &Canvas {
        if !self.dirty {
            return &self.screen;
        }
        let c = &mut self.screen;
        c.fill(c.bounds(), Theme::DESKTOP);
        // System 6's restrained stipple, rather than a wallpaper asset.
        for y in (0..c.height() as i32).step_by(4) {
            for x in ((y % 8 / 4 * 2)..c.width() as i32).step_by(4) {
                c.pixel(x, y, 0x555d61);
            }
        }
        for w in &self.windows {
            c.fill(
                Rect::new(w.x + 5, w.y + 5, w.frame().w, w.frame().h),
                0x24282a,
            );
            crate::ui::bevel(c, w.frame(), Theme::FACE, false);
            let active = Some(w.id) == self.focus;
            let title = Rect::new(w.x + 3, w.y + 3, w.frame().w - 6, TITLE_H - 5);
            c.fill(title, if active { Theme::TITLE } else { 0x686a6b });
            // A compact BeOS tab sits within the dominant NeXT charcoal title bar.
            let tab = Rect::new(
                title.x,
                title.y,
                (w.title.chars().count() as u32 * crate::canvas::char_w() as u32 + 24)
                    .min(title.w / 2),
                title.h,
            );
            crate::ui::bevel(
                c,
                tab,
                if active { Theme::TAB } else { Theme::SHADOW },
                false,
            );
            c.text(
                tab.x + 10,
                tab.y + 1,
                &w.title,
                if active { Theme::INK } else { Theme::PAPER },
                tab,
            );
            crate::ui::bevel(c, w.close(), Theme::FACE, false);
            c.text(w.close().x + 4, w.close().y - 1, "x", Theme::INK, w.close());
            c.blit(&w.front, w.content().x, w.content().y);
        }
        // Monochrome pointer; it is drawn by the compositor, never by clients.
        let (x, y) = self.pointer;
        for dy in 0i32..14 {
            for dx in 0..=dy / 2 {
                c.pixel(
                    x + dx,
                    y + dy,
                    if dx == 0 || dx == dy / 2 || dy == 13 {
                        0x101010
                    } else {
                        0xffffff
                    },
                );
            }
        }
        self.dirty = false;
        &self.screen
    }
}
