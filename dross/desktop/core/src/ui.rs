//! Small immediate-mode toolkit with stable widget IDs, pointer capture and keyboard
//! focus. Layout and drawing share rectangles, so hit testing cannot drift from paint.
use crate::{
    canvas::{CHAR_H, Canvas, Painter, Rect, char_w},
    protocol::{Input, Key, Mods},
};

pub struct Theme;
impl Theme {
    pub const FACE: u32 = 0xaaaaaa;
    pub const LIGHT: u32 = 0xf2f2ed;
    pub const SHADOW: u32 = 0x555555;
    pub const INK: u32 = 0x101010;
    pub const PAPER: u32 = 0xeeeeea;
    pub const TITLE: u32 = 0x252525;
    pub const DESKTOP: u32 = 0x5b6469;
    pub const TAB: u32 = 0xf3c54d;
    pub const PANEL: u32 = 0x8f999f;
    pub const TERMINAL: u32 = 0x202526;
    pub const TERMINAL_TEXT: u32 = 0xe2e3d8;
}

pub fn bevel(c: &mut impl Painter, r: Rect, face: u32, pressed: bool) {
    c.fill(r, face);
    let (light, dark) = if pressed {
        (Theme::SHADOW, Theme::LIGHT)
    } else {
        (Theme::LIGHT, Theme::SHADOW)
    };
    if r.w < 2 || r.h < 2 {
        return;
    }
    c.fill(Rect::new(r.x, r.y, r.w, 1), light);
    c.fill(Rect::new(r.x, r.y, 1, r.h), light);
    c.fill(Rect::new(r.x, r.y + r.h as i32 - 1, r.w, 1), dark);
    c.fill(Rect::new(r.x + r.w as i32 - 1, r.y, 1, r.h), dark);
}

/// A row layout, clipped to its parent. No independent widget allocations.
pub struct Row {
    remaining: Rect,
    gap: u32,
}
impl Row {
    pub fn new(area: Rect, gap: u32) -> Self {
        Self {
            remaining: area,
            gap,
        }
    }
    pub fn take(&mut self, width: u32) -> Rect {
        let width = width.min(self.remaining.w);
        let r = Rect::new(self.remaining.x, self.remaining.y, width, self.remaining.h);
        let advance = width.saturating_add(self.gap).min(self.remaining.w);
        self.remaining.x += advance as i32;
        self.remaining.w -= advance;
        r
    }
}

#[derive(Default)]
pub struct Ui {
    focus: Option<u32>,
    active: Option<u32>,
    pointer: (i32, i32),
    down: bool,
    press: bool,
    release: bool,
    activate: bool,
    tab: i8,
    first: Option<u32>,
    previous: Option<u32>,
    next_focus: bool,
    key: Option<Key>,
}
impl Ui {
    /// One event per frame. Call `end` even when there are no widgets.
    pub fn begin(&mut self, input: Option<Input>) {
        self.press = false;
        self.release = false;
        self.activate = false;
        self.tab = 0;
        self.key = None;
        self.first = None;
        self.previous = None;
        self.next_focus = false;
        match input {
            Some(Input::Pointer { x, y, down }) => {
                self.pointer = (x, y);
                self.press = down && !self.down;
                self.release = !down && self.down;
                self.down = down;
            }
            Some(Input::Key {
                key: Key::Tab,
                mods,
            }) => {
                self.tab = if mods.has(Mods::SHIFT) { -1 } else { 1 };
            }
            Some(Input::Key { key, .. }) => {
                self.key = Some(key);
                self.activate = matches!(key, Key::Enter | Key::Character(' '));
            }
            _ => {}
        }
    }
    fn register(&mut self, id: u32) -> bool {
        self.first.get_or_insert(id);
        let had_focus = self.focus == Some(id);
        if self.next_focus {
            self.focus = Some(id);
            self.next_focus = false;
            self.tab = 0;
        }
        if had_focus && self.tab == -1 {
            // Wrap to the last widget in end().
            self.focus = self.previous;
            self.tab = if self.previous.is_some() { 0 } else { -2 };
        } else if had_focus && self.tab == 1 {
            self.next_focus = true;
        }
        self.previous = Some(id);
        had_focus
    }
    pub fn button(&mut self, c: &mut Canvas, id: u32, r: Rect, label: &str) -> bool {
        let had_focus = self.register(id);
        let hovered = r.contains(self.pointer.0, self.pointer.1);
        if self.press && hovered {
            self.active = Some(id);
            self.focus = Some(id);
        }
        let clicked =
            (self.release && self.active == Some(id) && hovered) || (self.activate && had_focus);
        bevel(c, r, Theme::FACE, self.active == Some(id) && self.down);
        let shift = i32::from(self.active == Some(id) && self.down);
        let tx = r.x + ((r.w as i32 - label.chars().count() as i32 * char_w()) / 2).max(2) + shift;
        c.text(
            tx,
            r.y + ((r.h as i32 - CHAR_H) / 2).max(0) + shift,
            label,
            Theme::INK,
            r,
        );
        if self.focus == Some(id) && r.w >= 6 && r.h >= 6 {
            for x in (r.x + 3..r.x + r.w as i32 - 3).step_by(2) {
                c.pixel(x, r.y + 3, Theme::INK);
                c.pixel(x, r.y + r.h as i32 - 4, Theme::INK);
            }
        }
        clicked
    }
    /// Vertical scroll control with pointer capture and keyboard access. The value
    /// increases downwards; the client defines its mapping to document position.
    pub fn scrollbar(
        &mut self,
        c: &mut Canvas,
        id: u32,
        r: Rect,
        max: usize,
        value: usize,
        page: usize,
    ) -> usize {
        let focused = self.register(id);
        let mut value = value.min(max);
        let hover = r.contains(self.pointer.0, self.pointer.1);
        if self.press && hover {
            self.active = Some(id);
            self.focus = Some(id);
        }
        if self.active == Some(id) && self.down && r.h > 1 {
            value = ((self.pointer.1 - r.y).clamp(0, r.h as i32 - 1) as u64 * max as u64
                / (r.h - 1) as u64) as usize;
        }
        if focused {
            value = match self.key {
                Some(Key::Up) => value.saturating_sub(1),
                Some(Key::Down) => value.saturating_add(1).min(max),
                Some(Key::PageUp) => value.saturating_sub(page),
                Some(Key::PageDown) => value.saturating_add(page).min(max),
                Some(Key::Home) => 0,
                Some(Key::End) => max,
                _ => value,
            };
        }
        bevel(c, r, Theme::SHADOW, true);
        let thumb_h = if max == 0 {
            r.h
        } else {
            ((r.h as u64 * page as u64 / max.saturating_add(page).max(1) as u64) as u32)
                .max(16)
                .min(r.h)
        };
        let y = r.y
            + if max == 0 {
                0
            } else {
                (value as u64 * (r.h - thumb_h) as u64 / max as u64) as i32
            };
        bevel(
            c,
            Rect::new(r.x + 2, y, r.w.saturating_sub(4), thumb_h),
            Theme::FACE,
            false,
        );
        value
    }
    pub fn end(&mut self) {
        if self.tab == -2 {
            self.focus = self.previous;
        } else if self.next_focus || (self.tab == 1 && self.focus.is_none()) {
            self.focus = self.first;
        } else if self.tab == -1 && self.focus.is_none() {
            self.focus = self.previous;
        }
        if self.release {
            self.active = None;
        }
    }
    pub fn clear_focus(&mut self) {
        self.focus = None;
        self.active = None;
    }
    pub fn focused(&self) -> Option<u32> {
        self.focus
    }
}
