//! Bounded streaming UTF-8/ANSI terminal model, independent of its byte-stream peer.
//! Escape strings never allocate. Unsupported sequences are consumed, not printed.
use crate::{
    canvas::{CHAR_H, Canvas, Rect, char_w},
    protocol::{Fault, Key, Mods},
    ui::Theme,
};
use alloc::{collections::VecDeque, string::String, vec, vec::Vec};

const HISTORY: usize = 256;
const ANSI: [u32; 16] = [
    0x202526, 0xc65d5d, 0x8fba78, 0xdac477, 0x7fa3c5, 0xb794c2, 0x7dbbb9, 0xd9dbd3, 0x717875,
    0xf08080, 0xb8dd92, 0xf3d889, 0xa1c3e6, 0xd9b0e4, 0xa7dfda, 0xffffff,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub ch: char,
    pub fg: u32,
    pub bg: u32,
    pub bold: bool,
}
impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            fg: Theme::TERMINAL_TEXT,
            bg: Theme::TERMINAL,
            bold: false,
        }
    }
}
#[derive(Clone)]
struct Screen {
    cells: Vec<Cell>,
    x: usize,
    y: usize,
    saved: (usize, usize),
    top: usize,
    bottom: usize,
    wrap: bool,
}

pub struct Terminal {
    cols: usize,
    rows: usize,
    screen: Screen,
    alternate: Option<Screen>,
    history: VecDeque<Vec<Cell>>,
    style: Cell,
    inverse: bool,
    cursor: bool,
    state: u8,
    sequence: [u8; 128],
    sequence_len: usize,
    utf: [u8; 4],
    utf_len: usize,
    utf_need: usize,
    view: usize,
}
impl Terminal {
    pub fn new(cols: usize, rows: usize) -> Result<Self, Fault> {
        if cols == 0 || rows == 0 || cols > 256 || rows > 128 {
            return Err(Fault::Geometry);
        }
        Ok(Self {
            cols,
            rows,
            screen: Self::screen(cols, rows),
            alternate: None,
            history: VecDeque::new(),
            style: Cell::default(),
            inverse: false,
            cursor: true,
            state: 0,
            sequence: [0; 128],
            sequence_len: 0,
            utf: [0; 4],
            utf_len: 0,
            utf_need: 0,
            view: 0,
        })
    }
    fn screen(cols: usize, rows: usize) -> Screen {
        Screen {
            cells: vec![Cell::default(); cols * rows],
            x: 0,
            y: 0,
            saved: (0, 0),
            top: 0,
            bottom: rows - 1,
            wrap: false,
        }
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cells(&self) -> &[Cell] {
        &self.screen.cells
    }
    pub fn cursor(&self) -> (usize, usize) {
        (self.screen.x, self.screen.y)
    }
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
    pub fn view_offset(&self) -> usize {
        self.view
    }
    pub fn scroll_view(&mut self, lines: i32) {
        self.view = (self.view as i32 + lines).clamp(0, self.history.len() as i32) as usize;
    }
    pub fn clear(&mut self) {
        self.screen = Self::screen(self.cols, self.rows);
        self.history.clear();
        self.view = 0;
    }
    pub fn feed(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.byte(b);
        }
    }
    fn blank(&self) -> Cell {
        Cell {
            ch: ' ',
            ..self.style
        }
    }
    fn byte(&mut self, b: u8) {
        // OSC/DCS/PM/APC are ignored through BEL or ST, including arbitrarily long strings.
        if self.state == 3 || self.state == 4 {
            if b == 7 || (self.state == 4 && b == b'\\') {
                self.state = 0;
            } else {
                self.state = if b == 27 { 4 } else { 3 };
            }
            return;
        }
        if self.state == 2 || self.state == 5 {
            if b == 27 {
                self.state = 1;
                return;
            }
            if b == 0x18 || b == 0x1a {
                self.state = 0;
                return;
            }
            if (0x40..=0x7e).contains(&b) {
                if self.state == 2 {
                    self.csi(b);
                }
                self.state = 0;
            } else if self.sequence_len < self.sequence.len() && self.state == 2 {
                self.sequence[self.sequence_len] = b;
                self.sequence_len += 1;
            } else {
                self.state = 5;
            }
            return;
        }
        if self.state == 1 {
            self.state = 0;
            match b {
                b'[' => {
                    self.state = 2;
                    self.sequence_len = 0;
                }
                b']' | b'P' | b'^' | b'_' => self.state = 3,
                b'7' => self.screen.saved = (self.screen.x, self.screen.y),
                b'8' => {
                    (self.screen.x, self.screen.y) = self.screen.saved;
                    self.screen.wrap = false;
                }
                b'D' => self.newline(),
                b'E' => {
                    self.screen.x = 0;
                    self.newline();
                }
                b'M' => {
                    if self.screen.y == self.screen.top {
                        self.scroll_down();
                    } else {
                        self.screen.y = self.screen.y.saturating_sub(1);
                    }
                }
                b'c' => {
                    self.clear();
                    self.style = Cell::default();
                    self.inverse = false;
                    self.cursor = true;
                }
                _ => {}
            }
            return;
        }
        if self.utf_len != 0 {
            if b & 0xc0 == 0x80 {
                self.utf[self.utf_len] = b;
                self.utf_len += 1;
                if self.utf_len == self.utf_need {
                    let ch = core::str::from_utf8(&self.utf[..self.utf_len])
                        .ok()
                        .and_then(|s| s.chars().next())
                        .unwrap_or('\u{fffd}');
                    self.utf_len = 0;
                    self.put_char(ch);
                }
                return;
            }
            self.utf_len = 0;
            self.put_char('\u{fffd}');
        }
        match b {
            27 => self.state = 1,
            b'\r' => {
                self.screen.x = 0;
                self.screen.wrap = false;
            }
            b'\n' | 0x0b | 0x0c => self.newline(),
            8 => {
                self.screen.x = self.screen.x.saturating_sub(1);
                self.screen.wrap = false;
            }
            9 => {
                self.screen.x = ((self.screen.x / 8 + 1) * 8).min(self.cols - 1);
                self.screen.wrap = false;
            }
            0..=31 | 127 => {}
            32..=126 => self.put_char(b as char),
            _ => {
                self.utf_need = match b {
                    0xc2..=0xdf => 2,
                    0xe0..=0xef => 3,
                    0xf0..=0xf4 => 4,
                    _ => 0,
                };
                if self.utf_need == 0 {
                    self.put_char('\u{fffd}');
                } else {
                    self.utf[0] = b;
                    self.utf_len = 1;
                }
            }
        }
    }
    fn put_char(&mut self, ch: char) {
        if self.screen.wrap {
            self.screen.x = 0;
            self.newline();
        }
        let mut cell = Cell { ch, ..self.style };
        if self.inverse {
            core::mem::swap(&mut cell.fg, &mut cell.bg);
        }
        self.screen.cells[self.screen.y * self.cols + self.screen.x] = cell;
        if self.screen.x + 1 == self.cols {
            self.screen.wrap = true;
        } else {
            self.screen.x += 1;
        }
    }
    fn newline(&mut self) {
        self.screen.wrap = false;
        if self.screen.y == self.screen.bottom {
            self.scroll_up();
        } else {
            self.screen.y = (self.screen.y + 1).min(self.rows - 1);
        }
    }
    fn scroll_up(&mut self) {
        let start = self.screen.top * self.cols;
        let end = (self.screen.bottom + 1) * self.cols;
        if self.screen.top == 0 && self.screen.bottom + 1 == self.rows && self.alternate.is_none() {
            if self.history.len() == HISTORY {
                self.history.pop_front();
            }
            self.history
                .push_back(self.screen.cells[..self.cols].to_vec());
        }
        self.screen.cells.copy_within(start + self.cols..end, start);
        let blank = self.blank();
        self.screen.cells[end - self.cols..end].fill(blank);
    }
    fn scroll_down(&mut self) {
        let start = self.screen.top * self.cols;
        let end = (self.screen.bottom + 1) * self.cols;
        self.screen
            .cells
            .copy_within(start..end - self.cols, start + self.cols);
        let blank = self.blank();
        self.screen.cells[start..start + self.cols].fill(blank);
    }
    fn csi(&mut self, final_byte: u8) {
        let private = self.sequence_len > 0 && self.sequence[0] == b'?';
        let mut args = [0usize; 16];
        let mut count = 1;
        for &b in &self.sequence[usize::from(private)..self.sequence_len] {
            match b {
                b'0'..=b'9' => {
                    args[count - 1] = (args[count - 1] * 10 + (b - b'0') as usize).min(65535)
                }
                b';' if count < args.len() => count += 1,
                _ => return,
            }
        }
        let n = args[0].max(1);
        let blank = self.blank();
        if private {
            if final_byte == b'h' || final_byte == b'l' {
                let on = final_byte == b'h';
                for &arg in &args[..count] {
                    match arg {
                        25 => self.cursor = on,
                        1049 if on && self.alternate.is_none() => {
                            self.alternate = Some(core::mem::replace(
                                &mut self.screen,
                                Self::screen(self.cols, self.rows),
                            ));
                            self.view = 0;
                        }
                        1049 if !on => {
                            if let Some(screen) = self.alternate.take() {
                                self.screen = screen;
                            }
                        }
                        _ => {}
                    }
                }
            }
            return;
        }
        let pos = self.screen.y * self.cols + self.screen.x;
        match final_byte {
            b'A' => self.screen.y = self.screen.y.saturating_sub(n),
            b'B' => self.screen.y = (self.screen.y + n).min(self.rows - 1),
            b'C' => self.screen.x = (self.screen.x + n).min(self.cols - 1),
            b'D' => self.screen.x = self.screen.x.saturating_sub(n),
            b'E' => {
                self.screen.y = (self.screen.y + n).min(self.rows - 1);
                self.screen.x = 0;
            }
            b'F' => {
                self.screen.y = self.screen.y.saturating_sub(n);
                self.screen.x = 0;
            }
            b'H' | b'f' => {
                self.screen.y = args[0].max(1).min(self.rows) - 1;
                self.screen.x = args[1].max(1).min(self.cols) - 1;
            }
            b'G' => self.screen.x = n.min(self.cols) - 1,
            b'd' => self.screen.y = n.min(self.rows) - 1,
            b'J' => match args[0] {
                0 => self.screen.cells[pos..].fill(blank),
                1 => self.screen.cells[..=pos].fill(blank),
                2 => self.screen.cells.fill(blank),
                3 => {
                    self.history.clear();
                    self.view = 0;
                }
                _ => {}
            },
            b'K' => match args[0] {
                0 => self.screen.cells[pos..(self.screen.y + 1) * self.cols].fill(blank),
                1 => self.screen.cells[self.screen.y * self.cols..=pos].fill(blank),
                2 => self.screen.cells[self.screen.y * self.cols..(self.screen.y + 1) * self.cols]
                    .fill(blank),
                _ => {}
            },
            b'X' => {
                let end = (pos + n).min((self.screen.y + 1) * self.cols);
                self.screen.cells[pos..end].fill(blank);
            }
            b'S' => {
                for _ in 0..n.min(self.rows) {
                    self.scroll_up();
                }
            }
            b'T' => {
                for _ in 0..n.min(self.rows) {
                    self.scroll_down();
                }
            }
            b'm' => self.sgr(&args[..count]),
            b's' => self.screen.saved = (self.screen.x, self.screen.y),
            b'u' => (self.screen.x, self.screen.y) = self.screen.saved,
            b'r' => {
                let top = args[0].max(1).min(self.rows) - 1;
                let bottom = if args[1] == 0 {
                    self.rows - 1
                } else {
                    args[1].min(self.rows) - 1
                };
                if top < bottom {
                    self.screen.top = top;
                    self.screen.bottom = bottom;
                    self.screen.x = 0;
                    self.screen.y = 0;
                }
            }
            _ => {}
        }
        if final_byte != b'm' {
            self.screen.wrap = false;
        }
    }
    fn sgr(&mut self, args: &[usize]) {
        let mut i = 0;
        while i < args.len() {
            match args[i] {
                0 => {
                    self.style = Cell::default();
                    self.inverse = false;
                }
                1 => self.style.bold = true,
                22 => self.style.bold = false,
                7 => self.inverse = true,
                27 => self.inverse = false,
                30..=37 => self.style.fg = ANSI[args[i] - 30],
                40..=47 => self.style.bg = ANSI[args[i] - 40],
                90..=97 => self.style.fg = ANSI[args[i] - 90 + 8],
                100..=107 => self.style.bg = ANSI[args[i] - 100 + 8],
                39 => self.style.fg = Cell::default().fg,
                49 => self.style.bg = Cell::default().bg,
                38 | 48 => {
                    let fg = args[i] == 38;
                    let color = if args.get(i + 1) == Some(&5) && args.len() > i + 2 {
                        i += 2;
                        Some(indexed(args[i].min(255)))
                    } else if args.get(i + 1) == Some(&2) && args.len() > i + 4 {
                        let color = (args[i + 2].min(255) << 16
                            | args[i + 3].min(255) << 8
                            | args[i + 4].min(255)) as u32;
                        i += 4;
                        Some(color)
                    } else {
                        None
                    };
                    if let Some(color) = color {
                        if fg {
                            self.style.fg = color;
                        } else {
                            self.style.bg = color;
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }
    pub fn line(&self, row: usize) -> String {
        if row >= self.rows {
            return String::new();
        }
        self.screen.cells[row * self.cols..(row + 1) * self.cols]
            .iter()
            .map(|c| c.ch)
            .collect()
    }
    pub fn draw(&self, c: &mut Canvas, area: Rect, focused: bool) {
        c.fill(area, Theme::TERMINAL);
        for row in 0..self.rows {
            let logical = self.history.len() + row - self.view;
            let cells = if logical < self.history.len() {
                &self.history[logical][..]
            } else {
                &self.screen.cells[(logical - self.history.len()) * self.cols
                    ..(logical - self.history.len() + 1) * self.cols]
            };
            for (col, cell) in cells.iter().enumerate() {
                let rect = Rect::new(
                    area.x + col as i32 * char_w(),
                    area.y + row as i32 * CHAR_H,
                    char_w() as u32,
                    CHAR_H as u32,
                )
                .intersect(area);
                c.fill(rect, cell.bg);
                let mut bytes = [0; 4];
                c.text(
                    rect.x,
                    rect.y,
                    cell.ch.encode_utf8(&mut bytes),
                    cell.fg,
                    rect,
                );
                if cell.bold {
                    c.text(
                        rect.x + 1,
                        rect.y,
                        cell.ch.encode_utf8(&mut bytes),
                        cell.fg,
                        rect,
                    );
                }
            }
        }
        if focused && self.cursor && self.view == 0 {
            c.fill(
                Rect::new(
                    area.x + self.screen.x as i32 * char_w(),
                    area.y + self.screen.y as i32 * CHAR_H + 14,
                    char_w() as u32,
                    2,
                )
                .intersect(area),
                Theme::TAB,
            );
        }
    }
}

fn indexed(n: usize) -> u32 {
    if n < 16 {
        ANSI[n]
    } else if n >= 232 {
        let c = 8 + (n as u32 - 232) * 10;
        c * 0x010101
    } else {
        let n = n - 16;
        let level = |v: usize| if v == 0 { 0 } else { 55 + v as u32 * 40 };
        level(n / 36) << 16 | level(n / 6 % 6) << 8 | level(n % 6)
    }
}

/// VT-style input encoding, usable with a PTY or a native channel byte stream.
pub fn encode_key(key: Key, mods: Mods) -> Vec<u8> {
    let mut out = Vec::new();
    if mods.has(Mods::ALT) {
        out.push(27);
    }
    match key {
        Key::Character(c) if mods.has(Mods::CTRL) && c.is_ascii() => {
            let c = c.to_ascii_uppercase() as u8;
            if (b'@'..=b'_').contains(&c) {
                out.push(c & 31);
            }
        }
        Key::Character(c) => {
            let mut bytes = [0; 4];
            out.extend_from_slice(c.encode_utf8(&mut bytes).as_bytes());
        }
        _ => out.extend_from_slice(match key {
            Key::Enter => b"\r",
            Key::Backspace => b"\x7f",
            Key::Delete => b"\x1b[3~",
            Key::Left => b"\x1b[D",
            Key::Right => b"\x1b[C",
            Key::Up => b"\x1b[A",
            Key::Down => b"\x1b[B",
            Key::Home => b"\x1b[H",
            Key::End => b"\x1b[F",
            Key::Tab => b"\t",
            Key::Escape => b"\x1b",
            Key::PageUp => b"\x1b[5~",
            Key::PageDown => b"\x1b[6~",
            _ => b"",
        }),
    }
    out
}
