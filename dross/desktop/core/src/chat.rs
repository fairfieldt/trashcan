//! The chat window: an instant-message conversation with Oscar (desktop/chat drives it and does the networking).
//! A transcript with the speakers' names in colour, an input box, Oscar's buddy icon, a status line, and Clear
//! and Send. Enter sends; Page Up and Page Down scroll the transcript.
use crate::{
    canvas::{CHAR_H, Canvas, Rect, char_w},
    protocol::{Fault, HEADER, Input, Key, MAX_PACKET, Message, Packet},
    ui::{Row, Theme, Ui, bevel},
};
use alloc::{string::String, vec::Vec};

pub const WIDTH: u32 = 520;
pub const HEIGHT: u32 = 400;
/// The user's name in the transcript, and Oscar's (his green).
const YOU: u32 = 0xb0281e;
const OSCAR: u32 = 0x2f7a1f;
const MAX_INPUT: usize = 2000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Who {
    You,
    Oscar,
}

/// What the window asks of its driver.
#[derive(PartialEq, Eq, Debug)]
pub enum Request {
    Send(String),
    Reset,
}

pub struct ChatWindow {
    pub surface: Canvas,
    ui: Ui,
    pub window: u32,
    pub dirty: bool,
    pub closed: bool,
    focused: bool,
    in_flight: bool,
    serial: u32,
    entries: Vec<(Who, String)>,
    input: String,
    /// Lines scrolled back from the newest.
    scroll: usize,
    /// Waiting for Oscar's reply.
    pub busy: bool,
    status: String,
    requests: Vec<Request>,
}

impl ChatWindow {
    pub fn new(width: u32, height: u32) -> Result<Self, Fault> {
        if width < 360 || height < 240 {
            return Err(Fault::Geometry);
        }
        Ok(Self {
            surface: Canvas::new(width, height)?,
            ui: Ui::default(),
            window: 0,
            dirty: true,
            closed: false,
            focused: false,
            in_flight: false,
            serial: 1,
            entries: Vec::new(),
            input: String::new(),
            scroll: 0,
            busy: false,
            status: String::new(),
            requests: Vec::new(),
        })
    }

    pub fn create(&self) -> Packet {
        Packet::new(
            1,
            0,
            Message::Create { width: self.surface.width(), height: self.surface.height(), title: "Oscar".into() },
        )
    }

    /// The driver's work: messages to send, conversations to reset.
    pub fn take_requests(&mut self) -> Vec<Request> {
        core::mem::take(&mut self.requests)
    }

    /// Oscar's reply begins.
    pub fn reply_start(&mut self) {
        self.entries.push((Who::Oscar, String::new()));
        self.dirty = true;
    }

    /// More of Oscar's reply (already plain text).
    pub fn reply_text(&mut self, text: &str) {
        if let Some((Who::Oscar, t)) = self.entries.last_mut() {
            t.push_str(text);
        }
        self.dirty = true;
    }

    pub fn reply_done(&mut self) {
        self.busy = false;
        self.status.clear();
        self.dirty = true;
    }

    /// The reply failed: `why` goes on the status line, and the unanswered message is taken back.
    pub fn reply_failed(&mut self, why: &str) {
        if matches!(self.entries.last(), Some((Who::Oscar, t)) if t.is_empty()) {
            self.entries.pop();
        }
        if let Some((Who::You, t)) = self.entries.last() {
            if self.input.is_empty() {
                self.input = t.clone();
            }
            self.entries.pop();
        }
        self.busy = false;
        self.status = String::from(why);
        self.dirty = true;
    }

    pub fn receive(&mut self, p: Packet) -> Result<(), Fault> {
        if let Message::Created { width, height } = p.message {
            if self.window != 0 || width != self.surface.width() || height != self.surface.height() {
                return Err(Fault::Geometry);
            }
            self.window = p.window;
            self.dirty = true;
            return Ok(());
        }
        if p.window != self.window {
            return Err(Fault::Ownership);
        }
        match p.message {
            Message::Focus(f) => {
                self.focused = f;
                self.dirty = true;
                if !f {
                    self.ui.clear_focus();
                }
            }
            Message::Input(input) => self.input(input),
            Message::FrameDone if p.serial == self.serial => self.in_flight = false,
            Message::CloseRequested | Message::Closed => self.closed = true,
            Message::Error(e) => return Err(e),
            _ => {}
        }
        Ok(())
    }

    pub fn input(&mut self, input: Input) {
        if let Input::Key { key, mods } = input {
            match key {
                Key::Enter => self.send(),
                Key::Backspace => {
                    self.input.pop();
                }
                Key::PageUp => self.scroll += self.page() / 2,
                Key::PageDown => self.scroll = self.scroll.saturating_sub(self.page() / 2),
                Key::Character(c) if !mods.has(crate::protocol::Mods::CTRL) && self.input.len() < MAX_INPUT => {
                    self.input.push(c)
                }
                _ => {}
            }
            // Keys go to the input box, not the buttons.
            self.paint(None);
        } else {
            self.paint(Some(input));
        }
        self.dirty = true;
    }

    fn send(&mut self) {
        let text = String::from(self.input.trim());
        if self.busy || text.is_empty() {
            return;
        }
        self.input.clear();
        self.entries.push((Who::You, text.clone()));
        self.requests.push(Request::Send(text));
        self.busy = true;
        self.scroll = 0;
        self.status = String::from("Oscar is typing...");
    }

    fn clear(&mut self) {
        if self.busy {
            return;
        }
        self.entries.clear();
        self.scroll = 0;
        self.status.clear();
        self.requests.push(Request::Reset);
    }

    fn transcript_area(&self) -> Rect {
        let (w, h) = (self.surface.width(), self.surface.height());
        Rect::new(10, 10, w - 46, h - 136)
    }

    fn page(&self) -> usize {
        (self.transcript_area().h as usize - 8) / CHAR_H as usize
    }

    /// The transcript as display lines: (text, the speaker whose name starts the line, if one does).
    fn lines(&self, cols: usize) -> Vec<(String, Option<Who>)> {
        let mut out = Vec::new();
        for (who, text) in &self.entries {
            let name = if *who == Who::You { "You: " } else { "Oscar: " };
            let mut first = true;
            for line in wrap(&alloc::format!("{name}{text}"), cols) {
                out.push((line, first.then_some(*who)));
                first = false;
            }
        }
        out
    }

    fn paint(&mut self, input: Option<Input>) {
        let (w, h) = (self.surface.width(), self.surface.height());
        let area = self.transcript_area();
        let cols = ((area.w - 8) / char_w() as u32) as usize;
        let lines = self.lines(cols);
        let page = self.page();
        let max = lines.len().saturating_sub(page);
        self.scroll = self.scroll.min(max);

        let c = &mut self.surface;
        c.fill(c.bounds(), Theme::FACE);
        self.ui.begin(input);

        // Transcript, on paper.
        bevel(c, Rect::new(area.x - 2, area.y - 2, area.w + 4, area.h + 4), Theme::PAPER, true);
        let first = max - self.scroll;
        for (i, (line, who)) in lines.iter().skip(first).take(page).enumerate() {
            let y = area.y + 4 + i as i32 * CHAR_H;
            c.text(area.x + 4, y, line, Theme::INK, area);
            if let Some(who) = who {
                let (name, color) = if *who == Who::You { ("You:", YOU) } else { ("Oscar:", OSCAR) };
                // Bold: the name twice, a pixel apart.
                c.text(area.x + 4, y, name, color, area);
                c.text(area.x + 5, y, name, color, area);
            }
        }
        let value = self.ui.scrollbar(c, 1, Rect::new(w as i32 - 30, area.y - 2, 18, area.h + 4), max, first, page);
        self.scroll = max - value.min(max);

        // Input box and Oscar's buddy icon.
        let iy = area.y + area.h as i32 + 10;
        let input = Rect::new(10, iy, w - 86, 4 * CHAR_H as u32 + 8);
        bevel(c, Rect::new(input.x - 2, input.y - 2, input.w + 4, input.h + 4), Theme::PAPER, true);
        let icols = ((input.w - 8) / char_w() as u32) as usize;
        let mut ilines = wrap(&self.input, icols);
        if ilines.is_empty() {
            ilines.push(String::new());
        }
        let shown = ilines.len().saturating_sub(4);
        for (i, line) in ilines.iter().skip(shown).enumerate() {
            c.text(input.x + 4, input.y + 4 + i as i32 * CHAR_H, line, Theme::INK, input);
        }
        if self.focused && !self.busy {
            let last = ilines.len() - 1 - shown;
            let x = input.x + 4 + ilines[ilines.len() - 1].chars().count() as i32 * char_w();
            c.fill(Rect::new(x, input.y + 4 + last as i32 * CHAR_H + CHAR_H - 3, char_w() as u32, 2), Theme::INK);
        }
        let icon = Rect::new(w as i32 - 70, iy - 2, 60, input.h + 4);
        bevel(c, icon, Theme::PANEL, true);
        trash_can(c, icon.x + 6, icon.y + (icon.h as i32 - 48) / 2);

        // Status line, Clear and Send.
        let by = h as i32 - 36;
        c.text(12, by + 5, &self.status, Theme::INK, Rect::new(10, by, w - 180, 26));
        let mut row = Row::new(Rect::new(w as i32 - 166, by, 156, 26), 8);
        let clear = self.ui.button(c, 2, row.take(68), "Clear");
        let send = self.ui.button(c, 3, row.take(80), "Send");
        self.ui.end();
        if send {
            self.send();
        }
        if clear {
            self.clear();
        }
    }

    /// Uploads at most one frame until it is acknowledged, as the Trash client does.
    pub fn frame(&mut self, mut send: impl FnMut(Packet) -> Result<(), Fault>) -> Result<(), Fault> {
        if !self.dirty || self.in_flight || self.window == 0 || self.closed {
            return Ok(());
        }
        self.paint(None);
        self.serial = self.serial.checked_add(1).ok_or(Fault::Limit)?;
        let chunk = (MAX_PACKET - HEADER - 4) / 4;
        for (i, pixels) in self.surface.pixels().chunks(chunk).enumerate() {
            send(Packet::new(
                self.serial,
                self.window,
                Message::Upload { offset: (i * chunk) as u32, pixels: pixels.into() },
            ))?;
        }
        send(Packet::new(self.serial, self.window, Message::Present))?;
        self.in_flight = true;
        self.dirty = false;
        Ok(())
    }

    pub fn close_packet(&self) -> Packet {
        Packet::new(self.serial, self.window, Message::Close)
    }
}

/// Oscar's buddy icon, 48 pixels square: his can, the lid up, his eyes in the gap.
fn trash_can(c: &mut Canvas, x: i32, y: i32) {
    const METAL: u32 = 0x8c949a;
    const DARK: u32 = 0x3c4246;
    const FUR: u32 = 0x5f9a32;
    // Lid and handle, raised.
    c.fill(Rect::new(x + 2, y + 6, 44, 5), METAL);
    c.fill(Rect::new(x + 2, y + 10, 44, 1), DARK);
    c.fill(Rect::new(x + 18, y + 2, 12, 4), DARK);
    // Oscar between lid and can.
    c.fill(Rect::new(x + 6, y + 11, 36, 9), FUR);
    for ex in [x + 14, x + 26] {
        c.fill(Rect::new(ex, y + 12, 8, 7), 0xffffff);
        c.fill(Rect::new(ex + 3, y + 15, 3, 3), Theme::INK);
    }
    c.fill(Rect::new(x + 12, y + 11, 24, 1), DARK);
    // The can: rim, ribs, base.
    c.fill(Rect::new(x + 4, y + 20, 40, 26), METAL);
    c.fill(Rect::new(x + 3, y + 20, 42, 2), DARK);
    for rx in (x + 9..x + 42).step_by(7) {
        c.fill(Rect::new(rx, y + 24, 2, 19), DARK);
    }
    c.fill(Rect::new(x + 5, y + 45, 38, 2), DARK);
}

/// `text` in lines of at most `cols` characters, broken at spaces where possible.
pub fn wrap(text: &str, cols: usize) -> Vec<String> {
    let cols = cols.max(1);
    let mut out = Vec::new();
    for para in text.split('\n') {
        let mut line = String::new();
        for word in para.split(' ') {
            let mut word = word;
            while word.chars().count() > cols {
                if !line.is_empty() {
                    out.push(core::mem::take(&mut line));
                }
                let cut = word.char_indices().nth(cols).map_or(word.len(), |(i, _)| i);
                out.push(String::from(&word[..cut]));
                word = &word[cut..];
            }
            let need = line.chars().count() + usize::from(!line.is_empty()) + word.chars().count();
            if need > cols && !line.is_empty() {
                out.push(core::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        out.push(line);
    }
    out
}
