//! Trash desktop client: dock, front panel, terminal widget and command shell.
//! Both adapters send these pixels over the protocol; neither borrows server state.
use crate::{
    canvas::{CHAR_H, Canvas, Rect, char_w},
    protocol::{Fault, HEADER, Input, Key, MAX_PACKET, MAX_TEXT, Message, Mods, Packet},
    shell::{Environment, Shell},
    terminal::Terminal,
    ui::{Row, Theme, Ui, bevel},
};
use alloc::vec::Vec;

pub const WIDTH: u32 = 760;
pub const HEIGHT: u32 = 500;
pub struct Trash {
    pub surface: Canvas,
    pub terminal: Terminal,
    pub shell: Shell,
    ui: Ui,
    pub window: u32,
    pub dirty: bool,
    pub closed: bool,
    pub focused: bool,
    terminal_focus: bool,
    in_flight: bool,
    serial: u32,
    external: bool,
    pending_input: Vec<u8>,
}
impl Trash {
    pub fn new(width: u32, height: u32) -> Result<Self, Fault> {
        if width < 360 || height < 240 {
            return Err(Fault::Geometry);
        }
        let area = Self::terminal_area(width, height);
        let mut terminal = Terminal::new(
            area.w as usize / char_w() as usize,
            area.h as usize / CHAR_H as usize,
        )?;
        let shell = Shell::default();
        shell.start(&mut terminal);
        Ok(Self {
            surface: Canvas::new(width, height)?,
            terminal,
            shell,
            ui: Ui::default(),
            window: 0,
            dirty: true,
            closed: false,
            focused: false,
            terminal_focus: true,
            in_flight: false,
            serial: 1,
            external: false,
            pending_input: Vec::new(),
        })
    }
    pub fn create(&self) -> Packet {
        Packet::new(
            1,
            0,
            Message::Create {
                width: self.surface.width(),
                height: self.surface.height(),
                title: if self.external { "Terminal" } else { "Trash" }.into(),
            },
        )
    }
    pub fn use_byte_stream(&mut self) {
        self.external = true;
        self.terminal.clear();
        self.dirty = true;
    }
    pub fn take_input(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.pending_input)
    }
    pub fn feed(&mut self, bytes: &[u8]) {
        self.terminal.feed(bytes);
        self.dirty = true;
    }
    fn terminal_area(w: u32, h: u32) -> Rect {
        Rect::new(68, 10, w - 98, h - 64)
    }
    pub fn receive(&mut self, p: Packet, env: &mut impl Environment) -> Result<(), Fault> {
        if let Message::Created { width, height } = p.message {
            if self.window != 0 || width != self.surface.width() || height != self.surface.height()
            {
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
            Message::Input(input) => self.input(input, env),
            Message::FrameDone if p.serial == self.serial => self.in_flight = false,
            Message::CloseRequested => self.closed = true,
            Message::Closed => self.closed = true,
            Message::Error(e) => return Err(e),
            _ => {}
        }
        Ok(())
    }
    pub fn input(&mut self, input: Input, env: &mut impl Environment) {
        if let Input::Pointer { x, y, down: true } = input {
            self.terminal_focus =
                Self::terminal_area(self.surface.width(), self.surface.height()).contains(x, y);
            if self.terminal_focus {
                self.ui.clear_focus();
            }
        }
        if let Input::Key { key, mods } = input {
            if key == Key::Tab && mods.has(Mods::CTRL) {
                self.terminal_focus = !self.terminal_focus;
            } else if key == Key::PageUp {
                self.terminal.scroll_view(self.terminal.rows() as i32 / 2);
            } else if key == Key::PageDown {
                self.terminal
                    .scroll_view(-(self.terminal.rows() as i32 / 2));
            } else if self.terminal_focus {
                if self.external {
                    let bytes = crate::terminal::encode_key(key, mods);
                    if self.pending_input.len() + bytes.len() <= MAX_TEXT {
                        self.pending_input.extend(bytes);
                    }
                    self.terminal.scroll_view(-10000);
                } else {
                    self.shell.key(key, mods, &mut self.terminal, env);
                }
            }
        }
        let widget_input = if matches!(input, Input::Key { .. }) && self.terminal_focus {
            None
        } else {
            Some(input)
        };
        self.paint(widget_input, env);
        self.dirty = true;
        if self.shell.exited {
            self.closed = true;
        }
    }
    pub fn command(&mut self, command: &str, env: &mut impl Environment) {
        self.shell.command(command, &mut self.terminal, env);
        self.dirty = true;
        if self.shell.exited {
            self.closed = true;
        }
    }
    fn paint(&mut self, input: Option<Input>, env: &mut impl Environment) {
        let c = &mut self.surface;
        let bounds = c.bounds();
        c.fill(bounds, Theme::FACE);
        // NeXT-style vertical dock. Each tile has a real action.
        bevel(c, Rect::new(4, 4, 56, bounds.h - 8), Theme::SHADOW, true);
        self.ui.begin(input);
        let shell = self.ui.button(c, 1, Rect::new(8, 9, 48, 48), ">_");
        let files = !self.external && self.ui.button(c, 2, Rect::new(8, 61, 48, 48), "Files");
        let help = !self.external && self.ui.button(c, 3, Rect::new(8, 113, 48, 48), "Help");
        let area = Self::terminal_area(bounds.w, bounds.h);
        bevel(
            c,
            Rect::new(area.x - 2, area.y - 2, area.w + 4, area.h + 4),
            Theme::TERMINAL,
            true,
        );
        // CDE front panel: inset frame with small, grouped controls.
        let panel = Rect::new(66, bounds.h as i32 - 44, bounds.w - 72, 38);
        bevel(c, panel, Theme::PANEL, false);
        let mut row = Row::new(Rect::new(panel.x + 6, panel.y + 6, panel.w - 12, 26), 6);
        let clear = self.ui.button(c, 4, row.take(80), "Clear");
        let older = self.ui.button(c, 5, row.take(80), "Older");
        let newer = self.ui.button(c, 6, row.take(80), "Newer");
        let quit = self.ui.button(c, 7, row.take(64), "Quit");
        let max = self.terminal.history_len();
        let view = self.terminal.view_offset();
        let value = self.ui.scrollbar(
            c,
            8,
            Rect::new(bounds.w as i32 - 26, area.y - 2, 18, area.h + 4),
            max,
            max - view,
            self.terminal.rows(),
        );
        self.terminal
            .scroll_view((max - value) as i32 - view as i32);
        self.ui.end();
        if clear {
            self.terminal.clear();
            if !self.external {
                self.shell = Shell::default();
                self.shell.start(&mut self.terminal);
            }
        }
        if older {
            self.terminal.scroll_view(self.terminal.rows() as i32 / 2);
        }
        if newer {
            self.terminal
                .scroll_view(-(self.terminal.rows() as i32 / 2));
        }
        if files {
            self.shell.command("ls", &mut self.terminal, env);
        }
        if help {
            self.shell.command("help", &mut self.terminal, env);
        }
        if shell || files || help || clear {
            self.terminal_focus = true;
            self.ui.clear_focus();
        }
        if quit {
            self.closed = true;
        }
        self.terminal
            .draw(c, area, self.focused && self.terminal_focus);
    }
    /// Upload at most one frame until acknowledged. Chunking fits dross's channel
    /// limit and makes backpressure a transport concern rather than an allocation.
    pub fn frame(
        &mut self,
        env: &mut impl Environment,
        mut send: impl FnMut(Packet) -> Result<(), Fault>,
    ) -> Result<(), Fault> {
        if !self.dirty || self.in_flight || self.window == 0 || self.closed {
            return Ok(());
        }
        self.paint(None, env);
        self.serial = self.serial.checked_add(1).ok_or(Fault::Limit)?;
        let chunk = (MAX_PACKET - HEADER - 4) / 4;
        for (i, pixels) in self.surface.pixels().chunks(chunk).enumerate() {
            send(Packet::new(
                self.serial,
                self.window,
                Message::Upload {
                    offset: (i * chunk) as u32,
                    pixels: pixels.into(),
                },
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
