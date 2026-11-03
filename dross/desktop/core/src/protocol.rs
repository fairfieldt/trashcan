//! Versioned little-endian messages; never cast untrusted bytes to Rust structs.
use alloc::{string::String, vec::Vec};

pub const VERSION: u16 = 1;
pub const MAX_PACKET: usize = 60 * 1024;
pub const HEADER: usize = 20;
pub const MAX_TITLE: usize = 80;
pub const MAX_DIMENSION: u32 = 2048;
pub const MAX_WINDOW_PIXELS: usize = 512 * 1024;
pub const MAX_TEXT: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Fault {
    Malformed = 1,
    Version = 2,
    Ownership = 3,
    Limit = 4,
    Geometry = 5,
    Unknown = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Character(char),
    Enter,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Tab,
    Escape,
    PageUp,
    PageDown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods(pub u8);
impl Mods {
    pub const SHIFT: u8 = 1;
    pub const CTRL: u8 = 2;
    pub const ALT: u8 = 4;
    pub fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Pointer { x: i32, y: i32, down: bool },
    Key { key: Key, mods: Mods },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    Create {
        width: u32,
        height: u32,
        title: String,
    },
    Created {
        width: u32,
        height: u32,
    },
    /// Pixel offset in a tightly packed opaque XRGB8888 client surface.
    Upload {
        offset: u32,
        pixels: Vec<u32>,
    },
    Present,
    FrameDone,
    Close,
    Closed,
    CloseRequested,
    Focus(bool),
    Input(Input),
    Error(Fault),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub serial: u32,
    pub window: u32,
    pub message: Message,
}

fn put(out: &mut Vec<u8>, n: u32) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn get(bytes: &[u8], at: usize) -> Result<u32, Fault> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .ok_or(Fault::Malformed)?
            .try_into()
            .map_err(|_| Fault::Malformed)?,
    ))
}
fn boolean(n: u8) -> Result<bool, Fault> {
    match n {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Fault::Malformed),
    }
}

impl Packet {
    pub fn new(serial: u32, window: u32, message: Message) -> Self {
        Self {
            serial,
            window,
            message,
        }
    }
    pub fn encode(&self) -> Result<Vec<u8>, Fault> {
        let mut p = Vec::new();
        let op: u16 = match &self.message {
            Message::Create {
                width,
                height,
                title,
            } => {
                if title.len() > MAX_TITLE || title.chars().any(char::is_control) {
                    return Err(Fault::Malformed);
                }
                put(&mut p, *width);
                put(&mut p, *height);
                p.extend_from_slice(title.as_bytes());
                1
            }
            Message::Created { width, height } => {
                put(&mut p, *width);
                put(&mut p, *height);
                2
            }
            Message::Upload { offset, pixels } => {
                put(&mut p, *offset);
                for v in pixels {
                    put(&mut p, *v);
                }
                3
            }
            Message::Present => 4,
            Message::FrameDone => 5,
            Message::Close => 6,
            Message::Closed => 7,
            Message::CloseRequested => 8,
            Message::Focus(f) => {
                p.push(u8::from(*f));
                9
            }
            Message::Input(Input::Pointer { x, y, down }) => {
                put(&mut p, *x as u32);
                put(&mut p, *y as u32);
                p.push(u8::from(*down));
                10
            }
            Message::Input(Input::Key { key, mods }) => {
                put(&mut p, key_code(*key));
                p.push(mods.0);
                11
            }
            Message::Error(e) => {
                put(&mut p, *e as u32);
                12
            }
        };
        if p.len() + HEADER > MAX_PACKET {
            return Err(Fault::Limit);
        }
        let mut out = Vec::with_capacity(p.len() + HEADER);
        out.extend_from_slice(b"OSCR");
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&op.to_le_bytes());
        put(&mut out, self.serial);
        put(&mut out, self.window);
        put(&mut out, p.len() as u32);
        out.extend_from_slice(&p);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, Fault> {
        if bytes.len() < HEADER || bytes.len() > MAX_PACKET || &bytes[..4] != b"OSCR" {
            return Err(Fault::Malformed);
        }
        if u16::from_le_bytes([bytes[4], bytes[5]]) != VERSION {
            return Err(Fault::Version);
        }
        if get(bytes, 16)? as usize != bytes.len() - HEADER {
            return Err(Fault::Malformed);
        }
        let p = &bytes[HEADER..];
        let op = u16::from_le_bytes([bytes[6], bytes[7]]);
        let message = match (op, p.len()) {
            (1, n) if (8..=8 + MAX_TITLE).contains(&n) => {
                let title = core::str::from_utf8(&p[8..]).map_err(|_| Fault::Malformed)?;
                if title.chars().any(char::is_control) {
                    return Err(Fault::Malformed);
                }
                Message::Create {
                    width: get(p, 0)?,
                    height: get(p, 4)?,
                    title: String::from(title),
                }
            }
            (2, 8) => Message::Created {
                width: get(p, 0)?,
                height: get(p, 4)?,
            },
            (3, n) if n >= 4 && n % 4 == 0 => {
                let mut pixels = Vec::with_capacity((n - 4) / 4);
                for at in (4..n).step_by(4) {
                    pixels.push(get(p, at)?);
                }
                Message::Upload {
                    offset: get(p, 0)?,
                    pixels,
                }
            }
            (4, 0) => Message::Present,
            (5, 0) => Message::FrameDone,
            (6, 0) => Message::Close,
            (7, 0) => Message::Closed,
            (8, 0) => Message::CloseRequested,
            (9, 1) => Message::Focus(boolean(p[0])?),
            (10, 9) => Message::Input(Input::Pointer {
                x: get(p, 0)? as i32,
                y: get(p, 4)? as i32,
                down: boolean(p[8])?,
            }),
            (11, 5) if p[4] & !7 == 0 => Message::Input(Input::Key {
                key: decode_key(get(p, 0)?)?,
                mods: Mods(p[4]),
            }),
            (12, 4) => Message::Error(match get(p, 0)? {
                1 => Fault::Malformed,
                2 => Fault::Version,
                3 => Fault::Ownership,
                4 => Fault::Limit,
                5 => Fault::Geometry,
                6 => Fault::Unknown,
                _ => return Err(Fault::Malformed),
            }),
            _ => return Err(Fault::Malformed),
        };
        Ok(Self {
            serial: get(bytes, 8)?,
            window: get(bytes, 12)?,
            message,
        })
    }
}

fn key_code(key: Key) -> u32 {
    match key {
        Key::Character(c) => c as u32,
        Key::Enter => 0x110000,
        Key::Backspace => 0x110001,
        Key::Delete => 0x110002,
        Key::Left => 0x110003,
        Key::Right => 0x110004,
        Key::Up => 0x110005,
        Key::Down => 0x110006,
        Key::Home => 0x110007,
        Key::End => 0x110008,
        Key::Tab => 0x110009,
        Key::Escape => 0x11000a,
        Key::PageUp => 0x11000b,
        Key::PageDown => 0x11000c,
    }
}
fn decode_key(n: u32) -> Result<Key, Fault> {
    Ok(match n {
        0x110000 => Key::Enter,
        0x110001 => Key::Backspace,
        0x110002 => Key::Delete,
        0x110003 => Key::Left,
        0x110004 => Key::Right,
        0x110005 => Key::Up,
        0x110006 => Key::Down,
        0x110007 => Key::Home,
        0x110008 => Key::End,
        0x110009 => Key::Tab,
        0x11000a => Key::Escape,
        0x11000b => Key::PageUp,
        0x11000c => Key::PageDown,
        _ => Key::Character(
            char::from_u32(n)
                .filter(|c| !c.is_control())
                .ok_or(Fault::Malformed)?,
        ),
    })
}
