//! The seat: keyboard and mouse reports as the desktop's input packets (`desktop/core/src/protocol.rs`), for the
//! display server on the `seat` channel. A US layout; a held key repeats after 500 ms, 30 times a second; the
//! pointer moves over the display and stays on it.

use dross_rt::{sys, Handle};
use oscar_desktop::protocol::{Input, Key, Message, Mods, Packet};

const REPEAT_DELAY_NS: u64 = 500_000_000;
const REPEAT_EVERY_NS: u64 = 33_000_000;

pub struct Seat {
    ch: Handle,
    width: i32,
    height: i32,
    x: i32,
    y: i32,
    down: bool,
    /// The newest held key, its modifiers and when it repeats next.
    held: Option<(Key, Mods, u64)>,
    pending_keyboard: Option<Input>,
}

impl Seat {
    pub fn new(ch: Handle, width: u32, height: u32) -> Seat {
        let (width, height) = (width.max(1) as i32, height.max(1) as i32);
        Seat { ch, width, height, x: width / 2, y: height / 2, down: false, held: None, pending_keyboard: None }
    }

    fn send(&self, input: Input) {
        if let Ok(bytes) = Packet::new(0, 0, Message::Input(input)).encode() {
            // A full channel drops the event rather than stalling the USB driver.
            let _ = sys::channel_write(self.ch, &bytes, &[]);
        }
    }

    /// A boot keyboard report: modifiers, then up to six pressed usages. Sends the newly pressed keys.
    pub fn keyboard(&mut self, report: &[u8; 8], last: &[u8; 8]) {
        let mods = mods(report[0]);
        // Retain the newest complete physical state until the channel accepts it.
        // Coalescing snapshots cannot strand a released game key after backpressure.
        if report[2..].iter().all(|k| *k == 0 || (4..=0xdf).contains(k)) {
            self.pending_keyboard = Some(Input::KeyboardState { modifiers:report[0],keys:report[2..8].try_into().unwrap() });
            self.flush_keyboard();
        }
        let mut newest = None;
        for &k in &report[2..8] {
            if k != 0 && !last[2..8].contains(&k) {
                if let Some(key) = key(k, report[0] & 0x22 != 0) {
                    self.send(Input::Key { key, mods });
                    newest = Some(key);
                }
            }
        }
        match newest {
            Some(key) => self.held = Some((key, mods, sys::clock() + REPEAT_DELAY_NS)),
            None => {
                // Stop repeating once the repeating key is released.
                if let Some((held, ..)) = self.held {
                    if !report[2..8].iter().any(|&k| k != 0 && key(k, report[0] & 0x22 != 0) == Some(held)) {
                        self.held = None;
                    }
                }
            }
        }
    }

    /// A boot mouse report: buttons, then relative x and y.
    pub fn mouse(&mut self, buttons: u8, dx: i8, dy: i8) {
        self.x = (self.x + dx as i32).clamp(0, self.width - 1);
        self.y = (self.y + dy as i32).clamp(0, self.height - 1);
        self.down = buttons & 1 != 0;
        self.send(Input::Pointer { x: self.x, y: self.y, down: self.down });
    }

    /// Sends a repeat if one is due; the time of the next one, if a key is held.
    fn flush_keyboard(&mut self) {
        if let Some(input)=self.pending_keyboard {
            if let Ok(bytes)=Packet::new(0,0,Message::Input(input)).encode() {
                if sys::channel_write(self.ch,&bytes,&[]).is_ok() { self.pending_keyboard=None; }
            }
        }
    }
    pub fn repeat(&mut self) -> Option<u64> {
        self.flush_keyboard();
        let now=sys::clock();
        if let Some((key,mods,at))=self.held {
            if now>=at { self.send(Input::Key { key,mods }); self.held=Some((key,mods,now+REPEAT_EVERY_NS)); }
        }
        let repeat=self.held.map(|(..,at)|at);
        match (repeat,self.pending_keyboard.is_some()) {
            (Some(at),true)=>Some(at.min(now+1_000_000)),
            (None,true)=>Some(now+1_000_000),
            (at,false)=>at,
        }
    }

}

fn mods(m: u8) -> Mods {
    // Left and right Ctrl, Shift, Alt.
    let mut v = 0;
    if m & 0x11 != 0 {
        v |= Mods::CTRL;
    }
    if m & 0x22 != 0 {
        v |= Mods::SHIFT;
    }
    if m & 0x44 != 0 {
        v |= Mods::ALT;
    }
    Mods(v)
}

/// A HID keyboard usage on a US layout.
fn key(k: u8, shift: bool) -> Option<Key> {
    const DIGITS: &[u8; 10] = b"1234567890";
    const SHIFTED_DIGITS: &[u8; 10] = b"!@#$%^&*()";
    // Usages 0x2d-0x38: - = [ ] \ (non-US #) ; ' ` , . /
    const PUNCT: &[u8; 12] = b"-=[]\\\\;'`,./";
    const SHIFTED_PUNCT: &[u8; 12] = b"_+{}||:\"~<>?";
    let c = |b: u8| Some(Key::Character(b as char));
    match k {
        0x04..=0x1d => {
            let l = b'a' + k - 0x04;
            c(if shift { l.to_ascii_uppercase() } else { l })
        }
        0x1e..=0x27 => c(if shift { SHIFTED_DIGITS } else { DIGITS }[(k - 0x1e) as usize]),
        0x28 | 0x58 => Some(Key::Enter),
        0x29 => Some(Key::Escape),
        0x2a => Some(Key::Backspace),
        0x2b => Some(Key::Tab),
        0x2c => c(b' '),
        0x2d..=0x38 => c(if shift { SHIFTED_PUNCT } else { PUNCT }[(k - 0x2d) as usize]),
        0x4a => Some(Key::Home),
        0x4b => Some(Key::PageUp),
        0x4c => Some(Key::Delete),
        0x4d => Some(Key::End),
        0x4e => Some(Key::PageDown),
        0x4f => Some(Key::Right),
        0x50 => Some(Key::Left),
        0x51 => Some(Key::Down),
        0x52 => Some(Key::Up),
        _ => None,
    }
}
