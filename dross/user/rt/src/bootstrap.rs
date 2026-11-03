//! Bootstrap messages: the first message on a program's handle 1. Text lines; a line `@name` names the next
//! handle of the message, other lines are `key value`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::sys::{self, Handle};

pub struct Bootstrap {
    pub handles: Vec<(String, Handle)>,
    pub values: Vec<(String, String)>,
}

impl Bootstrap {
    pub fn read(ch: Handle) -> sys::Result<Bootstrap> {
        let mut bytes = alloc::vec![0u8; dross_abi::MSG_BYTES];
        let mut hs = alloc::vec![Handle(0); dross_abi::MSG_HANDLES];
        let (nb, nh) = sys::channel_read(ch, &mut bytes, &mut hs, crate::FOREVER)?;
        let text = core::str::from_utf8(&bytes[..nb]).unwrap_or("");
        let mut b = Bootstrap { handles: Vec::new(), values: Vec::new() };
        let mut next = 0;
        for line in text.lines() {
            if let Some(name) = line.strip_prefix('@') {
                if next < nh {
                    b.handles.push((String::from(name), hs[next]));
                    next += 1;
                }
            } else if let Some((k, v)) = line.split_once(' ') {
                b.values.push((String::from(k), String::from(v)));
            }
        }
        Ok(b)
    }

    pub fn handle(&self, name: &str) -> Option<Handle> {
        self.handles.iter().find(|(n, _)| n == name).map(|(_, h)| *h)
    }

    pub fn value(&self, key: &str) -> Option<&str> {
        self.values.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn value_u64(&self, key: &str) -> Option<u64> {
        let v = self.value(key)?;
        match v.strip_prefix("0x") {
            Some(h) => u64::from_str_radix(h, 16).ok(),
            None => v.parse().ok(),
        }
    }
}

/// Builds a bootstrap message for a child.
#[derive(Default)]
pub struct Builder {
    text: String,
    handles: Vec<Handle>,
}

impl Builder {
    pub fn handle(mut self, name: &str, h: Handle) -> Self {
        self.text.push('@');
        self.text.push_str(name);
        self.text.push('\n');
        self.handles.push(h);
        self
    }

    pub fn value(mut self, key: &str, v: &str) -> Self {
        self.text.push_str(key);
        self.text.push(' ');
        self.text.push_str(v);
        self.text.push('\n');
        self
    }

    /// Writes the message to a new channel; returns the end for the child.
    pub fn send(self) -> sys::Result<Handle> {
        let (mine, theirs) = sys::channel_create()?;
        sys::channel_write(mine, self.text.as_bytes(), &self.handles)?;
        let _ = sys::close(mine);
        Ok(theirs)
    }
}

/// Spawns module `image` with a bootstrap message on `cpu` (`u64::MAX`: any).
pub fn spawn(image: Handle, name: &str, b: Builder, cpu: u64) -> sys::Result<Handle> {
    let ch = b.send()?;
    sys::spawn(image, name, Some(ch), cpu)
}
