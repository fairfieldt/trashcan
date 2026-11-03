//! Owned adapter for init's existing text bootstrap. This format is private;
//! application networking uses the public framed service protocol.
use alloc::{string::String, vec::Vec};
use dross_rt::{
    Error, Handle,
    deadline::{Deadline, Duration},
    owned::{self, OwnedHandle},
};
pub struct Seed {
    caps: Vec<(String, OwnedHandle)>,
    values: Vec<(String, String)>,
}
fn string(s: &str) -> Result<String, Error> {
    let mut out = String::new();
    out.try_reserve_exact(s.len())
        .map_err(|_| Error::NoMemory)?;
    out.push_str(s);
    Ok(out)
}
impl Seed {
    pub fn read(raw: Handle) -> Result<Self, Error> {
        // Process entry transfers sole ownership of its bootstrap slot.
        let boot = unsafe { OwnedHandle::from_raw(raw) };
        let mut m = owned::receive(
            boot.as_handle(),
            Deadline::from_now(Duration::from_secs(15)),
        )?;
        let text = core::str::from_utf8(m.bytes()).map_err(|_| Error::InvalidArgs)?;
        let mut caps = Vec::new();
        let mut values = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let (name, value) = if let Some(name) = line.strip_prefix('@') {
                (name, None)
            } else {
                let (k, v) = line.split_once(' ').ok_or(Error::InvalidArgs)?;
                (k, Some(v))
            };
            if name.is_empty()
                || name.len() > 64
                || name.chars().any(char::is_control)
                || names.iter().any(|n| n == name)
            {
                return Err(Error::InvalidArgs);
            }
            names.try_reserve(1).map_err(|_| Error::NoMemory)?;
            names.push(string(name)?);
            if let Some(v) = value {
                values.try_reserve(1).map_err(|_| Error::NoMemory)?;
                values.push((string(name)?, string(v)?));
            } else {
                caps.try_reserve(1).map_err(|_| Error::NoMemory)?;
                caps.push((string(name)?, None));
            }
        }
        if caps.len() != m.handle_count() {
            return Err(Error::InvalidArgs);
        }
        for (i, (_, slot)) in caps.iter_mut().enumerate() {
            *slot = Some(m.take_handle(i).ok_or(Error::InvalidArgs)?);
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(caps.len())
            .map_err(|_| Error::NoMemory)?;
        for (n, c) in caps {
            owned.push((n, c.unwrap()));
        }
        Ok(Self {
            caps: owned,
            values,
        })
    }
    pub fn take(&mut self, name: &str) -> Option<OwnedHandle> {
        let i = self.caps.iter().position(|(n, _)| n == name)?;
        Some(self.caps.remove(i).1)
    }
    pub fn value(&self, name: &str) -> Option<&str> {
        self.values
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
    pub fn remaining(&self) -> usize {
        self.caps.len()
    }
}
