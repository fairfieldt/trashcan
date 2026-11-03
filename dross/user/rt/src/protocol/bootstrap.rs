//! Owned, field-encoded bootstrap v2. All metadata is checked before any
//! attachment is taken. Rejecting a message drops every received capability.
use super::{
    ProtocolError,
    frame::{Frame, Header, get16, get32},
    schema::*,
    valid_name,
};
use crate::owned::{Message, OwnedHandle};
use alloc::{string::String, vec::Vec};

pub struct Capability {
    pub protocol: u32,
    pub handle: OwnedHandle,
}
enum Data {
    Cap {
        protocol: u32,
        index: usize,
        handle: Option<OwnedHandle>,
    },
    Value(Vec<u8>),
}
struct Entry {
    name: String,
    data: Data,
}
pub struct BootstrapV2 {
    entries: Vec<Entry>,
}
pub struct Builder {
    entries: Vec<Entry>,
}
pub struct AddHandleError {
    pub error: ProtocolError,
    pub handle: OwnedHandle,
}
pub struct BuildError {
    pub error: ProtocolError,
    pub builder: Builder,
}
fn name_copy(name: &str) -> Result<String, ProtocolError> {
    let mut s = String::new();
    s.try_reserve_exact(name.len())
        .map_err(|_| ProtocolError::Allocation)?;
    s.push_str(name);
    Ok(s)
}
impl BootstrapV2 {
    pub fn decode(mut message: Message) -> Result<Self, ProtocolError> {
        let f = Frame::parse(message.bytes(), message.handle_count())?;
        if !f.header.supported(PROTOCOL_BOOTSTRAP, MAJOR_BOOTSTRAP)
            || f.header.kind != KIND_EVENT
            || f.header.ordinal != BOOTSTRAP_START
        {
            return Err(ProtocolError::Unsupported);
        }
        let b = f.body;
        if b.len() < 4 || get16(b, 2) != 0 {
            return Err(ProtocolError::Malformed);
        }
        let count = get16(b, 0) as usize;
        if count > MAX_BOOTSTRAP_ENTRIES {
            return Err(ProtocolError::Limit);
        }
        let mut entries: Vec<Entry> = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| ProtocolError::Allocation)?;
        let mut at = 4;
        let mut seen = 0u64;
        for _ in 0..count {
            if b.len() - at < 8 {
                return Err(ProtocolError::Malformed);
            }
            let kind = b[at];
            let reserved = b[at + 1];
            let nl = get16(b, at + 2) as usize;
            let dl = get32(b, at + 4) as usize;
            at += 8;
            let end = at
                .checked_add(nl)
                .and_then(|v| v.checked_add(dl))
                .filter(|&e| e <= b.len())
                .ok_or(ProtocolError::Malformed)?;
            let name =
                core::str::from_utf8(&b[at..at + nl]).map_err(|_| ProtocolError::Malformed)?;
            if reserved != 0 || !valid_name(name) || entries.iter().any(|e| e.name == name) {
                return Err(ProtocolError::Malformed);
            }
            let data = &b[at + nl..end];
            let data = match kind {
                1 => {
                    if dl != 8 || get32(data, 0) == 0 || get16(data, 6) != 0 {
                        return Err(ProtocolError::Malformed);
                    }
                    let index = get16(data, 4) as usize;
                    if index >= message.handle_count() || seen & (1u64 << index) != 0 {
                        return Err(ProtocolError::Malformed);
                    }
                    seen |= 1u64 << index;
                    Data::Cap {
                        protocol: get32(data, 0),
                        index,
                        handle: None,
                    }
                }
                2 => {
                    if dl > MAX_VALUE_BYTES {
                        return Err(ProtocolError::Limit);
                    }
                    let mut v = Vec::new();
                    v.try_reserve_exact(dl)
                        .map_err(|_| ProtocolError::Allocation)?;
                    v.extend_from_slice(data);
                    Data::Value(v)
                }
                _ => return Err(ProtocolError::Malformed),
            };
            entries.push(Entry {
                name: name_copy(name)?,
                data,
            });
            at = end;
        }
        let mask = if message.handle_count() == 64 {
            u64::MAX
        } else {
            (1u64 << message.handle_count()) - 1
        };
        if at != b.len() || seen != mask || message.remaining_handles() != message.handle_count() {
            return Err(ProtocolError::Malformed);
        }
        for e in &mut entries {
            if let Data::Cap { index, handle, .. } = &mut e.data {
                *handle = Some(
                    message
                        .take_handle(*index)
                        .ok_or(ProtocolError::Malformed)?,
                );
            }
        }
        Ok(Self { entries })
    }
    pub fn take(&mut self, name: &str) -> Option<Capability> {
        let e = self.entries.iter_mut().find(|e| e.name == name)?;
        match &mut e.data {
            Data::Cap {
                protocol, handle, ..
            } => Some(Capability {
                protocol: *protocol,
                handle: handle.take()?,
            }),
            _ => None,
        }
    }
    pub fn value(&self, name: &str) -> Option<&[u8]> {
        match &self.entries.iter().find(|e| e.name == name)?.data {
            Data::Value(v) => Some(v),
            _ => None,
        }
    }
    pub fn take_next(&mut self) -> Option<(String, Capability)> {
        let i = self.entries.iter().position(|e| {
            matches!(
                &e.data,
                Data::Cap {
                    handle: Some(_),
                    ..
                }
            )
        })?;
        let e = self.entries.remove(i);
        match e.data {
            Data::Cap {
                protocol,
                handle: Some(handle),
                ..
            } => Some((e.name, Capability { protocol, handle })),
            _ => None,
        }
    }
    pub fn remaining_handles(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| {
                matches!(
                    &e.data,
                    Data::Cap {
                        handle: Some(_),
                        ..
                    }
                )
            })
            .count()
    }
}
impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}
impl Builder {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
    fn check(&self, name: &str) -> Result<(), ProtocolError> {
        if !valid_name(name) || self.entries.iter().any(|e| e.name == name) {
            return Err(ProtocolError::Malformed);
        }
        if self.entries.len() >= MAX_BOOTSTRAP_ENTRIES {
            return Err(ProtocolError::Limit);
        }
        Ok(())
    }
    pub fn try_handle(
        &mut self,
        name: &str,
        protocol: u32,
        handle: OwnedHandle,
    ) -> Result<(), AddHandleError> {
        let prepare = (|| {
            self.check(name)?;
            if protocol == 0 {
                return Err(ProtocolError::Malformed);
            }
            let name = name_copy(name)?;
            self.entries
                .try_reserve(1)
                .map_err(|_| ProtocolError::Allocation)?;
            Ok(name)
        })();
        match prepare {
            Ok(name) => {
                self.entries.push(Entry {
                    name,
                    data: Data::Cap {
                        protocol,
                        index: 0,
                        handle: Some(handle),
                    },
                });
                Ok(())
            }
            Err(error) => Err(AddHandleError { error, handle }),
        }
    }
    pub fn try_value(&mut self, name: &str, value: &[u8]) -> Result<(), ProtocolError> {
        self.check(name)?;
        if value.len() > MAX_VALUE_BYTES {
            return Err(ProtocolError::Limit);
        }
        let name = name_copy(name)?;
        let mut v = Vec::new();
        v.try_reserve_exact(value.len())
            .map_err(|_| ProtocolError::Allocation)?;
        v.extend_from_slice(value);
        self.entries
            .try_reserve(1)
            .map_err(|_| ProtocolError::Allocation)?;
        self.entries.push(Entry {
            name,
            data: Data::Value(v),
        });
        Ok(())
    }
    pub fn build(mut self) -> Result<Message, BuildError> {
        let result = (|| {
            let handles = self
                .entries
                .iter()
                .filter(|e| matches!(e.data, Data::Cap { .. }))
                .count();
            let len = 4 + self
                .entries
                .iter()
                .map(|e| {
                    8 + e.name.len()
                        + match &e.data {
                            Data::Cap { .. } => 8,
                            Data::Value(v) => v.len(),
                        }
                })
                .sum::<usize>();
            let h = Header::new(
                PROTOCOL_BOOTSTRAP,
                MAJOR_BOOTSTRAP,
                KIND_EVENT,
                BOOTSTRAP_START,
                0,
                len,
                handles,
            )?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(32 + len)
                .map_err(|_| ProtocolError::Allocation)?;
            bytes.resize(32, 0);
            h.encode(&mut bytes)?;
            bytes.extend_from_slice(&(self.entries.len() as u16).to_le_bytes());
            bytes.extend_from_slice(&[0, 0]);
            let mut index = 0;
            for e in &self.entries {
                let (kind, dl) = match &e.data {
                    Data::Cap { .. } => (1, 8),
                    Data::Value(v) => (2, v.len()),
                };
                bytes.push(kind);
                bytes.push(0);
                bytes.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
                bytes.extend_from_slice(&(dl as u32).to_le_bytes());
                bytes.extend_from_slice(e.name.as_bytes());
                match &e.data {
                    Data::Cap { protocol, .. } => {
                        bytes.extend_from_slice(&protocol.to_le_bytes());
                        bytes.extend_from_slice(&(index as u16).to_le_bytes());
                        bytes.extend_from_slice(&[0, 0]);
                        index += 1;
                    }
                    Data::Value(v) => bytes.extend_from_slice(v),
                }
            }
            Message::try_from_bytes(&bytes).map_err(|_| ProtocolError::Allocation)
        })();
        let mut message = match result {
            Ok(m) => m,
            Err(error) => {
                return Err(BuildError {
                    error,
                    builder: self,
                });
            }
        };
        for i in 0..self.entries.len() {
            if let Data::Cap { handle, .. } = &mut self.entries[i].data {
                let h = handle.take().expect("builder owns cap");
                if let Err(e) = message.try_push_handle(h) {
                    *handle = Some(e.handle);
                    let mut index = 0;
                    for prev in &mut self.entries[..i] {
                        if let Data::Cap { handle, .. } = &mut prev.data {
                            *handle = message.take_handle(index);
                            index += 1;
                        }
                    }
                    return Err(BuildError {
                        error: ProtocolError::Allocation,
                        builder: self,
                    });
                }
            }
        }
        Ok(message)
    }
}
