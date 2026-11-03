//! Field-by-field little-endian service framing. Never copies Rust layouts.
use super::schema;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    Truncated,
    Length,
    Handles,
    Kind,
    Correlation,
    Reserved,
    Identifier,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub protocol: u32,
    pub major: u16,
    pub kind: u16,
    pub ordinal: u32,
    pub body_bytes: u32,
    pub request_id: u64,
    pub handle_count: u16,
    pub flags: u16,
}
pub struct Frame<'a> {
    pub header: Header,
    pub body: &'a [u8],
}
pub(crate) fn get16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
pub(crate) fn get32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
pub(crate) fn get64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
impl Header {
    pub fn new(
        protocol: u32,
        major: u16,
        kind: u16,
        ordinal: u32,
        request_id: u64,
        body_bytes: usize,
        handles: usize,
    ) -> Result<Self, FrameError> {
        if body_bytes > schema::MAX_MESSAGE_BYTES - schema::MAX_HEADER_BYTES {
            return Err(FrameError::Length);
        }
        if handles > schema::MAX_HANDLES {
            return Err(FrameError::Handles);
        }
        let h = Self {
            protocol,
            major,
            kind,
            ordinal,
            request_id,
            body_bytes: body_bytes as u32,
            handle_count: handles as u16,
            flags: 0,
        };
        h.validate()?;
        Ok(h)
    }
    fn validate(&self) -> Result<(), FrameError> {
        if self.protocol == 0 || self.major == 0 {
            return Err(FrameError::Identifier);
        }
        if ![schema::KIND_REQUEST, schema::KIND_REPLY, schema::KIND_EVENT].contains(&self.kind) {
            return Err(FrameError::Kind);
        }
        if (self.kind == schema::KIND_EVENT) != (self.request_id == 0) {
            return Err(FrameError::Correlation);
        }
        if self.handle_count as usize > schema::MAX_HANDLES {
            return Err(FrameError::Handles);
        }
        if self.body_bytes as usize > schema::MAX_MESSAGE_BYTES - schema::MAX_HEADER_BYTES {
            return Err(FrameError::Length);
        }
        Ok(())
    }
    pub fn supported(&self, protocol: u32, major: u16) -> bool {
        self.protocol == protocol && self.major == major && self.flags == 0
    }
    pub fn encode(&self, out: &mut [u8]) -> Result<(), FrameError> {
        self.validate()?;
        if out.len() < 32 {
            return Err(FrameError::Truncated);
        }
        out[..32].fill(0);
        out[0..4].copy_from_slice(&self.protocol.to_le_bytes());
        out[4..6].copy_from_slice(&self.major.to_le_bytes());
        out[6..8].copy_from_slice(&self.kind.to_le_bytes());
        out[8..12].copy_from_slice(&self.ordinal.to_le_bytes());
        out[12..16].copy_from_slice(&self.body_bytes.to_le_bytes());
        out[16..24].copy_from_slice(&self.request_id.to_le_bytes());
        out[24..26].copy_from_slice(&self.handle_count.to_le_bytes());
        out[26..28].copy_from_slice(&self.flags.to_le_bytes());
        Ok(())
    }
}
impl<'a> Frame<'a> {
    /// Structural validation retains unknown flags/major for a safe Unsupported
    /// reply. Call `supported` before interpreting any operation body.
    pub fn parse(bytes: &'a [u8], handles: usize) -> Result<Self, FrameError> {
        if bytes.len() < 32 {
            return Err(FrameError::Truncated);
        }
        let h = Header {
            protocol: get32(bytes, 0),
            major: get16(bytes, 4),
            kind: get16(bytes, 6),
            ordinal: get32(bytes, 8),
            body_bytes: get32(bytes, 12),
            request_id: get64(bytes, 16),
            handle_count: get16(bytes, 24),
            flags: get16(bytes, 26),
        };
        h.validate()?;
        if get32(bytes, 28) != 0 {
            return Err(FrameError::Reserved);
        }
        if h.handle_count as usize != handles {
            return Err(FrameError::Handles);
        }
        if bytes.len() != 32 + h.body_bytes as usize {
            return Err(FrameError::Length);
        }
        Ok(Self {
            header: h,
            body: &bytes[32..],
        })
    }
}
