//! Versioned native PCM messages. A channel endpoint grants one exclusive playback session.
#![no_std]
extern crate alloc;

pub const RATE: u32 = 48_000;
pub const CHANNELS: u32 = 2;
pub const FRAME_BYTES: u32 = 4;
pub const MAX_FRAMES: u32 = RATE;
pub const VERSION: u32 = 1;
pub const PLAY: u32 = 1;
pub const CLOSE: u32 = 2;
pub const INFO: u32 = 3;
pub const MESSAGE_BYTES: usize = 32;

/// PLAY moves exactly one Memory handle, containing signed16 LE interleaved stereo at byte0.
/// The client must finish writing before sending. The server snapshots before starting DMA.
/// Reply means playback has stopped, including on errors; received Memory is always closed.
/// One outstanding request per endpoint. All deadlines are absolute monotonic nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub op: u32,
    pub id: u64,
    pub deadline: u64,
    pub frames: u32,
}
impl Request {
    pub fn encode(self) -> [u8; MESSAGE_BYTES] {
        let mut out = [0; MESSAGE_BYTES];
        out[0..4].copy_from_slice(&VERSION.to_le_bytes());
        out[4..8].copy_from_slice(&self.op.to_le_bytes());
        out[8..16].copy_from_slice(&self.id.to_le_bytes());
        out[16..24].copy_from_slice(&self.deadline.to_le_bytes());
        out[24..28].copy_from_slice(&self.frames.to_le_bytes());
        out
    }
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != MESSAGE_BYTES
            || u32::from_le_bytes(bytes[0..4].try_into().ok()?) != VERSION
            || bytes[28..32] != [0; 4]
        {
            return None;
        }
        let result = Self {
            op: u32::from_le_bytes(bytes[4..8].try_into().ok()?),
            id: u64::from_le_bytes(bytes[8..16].try_into().ok()?),
            deadline: u64::from_le_bytes(bytes[16..24].try_into().ok()?),
            frames: u32::from_le_bytes(bytes[24..28].try_into().ok()?),
        };
        if !matches!(result.op, PLAY | CLOSE | INFO)
            || (result.op == PLAY && !(1..=MAX_FRAMES).contains(&result.frames))
            || (result.op != PLAY && result.frames != 0)
        {
            return None;
        }
        Some(result)
    }
}

/// Reply: version:u32, status:i32 (native Error or0), request id:u64,
/// played frames:u32, sample rate:u32, channels:u32, sample bits:u32.
pub fn reply(id: u64, status: i32, frames: u32) -> [u8; MESSAGE_BYTES] {
    let mut bytes = [0; MESSAGE_BYTES];
    bytes[0..4].copy_from_slice(&VERSION.to_le_bytes());
    bytes[4..8].copy_from_slice(&status.to_le_bytes());
    bytes[8..16].copy_from_slice(&id.to_le_bytes());
    bytes[16..20].copy_from_slice(&frames.to_le_bytes());
    bytes[20..24].copy_from_slice(&RATE.to_le_bytes());
    bytes[24..28].copy_from_slice(&CHANNELS.to_le_bytes());
    bytes[28..32].copy_from_slice(&16u32.to_le_bytes());
    bytes
}

/// Connection list expansion from HDA short/long form, including range entries.
/// Reject malformed ranges, duplicate/cyclic routes are handled by the route search.
pub fn connections(raw: &[u16], long: bool) -> Option<alloc::vec::Vec<u8>> {
    let mut out = alloc::vec::Vec::new();
    let flag = if long { 0x8000 } else { 0x80 };
    for &item in raw {
        let nid = item & !flag;
        if nid > 255 || nid == 0 {
            return None;
        }
        if item & flag != 0 {
            let &last = out.last()?;
            if nid <= last as u16 {
                return None;
            }
            for n in last + 1..=nid as u8 {
                out.push(n);
            }
        } else {
            out.push(nid as u8);
        }
        if out.len() > 256 {
            return None;
        }
    }
    Some(out)
}

#[cfg(feature = "native")]
pub mod hda;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn messages_are_strict_and_bounded() {
        let r = Request {
            op: PLAY,
            id: 42,
            deadline: 123,
            frames: MAX_FRAMES,
        };
        assert_eq!(Request::decode(&r.encode()), Some(r));
        let mut b = r.encode();
        b[31] = 1;
        assert_eq!(Request::decode(&b), None);
        assert_eq!(Request::decode(&b[..31]), None);
        assert_eq!(
            Request::decode(
                &Request {
                    frames: MAX_FRAMES + 1,
                    ..r
                }
                .encode()
            ),
            None
        );
        assert_eq!(Request::decode(&Request { frames: 0, ..r }.encode()), None);
    }
    #[test]
    fn connection_ranges_and_bad_tables() {
        assert_eq!(connections(&[2, 0x85, 9], false).unwrap(), [2, 3, 4, 5, 9]);
        assert_eq!(connections(&[2, 0x8004], true).unwrap(), [2, 3, 4]);
        assert!(connections(&[0x85], false).is_none());
        assert!(connections(&[5, 0x83], false).is_none());
        assert!(connections(&[256], true).is_none());
    }
}
