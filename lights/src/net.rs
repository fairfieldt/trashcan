//! Frames from the conductor to the nodes: UDP, one datagram per frame.
//!
//! "PLT1", a little-endian u32 sequence number, then one byte per LED (linear brightness 0-255).

use crate::panel::{COUNT, Frame};

pub const PORT: u16 = 7465;
const MAGIC: &[u8; 4] = b"PLT1";
pub const SIZE: usize = 8 + COUNT;

pub fn encode(seq: u32, frame: &Frame) -> [u8; SIZE] {
    let mut out = [0u8; SIZE];
    out[..4].copy_from_slice(MAGIC);
    out[4..8].copy_from_slice(&seq.to_le_bytes());
    for (o, v) in out[8..].iter_mut().zip(frame) {
        *o = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    out
}

pub fn decode(packet: &[u8]) -> Option<(u32, Frame)> {
    if packet.len() != SIZE || &packet[..4] != MAGIC {
        return None;
    }
    let seq = u32::from_le_bytes(packet[4..8].try_into().ok()?);
    let mut frame = [0.0; COUNT];
    for (f, &b) in frame.iter_mut().zip(&packet[8..]) {
        *f = b as f32 / 255.0;
    }
    Some((seq, frame))
}

/// Whether `seq` is newer than `last` (wrapping). A restarted conductor starts over; the node accepts it once
/// the old stream has been quiet for a second.
pub fn newer(seq: u32, last: u32) -> bool {
    let d = seq.wrapping_sub(last);
    d != 0 && d < u32::MAX / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut f = [0.0; COUNT];
        f[3] = 1.0;
        f[7] = 0.5;
        let (seq, g) = decode(&encode(9, &f)).unwrap();
        assert_eq!(seq, 9);
        assert_eq!(g[3], 1.0);
        assert!((g[7] - 0.5).abs() < 0.01);
        assert!(decode(b"nope").is_none());
    }

    #[test]
    fn order() {
        assert!(newer(5, 4));
        assert!(!newer(4, 5));
        assert!(newer(0, u32::MAX));
        assert!(!newer(1, 5000));
    }
}
