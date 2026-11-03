//! Capability-scoped Ethernet frame transport. One datagram per packet, no handles.
#![no_std]
extern crate alloc;
use alloc::vec::Vec;
pub const HEADER: usize = 8;
pub const MAX_FRAME: usize = 1514;
pub const MAX_PACKET: usize = HEADER + MAX_FRAME;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    pub up: bool,
    pub mbps: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Info {
    pub mac: [u8; 6],
    pub mtu: u16,
    pub link: Link,
}
#[derive(Debug, PartialEq, Eq)]
pub enum Packet<'a> {
    Info(Info),
    Frame(&'a [u8]),
    Link(Link),
}
#[derive(Debug, PartialEq, Eq)]
pub struct Invalid;
fn header(op: u16) -> Vec<u8> {
    let mut out = Vec::from(*b"OETH");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&op.to_le_bytes());
    out
}
pub fn encode(packet: Packet<'_>) -> Result<Vec<u8>, Invalid> {
    Ok(match packet {
        Packet::Info(info) => {
            if info.mtu != 1500 {
                return Err(Invalid);
            }
            let mut out = header(1);
            out.extend_from_slice(&info.mac);
            out.extend_from_slice(&info.mtu.to_le_bytes());
            out.push(u8::from(info.link.up));
            out.extend_from_slice(&[0; 3]);
            out.extend_from_slice(&info.link.mbps.to_le_bytes());
            out
        }
        Packet::Frame(frame) => {
            if !(14..=MAX_FRAME).contains(&frame.len()) {
                return Err(Invalid);
            }
            let mut out = header(2);
            out.extend_from_slice(frame);
            out
        }
        Packet::Link(link) => {
            let mut out = header(3);
            out.push(u8::from(link.up));
            out.extend_from_slice(&[0; 3]);
            out.extend_from_slice(&link.mbps.to_le_bytes());
            out
        }
    })
}
pub fn decode(bytes: &[u8]) -> Result<Packet<'_>, Invalid> {
    if bytes.len() < HEADER || bytes[..4] != *b"OETH" || bytes[4..6] != [1, 0] {
        return Err(Invalid);
    }
    match (
        u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
        bytes.len(),
    ) {
        (1, 24) if bytes[16] <= 1 && bytes[17..20] == [0; 3] => {
            let mtu = u16::from_le_bytes(bytes[14..16].try_into().unwrap());
            if mtu != 1500 {
                return Err(Invalid);
            }
            Ok(Packet::Info(Info {
                mac: bytes[8..14].try_into().unwrap(),
                mtu,
                link: Link {
                    up: bytes[16] != 0,
                    mbps: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
                },
            }))
        }
        (2, n) if (HEADER + 14..=MAX_PACKET).contains(&n) => Ok(Packet::Frame(&bytes[8..])),
        (3, 16) if bytes[8] <= 1 && bytes[9..12] == [0; 3] => Ok(Packet::Link(Link {
            up: bytes[8] != 0,
            mbps: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        })),
        _ => Err(Invalid),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip_and_truncation() {
        let info = Info {
            mac: [2, 1, 2, 3, 4, 5],
            mtu: 1500,
            link: Link {
                up: true,
                mbps: 1000,
            },
        };
        let wire = encode(Packet::Info(info)).unwrap();
        assert_eq!(decode(&wire), Ok(Packet::Info(info)));
        for n in 0..wire.len() {
            assert_eq!(decode(&wire[..n]), Err(Invalid));
        }
        let wire = encode(Packet::Frame(&[0; 1514])).unwrap();
        assert!(matches!(decode(&wire),Ok(Packet::Frame(f))if f.len()==1514));
    }
    #[test]
    fn rejects_reserved_flags_versions_and_lengths() {
        let mut wire = encode(Packet::Link(Link { up: false, mbps: 0 })).unwrap();
        wire[8] = 2;
        assert_eq!(decode(&wire), Err(Invalid));
        wire[8] = 0;
        wire[9] = 1;
        assert_eq!(decode(&wire), Err(Invalid));
        wire[9] = 0;
        wire[4] = 2;
        assert_eq!(decode(&wire), Err(Invalid));
        assert_eq!(encode(Packet::Frame(&[0; 13])), Err(Invalid));
        assert_eq!(encode(Packet::Frame(&[0; 1515])), Err(Invalid));
    }
}
