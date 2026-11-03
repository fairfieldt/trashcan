//! OETH: how a NIC driver and Oscar's network stack (netd) talk over a dross channel. Each message is one
//! datagram: "OETH", a u16 LE version (1) and a u16 LE opcode, then the body. The driver sends Info first, then
//! frames and link updates; the stack sends frames. Raw Ethernet frames of 14-1514 bytes, no FCS, no handles.

pub const MAGIC: &[u8; 4] = b"OETH";
pub const VERSION: u16 = 1;
pub const OP_INFO: u16 = 1;
pub const OP_FRAME: u16 = 2;
pub const OP_LINK: u16 = 3;
pub const MIN_FRAME: usize = 14;
pub const MAX_FRAME: usize = 1514;

fn header(op: u16, len: usize) -> Vec<u8> {
    let mut m = Vec::with_capacity(len);
    m.extend_from_slice(MAGIC);
    m.extend_from_slice(&VERSION.to_le_bytes());
    m.extend_from_slice(&op.to_le_bytes());
    m
}

/// Info (24 bytes): MAC at 8, IP MTU at 14, link at 16, speed in Mb/s at 20.
pub fn info(mac: &[u8; 6], mtu: u16, link: bool, mbps: u32) -> Vec<u8> {
    let mut m = header(OP_INFO, 24);
    m.extend_from_slice(mac);
    m.extend_from_slice(&mtu.to_le_bytes());
    m.push(link as u8);
    m.extend_from_slice(&[0; 3]);
    m.extend_from_slice(&mbps.to_le_bytes());
    m
}

pub fn frame(f: &[u8]) -> Vec<u8> {
    let mut m = header(OP_FRAME, 8 + f.len());
    m.extend_from_slice(f);
    m
}

/// Link update (16 bytes): link at 8, speed in Mb/s at 12.
pub fn link(up: bool, mbps: u32) -> Vec<u8> {
    let mut m = header(OP_LINK, 16);
    m.push(up as u8);
    m.extend_from_slice(&[0; 3]);
    m.extend_from_slice(&mbps.to_le_bytes());
    m
}

#[derive(Debug, PartialEq, Eq)]
pub enum Message<'a> {
    Info { mac: [u8; 6], mtu: u16, link: bool, mbps: u32 },
    Frame(&'a [u8]),
    Link { up: bool, mbps: u32 },
}

pub fn parse(m: &[u8]) -> Option<Message<'_>> {
    if m.len() < 8 || &m[..4] != MAGIC || u16::from_le_bytes([m[4], m[5]]) != VERSION {
        return None;
    }
    let u32_at = |o: usize| u32::from_le_bytes(m[o..o + 4].try_into().unwrap());
    match u16::from_le_bytes([m[6], m[7]]) {
        OP_INFO if m.len() == 24 => Some(Message::Info {
            mac: m[8..14].try_into().unwrap(),
            mtu: u16::from_le_bytes([m[14], m[15]]),
            link: m[16] != 0,
            mbps: u32_at(20),
        }),
        OP_FRAME if (8 + MIN_FRAME..=8 + MAX_FRAME).contains(&m.len()) => Some(Message::Frame(&m[8..])),
        OP_LINK if m.len() == 16 => Some(Message::Link { up: m[8] != 0, mbps: u32_at(12) }),
        _ => None,
    }
}

/// An ARP request (who has `target`, tell `ip`) from `mac`, as a broadcast frame.
pub fn arp_request(mac: &[u8; 6], ip: [u8; 4], target: [u8; 4]) -> Vec<u8> {
    let mut f = Vec::with_capacity(60);
    f.extend_from_slice(&[0xff; 6]);
    f.extend_from_slice(mac);
    f.extend_from_slice(&[0x08, 0x06, 0x00, 0x01, 0x08, 0x00, 6, 4, 0x00, 0x01]);
    f.extend_from_slice(mac);
    f.extend_from_slice(&ip);
    f.extend_from_slice(&[0; 6]);
    f.extend_from_slice(&target);
    f.resize(60, 0);
    f
}

/// The sender's MAC if `f` is an ARP reply from `target`.
pub fn arp_reply_from(f: &[u8], target: [u8; 4]) -> Option<[u8; 6]> {
    if f.len() < 42 || f[12..14] != [0x08, 0x06] || f[20..22] != [0x00, 0x02] || f[28..32] != target {
        return None;
    }
    Some(f[22..28].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_round_trip_at_the_contract_offsets() {
        let mac = [0, 0x3e, 0xe1, 0xc2, 0x50, 0xf8];
        let i = info(&mac, 1500, true, 1000);
        assert_eq!(i.len(), 24);
        assert_eq!(&i[..8], b"OETH\x01\x00\x01\x00");
        assert_eq!(&i[8..14], &mac);
        assert_eq!(&i[14..16], &1500u16.to_le_bytes());
        assert_eq!(i[16], 1);
        assert_eq!(&i[17..20], &[0, 0, 0]);
        assert_eq!(&i[20..24], &1000u32.to_le_bytes());
        assert_eq!(parse(&i), Some(Message::Info { mac, mtu: 1500, link: true, mbps: 1000 }));
        let l = link(false, 0);
        assert_eq!(l.len(), 16);
        assert_eq!(parse(&l), Some(Message::Link { up: false, mbps: 0 }));
        let a = arp_request(&mac, [192, 168, 0, 66], [192, 168, 0, 1]);
        let f = frame(&a);
        assert_eq!(parse(&f), Some(Message::Frame(&a[..])));
        assert_eq!(parse(&frame(&[0; 13])), None);
        assert_eq!(parse(&frame(&[0; 1515])), None);
        assert_eq!(parse(b"OETH\x02\x00\x02\x00"), None);
        // A reply: the request with the opcode, sender and target swapped.
        let mut r = a.clone();
        r[20..22].copy_from_slice(&[0, 2]);
        r[22..28].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
        r[28..32].copy_from_slice(&[192, 168, 0, 1]);
        assert_eq!(arp_reply_from(&r, [192, 168, 0, 1]), Some([1, 2, 3, 4, 5, 6]));
        assert_eq!(arp_reply_from(&a, [192, 168, 0, 1]), None);
    }
}
