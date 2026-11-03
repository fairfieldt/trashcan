//! The network log: kernel log records (/dev/kmsg) sent as UDP datagrams (broadcast by default) to the
//! netconsole port, so a machine
//! without a console (dross on a Mac Pro, which has no serial port) can be watched from the LAN with
//! `netlog-listen.py` (next to this crate) or `nc -kul 6666`.

/// One record of /dev/kmsg ("PRI,SEQ,USEC,FLAGS;MESSAGE\n", continuation lines start with a space): its message.
pub fn kmsg_message(record: &[u8]) -> Option<&[u8]> {
    let line = record.split(|&b| b == b'\n').next()?;
    let semi = line.iter().position(|&b| b == b';')?;
    Some(&line[semi + 1..])
}

fn checksum(header: &[u8]) -> u16 {
    let mut sum = 0u32;
    for c in header.chunks(2) {
        sum += u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)]) as u32;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

/// An Ethernet frame with an IPv4 UDP datagram from `src_ip:port` to `dst_ip:port` at `dst_mac` (broadcast:
/// 255.255.255.255 and ff:ff:ff:ff:ff:ff; macOS drops broadcasts from 0.0.0.0, so give a real source then).
pub fn udp_frame(
    src_mac: &[u8; 6],
    src_ip: [u8; 4],
    dst_mac: &[u8; 6],
    dst_ip: [u8; 4],
    port: u16,
    payload: &[u8],
    id: u16,
) -> Vec<u8> {
    let mut f = Vec::with_capacity(42 + payload.len());
    f.extend_from_slice(dst_mac);
    f.extend_from_slice(src_mac);
    f.extend_from_slice(&[0x08, 0x00]);
    let total = (20 + 8 + payload.len()) as u16;
    let mut ip = [0u8; 20];
    ip[0] = 0x45;
    ip[2..4].copy_from_slice(&total.to_be_bytes());
    ip[4..6].copy_from_slice(&id.to_be_bytes());
    ip[8] = 64;
    ip[9] = 17;
    ip[12..16].copy_from_slice(&src_ip);
    ip[16..20].copy_from_slice(&dst_ip);
    let c = checksum(&ip);
    ip[10..12].copy_from_slice(&c.to_be_bytes());
    f.extend_from_slice(&ip);
    f.extend_from_slice(&port.to_be_bytes());
    f.extend_from_slice(&port.to_be_bytes());
    f.extend_from_slice(&((8 + payload.len()) as u16).to_be_bytes());
    // No UDP checksum (allowed for IPv4).
    f.extend_from_slice(&[0, 0]);
    f.extend_from_slice(payload);
    while f.len() < 60 {
        f.push(0);
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_and_records() {
        let f =
            udp_frame(&[0, 0x3e, 0xe1, 0xc2, 0x50, 0xf8], [192, 168, 0, 66], &[0xff; 6], [255; 4], 6666, b"hello\n", 7);
        assert_eq!(&f[..6], &[0xff; 6]);
        assert_eq!(&f[12..14], &[0x08, 0x00]);
        // The header with its checksum sums to 0xffff.
        assert_eq!(checksum(&f[14..34]), 0);
        assert_eq!(u16::from_be_bytes([f[16], f[17]]), 20 + 8 + 6);
        assert_eq!(&f[36..38], &6666u16.to_be_bytes());
        assert_eq!(&f[42..48], b"hello\n");
        assert_eq!(f.len(), 60);
        assert_eq!(kmsg_message(b"6,12,345678,-;dross 0.1.0\n"), Some(b"dross 0.1.0".as_slice()));
        assert_eq!(kmsg_message(b"no separator"), None);
    }
}
