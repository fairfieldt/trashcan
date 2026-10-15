//! Records, the unit of the link protocol on every path. A record is a 48-byte header and up to `STREAM_PAYLOAD`
//! (USB4STREAM) or `TCP_PAYLOAD` bytes. On a stream each record is exactly one frame: the driver cuts a write into
//! 4 KiB frames, so records are written whole and every one but the last of a write fills its frame, and a lost or
//! damaged frame costs whole records. `read` returns the frames' bytes back to back, so a reader finds the records
//! by their headers and, after damage, by searching for the next header that checks out.

/// "TLNK" in the first four bytes.
pub const MAGIC: u32 = 0x4b4e_4c54;
pub const HEADER: usize = 48;
/// USB4STREAM frame size (`TB_MAX_FRAME_SIZE`).
pub const FRAME: usize = 4096;
pub const STREAM_PAYLOAD: usize = FRAME - HEADER;
pub const TCP_PAYLOAD: usize = 60 << 10;

/// Record kinds. DATA, ACK and NAK carry the byte streams; PROBE opens a stream epoch; the rest travel over TCP.
pub mod kind {
    pub const DATA: u8 = 1;
    /// Acknowledgement and credits only; with `PING` the other side answers at once.
    pub const ACK: u8 = 2;
    /// "Go back": `seq` is the next byte the sender of the NAK expects on the stream.
    pub const NAK: u8 = 3;
    pub const PROBE: u8 = 4;
    /// The connecting side opens stream epoch `epoch`.
    pub const START: u8 = 5;
    /// Stream epoch `epoch` failed on the sending side; close it.
    pub const FAIL: u8 = 6;
    /// The sending side closes the link.
    pub const BYE: u8 = 7;
    pub const HELLO: u8 = 8;
    pub const WELCOME: u8 = 9;
    /// The sending side's receive buffer is full: it stops reading until `OPEN`.
    pub const FULL: u8 = 10;
    pub const OPEN: u8 = 11;
    /// A further TCP connection of an existing link (another cable to the same machine), as its backup path.
    pub const JOIN: u8 = 12;
}

/// Flag: answer with an ACK at once (keepalive).
pub const PING: u8 = 1;
/// Flag of an ACK over TCP while a stream epoch opens: `epoch`, and the sender's frames of the epoch read (`rx`) and
/// written (`tx`), so that the other side can tell a direction that carries nothing from a peer that is only slow.
pub const REPORT: u8 = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Header {
    pub kind: u8,
    pub flags: u8,
    pub len: u16,
    /// Low 32 bits of the link's session id.
    pub session: u32,
    /// Stream epoch (0 on TCP).
    pub epoch: u32,
    /// Byte offset of the payload in the sender's byte stream (DATA), or the offset a NAK asks for.
    pub seq: u64,
    /// Every byte of the other side's stream before `ack` has arrived in order.
    pub ack: u64,
    /// Stream frames: this frame's number in the epoch.
    pub tx: u32,
    /// Stream frames: one more than the number of the last frame read from the sender's receive ring in this
    /// epoch, so every frame numbered below it has left that ring (read, or lost before it got there).
    pub rx: u32,
    /// CRC32C of the payload (0 without one).
    pub pcrc: u32,
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// CRC32C (Castagnoli) of `data`. On x86-64 with SSE4.2 (the Mac Pros) the crc32 instruction runs inline in `hw`:
/// the crc32c crate reaches it through a call per 8 bytes unless the whole build enables SSE4.2, 2.5-3.7 GB/s on the
/// Mac Pros (results/kernel/usb4stream-reliable-20261013).
pub fn crc(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if hw::on() {
        // SAFETY: the CPU has SSE4.2.
        return unsafe { hw::crc1(data) };
    }
    crc32c::crc32c(data)
}

/// The CRCs of `parts` into `out`, three at a time where three in a row have the same length (full stream records):
/// the instruction takes three cycles and starts one per cycle, so three chains keep it busy.
pub fn crcs(parts: &[&[u8]], out: &mut Vec<u32>) {
    out.clear();
    #[cfg(target_arch = "x86_64")]
    if hw::on() {
        let mut i = 0;
        while i < parts.len() {
            if i + 3 <= parts.len() && parts[i].len() == parts[i + 1].len() && parts[i].len() == parts[i + 2].len() {
                // SAFETY: the CPU has SSE4.2.
                out.extend_from_slice(&unsafe { hw::crc3(parts[i], parts[i + 1], parts[i + 2]) });
                i += 3;
            } else {
                // SAFETY: as above.
                out.push(unsafe { hw::crc1(parts[i]) });
                i += 1;
            }
        }
        return;
    }
    out.extend(parts.iter().map(|p| crc32c::crc32c(p)));
}

#[cfg(target_arch = "x86_64")]
mod hw {
    use std::arch::x86_64::{_mm_crc32_u64, _mm_crc32_u8};

    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::arch::is_x86_feature_detected!("sse4.2"))
    }

    fn word(w: &[u8]) -> u64 {
        u64::from_le_bytes(w.try_into().unwrap())
    }

    #[target_feature(enable = "sse4.2")]
    pub unsafe fn crc1(data: &[u8]) -> u32 {
        let mut c = u64::from(u32::MAX);
        let words = data.chunks_exact(8);
        let tail = words.remainder();
        for w in words {
            c = _mm_crc32_u64(c, word(w));
        }
        let mut c = c as u32;
        for &b in tail {
            c = _mm_crc32_u8(c, b);
        }
        !c
    }

    /// Three buffers of the same length.
    #[target_feature(enable = "sse4.2")]
    pub unsafe fn crc3(a: &[u8], b: &[u8], c: &[u8]) -> [u32; 3] {
        let (mut x, mut y, mut z) = (u64::from(u32::MAX), u64::from(u32::MAX), u64::from(u32::MAX));
        let (wa, wb, wc) = (a.chunks_exact(8), b.chunks_exact(8), c.chunks_exact(8));
        let (ta, tb, tc) = (wa.remainder(), wb.remainder(), wc.remainder());
        for ((p, q), r) in wa.zip(wb).zip(wc) {
            x = _mm_crc32_u64(x, word(p));
            y = _mm_crc32_u64(y, word(q));
            z = _mm_crc32_u64(z, word(r));
        }
        let (mut x, mut y, mut z) = (x as u32, y as u32, z as u32);
        for ((&p, &q), &r) in ta.iter().zip(tb).zip(tc) {
            x = _mm_crc32_u8(x, p);
            y = _mm_crc32_u8(y, q);
            z = _mm_crc32_u8(z, r);
        }
        [!x, !y, !z]
    }
}

impl Header {
    pub fn encode(&self) -> [u8; HEADER] {
        let mut b = [0u8; HEADER];
        b[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        b[4] = self.kind;
        b[5] = self.flags;
        b[6..8].copy_from_slice(&self.len.to_le_bytes());
        b[8..12].copy_from_slice(&self.session.to_le_bytes());
        b[12..16].copy_from_slice(&self.epoch.to_le_bytes());
        b[16..24].copy_from_slice(&self.seq.to_le_bytes());
        b[24..32].copy_from_slice(&self.ack.to_le_bytes());
        b[32..36].copy_from_slice(&self.tx.to_le_bytes());
        b[36..40].copy_from_slice(&self.rx.to_le_bytes());
        b[40..44].copy_from_slice(&self.pcrc.to_le_bytes());
        let h = crc(&b[..44]);
        b[44..48].copy_from_slice(&h.to_le_bytes());
        b
    }

    /// The header at the start of `b`, if its magic and header CRC check out.
    pub fn decode(b: &[u8]) -> Option<Header> {
        if b.len() < HEADER || u32_at(b, 0) != MAGIC || crc(&b[..44]) != u32_at(b, 44) {
            return None;
        }
        Some(Header {
            kind: b[4],
            flags: b[5],
            len: u16::from_le_bytes([b[6], b[7]]),
            session: u32_at(b, 8),
            epoch: u32_at(b, 12),
            seq: u64_at(b, 16),
            ack: u64_at(b, 24),
            tx: u32_at(b, 32),
            rx: u32_at(b, 36),
            pcrc: u32_at(b, 40),
        })
    }
}

/// Offset of the next possible header start in `b` (the magic's first byte), or the length of `b` minus the bytes
/// that could still begin one.
pub fn find_magic(b: &[u8]) -> usize {
    let m = MAGIC.to_le_bytes();
    let mut at = 0;
    while at < b.len() {
        match b[at..].iter().position(|&x| x == m[0]) {
            None => return b.len(),
            Some(i) => at += i,
        }
        let rest = &b[at..];
        let n = rest.len().min(4);
        if rest[..n] == m[..n] {
            return at;
        }
        at += 1;
    }
    b.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip_and_damage() {
        let h = Header { kind: kind::DATA, flags: PING, len: 4048, session: 7, epoch: 3, seq: 1 << 40, ack: 99, tx: 5, rx: 4, pcrc: 0xdead_beef };
        let b = h.encode();
        assert_eq!(Header::decode(&b), Some(h));
        for i in 0..HEADER {
            let mut d = b;
            d[i] ^= 0x10;
            assert_eq!(Header::decode(&d), None, "byte {i}");
        }
        assert_eq!(Header::decode(&b[..HEADER - 1]), None);
    }

    #[test]
    fn crcs_are_crc32c() {
        let data: Vec<u8> = (0..20000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        for len in [0, 1, 7, 8, 9, 63, 4048, 4049, 12345] {
            for off in [0, 1, 3, 8] {
                assert_eq!(crc(&data[off..off + len]), crc32c::crc32c(&data[off..off + len]), "{len} at {off}");
            }
        }
        let parts: Vec<&[u8]> = [(0, 4048), (4048, 4048), (8096, 4048), (1, 4048), (5, 100), (9, 100), (13, 100), (0, 0), (77, 3)]
            .iter()
            .map(|&(a, n)| &data[a..a + n])
            .collect();
        let mut out = Vec::new();
        crcs(&parts, &mut out);
        assert_eq!(out, parts.iter().map(|p| crc32c::crc32c(p)).collect::<Vec<_>>());
    }

    #[test]
    fn magic_search() {
        let m = MAGIC.to_le_bytes();
        assert_eq!(find_magic(&[]), 0);
        assert_eq!(find_magic(&[1, 2, 3]), 3);
        assert_eq!(find_magic(&[1, m[0], m[1], m[2], m[3], 9]), 1);
        assert_eq!(find_magic(&[m[0], 5, m[0], m[1], m[2], m[3]]), 2);
        // A possible start at the end is kept for the next read.
        assert_eq!(find_magic(&[7, 7, m[0], m[1]]), 2);
    }
}
