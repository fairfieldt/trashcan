//! UUIDs as Linux keeps them: 16 bytes in memory, printed in byte order (`%pUb`). Packets carry them as four dwords
//! read from those bytes little-endian (the struct layout on x86), which the control channel then sends big-endian.

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Uuid(pub [u8; 16]);

impl Uuid {
    /// `UUID_INIT(a, b, c, d0..d7)`: the first three fields big-endian.
    pub const fn init(a: u32, b: u16, c: u16, d: [u8; 8]) -> Self {
        let a = a.to_be_bytes();
        let b = b.to_be_bytes();
        let c = c.to_be_bytes();
        Uuid([a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]])
    }

    pub fn from_words(w: &[u32]) -> Self {
        let mut b = [0u8; 16];
        for (i, word) in w[..4].iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        Uuid(b)
    }

    pub fn words(&self) -> [u32; 4] {
        let mut w = [0u32; 4];
        for (i, word) in w.iter_mut().enumerate() {
            *word = u32::from_le_bytes(self.0[i * 4..i * 4 + 4].try_into().unwrap());
        }
        w
    }

    /// A random (version 4) UUID.
    pub fn random() -> Self {
        let mut b = [0u8; 16];
        // SAFETY: getentropy writes b.len() bytes into b.
        let n = unsafe { libc::getentropy(b.as_mut_ptr().cast(), b.len()) };
        assert_eq!(n, 0, "getentropy");
        b[6] = (b[6] & 0x0f) | 0x40;
        b[8] = (b[8] & 0x3f) | 0x80;
        Uuid(b)
    }
}

impl fmt::Display for Uuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, b) in self.0.iter().enumerate() {
            if matches!(i, 4 | 6 | 8 | 10) {
                f.write_str("-")?;
            }
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Uuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl FromStr for Uuid {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let hex: String = s.chars().filter(|&c| c != '-').collect();
        if hex.len() != 32 {
            return Err(format!("UUID {s:?}"));
        }
        let mut b = [0u8; 16];
        for (i, byte) in b.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(|_| format!("UUID {s:?}"))?;
        }
        Ok(Uuid(b))
    }
}

#[cfg(test)]
mod tests {
    use super::Uuid;

    #[test]
    fn init_prints_like_linux() {
        let xdp = Uuid::init(0xb638d70e, 0x42ff, 0x40bb, [0x97, 0xc2, 0x90, 0xe2, 0xc0, 0xb2, 0xff, 0x07]);
        assert_eq!(xdp.to_string(), "b638d70e-42ff-40bb-97c2-90e2c0b2ff07");
        assert_eq!(xdp.words()[0], 0x0ed7_38b6);
        assert_eq!(Uuid::from_words(&xdp.words()), xdp);
        assert_eq!("b638d70e-42ff-40bb-97c2-90e2c0b2ff07".parse::<Uuid>().unwrap(), xdp);
    }
}
