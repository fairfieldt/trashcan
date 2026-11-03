//! CRC-32C (Castagnoli), the checksum at the end of every control channel frame.

const POLY: u32 = 0x82f6_3b78; // reflected 0x1edc6f41

const TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// The standard CRC-32C of `data` (initial value and final XOR all ones). Linux's control channel sends
/// `cpu_to_be32(~crc32c(~0, data))`, which is this value, big-endian.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c = TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

#[cfg(test)]
mod tests {
    #[test]
    fn check_value() {
        assert_eq!(super::crc32c(b"123456789"), 0xe306_9283);
        assert_eq!(super::crc32c(b""), 0);
    }
}
