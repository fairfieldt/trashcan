use half::f16;
/// Q4 structure of arrays: all f32 scales, then aligned uvec4 packed blocks.
pub fn repack_q4(
    src: &[u8],
    k: usize,
    n: usize,
    row: usize,
    rows: usize,
    col: usize,
    cols: usize,
) -> Vec<u8> {
    assert!(
        row + rows <= n && col + cols <= k && col.is_multiple_of(32) && cols.is_multiple_of(32)
    );
    let count = rows * cols / 32;
    let mut out = vec![0u8; count * 20];
    let mut index = 0;
    for r in row..row + rows {
        for b in col / 32..(col + cols) / 32 {
            let at = (r * k / 32 + b) * 18;
            let d = f16::from_le_bytes(src[at..at + 2].try_into().unwrap()).to_f32();
            out[index * 4..index * 4 + 4].copy_from_slice(&d.to_le_bytes());
            out[count * 4 + index * 16..count * 4 + (index + 1) * 16]
                .copy_from_slice(&src[at + 2..at + 18]);
            index += 1;
        }
    }
    out
}

/// Q4_0 blocks as stored (f16 scale, 16 bytes of nibbles) split into all f16 scales, padded to 16 bytes, then
/// the 16-byte blocks: 18 bytes per 32 weights, the GGUF size (`repack_q4` widens scales to f32, 20 bytes).
/// Returns the bytes and the offset of the blocks in u32 words.
pub fn repack_q4h(src: &[u8]) -> (Vec<u8>, usize) {
    let count = src.len() / 18;
    let blocks_at = (count * 2).next_multiple_of(16);
    let mut out = vec![0u8; blocks_at + count * 16];
    for (i, b) in src.chunks_exact(18).enumerate() {
        out[i * 2..i * 2 + 2].copy_from_slice(&b[..2]);
        out[blocks_at + i * 16..blocks_at + (i + 1) * 16].copy_from_slice(&b[2..]);
    }
    (out, blocks_at / 4)
}

/// IQ4_XS superblocks (256 weights: f16 d, u16 high scale bits, 4 bytes of low scale bits, 128 bytes of nibbles)
/// split into all 8-byte headers, padded to 16 bytes, then the nibbles: 136 bytes per 256 weights, as stored.
/// The nibbles of each 32-weight sub-block keep the Q4_0 order. Returns the bytes and the blocks' word offset.
pub fn repack_iq4xs(src: &[u8]) -> (Vec<u8>, usize) {
    let count = src.len() / 136;
    let blocks_at = (count * 8).next_multiple_of(16);
    let mut out = vec![0u8; blocks_at + count * 128];
    for (i, b) in src.chunks_exact(136).enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&b[..8]);
        out[blocks_at + i * 128..blocks_at + (i + 1) * 128].copy_from_slice(&b[8..]);
    }
    (out, blocks_at / 4)
}

/// Q8_0 blocks split into all f16 scales, padded to 64 bytes (so pairs of blocks are whole cache lines), then the
/// 32-byte blocks: 34 bytes per 32 weights, as stored. Returns the bytes and the offset of the blocks in u32 words.
pub fn repack_q8h(src: &[u8]) -> (Vec<u8>, usize) {
    let count = src.len() / 34;
    let blocks_at = (count * 2).next_multiple_of(64);
    let mut out = vec![0u8; blocks_at + count * 32];
    for (i, b) in src.chunks_exact(34).enumerate() {
        out[i * 2..i * 2 + 2].copy_from_slice(&b[..2]);
        out[blocks_at + i * 32..blocks_at + (i + 1) * 32].copy_from_slice(&b[2..]);
    }
    (out, blocks_at / 4)
}

/// Q6 head: f32 (signed scale * block scale) per 16 elements, then signed bytes.
/// Larger than GGUF, but coalesced vector loads avoid byte-plane extraction in every head call.
pub fn repack_q6(src: &[u8]) -> Vec<u8> {
    let blocks = src.len() / 210;
    let count = blocks * 256;
    let mut out = vec![0u8; count * 5 / 4];
    for (b, s) in src.chunks_exact(210).enumerate() {
        let d = f16::from_le_bytes(s[208..210].try_into().unwrap()).to_f32();
        for j in 0..256 {
            let h = j / 128;
            let l = j % 32;
            let group = j % 128 / 32;
            let lo = s[h * 64 + l + group % 2 * 32] as u32;
            let hi = s[128 + h * 32 + l] as u32;
            let q = ((lo >> (group / 2 * 4)) & 15) | (((hi >> (group * 2)) & 3) << 4);
            out[count / 4 + b * 256 + j] = (q as i32 - 32) as i8 as u8;
            if j % 16 == 0 {
                let scale = d * (s[192 + h * 8 + l / 16 + group * 2] as i8) as f32;
                let at = (b * 16 + j / 16) * 4;
                out[at..at + 4].copy_from_slice(&scale.to_le_bytes());
            }
        }
    }
    out
}

pub fn q6_row(src: &[u8]) -> Vec<f32> {
    let mut out = vec![0.; src.len() / 210 * 256];
    for (b, s) in src.chunks_exact(210).enumerate() {
        let d = f16::from_le_bytes(s[208..210].try_into().unwrap()).to_f32();
        for j in 0..256 {
            let h = j / 128;
            let l = j % 32;
            let group = j % 128 / 32;
            let lo = s[h * 64 + l + group % 2 * 32] as u32;
            let hi = s[128 + h * 32 + l] as u32;
            let q = ((lo >> (group / 2 * 4)) & 15) | (((hi >> (group * 2)) & 3) << 4);
            let scale = s[192 + h * 8 + l / 16 + group * 2] as i8;
            out[b * 256 + j] = d * scale as f32 * (q as i32 - 32) as f32;
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn q6_signed_scales_and_planes() {
        let mut b = [0u8; 210];
        b[..128].fill(0x21);
        b[128..192].fill(0b11100100);
        b[192..208].fill((-2i8) as u8);
        b[208..].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        let v = q6_row(&b);
        assert_eq!((v[0], v[32], v[64], v[96]), (31., 15., -2., -18.));
        for (i, scale) in b[192..208].iter_mut().enumerate() {
            *scale = (i as i8 - 8) as u8;
        }
        for (i, q) in b[..192].iter_mut().enumerate() {
            *q = (i * 73 + 11) as u8;
        }
        let expected = q6_row(&b);
        let repacked = repack_q6(&b);
        for j in 0..256 {
            let scale =
                f32::from_le_bytes(repacked[j / 16 * 4..j / 16 * 4 + 4].try_into().unwrap());
            assert_eq!(scale * (repacked[64 + j] as i8) as f32, expected[j]);
        }
    }
    #[test]
    fn q4_shard() {
        let mut b = vec![0u8; 4 * 18];
        for i in 0..4 {
            b[i * 18..i * 18 + 2].copy_from_slice(&f16::from_f32(i as f32).to_le_bytes());
            b[i * 18 + 2..i * 18 + 18].fill(i as u8);
        }
        let r = repack_q4(&b, 64, 2, 1, 1, 32, 32);
        assert_eq!(r.len(), 20);
        assert_eq!(f32::from_le_bytes(r[..4].try_into().unwrap()), 3.);
        assert!(r[4..].iter().all(|x| *x == 3));
    }
}
