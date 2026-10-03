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
