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

/// The two planes of a matrix repacked for the GPU: per GGUF block a head (its scale or header) in the first plane,
/// padded with zeros to `Planes::body_at` bytes, then a body (its quants) in the second. `Kind::fill` repacks any run
/// of whole blocks, so a matrix can be written in pieces and in parallel, straight into staging memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Q4_0 with f32 scales (the Gemma kernels; `repack_q4` of a whole matrix): 20 bytes per 32 weights.
    Q4,
    /// Q4_0 or IQ4_NL with the f16 scales as stored: 18 bytes per 32 weights, the GGUF size.
    Q4h,
    /// Q8_0 with f16 scales: 34 bytes per 32 weights, as stored.
    Q8h,
    /// IQ4_XS superblocks (256 weights: f16 d, u16 high scale bits, 4 bytes of low scale bits, 128 bytes of
    /// nibbles): the 8-byte headers, then the nibbles, 136 bytes per 256 weights as stored. The nibbles of each
    /// 32-weight sub-block keep the Q4_0 order.
    Iq4xs,
    /// Q6_K head: an f32 (signed scale * block scale) per 16 weights, then signed bytes. Larger than the GGUF, but
    /// coalesced vector loads avoid byte-plane extraction in every head call.
    Q6,
}

/// The integer values of the NVFP4 nibbles (twice E2M1), multiplied by the halved UE4M3 scales.
pub const NVFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// A UE4M3 scale as ggml reads it (`ggml_ue4m3_to_fp32`): the sign bit ignored, 0x00 and 0x7f (NaN) as zero, halved
/// to match `NVFP4_VALUES`. Every result is exact in f16.
pub fn ue4m3_half(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.;
    }
    let (e, m) = ((x >> 3) & 15, (x & 7) as f32);
    let raw = if e == 0 { m * (2f32).powi(-9) } else { (1. + m / 8.) * (2f32).powi(e as i32 - 7) };
    raw * 0.5
}

/// One ggml NVFP4 block of 64 weights, dequantized as ggml does (`dequantize_row_nvfp4`).
pub fn nvfp4_block(b: &[u8; 36]) -> [f32; 64] {
    let mut out = [0f32; 64];
    for s in 0..4 {
        let d = ue4m3_half(b[s]);
        for j in 0..8 {
            let q = b[4 + s * 8 + j];
            out[s * 16 + j] = NVFP4_VALUES[(q & 15) as usize] as f32 * d;
            out[s * 16 + j + 8] = NVFP4_VALUES[(q >> 4) as usize] as f32 * d;
        }
    }
    out
}

impl Kind {
    /// Bytes per block in the GGUF, in the head plane and in the body plane.
    pub const fn sizes(self) -> (usize, usize, usize) {
        match self {
            Kind::Q4 => (18, 4, 16),
            Kind::Q4h => (18, 2, 16),
            Kind::Q8h => (34, 2, 32),
            Kind::Iq4xs => (136, 8, 128),
            Kind::Q6 => (210, 64, 256),
        }
    }

    /// Repack the whole blocks `src` into their heads and bodies.
    pub fn fill(self, src: &[u8], heads: &mut [u8], bodies: &mut [u8]) {
        match self {
            Kind::Q4 => {
                for (b, (h, o)) in blocks::<18, 4, 16>(src, heads, bodies) {
                    h.copy_from_slice(&f16::from_le_bytes([b[0], b[1]]).to_f32().to_le_bytes());
                    o.copy_from_slice(&b[2..]);
                }
            }
            Kind::Q4h => split::<18, 2, 16>(src, heads, bodies),
            Kind::Q8h => split::<34, 2, 32>(src, heads, bodies),
            Kind::Iq4xs => split::<136, 8, 128>(src, heads, bodies),
            Kind::Q6 => {
                for (s, (scales, q)) in blocks::<210, 64, 256>(src, heads, bodies) {
                    let d = f16::from_le_bytes([s[208], s[209]]).to_f32();
                    for j in 0..256 {
                        let h = j / 128;
                        let l = j % 32;
                        let group = j % 128 / 32;
                        let lo = s[h * 64 + l + group % 2 * 32] as u32;
                        let hi = s[128 + h * 32 + l] as u32;
                        let v = ((lo >> (group / 2 * 4)) & 15) | (((hi >> (group * 2)) & 3) << 4);
                        q[j] = (v as i32 - 32) as i8 as u8;
                        if j % 16 == 0 {
                            let scale = d * (s[192 + h * 8 + l / 16 + group * 2] as i8) as f32;
                            scales[j / 4..j / 4 + 4].copy_from_slice(&scale.to_le_bytes());
                        }
                    }
                }
            }
        }
    }
}

/// The blocks of `src` with their heads and bodies.
#[allow(clippy::type_complexity)]
fn blocks<'a, const S: usize, const H: usize, const B: usize>(
    src: &'a [u8],
    heads: &'a mut [u8],
    bodies: &'a mut [u8],
) -> impl Iterator<Item = (&'a [u8; S], (&'a mut [u8; H], &'a mut [u8; B]))> {
    let n = src.len() / S;
    assert!(src.len() == n * S && heads.len() == n * H && bodies.len() == n * B, "block run mismatch");
    src.as_chunks::<S>().0.iter().zip(heads.as_chunks_mut::<H>().0.iter_mut().zip(bodies.as_chunks_mut::<B>().0))
}

/// Blocks stored as their head followed by their body.
fn split<const S: usize, const H: usize, const B: usize>(src: &[u8], heads: &mut [u8], bodies: &mut [u8]) {
    for (b, (h, o)) in blocks::<S, H, B>(src, heads, bodies) {
        h.copy_from_slice(&b[..H]);
        o.copy_from_slice(&b[H..]);
    }
}

/// A repacked matrix of `count` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Planes {
    pub kind: Kind,
    pub count: usize,
    /// Byte offset of the body plane: the heads, padded to 16 bytes where the kernels load bodies as aligned vectors
    /// (to 64 for Q8_0, so pairs of its blocks are whole cache lines).
    pub body_at: usize,
}

impl Planes {
    pub fn new(kind: Kind, count: usize) -> Self {
        let heads = count * kind.sizes().1;
        let body_at = match kind {
            Kind::Q4 | Kind::Q6 => heads,
            Kind::Q8h => heads.next_multiple_of(64),
            Kind::Q4h | Kind::Iq4xs => heads.next_multiple_of(16),
        };
        Self { kind, count, body_at }
    }
    /// The planes of the blocks in `bytes` GGUF bytes.
    pub fn of(kind: Kind, bytes: usize) -> Self {
        Self::new(kind, bytes / kind.sizes().0)
    }
    pub fn size(&self) -> usize {
        self.body_at + self.count * self.kind.sizes().2
    }
    /// The whole repacked matrix.
    pub fn pack(&self, src: &[u8]) -> Vec<u8> {
        let (s, h, _) = self.kind.sizes();
        let mut out = vec![0u8; self.size()];
        let (heads, bodies) = out.split_at_mut(self.body_at);
        self.kind.fill(&src[..self.count * s], &mut heads[..self.count * h], bodies);
        out
    }
}

/// Q8_0 blocks as the GPU reads them (`Kind::Q8h`): the bytes and the offset of the blocks in u32 words.
pub fn repack_q8h(src: &[u8]) -> (Vec<u8>, usize) {
    let p = Planes::of(Kind::Q8h, src.len());
    (p.pack(src), p.body_at / 4)
}

/// Q6 head (`Kind::Q6`): f32 (signed scale * block scale) per 16 elements, then signed bytes.
pub fn repack_q6(src: &[u8]) -> Vec<u8> {
    Planes::of(Kind::Q6, src.len()).pack(src)
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

    #[test]
    fn ue4m3_matches_ggml() {
        assert_eq!(ue4m3_half(0x38), 0.5); // exponent 7, mantissa 0: 1.0, halved
        assert_eq!(ue4m3_half(0x7e), 448. * 0.5);
        assert_eq!(ue4m3_half(0x01), (2f32).powi(-10));
        assert_eq!(ue4m3_half(0xb8), 0.5); // the sign bit is ignored
        assert_eq!((ue4m3_half(0), ue4m3_half(0x7f)), (0., 0.));
        for x in 0..=255u8 {
            let v = ue4m3_half(x);
            assert_eq!(f16::from_f32(v).to_f32(), v, "{x:#x} is not exact in f16");
        }
    }

    /// The repacks as they were written before the planes (whole matrices into one buffer; the heads padded to `align`
    /// bytes, 64 for Q8_0 since 7da6f93).
    fn split_old(src: &[u8], s: usize, h: usize, align: usize) -> Vec<u8> {
        let count = src.len() / s;
        let blocks_at = (count * h).next_multiple_of(align);
        let mut out = vec![0u8; blocks_at + count * (s - h)];
        for (i, b) in src.chunks_exact(s).enumerate() {
            out[i * h..(i + 1) * h].copy_from_slice(&b[..h]);
            out[blocks_at + i * (s - h)..blocks_at + (i + 1) * (s - h)].copy_from_slice(&b[h..]);
        }
        out
    }
    fn q6_old(src: &[u8]) -> Vec<u8> {
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

    #[test]
    fn planes_match_the_whole_matrix_repacks_and_fill_in_pieces() {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut random = |n: usize| -> Vec<u8> {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    // Finite f16 scales: clear the top exponent bit of every byte.
                    (x >> 24) as u8 & 0xbf
                })
                .collect()
        };
        for count in [1, 3, 8, 37, 64] {
            for kind in [Kind::Q4, Kind::Q4h, Kind::Q8h, Kind::Iq4xs, Kind::Q6] {
                let (s, h, b) = kind.sizes();
                let src = random(count * s);
                let p = Planes::new(kind, count);
                let whole = p.pack(&src);
                let old = match kind {
                    Kind::Q4 => repack_q4(&src, 32, count, 0, count, 0, 32),
                    Kind::Q6 => q6_old(&src),
                    Kind::Q8h => split_old(&src, s, h, 64),
                    _ => split_old(&src, s, h, 16),
                };
                assert_eq!(whole, old, "{kind:?} x {count}");
                // Pieces of any length give the same bytes.
                let mut out = vec![0u8; p.size()];
                let (heads, bodies) = out.split_at_mut(p.body_at);
                let mut at = 0;
                for n in [2, 1, 5].iter().cycle() {
                    let n = (*n).min(count - at);
                    kind.fill(&src[at * s..(at + n) * s], &mut heads[at * h..(at + n) * h], &mut bodies[at * b..(at + n) * b]);
                    at += n;
                    if at == count {
                        break;
                    }
                }
                assert_eq!(out, old, "{kind:?} x {count} in pieces");
            }
        }
    }
}
