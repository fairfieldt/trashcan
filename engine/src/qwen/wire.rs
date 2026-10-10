//! Compact all-reduce payloads (docs/tp8-optimization.md W1; the default below prefill sizes): f16 values with one
//! exponent shift per block of `BLOCK` values, so a block whose largest magnitude reaches 2^15 is scaled down by a
//! power of two (exactly) before rounding and Qwen's massive activations cannot overflow f16. A range of `m`
//! values travels as `[m f16 values, little endian][one shift byte per block][zero padding to 4 bytes]`, its
//! blocks counted from the range's start; a message is the concatenation of its ranges.
//!
//! Rounding is to nearest even, with F16C when the CPU has it and the same bits in software otherwise, so every
//! machine rounds a value the same way. `Mesh` keeps the machines bitwise identical by rounding a running sum in
//! place before sending it, so both partners add the same two rounded values. Prefill-sized all-reduces stay f32.

/// Values per exponent shift.
pub const BLOCK: usize = 256;

/// The payload format of the all-reduces of a tensor-parallel session; the driver chooses it (`from_env`) and
/// passes it to the other machines at setup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wire {
    F32,
    F16,
}

impl Wire {
    /// The format of the all-reduces below prefill sizes: f16 (decode 3.6% and MTP 2.1% faster on TP8,
    /// results/qwen/tp8-opt-20261008), or with `TRASHCAN_TP_WIRE=f32` exact f32 sums, for comparing layouts bit for
    /// bit and validating against references.
    pub fn from_env() -> Self {
        if std::env::var("TRASHCAN_TP_WIRE").as_deref() == Ok("f32") {
            Wire::F32
        } else {
            Wire::F16
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Wire::F32 => "f32",
            Wire::F16 => "f16",
        }
    }

    pub fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "f32" => Ok(Wire::F32),
            "f16" => Ok(Wire::F16),
            _ => anyhow::bail!("unknown all-reduce payload format {name:?}"),
        }
    }
}

/// Bytes of one encoded range of `m` values.
pub fn range_bytes(m: usize) -> usize {
    (2 * m + m.div_ceil(BLOCK)).next_multiple_of(4)
}

/// Bytes of a message carrying `ranges` (element ranges `(start, end)`).
pub fn message_bytes(ranges: &[(usize, usize)]) -> usize {
    ranges.iter().map(|&(a, b)| range_bytes(b - a)).sum()
}

/// Shift `s` for a block whose largest magnitude is `max`: the block travels as f16(v * 2^-s), below 2^15
/// (`max` from 2^15 up gets s >= 1). Infinities and NaN give the largest shift; their values stay garbage.
fn shift_of(max: f32) -> u8 {
    let exponent = ((max.to_bits() >> 23) & 0xff) as i32 - 127;
    (exponent - 14).clamp(0, 100) as u8
}

/// 2^e as an f32, for |e| <= 100.
fn pow2(e: i32) -> f32 {
    f32::from_bits(((127 + e) as u32) << 23)
}

/// Whether F16C (and AVX for its 256-bit form) is available, detected once.
fn f16c() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx") && std::env::var_os("TRASHCAN_WIRE_SOFT").is_none())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn max_abs(v: &[f32]) -> f32 {
    // Eight running maxima, so the loop vectorizes; max is exact, so the order does not matter.
    let mut m = [0f32; 8];
    let mut chunks = v.chunks_exact(8);
    for c in &mut chunks {
        for j in 0..8 {
            m[j] = m[j].max(c[j].abs());
        }
    }
    let mut out = m.iter().fold(0f32, |a, &b| a.max(b));
    for &x in chunks.remainder() {
        out = out.max(x.abs());
    }
    // A NaN is ignored here and travels as an f16 NaN: the block is garbage either way, the same on every machine.
    out
}

/// Encode one block (at most `BLOCK` values) into `out` (2 bytes per value) and return its shift; with `round`,
/// also replace every value by the one the block carries.
fn encode_block(v: &mut [f32], round: bool, out: &mut [u8]) -> u8 {
    #[cfg(target_arch = "x86_64")]
    if f16c() {
        // SAFETY: F16C and AVX were detected above.
        return unsafe { x86::encode_block(v, round, out) };
    }
    let s = shift_of(max_abs(v));
    let (down, up) = (pow2(-(s as i32)), pow2(s as i32));
    for (x, o) in v.iter_mut().zip(out.chunks_exact_mut(2)) {
        let h = half::f16::from_f32(*x * down);
        o.copy_from_slice(&h.to_bits().to_le_bytes());
        if round {
            *x = h.to_f32() * up;
        }
    }
    s
}

/// Decode one block of `dst.len()` values with shift `s` into `dst`, or add them to it.
fn decode_block(bytes: &[u8], s: u8, dst: &mut [f32], add: bool) {
    let up = pow2(s as i32);
    #[cfg(target_arch = "x86_64")]
    if f16c() {
        // SAFETY: F16C and AVX were detected above.
        unsafe { x86::decode(bytes, up, dst, add) };
        return;
    }
    for (x, b) in dst.iter_mut().zip(bytes.chunks_exact(2)) {
        let v = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32() * up;
        if add {
            *x += v;
        } else {
            *x = v;
        }
    }
}

/// Encode the values `src[a..b]` of every range into `out` (`message_bytes(ranges)` long), rounding them in place
/// with `round`.
pub fn encode(src: &mut [f32], ranges: &[(usize, usize)], round: bool, out: &mut [u8]) {
    let mut at = 0;
    for &(a, b) in ranges {
        let m = b - a;
        let (vals, rest) = out[at..at + range_bytes(m)].split_at_mut(2 * m);
        encode_piece(&mut src[a..b], round, vals, rest);
        rest[m.div_ceil(BLOCK)..].fill(0);
        at += range_bytes(m);
    }
}

/// Decode a message carrying `ranges` into `dst[a..b]` of each range, or add it there.
pub fn decode(bytes: &[u8], ranges: &[(usize, usize)], dst: &mut [f32], add: bool) {
    let mut at = 0;
    for &(a, b) in ranges {
        let m = b - a;
        let (vals, shifts) = bytes[at..at + range_bytes(m)].split_at(2 * m);
        decode_piece(vals, shifts, &mut dst[a..b], add);
        at += range_bytes(m);
    }
}

/// `dst = dq(a) + dq(b)` for two messages carrying the same `ranges`: the sum both partners of a round compute from
/// what they sent and received, in one pass.
pub fn decode_sum(a: &[u8], b: &[u8], ranges: &[(usize, usize)], dst: &mut [f32]) {
    let mut at = 0;
    for &(start, end) in ranges {
        let m = end - start;
        let len = range_bytes(m);
        let (av, ash) = a[at..at + len].split_at(2 * m);
        let (bv, bsh) = b[at..at + len].split_at(2 * m);
        #[cfg(target_arch = "x86_64")]
        if f16c() {
            // SAFETY: F16C and AVX were detected above.
            unsafe { x86::decode_sum(av, ash, bv, bsh, &mut dst[start..end]) };
            at += len;
            continue;
        }
        for (i, d) in dst[start..end].chunks_mut(BLOCK).enumerate() {
            let (ua, ub) = (pow2(ash[i] as i32), pow2(bsh[i] as i32));
            for (j, x) in d.iter_mut().enumerate() {
                let k = 2 * (i * BLOCK + j);
                let va = half::f16::from_bits(u16::from_le_bytes([av[k], av[k + 1]])).to_f32() * ua;
                let vb = half::f16::from_bits(u16::from_le_bytes([bv[k], bv[k + 1]])).to_f32() * ub;
                *x = va + vb;
            }
        }
        at += len;
    }
}

/// Encode consecutive blocks of `v`: values into `vals`, one shift per block into `shifts`.
fn encode_piece(v: &mut [f32], round: bool, vals: &mut [u8], shifts: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if f16c() {
        // SAFETY: F16C and AVX were detected above.
        return unsafe { x86::encode_piece(v, round, vals, shifts) };
    }
    for (i, b) in v.chunks_mut(BLOCK).enumerate() {
        shifts[i] = encode_block(b, round, &mut vals[i * 2 * BLOCK..i * 2 * BLOCK + 2 * b.len()]);
    }
}

/// Decode consecutive blocks into `dst`, or add them to it.
fn decode_piece(vals: &[u8], shifts: &[u8], dst: &mut [f32], add: bool) {
    #[cfg(target_arch = "x86_64")]
    if f16c() {
        // SAFETY: F16C and AVX were detected above.
        return unsafe { x86::decode_piece(vals, shifts, dst, add) };
    }
    for (i, b) in dst.chunks_mut(BLOCK).enumerate() {
        decode_block(&vals[i * 2 * BLOCK..i * 2 * BLOCK + 2 * b.len()], shifts[i], b, add);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{pow2, shift_of};
    use std::arch::x86_64::*;

    /// `super::encode_block` with F16C. The first pass converts without a shift while it finds the block's largest
    /// magnitude; only a block that reaches 2^15 (a massive activation) is converted again with its shift. Rounding
    /// in place then reads the stored f16 values back. The bits are those of the software path.
    #[target_feature(enable = "avx,f16c")]
    #[inline]
    pub unsafe fn encode_block(v: &mut [f32], round: bool, out: &mut [u8]) -> u8 {
        let n = v.len() / 8 * 8;
        let sign = _mm256_set1_ps(-0.0);
        let mut mx = _mm256_setzero_ps();
        let mut i = 0;
        while i < n {
            let x = _mm256_loadu_ps(v.as_ptr().add(i));
            mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, x));
            _mm_storeu_si128(out.as_mut_ptr().add(2 * i) as *mut __m128i, _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x));
            i += 8;
        }
        let m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps::<1>(mx));
        let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let mut max = _mm_cvtss_f32(_mm_max_ss(m2, _mm_shuffle_ps::<1>(m2, m2)));
        for &x in &v[n..] {
            max = max.max(x.abs());
        }
        let s = shift_of(max);
        let (down, up) = (pow2(-(s as i32)), pow2(s as i32));
        if s != 0 {
            let d = _mm256_set1_ps(down);
            let mut i = 0;
            while i < n {
                let x = _mm256_mul_ps(_mm256_loadu_ps(v.as_ptr().add(i)), d);
                _mm_storeu_si128(out.as_mut_ptr().add(2 * i) as *mut __m128i, _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x));
                i += 8;
            }
        }
        for j in n..v.len() {
            let h = _mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_set_ss(v[j] * down));
            out[2 * j..2 * j + 2].copy_from_slice(&(_mm_extract_epi16::<0>(h) as u16).to_le_bytes());
        }
        if round {
            decode(out, up, v, false);
        }
        s
    }

    /// `encode_block` over consecutive blocks, inlined into one loop.
    #[target_feature(enable = "avx,f16c")]
    pub unsafe fn encode_piece(v: &mut [f32], round: bool, vals: &mut [u8], shifts: &mut [u8]) {
        for (i, b) in v.chunks_mut(super::BLOCK).enumerate() {
            shifts[i] = encode_block(b, round, &mut vals[i * 2 * super::BLOCK..i * 2 * super::BLOCK + 2 * b.len()]);
        }
    }

    /// `decode` over consecutive blocks, inlined into one loop.
    #[target_feature(enable = "avx,f16c")]
    pub unsafe fn decode_piece(vals: &[u8], shifts: &[u8], dst: &mut [f32], add: bool) {
        for (i, b) in dst.chunks_mut(super::BLOCK).enumerate() {
            decode(&vals[i * 2 * super::BLOCK..i * 2 * super::BLOCK + 2 * b.len()], pow2(shifts[i] as i32), b, add);
        }
    }

    /// `dst = a * 2^sa + b * 2^sb` block by block.
    #[target_feature(enable = "avx,f16c")]
    pub unsafe fn decode_sum(av: &[u8], ash: &[u8], bv: &[u8], bsh: &[u8], dst: &mut [f32]) {
        for (i, d) in dst.chunks_mut(super::BLOCK).enumerate() {
            let at = 2 * i * super::BLOCK;
            let (ua, ub) = (_mm256_set1_ps(pow2(ash[i] as i32)), _mm256_set1_ps(pow2(bsh[i] as i32)));
            let n = d.len() / 8 * 8;
            let mut j = 0;
            while j < n {
                let x = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(av.as_ptr().add(at + 2 * j) as *const __m128i)), ua);
                let y = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(bv.as_ptr().add(at + 2 * j) as *const __m128i)), ub);
                _mm256_storeu_ps(d.as_mut_ptr().add(j), _mm256_add_ps(x, y));
                j += 8;
            }
            for j in n..d.len() {
                let h = |v: &[u8]| _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(u16::from_le_bytes([v[at + 2 * j], v[at + 2 * j + 1]]) as i32)));
                d[j] = h(av) * pow2(ash[i] as i32) + h(bv) * pow2(bsh[i] as i32);
            }
        }
    }

    /// `dst = h * up` (or `dst += h * up`) for the f16 values in `bytes`, eight at a time.
    #[target_feature(enable = "avx,f16c")]
    #[inline]
    pub unsafe fn decode(bytes: &[u8], up: f32, dst: &mut [f32], add: bool) {
        let u = _mm256_set1_ps(up);
        let n = dst.len() / 8 * 8;
        let mut i = 0;
        while i < n {
            let v = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(bytes.as_ptr().add(2 * i) as *const __m128i)), u);
            let p = dst.as_mut_ptr().add(i);
            _mm256_storeu_ps(p, if add { _mm256_add_ps(_mm256_loadu_ps(p), v) } else { v });
            i += 8;
        }
        for j in n..dst.len() {
            let h = _mm_cvtsi32_si128(u16::from_le_bytes([bytes[2 * j], bytes[2 * j + 1]]) as i32);
            let v = _mm_cvtss_f32(_mm_cvtph_ps(h)) * up;
            if add {
                dst[j] += v;
            } else {
                dst[j] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let u = state as f32 / u32::MAX as f32 - 0.5;
                // Mostly ordinary residual magnitudes, some tiny ones, a few massive activations.
                match i % 97 {
                    0 => u * 3.0e5,
                    1 => u * 1e-6,
                    2 => u * 7.0e4,
                    _ => u * 8.0,
                }
            })
            .collect()
    }

    #[test]
    fn layout_sizes_are_aligned() {
        for m in [0, 1, 7, 255, 256, 257, 2560, 5120, 7680] {
            assert_eq!(range_bytes(m) % 4, 0);
            assert!(range_bytes(m) >= 2 * m + m.div_ceil(BLOCK));
        }
        assert_eq!(message_bytes(&[(0, 2560), (5120, 7680)]), 2 * range_bytes(2560));
    }

    #[test]
    fn round_trip_is_close_range_safe_and_idempotent() {
        let v = samples(5120 + 77, 7);
        let mut rounded = v.clone();
        let ranges = [(3, 2563), (2563, v.len())];
        let mut bytes = vec![0u8; message_bytes(&ranges)];
        encode(&mut rounded, &ranges, true, &mut bytes);
        let mut back = vec![0f32; v.len()];
        decode(&bytes, &ranges, &mut back, false);
        for (a, b) in ranges {
            for (k, block) in v[a..b].chunks(BLOCK).enumerate() {
                // Relative error 2^-11, plus half an f16 subnormal step at the block's scale for tiny values that
                // share a block with a massive one.
                let floor = 2f32.powi(shift_of(max_abs(block)) as i32 - 25);
                for (j, &x) in block.iter().enumerate() {
                    let i = a + k * BLOCK + j;
                    assert_eq!(back[i].to_bits(), rounded[i].to_bits(), "decode equals the in-place rounding at {i}");
                    assert!(back[i].is_finite());
                    assert!((back[i] - x).abs() <= x.abs() * 2f32.powi(-11) + floor, "{} vs {x} at {i}", back[i]);
                }
            }
        }
        // Rounded values travel exactly: encoding them again gives the same bytes.
        let mut again = rounded.clone();
        let mut bytes2 = vec![0u8; message_bytes(&ranges)];
        encode(&mut again, &ranges, true, &mut bytes2);
        assert_eq!(bytes, bytes2);
        assert!(again.iter().zip(&rounded).all(|(a, b)| a.to_bits() == b.to_bits()));
        // Decoding with add adds to what is there.
        let mut acc = vec![1f32; v.len()];
        decode(&bytes, &ranges, &mut acc, true);
        for i in 3..v.len() {
            assert_eq!(acc[i].to_bits(), (1f32 + rounded[i]).to_bits());
        }
    }

    #[test]
    fn hardware_and_software_rounding_agree_bit_for_bit() {
        // The software conversion (`from_f32_const`) against whatever `encode_block` picked (F16C on x86-64 hosts
        // that have it), including ties, subnormal results and shifted blocks.
        let mut v = samples(4096, 11);
        v.extend([0.0, -0.0, 1.0 + 2f32.powi(-11), 1.0 + 3.0 * 2f32.powi(-11), 6.0e-8, -3.0e-5, 32767.0, 32768.0, 65504.0, 1.0e9]);
        for block in v.chunks(BLOCK) {
            let (mut hw, mut sw) = (block.to_vec(), block.to_vec());
            let mut hb = vec![0u8; 2 * block.len()];
            let s = encode_block(&mut hw, true, &mut hb);
            let (down, up) = (pow2(-(s as i32)), pow2(s as i32));
            for (i, x) in sw.iter_mut().enumerate() {
                let h = half::f16::from_f32_const(*x * down);
                assert_eq!(h.to_bits().to_le_bytes(), [hb[2 * i], hb[2 * i + 1]], "value {x}");
                *x = h.to_f32_const() * up;
            }
            assert!(hw.iter().zip(&sw).all(|(a, b)| a.to_bits() == b.to_bits()));
            let mut d = vec![0f32; block.len()];
            decode_block(&hb, s, &mut d, false);
            assert!(d.iter().zip(&sw).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn decode_sum_adds_the_two_rounded_messages() {
        let (a, b) = (samples(5120, 5), samples(5120, 9));
        let ranges = [(0, 2560), (2560, 5120)];
        let (mut ra, mut rb) = (a.clone(), b.clone());
        let (mut ba, mut bb) = (vec![0u8; message_bytes(&ranges)], vec![0u8; message_bytes(&ranges)]);
        encode(&mut ra, &ranges, true, &mut ba);
        encode(&mut rb, &ranges, true, &mut bb);
        let mut sum = vec![0f32; 5120];
        decode_sum(&ba, &bb, &ranges, &mut sum);
        assert!(sum.iter().zip(ra.iter().zip(&rb)).all(|(s, (x, y))| s.to_bits() == (x + y).to_bits()));
        // Encoding without rounding gives the same bytes.
        let mut plain = vec![0u8; message_bytes(&ranges)];
        encode(&mut a.clone(), &ranges, false, &mut plain);
        assert_eq!(plain, ba);
    }
}
