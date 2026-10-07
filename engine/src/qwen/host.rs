//! CPU-only tensor-parallel reduction candidate. Keep the rank accumulation order fixed.

/// The common two-local-rank case writes its sum once instead of copying then adding in place.
pub(super) fn sum_partials(out: &mut [f32], parts: &[&[f32]], at: usize) {
    match parts {
        [a, b] => {
            for ((o, a), b) in out.iter_mut().zip(&a[at..]).zip(&b[at..]) {
                *o = *a + *b;
            }
        }
        [first, rest @ ..] => {
            out.copy_from_slice(&first[at..at + out.len()]);
            for p in rest {
                for (o, p) in out.iter_mut().zip(&p[at..]) {
                    *o += *p;
                }
            }
        }
        [] => panic!("a tensor-parallel reduction needs local ranks"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n).map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Mix signs and exponents, including signed zeros.
            if i % 97 == 0 { -0.0 } else { (state as i32 as f32) * 2f32.powi((i % 15) as i32 - 38) }
        }).collect()
    }

    fn bits(x: &[f32]) -> Vec<u32> { x.iter().map(|v| v.to_bits()).collect() }

    #[test]
    fn sums_keep_rank_order_and_chunk_boundaries() {
        let parts: Vec<_> = (1..=8).map(|s| values(513, s)).collect();
        for ranks in 1..=8 {
            let refs: Vec<_> = parts[..ranks].iter().map(Vec::as_slice).collect();
            for (at, len) in [(0, 513), (1, 127), (257, 256), (513, 0)] {
                let mut expected = refs[0][at..at + len].to_vec();
                for p in &refs[1..] {
                    for (a, b) in expected.iter_mut().zip(&p[at..]) { *a += b; }
                }
                let mut out = vec![0.; len];
                sum_partials(&mut out, &refs, at);
                assert_eq!(bits(&out), bits(&expected));
            }
        }
    }
}
