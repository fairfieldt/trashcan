//! CPU-only tensor-parallel reduction candidates. Keep the rank and RMS accumulation order fixed.

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

/// Fuse remote addition, residual update and squared norm as `residual + (local + remote)`,
/// exactly as the existing host path (including operand order).
/// The eight accumulation chains and the final reduction tree are unchanged. Output is a separate pass
/// because its scale depends on the whole row. Used only with TRASHCAN_TP_HOST_FUSED=1.
pub(super) fn add_norm(resid: &mut [f32], local: &[f32], remote: Option<&[f32]>, w: &[f32], eps: f32, out: &mut [f32]) {
    assert!(!w.is_empty() && w.len() % 8 == 0);
    assert_eq!(resid.len(), local.len());
    assert_eq!(resid.len(), out.len());
    assert_eq!(resid.len() % w.len(), 0);
    if let Some(remote) = remote {
        assert_eq!(resid.len(), remote.len());
    }
    for (row, (x, o)) in resid.chunks_exact_mut(w.len()).zip(out.chunks_exact_mut(w.len())).enumerate() {
        let at = row * w.len();
        let mut acc = [0f32; 8];
        if let Some(remote) = remote {
            for ((chunk, l), r) in x.as_chunks_mut::<8>().0.iter_mut()
                .zip(local[at..at + w.len()].as_chunks::<8>().0)
                .zip(remote[at..at + w.len()].as_chunks::<8>().0) {
                for j in 0..8 {
                    chunk[j] += l[j] + r[j];
                    acc[j] += chunk[j] * chunk[j];
                }
            }
        } else {
            for (chunk, l) in x.as_chunks_mut::<8>().0.iter_mut()
                .zip(local[at..at + w.len()].as_chunks::<8>().0) {
                for j in 0..8 {
                    chunk[j] += l[j];
                    acc[j] += chunk[j] * chunk[j];
                }
            }
        }
        let ss = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        let inv = 1.0 / (ss / w.len() as f32 + eps).sqrt();
        for ((o, &v), &wv) in o.iter_mut().zip(x.iter()).zip(w) {
            *o = v * inv * wv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n).map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Mix signs and exponents, including exact cancellation and signed zeros.
            if i % 97 == 0 { -0.0 } else { (state as i32 as f32) * 2f32.powi((i % 15) as i32 - 38) }
        }).collect()
    }

    fn reference(resid: &mut [f32], local: &[f32], remote: Option<&[f32]>, w: &[f32], eps: f32, out: &mut [f32]) {
        let mut sum = local.to_vec();
        if let Some(remote) = remote {
            for (a, b) in sum.iter_mut().zip(remote) { *a += b; }
        }
        for (a, b) in resid.iter_mut().zip(sum) { *a += b; }
        for (x, o) in resid.chunks_exact(w.len()).zip(out.chunks_exact_mut(w.len())) {
            let mut acc = [0f32; 8];
            for c in x.chunks_exact(8) {
                for j in 0..8 { acc[j] += c[j] * c[j]; }
            }
            let ss = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            let inv = 1.0 / (ss / w.len() as f32 + eps).sqrt();
            for ((o, &v), &wv) in o.iter_mut().zip(x).zip(w) { *o = v * inv * wv; }
        }
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

    #[test]
    fn fused_norm_preserves_bits_across_layers_and_verification_widths() {
        for d in [8, 4096, 5120] {
            for n in [1, 2, 3, 4, 8] {
                for peer in [false, true] {
                    let mut expected = values(d * n, 1);
                    let mut actual = expected.clone();
                    let local = values(d * n, 2);
                    let remote = values(d * n, 3);
                    let weights = values(d, 4);
                    let mut old_out = vec![0.; d * n];
                    let mut new_out = old_out.clone();
                    for _ in 0..128 {
                        reference(&mut expected, &local, peer.then_some(&remote), &weights, 1e-6, &mut old_out);
                        add_norm(&mut actual, &local, peer.then_some(&remote), &weights, 1e-6, &mut new_out);
                        assert_eq!(bits(&actual), bits(&expected));
                        assert_eq!(bits(&new_out), bits(&old_out));
                    }
                }
            }
        }
    }
}
