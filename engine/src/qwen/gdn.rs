//! Deterministic checks and timing for the opt-in prefill DeltaNet preparation pass.
use crate::{gpu::Instance, vulkan::{Device, Push}};
use anyhow::{ensure, Result};
use ash::vk;
use std::{rc::Rc, time::{Duration, Instant}};

/// Model-free comparison: include preparation and its barrier in the candidate timing,
/// restore inputs before every replay, and compare both outputs and the complete final state.
pub fn bench(n: usize, k_heads: usize, v_heads: usize, position: usize, iterations: u32) -> Result<()> {
    ensure!(n > 0 && n <= 512 && k_heads > 0 && k_heads <= 16 && v_heads > 0 && v_heads <= 48
        && v_heads.is_multiple_of(k_heads) && iterations > 0 && position <= u32::MAX as usize,
        "use 1-512 tokens, 1-16 key heads, 1-48 value heads (a multiple of key heads), positive iterations and a u32 position");
    let (dim, channels) = (128usize, (2 * k_heads + v_heads) * 128);
    let mut random = 0x938fa31u32;
    let mut sample = || {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        (random as f64 / u32::MAX as f64 * 2. - 1.) as f32
    };
    let mut input: Vec<_> = (0..n * channels).map(|_| sample()).collect();
    // Include zero and tiny norms: the epsilon is inside the square root, without division by dim.
    for t in 0..n {
        if t % 7 == 0 { input[t * channels..t * channels + 2 * k_heads * dim].fill(0.); }
        else if t % 7 == 1 {
            for value in &mut input[t * channels..t * channels + 2 * k_heads * dim] { *value *= 1e-4; }
        }
    }
    let gates: Vec<_> = (0..n * 2 * v_heads).map(|i| {
        if i % 9 == 0 { 21. } else if i % 11 == 0 { -30. } else { sample() * 4. }
    }).collect();
    let mut dt_a: Vec<_> = (0..v_heads).map(|_| sample()).collect();
    dt_a.extend((0..v_heads).map(|_| -0.1 - sample().abs()));
    let initial: Vec<_> = (0..v_heads * dim * dim).map(|_| sample() * 0.1).collect();
    let inst = Rc::new(Instance::new()?);
    let devices = inst.amd_gpus()?;
    ensure!(!devices.is_empty(), "no AMD GPUs found");
    for (gpu, pd) in devices.into_iter().enumerate() {
        let d = Device::new(inst.clone(), pd)?;
        let source = d.upload(bytemuck::cast_slice(&input))?;
        let raw_gates = d.upload(bytemuck::cast_slice(&gates))?;
        let original_state = d.upload(bytemuck::cast_slice(&initial))?;
        let weights = d.upload(bytemuck::cast_slice(&dt_a))?;
        let params = d.upload(bytemuck::cast_slice(&[position as u32, n as u32, 0, 0]))?;
        let conv = d.buffer(source.size, false, false, None)?;
        let ab = d.buffer(raw_gates.size, false, false, None)?;
        let state = d.buffer(original_state.size, false, false, None)?;
        let out = d.buffer(n * v_heads * dim * 4, false, false, None)?;
        let query = unsafe { d.dev.create_query_pool(&vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP).query_count(2), None)? };
        let mut p = Push { ptr: [conv.address, ab.address, state.address, out.address,
            weights.address, 0, params.address, 0], ..Default::default() };
        p.u[0] = channels as u32; p.u[4] = v_heads as u32; p.u[5] = k_heads as u32;
        p.u[6] = dim as u32; p.u[12] = 1e-6f32.to_bits(); p.u[15] = n as u32;
        let mut reference: Option<[Vec<u8>; 2]> = None;
        for prepared in [false, true] {
            let cmd = d.begin()?;
            unsafe { d.dev.cmd_reset_query_pool(cmd, query, 0, 2); }
            d.copy(cmd, &source, 0, &conv, 0, source.size);
            d.copy(cmd, &raw_gates, 0, &ab, 0, raw_gates.size);
            d.copy(cmd, &original_state, 0, &state, 0, original_state.size);
            unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query, 0); }
            if prepared {
                let prep = Push { ptr: [conv.address, weights.address, ab.address, 0, 0, 0, params.address, 0], ..p };
                d.dispatch(cmd, "qwen_gdn_prepare", prep, k_heads.max(v_heads).div_ceil(4) as u32, n as u32);
            }
            d.dispatch(cmd, if prepared { "qwen_gdn_prepared" } else { "qwen_gdn" }, p,
                (v_heads * dim / 16) as u32, 1);
            unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query, 1); }
            d.finish(cmd)?;
            let warmup = Instant::now();
            while warmup.elapsed() < Duration::from_millis(300) { d.run_reusable(cmd)?; }
            let mut times = Vec::new();
            for _ in 0..iterations {
                d.run_reusable(cmd)?;
                let mut ticks = [0u64; 2];
                unsafe { d.dev.get_query_pool_results(query, 0, &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; }
                times.push((ticks[1] - ticks[0]) as f64 * d.period as f64 / 1e6);
            }
            let values = [out.download()?, state.download()?];
            let mut max_abs = [0f32; 2];
            let mut max_rel = [0f32; 2];
            let mut bitwise_equal = true;
            for (i, bytes) in values.iter().enumerate() {
                let floats: &[f32] = bytemuck::cast_slice(bytes);
                ensure!(floats.iter().all(|v| v.is_finite()), "non-finite DeltaNet result on GPU{gpu}");
                if let Some(expected) = &reference {
                    let refs: &[f32] = bytemuck::cast_slice(&expected[i]);
                    for (&a, &b) in floats.iter().zip(refs) {
                        let error = (a - b).abs();
                        max_abs[i] = max_abs[i].max(error);
                        max_rel[i] = max_rel[i].max(error / b.abs().max(1e-6));
                        bitwise_equal &= a.to_bits() == b.to_bits();
                        ensure!(error <= 2e-5 + 2e-5 * b.abs(), "DeltaNet {} mismatch on GPU{gpu}: {a} vs {b}",
                            if i == 0 { "output" } else { "state" });
                    }
                }
            }
            if reference.is_none() { reference = Some(values); }
            println!("{}", serde_json::json!({"gpu":gpu,"n":n,"k_heads":k_heads,"v_heads":v_heads,
                "position":position,"prepared":prepared,"ms":times.iter().sum::<f64>()/times.len() as f64,
                "samples_ms":times,"max_abs_error":max_abs,"max_rel_error":max_rel,
                "bitwise_equal":bitwise_equal,"pass":true}));
            unsafe { d.dev.free_command_buffers(d.pool, &[cmd]); }
        }
        unsafe { d.dev.destroy_query_pool(query, None); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    const DIM: usize = 128;

    // Model the shader's two elements per lane and a fixed f32 reduction tree. Vulkan's exact
    // subgroup tree and transcendental approximations still require the GPU comparison above.
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut lanes = [0f32; 64];
        for l in 0..64 { lanes[l] = a[l] * b[l] + a[l + 64] * b[l + 64]; }
        for stride in [32, 16, 8, 4, 2, 1] {
            for l in 0..stride { lanes[l] += lanes[l + stride]; }
        }
        lanes[0]
    }

    fn normalize(q: &mut [f32], k: &mut [f32], eps: f32) {
        let qscale = (dot(q, q) + eps).sqrt().recip() * (DIM as f32).sqrt().recip();
        let kscale = (dot(k, k) + eps).sqrt().recip();
        for v in q { *v *= qscale; }
        for v in k { *v *= kscale; }
    }

    fn gates(alpha: f32, beta: f32, dt: f32, a: f32) -> (f32, f32) {
        let x = alpha + dt;
        (((if x > 20. { x } else { (1. + x.exp()).ln() }) * a).exp(),
            1. / (1. + (-beta).exp()))
    }

    fn prepare(input: &mut [f32], ab: &mut [f32], dt_a: &[f32], kh: usize, vh: usize, eps: f32) {
        let channels = (2 * kh + vh) * DIM;
        for (t, row) in input.chunks_exact_mut(channels).enumerate() {
            for head in 0..kh {
                let (q, rest) = row.split_at_mut(kh * DIM);
                normalize(&mut q[head * DIM..(head + 1) * DIM],
                    &mut rest[head * DIM..(head + 1) * DIM], eps);
            }
            for head in 0..vh {
                let at = t * 2 * vh + head;
                let (decay, beta) = gates(ab[at], ab[at + vh], dt_a[head], dt_a[vh + head]);
                ab[at] = decay; ab[at + vh] = beta;
            }
        }
    }

    #[allow(clippy::too_many_arguments)] // Mirror the kernel's independent inputs and state controls.
    fn recurrence(input: &[f32], ab: &[f32], dt_a: &[f32], state: &mut [f32], kh: usize, vh: usize,
        eps: f32, prepared: bool, fresh: bool) -> Vec<f32> {
        if fresh { state.fill(0.); }
        let channels = (2 * kh + vh) * DIM;
        let mut output = vec![0.; input.len() / channels * vh * DIM];
        for (t, row) in input.chunks_exact(channels).enumerate() {
            for head in 0..vh {
                let key = head % kh;
                let mut q = row[key * DIM..(key + 1) * DIM].to_vec();
                let mut k = row[(kh + key) * DIM..(kh + key + 1) * DIM].to_vec();
                let at = t * 2 * vh + head;
                let (decay, beta) = if prepared { (ab[at], ab[at + vh]) } else {
                    normalize(&mut q, &mut k, eps);
                    gates(ab[at], ab[at + vh], dt_a[head], dt_a[head + vh])
                };
                for value in 0..DIM {
                    let s = &mut state[(head * DIM + value) * DIM..(head * DIM + value + 1) * DIM];
                    for v in &mut *s { *v *= decay; }
                    let delta = (row[(2 * kh + head) * DIM + value] - dot(s, &k)) * beta;
                    for (v, k) in s.iter_mut().zip(&k) { *v += k * delta; }
                    output[(t * vh + head) * DIM + value] = dot(s, &q);
                }
            }
        }
        output
    }

    #[test]
    fn preparation_owns_every_element_once_and_preserves_values() {
        // TP4 27B (4/12), TP2 9B (8/16), an uneven final workgroup and the full 27B.
        for (kh, vh) in [(4usize, 12usize), (8, 16), (1, 3), (16, 48)] {
            let mut qk_writers = vec![0; 2 * kh * DIM];
            let mut gate_writers = vec![0; 2 * vh];
            for group in 0..kh.max(vh).div_ceil(4) {
                for wave in 0..4 {
                    let head = group * 4 + wave;
                    for lane in 0..64 {
                        if head < kh {
                            for base in [head * DIM, (head + kh) * DIM] {
                                qk_writers[base + lane] += 1; qk_writers[base + lane + 64] += 1;
                            }
                        }
                        if head < vh && lane == 0 { gate_writers[head] += 1; gate_writers[head + vh] += 1; }
                    }
                }
            }
            assert!(qk_writers.iter().chain(&gate_writers).all(|&n| n == 1));
            let mut input = vec![0.1; (2 * kh + vh) * DIM];
            let values = input[2 * kh * DIM..].to_vec();
            prepare(&mut input, &mut vec![0.; 2 * vh], &vec![-0.1; 2 * vh], kh, vh, 1e-6);
            assert_eq!(values, input[2 * kh * DIM..]);
        }
    }

    #[test]
    fn preparation_matches_f32_recurrence_and_microbatch_continuation() {
        for (kh, vh, n) in [(1, 3, 9), (4, 12, 17), (8, 16, 9)] {
            let channels = (2 * kh + vh) * DIM;
            let input: Vec<f32> = (0..n * channels).map(|i| {
                let t = i / channels;
                let scale = [0., 1e-5, 0.2, 1.][t % 4];
                ((i * 43 % 193) as f32 - 96.) / 97. * scale
            }).collect();
            let ab: Vec<f32> = (0..n * 2 * vh).map(|i| [-30., -1., 0., 19.9, 20., 20.1, 30.][i % 7]).collect();
            let dt_a: Vec<f32> = (0..2 * vh).map(|i| if i < vh { 0.1 } else { -0.1 }).collect();
            for eps in [1e-6, 1e-5] {
                for fresh in [false, true] {
                    let initial: Vec<f32> = (0..vh * DIM * DIM).map(|i| (i % 19) as f32 / 100.).collect();
                    let mut baseline = initial.clone();
                    let output = recurrence(&input, &ab, &dt_a, &mut baseline, kh, vh, eps, false, fresh);
                    let mut state = initial.clone();
                    let mut split_output = Vec::new();
                    // Uneven micro-batches, with a nonzero inherited state in the second one.
                    for (first, end) in [(0, 5), (5, n)] {
                        let mut rows = input[first * channels..end * channels].to_vec();
                        let mut gates = ab[first * 2 * vh..end * 2 * vh].to_vec();
                        prepare(&mut rows, &mut gates, &dt_a, kh, vh, eps);
                        split_output.extend(recurrence(&rows, &gates, &dt_a, &mut state,
                            kh, vh, eps, true, fresh && first == 0));
                    }
                    assert!(output.iter().chain(&baseline).all(|v| v.is_finite()));
                    assert!(output.iter().zip(&split_output).all(|(a,b)| a.to_bits() == b.to_bits()));
                    assert!(baseline.iter().zip(&state).all(|(a,b)| a.to_bits() == b.to_bits()));
                }
            }
        }
    }
}
