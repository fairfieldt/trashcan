use crate::{
    gguf::{GgmlType, Gguf},
    gpu::Instance,
    quant,
    vulkan::{Device, Push},
};
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::{
    path::Path,
    rc::Rc,
    time::{Duration, Instant},
};

pub fn norm(iterations: u32) -> Result<()> {
    ensure!(iterations > 0, "positive iterations required");
    const N: usize = 3840;
    let inst = Rc::new(Instance::new()?);
    for (gpu, pd) in inst.amd_gpus()?.into_iter().enumerate() {
        let d = Device::new(inst.clone(), pd)?;
        // Cached descriptors must not outlive buffers and see recycled Vulkan handles.
        let mut keepalive = Vec::new();
        let query = unsafe {
            d.dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?
        };
        for columns in [1usize, 4] {
            for seed in 1..=4u32 {
                let mut state = seed;
                let mut random = |count: usize| -> Vec<f32> {
                    (0..count)
                        .map(|i| {
                            state ^= state << 13;
                            state ^= state >> 17;
                            state ^= state << 5;
                            let scale = [0.001, 0.1, 1., 10.][i % 4];
                            (state as f64 / u32::MAX as f64 * 2. - 1.) as f32 * scale
                        })
                        .collect()
                };
                let input = random(N * columns);
                let peer = random(N * columns);
                let residual = random(N * columns);
                let post_weight = random(N);
                let next_weight = random(N);
                let scale = 0.875f32;
                let mut expected = [Vec::new(), Vec::new(), Vec::new()];
                for t in 0..columns {
                    let x: Vec<f32> = (0..N).map(|i| input[t * N + i] + peer[t * N + i]).collect();
                    let inv = (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / N as f64
                        + 1e-6)
                        .sqrt()
                        .recip();
                    let y: Vec<f32> = (0..N)
                        .map(|i| (x[i] as f64 * inv * post_weight[i] as f64) as f32)
                        .collect();
                    let r: Vec<f32> = (0..N)
                        .map(|i| (residual[t * N + i] + y[i]) * scale)
                        .collect();
                    let ni = (r.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / N as f64 + 1e-6)
                        .sqrt()
                        .recip();
                    expected[0].extend(&y);
                    expected[1].extend(&r);
                    expected[2]
                        .extend((0..N).map(|i| (r[i] as f64 * ni * next_weight[i] as f64) as f32));
                }
                let a = d.upload(bytemuck::cast_slice(&input))?;
                let receive = d.upload(bytemuck::cast_slice(&peer))?;
                let initial = d.upload(bytemuck::cast_slice(&residual))?;
                let e = d.buffer(N * columns * 4, false, false, None)?;
                let b = d.upload(bytemuck::cast_slice(&post_weight))?;
                let f = d.upload(bytemuck::cast_slice(&next_weight))?;
                let c = d.buffer(N * columns * 4, false, false, None)?;
                let extra = d.buffer(N * columns * 4, false, false, None)?;
                let mut p = Push {
                    ptr: [
                        a.address,
                        b.address,
                        c.address,
                        0,
                        e.address,
                        f.address,
                        0,
                        extra.address,
                    ],
                    ..Push::default()
                };
                p.u[0] = N as u32;
                p.u[3] = N as u32;
                p.u[11] = 7;
                p.u[12] = scale.to_bits();
                p.u[15] = columns as u32;
                let mut baseline: Option<Vec<Vec<u8>>> = None;
                for shader in ["norm", "norm_fused"] {
                    let cmd = d.begin()?;
                    unsafe {
                        d.dev.cmd_reset_query_pool(cmd, query, 0, 2);
                    }
                    // Restore the residual before every replay. Time only the norm dispatch.
                    d.copy(cmd, &initial, 0, &e, 0, initial.size);
                    unsafe {
                        d.dev.cmd_write_timestamp(
                            cmd,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            query,
                            0,
                        );
                    }
                    d.dispatch_receive(cmd, shader, p, 1, columns as u32, &receive)?;
                    unsafe {
                        d.dev.cmd_write_timestamp(
                            cmd,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            query,
                            1,
                        );
                    }
                    d.finish(cmd)?;
                    let warmup = Instant::now();
                    while warmup.elapsed() < Duration::from_millis(300) {
                        d.run_reusable(cmd)?;
                    }
                    for repetition in 1..=2 {
                        let mut times = Vec::new();
                        for _ in 0..iterations {
                            d.run_reusable(cmd)?;
                            let mut ticks = [0u64; 2];
                            unsafe {
                                d.dev.get_query_pool_results(
                                    query,
                                    0,
                                    &mut ticks,
                                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                                )?;
                            }
                            times.push((ticks[1] - ticks[0]) as f64 * d.period as f64 / 1000.);
                        }
                        let outputs = vec![c.download()?, e.download()?, extra.download()?];
                        let mut worst = 0f64;
                        for (bytes, cpu) in outputs.iter().zip(&expected) {
                            for (bytes, &want) in bytes.chunks_exact(4).zip(cpu) {
                                let got = f32::from_le_bytes(bytes.try_into()?);
                                let error =
                                    (got as f64 - want as f64).abs() / (1. + (want as f64).abs());
                                ensure!(
                                    got.is_finite() && error < 1e-5,
                                    "{shader} CPU norm mismatch: {got} vs {want}"
                                );
                                worst = worst.max(error);
                            }
                        }
                        let identical = baseline.as_ref().is_none_or(|old| *old == outputs);
                        ensure!(identical, "fused norm differs from generic norm on seed {seed}, columns {columns}, GPU {gpu}");
                        if baseline.is_none() {
                            baseline = Some(outputs);
                        }
                        times.sort_by(f64::total_cmp);
                        println!(
                            "{}",
                            serde_json::json!({"gpu":gpu,"shader":shader,"seed":seed,"columns":columns,"repetition":repetition,"iterations":iterations,"median_us":times[times.len()/2],"cpu_worst_scaled_error":worst,"bit_identical":identical,"pass":true})
                        );
                    }
                    unsafe {
                        d.dev.free_command_buffers(d.pool, &[cmd]);
                    }
                }
                keepalive.extend([a, receive, initial, e, b, f, c, extra]);
            }
        }
        unsafe {
            d.dev.destroy_query_pool(query, None);
        }
    }
    Ok(())
}

pub fn attention(
    iterations: u32,
    tokens: usize,
    partitions: Option<u32>,
    kernel: &str,
) -> Result<()> {
    ensure!(
        iterations > 0 && (1..=131072).contains(&tokens),
        "positive iterations, tokens 1..131072"
    );
    ensure!(
        matches!(kernel, "scalar" | "selected"),
        "attention kernel must be scalar or selected"
    );
    let partition_counts = partitions.map_or_else(|| vec![8, 16, 32, 64, 128], |n| vec![n]);
    ensure!(
        partition_counts
            .iter()
            .all(|n| [8, 16, 32, 64, 128].contains(n)),
        "partitions must be 8, 16, 32, 64 or 128"
    );
    let inst = Rc::new(Instance::new()?);
    for (gpu, pd) in inst.amd_gpus()?.into_iter().enumerate() {
        let d = Device::new(inst.clone(), pd)?;
        for (dim, kv) in [(256usize, 4usize), (512, 1)] {
            let inputs: Vec<f32> = (0..8 * dim)
                .map(|i| ((i * 13 % 47) as f32 - 23.) / 64.)
                .collect();
            let keys: Vec<u16> = (0..tokens * kv * dim)
                .map(|i| {
                    half::f16::from_f32(((i * 17 + i / dim * 7) % 67) as f32 / 128. - 0.25)
                        .to_bits()
                })
                .collect();
            let values: Vec<u16> = (0..keys.len())
                .map(|i| {
                    half::f16::from_f32(((i * 19 + i / dim * 3) % 71) as f32 / 64. - 0.5).to_bits()
                })
                .collect();
            let qbuf = d.upload(bytemuck::cast_slice(&inputs))?;
            let kbuf = d.upload(bytemuck::cast_slice(&keys))?;
            let vbuf = d.upload(bytemuck::cast_slice(&values))?;
            let output = d.buffer(8 * dim * 4, false, false, None)?;
            let scratch = d.buffer(8 * 128 * (dim + 2) * 4, false, false, None)?;
            let params = d.upload(bytemuck::cast_slice(&[(tokens - 1) as u32, 1u32, 0, 0]))?;
            let mut expected = Vec::new();
            for head in [0, 7] {
                let kh = head / (8 / kv);
                let scores: Vec<f64> = (0..tokens)
                    .map(|t| {
                        (0..dim)
                            .map(|j| {
                                inputs[head * dim + j] as f64
                                    * half::f16::from_bits(keys[(t * kv + kh) * dim + j]).to_f64()
                            })
                            .sum::<f64>()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let denominator: f64 = weights.iter().sum();
                for j in [0, dim / 2, dim - 1] {
                    let sum: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(t, w)| {
                            w * half::f16::from_bits(values[(t * kv + kh) * dim + j]).to_f64()
                        })
                        .sum();
                    expected.push((head * dim + j, sum / denominator));
                }
            }
            let query = unsafe {
                d.dev.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                )?
            };
            for &parts in &partition_counts {
                let (partial_shader, merge_shader, heads_per_group) = if kernel == "selected" {
                    crate::vulkan::selected_attention_shaders(parts, dim, kv)?
                } else {
                    let (partial, merge) = crate::vulkan::attention_shaders(parts)?;
                    (partial, merge, 1)
                };
                let mut p = Push {
                    ptr: [
                        qbuf.address,
                        kbuf.address,
                        vbuf.address,
                        output.address,
                        scratch.address,
                        0,
                        params.address,
                        0,
                    ],
                    ..Push::default()
                };
                p.u[4] = 8;
                p.u[5] = kv as u32;
                p.u[6] = dim as u32;
                p.u[7] = tokens as u32;
                let cmd = d.begin()?;
                unsafe {
                    d.dev.cmd_reset_query_pool(cmd, query, 0, 2);
                    d.dev
                        .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, query, 0);
                }
                for _ in 0..iterations {
                    d.dispatch(
                        cmd,
                        partial_shader,
                        p,
                        8u32.div_ceil(heads_per_group) * parts,
                        1,
                    );
                    d.dispatch(cmd, merge_shader, p, 8, 1);
                }
                unsafe {
                    d.dev.cmd_write_timestamp(
                        cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        query,
                        1,
                    );
                }
                d.finish(cmd)?;
                let warmup = Instant::now();
                while warmup.elapsed() < Duration::from_millis(300) {
                    d.run_reusable(cmd)?;
                }
                d.run(cmd)?;
                let mut times = [0u64; 2];
                unsafe {
                    d.dev.get_query_pool_results(
                        query,
                        0,
                        &mut times,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )?;
                }
                let result = output.download()?;
                for &(i, wanted) in &expected {
                    let got = f32::from_le_bytes(result[i * 4..i * 4 + 4].try_into()?) as f64;
                    ensure!(
                        got.is_finite() && (got - wanted).abs() < 0.0002,
                        "attention dim {dim}, partitions {parts}, value {i}: {got} vs {wanted}"
                    );
                }
                let reference_cmd = d.begin()?;
                let mut reference_push = p;
                reference_push.u[13] = parts;
                d.dispatch(
                    reference_cmd,
                    "attention_merge_reference",
                    reference_push,
                    8,
                    1,
                );
                d.finish(reference_cmd)?;
                d.run(reference_cmd)?;
                ensure!(result == output.download()?, "attention merge differs from serial reference: GPU {gpu}, dim {dim}, partitions {parts}");
                println!(
                    "{}",
                    serde_json::json!({"gpu":gpu,"tokens":tokens,"dim":dim,"kv_heads":kv,"partitions":parts,"kernel":kernel,"partial_shader":partial_shader,"heads_per_group":heads_per_group,"iterations":iterations,"milliseconds":(times[1]-times[0]) as f64 * d.period as f64 / 1e6 / iterations as f64,"merge_bit_identical":true,"pass":true})
                );
            }
            unsafe {
                d.dev.destroy_query_pool(query, None);
            }
        }
    }
    Ok(())
}

pub fn bench(
    path: &Path,
    iterations: u32,
    batch: usize,
    kernel: Option<&str>,
    tensor: &str,
    warmup_ms: u64,
    shape: Option<(usize, usize)>,
) -> Result<()> {
    ensure!(
        iterations > 0 && (1..=512).contains(&batch),
        "iterations positive, batch 1..512"
    );
    let shader = kernel.unwrap_or(if batch == 1 {
        "matvecq4"
    } else if batch <= 8 {
        "multivec4"
    } else {
        "gemm"
    });
    ensure!(
        [
            "matvecq4",
            "matvecq4narrow",
            "multivec",
            "multivec2",
            "multivec4",
            "multivec4x2",
            "multivec4x3",
            "multivec4_reference",
            "headvec",
            "gemm",
            "qwen_q4c2",
            "qwen_q4c3",
            "qwen_q4c4"
        ]
        .contains(&shader),
        "unsupported kernel"
    );
    ensure!(
        batch == 1 || !matches!(shader, "matvecq4" | "matvecq4narrow" | "headvec"),
        "single-column kernel needs batch 1"
    );
    let g = Gguf::open(path)?;
    let t = g.tensor(tensor).context("benchmark tensor not found")?;
    ensure!(
        matches!(t.ty, GgmlType::Q4_0 | GgmlType::Q6K) && t.dims.len() == 2,
        "benchmark needs a Q4_0 or Q6_K matrix"
    );
    let kind = u32::from(t.ty == GgmlType::Q6K);
    let (full_k, full_n) = (t.dims[0] as usize, t.dims[1] as usize);
    let column_split =
        tensor.ends_with("ffn_down.weight") || tensor.ends_with("attn_output.weight");
    let replicated = tensor.ends_with("attn_k.weight") || tensor.ends_with("attn_v.weight");
    // Global KV is replicated; sliding KV and Q/gate/up are split by rows.
    let (k, n) = if let Some((rows, cols)) = shape {
        ensure!(kind == 0 && rows <= full_n && cols <= full_k, "--rows/--cols cut a Q4 matrix");
        (cols, rows)
    } else if column_split {
        (full_k / 2, full_n)
    } else if replicated && full_n == 512 {
        (full_k, full_n)
    } else {
        (full_k, full_n / 2)
    };
    let shader = if kernel.is_some() {
        shader
    } else if batch == 1 {
        if kind == 1 {
            "headvec"
        } else {
            crate::vulkan::q4_decode_shader(n, k)
        }
    } else if batch == 2 {
        "multivec4x2"
    } else if batch == 3 {
        "multivec4x3"
    } else if batch <= 8 {
        "multivec4"
    } else {
        "gemm"
    };
    let group_rows = if shader == "gemm" {
        64
    } else if matches!(
        shader,
        "matvecq4"

            | "matvecq4narrow"
            | "headvec"
            | "multivec4"
            | "multivec4x2"
            | "multivec4x3"
            | "multivec4_reference"
    ) || shader.starts_with("qwen_q4c")
    {
        16
    } else {
        4
    };
    let group_cols = if let Some(c) = shader.strip_prefix("qwen_q4c") {
        c[..1].parse::<usize>()?
    } else if shader == "gemm" {
        64
    } else if matches!(shader, "multivec4" | "multivec4_reference") {
        4
    } else if shader == "multivec4x2" {
        2
    } else if shader == "multivec4x3" {
        3
    } else if shader == "multivec2" {
        2
    } else {
        8
    };
    ensure!(
        k.is_multiple_of(32) && n.is_multiple_of(group_rows),
        "unsupported benchmark shape"
    );
    ensure!(
        kind == 0 || (k == full_k && shader != "gemm"),
        "Q6 head needs complete rows and a vector kernel"
    );
    ensure!(
        shader != "headvec" || kind == 1,
        "headvec needs a Q6_K matrix"
    );
    let data = if kind == 0 {
        quant::repack_q4(g.data(t), full_k, full_n, 0, n, 0, k)
    } else {
        quant::repack_q6(&g.data(t)[..n * k / 256 * 210])
    };
    let mut state = 0x12345678u32;
    let inputs: Vec<f32> = (0..k * batch)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f64 / u32::MAX as f64 * 2. - 1.) as f32 * [0.001, 0.1, 1., 10.][i % 4]
        })
        .collect();
    let inst = Rc::new(Instance::new()?);
    for (gpu, pd) in inst.amd_gpus()?.into_iter().enumerate() {
        let d = Device::new(inst.clone(), pd)?;
        let w = d.upload_matrix(&data)?;
        let x = d.upload(bytemuck::cast_slice(&inputs))?;
        let y = d.buffer(n * batch * 4, false, false, None)?;
        let params = d.upload(bytemuck::cast_slice(&[0u32, batch as u32]))?;
        let q = unsafe {
            d.dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?
        };
        let mut p = Push::default();
        p.ptr[0] = 0;
        p.ptr[1] = x.address;
        p.ptr[2] = y.address;
        p.ptr[6] = params.address;
        p.u[0] = n as u32;
        p.u[1] = k as u32;
        p.u[3] = (n * k / if kind == 0 { 32 } else { 16 }) as u32;
        p.u[9] = kind;
        let dispatch = |cmd| -> Result<()> {
            d.dispatch_matrix(
                cmd,
                shader,
                p,
                n.div_ceil(group_rows) as u32,
                batch.div_ceil(group_cols) as u32,
                &w,
            )?;
            Ok(())
        };
        let cmd = d.begin()?;
        unsafe {
            d.dev.cmd_reset_query_pool(cmd, q, 0, 2);
            d.dev
                .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
        }
        for _ in 0..iterations {
            dispatch(cmd)?;
        }
        unsafe {
            d.dev
                .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1);
        }
        d.finish(cmd)?;
        let warmup = Instant::now();
        while warmup.elapsed() < Duration::from_millis(warmup_ms) {
            d.run_reusable(cmd)?;
        }
        d.run(cmd)?;
        let mut ts = [0u64; 2];
        unsafe {
            d.dev.get_query_pool_results(
                q,
                0,
                &mut ts,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
            d.dev.destroy_query_pool(q, None);
        }
        let seconds = (ts[1] - ts[0]) as f64 * d.period as f64 * 1e-9 / iterations as f64;
        let result = y.download()?;
        // Keep weights, inputs and both outputs alive until the Device is dropped;
        // its cached weight descriptors must never see a recycled VkBuffer handle.
        let reference = d.buffer(n * batch * 4, false, false, None)?;
        let compare_bits = (2..=8).contains(&batch) && shader != "gemm";
        if shader.starts_with("qwen_q4c") {
            // Each column must match the one-column decode kernel for this shape bit for bit.
            let single = crate::vulkan::q4_decode_shader(n, k);
            let cmd = d.begin()?;
            for col in 0..batch {
                let mut rp = p;
                rp.ptr[1] = x.address + (col * k * 4) as u64;
                rp.ptr[2] = reference.address + (col * n * 4) as u64;
                d.dispatch_matrix(cmd, single, rp, n.div_ceil(16) as u32, 1, &w)?;
            }
            d.finish(cmd)?;
            d.run(cmd)?;
            ensure!(result == reference.download()?, "matrix output bits differ from {single} per column: GPU {gpu}, {tensor}, batch {batch}, {shader}");
        } else if compare_bits {
            let mut rp = p;
            rp.ptr[2] = reference.address;
            let cmd = d.begin()?;
            d.dispatch_matrix(
                cmd,
                "multivec4_reference",
                rp,
                n.div_ceil(16) as u32,
                batch.div_ceil(4) as u32,
                &w,
            )?;
            d.finish(cmd)?;
            d.run(cmd)?;
            ensure!(result == reference.download()?, "matrix output bits differ from original multivec4: GPU {gpu}, {tensor}, batch {batch}, {shader}");
        }
        for row in [0, n / 2, n - 1] {
            for col in 0..batch {
                let mut expected = 0f64;
                if kind == 1 {
                    for j in 0..k {
                        let scale_at = (row * k / 16 + j / 16) * 4;
                        let scale =
                            f32::from_le_bytes(data[scale_at..scale_at + 4].try_into()?) as f64;
                        let q = data[n * k / 4 + row * k + j] as i8;
                        expected += scale * q as f64 * inputs[col * k + j] as f64;
                    }
                } else {
                    for b in 0..k / 32 {
                        let block = row * k / 32 + b;
                        let scale =
                            f32::from_le_bytes(data[block * 4..block * 4 + 4].try_into()?) as f64;
                        for (j, q) in data
                            [n * k / 32 * 4 + block * 16..n * k / 32 * 4 + (block + 1) * 16]
                            .iter()
                            .enumerate()
                        {
                            expected += scale
                                * ((q & 15) as i32 - 8) as f64
                                * inputs[col * k + b * 32 + j] as f64;
                            expected += scale
                                * ((q >> 4) as i32 - 8) as f64
                                * inputs[col * k + b * 32 + j + 16] as f64;
                        }
                    }
                }
                let at = (col * n + row) * 4;
                let got = f32::from_le_bytes(result[at..at + 4].try_into()?);
                ensure!(
                    (got as f64 - expected).abs() < 0.002 + 0.0002 * expected.abs(),
                    "matvec row {row} col {col}: {got} vs {expected}"
                );
            }
        }
        println!(
            "{}",
            serde_json::json!({"gpu":gpu,"matrix_GFLOP_s":(n*k*batch*2) as f64/seconds/1e9,"kernel":shader,"tensor":tensor,"batch":batch,"rows":n,"cols":k,"milliseconds":seconds*1e3,"repacked_GB_s":data.len()as f64/seconds/1e9,"original_GB_s":(n*k/if kind == 0 {32} else {256}*if kind == 0 {18} else {210})as f64/seconds/1e9,"iterations":iterations,"warmup_ms":warmup_ms,"reference_bit_identical":compare_bits,"cpu_rows_checked":3})
        );
    }
    Ok(())
}

pub fn selection(iterations: u32, batch: usize) -> Result<()> {
    ensure!(
        iterations > 0 && (1..=8).contains(&batch),
        "iterations positive, batch 1..8"
    );
    // An incomplete final chunk exercises padding, and repeated scores exercise token-ID ties.
    let n = 131061usize;
    let offset = 131072u32;
    let suppressed = [offset + n as u32 / 3, offset + n as u32 / 3 + 1];
    let chunks = n.div_ceil(1024);
    let inputs: Vec<f32> = (0..n * batch)
        .map(|i| {
            let id = i % n;
            if suppressed.contains(&(id as u32 + offset)) {
                32.
            } else {
                ((id * 73 + i / n * 31) % 4096) as f32 / 128. - 16.
            }
        })
        .collect();
    let inst = Rc::new(Instance::new()?);
    for (gpu, pd) in inst.amd_gpus()?.into_iter().enumerate() {
        let d = Device::new(inst.clone(), pd)?;
        let x = d.upload(bytemuck::cast_slice(&inputs))?;
        let scratch = d.buffer(chunks * 64 * batch * 8, false, false, None)?;
        let output = d.buffer(64 * batch * 8, false, false, None)?;
        let params = d.upload(bytemuck::cast_slice(&[0u32, batch as u32]))?;
        let q = unsafe {
            d.dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?
        };
        for flags in [0u32, 1, 3] {
            let expected: Vec<Vec<(f32, u32)>> = inputs
                .chunks(n)
                .map(|column| {
                    let mut ranked: Vec<_> = column
                        .iter()
                        .enumerate()
                        .map(|(id, &v)| {
                            let token = offset + id as u32;
                            let v = if flags & 2 != 0 && suppressed.contains(&token) {
                                -1e30
                            } else if flags & 1 != 0 {
                                30. * (v / 30.).tanh()
                            } else {
                                v
                            };
                            (v, token)
                        })
                        .collect();
                    ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                    ranked.truncate(64);
                    ranked
                })
                .collect();
            for k in [1usize, 8, 64] {
                let cmd = d.begin()?;
                unsafe {
                    d.dev.cmd_reset_query_pool(cmd, q, 0, 2);
                    d.dev
                        .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
                }
                for _ in 0..iterations {
                    let mut p = Push::default();
                    p.ptr[0] = x.address;
                    p.ptr[2] = scratch.address;
                    p.ptr[6] = params.address;
                    p.u[0] = n as u32;
                    p.u[1] = k as u32;
                    p.u[2] = (chunks * k) as u32;
                    p.u[10] = offset;
                    p.u[11] = flags;
                    p.u[12] = suppressed[0];
                    p.u[13] = suppressed[1];
                    d.dispatch(cmd, "topk", p, chunks as u32, batch as u32);
                    let mut p = Push::default();
                    p.ptr[0] = scratch.address;
                    p.ptr[2] = output.address;
                    p.ptr[6] = params.address;
                    p.u[0] = (chunks * k) as u32;
                    p.u[1] = k as u32;
                    d.dispatch(cmd, "merge", p, 1, batch as u32);
                }
                unsafe {
                    d.dev
                        .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1);
                }
                d.finish(cmd)?;
                let warmup = Instant::now();
                while warmup.elapsed() < Duration::from_millis(500) {
                    d.run_reusable(cmd)?;
                }
                d.run(cmd)?;
                let mut ts = [0u64; 2];
                unsafe {
                    d.dev.get_query_pool_results(
                        q,
                        0,
                        &mut ts,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )?;
                }
                let milliseconds =
                    (ts[1] - ts[0]) as f64 * d.period as f64 / 1e6 / iterations as f64;
                let bytes = output.download()?;
                for (col, ranks) in expected.iter().enumerate() {
                    for (rank, &(value, token)) in ranks[..k].iter().enumerate() {
                        let at = (col * k + rank) * 8;
                        let got = f32::from_le_bytes(bytes[at..at + 4].try_into()?);
                        let id = u32::from_le_bytes(bytes[at + 4..at + 8].try_into()?);
                        ensure!(id == token && (got - value).abs() < 0.00005,
                            "selection GPU{gpu} flags {flags} k {k} col {col} rank {rank}: ({got},{id}) vs ({value},{token})");
                    }
                }
                println!(
                    "{}",
                    serde_json::json!({"gpu":gpu,"batch":batch,"top_k":k,"flags":flags,"milliseconds":milliseconds,"iterations":iterations,"pass":true})
                );
            }
        }
        unsafe {
            d.dev.destroy_query_pool(q, None);
        }
    }
    Ok(())
}

/// Q4_0 GEMM kernels (`gemm` and its variants) at (rows, k) shapes for `n` tokens on GPU `gpu`, random weights
/// and activations: milliseconds and TFLOP/s per call, and every variant's output checked bit for bit against
/// `gemm`. One JSON line per shape and kernel.
/// Workgroups of a GEMM kernel: the 64 x 64 tiled kernels, or `qwen4_q8_gemm_sTrR[p]` (64 R rows by T tokens, the
/// token tile first), each times `parts` K splits.
fn gemm_groups(kernel: &str, rows: usize, n: usize, parts: usize) -> (usize, usize) {
    let parts = parts.max(1);
    let tr = kernel.strip_prefix("qwen4_q8_gemm_s").and_then(|c| {
        let (t, r) = c.trim_end_matches(['p', 'o', 'd', 'm']).split_once('r')?;
        Some((t.parse::<usize>().ok()?, r.parse::<usize>().ok()?, c.contains('o')))
    });
    match tr {
        Some((t, r, false)) => (n.div_ceil(t), rows.div_ceil(64 * r) * parts),
        Some((t, r, true)) => (rows.div_ceil(64 * r) * parts, n.div_ceil(t)),
        None => (rows.div_ceil(64) * parts, n.div_ceil(64)),
    }
}

/// Kernels named `qwen4_q8_*` read random Q8_0 weights (`quant::repack_q8h`), the others Q4_0; `KERNEL:PARTS` splits K.
pub fn gemm(shapes: &[(usize, usize)], n: usize, kernels: &[String], iterations: u32, gpu: usize, warmup_ms: u64) -> Result<()> {
    ensure!(iterations > 0 && n > 0, "positive iterations and tokens");
    let inst = Rc::new(Instance::new()?);
    let pd = *inst.amd_gpus()?.get(gpu).context("GPU not found")?;
    let d = Device::new(inst.clone(), pd)?;
    let mut state = 0x2545_f491u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    // Buffers stay alive until the device goes: its cached descriptors must never see a recycled handle.
    let mut keep = Vec::new();
    let q = unsafe { d.dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)? };
    for &(rows, k) in shapes {
        ensure!(rows.is_multiple_of(64) && k.is_multiple_of(32), "rows must be a multiple of 64 and k of 32");
        let q8 = kernels.iter().any(|k| k.starts_with("qwen4_q8"));
        let bytes = if q8 { 34 } else { 18 };
        let mut blocks = vec![0u8; rows * k / 32 * bytes];
        for b in blocks.chunks_exact_mut(bytes) {
            let scale = half::f16::from_f32(0.001 + (next() % 1000) as f32 * 5e-5);
            b[..2].copy_from_slice(&scale.to_le_bytes());
            for byte in &mut b[2..] {
                *byte = next() as u8;
            }
        }
        let (w, stride) = if q8 {
            let (packed, stride) = quant::repack_q8h(&blocks);
            (d.upload_matrix(&packed)?, stride as u32)
        } else {
            (d.upload_matrix(&quant::repack_q4(&blocks, k, rows, 0, rows, 0, k))?, (rows * k / 32) as u32)
        };
        let input: Vec<f32> = (0..n * k).map(|_| (next() as f64 / u32::MAX as f64 * 2. - 1.) as f32).collect();
        let x = d.upload(bytemuck::cast_slice(&input))?;
        // The normed streams the mixer up projections (`_mix`, `...m`) fold into their output.
        let normed: Vec<f32> = (0..n * rows).map(|_| (next() as f64 / u32::MAX as f64 * 2. - 1.) as f32).collect();
        let xn = d.upload(bytemuck::cast_slice(&normed))?;
        let mut reference: Option<Vec<u8>> = None;
        for spec in kernels {
            let (kernel, parts) = match spec.split_once(':') {
                Some((kn, pp)) => (kn, pp.parse::<usize>()?),
                None => (spec.as_str(), 0),
            };
            let (gx, gy) = gemm_groups(kernel, rows, n, parts);
            // Zeroed, so kernels that write part of it (the mixer up projections) compare equal elsewhere.
            let y = d.upload(&vec![0u8; n * rows * 4 * parts.max(1)])?;
            let mut p = Push::default();
            p.ptr[1] = x.address;
            p.ptr[2] = y.address;
            p.ptr[3] = xn.address;
            p.u[0] = rows as u32;
            p.u[1] = k as u32;
            p.u[2] = parts as u32;
            p.u[3] = stride;
            p.u[15] = n as u32;
            let cmd = d.begin()?;
            unsafe {
                d.dev.cmd_reset_query_pool(cmd, q, 0, 2);
                d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
            }
            for _ in 0..iterations {
                d.dispatch_matrix(cmd, kernel, p, gx as u32, gy as u32, &w)?;
            }
            unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1) };
            d.finish(cmd)?;
            let warmup = Instant::now();
            while warmup.elapsed() < Duration::from_millis(warmup_ms) {
                d.run_reusable(cmd)?;
            }
            d.run(cmd)?;
            let mut ts = [0u64; 2];
            unsafe { d.dev.get_query_pool_results(q, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
            let ms = (ts[1] - ts[0]) as f64 * d.period as f64 * 1e-6 / iterations as f64;
            let out = y.download()?;
            let (mismatches, max_diff) = match &reference {
                None => (0, 0f32),
                Some(r) => {
                    let (a, b): (&[f32], &[f32]) = (bytemuck::cast_slice(r), bytemuck::cast_slice(&out));
                    let m = a.iter().zip(b).filter(|(u, v)| u.to_bits() != v.to_bits()).count();
                    (m, a.iter().zip(b).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max))
                }
            };
            if reference.is_none() {
                reference = Some(out);
            }
            let tflops = 2. * (rows * k * n) as f64 / (ms * 1e-3) / 1e12;
            println!("{}", serde_json::json!({"gpu": gpu, "rows": rows, "k": k, "n": n, "kernel": spec, "ms": ms, "tflops": tflops, "mismatches": mismatches, "max_diff": max_diff}));
            keep.push(y);
        }
        keep.push(w);
        keep.push(x);
        keep.push(xn);
    }
    unsafe { d.dev.destroy_query_pool(q, None) };
    drop(keep);
    Ok(())
}

/// Rows one workgroup of a decode or verification Q4_0 kernel covers (rows per wave times waves), or None for a
/// GEMM kernel (64-row tiles).
fn rows_per_workgroup(kernel: &str) -> Option<usize> {
    Some(match kernel {
        "matvecq4" | "matvecq4narrow" | "multivec4" | "qwen_q4c2" | "qwen_q4c3" | "qwen_q4c4" => 16,
        "matvecq4_r2" | "matvecq4_r2u2" => 8,
        _ => return None,
    })
}

/// Time Q4_0 decode and verification kernels (1 to 8 columns) at ROWSxK shapes with random weights, checking every
/// kernel against the first bit for bit. `group` dispatches share one trailing barrier (the first `group - 1` defer
/// it), as a layer's independent projections do, so their overlap is measured too; each timed unit is one group.
pub fn matvec(shapes: &[(usize, usize)], n: usize, kernels: &[String], iterations: u32, gpu: usize, warmup_ms: u64, group: usize) -> Result<()> {
    ensure!(iterations > 0 && (1..=8).contains(&n) && group > 0, "positive iterations and group, 1 to 8 columns");
    let inst = Rc::new(Instance::new()?);
    let pd = *inst.amd_gpus()?.get(gpu).context("GPU not found")?;
    let d = Device::new(inst.clone(), pd)?;
    let mut state = 0x2545_f491u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut keep = Vec::new();
    let q = unsafe { d.dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None)? };
    for &(rows, k) in shapes {
        ensure!(rows.is_multiple_of(16) && k.is_multiple_of(32), "rows must be a multiple of 16 and k of 32");
        let mut blocks = vec![0u8; rows * k / 32 * 18];
        for b in blocks.chunks_exact_mut(18) {
            let scale = half::f16::from_f32(0.001 + (next() % 1000) as f32 * 5e-5);
            b[..2].copy_from_slice(&scale.to_le_bytes());
            for byte in &mut b[2..] {
                *byte = next() as u8;
            }
        }
        // One matrix per dispatch of the group, so a group streams as many bytes as the projections it stands for.
        let ws: Vec<_> = (0..group).map(|_| d.upload_matrix(&quant::repack_q4(&blocks, k, rows, 0, rows, 0, k))).collect::<Result<_>>()?;
        let input: Vec<f32> = (0..n * k).map(|_| (next() as f64 / u32::MAX as f64 * 2. - 1.) as f32).collect();
        let x = d.upload(bytemuck::cast_slice(&input))?;
        let mut reference: Option<Vec<u8>> = None;
        for kernel in kernels {
            let per = rows_per_workgroup(kernel);
            let (gx, gy) = match per {
                Some(r) => (rows.div_ceil(r), if kernel.starts_with("multivec4") { n.div_ceil(4) } else { 1 }),
                None => (rows.div_ceil(64), n.div_ceil(64)),
            };
            ensure!(per.is_none() || kernel.starts_with("qwen_q4c") || kernel.starts_with("multivec") || n == 1, "{kernel} takes one column");
            let ys: Vec<_> = (0..group).map(|_| d.buffer(n * rows * 4, false, false, None)).collect::<Result<_>>()?;
            let cmd = d.begin()?;
            unsafe {
                d.dev.cmd_reset_query_pool(cmd, q, 0, 2);
                d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, q, 0);
            }
            for _ in 0..iterations {
                for (g, (w, y)) in ws.iter().zip(&ys).enumerate() {
                    let mut p = Push::default();
                    p.ptr[1] = x.address;
                    p.ptr[2] = y.address;
                    p.u[0] = rows as u32;
                    p.u[1] = k as u32;
                    p.u[3] = (rows * k / 32) as u32;
                    p.u[15] = n as u32;
                    d.dispatch_matrix(cmd, kernel, p.defer_barrier(g + 1 < group), gx as u32, gy as u32, w)?;
                }
            }
            unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 1) };
            d.finish(cmd)?;
            let warmup = Instant::now();
            while warmup.elapsed() < Duration::from_millis(warmup_ms) {
                d.run_reusable(cmd)?;
            }
            let mut samples = Vec::new();
            for _ in 0..5 {
                d.run(cmd)?;
                let mut ts = [0u64; 2];
                unsafe { d.dev.get_query_pool_results(q, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
                samples.push((ts[1] - ts[0]) as f64 * d.period as f64 * 1e-3 / iterations as f64);
            }
            samples.sort_by(f64::total_cmp);
            let us = samples[samples.len() / 2];
            let out = ys[0].download()?;
            let (mismatches, max_diff) = match &reference {
                None => (0, 0f32),
                Some(r) => {
                    let (a, b): (&[f32], &[f32]) = (bytemuck::cast_slice(r), bytemuck::cast_slice(&out));
                    let m = a.iter().zip(b).filter(|(u, v)| u.to_bits() != v.to_bits()).count();
                    (m, a.iter().zip(b).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max))
                }
            };
            if reference.is_none() {
                reference = Some(out);
            }
            let bytes = (rows * k / 32 * 20 * group) as f64;
            println!("{}", serde_json::json!({"gpu": gpu, "rows": rows, "k": k, "n": n, "group": group, "kernel": kernel, "us": us, "gb_s": bytes / (us * 1e-6) / 1e9, "mismatches": mismatches, "max_diff": max_diff}));
            keep.extend(ys);
        }
        keep.extend(ws);
        keep.push(x);
    }
    unsafe { d.dev.destroy_query_pool(q, None) };
    drop(keep);
    Ok(())
}

/// Time split-K decode attention (partial + merge) at a tensor-parallel rank's shape: `heads` query heads over
/// `kv` KV heads of 256, `n` columns after `pos` cached keys, `parts` partitions; each merge kernel named in
/// `merges` is checked against the first bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn decode_attention(pos: usize, n: usize, heads: usize, kv: usize, parts: u32, merges: &[String], iterations: u32, gpu: usize) -> Result<()> {
    const DIM: usize = 256;
    ensure!(n > 0 && n <= 8 && heads > 0 && heads.is_multiple_of(kv) && iterations > 0, "invalid decode attention shape");
    let inst = Rc::new(Instance::new()?);
    let pd = *inst.amd_gpus()?.get(gpu).context("GPU not found")?;
    let d = Device::new(inst.clone(), pd)?;
    let cap = pos + n + 3;
    let mut state = 0x9e37_79b9u32;
    let mut sample = || { state ^= state << 13; state ^= state >> 17; state ^= state << 5; (state as f64 / u32::MAX as f64 * 2. - 1.) as f32 };
    let q: Vec<f32> = (0..n * heads * DIM).map(|_| sample() * 0.25).collect();
    let k16: Vec<u16> = (0..cap * kv * DIM).map(|_| half::f16::from_f32(sample() * 2.0).to_bits()).collect();
    let v16: Vec<u16> = (0..cap * kv * DIM).map(|_| half::f16::from_f32(sample()).to_bits()).collect();
    let qb = d.upload(bytemuck::cast_slice(&q))?;
    let kb = d.upload(bytemuck::cast_slice(&k16))?;
    let vb = d.upload(bytemuck::cast_slice(&v16))?;
    let params = d.upload(bytemuck::cast_slice(&[pos as u32, n as u32, 0u32, 0]))?;
    let out = d.buffer(n * heads * DIM * 4, false, false, None)?;
    let scratch = d.buffer(n * heads * parts as usize * (DIM + 2) * 4, false, false, None)?;
    let (partial, _) = crate::vulkan::attention_shaders(parts)?;
    let mut p = Push { ptr: [qb.address, kb.address, vb.address, out.address, scratch.address, 0, params.address, 0], ..Default::default() };
    p.u[4] = heads as u32;
    p.u[5] = kv as u32;
    p.u[6] = DIM as u32;
    p.u[7] = cap as u32;
    let qp = unsafe { d.dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(4), None)? };
    let mut reference: Option<Vec<u8>> = None;
    for merge in merges {
        let cmd = d.begin()?;
        unsafe {
            d.dev.cmd_reset_query_pool(cmd, qp, 0, 4);
            d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
        }
        for _ in 0..iterations {
            d.dispatch(cmd, partial, p, heads as u32 * parts, n as u32);
        }
        unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1) };
        unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 2) };
        for _ in 0..iterations {
            d.dispatch(cmd, merge, p, heads as u32, n as u32);
        }
        unsafe { d.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 3) };
        d.finish(cmd)?;
        let warmup = Instant::now();
        while warmup.elapsed() < Duration::from_millis(300) {
            d.run_reusable(cmd)?;
        }
        let mut samples = Vec::new();
        for _ in 0..5 {
            d.run(cmd)?;
            let mut ts = [0u64; 4];
            unsafe { d.dev.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)? };
            let us = |a: u64, b: u64| (b - a) as f64 * d.period as f64 * 1e-3 / iterations as f64;
            samples.push((us(ts[0], ts[1]), us(ts[2], ts[3])));
        }
        samples.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (partial_us, merge_us) = samples[samples.len() / 2];
        let bytes = out.download()?;
        let mismatches = match &reference {
            None => 0,
            Some(r) => bytemuck::cast_slice::<u8, f32>(r).iter().zip(bytemuck::cast_slice::<u8, f32>(&bytes)).filter(|(a, b)| a.to_bits() != b.to_bits()).count(),
        };
        if reference.is_none() {
            reference = Some(bytes);
        }
        println!("{}", serde_json::json!({"gpu": gpu, "pos": pos, "n": n, "heads": heads, "kv": kv, "parts": parts, "partial": partial, "merge": merge, "partial_us": partial_us, "merge_us": merge_us, "mismatches": mismatches}));
    }
    unsafe { d.dev.destroy_query_pool(qp, None) };
    drop((qb, kb, vb, params, out, scratch));
    Ok(())
}

/// Deterministic attention checks for Qwen's GQA ratio six, padded tiles, tails, and later positions.
pub fn qwen_attention(tokens: usize, n: usize, heads: usize, iterations: usize) -> Result<()> {
    ensure!(tokens >= n && n > 0 && heads > 0 && heads <= 64 && iterations > 0, "invalid attention shape");
    let inst = Rc::new(Instance::new()?);
    for (gpu, pd) in inst.amd_gpus()?.into_iter().enumerate() {
        let d = Device::new(inst.clone(), pd)?;
        let mut state = 1234567u32;
        let mut sample = || { state ^= state << 13; state ^= state >> 17; state ^= state << 5; (state as f64 / u32::MAX as f64 * 2. - 1.) as f32 };
        let queries: Vec<_> = (0..n * heads * 256).map(|_| sample() / 16.).collect();
        let keys: Vec<_> = (0..tokens * 256).map(|_| half::f16::from_f32(sample()).to_bits()).collect();
        let values: Vec<_> = (0..tokens * 256).map(|_| half::f16::from_f32(sample()).to_bits()).collect();
        let q = d.upload(bytemuck::cast_slice(&queries))?;
        let k = d.upload(bytemuck::cast_slice(&keys))?;
        let v = d.upload(bytemuck::cast_slice(&values))?;
        let params = d.upload(bytemuck::cast_slice(&[(tokens - n) as u32, n as u32, 0u32, 0]))?;
        let out = d.buffer(n * heads * 256 * 4, false, false, None)?;
        let stride = tokens.next_multiple_of(64);
        let scratch = d.buffer(64 * heads * (stride + 1) * 4, false, false, None)?;
        let mut p = Push { ptr: [q.address, k.address, v.address, out.address, scratch.address, scratch.address + (64 * heads * stride * 4) as u64, params.address, 0], ..Default::default() };
        p.u[0] = stride as u32; p.u[1] = heads as u32; p.u[4] = heads as u32;
        p.u[5] = 1; p.u[6] = 256; p.u[7] = tokens as u32; p.u[15] = n as u32;
        let mut reference = None;
        for (variant, tiled) in [("scalar", false), ("tiled64", true)] {
            let cmd = d.begin()?;
            for _ in 0..iterations {
                if tiled {
                    let tq = 1usize << (64 / heads).ilog2();
                    for first in (0..n).step_by(64) {
                        p.u[13] = first as u32;
                        let nb = (n - first).min(64);
                        d.dispatch(cmd, "qwen_attn_scores", p, stride.div_ceil(64) as u32, nb.div_ceil(tq) as u32);
                        d.dispatch(cmd, "qwen_attn_softmax", p, heads as u32, nb as u32);
                        d.dispatch(cmd, "qwen_attn_values", p, 4, nb.div_ceil(tq) as u32);
                    }
                } else { d.dispatch(cmd, "attention", p, heads as u32, n as u32); }
            }
            d.finish(cmd)?;
            d.run_reusable(cmd)?;
            let start = Instant::now(); d.run(cmd)?;
            let ms = start.elapsed().as_secs_f64() * 1e3 / iterations as f64;
            let bytes = out.download()?;
            let floats: &[f32] = bytemuck::cast_slice(&bytes);
            let error = match &reference {
                None => { reference = Some(floats.to_vec()); 0f32 },
                Some(r) => floats.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max),
            };
            ensure!(floats.iter().all(|v| v.is_finite()) && error < 2e-5, "attention mismatch GPU{gpu}: {error}");
            println!("{}", serde_json::json!({"gpu":gpu,"tokens":tokens,"n":n,"heads":heads,"tiled":tiled,"variant":variant,"ms":ms,"max_abs_error":error,"pass":true}));
        }
    }
    Ok(())
}
