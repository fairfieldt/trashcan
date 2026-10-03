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
pub fn bench(
    path: &Path,
    iterations: u32,
    batch: usize,
    kernel: Option<&str>,
    tensor: &str,
    warmup_ms: u64,
) -> Result<()> {
    ensure!(
        iterations > 0 && (1..=512).contains(&batch),
        "iterations positive, batch 1..512"
    );
    let shader = kernel.unwrap_or(if batch == 1 {
        "matvecq4"
    } else if batch <= 4 {
        "multivec4"
    } else if batch <= 8 {
        "multivec"
    } else {
        "gemm"
    });
    ensure!(
        ["matvecq4", "multivec", "multivec2", "multivec4", "gemm"].contains(&shader),
        "unsupported kernel"
    );
    ensure!(
        batch == 1 || shader != "matvecq4",
        "single-column kernel needs batch 1"
    );
    let group_rows = if shader == "gemm" {
        64
    } else if matches!(shader, "matvecq4" | "multivec4") {
        16
    } else {
        4
    };
    let group_cols = if shader == "gemm" {
        64
    } else if shader == "multivec4" {
        4
    } else if shader == "multivec2" {
        2
    } else {
        8
    };
    let g = Gguf::open(path)?;
    let t = g.tensor(tensor).context("benchmark tensor not found")?;
    ensure!(
        t.ty == GgmlType::Q4_0 && t.dims.len() == 2,
        "benchmark needs a Q4_0 matrix"
    );
    let (full_k, full_n) = (t.dims[0] as usize, t.dims[1] as usize);
    let column_split =
        tensor.ends_with("ffn_down.weight") || tensor.ends_with("attn_output.weight");
    let replicated = tensor.ends_with("attn_k.weight") || tensor.ends_with("attn_v.weight");
    // Global KV is replicated; sliding KV and Q/gate/up are split by rows.
    let (k, n) = if column_split {
        (full_k / 2, full_n)
    } else if replicated && full_n == 512 {
        (full_k, full_n)
    } else {
        (full_k, full_n / 2)
    };
    ensure!(
        k.is_multiple_of(32) && n.is_multiple_of(group_rows),
        "unsupported benchmark shape"
    );
    let data = quant::repack_q4(g.data(t), full_k, full_n, 0, n, 0, k);
    let inputs: Vec<f32> = (0..k * batch)
        .map(|i| ((i * 13 + i / k * 7) % 29) as f32 / 16. - 0.875)
        .collect();
    let inst = Rc::new(Instance::new()?);
    for pd in inst.amd_gpus()? {
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
        p.u[3] = (n * k / 32) as u32;
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
        for row in [0, n / 2, n - 1] {
            for col in 0..batch {
                let mut expected = 0f64;
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
            serde_json::json!({"matrix_GFLOP_s":(n*k*batch*2) as f64/seconds/1e9,"kernel":shader,"tensor":tensor,"batch":batch,"rows":n,"cols":k,"milliseconds":seconds*1e3,"repacked_GB_s":data.len()as f64/seconds/1e9,"original_GB_s":(n*k/32*18)as f64/seconds/1e9,"iterations":iterations,"warmup_ms":warmup_ms})
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
