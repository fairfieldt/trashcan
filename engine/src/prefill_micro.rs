//! Causal/window/ring and padding checks for the bounded prefill path.
use crate::{
    gpu::Instance,
    vulkan::{prefill_attention, Device, Push},
};
use anyhow::{ensure, Result};
use ash::vk;
use std::{
    rc::Rc,
    time::{Duration, Instant},
};

pub fn bench(position: usize, batch: usize, iterations: u32, sliding: bool) -> Result<()> {
    ensure!(
        (1..=512).contains(&batch) && iterations > 0 && position + batch <= 131072,
        "batch 1..512, positive iterations, end <=131072"
    );
    let (dim, kv) = if sliding {
        (256usize, 4usize)
    } else {
        (512, 1)
    };
    let cap = if sliding {
        1024 + batch
    } else {
        position + batch + 64
    };
    let stride = (if sliding { 1055usize } else { cap }).div_ceil(64) * 64 + 16;
    let mut state = 0x12345678u32;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f64 / u32::MAX as f64 * 2. - 1.) as f32
    };
    let inputs: Vec<f32> = (0..batch * 8 * dim).map(|_| random() * 0.1).collect();
    // Unwritten slots contain NaN. Future/padded scores must be masked before
    // exp, and padded V loads must be zero rather than evaluating 0 * NaN.
    let mut keys = vec![half::f16::NAN.to_bits(); cap * kv * dim];
    let mut values = keys.clone();
    for key in (position + batch).saturating_sub(cap)..position + batch {
        for j in 0..kv * dim {
            let i = (key % cap) * kv * dim + j;
            keys[i] = half::f16::from_f32(random()).to_bits();
            values[i] = half::f16::from_f32(random()).to_bits();
        }
    }
    let mut expected = Vec::new();
    for t in [0, batch / 2, batch - 1] {
        let pos = position + t;
        let start = if sliding {
            (pos + 1).saturating_sub(1024)
        } else {
            0
        };
        for head in [0, 7] {
            let kh = head / (8 / kv);
            let scores: Vec<f64> = (start..=pos)
                .map(|key| {
                    (0..dim)
                        .map(|j| {
                            inputs[(t * 8 + head) * dim + j] as f64
                                * half::f16::from_bits(keys[((key % cap) * kv + kh) * dim + j])
                                    .to_f64()
                        })
                        .sum()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let den: f64 = weights.iter().sum();
            for j in [0, dim / 2, dim - 1] {
                let v: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        w * half::f16::from_bits(values[(((start + i) % cap) * kv + kh) * dim + j])
                            .to_f64()
                    })
                    .sum();
                expected.push(((t * 8 + head) * dim + j, v / den));
            }
        }
    }
    let inst = Rc::new(Instance::new()?);
    for (gpu, pd) in inst.amd_gpus()?.into_iter().enumerate() {
        let d = Device::new(inst.clone(), pd)?;
        let q = d.upload(bytemuck::cast_slice(&inputs))?;
        let k = d.upload(bytemuck::cast_slice(&keys))?;
        let v = d.upload(bytemuck::cast_slice(&values))?;
        let poison = vec![f32::NAN; inputs.len()];
        let scalar_out = d.upload(bytemuck::cast_slice(&poison))?;
        let tiled_out = d.upload(bytemuck::cast_slice(&poison))?;
        let scores = d.upload(bytemuck::cast_slice(&vec![
            f32::NAN;
            batch.min(64) * 8 * stride
        ]))?;
        let den = d.buffer(batch.min(64) * 8 * 4, false, false, None)?;
        let params = d.upload(bytemuck::cast_slice(&[position as u32, batch as u32, 0, 0]))?;
        let query = unsafe {
            d.dev.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?
        };
        let mut reference: Option<Vec<f32>> = None;
        for (shader, out) in [("scalar", &scalar_out), ("tiled", &tiled_out)] {
            let mut p = Push {
                ptr: [
                    q.address,
                    k.address,
                    v.address,
                    out.address,
                    scores.address,
                    den.address,
                    params.address,
                    0,
                ],
                ..Push::default()
            };
            p.u[0] = stride as u32;
            p.u[4] = 8;
            p.u[5] = kv as u32;
            p.u[6] = dim as u32;
            p.u[7] = cap as u32;
            p.u[8] = if sliding { 1024 } else { 0 };
            let cmd = d.begin()?;
            unsafe {
                d.dev.cmd_reset_query_pool(cmd, query, 0, 2);
                d.dev
                    .cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, query, 0);
            }
            if shader == "scalar" {
                d.dispatch(cmd, "attention", p, 8, batch as u32);
            } else {
                prefill_attention(&d, cmd, p, batch);
            }
            unsafe {
                d.dev
                    .cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query, 1);
            }
            d.finish(cmd)?;
            let warm = Instant::now();
            while warm.elapsed() < Duration::from_millis(300) {
                d.run_reusable(cmd)?;
            }
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
                times.push((ticks[1] - ticks[0]) as f64 * d.period as f64 / 1e6);
            }
            let bytes = out.download()?;
            let result: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            ensure!(
                result.iter().all(|x| x.is_finite()),
                "{shader} has nonfinite outputs"
            );
            let cpu_error = expected
                .iter()
                .map(|&(i, w)| (result[i] as f64 - w).abs())
                .fold(0f64, f64::max);
            ensure!(cpu_error < 0.0002, "{shader}: CPU error {cpu_error}");
            let scalar_error = reference.as_ref().map(|r| {
                r.iter()
                    .zip(&result)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max)
            });
            ensure!(
                scalar_error.is_none_or(|x| x < 0.0003),
                "tiled/scalar error {scalar_error:?}"
            );
            if shader == "scalar" {
                reference = Some(result);
            }
            println!(
                "{}",
                serde_json::json!({"gpu":gpu,"kernel":shader,"position":position,"batch":batch,"dim":dim,"kv_heads":kv,"sliding":sliding,"iterations":iterations,"milliseconds":times.iter().sum::<f64>()/times.len() as f64,"times_ms":times,"cpu_checks":expected.len(),"cpu_max_error":cpu_error,"scalar_max_error":scalar_error,"score_bytes":scores.size,"pass":true})
            );
            unsafe {
                d.dev.free_command_buffers(d.pool, &[cmd]);
            }
        }
        unsafe {
            d.dev.destroy_query_pool(query, None);
        }
    }
    Ok(())
}
