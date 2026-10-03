use crate::{
    gguf::Gguf,
    gpu::Instance,
    quant,
    vulkan::{Device, Push},
};
use anyhow::{ensure, Result};
use ash::vk;
use std::{path::Path, rc::Rc};
pub fn bench(path: &Path, iterations: u32, batch: usize, kernel: Option<&str>) -> Result<()> {
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
    let t = g.tensor("blk.0.ffn_gate.weight").unwrap();
    let (k, n) = (3840, 7680);
    let data = quant::repack_q4(g.data(t), k, n * 2, 0, n, 0, k);
    let inst = Rc::new(Instance::new()?);
    for pd in inst.amd_gpus()? {
        let d = Device::new(inst.clone(), pd)?;
        let w = d.upload_matrix(&data)?;
        let x = d.upload(bytemuck::cast_slice(&vec![1f32; k * batch]))?;
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
        dispatch(cmd)?;
        d.finish(cmd)?;
        d.run(cmd)?;
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
            let mut expected = 0f32;
            for block in row * k / 32..(row + 1) * k / 32 {
                let scale = f32::from_le_bytes(data[block * 4..block * 4 + 4].try_into()?);
                for q in &data[n * k / 32 * 4 + block * 16..n * k / 32 * 4 + (block + 1) * 16] {
                    expected += scale * ((q & 15) as i32 - 8) as f32;
                    expected += scale * ((q >> 4) as i32 - 8) as f32;
                }
            }
            for col in 0..batch {
                let at = (col * n + row) * 4;
                let got = f32::from_le_bytes(result[at..at + 4].try_into()?);
                ensure!(
                    (got - expected).abs() < 0.02,
                    "matvec row {row} col {col}: {got} vs {expected}"
                );
            }
        }
        println!(
            "{}",
            serde_json::json!({"matrix_GFLOP_s":(n*k*batch*2) as f64/seconds/1e9,"kernel":shader,"batch":batch,"rows":n,"cols":k,"milliseconds":seconds*1e3,"repacked_GB_s":data.len()as f64/seconds/1e9,"original_GB_s":(n*k/32*18)as f64/seconds/1e9,"iterations":iterations})
        );
    }
    Ok(())
}
