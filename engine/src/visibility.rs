use crate::{
    gpu::Instance,
    vulkan::{Device, Push},
};
use anyhow::{ensure, Result};
use ash::vk;
use std::rc::Rc;
/// Exercise receiver L2 repeatedly with fresh payloads on the same dma-buf slot.
pub fn test(rounds: u32, elements: usize) -> Result<()> {
    ensure!(
        rounds > 0 && elements > 0,
        "rounds/elements must be positive"
    );
    let inst = Rc::new(Instance::new()?);
    let pd = inst.amd_gpus()?;
    ensure!(pd.len() == 2, "two GPUs required");
    let mut gpu = Vec::new();
    for p in pd {
        gpu.push(Device::new(inst.clone(), p)?);
    }
    let mut recv = Vec::new();
    let mut source = Vec::new();
    let mut params = Vec::new();
    let mut checks = Vec::new();
    let mut own = Vec::new();
    let mut peer = Vec::new();
    let mut fences = Vec::new();
    for d in &gpu {
        recv.push(d.external_buffer(elements * 4, None)?);
        source.push(d.buffer(elements * 4, false, false, None)?);
        params.push(d.buffer(4, true, false, None)?);
        checks.push(d.buffer(elements.div_ceil(256) * 4, true, false, None)?);
        own.push(d.timeline(true)?);
        peer.push(d.timeline(false)?);
        fences.push(unsafe { d.dev.create_fence(&vk::FenceCreateInfo::default(), None)? });
    }
    let mut imports = Vec::new();
    for g in 0..2 {
        imports.push(gpu[g].external_buffer(recv[1 - g].size, Some(recv[1 - g].export()?))?);
        let ex = ash::khr::external_semaphore_fd::Device::new(&inst.instance, &gpu[1 - g].dev);
        let fd = unsafe {
            ex.get_semaphore_fd(
                &vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(own[1 - g])
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
            )?
        };
        let im = ash::khr::external_semaphore_fd::Device::new(&inst.instance, &gpu[g].dev);
        unsafe {
            im.import_semaphore_fd(
                &vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(peer[g])
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                    .fd(fd),
            )?;
        }
    }
    let mut send = Vec::new();
    let mut check = Vec::new();
    for g in 0..2 {
        let d = &gpu[g];
        let cmd = d.begin()?;
        let mut p = Push::default();
        p.ptr[0] = source[g].address;
        p.ptr[6] = params[g].address;
        p.u[0] = elements as u32;
        d.dispatch_receive(
            cmd,
            "visibility",
            p,
            elements.div_ceil(256) as u32,
            1,
            &source[g],
        )?;
        d.copy(cmd, &source[g], 0, &imports[g], 0, elements * 4);
        d.finish(cmd)?;
        send.push(cmd);
        let cmd = d.begin()?;
        d.barrier(cmd);
        p.ptr[0] = recv[g].address;
        p.ptr[2] = checks[g].address;
        p.u[9] = 1;
        d.dispatch_receive(
            cmd,
            "visibility",
            p,
            elements.div_ceil(256) as u32,
            1,
            &recv[g],
        )?;
        d.finish(cmd)?;
        check.push(cmd);
    }
    for round in 1..=rounds {
        for p in &params {
            p.write(bytemuck::bytes_of(&round))?;
        }
        for g in 0..2 {
            gpu[g].submit(
                send[g],
                None,
                Some((own[g], round as u64)),
                vk::Fence::null(),
            )?;
        }
        for g in 0..2 {
            gpu[g].submit(check[g], Some((peer[g], round as u64)), None, fences[g])?;
        }
        for g in 0..2 {
            unsafe {
                gpu[g]
                    .dev
                    .wait_for_fences(&[fences[g]], true, 60_000_000_000)?;
                gpu[g].dev.reset_fences(&[fences[g]])?;
            }
            let data = checks[g].read(elements.div_ceil(256) * 4)?;
            ensure!(
                data.iter().all(|x| *x == 0),
                "stale P2P payload at round {round} GPU{g}"
            );
        }
    }
    for g in 0..2 {
        unsafe {
            gpu[g]
                .dev
                .free_command_buffers(gpu[g].pool, &[send[g], check[g]]);
            gpu[g].dev.destroy_fence(fences[g], None);
            gpu[g].dev.destroy_semaphore(own[g], None);
            gpu[g].dev.destroy_semaphore(peer[g], None);
        }
    }
    println!(
        "P2P visibility passed: {rounds} changing rounds x {elements} elements x two directions"
    );
    Ok(())
}
