//! vkcompute: Vulkan compute checks on one GPU, for comparing gpud with amdgpu on the same card model.
//!
//! 1. host: c = a * b + i on 4M floats in host-visible memory, checked on the CPU.
//! 2. device: the same on device-local buffers, uploaded and read back with transfer commands.
//! 3. bandwidth: a 16-byte-per-thread copy between two 256 MiB device-local buffers, timed with GPU timestamps.
//!
//! usage: vkcompute [--bus N] (picks the device on PCI bus N when several are visible)

use anyhow::{Context, Result, bail, ensure};
use ash::vk;
use std::time::Instant;

const FMA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fma.spv"));
const COPY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy.spv"));

struct Gpu {
    _entry: ash::Entry,
    _instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    mem: vk::PhysicalDeviceMemoryProperties,
    timestamp_period: f32,
}

struct Buffer {
    buffer: vk::Buffer,
    _memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
}

// Objects live until the process exits.
struct Pipeline {
    _set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    _pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl Gpu {
    fn new(bus: Option<u32>) -> Result<Gpu> {
        // SAFETY: loads the system Vulkan loader.
        let entry = unsafe { ash::Entry::load()? };
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
        // SAFETY: valid create info.
        let instance =
            unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)? };
        // SAFETY: valid instance.
        let pdevs = unsafe { instance.enumerate_physical_devices()? };
        let mut chosen = None;
        for &p in &pdevs {
            let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
            let (name, period) = {
                let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
                // SAFETY: valid physical device; RADV supports VK_EXT_pci_bus_info.
                unsafe { instance.get_physical_device_properties2(p, &mut props2) };
                let name = props2.properties.device_name_as_c_str().unwrap_or(c"?").to_string_lossy().into_owned();
                (name, props2.properties.limits.timestamp_period)
            };
            println!(
                "device     {name} at {:04x}:{:02x}:{:02x}.{}",
                pci.pci_domain, pci.pci_bus, pci.pci_device, pci.pci_function
            );
            if chosen.is_none() && bus.is_none_or(|b| b == pci.pci_bus) {
                chosen = Some((p, period));
            }
        }
        let (pdev, timestamp_period) = chosen.context("no matching Vulkan device")?;
        // SAFETY: valid physical device.
        let families = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
        let qfi =
            families.iter().position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE)).context("no compute queue")?
                as u32;
        let prio = [1.0];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfi).queue_priorities(&prio)];
        // SAFETY: valid create info.
        let device =
            unsafe { instance.create_device(pdev, &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None)? };
        // SAFETY: the queue was requested above.
        let queue = unsafe { device.get_device_queue(qfi, 0) };
        // SAFETY: valid device.
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qfi)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        // SAFETY: valid physical device.
        let mem = unsafe { instance.get_physical_device_memory_properties(pdev) };
        println!("queue      family {qfi} ({:?})", families[qfi as usize].queue_flags);
        Ok(Gpu { _entry: entry, _instance: instance, device, queue, pool, mem, timestamp_period })
    }

    fn buffer(&self, size: u64, host: bool) -> Result<Buffer> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        // SAFETY: valid device and create info.
        let buffer =
            unsafe { self.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)? };
        // SAFETY: valid buffer.
        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let want = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let ty = (0..self.mem.memory_type_count)
            .find(|&i| {
                let f = self.mem.memory_types[i as usize].property_flags;
                req.memory_type_bits & (1 << i) != 0
                    && f.contains(want)
                    && (host || !f.contains(vk::MemoryPropertyFlags::HOST_VISIBLE))
            })
            .context("no memory type")?;
        // SAFETY: valid allocation info.
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty),
                None,
            )?
        };
        // SAFETY: buffer and memory belong to this device.
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };
        let ptr = if host {
            // SAFETY: host-visible memory, mapped whole.
            unsafe { self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?.cast() }
        } else {
            std::ptr::null_mut()
        };
        Ok(Buffer { buffer, _memory: memory, ptr, size })
    }

    fn pipeline(&self, spv: &[u8], buffers: &[&Buffer]) -> Result<Pipeline> {
        let d = &self.device;
        let bindings: Vec<_> = (0..buffers.len() as u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
        let pc = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(4)];
        // SAFETY: valid create infos throughout; objects live as long as the Pipeline.
        unsafe {
            let set_layout = d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let sl = [set_layout];
            let layout = d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&sl).push_constant_ranges(&pc),
                None,
            )?;
            let module = d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0];
            d.destroy_shader_module(module, None);
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(buffers.len() as u32)];
            let pool = d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
                None,
            )?;
            let set = d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sl),
            )?[0];
            let infos: Vec<_> = buffers
                .iter()
                .map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buffer).range(vk::WHOLE_SIZE)])
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            d.update_descriptor_sets(&writes, &[]);
            Ok(Pipeline { _set_layout: set_layout, layout, pipeline, _pool: pool, set })
        }
    }

    /// Records with f, submits, waits; returns the wall time of submit-to-signal in milliseconds.
    fn run(&self, f: impl FnOnce(vk::CommandBuffer)) -> Result<f64> {
        let d = &self.device;
        // SAFETY: valid pool; the command buffer is used once and freed.
        unsafe {
            let cb = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1),
            )?[0];
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            f(cb);
            d.end_command_buffer(cb)?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbs = [cb];
            let t0 = Instant::now();
            d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence)?;
            d.wait_for_fences(&[fence], true, 10_000_000_000)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            d.destroy_fence(fence, None);
            d.free_command_buffers(self.pool, &cbs);
            Ok(ms)
        }
    }

    fn barrier(&self, cb: vk::CommandBuffer) {
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
            )];
        // SAFETY: recording command buffer.
        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            )
        };
    }

    fn dispatch(&self, cb: vk::CommandBuffer, p: &Pipeline, n: u32, groups: u32) {
        let d = &self.device;
        // SAFETY: recording command buffer; the pipeline's objects are alive.
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &[p.set], &[]);
            d.cmd_push_constants(cb, p.layout, vk::ShaderStageFlags::COMPUTE, 0, &n.to_le_bytes());
            d.cmd_dispatch(cb, groups, 1, 1);
        }
    }
}

fn fill(b: &Buffer, f: impl Fn(usize) -> f32) {
    let n = b.size as usize / 4;
    // SAFETY: host-visible mapping of b.size bytes.
    let s = unsafe { std::slice::from_raw_parts_mut(b.ptr.cast::<f32>(), n) };
    for (i, v) in s.iter_mut().enumerate() {
        *v = f(i);
    }
}

/// a[i] = i % 1000, b[i] = 0.5: every product and sum is exact in f32 for i < 2^24.
fn check(c: &Buffer) -> Result<()> {
    let n = c.size as usize / 4;
    // SAFETY: host-visible mapping of c.size bytes, read after the fence.
    let s = unsafe { std::slice::from_raw_parts(c.ptr.cast::<f32>(), n) };
    let bad = s.iter().enumerate().filter(|&(i, &v)| v != (i % 1000) as f32 * 0.5 + i as f32).count();
    if bad > 0 {
        let i = s.iter().enumerate().position(|(i, &v)| v != (i % 1000) as f32 * 0.5 + i as f32).unwrap();
        bail!("{bad} of {n} wrong; first at {i}: {} (expected {})", s[i], (i % 1000) as f32 * 0.5 + i as f32);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let bus = args.iter().position(|a| a == "--bus").and_then(|i| args.get(i + 1)).map(|s| s.parse()).transpose()?;
    let gpu = Gpu::new(bus)?;
    const N: u32 = 4 << 20;
    let bytes = N as u64 * 4;
    let groups = N.div_ceil(256);

    // 1. host-visible
    let (a, b, c) = (gpu.buffer(bytes, true)?, gpu.buffer(bytes, true)?, gpu.buffer(bytes, true)?);
    fill(&a, |i| (i % 1000) as f32);
    fill(&b, |_| 0.5);
    fill(&c, |_| -1.0);
    let fma = gpu.pipeline(FMA, &[&a, &b, &c])?;
    let ms = gpu.run(|cb| {
        gpu.dispatch(cb, &fma, N, groups);
        gpu.barrier(cb);
    })?;
    check(&c).context("host-visible kernel")?;
    println!("ok   c = a * b + i on {N} floats in host-visible memory ({ms:.2} ms)");

    // 2. device-local, through transfers
    let (da, db, dc) = (gpu.buffer(bytes, false)?, gpu.buffer(bytes, false)?, gpu.buffer(bytes, false)?);
    fill(&c, |_| -1.0);
    let fma_dev = gpu.pipeline(FMA, &[&da, &db, &dc])?;
    let region = [vk::BufferCopy::default().size(bytes)];
    let ms = gpu.run(|cb| {
        // SAFETY: recording command buffer; buffers are alive.
        unsafe {
            gpu.device.cmd_copy_buffer(cb, a.buffer, da.buffer, &region);
            gpu.device.cmd_copy_buffer(cb, b.buffer, db.buffer, &region);
        }
        gpu.barrier(cb);
        gpu.dispatch(cb, &fma_dev, N, groups);
        gpu.barrier(cb);
        // SAFETY: as above.
        unsafe { gpu.device.cmd_copy_buffer(cb, dc.buffer, c.buffer, &region) };
        gpu.barrier(cb);
    })?;
    check(&c).context("device-local kernel")?;
    println!("ok   the same on device-local buffers, uploaded and read back ({ms:.2} ms)");

    // 3. bandwidth
    const BIG: u64 = 256 << 20;
    let (src, dst) = (gpu.buffer(BIG, false)?, gpu.buffer(BIG, false)?);
    let copy = gpu.pipeline(COPY, &[&src, &dst])?;
    let n16 = (BIG / 16) as u32;
    // SAFETY: valid device.
    let qp = unsafe {
        gpu.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2),
            None,
        )?
    };
    const ITERS: u32 = 20;
    let ms = gpu.run(|cb| {
        // SAFETY: recording command buffer.
        unsafe {
            gpu.device.cmd_reset_query_pool(cb, qp, 0, 2);
            gpu.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, qp, 0);
        }
        for _ in 0..ITERS {
            gpu.dispatch(cb, &copy, n16, n16.div_ceil(256));
            gpu.barrier(cb);
        }
        // SAFETY: as above.
        unsafe { gpu.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, qp, 1) };
    })?;
    let mut ts = [0u64; 2];
    // SAFETY: two timestamps were written.
    unsafe {
        gpu.device.get_query_pool_results(qp, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?
    };
    let gpu_ms = (ts[1] - ts[0]) as f64 * gpu.timestamp_period as f64 / 1e6;
    let gbs = (2 * BIG * ITERS as u64) as f64 / (gpu_ms / 1e3) / 1e9;
    ensure!(gpu_ms > 0.0, "no GPU time");
    println!(
        "ok   {ITERS} copies of 256 MiB in VRAM: {gpu_ms:.2} ms on the GPU ({gbs:.1} GB/s read+write), {ms:.2} ms wall"
    );

    Ok(())
}
