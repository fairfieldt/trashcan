//! Owned Vulkan resources with cached descriptors. Device addresses are deliberately disabled:
//! RADV otherwise submits every allocation, serializing imported peer buffers.
use crate::gpu::Instance;
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
};

pub struct Device {
    pub inst: Rc<Instance>,
    pub dev: ash::Device,
    pub queue: vk::Queue,
    /// A second queue of the same family (the first one again if the family has only one).
    pub queue2: vk::Queue,
    pub pool: vk::CommandPool,
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    receive_layout: vk::DescriptorSetLayout,
    receive_sets: RefCell<HashMap<vk::Buffer, vk::DescriptorSet>>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: RefCell<HashMap<vk::Buffer, vk::DescriptorSet>>,
    pub pipelines: HashMap<&'static str, vk::Pipeline>,
    pub period: f32,
    binding_layout: vk::DescriptorSetLayout,
    bindings: RefCell<HashMap<[u64; 8], vk::DescriptorSet>>,
    buffers: RefCell<HashMap<u64, (vk::Buffer, usize)>>,
    next_buffer: Cell<u64>,
    pub device_bytes: Cell<u64>,
    dummy: vk::Buffer,
    dummy_memory: vk::DeviceMemory,
    /// Timestamp recorder of the graph being recorded, when profiling (`Stamps`).
    pub stamps: RefCell<Option<Stamps>>,
    calibrated: Option<ash::ext::calibrated_timestamps::Device>,
    /// Device-local memory shared by many buffers, once `use_arena` is called.
    arena: RefCell<Option<Vec<Chunk>>>,
}

/// One allocation of the arena: memory type, size, bytes handed out.
struct Chunk {
    memory: vk::DeviceMemory,
    ty: u32,
    size: u64,
    used: u64,
}

/// Arena allocations; buffers larger than half of this get their own.
const CHUNK: u64 = 256 << 20;

/// GPU timestamps written into a graph's command buffers for profiling (`TRASHCAN_TP_PROFILE`): segment
/// boundaries, and with `dispatches` the start (top of pipe, after the preceding barrier's flushes) and end
/// (bottom of pipe) of every dispatch and copy.
pub struct Stamps {
    pub pool: vk::QueryPool,
    pub labels: Vec<String>,
    pub dispatches: bool,
}
pub const STAMP_CAPACITY: u32 = 16384;

/// CLOCK_MONOTONIC in nanoseconds, the host time domain of `Device::calibrate`.
pub fn host_ns() -> u64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Push {
    pub ptr: [u64; 8],
    pub u: [u32; 16],
}
unsafe impl bytemuck::Zeroable for Push {}
unsafe impl bytemuck::Pod for Push {}
impl Push {
    pub fn with_extra(mut self, address: u64) -> Self {
        self.ptr[7] = address;
        self
    }
    /// The next barrier covers every preceding dispatch in this independent group.
    pub fn defer_barrier(mut self, defer: bool) -> Self {
        self.u[14] = u32::from(defer);
        self
    }
}

/// Choose the measured single-column Q4 layout from the sharded dimensions.
pub fn q4_decode_shader(rows: usize, cols: usize) -> &'static str {
    if rows <= 4096 && cols <= 4096 {
        "matvecq4narrow"
    } else {
        "matvecq4"
    }
}

/// Reuse a bounded score slab across query blocks. Every dispatch keeps the
/// normal compute barrier, including the value read before scratch is reused.
pub fn prefill_attention(d: &Device, cmd: vk::CommandBuffer, mut p: Push, n: usize) {
    let kv = p.u[5];
    let tq = 64 / (p.u[4] / kv);
    p.u[1] = (p.u[4] / kv).ilog2();
    let keys = if p.u[8] > 0 { p.u[8] + tq - 1 } else { p.u[7] };
    for first in (0..n).step_by(64) {
        p.u[13] = first as u32;
        let nb = (n - first).min(64) as u32;
        let groups = kv * nb.div_ceil(tq);
        d.dispatch(cmd, "attn_scores", p, keys.div_ceil(64), groups);
        d.dispatch(cmd, "attn_softmax", p, p.u[4], nb);
        d.dispatch(cmd, "attn_values", p, p.u[6] / 64, groups);
    }
}

/// Scalar reference pipelines with matching partition counts and scratch layout.
pub fn attention_shaders(parts: u32) -> Result<(&'static str, &'static str)> {
    match parts {
        8 => Ok(("attention_partial8", "attention_merge8")),
        16 => Ok(("attention_partial16", "attention_merge16")),
        32 => Ok(("attention_partial", "attention_merge")),
        64 => Ok(("attention_partial64", "attention_merge64")),
        128 => Ok(("attention_partial128", "attention_merge128")),
        _ => anyhow::bail!("attention partitions must be 8, 16, 32, 64 or 128"),
    }
}

/// Share KV across the eight local query heads for measured model shapes.
/// Tiny contexts and unmeasured shape/partition combinations retain scalar loads.
pub fn selected_attention_shaders(
    parts: u32,
    dim: usize,
    kv_heads: usize,
) -> Result<(&'static str, &'static str, u32)> {
    let (scalar, merge) = attention_shaders(parts)?;
    let (partial, heads_per_group) = match (parts, dim, kv_heads) {
        (16, 256, 4) => ("attention_partial_heads2_16", 2),
        (32, 256, 4) => ("attention_partial_heads2", 2),
        (16, 512, 1) => ("attention_partial_heads8_16", 8),
        (32, 512, 1) => ("attention_partial_heads8", 8),
        (64, 512, 1) => ("attention_partial_heads8_64", 8),
        _ => (scalar, 1),
    };
    Ok((partial, merge, heads_per_group))
}

impl Device {
    pub fn new(inst: Rc<Instance>, pd: vk::PhysicalDevice) -> Result<Rc<Self>> {
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
        unsafe {
            inst.instance
                .get_physical_device_properties2(pd, &mut properties);
        }
        ensure!(
            subgroup.subgroup_size == 64
                && subgroup
                    .supported_stages
                    .contains(vk::ShaderStageFlags::COMPUTE)
                && subgroup
                    .supported_operations
                    .contains(vk::SubgroupFeatureFlags::ARITHMETIC),
            "kernels require 64-lane compute subgroups with arithmetic"
        );
        let (family, available) = inst.compute_family(pd)?;
        // A second queue of the family lets tensor-parallel micro-batches run concurrently.
        let priority = [1.0, 1.0];
        let queues = available.clamp(1, 2) as usize;
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priority[..queues])];
        let mut extensions = vec![
            ash::khr::external_memory_fd::NAME.as_ptr(),
            ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ash::khr::external_semaphore_fd::NAME.as_ptr(),
        ];
        // Profiling aligns GPU timestamps with host time.
        let calibrate = std::env::var_os("TRASHCAN_TP_PROFILE").is_some()
            && unsafe { inst.instance.enumerate_device_extension_properties(pd)? }
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::calibrated_timestamps::NAME));
        if calibrate {
            extensions.push(ash::ext::calibrated_timestamps::NAME.as_ptr());
        }
        let mut f11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(false)
            .timeline_semaphore(true);
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_int16(true)
            .shader_int64(true);
        let ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut f11)
            .push_next(&mut f12);
        let dev = unsafe { inst.instance.create_device(pd, &ci, None)? };
        let calibrated = calibrate.then(|| ash::ext::calibrated_timestamps::Device::new(&inst.instance, &dev));
        let queue = unsafe { dev.get_device_queue(family, 0) };
        let queue2 = if queues > 1 { unsafe { dev.get_device_queue(family, 1) } } else { queue };
        let pool = unsafe {
            dev.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let descriptor_layout = unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 65536,
        }];
        let descriptor_pool = unsafe {
            dev.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(8192)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };
        let receive_layout = unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };
        let io_bindings: Vec<_> = (0..8)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let binding_layout = unsafe {
            dev.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&io_bindings),
                None,
            )?
        };
        let set_layouts = [descriptor_layout, receive_layout, binding_layout];
        let range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(128)];
        let layout = unsafe {
            dev.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&range),
                None,
            )?
        };
        let mut pipelines = HashMap::new();
        for (name, bytes) in [
            (
                "matvec",
                include_bytes!(concat!(env!("OUT_DIR"), "/matvec.spv")).as_slice(),
            ),
            (
                "headvec",
                include_bytes!(concat!(env!("OUT_DIR"), "/headvec.spv")).as_slice(),
            ),
            (
                "norm",
                include_bytes!(concat!(env!("OUT_DIR"), "/norm.spv")).as_slice(),
            ),
            (
                "norm_fused",
                include_bytes!(concat!(env!("OUT_DIR"), "/norm_fused.spv")).as_slice(),
            ),
            (
                "norm_rope",
                include_bytes!(concat!(env!("OUT_DIR"), "/norm_rope.spv")).as_slice(),
            ),
            (
                "rope",
                include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv")).as_slice(),
            ),
            (
                "attention",
                include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv")).as_slice(),
            ),
            (
                "attention_partial",
                include_bytes!(concat!(env!("OUT_DIR"), "/attention_partial.spv")).as_slice(),
            ),
            (
                "attention_partial_heads2",
                include_bytes!(concat!(env!("OUT_DIR"), "/attention_partial_heads2.spv"))
                    .as_slice(),
            ),
            (
                "attention_partial_heads8",
                include_bytes!(concat!(env!("OUT_DIR"), "/attention_partial_heads8.spv"))
                    .as_slice(),
            ),
            (
                "attention_merge",
                include_bytes!(concat!(env!("OUT_DIR"), "/attention_merge.spv")).as_slice(),
            ),
            (
                "attn_scores",
                include_bytes!(concat!(env!("OUT_DIR"), "/attn_scores.spv")).as_slice(),
            ),
            (
                "attn_softmax",
                include_bytes!(concat!(env!("OUT_DIR"), "/attn_softmax.spv")).as_slice(),
            ),
            (
                "attn_values",
                include_bytes!(concat!(env!("OUT_DIR"), "/attn_values.spv")).as_slice(),
            ),
            (
                "attention_merge_reference",
                include_bytes!(concat!(env!("OUT_DIR"), "/attention_merge_reference.spv"))
                    .as_slice(),
            ),
            (
                "geglu",
                include_bytes!(concat!(env!("OUT_DIR"), "/geglu.spv")).as_slice(),
            ),
            (
                "topk",
                include_bytes!(concat!(env!("OUT_DIR"), "/topk.spv")).as_slice(),
            ),
            (
                "merge",
                include_bytes!(concat!(env!("OUT_DIR"), "/merge.spv")).as_slice(),
            ),
            (
                "matvecq4",
                include_bytes!(concat!(env!("OUT_DIR"), "/matvecq4.spv")).as_slice(),
            ),
            (
                "matvecq4narrow",
                include_bytes!(concat!(env!("OUT_DIR"), "/matvecq4narrow.spv")).as_slice(),
            ),
            (
                "multivec",
                include_bytes!(concat!(env!("OUT_DIR"), "/multivec.spv")).as_slice(),
            ),
            (
                "multivec2",
                include_bytes!(concat!(env!("OUT_DIR"), "/multivec2.spv")).as_slice(),
            ),
            (
                "multivec4",
                include_bytes!(concat!(env!("OUT_DIR"), "/multivec4.spv")).as_slice(),
            ),
            (
                "multivec4x2",
                include_bytes!(concat!(env!("OUT_DIR"), "/multivec4x2.spv")).as_slice(),
            ),
            (
                "multivec4x3",
                include_bytes!(concat!(env!("OUT_DIR"), "/multivec4x3.spv")).as_slice(),
            ),
            (
                "multivec4_reference",
                include_bytes!(concat!(env!("OUT_DIR"), "/multivec4_reference.spv")).as_slice(),
            ),
            (
                "visibility",
                include_bytes!(concat!(env!("OUT_DIR"), "/visibility.spv")).as_slice(),
            ),
            (
                "gemm",
                include_bytes!(concat!(env!("OUT_DIR"), "/gemm.spv")).as_slice(),
            ),
            ("gemm2", include_bytes!(concat!(env!("OUT_DIR"), "/gemm2.spv")).as_slice()),
            (
                "qwen_add_norm",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_add_norm.spv")).as_slice(),
            ),
            (
                "qwen_add_norm_wide",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_add_norm_wide.spv")).as_slice(),
            ),
            (
                "qwen_matf32",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_matf32.spv")).as_slice(),
            ),
            (
                "qwen_conv",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_conv.spv")).as_slice(),
            ),
            (
                "qwen_gdn",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_gdn.spv")).as_slice(),
            ),
            (
                "qwen_gated_norm",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_gated_norm.spv")).as_slice(),
            ),
            (
                "qwen_qk",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_qk.spv")).as_slice(),
            ),
            (
                "qwen_gate",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_gate.spv")).as_slice(),
            ),
            (
                "qwen_swiglu",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_swiglu.spv")).as_slice(),
            ),
            (
                "qwen_q4c2",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_q4c2.spv")).as_slice(),
            ),
            (
                "qwen_q4c3",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_q4c3.spv")).as_slice(),
            ),
            (
                "qwen_q4c4",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_q4c4.spv")).as_slice(),
            ),
            (
                "qwen_topk",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen_topk.spv")).as_slice(),
            ),
            // Qwen3.8-Flash-Next (qwen4exp)
            ("qwen4_hc_norm", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_hc_norm.spv")).as_slice()),
            ("qwen4_hc_mix", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_hc_mix.spv")).as_slice()),
            ("qwen4_act", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_act.spv")).as_slice()),
            ("qwen4_gated_norm", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_gated_norm.spv")).as_slice()),
            ("qwen4_topk", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_topk.spv")).as_slice()),
            ("qwen4_moe_gu", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_moe_gu.spv")).as_slice()),
            ("qwen4_moe_down", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_moe_down.spv")).as_slice()),
            ("qwen4_ple_gate", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_ple_gate.spv")).as_slice()),
            ("qwen4_ple_conv", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_ple_conv.spv")).as_slice()),
            ("qwen4_moe_route", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_moe_route.spv")).as_slice()),
            ("qwen4_moe_gu_grouped", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_moe_gu_grouped.spv")).as_slice()),
            ("qwen4_moe_down_grouped", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_moe_down_grouped.spv")).as_slice()),
            ("qwen4_moe_sum", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_moe_sum.spv")).as_slice()),
            ("qwen4_idx_store", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_idx_store.spv")).as_slice()),
            ("qwen4_idx_pool", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_idx_pool.spv")).as_slice()),
            ("qwen4_idx_select", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_idx_select.spv")).as_slice()),
            ("qwen4_idx_score", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_idx_score.spv")).as_slice()),
            ("qwen4_attention_sparse", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_attention_sparse.spv")).as_slice()),
            (
                "qwen4_attention_sparse_full",
                include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_attention_sparse_full.spv")).as_slice(),
            ),
            ("qwen4_attention_sparse_q8", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_attention_sparse_q8.spv")).as_slice()),
            ("qwen4_attention_sparse_full_q8", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_attention_sparse_full_q8.spv")).as_slice()),
            ("qwen4_attention_sparse_k16v8", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_attention_sparse_k16v8.spv")).as_slice()),
            ("qwen4_attention_sparse_full_k16v8", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_attention_sparse_full_k16v8.spv")).as_slice()),
            ("qwen4_kv_store", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_kv_store.spv")).as_slice()),
            ("qwen4_q8_mv", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_q8_mv.spv")).as_slice()),
            ("qwen4_q8_gemm", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_q8_gemm.spv")).as_slice()),
            ("qwen4_mix_down", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_mix_down.spv")).as_slice()),
            ("qwen4_mix_up", include_bytes!(concat!(env!("OUT_DIR"), "/qwen4_mix_up.spv")).as_slice()),
            // Cross-machine all-reduce benchmark
            ("allreduce_partial", include_bytes!(concat!(env!("OUT_DIR"), "/allreduce_partial.spv")).as_slice()),
            ("allreduce_check", include_bytes!(concat!(env!("OUT_DIR"), "/allreduce_check.spv")).as_slice()),
            ("allreduce_work", include_bytes!(concat!(env!("OUT_DIR"), "/allreduce_work.spv")).as_slice()),
        ] {
            ensure!(
                !bytes.is_empty(),
                "shaders missing: build with glslc on incite"
            );
            let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
            let module = unsafe {
                dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
            };
            let variants: &[(&'static str, Option<u32>)] = match name {
                "attention_partial" => &[
                    ("attention_partial", None),
                    ("attention_partial8", Some(8)),
                    ("attention_partial16", Some(16)),
                    ("attention_partial64", Some(64)),
                    ("attention_partial128", Some(128)),
                ],
                "qwen_add_norm_wide" => &[("qwen_add_norm_j16", Some(16)), ("qwen_add_norm_j20", Some(20))],
                "attention_partial_heads2" => &[
                    ("attention_partial_heads2", None),
                    ("attention_partial_heads2_16", Some(16)),
                ],
                "attention_partial_heads8" => &[
                    ("attention_partial_heads8", None),
                    ("attention_partial_heads8_16", Some(16)),
                    ("attention_partial_heads8_64", Some(64)),
                ],
                "qwen4_q8_mv" => &[("qwen4_q8_mv", Some(1)), ("qwen4_q8_mv4", Some(4))],
                "qwen4_moe_gu" => &[
                    ("qwen4_moe_gu", Some(0)),
                    ("qwen4_moe_gu_iq4nl", Some(1)),
                    ("qwen4_moe_gu_iq4xs", Some(2)),
                ],
                "qwen4_moe_down" => &[
                    ("qwen4_moe_down", Some(0)),
                    ("qwen4_moe_down_iq4nl", Some(1)),
                    ("qwen4_moe_down_iq4xs", Some(2)),
                    ("qwen4_moe_down_q2", Some(3)),
                ],
                "qwen4_moe_gu_grouped" => &[
                    ("qwen4_moe_gu_grouped", Some(0)),
                    ("qwen4_moe_gu_grouped_iq4nl", Some(1)),
                    ("qwen4_moe_gu_grouped_iq4xs", Some(2)),
                ],
                "qwen4_moe_down_grouped" => &[
                    ("qwen4_moe_down_grouped", Some(0)),
                    ("qwen4_moe_down_grouped_iq4nl", Some(1)),
                    ("qwen4_moe_down_grouped_iq4xs", Some(2)),
                    ("qwen4_moe_down_grouped_q2", Some(3)),
                ],
                "qwen4_mix_down" => &[("qwen4_mix_down", Some(0)), ("qwen4_mix_down_q8", Some(1))],
                "qwen4_mix_up" => &[("qwen4_mix_up", Some(0)), ("qwen4_mix_up_q8", Some(1))],
                "qwen4_attention_sparse" => &[
                    ("qwen4_attention_sparse", None),
                    ("qwen4_attention_sparse8", Some(8)),
                    ("qwen4_attention_sparse16", Some(16)),
                    ("qwen4_attention_sparse64", Some(64)),
                ],
                "qwen4_attention_sparse_q8" => &[
                    ("qwen4_attention_sparse_q8", None),
                    ("qwen4_attention_sparse_q8_8", Some(8)),
                    ("qwen4_attention_sparse_q8_16", Some(16)),
                    ("qwen4_attention_sparse_q8_64", Some(64)),
                ],
                "qwen4_attention_sparse_k16v8" => &[
                    ("qwen4_attention_sparse_k16v8", None),
                    ("qwen4_attention_sparse_k16v8_8", Some(8)),
                    ("qwen4_attention_sparse_k16v8_16", Some(16)),
                    ("qwen4_attention_sparse_k16v8_64", Some(64)),
                ],
                "attention_merge" => &[
                    ("attention_merge", None),
                    ("attention_merge8", Some(8)),
                    ("attention_merge16", Some(16)),
                    ("attention_merge64", Some(64)),
                    ("attention_merge128", Some(128)),
                ],
                _ => &[(name, None)],
            };
            for &(variant, parts) in variants {
                let entries = [vk::SpecializationMapEntry::default()
                    .constant_id(0)
                    .offset(0)
                    .size(std::mem::size_of::<u32>())];
                let data = parts.unwrap_or(32).to_ne_bytes();
                let specialization = vk::SpecializationInfo::default()
                    .map_entries(&entries)
                    .data(&data);
                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                if parts.is_some() {
                    stage = stage.specialization_info(&specialization);
                }
                let ci = [vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)];
                let pipe =
                    unsafe { dev.create_compute_pipelines(vk::PipelineCache::null(), &ci, None) }
                        .map_err(|(_, e)| anyhow::anyhow!("{variant}: {e}"))?[0];
                pipelines.insert(variant, pipe);
            }
            unsafe {
                dev.destroy_shader_module(module, None);
            }
        }
        let mem = unsafe { inst.instance.get_physical_device_memory_properties(pd) };
        let period = unsafe { inst.instance.get_physical_device_properties(pd) }
            .limits
            .timestamp_period;
        let dummy = unsafe {
            dev.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(16)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
                None,
            )?
        };
        let req = unsafe { dev.get_buffer_memory_requirements(dummy) };
        let ty = (0..mem.memory_type_count)
            .find(|&i| req.memory_type_bits & (1 << i) != 0)
            .context("dummy memory type")?;
        let dummy_memory = unsafe {
            dev.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty),
                None,
            )?
        };
        unsafe {
            dev.bind_buffer_memory(dummy, dummy_memory, 0)?;
        }
        Ok(Rc::new(Self {
            inst,
            dev,
            queue,
            queue2,
            pool,
            mem,
            layout,
            descriptor_layout,
            receive_layout,
            receive_sets: RefCell::new(HashMap::new()),
            descriptor_pool,
            descriptor_sets: RefCell::new(HashMap::new()),
            pipelines,
            period,
            binding_layout,
            bindings: RefCell::new(HashMap::new()),
            buffers: RefCell::new(HashMap::new()),
            next_buffer: Cell::new(1),
            device_bytes: Cell::new(0),
            dummy,
            dummy_memory,
            stamps: RefCell::new(None),
            calibrated,
            arena: RefCell::new(None),
        }))
    }
    /// Place later device-local buffers (not host-visible, not external) in shared allocations of `CHUNK`
    /// bytes. A submission names every memory object its command buffers use and the kernel validates each
    /// of them, so a forward over hundreds of per-buffer allocations waits about a millisecond before it
    /// starts. Arena memory is only returned when the device is destroyed.
    pub fn use_arena(&self) {
        self.arena.borrow_mut().get_or_insert_with(Vec::new);
    }
    /// Memory and offset for `buf` in the arena, if the arena is on and the buffer fits a chunk.
    fn arena_place(&self, buf: vk::Buffer, ty: u32, req: &vk::MemoryRequirements) -> Result<Option<(vk::DeviceMemory, u64)>> {
        let mut arena = self.arena.borrow_mut();
        let Some(chunks) = arena.as_mut() else { return Ok(None) };
        if req.size > CHUNK / 2 {
            return Ok(None);
        }
        let fits = |c: &Chunk| c.ty == ty && c.used.next_multiple_of(req.alignment) + req.size <= c.size;
        if !chunks.iter().any(fits) {
            let ai = vk::MemoryAllocateInfo::default().allocation_size(CHUNK).memory_type_index(ty);
            let memory = unsafe { self.dev.allocate_memory(&ai, None)? };
            chunks.push(Chunk { memory, ty, size: CHUNK, used: 0 });
        }
        let c = chunks.iter_mut().find(|c| fits(c)).unwrap();
        let offset = c.used.next_multiple_of(req.alignment);
        c.used = offset + req.size;
        unsafe { self.dev.bind_buffer_memory(buf, c.memory, offset)? };
        Ok(Some((c.memory, offset)))
    }
    /// Write a timestamp into the active recorder, if any: at the top of the pipe (when the command processor
    /// gets there, after pending barrier flushes) or the bottom (when all preceding work has finished).
    pub fn stamp(&self, cmd: vk::CommandBuffer, bottom: bool, label: impl FnOnce() -> String) {
        if let Some(s) = self.stamps.borrow_mut().as_mut() {
            let i = s.labels.len() as u32;
            if i < STAMP_CAPACITY {
                s.labels.push(label());
                let stage = if bottom { vk::PipelineStageFlags::BOTTOM_OF_PIPE } else { vk::PipelineStageFlags::TOP_OF_PIPE };
                unsafe { self.dev.cmd_write_timestamp(cmd, stage, s.pool, i) };
            }
        }
    }
    fn stamping_dispatches(&self) -> bool {
        self.stamps.borrow().as_ref().is_some_and(|s| s.dispatches)
    }
    /// (device ticks, CLOCK_MONOTONIC ns) sampled together, on a device created for profiling.
    pub fn calibrate(&self) -> Option<(u64, u64)> {
        let c = self.calibrated.as_ref()?;
        let info = [
            vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::DEVICE),
            vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::CLOCK_MONOTONIC),
        ];
        let (t, _) = unsafe { c.get_calibrated_timestamps(&info) }.ok()?;
        Some((t[0], t[1]))
    }
    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Result<u32> {
        (0..self.mem.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && self.mem.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .context("no compatible Vulkan memory type")
    }
    pub fn buffer(
        self: &Rc<Self>,
        size: usize,
        host: bool,
        external: bool,
        import: Option<i32>,
    ) -> Result<Buffer> {
        self.buffer_impl(size, host, external, import)
    }
    fn buffer_impl(
        self: &Rc<Self>,
        size: usize,
        host: bool,
        external: bool,
        import: Option<i32>,
    ) -> Result<Buffer> {
        let size = size.max(4).next_multiple_of(4);
        let mut ext = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut ci = vk::BufferCreateInfo::default().size(size as u64).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        if external {
            ci = ci.push_next(&mut ext);
        }
        let buf = unsafe { self.dev.create_buffer(&ci, None)? };
        let req = unsafe { self.dev.get_buffer_memory_requirements(buf) };
        let mut bits = req.memory_type_bits;
        let fd_ext = ash::khr::external_memory_fd::Device::new(&self.inst.instance, &self.dev);
        if let Some(fd) = import {
            let mut props = vk::MemoryFdPropertiesKHR::default();
            unsafe {
                fd_ext.get_memory_fd_properties(
                    vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                    fd,
                    &mut props,
                )?
            };
            bits &= props.memory_type_bits;
        }
        let flags = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else if import.is_some() {
            vk::MemoryPropertyFlags::empty()
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let ty = if host {
            self.memory_type(bits, flags | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|_| self.memory_type(bits, flags))?
        } else {
            self.memory_type(bits, flags)?
        };
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buf);
        let mut exp = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut imp = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(import.unwrap_or(-1));
        let mut ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(ty);
        if external {
            ai = ai.push_next(&mut dedicated);
            ai = if import.is_some() {
                ai.push_next(&mut imp)
            } else {
                ai.push_next(&mut exp)
            };
        }
        let placed = if host || external || import.is_some() { None } else { self.arena_place(buf, ty, &req)? };
        let (memory, owned) = match placed {
            Some((memory, _)) => (memory, false),
            None => {
                let memory = unsafe { self.dev.allocate_memory(&ai, None)? };
                unsafe { self.dev.bind_buffer_memory(buf, memory, 0)? };
                (memory, true)
            }
        };
        let ptr = if host {
            unsafe {
                self.dev
                    .map_memory(memory, 0, size as u64, vk::MemoryMapFlags::empty())?
            }
            .cast()
        } else {
            std::ptr::null_mut()
        };
        let id = self.next_buffer.get();
        self.next_buffer.set(id + 1);
        let address = id << 32;
        self.buffers.borrow_mut().insert(id, (buf, size));
        // Imported allocations alias the peer's memory; count only locally owned VRAM.
        let device_bytes = if !host && import.is_none() {
            req.size
        } else {
            0
        };
        self.device_bytes
            .set(self.device_bytes.get() + device_bytes);
        Ok(Buffer {
            gpu: self.clone(),
            buf,
            memory,
            address,
            size,
            ptr,
            device_bytes,
            owned,
        })
    }
    pub fn begin(&self) -> Result<vk::CommandBuffer> {
        let cmd = unsafe {
            self.dev.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        unsafe {
            self.dev
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
        }
        Ok(cmd)
    }
    pub fn barrier(&self, cmd: vk::CommandBuffer) {
        let b = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        unsafe {
            self.dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &b,
                &[],
                &[],
            );
        }
    }
    /// Order host access around a submission boundary: `to_host` makes shader writes visible to the host
    /// once the submission completes; otherwise host writes made before the submission become visible to
    /// shaders and copies.
    pub fn host_barrier(&self, cmd: vk::CommandBuffer, to_host: bool) {
        if to_host {
            self.scoped_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::HOST_READ,
            );
        } else {
            self.scoped_barrier(
                cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::HOST_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ,
            );
        }
    }
    /// Make the command processor wait until the host sets `event` (RADV: a WAIT_REG_MEM on the event's host
    /// memory, no kernel involvement), then make host writes made before the set visible to shaders and copies.
    pub fn wait_host_event(&self, cmd: vk::CommandBuffer, event: vk::Event) {
        let b = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ)];
        unsafe {
            self.dev.cmd_wait_events(
                cmd,
                &[event],
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                &b,
                &[],
                &[],
            )
        };
    }
    fn scoped_barrier(
        &self,
        cmd: vk::CommandBuffer,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        write: vk::AccessFlags,
        read: vk::AccessFlags,
    ) {
        let b = [vk::MemoryBarrier::default()
            .src_access_mask(write)
            .dst_access_mask(read)];
        unsafe {
            self.dev.cmd_pipeline_barrier(
                cmd,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &b,
                &[],
                &[],
            );
        }
    }
    pub fn external_buffer(self: &Rc<Self>, size: usize, import: Option<i32>) -> Result<Buffer> {
        self.buffer_impl(size, false, true, import)
    }
    pub fn dispatch_norm(
        &self,
        cmd: vk::CommandBuffer,
        p: Push,
        x: u32,
        y: u32,
        receive: &Buffer,
    ) -> Result<()> {
        // The fixed-width kernel preserves norm's per-lane FMA chains and reduction tree.
        let shader = if x == 1
            && p.u[0] == 3840
            && p.u[3] == 3840
            && p.u[11] == 7
            && p.ptr[1] != 0
            && p.ptr[5] != 0
            && p.ptr[7] != 0
        {
            "norm_fused"
        } else {
            "norm"
        };
        self.dispatch_receive(cmd, shader, p, x, y, receive)
    }
    pub fn dispatch_receive(
        &self,
        cmd: vk::CommandBuffer,
        shader: &str,
        p: Push,
        x: u32,
        y: u32,
        receive: &Buffer,
    ) -> Result<()> {
        let existing = { self.receive_sets.borrow().get(&receive.buf).copied() };
        let set = if let Some(set) = existing {
            set
        } else {
            let layouts = [self.receive_layout];
            let set = unsafe {
                self.dev.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&layouts),
                )?
            }[0];
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(receive.buf)
                .offset(0)
                .range(receive.size as u64)];
            let write = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info)];
            unsafe {
                self.dev.update_descriptor_sets(&write, &[]);
            }
            self.receive_sets.borrow_mut().insert(receive.buf, set);
            set
        };
        unsafe {
            self.dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                1,
                &[set],
                &[],
            );
        }
        self.dispatch(cmd, shader, p, x, y);
        Ok(())
    }
    pub fn dispatch_matrix(
        &self,
        cmd: vk::CommandBuffer,
        shader: &str,
        p: Push,
        x: u32,
        y: u32,
        weight: &Buffer,
    ) -> Result<()> {
        let existing = { self.descriptor_sets.borrow().get(&weight.buf).copied() };
        let set = if let Some(set) = existing {
            set
        } else {
            let layouts = [self.descriptor_layout];
            let set = unsafe {
                self.dev.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&layouts),
                )?
            }[0];
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(weight.buf)
                .offset(0)
                .range(weight.size as u64)];
            let write = [vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info)];
            unsafe {
                self.dev.update_descriptor_sets(&write, &[]);
            }
            self.descriptor_sets.borrow_mut().insert(weight.buf, set);
            set
        };
        unsafe {
            self.dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
        }
        // Profiling labels a matrix by its rows and row length.
        self.dispatch_as(cmd, shader, p, x, y, || format!("{shader} {}x{}", p.u[0], p.u[1]));
        Ok(())
    }
    pub fn dispatch(&self, cmd: vk::CommandBuffer, shader: &str, p: Push, x: u32, y: u32) {
        self.dispatch_as(cmd, shader, p, x, y, || format!("{shader} {x}x{y}"));
    }
    /// `dispatch` with `label` for its profiling timestamps (`Stamps`).
    fn dispatch_as(&self, cmd: vk::CommandBuffer, shader: &str, p: Push, x: u32, y: u32, label: impl Fn() -> String) {
        let key = p.ptr;
        let cached = { self.bindings.borrow().get(&key).copied() };
        let set = if let Some(set) = cached {
            set
        } else {
            let layouts = [self.binding_layout];
            let set = unsafe {
                self.dev
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.descriptor_pool)
                            .set_layouts(&layouts),
                    )
                    .expect("graph descriptor allocation")
            }[0];
            let entries = self.buffers.borrow();
            let info: Vec<_> = key
                .iter()
                .map(|address| {
                    if *address == 0 {
                        return vk::DescriptorBufferInfo::default()
                            .buffer(self.dummy)
                            .range(16);
                    }
                    let (buf, size) = entries.get(&(address >> 32)).expect("live graph buffer");
                    let offset = address & 0xffffffff;
                    assert!(offset < *size as u64);
                    vk::DescriptorBufferInfo::default()
                        .buffer(*buf)
                        .offset(offset)
                        .range(*size as u64 - offset)
                })
                .collect();
            let writes: Vec<_> = info
                .iter()
                .enumerate()
                .map(|(binding, i)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(i))
                })
                .collect();
            unsafe {
                self.dev.update_descriptor_sets(&writes, &[]);
            }
            self.bindings.borrow_mut().insert(key, set);
            set
        };
        unsafe {
            self.dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                2,
                &[set],
                &[],
            );
        }
        unsafe {
            self.dev
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[shader]);
            self.dev.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&p),
            );
        }
        let stamp = self.stamping_dispatches();
        if stamp {
            self.stamp(cmd, false, &label);
        }
        unsafe { self.dev.cmd_dispatch(cmd, x, y, 1) };
        if stamp {
            self.stamp(cmd, true, &label);
        }
        if p.u[14] == 0 {
            self.scoped_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
        }
    }
    pub fn copy(
        &self,
        cmd: vk::CommandBuffer,
        a: &Buffer,
        ao: usize,
        b: &Buffer,
        bo: usize,
        size: usize,
    ) {
        self.scoped_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        let stamp = self.stamping_dispatches();
        if stamp {
            self.stamp(cmd, false, || format!("copy {size}"));
        }
        unsafe {
            self.dev.cmd_copy_buffer(
                cmd,
                a.buf,
                b.buf,
                &[vk::BufferCopy {
                    src_offset: ao as u64,
                    dst_offset: bo as u64,
                    size: size as u64,
                }],
            );
        }
        if stamp {
            self.stamp(cmd, true, || format!("copy {size}"));
        }
        self.scoped_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
    }
    pub fn finish(&self, cmd: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.dev.end_command_buffer(cmd)?;
        }
        Ok(())
    }
    pub fn submit(
        &self,
        cmd: vk::CommandBuffer,
        wait: Option<(vk::Semaphore, u64)>,
        signal: Option<(vk::Semaphore, u64)>,
        fence: vk::Fence,
    ) -> Result<()> {
        let commands = [cmd];
        let ws = wait.map(|x| vec![x.0]).unwrap_or_default();
        let wv = wait.map(|x| vec![x.1]).unwrap_or_default();
        let ss = signal.map(|x| vec![x.0]).unwrap_or_default();
        let sv = signal.map(|x| vec![x.1]).unwrap_or_default();
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let mut ti = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wv)
            .signal_semaphore_values(&sv);
        let ci = [vk::SubmitInfo::default()
            .command_buffers(&commands)
            .wait_semaphores(&ws)
            .wait_dst_stage_mask(if ws.is_empty() { &[] } else { &stages })
            .signal_semaphores(&ss)
            .push_next(&mut ti)];
        unsafe {
            self.dev.queue_submit(self.queue, &ci, fence)?;
        }
        Ok(())
    }
    /// Submit `cmds` as one submission, signalling `signal` when all have finished.
    pub fn submit_all(&self, cmds: &[vk::CommandBuffer], signal: (vk::Semaphore, u64)) -> Result<()> {
        let (ss, sv) = ([signal.0], [signal.1]);
        let mut ti = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&sv);
        let ci = [vk::SubmitInfo::default().command_buffers(cmds).signal_semaphores(&ss).push_next(&mut ti)];
        unsafe { self.dev.queue_submit(self.queue, &ci, vk::Fence::null())? };
        Ok(())
    }
    /// Submit `cmd` to `queue`, waiting for every timeline value in `waits` and signalling those in `signals`
    /// (at most four each, without allocating).
    pub fn submit_on(&self, queue: vk::Queue, cmd: vk::CommandBuffer, waits: &[(vk::Semaphore, u64)], signals: &[(vk::Semaphore, u64)]) -> Result<()> {
        ensure!(waits.len() <= 4 && signals.len() <= 4, "at most four waits and signals per submission");
        let commands = [cmd];
        let (mut ws, mut wv, mut ss, mut sv) = ([vk::Semaphore::null(); 4], [0u64; 4], [vk::Semaphore::null(); 4], [0u64; 4]);
        for (i, &(s, v)) in waits.iter().enumerate() {
            (ws[i], wv[i]) = (s, v);
        }
        for (i, &(s, v)) in signals.iter().enumerate() {
            (ss[i], sv[i]) = (s, v);
        }
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS; 4];
        let (w, s) = (waits.len(), signals.len());
        let mut ti = vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wv[..w]).signal_semaphore_values(&sv[..s]);
        let ci = [vk::SubmitInfo::default()
            .command_buffers(&commands)
            .wait_semaphores(&ws[..w])
            .wait_dst_stage_mask(&stages[..w])
            .signal_semaphores(&ss[..s])
            .push_next(&mut ti)];
        unsafe { self.dev.queue_submit(queue, &ci, vk::Fence::null())? };
        Ok(())
    }
    /// Submit the full schedule in one API call per GPU. Each element still has
    /// its own timeline dependency, so the GPUs advance together at allreduces.
    pub fn submit_sequence(
        &self,
        cmds: &[vk::CommandBuffer],
        peer: vk::Semaphore,
        own: vk::Semaphore,
        counter: u64,
        fence: vk::Fence,
    ) -> Result<()> {
        let waits = [peer];
        let signals = [own];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
        let wv: Vec<[u64; 1]> = (0..cmds.len()).map(|i| [counter + i as u64]).collect();
        let sv: Vec<[u64; 1]> = (0..cmds.len()).map(|i| [counter + i as u64 + 1]).collect();
        let mut timelines: Vec<_> = (0..cmds.len())
            .map(|i| {
                vk::TimelineSemaphoreSubmitInfo::default()
                    .wait_semaphore_values(if i > 0 { &wv[i] } else { &[] })
                    .signal_semaphore_values(if i + 1 < cmds.len() { &sv[i] } else { &[] })
            })
            .collect();
        let submits: Vec<_> = timelines
            .iter_mut()
            .enumerate()
            .map(|(i, t)| {
                vk::SubmitInfo::default()
                    .command_buffers(&cmds[i..i + 1])
                    .wait_semaphores(if i > 0 { &waits } else { &[] })
                    .wait_dst_stage_mask(if i > 0 { &stages } else { &[] })
                    .signal_semaphores(if i + 1 < cmds.len() { &signals } else { &[] })
                    .push_next(t)
            })
            .collect();
        unsafe {
            self.dev.queue_submit(self.queue, &submits, fence)?;
        }
        Ok(())
    }
    pub fn run(&self, cmd: vk::CommandBuffer) -> Result<()> {
        self.run_reusable(cmd)?;
        unsafe {
            self.dev.free_command_buffers(self.pool, &[cmd]);
        }
        Ok(())
    }
    /// Wait for completion without freeing the command buffer; benchmarks can resubmit it.
    pub fn run_reusable(&self, cmd: vk::CommandBuffer) -> Result<()> {
        let fence = unsafe {
            self.dev
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        self.submit(cmd, None, None, fence)?;
        unsafe {
            self.dev.wait_for_fences(&[fence], true, 60_000_000_000)?;
            self.dev.destroy_fence(fence, None);
        }
        Ok(())
    }
    pub fn upload_matrix(self: &Rc<Self>, data: &[u8]) -> Result<Buffer> {
        self.upload(data)
    }
    pub fn upload(self: &Rc<Self>, data: &[u8]) -> Result<Buffer> {
        let dst = self.buffer(data.len(), false, false, None)?;
        let staging = self.buffer(data.len(), true, false, None)?;
        staging.write(data)?;
        let cmd = self.begin()?;
        self.copy(cmd, &staging, 0, &dst, 0, staging.size);
        self.finish(cmd)?;
        self.run(cmd)?;
        Ok(dst)
    }
    pub fn timeline(&self, external: bool) -> Result<vk::Semaphore> {
        let mut ty =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let mut ext = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let mut ci = vk::SemaphoreCreateInfo::default().push_next(&mut ty);
        if external {
            ci = ci.push_next(&mut ext);
        }
        Ok(unsafe { self.dev.create_semaphore(&ci, None)? })
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = self.dev.device_wait_idle();
            for p in self.pipelines.values() {
                self.dev.destroy_pipeline(*p, None);
            }
            self.dev.destroy_pipeline_layout(self.layout, None);
            self.dev.destroy_descriptor_pool(self.descriptor_pool, None);
            self.dev
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.dev
                .destroy_descriptor_set_layout(self.receive_layout, None);
            self.dev.destroy_command_pool(self.pool, None);
            self.dev
                .destroy_descriptor_set_layout(self.binding_layout, None);
            self.dev.destroy_buffer(self.dummy, None);
            self.dev.free_memory(self.dummy_memory, None);
            for c in self.arena.borrow().iter().flatten() {
                self.dev.free_memory(c.memory, None);
            }
            self.dev.destroy_device(None);
        }
    }
}
pub struct Buffer {
    pub gpu: Rc<Device>,
    pub buf: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub address: u64,
    pub size: usize,
    pub ptr: *mut u8,
    device_bytes: u64,
    /// The memory is this buffer's own (not an arena chunk).
    owned: bool,
}
impl Buffer {
    pub fn copy_host(&self, other: &Buffer, size: usize) -> Result<()> {
        ensure!(
            !self.ptr.is_null() && !other.ptr.is_null() && size <= self.size && size <= other.size,
            "invalid host copy"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(other.ptr, self.ptr, size);
        }
        Ok(())
    }
    pub fn write(&self, bytes: &[u8]) -> Result<()> {
        ensure!(
            !self.ptr.is_null() && bytes.len() <= self.size,
            "invalid host write"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr, bytes.len());
        }
        Ok(())
    }
    pub fn write_at(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        ensure!(
            !self.ptr.is_null() && offset + bytes.len() <= self.size,
            "invalid host write"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
        }
        Ok(())
    }
    pub fn read(&self, size: usize) -> Result<Vec<u8>> {
        ensure!(
            !self.ptr.is_null() && size <= self.size,
            "invalid host read"
        );
        Ok(unsafe { std::slice::from_raw_parts(self.ptr, size) }.to_vec())
    }
    pub fn export(&self) -> Result<i32> {
        let ex = ash::khr::external_memory_fd::Device::new(&self.gpu.inst.instance, &self.gpu.dev);
        Ok(unsafe {
            ex.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(self.memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
            )?
        })
    }
    pub fn download_range(&self, offset: usize, size: usize) -> Result<Vec<u8>> {
        ensure!(offset + size <= self.size, "download out of bounds");
        let h = self.gpu.buffer(size, true, false, None)?;
        let cmd = self.gpu.begin()?;
        self.gpu.copy(cmd, self, offset, &h, 0, size);
        self.gpu.finish(cmd)?;
        self.gpu.run(cmd)?;
        h.read(size)
    }
    pub fn download(&self) -> Result<Vec<u8>> {
        let h = self.gpu.buffer(self.size, true, false, None)?;
        let cmd = self.gpu.begin()?;
        self.gpu.copy(cmd, self, 0, &h, 0, self.size);
        self.gpu.finish(cmd)?;
        self.gpu.run(cmd)?;
        h.read(self.size)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        self.gpu.buffers.borrow_mut().remove(&(self.address >> 32));
        self.gpu
            .device_bytes
            .set(self.gpu.device_bytes.get() - self.device_bytes);
        unsafe {
            if !self.ptr.is_null() {
                self.gpu.dev.unmap_memory(self.memory);
            }
            self.gpu.dev.destroy_buffer(self.buf, None);
            if self.owned {
                self.gpu.dev.free_memory(self.memory, None);
            }
        }
    }
}

#[cfg(test)]
mod attention_selection_tests {
    use super::{attention_shaders, selected_attention_shaders};

    #[test]
    fn shared_shapes_keep_the_matching_partition_merge() {
        for (parts, dim, kv, expected) in [
            (16, 256, 4, "attention_partial_heads2_16"),
            (32, 256, 4, "attention_partial_heads2"),
            (16, 512, 1, "attention_partial_heads8_16"),
            (32, 512, 1, "attention_partial_heads8"),
            (64, 512, 1, "attention_partial_heads8_64"),
        ] {
            let (partial, merge, group_heads) = selected_attention_shaders(parts, dim, kv).unwrap();
            assert_eq!(group_heads, if kv == 4 { 2 } else { 8 });
            assert_eq!(partial, expected);
            assert_eq!(merge, attention_shaders(parts).unwrap().1);
        }
    }

    #[test]
    fn tiny_and_unmeasured_cases_retain_scalar_loads() {
        for (parts, dim, kv) in [
            (8, 256, 4),
            (8, 512, 1),
            (64, 256, 4),
            (128, 512, 1),
            (32, 256, 1),
        ] {
            assert_eq!(selected_attention_shaders(parts, dim, kv).unwrap(), {
                let (partial, merge) = attention_shaders(parts).unwrap();
                (partial, merge, 1)
            });
        }
        assert!(selected_attention_shaders(17, 512, 1).is_err());
    }
}
