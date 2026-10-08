//! Vulkan device discovery for the two D700s, and a report of everything the engine depends on.

use anyhow::{bail, Context, Result};
use ash::vk;
use std::ffi::CStr;

/// Device extensions the engine needs: dma-buf export/import for P2P between the GPUs, cross-device timeline
/// semaphores, host-pointer import, and PCI addresses to tell the cards apart.
pub const REQUIRED_EXTENSIONS: [&CStr; 6] = [
    ash::khr::external_memory_fd::NAME,
    ash::ext::external_memory_dma_buf::NAME,
    ash::khr::external_semaphore_fd::NAME,
    ash::ext::external_memory_host::NAME,
    ash::ext::pci_bus_info::NAME,
    ash::amd::shader_core_properties::NAME,
];

pub struct Instance {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
}

impl Instance {
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.context("loading the Vulkan loader")?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"trashcan")
            .api_version(vk::API_VERSION_1_3);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )?
        };
        Ok(Instance {
            _entry: entry,
            instance,
        })
    }

    /// Every Vulkan device, in the loader's order (`qwen-kernel-check --any-device`).
    pub fn all_devices(&self) -> Result<Vec<vk::PhysicalDevice>> {
        Ok(unsafe { self.instance.enumerate_physical_devices()? })
    }

    /// The discrete AMD GPUs, ordered by PCI bus number so indices are stable across runs.
    pub fn amd_gpus(&self) -> Result<Vec<vk::PhysicalDevice>> {
        let mut gpus = Vec::new();
        for pd in unsafe { self.instance.enumerate_physical_devices()? } {
            let p = unsafe { self.instance.get_physical_device_properties(pd) };
            if p.vendor_id == 0x1002 && p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                gpus.push((self.pci(pd)?.pci_bus, pd));
            }
        }
        gpus.sort_by_key(|g| g.0);
        Ok(gpus.into_iter().map(|g| g.1).collect())
    }

    fn pci(
        &self,
        pd: vk::PhysicalDevice,
    ) -> Result<vk::PhysicalDevicePCIBusInfoPropertiesEXT<'static>> {
        let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
        unsafe { self.instance.get_physical_device_properties2(pd, &mut p2) };
        Ok(vk::PhysicalDevicePCIBusInfoPropertiesEXT {
            p_next: std::ptr::null_mut(),
            ..pci
        })
    }

    pub fn extensions(&self, pd: vk::PhysicalDevice) -> Result<Vec<String>> {
        let exts = unsafe { self.instance.enumerate_device_extension_properties(pd)? };
        Ok(exts
            .iter()
            .filter_map(|e| e.extension_name_as_c_str().ok())
            .map(|n| n.to_string_lossy().into_owned())
            .collect())
    }

    /// Queue family with compute and without graphics (the async compute engines), else any compute family.
    pub fn compute_family(&self, pd: vk::PhysicalDevice) -> Result<(u32, u32)> {
        let fams = unsafe {
            self.instance
                .get_physical_device_queue_family_properties(pd)
        };
        let pick = |want_no_gfx: bool| {
            fams.iter().enumerate().find(|(_, f)| {
                f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && (!want_no_gfx || !f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            })
        };
        let (i, f) = pick(true)
            .or_else(|| pick(false))
            .context("no compute queue family")?;
        Ok((i as u32, f.queue_count))
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn check(ok: bool) -> &'static str {
    if ok {
        "ok"
    } else {
        "MISSING"
    }
}

/// Print what the engine depends on for each GPU, then create a logical device with all of it enabled.
pub fn report() -> Result<()> {
    let inst = Instance::new()?;
    let gpus = inst.amd_gpus()?;
    if gpus.is_empty() {
        bail!("no discrete AMD GPUs found");
    }
    let vi = &inst.instance;
    let mut all_ok = true;

    for (idx, &pd) in gpus.iter().enumerate() {
        let exts = inst.extensions(pd)?;
        let has = |n: &CStr| exts.iter().any(|e| e.as_bytes() == n.to_bytes());

        let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
        let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
        let mut drv = vk::PhysicalDeviceDriverProperties::default();
        let mut amd = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut pci)
            .push_next(&mut sub)
            .push_next(&mut drv)
            .push_next(&mut amd);
        unsafe { vi.get_physical_device_properties2(pd, &mut p2) };
        let p = p2.properties;
        let lim = p.limits;

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        unsafe { vi.get_physical_device_features2(pd, &mut f2) };
        let f = f2.features;

        let name = p.device_name_as_c_str()?.to_string_lossy();
        let driver = drv.driver_info_as_c_str()?.to_string_lossy();
        println!(
            "GPU{idx}: {name} @ {:02x}:{:02x}.{:x}  ({}, Vulkan {}.{}.{})",
            pci.pci_bus,
            pci.pci_device,
            pci.pci_function,
            driver,
            vk::api_version_major(p.api_version),
            vk::api_version_minor(p.api_version),
            vk::api_version_patch(p.api_version)
        );

        let cus = amd.shader_engine_count
            * amd.shader_arrays_per_engine_count
            * amd.compute_units_per_shader_array;
        println!(
            "  shape: {} shader engines x {} arrays x {} CUs = {} CUs, {} SIMDs/CU, wave{} , {} wave slots/SIMD, {} VGPRs/SIMD, {} SGPRs/SIMD",
            amd.shader_engine_count,
            amd.shader_arrays_per_engine_count,
            amd.compute_units_per_shader_array,
            cus,
            amd.simd_per_compute_unit,
            amd.wavefront_size,
            amd.wavefronts_per_simd,
            amd.vgprs_per_simd,
            amd.sgprs_per_simd
        );
        println!(
            "  compute: max workgroup {} invocations {:?}, shared mem {} KiB, subgroup {} (ops {:?})",
            lim.max_compute_work_group_invocations,
            lim.max_compute_work_group_size,
            lim.max_compute_shared_memory_size / 1024,
            sub.subgroup_size,
            sub.supported_operations
        );
        println!(
            "  limits: storage buffer range {:.0} MiB, max allocation count {}, timestamp period {} ns",
            mib(lim.max_storage_buffer_range as u64),
            lim.max_memory_allocation_count,
            lim.timestamp_period
        );

        let mem = unsafe { vi.get_physical_device_memory_properties(pd) };
        for (i, h) in mem.memory_heaps_as_slice().iter().enumerate() {
            println!("  heap {i}: {:.0} MiB {:?}", mib(h.size), h.flags);
        }
        for (i, t) in mem.memory_types_as_slice().iter().enumerate() {
            println!(
                "  memory type {i}: heap {} {:?}",
                t.heap_index, t.property_flags
            );
        }
        for (i, q) in unsafe { vi.get_physical_device_queue_family_properties(pd) }
            .iter()
            .enumerate()
        {
            println!(
                "  queue family {i}: {} x {:?}",
                q.queue_count, q.queue_flags
            );
        }

        let mut ok = true;
        for e in REQUIRED_EXTENSIONS {
            let h = has(e);
            ok &= h;
            println!("  [{}] {}", check(h), e.to_string_lossy());
        }
        let features = [
            ("shaderInt16", f.shader_int16),
            ("shaderInt64", f.shader_int64),
            ("storageBuffer16BitAccess", f11.storage_buffer16_bit_access),
            ("timelineSemaphore", f12.timeline_semaphore),
            ("vulkanMemoryModel", f12.vulkan_memory_model),
            ("bufferDeviceAddress", f12.buffer_device_address),
            ("subgroupSizeControl", f13.subgroup_size_control),
        ];
        for (n, v) in features {
            ok &= v == vk::TRUE;
            println!("  [{}] {n}", check(v == vk::TRUE));
        }
        println!(
            "  [info] shaderFloat16 = {}, shaderInt8 = {} (not required)",
            f12.shader_float16, f12.shader_int8
        );
        ok &= sub.subgroup_size == 64;
        all_ok &= ok;

        // Create a device with everything enabled, to prove the combination works.
        let (qfam, qcount) = inst.compute_family(pd)?;
        let prios = vec![1.0f32; qcount as usize];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfam)
            .queue_priorities(&prios)];
        let ext_ptrs: Vec<_> = REQUIRED_EXTENSIONS.iter().map(|e| e.as_ptr()).collect();
        let mut e11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .vulkan_memory_model(true)
            .buffer_device_address(true);
        let mut e13 = vk::PhysicalDeviceVulkan13Features::default().subgroup_size_control(true);
        let mut ef = vk::PhysicalDeviceFeatures2::default()
            .features(
                vk::PhysicalDeviceFeatures::default()
                    .shader_int16(true)
                    .shader_int64(true),
            )
            .push_next(&mut e11)
            .push_next(&mut e12)
            .push_next(&mut e13);
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut ef);
        match unsafe { vi.create_device(pd, &dci, None) } {
            Ok(dev) => {
                println!("  device created: queue family {qfam} with {qcount} queues");
                unsafe { dev.destroy_device(None) };
            }
            Err(e) => {
                all_ok = false;
                println!("  device creation FAILED: {e}");
            }
        }
    }

    if gpus.len() != 2 {
        all_ok = false;
        println!("expected 2 GPUs, found {}", gpus.len());
    }
    println!(
        "{}",
        if all_ok {
            "all requirements met"
        } else {
            "requirements NOT met"
        }
    );
    Ok(())
}
