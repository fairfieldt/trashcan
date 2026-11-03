//! Bounded offscreen Vulkan graphics renderer. Native window policy supplies Painter
//! operations; neither the API nor the scene depends on Linux, DRM or Mesa.
//! The present adapter still owns scanout. A readback fence is not a display fence.
use anyhow::{Context, Result, bail, ensure};
use ash::vk;
use noto_sans_mono_bitmap::{FontWeight, RasterHeight, get_raster};
use oscar_desktop::canvas::{Canvas, Painter, Rect, char_w};
use std::{ffi::CStr, path::Path};

const MAX_WORDS: usize = 3 * 1024 * 1024;
const MAX_DRAWS: usize = 8192;
const WAIT_NS: u64 = 500_000_000;
const FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
const VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quad.vert.spv"));
const FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/quad.frag.spv"));

struct Draw {
    push: [u32; 12],
    clip: Rect,
}
/// CPU command recording has fixed limits and owns its upload data. A client cannot
/// change a surface after recording and race the GPU's read of that snapshot.
pub struct Frame {
    width: u32,
    height: u32,
    words: Vec<u32>,
    draws: Vec<Draw>,
    failed: bool,
}
impl Frame {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        ensure!(
            width > 0
                && height > 0
                && width <= 4096
                && height <= 4096
                && width as u64 * height as u64 <= 2 * 1024 * 1024,
            "invalid frame dimensions"
        );
        Ok(Self {
            width,
            height,
            words: Vec::new(),
            draws: Vec::new(),
            failed: false,
        })
    }
    pub fn reset(&mut self) {
        self.words.clear();
        self.draws.clear();
        self.failed = false;
    }
    pub fn check(&self) -> Result<()> {
        ensure!(
            !self.failed,
            "composition exceeds bounded recording storage"
        );
        Ok(())
    }
    fn upload(&mut self, words: impl IntoIterator<Item = u32>, n: usize) -> Option<u32> {
        if n > MAX_WORDS.saturating_sub(self.words.len()) || self.words.try_reserve(n).is_err() {
            self.failed = true;
            return None;
        }
        let at = self.words.len();
        self.words.extend(words);
        Some(at as u32)
    }
    fn draw(&mut self, rect: Rect, clip: Rect, color: u32, mode: u32, offset: u32, stride: u32) {
        let clip = clip.intersect(rect).intersect(self.bounds());
        if clip.w == 0 || clip.h == 0 {
            return;
        }
        if self.draws.len() == MAX_DRAWS || self.draws.try_reserve(1).is_err() {
            self.failed = true;
            return;
        }
        self.draws.push(Draw {
            push: [
                rect.x as u32,
                rect.y as u32,
                rect.w,
                rect.h,
                self.width,
                self.height,
                color,
                mode,
                offset,
                stride,
                0,
                0,
            ],
            clip,
        });
    }
}
impl Painter for Frame {
    fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width, self.height)
    }
    fn fill(&mut self, rect: Rect, color: u32) {
        self.draw(rect, self.bounds(), color & 0xffffff, 0, 0, 0);
    }
    fn desktop(&mut self, color: u32) {
        self.draw(self.bounds(), self.bounds(), color, 3, 0, 0);
    }
    fn blit(&mut self, source: &Canvas, x: i32, y: i32) {
        if let Some(offset) = self.upload(source.pixels().iter().copied(), source.pixels().len()) {
            self.draw(
                Rect::new(x, y, source.width(), source.height()),
                self.bounds(),
                0,
                1,
                offset,
                source.width(),
            );
        }
    }
    fn text(&mut self, x: i32, y: i32, text: &str, color: u32, clip: Rect) {
        let mut x = x;
        for ch in text.chars() {
            if x as i64 >= clip.x as i64 + clip.w as i64 {
                break;
            }
            if let Some(glyph) = get_raster(ch, FontWeight::Regular, RasterHeight::Size16)
                .or_else(|| get_raster('?', FontWeight::Regular, RasterHeight::Size16))
            {
                let n = glyph.width() * glyph.height();
                let words = glyph
                    .rows()
                    .flat_map(|r| r.iter().map(|&a| (color & 0xffffff) | ((a as u32) << 24)));
                if let Some(offset) = self.upload(words, n) {
                    self.draw(
                        Rect::new(x, y, glyph.width() as u32, glyph.height() as u32),
                        clip,
                        0,
                        2,
                        offset,
                        glyph.width() as u32,
                    );
                }
            }
            x = x.saturating_add(char_w());
        }
    }
}

/// A renderer-owned process mapping of imported native Memory. It is separate
/// from the application's window/scanout mapping, including during teardown.
struct NativeMapping {
    memory: dross_rt::Handle,
    at: u64,
    bytes: u64,
}
impl NativeMapping {
    fn alias(pointer: *mut u8, bytes: u64) -> Result<Self> {
        use dross_rt::sys;
        let (memory, offset, remaining) = sys::vm_memory(pointer as u64)
            .map_err(|error| anyhow::anyhow!("native import authority: {error:?}"))?;
        if remaining < bytes {
            let _ = sys::close(memory);
            bail!("native import exceeds the original mapping");
        }
        match sys::vm_map(
            memory,
            offset,
            bytes,
            0,
            dross_abi::vm::READ | dross_abi::vm::WRITE,
        ) {
            Ok(at) => Ok(Self { memory, at, bytes }),
            Err(error) => {
                let _ = sys::close(memory);
                bail!("native import alias: {error:?}");
            }
        }
    }
}
impl Drop for NativeMapping {
    fn drop(&mut self) {
        let _ = dross_rt::sys::vm_unmap(self.at, self.bytes);
        let _ = dross_rt::sys::close(self.memory);
    }
}
#[derive(Default)]
struct Buffer {
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    imported: bool,
    native: Option<NativeMapping>,
}
/// One queue and one bounded submission in flight. All waits have a finite timeout.
pub struct Renderer {
    entry: Option<ash::Entry>,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    descriptor_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pass: vk::RenderPass,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    shaders: Vec<vk::ShaderModule>,
    upload: Buffer,
    readback: Buffer,
    width: u32,
    height: u32,
    poisoned: bool,
    pub device_name: String,
    pub imported_alignment: u64,
    physical_device: vk::PhysicalDevice,
    external_host: bool,
    output_pitch: u32,
}
impl Renderer {
    /// library is only a loader location (e.g. MoltenVK for host correctness tests).
    /// On Oscar, None uses the Vulkan loader and gpud's normal DRM shim.
    pub fn new(width: u32, height: u32, bus: Option<u32>, library: Option<&Path>) -> Result<Self> {
        ensure!(
            width > 0
                && height > 0
                && width <= 4096
                && height <= 4096
                && width as u64 * height as u64 <= 2 * 1024 * 1024,
            "invalid target dimensions"
        );
        // SAFETY: ash owns the dynamic library for the instance's whole lifetime.
        let entry = unsafe {
            match library {
                Some(p) => ash::Entry::load_from(p)?,
                None => ash::Entry::load()?,
            }
        };
        let mut extensions = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        // SAFETY: loader enumeration has no instance dependencies.
        let available = unsafe { entry.enumerate_instance_extension_properties(None)? };
        if available.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == ash::khr::portability_enumeration::NAME) {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let app = vk::ApplicationInfo::default()
            .application_name(c"Oscar compositor")
            .api_version(vk::API_VERSION_1_1);
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .flags(flags)
                    .enabled_extension_names(&extensions),
                None,
            )?
        };
        let selection = (|| -> Result<_> {
            for pdev in unsafe { instance.enumerate_physical_devices()? } {
                let props = unsafe { instance.get_physical_device_properties(pdev) };
                let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                if let Some(bus) = bus {
                    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
                    unsafe {
                        instance.get_physical_device_properties2(
                            pdev,
                            &mut vk::PhysicalDeviceProperties2::default().push_next(&mut pci),
                        )
                    };
                    if pci.pci_bus != bus {
                        continue;
                    }
                }
                let families =
                    unsafe { instance.get_physical_device_queue_family_properties(pdev) };
                if let Some(qfi) = families.iter().position(|q| {
                    q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                }) {
                    let format =
                        unsafe { instance.get_physical_device_format_properties(pdev, FORMAT) };
                    ensure!(
                        format.optimal_tiling_features.contains(
                            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                                | vk::FormatFeatureFlags::TRANSFER_SRC
                        ),
                        "B8G8R8A8 target unsupported"
                    );
                    return Ok((pdev, qfi as u32, name));
                }
            }
            bail!("no matching Vulkan graphics queue")
        })();
        let (pdev, qfi, name) = match selection {
            Ok(s) => s,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e);
            }
        };
        let priority = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi)
            .queue_priorities(&priority)];
        let mut device_extensions = Vec::new();
        let mut external_host = false;
        for ext in unsafe { instance.enumerate_device_extension_properties(pdev)? } {
            if unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }
                == ash::ext::external_memory_host::NAME
            {
                external_host = true;
                device_extensions.push(ash::ext::external_memory_host::NAME.as_ptr());
            }
            if unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) }
                == ash::khr::portability_subset::NAME
            {
                device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
            }
        }
        let device = match unsafe {
            instance.create_device(
                pdev,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&device_extensions),
                None,
            )
        } {
            Ok(d) => d,
            Err(e) => {
                unsafe { instance.destroy_instance(None) };
                return Err(e.into());
            }
        };
        let queue = unsafe { device.get_device_queue(qfi, 0) };
        let mut r = Self {
            entry: Some(entry),
            instance,
            device,
            queue,
            pool: vk::CommandPool::null(),
            cb: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            descriptor: vk::DescriptorSet::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pass: vk::RenderPass::null(),
            image: vk::Image::null(),
            image_memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            framebuffer: vk::Framebuffer::null(),
            shaders: Vec::new(),
            upload: Buffer::default(),
            readback: Buffer::default(),
            width,
            height,
            poisoned: false,
            device_name: name,
            imported_alignment: 0,
            physical_device: pdev,
            external_host,
            output_pitch: width,
        };
        r.initialize(pdev, qfi)?;
        Ok(r)
    }
    fn memory_type(
        &self,
        pdev: vk::PhysicalDevice,
        bits: u32,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<u32> {
        let mem = unsafe { self.instance.get_physical_device_memory_properties(pdev) };
        (0..mem.memory_type_count)
            .find(|i| {
                bits & (1 << i) != 0 && mem.memory_types[*i as usize].property_flags.contains(flags)
            })
            .context("compatible memory type")
    }
    fn buffer(
        &self,
        pdev: vk::PhysicalDevice,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<Buffer> {
        let d = &self.device;
        let handle = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };
        let mut b = Buffer {
            handle,
            ..Buffer::default()
        };
        let result = (|| -> Result<()> {
            let requirements = unsafe { d.get_buffer_memory_requirements(handle) };
            let ty = self.memory_type(
                pdev,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            b.memory = unsafe {
                d.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(ty),
                    None,
                )?
            };
            unsafe {
                d.bind_buffer_memory(handle, b.memory, 0)?;
                b.mapped = d
                    .map_memory(b.memory, 0, size, vk::MemoryMapFlags::empty())?
                    .cast();
            }
            Ok(())
        })();
        if let Err(e) = result {
            unsafe {
                d.destroy_buffer(b.handle, None);
                if b.memory != vk::DeviceMemory::null() {
                    d.free_memory(b.memory, None);
                }
            }
            return Err(e);
        }
        Ok(b)
    }
    /// Import server-owned native Memory as the GPU transfer destination.
    /// The driver retains its own fd, CPU mapping and IOMMU mapping through the
    /// last completion fence, independently of the exporting process.
    ///
    /// # Safety
    /// `pointer..pointer+bytes` must be a writable native Memory mapping during
    /// this call. The renderer then owns an independent alias; do not change its
    /// contents while rendering. Reuse only after render returns successfully
    /// (a finite GPU completion fence).
    pub unsafe fn import_output(&mut self, pointer: *mut u8, bytes: u64, pitch: u32) -> Result<()> {
        ensure!(
            !self.poisoned && self.external_host,
            "external host memory unsupported"
        );
        ensure!(
            pitch >= self.width
                && pitch <= 8192
                && bytes <= 64 << 20
                && bytes >= pitch as u64 * self.height as u64 * 4,
            "output mapping too small"
        );
        let mapping = NativeMapping::alias(pointer, bytes)?;
        let pointer = mapping.at as *mut u8;
        let mut props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        unsafe {
            self.instance.get_physical_device_properties2(
                self.physical_device,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut props),
            );
        }
        let align = props.min_imported_host_pointer_alignment.max(1);
        ensure!(
            (pointer as u64).is_multiple_of(align) && bytes.is_multiple_of(align),
            "host memory alignment {align} required"
        );
        let d = &self.device;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let handle = unsafe {
            d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut external),
                None,
            )?
        };
        let mut replacement = Buffer {
            handle,
            mapped: pointer,
            imported: true,
            native: Some(mapping),
            ..Buffer::default()
        };
        let result = (|| -> Result<()> {
            let requirements = unsafe { d.get_buffer_memory_requirements(handle) };
            ensure!(
                requirements.size <= bytes,
                "native mapping must cover Vulkan requirement {}",
                requirements.size
            );
            let loader = ash::ext::external_memory_host::Device::new(&self.instance, d);
            let mut host = vk::MemoryHostPointerPropertiesEXT::default();
            unsafe {
                (loader.fp().get_memory_host_pointer_properties_ext)(
                    d.handle(),
                    vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                    pointer.cast(),
                    &mut host,
                )
                .result()?;
            }
            let ty = self.memory_type(
                self.physical_device,
                host.memory_type_bits & requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .host_pointer(pointer.cast());
            replacement.memory = unsafe {
                d.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(bytes)
                        .memory_type_index(ty)
                        .push_next(&mut import),
                    None,
                )?
            };
            unsafe {
                d.bind_buffer_memory(handle, replacement.memory, 0)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            unsafe {
                d.destroy_buffer(handle, None);
                if replacement.memory != vk::DeviceMemory::null() {
                    d.free_memory(replacement.memory, None);
                }
            }
            return Err(error);
        }
        let old = core::mem::replace(&mut self.readback, replacement);
        unsafe {
            if !old.imported && !old.mapped.is_null() {
                d.unmap_memory(old.memory);
            }
            d.destroy_buffer(old.handle, None);
            d.free_memory(old.memory, None);
        }
        self.output_pitch = pitch;
        self.imported_alignment = align;
        Ok(())
    }
    fn initialize(&mut self, pdev: vk::PhysicalDevice, qfi: u32) -> Result<()> {
        let d = &self.device;
        self.pool = unsafe {
            d.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qfi)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        self.cb = unsafe {
            d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .command_buffer_count(1),
            )?[0]
        };
        self.fence = unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? };
        self.upload = self.buffer(
            pdev,
            (MAX_WORDS * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        self.readback = self.buffer(
            pdev,
            self.width as u64 * self.height as u64 * 4,
            vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        self.descriptor_layout = unsafe {
            d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )?
        };
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        self.descriptor_pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?
        };
        self.descriptor = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&[self.descriptor_layout]),
            )?[0]
        };
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(self.upload.handle)
            .range((MAX_WORDS * 4) as u64)];
        unsafe {
            d.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info)],
                &[],
            );
        }
        let range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(48)];
        self.layout = unsafe {
            d.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[self.descriptor_layout])
                    .push_constant_ranges(&range),
                None,
            )?
        };
        self.image = unsafe {
            d.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(FORMAT)
                    .extent(vk::Extent3D {
                        width: self.width,
                        height: self.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };
        let requirements = unsafe { d.get_image_memory_requirements(self.image) };
        let ty = self.memory_type(
            pdev,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.image_memory = unsafe {
            d.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(ty),
                None,
            )?
        };
        unsafe {
            d.bind_image_memory(self.image, self.image_memory, 0)?;
        }
        let subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        self.view = unsafe {
            d.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(self.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(FORMAT)
                    .subresource_range(subresource),
                None,
            )?
        };
        let attachments = [vk::AttachmentDescription::default()
            .format(FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
        let color = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color)];
        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        self.pass = unsafe {
            d.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies),
                None,
            )?
        };
        self.framebuffer = unsafe {
            d.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.pass)
                    .attachments(&[self.view])
                    .width(self.width)
                    .height(self.height)
                    .layers(1),
                None,
            )?
        };
        for code in [VERT, FRAG] {
            let words = ash::util::read_spv(&mut std::io::Cursor::new(code))?;
            self.shaders.push(unsafe {
                d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
            });
        }
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.shaders[0])
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.shaders[1])
                .name(c"main"),
        ];
        let vertex = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let samples = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamics = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
        let create = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex)
            .input_assembly_state(&assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&samples)
            .color_blend_state(&blending)
            .dynamic_state(&dynamics)
            .layout(self.layout)
            .render_pass(self.pass)];
        self.pipeline = unsafe {
            d.create_graphics_pipelines(vk::PipelineCache::null(), &create, None)
                .map_err(|(created, e)| {
                    for pipeline in created {
                        if pipeline != vk::Pipeline::null() {
                            d.destroy_pipeline(pipeline, None);
                        }
                    }
                    e
                })?[0]
        };
        Ok(())
    }
    pub fn render(&mut self, frame: &Frame) -> Result<&[u32]> {
        ensure!(!self.poisoned, "renderer lost; restart required");
        frame.check()?;
        ensure!(
            (frame.width, frame.height) == (self.width, self.height),
            "frame dimensions differ"
        );
        let d = &self.device;
        // SAFETY: recording owns upload words; no preceding submission remains in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(
                frame.words.as_ptr(),
                self.upload.mapped.cast(),
                frame.words.len(),
            );
            d.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            d.begin_command_buffer(
                self.cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)];
            d.cmd_pipeline_barrier(
                self.cb,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &host,
                &[],
                &[],
            );
            let area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.width,
                    height: self.height,
                },
            };
            let clear = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0., 0., 0., 1.],
                },
            }];
            d.cmd_begin_render_pass(
                self.cb,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.pass)
                    .framebuffer(self.framebuffer)
                    .render_area(area)
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            d.cmd_bind_pipeline(self.cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            d.cmd_bind_descriptor_sets(
                self.cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.descriptor],
                &[],
            );
            d.cmd_set_viewport(
                self.cb,
                0,
                &[vk::Viewport {
                    x: 0.,
                    y: 0.,
                    width: self.width as f32,
                    height: self.height as f32,
                    min_depth: 0.,
                    max_depth: 1.,
                }],
            );
            for draw in &frame.draws {
                d.cmd_set_scissor(
                    self.cb,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D {
                            x: draw.clip.x,
                            y: draw.clip.y,
                        },
                        extent: vk::Extent2D {
                            width: draw.clip.w,
                            height: draw.clip.h,
                        },
                    }],
                );
                let bytes = std::slice::from_raw_parts(draw.push.as_ptr().cast::<u8>(), 48);
                d.cmd_push_constants(
                    self.cb,
                    self.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytes,
                );
                d.cmd_draw(self.cb, 6, 1, 0, 0);
            }
            d.cmd_end_render_pass(self.cb);
            let region = [vk::BufferImageCopy::default()
                .buffer_row_length(self.output_pitch)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                })];
            d.cmd_copy_image_to_buffer(
                self.cb,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback.handle,
                &region,
            );
            let read = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            d.cmd_pipeline_barrier(
                self.cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &read,
                &[],
                &[],
            );
            d.end_command_buffer(self.cb)?;
            d.reset_fences(&[self.fence])?;
            let cbs = [self.cb];
            if let Err(e) = d.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                self.fence,
            ) {
                self.poisoned = true;
                return Err(e.into());
            }
            if let Err(e) = d.wait_for_fences(&[self.fence], true, WAIT_NS) {
                self.poisoned = true;
                return Err(anyhow::anyhow!(
                    "graphics fence failed within {} ms: {e:?}",
                    WAIT_NS / 1_000_000
                ));
            }
            Ok(std::slice::from_raw_parts(
                self.readback.mapped.cast(),
                self.output_pitch as usize * self.height as usize,
            ))
        }
    }
}
impl Drop for Renderer {
    fn drop(&mut self) {
        // Do not free backing that a timed-out GPU may still access. The server must
        // exit. gpud retains imported and owned backing until its tracked fences finish.
        // No unbounded device_wait_idle on shutdown.
        if self.poisoned {
            // The caller may drop its window/scanout now. Keep Vulkan's separate
            // CPU alias valid for worker threads until the process itself exits.
            if let Some(mapping) = self.readback.native.take() {
                core::mem::forget(mapping);
            }
            // The leaked Vulkan objects may still have driver worker threads.
            // Keep their dynamic library loaded until process exit as well.
            if let Some(entry) = self.entry.take() {
                core::mem::forget(entry);
            }
            return;
        }
        let d = &self.device;
        unsafe {
            if self.pipeline != vk::Pipeline::null() {
                d.destroy_pipeline(self.pipeline, None);
            }
            for &s in &self.shaders {
                d.destroy_shader_module(s, None);
            }
            if self.framebuffer != vk::Framebuffer::null() {
                d.destroy_framebuffer(self.framebuffer, None);
            }
            if self.pass != vk::RenderPass::null() {
                d.destroy_render_pass(self.pass, None);
            }
            if self.view != vk::ImageView::null() {
                d.destroy_image_view(self.view, None);
            }
            if self.image != vk::Image::null() {
                d.destroy_image(self.image, None);
            }
            if self.image_memory != vk::DeviceMemory::null() {
                d.free_memory(self.image_memory, None);
            }
            if self.layout != vk::PipelineLayout::null() {
                d.destroy_pipeline_layout(self.layout, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                d.destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.descriptor_layout != vk::DescriptorSetLayout::null() {
                d.destroy_descriptor_set_layout(self.descriptor_layout, None);
            }
            for b in [&self.upload, &self.readback] {
                if !b.imported && !b.mapped.is_null() {
                    d.unmap_memory(b.memory);
                }
                if b.handle != vk::Buffer::null() {
                    d.destroy_buffer(b.handle, None);
                }
                if b.memory != vk::DeviceMemory::null() {
                    d.free_memory(b.memory, None);
                }
            }
            if self.fence != vk::Fence::null() {
                d.destroy_fence(self.fence, None);
            }
            if self.pool != vk::CommandPool::null() {
                d.destroy_command_pool(self.pool, None);
            }
            d.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
    }
}
