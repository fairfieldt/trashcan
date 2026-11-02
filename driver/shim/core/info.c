// SPDX-License-Identifier: MIT
/*
 * Answers to DRM_AMDGPU_INFO queries, ported from amdgpu_info_ioctl and amdgpu_hw_ip_info (Linux 7.3
 * amdgpu_kms.c, Copyright Advanced Micro Devices, Inc., MIT license), reduced to what a Southern Islands device
 * without display, video, DPM or partitions has. Memory heaps describe gpud's placement: no CPU-visible VRAM
 * (clients' host-visible memory is host pages), VRAM above gpud's kernel carve-out.
 */
#include "amdgpu.h"
#include "amd_pcie.h"
#include <gpud_core.h>

u64 gpud_vram_free_bytes(void);
extern u64 gpud_client_vram_start;
extern u64 gpud_gtt_size;

static int put(void *out, u32 size, const void *val, size_t len)
{
	memcpy(out, val, min_t(size_t, size, len));
	return 0;
}

static int hw_ip_info(struct amdgpu_device *adev, const struct drm_amdgpu_info *info,
		      struct drm_amdgpu_info_hw_ip *result)
{
	enum amd_ip_block_type type;
	struct amdgpu_ip_block *block;
	unsigned int num_rings = 0, i;
	u32 ib_start_alignment = 0, ib_size_alignment = 0;

	if (info->query_hw_ip.ip_instance >= AMDGPU_HW_IP_INSTANCE_MAX_COUNT)
		return -EINVAL;
	switch (info->query_hw_ip.type) {
	case AMDGPU_HW_IP_GFX:
		type = AMD_IP_BLOCK_TYPE_GFX;
		for (i = 0; i < adev->gfx.num_gfx_rings; i++)
			num_rings += adev->gfx.gfx_ring[i].sched.ready;
		ib_start_alignment = 32;
		ib_size_alignment = 32;
		break;
	case AMDGPU_HW_IP_COMPUTE:
		type = AMD_IP_BLOCK_TYPE_GFX;
		for (i = 0; i < adev->gfx.num_compute_rings; i++)
			num_rings += adev->gfx.compute_ring[i].sched.ready;
		ib_start_alignment = 32;
		ib_size_alignment = 32;
		break;
	case AMDGPU_HW_IP_DMA:
		type = AMD_IP_BLOCK_TYPE_SDMA;
		for (i = 0; i < (unsigned)adev->sdma.num_instances; i++)
			num_rings += adev->sdma.instance[i].ring.sched.ready;
		ib_start_alignment = 256;
		ib_size_alignment = 4;
		break;
	case AMDGPU_HW_IP_UVD:
	case AMDGPU_HW_IP_VCE:
	case AMDGPU_HW_IP_UVD_ENC:
	case AMDGPU_HW_IP_VCN_DEC:
	case AMDGPU_HW_IP_VCN_ENC:
	case AMDGPU_HW_IP_VCN_JPEG:
		/* video blocks aren't run */
		memset(result, 0, sizeof(*result));
		return 0;
	default:
		return -EINVAL;
	}
	block = amdgpu_device_ip_get_ip_block(adev, type);
	memset(result, 0, sizeof(*result));
	if (block && block->status.valid) {
		result->hw_ip_version_major = block->version->major;
		result->hw_ip_version_minor = block->version->minor;
	}
	result->available_rings = (1 << num_rings) - 1;
	result->ib_start_alignment = ib_start_alignment;
	result->ib_size_alignment = ib_size_alignment;
	return 0;
}

static void firmware_info(struct amdgpu_device *adev, u32 fw_type, struct drm_amdgpu_info_firmware *fw)
{
	memset(fw, 0, sizeof(*fw));
	switch (fw_type) {
	case AMDGPU_INFO_FW_GMC:
		fw->ver = adev->gmc.fw_version;
		break;
	case AMDGPU_INFO_FW_GFX_ME:
		fw->ver = adev->gfx.me_fw_version;
		fw->feature = adev->gfx.me_feature_version;
		break;
	case AMDGPU_INFO_FW_GFX_PFP:
		fw->ver = adev->gfx.pfp_fw_version;
		fw->feature = adev->gfx.pfp_feature_version;
		break;
	case AMDGPU_INFO_FW_GFX_CE:
		fw->ver = adev->gfx.ce_fw_version;
		fw->feature = adev->gfx.ce_feature_version;
		break;
	case AMDGPU_INFO_FW_GFX_RLC:
		fw->ver = adev->gfx.rlc_fw_version;
		fw->feature = adev->gfx.rlc_feature_version;
		break;
	default:
		break;
	}
}

static u64 vram_usable(struct amdgpu_device *adev)
{
	return adev->gmc.real_vram_size - gpud_client_vram_start - AMDGPU_VM_RESERVED_VRAM;
}

int gpud_info(struct amdgpu_device *adev, const struct drm_amdgpu_info *info, void *out, u32 size)
{
	u32 ui32 = 0;
	u64 ui64 = 0;

	if (!size)
		return -EINVAL;
	switch (info->query) {
	case AMDGPU_INFO_ACCEL_WORKING:
		ui32 = adev->accel_working;
		return put(out, size, &ui32, 4);
	case AMDGPU_INFO_HW_IP_INFO: {
		struct drm_amdgpu_info_hw_ip ip;
		int r = hw_ip_info(adev, info, &ip);

		return r ? r : put(out, size, &ip, sizeof(ip));
	}
	case AMDGPU_INFO_HW_IP_COUNT:
		switch (info->query_hw_ip.type) {
		case AMDGPU_HW_IP_GFX:
		case AMDGPU_HW_IP_COMPUTE:
			ui32 = 1;
			break;
		case AMDGPU_HW_IP_DMA:
			ui32 = adev->sdma.num_instances;
			break;
		default:
			return -EINVAL;
		}
		return put(out, size, &ui32, 4);
	case AMDGPU_INFO_TIMESTAMP:
		ui64 = amdgpu_gfx_get_gpu_clock_counter(adev);
		return put(out, size, &ui64, 8);
	case AMDGPU_INFO_FW_VERSION: {
		struct drm_amdgpu_info_firmware fw;

		if (info->query_fw.ip_instance != 0)
			return -EINVAL;
		firmware_info(adev, info->query_fw.fw_type, &fw);
		return put(out, size, &fw, sizeof(fw));
	}
	case AMDGPU_INFO_NUM_BYTES_MOVED:
	case AMDGPU_INFO_NUM_EVICTIONS:
	case AMDGPU_INFO_NUM_VRAM_CPU_PAGE_FAULTS:
	case AMDGPU_INFO_VIS_VRAM_USAGE:
	case AMDGPU_INFO_GTT_USAGE:
		return put(out, size, &ui64, 8);
	case AMDGPU_INFO_VRAM_USAGE:
		ui64 = vram_usable(adev) + AMDGPU_VM_RESERVED_VRAM - gpud_vram_free_bytes();
		return put(out, size, &ui64, 8);
	case AMDGPU_INFO_VRAM_GTT: {
		struct drm_amdgpu_info_vram_gtt vg = {
			.vram_size = vram_usable(adev),
			.vram_cpu_accessible_size = 0,
			.gtt_size = gpud_gtt_size,
		};

		return put(out, size, &vg, sizeof(vg));
	}
	case AMDGPU_INFO_MEMORY: {
		struct drm_amdgpu_memory_info mem;

		memset(&mem, 0, sizeof(mem));
		mem.vram.total_heap_size = adev->gmc.real_vram_size;
		mem.vram.usable_heap_size = vram_usable(adev);
		mem.vram.heap_usage = mem.vram.usable_heap_size + AMDGPU_VM_RESERVED_VRAM - gpud_vram_free_bytes();
		mem.vram.max_allocation = mem.vram.usable_heap_size * 3 / 4;
		mem.gtt.total_heap_size = gpud_gtt_size;
		mem.gtt.usable_heap_size = gpud_gtt_size;
		mem.gtt.max_allocation = gpud_gtt_size * 3 / 4;
		return put(out, size, &mem, sizeof(mem));
	}
	case AMDGPU_INFO_READ_MMR_REG: {
		unsigned int se_num = (info->read_mmr_reg.instance >> AMDGPU_INFO_MMR_SE_INDEX_SHIFT) &
				      AMDGPU_INFO_MMR_SE_INDEX_MASK;
		unsigned int sh_num = (info->read_mmr_reg.instance >> AMDGPU_INFO_MMR_SH_INDEX_SHIFT) &
				      AMDGPU_INFO_MMR_SH_INDEX_MASK;
		u32 regs[128];

		if (se_num == AMDGPU_INFO_MMR_SE_INDEX_MASK)
			se_num = 0xffffffff;
		else if (se_num >= AMDGPU_GFX_MAX_SE)
			return -EINVAL;
		if (sh_num == AMDGPU_INFO_MMR_SH_INDEX_MASK)
			sh_num = 0xffffffff;
		else if (sh_num >= AMDGPU_GFX_MAX_SH_PER_SE)
			return -EINVAL;
		if (info->read_mmr_reg.count > 128)
			return -EINVAL;
		for (u32 i = 0; i < info->read_mmr_reg.count; i++)
			if (amdgpu_asic_read_register(adev, se_num, sh_num, info->read_mmr_reg.dword_offset + i, &regs[i]))
				return -EFAULT;
		return put(out, size, regs, info->read_mmr_reg.count * 4);
	}
	case AMDGPU_INFO_DEV_INFO: {
		struct drm_amdgpu_info_device *d = calloc(1, sizeof(*d));
		u64 vm_size;
		u32 pcie_gen_mask, pcie_width_mask;
		int r;

		if (!d)
			return -ENOMEM;
		d->device_id = adev->pdev->device;
		d->chip_rev = adev->rev_id;
		d->external_rev = adev->external_rev_id;
		d->pci_rev = adev->pdev->revision;
		d->family = adev->family;
		d->num_shader_engines = adev->gfx.config.max_shader_engines;
		d->num_shader_arrays_per_engine = adev->gfx.config.max_sh_per_se;
		d->gpu_counter_freq = amdgpu_asic_get_xclk(adev) * 10;
		d->max_engine_clock = d->min_engine_clock = adev->clock.default_sclk * 10;
		d->max_memory_clock = d->min_memory_clock = adev->clock.default_mclk * 10;
		d->enabled_rb_pipes_mask = adev->gfx.config.backend_enable_mask;
		d->num_rb_pipes = adev->gfx.config.max_backends_per_se * adev->gfx.config.max_shader_engines;
		d->num_hw_gfx_contexts = adev->gfx.config.max_hw_contexts;
		d->ids_flags = AMDGPU_IDS_FLAGS_GANG_SUBMIT;
		vm_size = adev->vm_manager.max_pfn * AMDGPU_GPU_PAGE_SIZE - AMDGPU_VA_RESERVED_TOP;
		d->virtual_address_offset = AMDGPU_VA_RESERVED_BOTTOM;
		d->virtual_address_max = vm_size;
		d->virtual_address_alignment = max_t(u32, PAGE_SIZE, AMDGPU_GPU_PAGE_SIZE);
		d->pte_fragment_size = (1 << adev->vm_manager.fragment_size) * AMDGPU_GPU_PAGE_SIZE;
		d->gart_page_size = max_t(u32, PAGE_SIZE, AMDGPU_GPU_PAGE_SIZE);
		d->cu_active_number = adev->gfx.cu_info.number;
		d->cu_ao_mask = adev->gfx.cu_info.ao_cu_mask;
		d->ce_ram_size = adev->gfx.ce_ram_size;
		memcpy(&d->cu_ao_bitmap[0], &adev->gfx.cu_info.ao_cu_bitmap[0], sizeof(adev->gfx.cu_info.ao_cu_bitmap));
		memcpy(&d->cu_bitmap[0], &adev->gfx.cu_info.bitmap[0], sizeof(d->cu_bitmap));
		d->vram_type = adev->gmc.vram_type;
		d->vram_bit_width = adev->gmc.vram_width;
		d->gc_double_offchip_lds_buf = adev->gfx.config.double_offchip_lds_buf;
		d->wave_front_size = adev->gfx.cu_info.wave_front_size;
		d->num_shader_visible_vgprs = adev->gfx.config.max_gprs;
		d->num_cu_per_sh = adev->gfx.config.max_cu_per_sh;
		d->num_tcc_blocks = adev->gfx.config.max_texture_channel_caches;
		d->gs_vgt_table_depth = adev->gfx.config.gs_vgt_table_depth;
		d->gs_prim_buffer_depth = adev->gfx.config.gs_prim_buffer_depth;
		d->max_gs_waves_per_vgt = adev->gfx.config.max_gs_threads;
		d->tcc_disabled_mask = adev->gfx.config.tcc_disabled_mask;
		pcie_gen_mask = adev->pm.pcie_gen_mask & (adev->pm.pcie_gen_mask >> CAIL_PCIE_LINK_SPEED_SUPPORT_SHIFT);
		pcie_width_mask = adev->pm.pcie_mlw_mask & (adev->pm.pcie_mlw_mask >> CAIL_PCIE_LINK_WIDTH_SUPPORT_SHIFT);
		d->pcie_gen = fls(pcie_gen_mask);
		d->pcie_num_lanes = pcie_width_mask & CAIL_ASIC_PCIE_LINK_WIDTH_SUPPORT_X32 ? 32 :
				    pcie_width_mask & CAIL_ASIC_PCIE_LINK_WIDTH_SUPPORT_X16 ? 16 :
				    pcie_width_mask & CAIL_ASIC_PCIE_LINK_WIDTH_SUPPORT_X12 ? 12 :
				    pcie_width_mask & CAIL_ASIC_PCIE_LINK_WIDTH_SUPPORT_X8  ? 8 :
				    pcie_width_mask & CAIL_ASIC_PCIE_LINK_WIDTH_SUPPORT_X4  ? 4 :
				    pcie_width_mask & CAIL_ASIC_PCIE_LINK_WIDTH_SUPPORT_X2  ? 2 : 1;
		r = put(out, size, d, sizeof(*d));
		free(d);
		return r;
	}
	case AMDGPU_INFO_VIDEO_CAPS: {
		struct drm_amdgpu_info_video_caps caps;

		memset(&caps, 0, sizeof(caps));
		return put(out, size, &caps, sizeof(caps));
	}
	case AMDGPU_INFO_GPUVM_FAULT: {
		struct drm_amdgpu_info_gpuvm_fault fault;

		memset(&fault, 0, sizeof(fault));
		return put(out, size, &fault, sizeof(fault));
	}
	default:
		return -EINVAL;
	}
}
