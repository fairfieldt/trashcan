// SPDX-License-Identifier: MIT
/*
 * gpud's device core, in place of amdgpu_device.c, amdgpu_ip.c and amdgpu_reg_access.c: register access, the IP
 * block sequence (amdgpu_device_ip_early_init/ip_init/ip_late_init, reduced to the blocks gpud runs), and the
 * device-level helpers the SI files call. Function bodies marked "Linux 7.3" are copied from the named file
 * (Copyright Advanced Micro Devices, Inc., MIT license).
 */
#include "amdgpu.h"
#include "gpud_dropped_ip.h"
#include "si.h"
#include "gca/gfx_6_0_d.h"
#include "gmc/gmc_6_0_d.h"
#include "oss/oss_1_0_d.h"
#include "gca/gfx_6_0_sh_mask.h"
#include "gmc/gmc_6_0_sh_mask.h"
#include "oss/oss_1_0_sh_mask.h"
#include <gpud_core.h>
#include "amd_pcie.h"
#include <linux/pci.h>

extern const struct gpud_platform *lkpi_platform;
extern const char *lkpi_firmware_dir;

/* amdgpu_drv.c defaults */
int amdgpu_pcie_gen2 = -1;
int amdgpu_reset_method = -1;
int amdgpu_emu_mode;
int amdgpu_vm_fault_stop;
int amdgpu_dpm = -1;
int amdgpu_sched_hw_submission = 2;
int amdgpu_vm_size = -1;
int amdgpu_vm_block_size = -1;
int amdgpu_gart_size = -1;
int amdgpu_gtt_size = -1;
uint amdgpu_cg_mask = 0xffffffff;
uint amdgpu_pg_mask = 0xffffffff;
int amdgpu_disable_cu_count;
char *amdgpu_disable_cu;
int amdgpu_aspm = -1;
int amdgpu_msi = -1;
int amdgpu_num_kcq = -1;
int amdgpu_vm_update_mode = -1;
int amdgpu_dc = -1;
int amdgpu_lockup_timeout;

/* ---- registers (amdgpu_reg_access.c, without SR-IOV, KIQ and RAS) ---- */
uint32_t amdgpu_device_rreg(struct amdgpu_device *adev, uint32_t reg, uint32_t acc_flags)
{
	(void)acc_flags;
	if ((u64)reg * 4 < adev->rmmio_size)
		return readl((u8 __iomem *)adev->rmmio + reg * 4);
	return amdgpu_reg_pcie_rd32(adev, reg * 4);
}

void amdgpu_device_wreg(struct amdgpu_device *adev, uint32_t reg, uint32_t v, uint32_t acc_flags)
{
	(void)acc_flags;
	if ((u64)reg * 4 < adev->rmmio_size)
		writel(v, (u8 __iomem *)adev->rmmio + reg * 4);
	else
		amdgpu_reg_pcie_wr32(adev, reg * 4, v);
}

uint32_t amdgpu_reg_smc_rd32(struct amdgpu_device *adev, uint32_t reg) { return adev->reg.smc.rreg(adev, reg); }
void amdgpu_reg_smc_wr32(struct amdgpu_device *adev, uint32_t reg, uint32_t v) { adev->reg.smc.wreg(adev, reg, v); }
uint32_t amdgpu_reg_uvd_ctx_rd32(struct amdgpu_device *adev, uint32_t reg) { return adev->reg.uvd_ctx.rreg(adev, reg); }
void amdgpu_reg_uvd_ctx_wr32(struct amdgpu_device *adev, uint32_t reg, uint32_t v) { adev->reg.uvd_ctx.wreg(adev, reg, v); }
uint32_t amdgpu_reg_pcie_rd32(struct amdgpu_device *adev, uint32_t reg) { return adev->reg.pcie.rreg(adev, reg); }
void amdgpu_reg_pcie_wr32(struct amdgpu_device *adev, uint32_t reg, uint32_t v) { adev->reg.pcie.wreg(adev, reg, v); }
uint32_t amdgpu_reg_pciep_rd32(struct amdgpu_device *adev, uint32_t reg) { return adev->reg.pcie.port_rreg(adev, reg); }
void amdgpu_reg_pciep_wr32(struct amdgpu_device *adev, uint32_t reg, uint32_t v) { adev->reg.pcie.port_wreg(adev, reg, v); }

/* Linux 7.3 amdgpu_device.c */
void amdgpu_device_program_register_sequence(struct amdgpu_device *adev, const u32 *registers, const u32 array_size)
{
	u32 tmp, reg, and_mask, or_mask;
	int i;

	if (array_size % 3)
		return;

	for (i = 0; i < array_size; i += 3) {
		reg = registers[i + 0];
		and_mask = registers[i + 1];
		or_mask = registers[i + 2];

		if (and_mask == 0xffffffff) {
			tmp = or_mask;
		} else {
			tmp = RREG32(reg);
			tmp &= ~and_mask;
			if (adev->family >= AMDGPU_FAMILY_AI)
				tmp |= (or_mask & and_mask);
			else
				tmp |= or_mask;
		}
		WREG32(reg, tmp);
	}
}

void amdgpu_device_flush_hdp(struct amdgpu_device *adev, struct amdgpu_ring *ring)
{
	if (ring && ring->funcs->emit_hdp_flush)
		amdgpu_ring_emit_hdp_flush(ring);
	else
		amdgpu_asic_flush_hdp(adev, ring);
}

void amdgpu_device_invalidate_hdp(struct amdgpu_device *adev, struct amdgpu_ring *ring)
{
	amdgpu_asic_invalidate_hdp(adev, ring);
}

/* No display, no ASPM, no resets yet (a bus reset needs ASIC init from the VBIOS first). */
bool amdgpu_device_has_dc_support(struct amdgpu_device *adev) { (void)adev; return false; }
bool amdgpu_device_should_use_aspm(struct amdgpu_device *adev) { (void)adev; return false; }
int amdgpu_device_pci_reset(struct amdgpu_device *adev) { (void)adev; return -ENOTSUPP; }
void amdgpu_device_pci_config_reset(struct amdgpu_device *adev) { dev_warn(adev->dev, "config reset not supported\n"); }
int amdgpu_device_resize_fb_bar(struct amdgpu_device *adev) { (void)adev; return 0; }
uint32_t amdgpu_get_soft_full_reset_mask(struct amdgpu_ring *ring) { (void)ring; return 0; }
bool amdgpu_read_bios(struct amdgpu_device *adev) { (void)adev; return false; }
void amdgpu_atombios_scratch_regs_engine_hung(struct amdgpu_device *adev, bool hung) { (void)adev; (void)hung; }
int amdgpu_gtt_mgr_recover(void *mgr) { (void)mgr; return 0; }
void amdgpu_gem_force_release(struct amdgpu_device *adev) { (void)adev; }
void amdgpu_vmid_reset_all(struct amdgpu_device *adev) { (void)adev; }

void amdgpu_vm_update_fault_cache(struct amdgpu_device *adev, unsigned int pasid, uint64_t addr, uint32_t status,
				  unsigned int vmhub)
{
	dev_err(adev->dev, "VM fault at %#llx (status %#x, pasid %u, hub %u)\n", (unsigned long long)addr, status, pasid,
		vmhub);
}

void amdgpu_sdma_set_buffer_funcs_scheds(struct amdgpu_device *adev, const struct amdgpu_buffer_funcs *buffer_funcs)
{
	adev->mman.buffer_funcs = buffer_funcs;
}

void amdgpu_sdma_set_vm_pte_scheds(struct amdgpu_device *adev, const struct amdgpu_vm_pte_funcs *vm_pte_funcs)
{
	adev->vm_manager.vm_pte_funcs = vm_pte_funcs;
}

/* ---- IP blocks ---- */
static const struct amd_ip_funcs dropped_ip_funcs = { .name = "dropped" };
const struct amdgpu_ip_block_version dce_v6_0_ip_block = { .type = AMD_IP_BLOCK_TYPE_DCE, .funcs = &dropped_ip_funcs };
const struct amdgpu_ip_block_version dce_v6_4_ip_block = { .type = AMD_IP_BLOCK_TYPE_DCE, .funcs = &dropped_ip_funcs };
const struct amdgpu_ip_block_version dm_ip_block = { .type = AMD_IP_BLOCK_TYPE_DCE, .funcs = &dropped_ip_funcs };
const struct amdgpu_ip_block_version amdgpu_vkms_ip_block = { .type = AMD_IP_BLOCK_TYPE_DCE, .funcs = &dropped_ip_funcs };
const struct amdgpu_ip_block_version uvd_v3_1_ip_block = { .type = AMD_IP_BLOCK_TYPE_UVD, .funcs = &dropped_ip_funcs };
const struct amdgpu_ip_block_version vce_v1_0_ip_block = { .type = AMD_IP_BLOCK_TYPE_VCE, .funcs = &dropped_ip_funcs };
const struct amdgpu_ip_block_version si_smu_ip_block = { .type = AMD_IP_BLOCK_TYPE_SMC, .funcs = &dropped_ip_funcs };

int amdgpu_device_ip_block_add(struct amdgpu_device *adev, const struct amdgpu_ip_block_version *v)
{
	if (!v)
		return -EINVAL;
	dev_info(adev->dev, "ip block %d: %s v%d.%d.%d\n", adev->num_ip_blocks, v->funcs->name, v->major, v->minor,
		 v->rev);
	adev->ip_blocks[adev->num_ip_blocks].adev = adev;
	adev->ip_blocks[adev->num_ip_blocks++].version = v;
	return 0;
}

struct amdgpu_ip_block *amdgpu_device_ip_get_ip_block(struct amdgpu_device *adev, enum amd_ip_block_type type)
{
	for (int i = 0; i < adev->num_ip_blocks; i++)
		if (adev->ip_blocks[i].version->type == type)
			return &adev->ip_blocks[i];
	return NULL;
}

/* Registers that tell why a ring stalled: CP and RLC state, ring pointers, GART setup and VM faults. */
void gpud_debug_dump(struct amdgpu_device *adev)
{
	static const struct { const char *name; u32 reg; } regs[] = {
		{ "GRBM_STATUS", mmGRBM_STATUS }, { "GRBM_STATUS2", mmGRBM_STATUS2 }, { "SRBM_STATUS", mmSRBM_STATUS },
		{ "CP_STAT", mmCP_STAT }, { "CP_BUSY_STAT", mmCP_BUSY_STAT }, { "CP_STALLED_STAT1", mmCP_STALLED_STAT1 },
		{ "CP_STALLED_STAT2", mmCP_STALLED_STAT2 }, { "CP_ME_CNTL", mmCP_ME_CNTL },
		{ "CP_ME_HEADER_DUMP", mmCP_ME_HEADER_DUMP }, { "CP_PFP_HEADER_DUMP", mmCP_PFP_HEADER_DUMP },
		{ "CP_RB0_BASE", mmCP_RB0_BASE }, { "CP_RB0_CNTL", mmCP_RB0_CNTL }, { "CP_RB0_RPTR", mmCP_RB0_RPTR },
		{ "CP_RB0_WPTR", mmCP_RB0_WPTR }, { "CP_RB0_RPTR_ADDR", mmCP_RB0_RPTR_ADDR }, { "RLC_CNTL", mmRLC_CNTL },
		{ "RLC_STAT", mmRLC_STAT }, { "SCRATCH_REG0", mmSCRATCH_REG0 }, { "VM_L2_CNTL", mmVM_L2_CNTL },
		{ "MC_VM_MX_L1_TLB_CNTL", mmMC_VM_MX_L1_TLB_CNTL }, { "VM_CONTEXT0_CNTL", mmVM_CONTEXT0_CNTL },
		{ "VM_CONTEXT0_PT_BASE", mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR },
		{ "VM_CONTEXT0_PT_START", mmVM_CONTEXT0_PAGE_TABLE_START_ADDR },
		{ "VM_CONTEXT0_PT_END", mmVM_CONTEXT0_PAGE_TABLE_END_ADDR },
		{ "VM_CONTEXT0_FAULT_DEFAULT", mmVM_CONTEXT0_PROTECTION_FAULT_DEFAULT_ADDR },
		{ "VM_CONTEXT1_FAULT_STATUS", mmVM_CONTEXT1_PROTECTION_FAULT_STATUS },
		{ "VM_CONTEXT1_FAULT_ADDR", mmVM_CONTEXT1_PROTECTION_FAULT_ADDR },
		{ "MC_VM_SYS_APERTURE_DEFAULT", mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR },
		{ "MC_VM_FB_LOCATION", mmMC_VM_FB_LOCATION }, { "MC_VM_AGP_BOT", mmMC_VM_AGP_BOT },
		{ "MC_VM_AGP_TOP", mmMC_VM_AGP_TOP }, { "DMA0_STATUS", mmDMA_STATUS_REG },
		{ "DMA0_RB_CNTL", mmDMA_GFX_RB_CNTL }, { "DMA0_RB_RPTR", mmDMA_GFX_RB_RPTR },
		{ "DMA0_RB_WPTR", mmDMA_GFX_RB_WPTR }, { "CP_RB_VMID", mmCP_RB_VMID },
		{ "CP_RB1_BASE", mmCP_RB1_BASE }, { "CP_RB1_CNTL", mmCP_RB1_CNTL }, { "CP_RB1_RPTR", mmCP_RB1_RPTR },
		{ "CP_RB1_WPTR", mmCP_RB1_WPTR }, { "CP_RB2_BASE", mmCP_RB2_BASE }, { "CP_RB2_CNTL", mmCP_RB2_CNTL },
		{ "CP_RB2_RPTR", mmCP_RB2_RPTR }, { "CP_RB2_WPTR", mmCP_RB2_WPTR }, { "CP_ROQ_RB_STAT", mmCP_ROQ_RB_STAT },
		{ "CP_MEQ_STAT", mmCP_MEQ_STAT }, { "CP_CE_HEADER_DUMP", mmCP_CE_HEADER_DUMP },
		{ "CP_INT_STAT_DEBUG", mmCP_INT_STAT_DEBUG }, { "CP_COHER_STATUS", mmCP_COHER_STATUS },
	};
	for (unsigned i = 0; i < ARRAY_SIZE(regs); i++)
		lkpi_log(LKPI_WARN, "  %-28s %#010x\n", regs[i].name, RREG32(regs[i].reg));
	for (unsigned i = 0; i < adev->num_rings; i++) {
		struct amdgpu_ring *ring = adev->rings[i];
		u64 pte = adev->gart.ptr ? readq((u8 *)adev->gart.ptr + (ring->gpu_addr >> 12) * 8) : 0;

		lkpi_log(LKPI_WARN, "  ring %-10s gpu %#llx wptr %llu rptr(wb) %u fence(wb) %u ready %d GART PTE %#llx\n",
			 ring->name, (unsigned long long)ring->gpu_addr, (unsigned long long)ring->wptr,
			 ring->rptr_cpu_addr ? *ring->rptr_cpu_addr : 0, ring->fence_cpu_addr ? *ring->fence_cpu_addr : 0,
			 ring->sched.ready, (unsigned long long)pte);
		lkpi_log(LKPI_WARN, "    %08x %08x %08x %08x %08x %08x %08x %08x\n", ring->ring[0], ring->ring[1],
			 ring->ring[2], ring->ring[3], ring->ring[4], ring->ring[5], ring->ring[6], ring->ring[7]);
	}
}

/*
 * The register part of gfx_v6_0_soft_reset (Linux 7.3 gfx_v6_0.c), run before GFX init: the CP and the graphics
 * pipeline start from reset rather than from whatever the previous driver left when it halted them (a halted
 * CP keeps its constant/draw engine state across a microcode reload, and its ME then stalls on the first packets).
 */
static void gpud_gfx_soft_reset(struct amdgpu_device *adev)
{
	u32 grbm = REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_CP, 1) | REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_CB, 1) |
		   REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_RLC, 1) | REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_DB, 1) |
		   REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_GDS, 1) | REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_PA, 1) |
		   REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_SC, 1) | REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_BCI, 1) |
		   REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_SPI, 1) | REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_SX, 1) |
		   REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_TC, 1) | REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_TA, 1) |
		   REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_VGT, 1) | REG_SET_FIELD(0, GRBM_SOFT_RESET, SOFT_RESET_IA, 1);
	u32 srbm = REG_SET_FIELD(0, SRBM_SOFT_RESET, SOFT_RESET_GRBM, 1) | REG_SET_FIELD(0, SRBM_SOFT_RESET, SOFT_RESET_SEM, 1);
	u32 tmp;

	/* Halt the CP first, as gfx_v6_0_cp_gfx_enable(false) does. */
	WREG32(mmCP_ME_CNTL, CP_ME_CNTL__ME_HALT_MASK | CP_ME_CNTL__PFP_HALT_MASK | CP_ME_CNTL__CE_HALT_MASK);
	udelay(50);
	tmp = RREG32(mmGMCON_DEBUG);
	tmp = REG_SET_FIELD(tmp, GMCON_DEBUG, GFX_STALL, 1);
	tmp = REG_SET_FIELD(tmp, GMCON_DEBUG, GFX_CLEAR, 1);
	WREG32(mmGMCON_DEBUG, tmp);
	udelay(100);

	tmp = RREG32(mmGRBM_SOFT_RESET) | grbm;
	WREG32(mmGRBM_SOFT_RESET, tmp);
	tmp = RREG32(mmGRBM_SOFT_RESET);
	udelay(100);
	WREG32(mmGRBM_SOFT_RESET, tmp & ~grbm);
	RREG32(mmGRBM_SOFT_RESET);
	udelay(100);

	tmp = RREG32(mmSRBM_SOFT_RESET) | srbm;
	WREG32(mmSRBM_SOFT_RESET, tmp);
	tmp = RREG32(mmSRBM_SOFT_RESET);
	udelay(100);
	WREG32(mmSRBM_SOFT_RESET, tmp & ~srbm);
	RREG32(mmSRBM_SOFT_RESET);
	udelay(100);

	tmp = RREG32(mmGMCON_DEBUG);
	tmp = REG_SET_FIELD(tmp, GMCON_DEBUG, GFX_STALL, 0);
	tmp = REG_SET_FIELD(tmp, GMCON_DEBUG, GFX_CLEAR, 0);
	WREG32(mmGMCON_DEBUG, tmp);
	udelay(100);
	dev_info(adev->dev, "GFX soft reset: GRBM_STATUS %#x, CP_STAT %#x\n", RREG32(mmGRBM_STATUS), RREG32(mmCP_STAT));
}

static bool gpud_runs(const struct amdgpu_ip_block *b)
{
	switch (b->version->type) {
	case AMD_IP_BLOCK_TYPE_COMMON:
	case AMD_IP_BLOCK_TYPE_GMC:
	case AMD_IP_BLOCK_TYPE_IH:
	case AMD_IP_BLOCK_TYPE_GFX:
	case AMD_IP_BLOCK_TYPE_SDMA:
		return true;
	default:
		return false;
	}
}

extern int gpud_mem_scratch_init(struct amdgpu_device *adev);
extern int gpud_ib_pool_init(struct amdgpu_device *adev);
void gpud_vram_init(struct amdgpu_device *adev, u64 start);

/* Clients get VRAM above 64 MiB (below it: the VBIOS and gpud's kernel objects); host memory up to 32 GiB. */
u64 gpud_client_vram_start = 64ULL << 20;
u64 gpud_gtt_size = 32ULL << 30;

static int hw_init_block(struct amdgpu_ip_block *b)
{
	int r;

	/* What the CPU wrote into VRAM through BAR0 (page tables, RLC buffers) reaches the GPU's view first. */
	amdgpu_device_flush_hdp(b->adev, NULL);
	r = b->version->funcs->hw_init ? b->version->funcs->hw_init(b) : 0;

	if (r)
		dev_err(b->adev->dev, "hw_init of %s failed: %d\n", b->version->funcs->name, r);
	else
		b->status.hw = true;
	return r;
}

/* amdgpu_device_ip_early_init + amdgpu_device_ip_init + amdgpu_device_ip_late_init for the blocks gpud runs. */
int gpud_device_init(struct amdgpu_device *adev)
{
	int r, i, failed = 0;

	r = si_set_ip_blocks(adev);
	if (r)
		return r;
	for (i = 0; i < adev->num_ip_blocks; i++) {
		struct amdgpu_ip_block *b = &adev->ip_blocks[i];

		if (!gpud_runs(b))
			continue;
		r = b->version->funcs->early_init ? b->version->funcs->early_init(b) : 0;
		if (r == -ENOENT)
			continue;
		if (r) {
			dev_err(adev->dev, "early_init of %s failed: %d\n", b->version->funcs->name, r);
			return r;
		}
		b->status.valid = true;
	}

	for (i = 0; i < adev->num_ip_blocks; i++) {
		struct amdgpu_ip_block *b = &adev->ip_blocks[i];

		if (!b->status.valid)
			continue;
		r = b->version->funcs->sw_init ? b->version->funcs->sw_init(b) : 0;
		if (r) {
			dev_err(adev->dev, "sw_init of %s failed: %d\n", b->version->funcs->name, r);
			return r;
		}
		b->status.sw = true;
		if (b->version->type == AMD_IP_BLOCK_TYPE_COMMON) {
			r = hw_init_block(b);
			if (r)
				return r;
		} else if (b->version->type == AMD_IP_BLOCK_TYPE_GMC) {
			r = gpud_mem_scratch_init(adev);
			if (r)
				return r;
			r = hw_init_block(b);
			if (r)
				return r;
			r = amdgpu_wb_init(adev);
			if (r)
				return r;
		}
	}

	r = gpud_ib_pool_init(adev);
	if (r)
		return r;
	gpud_gfx_soft_reset(adev);

	/* phase 1: IH; phase 2: everything else */
	for (int phase = 1; phase <= 2; phase++) {
		for (i = 0; i < adev->num_ip_blocks; i++) {
			struct amdgpu_ip_block *b = &adev->ip_blocks[i];

			if (!b->status.sw || b->status.hw)
				continue;
			if ((phase == 1) != (b->version->type == AMD_IP_BLOCK_TYPE_IH))
				continue;
			r = hw_init_block(b);
			if (r && phase == 1)
				return r;
			/* Keep going past a failed engine so the others can be brought up and diagnosed. */
			if (r) {
				gpud_debug_dump(adev);
				failed = r;
			}
		}
	}

	for (i = 0; i < adev->num_ip_blocks; i++) {
		struct amdgpu_ip_block *b = &adev->ip_blocks[i];

		if (!b->status.hw || !b->version->funcs->late_init)
			continue;
		r = b->version->funcs->late_init(b);
		if (r) {
			dev_err(adev->dev, "late_init of %s failed: %d\n", b->version->funcs->name, r);
			return r;
		}
		b->status.late_initialized = true;
	}
	gpud_vram_init(adev, gpud_client_vram_start);
	adev->accel_working = !failed;
	return failed;
}

void gpud_device_fini(struct amdgpu_device *adev)
{
	for (int i = adev->num_ip_blocks - 1; i >= 0; i--) {
		struct amdgpu_ip_block *b = &adev->ip_blocks[i];

		if (b->status.hw && b->version->funcs->hw_fini)
			b->version->funcs->hw_fini(b);
		b->status.hw = false;
	}
}

static struct pci_bus gpud_bus;

struct amdgpu_device *gpud_device_create(const struct gpud_platform *platform, const struct gpud_device_info *info)
{
	struct amdgpu_device *adev = calloc(1, sizeof(*adev));
	struct pci_dev *pdev = calloc(1, sizeof(*pdev));

	if (!adev || !pdev)
		return NULL;
	lkpi_platform = platform;
	if (info->firmware_dir)
		lkpi_firmware_dir = info->firmware_dir;
	gpud_bus.number = info->bus;
	pdev->bus = &gpud_bus;
	pdev->devfn = info->devfn;
	pdev->vendor = info->vendor;
	pdev->device = info->device;
	pdev->subsystem_vendor = info->subsystem_vendor;
	pdev->subsystem_device = info->subsystem_device;
	pdev->revision = info->revision;
	pdev->platform = platform;
	pdev->bar_start[0] = info->vram_bar_addr;
	pdev->bar_len[0] = info->vram_size;
	pdev->dev.name = "gpud";

	adev->pdev = pdev;
	adev->dev = &pdev->dev;
	adev->ddev.dev = &pdev->dev;
	adev->rmmio = info->mmio;
	adev->rmmio_size = info->mmio_size;
	adev->asic_type = CHIP_TAHITI;
	adev->family = AMDGPU_FAMILY_SI;
	adev->usec_timeout = AMDGPU_MAX_USEC_TIMEOUT;
	adev->gfx_timeout = adev->compute_timeout = adev->sdma_timeout = 2000;
	/*
	 * amdgpu reads these from the VBIOS (atombios FirmwareInfo, amdgpu_device_get_pcie_info); gpud doesn't parse
	 * the VBIOS yet, so they are amdgpu's values for this card (results/driver/gpud-*: drminfo on the native
	 * D700): a 27 MHz reference clock, 850 MHz engine and 1370 MHz memory clocks at the top DPM level, PCIe gen 3
	 * x16 for chip and platform.
	 */
	adev->clock.spll.reference_freq = 2700;
	adev->clock.default_sclk = 85000;
	adev->clock.default_mclk = 137000;
	adev->pm.pcie_gen_mask = CAIL_PCIE_LINK_SPEED_SUPPORT_GEN1 | CAIL_PCIE_LINK_SPEED_SUPPORT_GEN2 |
				 CAIL_PCIE_LINK_SPEED_SUPPORT_GEN3 | CAIL_ASIC_PCIE_LINK_SPEED_SUPPORT_GEN1 |
				 CAIL_ASIC_PCIE_LINK_SPEED_SUPPORT_GEN2 | CAIL_ASIC_PCIE_LINK_SPEED_SUPPORT_GEN3;
	adev->pm.pcie_mlw_mask = CAIL_PCIE_LINK_WIDTH_SUPPORT_X16 | CAIL_ASIC_PCIE_LINK_WIDTH_SUPPORT_X16;
	adev->accel_working = false;
	mutex_init(&adev->srbm_mutex);
	mutex_init(&adev->grbm_idx_mutex);
	mutex_init(&adev->gfx.gpu_clock_mutex);
	spin_lock_init(&adev->mmio_idx_lock);
	spin_lock_init(&adev->reg.smc.lock);
	spin_lock_init(&adev->reg.pcie.lock);
	spin_lock_init(&adev->reg.uvd_ctx.lock);
	spin_lock_init(&adev->irq.lock);
	spin_lock_init(&adev->wb.lock);
	adev->gpud = (void *)info;
	extern int gpud_mem_create(struct amdgpu_device *adev, const struct gpud_device_info *info);
	if (gpud_mem_create(adev, info)) {
		free(pdev);
		free(adev);
		return NULL;
	}
	return adev;
}

int gpud_ring_count(struct amdgpu_device *adev) { return adev->num_rings; }
const char *gpud_ring_name(struct amdgpu_device *adev, int ring) { return adev->rings[ring]->name; }
int gpud_ring_ready(struct amdgpu_device *adev, int ring) { return adev->rings[ring]->sched.ready; }
int gpud_ring_type(struct amdgpu_device *adev, int ring) { return adev->rings[ring]->funcs->type; }
