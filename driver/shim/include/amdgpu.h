/* SPDX-License-Identifier: MIT */
/*
 * gpud's amdgpu.h: the device structure and core API that Linux's Southern Islands files (driver/linux, unmodified)
 * compile against. Linux's own amdgpu.h embeds about eighty subsystems; this one keeps the members the SI hardware
 * files use, with the same names and types. The "funcs" tables are copied verbatim from Linux 7.3-rc7
 * (amdgpu.h, amdgpu_gfx.h, amdgpu_gmc.h, amdgpu_vm.h, amdgpu_sdma.h, amdgpu_mode.h; Copyright Advanced Micro
 * Devices, Inc., MIT license) so the files' designated initializers match.
 */
#ifndef GPUD_AMDGPU_H
#define GPUD_AMDGPU_H

#include <lkpi.h>
#include <lkpi_drm.h>
#include <drm/amdgpu_drm.h>

#include "amd_shared.h"
#include "amdgpu_ip.h"
#include "amdgpu_reg_access.h"
#include "amdgpu_ring.h"
#include "amdgpu_ih.h"
#include "amdgpu_irq.h"
#include "amdgpu_ucode.h"
#include "amdgpu_gart.h"
#include "amdgpu_wb.h"
#include "clearstate_defs.h"
#include "amdgpu_rlc.h"

/* ---- module parameters (amdgpu_drv.c), at their defaults ---- */
extern int amdgpu_pcie_gen2;
extern int amdgpu_reset_method;
extern int amdgpu_emu_mode;
extern int amdgpu_vm_fault_stop;
extern int amdgpu_dpm;
extern int amdgpu_sched_hw_submission;
extern int amdgpu_vm_size;
extern int amdgpu_vm_block_size;
extern int amdgpu_gart_size;
extern int amdgpu_gtt_size;
extern uint amdgpu_cg_mask;
extern uint amdgpu_pg_mask;
extern int amdgpu_disable_cu_count;
extern char *amdgpu_disable_cu;
extern int amdgpu_aspm;
extern int amdgpu_msi;
extern int amdgpu_num_kcq;
extern int amdgpu_vm_update_mode;
extern int amdgpu_dc;
extern int amdgpu_lockup_timeout;

#define AMDGPU_VM_FAULT_STOP_NEVER 0
#define AMDGPU_VM_FAULT_STOP_FIRST 1
#define AMDGPU_VM_FAULT_STOP_ALWAYS 2
#define AMDGPU_MAX_USEC_TIMEOUT 100000
#define AMDGPU_FENCE_JIFFIES_TIMEOUT (HZ / 2)
#define AMDGPU_MAX_PPLL 3
#define AMDGPU_MAX_VMHUBS 13
#define AMDGPU_GFXHUB(x) (x)
#define AMDGPU_MMHUB0(x) (8 + (x))
#define AMDGPU_MMHUB1(x) (8 + 4 + (x))
#define AMDGPU_GFX_MAX_SE 4
#define AMDGPU_GFX_MAX_SH_PER_SE 2
#define AMDGPU_MAX_GC_INSTANCES 8
#define AMDGPU_MAX_SDMA_INSTANCES 16
#define AMDGPU_MAX_GFX_QUEUES 8
#define AMDGPU_MAX_COMPUTE_QUEUES 8
#define AMDGPU_IB_POOL_SIZE (1024 * 1024)
#define AMDGPU_RESET_MAGIC_NUM 64
#define AMDGPU_REGS_NO_KIQ (1 << 1)
#define AMDGPU_REGS_RLC (1 << 2)
#define AMDGPU_VA_RESERVED_BOTTOM (1ULL << 16)
#define AMD_IS_MOBILITY (1UL << 16)
#define AMD_IS_APU (1UL << 17)
#define AMDGPU_HAVE_CTX_SWITCH (1 << 0)

/* ---- enums and limits from Linux's amdgpu.h, amdgpu_sdma.h, amdgpu_gmc.h, amdgpu_vm.h, amdgpu_reset.h ---- */
enum amdgpu_cp_irq {
	AMDGPU_CP_IRQ_GFX_ME0_PIPE0_EOP = 0,
	AMDGPU_CP_IRQ_GFX_ME0_PIPE1_EOP,
	AMDGPU_CP_IRQ_COMPUTE_MEC1_PIPE0_EOP,
	AMDGPU_CP_IRQ_COMPUTE_MEC1_PIPE1_EOP,
	AMDGPU_CP_IRQ_COMPUTE_MEC1_PIPE2_EOP,
	AMDGPU_CP_IRQ_COMPUTE_MEC1_PIPE3_EOP,
	AMDGPU_CP_IRQ_COMPUTE_MEC2_PIPE0_EOP,
	AMDGPU_CP_IRQ_COMPUTE_MEC2_PIPE1_EOP,
	AMDGPU_CP_IRQ_COMPUTE_MEC2_PIPE2_EOP,
	AMDGPU_CP_IRQ_COMPUTE_MEC2_PIPE3_EOP,
	AMDGPU_CP_IRQ_LAST
};
enum amdgpu_sdma_irq {
	AMDGPU_SDMA_IRQ_INSTANCE0 = 0,
	AMDGPU_SDMA_IRQ_INSTANCE1,
	AMDGPU_SDMA_IRQ_LAST = 16
};
enum amdgpu_gart_placement {
	AMDGPU_GART_PLACEMENT_BEST_FIT = 0,
	AMDGPU_GART_PLACEMENT_HIGH,
	AMDGPU_GART_PLACEMENT_LOW,
};
enum amd_reset_method {
	AMD_RESET_METHOD_NONE = -1,
	AMD_RESET_METHOD_LEGACY = 0,
	AMD_RESET_METHOD_MODE0,
	AMD_RESET_METHOD_MODE1,
	AMD_RESET_METHOD_MODE2,
	AMD_RESET_METHOD_LINK,
	AMD_RESET_METHOD_BACO,
	AMD_RESET_METHOD_PCI,
	AMD_RESET_METHOD_ON_INIT,
};
#define AMDGPU_RESET_TYPE_FULL (1 << 0)
#define AMDGPU_RESET_TYPE_SOFT_RECOVERY (1 << 1)
#define AMDGPU_RESET_TYPE_PER_QUEUE (1 << 2)
#define AMDGPU_RESET_TYPE_PER_PIPE (1 << 3)
#define AMDGPU_RESET_TYPE_IP_BLOCK_SOFT_RESET (1 << 4)
#define AMDGPU_VBIOS_VGA_ALLOCATION (9 * 1024 * 1024)
#define AMDGPU_VA_RESERVED_CSA_SIZE (2ULL << 20)
#define AMDGPU_VA_RESERVED_SEQ64_SIZE (2ULL << 20)
#define AMDGPU_VA_RESERVED_TRAP_SIZE (1ULL << 16)
#define AMDGPU_VA_RESERVED_TOP (AMDGPU_VA_RESERVED_TRAP_SIZE + AMDGPU_VA_RESERVED_SEQ64_SIZE + AMDGPU_VA_RESERVED_CSA_SIZE)
/* No SR-IOV: gpud owns the physical function. */
#define amdgpu_sriov_vf(adev) false
#define amdgpu_sriov_runtime(adev) false
#define amdgpu_sriov_is_pp_one_vf(adev) false
#define amdgpu_reset_domain_schedule(domain, work) false
void amdgpu_vmid_reset_all(struct amdgpu_device *adev);

/* ---- buffer objects: gpud's kernel allocations, never moved ---- */
struct amdgpu_bo {
	u32 domain;
	u64 size;
	u64 gpu_addr;  /* MC address */
	void *cpu_ptr; /* CPU mapping (VRAM through BAR0, host memory directly), or NULL */
	u64 vram_offset; /* VRAM: offset from the start of VRAM */
	void *host;    /* GTT: the platform's DMA buffer */
	u64 iova;      /* GTT: device address of the pages */
};
int amdgpu_bo_create_kernel(struct amdgpu_device *adev, unsigned long size, int align, u32 domain,
			    struct amdgpu_bo **bo_ptr, u64 *gpu_addr, void **cpu_addr);
int amdgpu_bo_create_reserved(struct amdgpu_device *adev, unsigned long size, int align, u32 domain,
			      struct amdgpu_bo **bo_ptr, u64 *gpu_addr, void **cpu_addr);
void amdgpu_bo_free_kernel(struct amdgpu_bo **bo, u64 *gpu_addr, void **cpu_addr);
static inline void amdgpu_bo_kunmap(struct amdgpu_bo *bo) { (void)bo; }
static inline void amdgpu_bo_unreserve(struct amdgpu_bo *bo) { (void)bo; }
static inline int amdgpu_bo_reserve(struct amdgpu_bo *bo, bool no_intr) { (void)bo; (void)no_intr; return 0; }
static inline int amdgpu_bo_kmap(struct amdgpu_bo *bo, void **ptr) { if (ptr) *ptr = bo->cpu_ptr; return 0; }
u64 amdgpu_bo_gpu_offset(struct amdgpu_bo *bo);
int amdgpu_bo_init(struct amdgpu_device *adev);
void amdgpu_bo_fini(struct amdgpu_device *adev);

/* ---- jobs: only the VMID reaches the ring functions ---- */
struct amdgpu_vm;
struct amdgpu_job {
	unsigned vmid;
	u64 vm_pd_addr;
};
#define AMDGPU_JOB_GET_VMID(job) ((job) ? (job)->vmid : 0)
void amdgpu_vm_flush(struct amdgpu_ring *ring, struct amdgpu_job *job, bool *need_pipe_sync, bool *emit_spm,
		     bool *emit_gds);

/* ---- GFX (amdgpu_gfx.h, trimmed) ---- */
struct amdgpu_rb_config {
	uint32_t rb_backend_disable;
	uint32_t user_rb_backend_disable;
	uint32_t raster_config;
	uint32_t raster_config_1;
};
struct gb_addr_config {
	uint16_t pipe_interleave_size;
	uint8_t num_pipes;
	uint8_t max_compress_frags;
	uint8_t num_banks;
	uint8_t num_se;
	uint8_t num_rb_per_se;
	uint8_t num_pkrs;
};
struct amdgpu_gfx_config {
	unsigned max_shader_engines;
	unsigned max_tile_pipes;
	unsigned max_cu_per_sh;
	unsigned max_sh_per_se;
	unsigned max_backends_per_se;
	unsigned max_texture_channel_caches;
	unsigned max_gprs;
	unsigned max_gs_threads;
	unsigned max_hw_contexts;
	unsigned sc_prim_fifo_size_frontend;
	unsigned sc_prim_fifo_size_backend;
	unsigned sc_hiz_tile_fifo_size;
	unsigned sc_earlyz_tile_fifo_size;
	unsigned num_tile_pipes;
	unsigned backend_enable_mask;
	unsigned mem_max_burst_length_bytes;
	unsigned mem_row_size_in_kb;
	unsigned shader_engine_tile_size;
	unsigned num_gpus;
	unsigned multi_gpu_tile_size;
	unsigned mc_arb_ramcfg;
	unsigned num_banks;
	unsigned num_ranks;
	unsigned gb_addr_config;
	unsigned num_rbs;
	unsigned gs_vgt_table_depth;
	unsigned gs_prim_buffer_depth;
	uint32_t tile_mode_array[32];
	uint32_t macrotile_mode_array[16];
	struct gb_addr_config gb_addr_config_fields;
	struct amdgpu_rb_config rb_config[AMDGPU_GFX_MAX_SE][AMDGPU_GFX_MAX_SH_PER_SE];
	uint32_t double_offchip_lds_buf;
	uint64_t tcc_disabled_mask;
};
struct amdgpu_cu_info {
	uint32_t simd_per_cu;
	uint32_t max_waves_per_simd;
	uint32_t wave_front_size;
	uint32_t max_scratch_slots_per_cu;
	uint32_t lds_size;
	uint32_t number;
	uint32_t ao_cu_mask;
	uint32_t ao_cu_bitmap[4][4];
	uint32_t bitmap[AMDGPU_MAX_GC_INSTANCES][4][4];
};
struct amdgpu_gfx_shadow_info;
enum amdgpu_gfx_partition { AMDGPU_GFX_PARTITION_DUMMY };
struct amdgpu_gfx_funcs {
	uint64_t (*get_gpu_clock_counter)(struct amdgpu_device *adev);
	void (*select_se_sh)(struct amdgpu_device *adev, u32 se_num, u32 sh_num, u32 instance, int xcc_id);
	void (*read_wave_data)(struct amdgpu_device *adev, uint32_t xcc_id, uint32_t simd, uint32_t wave,
			       uint32_t *dst, int *no_fields);
	void (*read_wave_vgprs)(struct amdgpu_device *adev, uint32_t xcc_id, uint32_t simd, uint32_t wave,
				uint32_t thread, uint32_t start, uint32_t size, uint32_t *dst);
	void (*read_wave_sgprs)(struct amdgpu_device *adev, uint32_t xcc_id, uint32_t simd, uint32_t wave,
				uint32_t start, uint32_t size, uint32_t *dst);
	void (*select_me_pipe_q)(struct amdgpu_device *adev, u32 me, u32 pipe, u32 queue, u32 vmid, u32 xcc_id);
	void (*init_spm_golden)(struct amdgpu_device *adev);
	void (*update_perfmon_mgcg)(struct amdgpu_device *adev, bool enable);
	int (*get_gfx_shadow_info)(struct amdgpu_device *adev, struct amdgpu_gfx_shadow_info *shadow_info,
				   bool skip_check);
	enum amdgpu_gfx_partition (*query_partition_mode)(struct amdgpu_device *adev);
	int (*switch_partition_mode)(struct amdgpu_device *adev, int num_xccs_per_xcp);
	int (*ih_node_to_logical_xcc)(struct amdgpu_device *adev, int ih_node);
	int (*get_xccs_per_xcp)(struct amdgpu_device *adev);
	void (*get_hdp_flush_mask)(struct amdgpu_ring *ring, uint32_t *ref_and_mask, uint32_t *reg_mem_engine);
};
struct amdgpu_gfx {
	struct mutex gpu_clock_mutex;
	struct amdgpu_gfx_config config;
	struct amdgpu_rlc rlc;
	const struct firmware *me_fw;
	uint32_t me_fw_version;
	const struct firmware *pfp_fw;
	uint32_t pfp_fw_version;
	const struct firmware *ce_fw;
	uint32_t ce_fw_version;
	const struct firmware *rlc_fw;
	uint32_t rlc_fw_version;
	const struct firmware *mec_fw;
	const struct firmware *mec2_fw;
	uint32_t me_feature_version;
	uint32_t ce_feature_version;
	uint32_t pfp_feature_version;
	uint32_t rlc_feature_version;
	unsigned ce_ram_size;
	struct amdgpu_cu_info cu_info;
	const struct amdgpu_gfx_funcs *funcs;
	struct amdgpu_ring gfx_ring[AMDGPU_MAX_GFX_RINGS];
	unsigned num_gfx_rings;
	struct amdgpu_ring compute_ring[AMDGPU_MAX_COMPUTE_RINGS];
	unsigned num_compute_rings;
	struct amdgpu_irq_src eop_irq;
	struct amdgpu_irq_src priv_reg_irq;
	struct amdgpu_irq_src priv_inst_irq;
	uint32_t gfx_supported_reset;
	uint32_t compute_supported_reset;
	uint16_t xcc_mask;
};
#define amdgpu_gfx_get_gpu_clock_counter(adev) (adev)->gfx.funcs->get_gpu_clock_counter((adev))
#define amdgpu_gfx_select_se_sh(adev, se, sh, instance, xcc_id) \
	((adev)->gfx.funcs->select_se_sh((adev), (se), (sh), (instance), (xcc_id)))
u32 amdgpu_gfx_create_bitmask(u32 bit_width);
void amdgpu_gfx_parse_disable_cu(struct amdgpu_device *adev, unsigned int *mask, unsigned int max_se,
				 unsigned int max_sh);
int amdgpu_gfx_get_num_kcq(struct amdgpu_device *adev);
u32 amdgpu_gfx_csb_preamble_start(u32 *buffer);
u32 amdgpu_gfx_csb_data_parser(struct amdgpu_device *adev, u32 *buffer, u32 count);
void amdgpu_gfx_csb_preamble_end(u32 *buffer, u32 count);

/* ---- GMC (amdgpu_gmc.h, trimmed) ---- */
enum amdgpu_memory_partition { AMDGPU_MEMORY_PARTITION_DUMMY };
struct amdgpu_gmc_funcs {
	void (*flush_gpu_tlb)(struct amdgpu_device *adev, uint32_t vmid, uint32_t vmhub, uint32_t flush_type);
	void (*flush_gpu_tlb_pasid)(struct amdgpu_device *adev, uint16_t pasid, uint32_t flush_type, bool all_hub,
				    uint32_t inst);
	uint64_t (*emit_flush_gpu_tlb)(struct amdgpu_ring *ring, unsigned vmid, uint64_t pd_addr);
	void (*emit_pasid_mapping)(struct amdgpu_ring *ring, unsigned vmid, unsigned pasid);
	void (*set_prt)(struct amdgpu_device *adev, bool enable);
	void (*get_vm_pde)(struct amdgpu_device *adev, int level, u64 *dst, u64 *flags);
	void (*get_vm_pte)(struct amdgpu_device *adev, struct amdgpu_vm *vm, struct amdgpu_bo *bo, uint32_t vm_flags,
			   uint64_t *pte_flags);
	void (*override_vm_pte_flags)(struct amdgpu_device *dev, struct amdgpu_vm *vm, uint64_t addr,
				      uint64_t *flags);
	unsigned int (*get_vbios_fb_size)(struct amdgpu_device *adev);
	unsigned int (*get_dcc_alignment)(struct amdgpu_device *adev);
	enum amdgpu_memory_partition (*query_mem_partition_mode)(struct amdgpu_device *adev);
	int (*request_mem_partition_mode)(struct amdgpu_device *adev, int nps_mode);
	bool (*need_reset_on_init)(struct amdgpu_device *adev);
};
struct amdgpu_gmc {
	resource_size_t aper_size;
	resource_size_t aper_base;
	u64 mc_vram_size;
	u64 visible_vram_size;
	u64 agp_size;
	u64 agp_start;
	u64 agp_end;
	u64 gart_size;
	u64 gart_start;
	u64 gart_end;
	u64 vram_start;
	u64 vram_end;
	u64 fb_start;
	u64 fb_end;
	unsigned vram_width;
	u64 real_vram_size;
	int vram_mtrr;
	u64 mc_mask;
	const struct firmware *fw;
	uint32_t fw_version;
	struct amdgpu_irq_src vm_fault;
	uint32_t vram_type;
	uint8_t vram_vendor;
	uint32_t srbm_soft_reset;
	bool prt_warning;
	uint32_t sdpif_register;
	u64 shared_aperture_start;
	u64 shared_aperture_end;
	u64 private_aperture_start;
	u64 private_aperture_end;
	const struct amdgpu_gmc_funcs *gmc_funcs;
	u64 pte_addr_mask;
};
#define amdgpu_gmc_emit_flush_gpu_tlb(r, vmid, addr) (r)->adev->gmc.gmc_funcs->emit_flush_gpu_tlb((r), (vmid), (addr))
#define amdgpu_gmc_get_vm_pde(adev, level, dst, flags) (adev)->gmc.gmc_funcs->get_vm_pde((adev), (level), (dst), (flags))
#define amdgpu_gmc_get_vm_pte(adev, vm, bo, vm_flags, pte_flags) \
	(adev)->gmc.gmc_funcs->get_vm_pte((adev), (vm), (bo), (vm_flags), (pte_flags))
void amdgpu_gmc_vram_location(struct amdgpu_device *adev, struct amdgpu_gmc *mc, u64 base);
void amdgpu_gmc_gart_location(struct amdgpu_device *adev, struct amdgpu_gmc *mc,
			      enum amdgpu_gart_placement gart_placement);
void amdgpu_gmc_set_agp_default(struct amdgpu_device *adev, struct amdgpu_gmc *mc);
void amdgpu_gmc_set_gart_size(struct amdgpu_device *adev, u64 default_size);
int amdgpu_gmc_vram_checking(struct amdgpu_device *adev);

/* ---- VM (amdgpu_vm.h, trimmed) ---- */
#define AMDGPU_VM_PTE_COUNT(adev) (1 << (adev)->vm_manager.block_size)
#define AMDGPU_PTE_VALID (1ULL << 0)
#define AMDGPU_PTE_SYSTEM (1ULL << 1)
#define AMDGPU_PTE_SNOOPED (1ULL << 2)
#define AMDGPU_PTE_EXECUTABLE (1ULL << 4)
#define AMDGPU_PTE_READABLE (1ULL << 5)
#define AMDGPU_PTE_WRITEABLE (1ULL << 6)
#define AMDGPU_PTE_FRAG(x) ((x & 0x1fULL) << 7)
#define AMDGPU_PTE_PRT (1ULL << 51)
#define AMDGPU_VM_PDB0 0
#define AMDGPU_VM_PDB1 1
#define AMDGPU_VM_PDB2 2
#define AMDGPU_VM_PTB 3
#define AMDGPU_NUM_VMID 16
#define AMDGPU_VM_RESERVED_VRAM (8ULL << 20)
struct amdgpu_vm_pte_funcs {
	unsigned copy_pte_num_dw;
	void (*copy_pte)(struct amdgpu_ib *ib, uint64_t pe, uint64_t src, unsigned count);
	void (*write_pte)(struct amdgpu_ib *ib, uint64_t pe, uint64_t value, unsigned count, uint32_t incr);
	void (*set_pte_pde)(struct amdgpu_ib *ib, uint64_t pe, uint64_t addr, unsigned count, uint32_t incr,
			    uint64_t flags);
};
struct amdgpu_vm_manager {
	unsigned seqno[AMDGPU_MAX_RINGS];
	uint64_t max_pfn;
	uint32_t num_level;
	uint32_t block_size;
	uint32_t fragment_size;
	enum amdgpu_vm_level { AMDGPU_VM_LEVEL_DUMMY } root_level;
	u64 vram_base_offset;
	unsigned first_kfd_vmid;
	u64 saved_table_addr[AMDGPU_NUM_VMID];
	const struct amdgpu_vm_pte_funcs *vm_pte_funcs;
	struct drm_gpu_scheduler *vm_pte_scheds[AMDGPU_MAX_RINGS];
	unsigned vm_pte_num_scheds;
};
void amdgpu_vm_manager_init(struct amdgpu_device *adev);
void amdgpu_vm_manager_fini(struct amdgpu_device *adev);
void amdgpu_vm_adjust_size(struct amdgpu_device *adev, uint32_t min_vm_size, uint32_t fragment_size_default,
			   unsigned max_level, unsigned max_bits);
void amdgpu_vm_update_fault_cache(struct amdgpu_device *adev, unsigned int pasid, uint64_t addr, uint32_t status,
				  unsigned int vmhub);
void amdgpu_sdma_set_vm_pte_scheds(struct amdgpu_device *adev, const struct amdgpu_vm_pte_funcs *vm_pte_funcs);

/* ---- SDMA (amdgpu_sdma.h, trimmed) ---- */
struct amdgpu_sdma_instance {
	const struct firmware *fw;
	uint32_t fw_version;
	uint32_t feature_version;
	struct amdgpu_ring ring;
	struct amdgpu_ring page;
	bool burst_nop;
};
struct amdgpu_sdma {
	struct amdgpu_sdma_instance instance[AMDGPU_MAX_SDMA_INSTANCES];
	struct amdgpu_irq_src trap_irq;
	struct amdgpu_irq_src illegal_inst_irq;
	int num_instances;
	uint32_t srbm_soft_reset;
	bool has_page_queue;
};
struct amdgpu_buffer_funcs {
	uint32_t copy_max_bytes;
	unsigned copy_num_dw;
	void (*emit_copy_buffer)(struct amdgpu_ib *ib, uint64_t src_offset, uint64_t dst_offset, uint32_t byte_count,
				 uint32_t copy_flags);
	uint32_t fill_max_bytes;
	unsigned fill_num_dw;
	void (*emit_fill_buffer)(struct amdgpu_ib *ib, uint32_t src_data, uint64_t dst_offset, uint32_t byte_count);
};
void amdgpu_sdma_set_buffer_funcs_scheds(struct amdgpu_device *adev, const struct amdgpu_buffer_funcs *buffer_funcs);

/* ---- clocks, modes, power (amdgpu.h, amdgpu_mode.h, trimmed) ---- */
struct amdgpu_pll {
	uint32_t reference_freq;
	uint32_t reference_div;
	uint32_t post_div;
	uint32_t pll_in_min;
	uint32_t pll_in_max;
	uint32_t pll_out_min;
	uint32_t pll_out_max;
	uint32_t lcd_pll_out_min;
	uint32_t lcd_pll_out_max;
	uint32_t best_vco;
	uint32_t min_ref_div;
	uint32_t max_ref_div;
	uint32_t min_post_div;
	uint32_t max_post_div;
	uint32_t min_feedback_div;
	uint32_t max_feedback_div;
	uint32_t min_frac_feedback_div;
	uint32_t max_frac_feedback_div;
	uint32_t flags;
	uint32_t id;
};
struct amdgpu_clock {
	struct amdgpu_pll ppll[AMDGPU_MAX_PPLL];
	struct amdgpu_pll spll;
	struct amdgpu_pll mpll;
	uint32_t default_mclk;
	uint32_t default_sclk;
	uint32_t default_dispclk;
	uint32_t dp_extclk;
	uint32_t max_pixel_clock;
};
struct atom_context;
struct card_info;
struct amdgpu_mode_info {
	struct atom_context *atom_context;
	struct card_info *atom_card_info;
	bool mode_config_initialized;
	int num_crtc;
	int num_hpd;
	int num_dig;
	/* firmware flags */
	u32 firmware_flags;
};
#define AMDGPU_MAX_I2C_BUS 16
struct amdgpu_i2c_chan;
#include "amdgpu_mode.h"
#include <drm/drm_fixed.h>

/* Power management: Linux's own headers (kgd_pp_interface.h, dm_pp_interface.h, pm/inc/amdgpu_dpm.h). */
#include "kgd_pp_interface.h"
#include "dm_pp_interface.h"
#include "amdgpu_dpm.h"
struct amd_powerplay {
	void *pp_handle;
	const struct amd_pm_funcs *pp_funcs;
};
enum amdgpu_thermal_irq {
	AMDGPU_THERMAL_IRQ_LOW_TO_HIGH = 0,
	AMDGPU_THERMAL_IRQ_HIGH_TO_LOW,

	AMDGPU_THERMAL_IRQ_LAST
};
struct amdgpu_mman {
	struct { int dummy; } gtt_mgr;
	void *gtt_mgr_priv; /* gpud: kernel memory allocator state */
	bool buffer_funcs_enabled;
	const struct amdgpu_buffer_funcs *buffer_funcs;
};
struct amdgpu_mem_scratch {
	struct amdgpu_bo *robj;
	uint32_t *ptr;
	u64 gpu_addr;
};
int amdgpu_gtt_mgr_recover(void *mgr);
void amdgpu_gem_force_release(struct amdgpu_device *adev);

/* ---- ASIC functions (amdgpu.h) ---- */
struct amdgpu_video_codecs;
enum amdgpu_reg_state { AMDGPU_REG_STATE_TYPE_INVALID = 0 };
struct amdgpu_asic_funcs {
	bool (*read_disabled_bios)(struct amdgpu_device *adev);
	bool (*read_bios_from_rom)(struct amdgpu_device *adev, u8 *bios, u32 length_bytes);
	int (*read_register)(struct amdgpu_device *adev, u32 se_num, u32 sh_num, u32 reg_offset, u32 *value);
	void (*set_vga_state)(struct amdgpu_device *adev, bool state);
	int (*reset)(struct amdgpu_device *adev);
	enum amd_reset_method (*reset_method)(struct amdgpu_device *adev);
	u32 (*get_xclk)(struct amdgpu_device *adev);
	int (*set_uvd_clocks)(struct amdgpu_device *adev, u32 vclk, u32 dclk);
	int (*set_vce_clocks)(struct amdgpu_device *adev, u32 evclk, u32 ecclk);
	int (*get_pcie_lanes)(struct amdgpu_device *adev);
	void (*set_pcie_lanes)(struct amdgpu_device *adev, int lanes);
	u32 (*get_config_memsize)(struct amdgpu_device *adev);
	void (*flush_hdp)(struct amdgpu_device *adev, struct amdgpu_ring *ring);
	void (*invalidate_hdp)(struct amdgpu_device *adev, struct amdgpu_ring *ring);
	void (*init_doorbell_index)(struct amdgpu_device *adev);
	void (*get_pcie_usage)(struct amdgpu_device *adev, uint64_t *count0, uint64_t *count1);
	bool (*need_reset_on_init)(struct amdgpu_device *adev);
	uint64_t (*get_pcie_replay_count)(struct amdgpu_device *adev);
	int (*supports_baco)(struct amdgpu_device *adev);
	void (*pre_asic_init)(struct amdgpu_device *adev);
	int (*update_umd_stable_pstate)(struct amdgpu_device *adev, bool enter);
	int (*query_video_codecs)(struct amdgpu_device *adev, bool encode, const struct amdgpu_video_codecs **codecs);
	u64 (*encode_ext_smn_addressing)(int ext_id);
	ssize_t (*get_reg_state)(struct amdgpu_device *adev, enum amdgpu_reg_state reg_state, void *buf,
				 size_t max_size);
};
#define amdgpu_asic_set_vga_state(adev, state) ((adev)->asic_funcs->set_vga_state ? (adev)->asic_funcs->set_vga_state((adev), (state)) : 0)
#define amdgpu_asic_reset(adev) (adev)->asic_funcs->reset((adev))
#define amdgpu_asic_get_xclk(adev) (adev)->asic_funcs->get_xclk((adev))
#define amdgpu_asic_get_config_memsize(adev) (adev)->asic_funcs->get_config_memsize((adev))
#define amdgpu_asic_read_register(adev, se, sh, offset, v) \
	((adev)->asic_funcs->read_register((adev), (se), (sh), (offset), (v)))
#define amdgpu_asic_flush_hdp(adev, r) \
	((adev)->asic_funcs->flush_hdp ? (adev)->asic_funcs->flush_hdp((adev), (r)) : (void)0)
#define amdgpu_asic_invalidate_hdp(adev, r) \
	((adev)->asic_funcs->invalidate_hdp ? (adev)->asic_funcs->invalidate_hdp((adev), (r)) : (void)0)
#define amdgpu_asic_need_reset_on_init(adev) (adev)->asic_funcs->need_reset_on_init((adev))
#define amdgpu_asic_set_uvd_clocks(adev, v, d) (adev)->asic_funcs->set_uvd_clocks((adev), (v), (d))
#define amdgpu_asic_set_vce_clocks(adev, ev, ec) (adev)->asic_funcs->set_vce_clocks((adev), (ev), (ec))
#define amdgpu_get_pcie_lanes(adev) (adev)->asic_funcs->get_pcie_lanes((adev))
#define amdgpu_set_pcie_lanes(adev, l) (adev)->asic_funcs->set_pcie_lanes((adev), (l))

/* ---- the device ---- */
struct amdgpu_device {
	struct device *dev;
	struct pci_dev *pdev;
	struct drm_device ddev;
	enum amd_asic_type asic_type;
	uint32_t family;
	uint32_t rev_id;
	uint32_t external_rev_id;
	unsigned long flags;
	unsigned long apu_flags;
	int usec_timeout;
	const struct amdgpu_asic_funcs *asic_funcs;
	bool shutdown;
	bool need_swiotlb;
	bool accel_working;
	struct mutex srbm_mutex;
	struct mutex grbm_idx_mutex;
	uint8_t *bios;
	uint32_t bios_size;
	uint32_t bios_scratch_reg_offset;
	bool is_atom_fw;
	bool dc_enabled;
	struct amdgpu_i2c_chan *i2c_bus[AMDGPU_MAX_I2C_BUS];
	resource_size_t rmmio_base;
	resource_size_t rmmio_size;
	void __iomem *rmmio;
	spinlock_t mmio_idx_lock;
	struct amdgpu_reg_access reg;
	struct amdgpu_clock clock;
	struct amdgpu_gmc gmc;
	struct amdgpu_gart gart;
	dma_addr_t dummy_page_addr;
	struct amdgpu_vm_manager vm_manager;
	DECLARE_BITMAP(vmhubs_mask, AMDGPU_MAX_VMHUBS);
	struct amdgpu_mman mman;
	struct amdgpu_mem_scratch mem_scratch;
	struct amdgpu_wb wb;
	bool enable_virtual_display;
	struct amdgpu_mode_info mode_info;
	unsigned num_rings;
	struct amdgpu_ring *rings[AMDGPU_MAX_RINGS];
	bool ib_pool_ready;
	struct amdgpu_irq irq;
	struct amd_powerplay powerplay;
	struct amdgpu_pm pm;
	u64 cg_flags;
	u32 pg_flags;
	struct amdgpu_gfx gfx;
	struct amdgpu_sdma sdma;
	struct amdgpu_firmware firmware;
	struct amdgpu_ip_block ip_blocks[AMDGPU_MAX_IP_NUM];
	int num_ip_blocks;
	bool has_uvd;
	bool has_hw_reset;
	bool in_suspend;
	bool debug_disable_ip_block_soft_reset;
	bool no_hw_access;
	long gfx_timeout;
	long sdma_timeout;
	long compute_timeout;
	void *reset_domain;
	struct { struct work_struct flr_work; } virt;
	/* gpud */
	void *gpud;
};
static inline struct drm_device *adev_to_drm(struct amdgpu_device *adev) { return &adev->ddev; }
static inline struct amdgpu_device *drm_to_adev(struct drm_device *ddev) { return container_of(ddev, struct amdgpu_device, ddev); }

/* ---- VRAM the firmware uses (amdgpu_ttm.h): kept out of gpud's client VRAM ---- */
enum amdgpu_resv_region_id {
	AMDGPU_RESV_STOLEN_VGA,
	AMDGPU_RESV_STOLEN_EXTENDED,
	AMDGPU_RESV_STOLEN_RESERVED,
	AMDGPU_RESV_FW,
	AMDGPU_RESV_FW_EXTEND,
	AMDGPU_RESV_FW_VRAM_USAGE,
	AMDGPU_RESV_DRV_VRAM_USAGE,
	AMDGPU_RESV_MEM_TRAIN,
	AMDGPU_RESV_MAX
};
void amdgpu_ttm_init_vram_resv(struct amdgpu_device *adev, enum amdgpu_resv_region_id id, uint64_t offset,
			       uint64_t size, bool needs_cpu_map);

/* ---- registers ---- */
#define RREG32_NO_KIQ(reg) amdgpu_device_rreg(adev, (reg), AMDGPU_REGS_NO_KIQ)
#define WREG32_NO_KIQ(reg, v) amdgpu_device_wreg(adev, (reg), (v), AMDGPU_REGS_NO_KIQ)
#define RREG32(reg) amdgpu_device_rreg(adev, (reg), 0)
#define DREG32(reg) printk("REGISTER: " #reg " : 0x%08X\n", amdgpu_device_rreg(adev, (reg), 0))
#define WREG32(reg, v) amdgpu_device_wreg(adev, (reg), (v), 0)
#define REG_SET(FIELD, v) (((v) << FIELD##_SHIFT) & FIELD##_MASK)
#define REG_GET(FIELD, v) (((v) << FIELD##_SHIFT) & FIELD##_MASK)
#define RREG32_PCIE(reg) amdgpu_reg_pcie_rd32(adev, (reg))
#define WREG32_PCIE(reg, v) amdgpu_reg_pcie_wr32(adev, (reg), (v))
#define RREG32_PCIE_PORT(reg) amdgpu_reg_pciep_rd32(adev, (reg))
#define WREG32_PCIE_PORT(reg, v) amdgpu_reg_pciep_wr32(adev, (reg), (v))
#define RREG32_SMC(reg) amdgpu_reg_smc_rd32(adev, (reg))
#define WREG32_SMC(reg, v) amdgpu_reg_smc_wr32(adev, (reg), (v))
#define RREG32_UVD_CTX(reg) amdgpu_reg_uvd_ctx_rd32(adev, (reg))
#define WREG32_UVD_CTX(reg, v) amdgpu_reg_uvd_ctx_wr32(adev, (reg), (v))
#define WREG32_P(reg, val, mask)                 \
	do {                                     \
		uint32_t tmp_ = RREG32(reg);     \
		tmp_ &= (mask);                  \
		tmp_ |= ((val) & ~(mask));       \
		WREG32(reg, tmp_);               \
	} while (0)
#define WREG32_AND(reg, and) WREG32_P(reg, 0, and)
#define WREG32_OR(reg, or) WREG32_P(reg, or, ~(or))
#define WREG32_SMC_P(_Reg, _Val, _Mask)                         \
	do {                                                    \
		u32 tmp = RREG32_SMC(_Reg);                     \
		tmp &= (_Mask);                                 \
		tmp |= ((_Val) & ~(_Mask));                     \
		WREG32_SMC(_Reg, tmp);                          \
	} while (0)
#define REG_FIELD_SHIFT(reg, field) reg##__##field##__SHIFT
#define REG_FIELD_MASK(reg, field) reg##__##field##_MASK
#define REG_SET_FIELD(orig_val, reg, field, field_val) \
	(((orig_val) & ~REG_FIELD_MASK(reg, field)) |  \
	 (REG_FIELD_MASK(reg, field) & ((field_val) << REG_FIELD_SHIFT(reg, field))))
#define REG_GET_FIELD(value, reg, field) (((value) & REG_FIELD_MASK(reg, field)) >> REG_FIELD_SHIFT(reg, field))
#define WREG32_FIELD(reg, field, val) WREG32(mm##reg, (RREG32(mm##reg) & ~REG_FIELD_MASK(reg, field)) | (val) << REG_FIELD_SHIFT(reg, field))
#define WREG32_FIELD_OFFSET(reg, offset, field, val) \
	WREG32(mm##reg + offset, (RREG32(mm##reg + offset) & ~REG_FIELD_MASK(reg, field)) | (val) << REG_FIELD_SHIFT(reg, field))

/* ---- device core (gpud's replacement for amdgpu_device.c) ---- */
void amdgpu_device_program_register_sequence(struct amdgpu_device *adev, const u32 *registers,
					     const u32 array_size);
bool amdgpu_device_has_dc_support(struct amdgpu_device *adev);
bool amdgpu_device_should_use_aspm(struct amdgpu_device *adev);
int amdgpu_device_pci_reset(struct amdgpu_device *adev);
void amdgpu_device_pci_config_reset(struct amdgpu_device *adev);
int amdgpu_device_resize_fb_bar(struct amdgpu_device *adev);
uint32_t amdgpu_get_soft_full_reset_mask(struct amdgpu_ring *ring);
bool amdgpu_device_need_post(struct amdgpu_device *adev);
void amdgpu_device_flush_hdp(struct amdgpu_device *adev, struct amdgpu_ring *ring);
void amdgpu_device_invalidate_hdp(struct amdgpu_device *adev, struct amdgpu_ring *ring);
bool amdgpu_read_bios(struct amdgpu_device *adev);
void amdgpu_atombios_scratch_regs_engine_hung(struct amdgpu_device *adev, bool hung);
int amdgpu_ucode_request(struct amdgpu_device *adev, const struct firmware **fw, enum amdgpu_ucode_required required,
			 const char *fmt, ...);
void amdgpu_ucode_release(const struct firmware **fw);

#endif
