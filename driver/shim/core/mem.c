// SPDX-License-Identifier: MIT
/*
 * Memory for the hardware code: kernel buffer objects without TTM, the GART table, and the GMC/VM sizing helpers.
 *
 * VRAM objects come from a carve-out inside the CPU-visible window (BAR0), so every kernel object has a CPU
 * pointer; GTT objects are host pages from the platform's DMA allocator, mapped through the GART like amdgpu's
 * pinned GTT buffers. Nothing moves or is evicted. Function bodies marked "Linux 7.3" are copied from the named
 * file (Copyright Advanced Micro Devices, Inc., MIT license).
 */
#include "amdgpu.h"
#include <gpud_core.h>

extern const struct gpud_platform *lkpi_platform;

/* VRAM below 16 MiB is left to the VBIOS (amdgpu reserves 9 MiB, AMDGPU_VBIOS_VGA_ALLOCATION); kernel objects
 * take 16-64 MiB; the rest of the window and all VRAM above it is for clients. */
#define GPUD_KERNEL_VRAM_START (16ULL << 20)
#define GPUD_KERNEL_VRAM_END (64ULL << 20)

/* First-fit range allocator. */
struct gpud_range {
	u64 start, end;
};
struct gpud_ranges {
	struct gpud_range r[256];
	int n;
	struct mutex lock;
};

static void ranges_init(struct gpud_ranges *a, u64 start, u64 end)
{
	a->r[0] = (struct gpud_range){ start, end };
	a->n = 1;
	mutex_init(&a->lock);
}

static int ranges_alloc(struct gpud_ranges *a, u64 size, u64 align, u64 *out)
{
	mutex_lock(&a->lock);
	for (int i = 0; i < a->n; i++) {
		u64 s = ALIGN(a->r[i].start, align), e = s + size;

		if (e > a->r[i].end)
			continue;
		if (s > a->r[i].start && e < a->r[i].end) {
			if (a->n == ARRAY_SIZE(a->r))
				break;
			memmove(&a->r[i + 2], &a->r[i + 1], (a->n - i - 1) * sizeof(a->r[0]));
			a->r[i + 1] = (struct gpud_range){ e, a->r[i].end };
			a->r[i].end = s;
			a->n++;
		} else if (s > a->r[i].start) {
			a->r[i].end = s;
		} else if (e < a->r[i].end) {
			a->r[i].start = e;
		} else {
			memmove(&a->r[i], &a->r[i + 1], (a->n - i - 1) * sizeof(a->r[0]));
			a->n--;
		}
		*out = s;
		mutex_unlock(&a->lock);
		return 0;
	}
	mutex_unlock(&a->lock);
	return -ENOMEM;
}

static void ranges_free(struct gpud_ranges *a, u64 start, u64 size)
{
	int i;

	mutex_lock(&a->lock);
	for (i = 0; i < a->n && a->r[i].start < start; i++)
		;
	if (i > 0 && a->r[i - 1].end == start) {
		a->r[i - 1].end += size;
		if (i < a->n && a->r[i].start == a->r[i - 1].end) {
			a->r[i - 1].end = a->r[i].end;
			memmove(&a->r[i], &a->r[i + 1], (a->n - i - 1) * sizeof(a->r[0]));
			a->n--;
		}
	} else if (i < a->n && a->r[i].start == start + size) {
		a->r[i].start = start;
	} else if (a->n < (int)ARRAY_SIZE(a->r)) {
		memmove(&a->r[i + 1], &a->r[i], (a->n - i) * sizeof(a->r[0]));
		a->r[i] = (struct gpud_range){ start, start + size };
		a->n++;
	}
	mutex_unlock(&a->lock);
}

struct gpud_mem {
	u8 __iomem *vram_cpu; /* BAR0 mapping */
	u64 vram_cpu_size;
	struct gpud_ranges kernel_vram; /* offsets from the start of VRAM */
	struct gpud_ranges gart;	/* offsets into the GART aperture */
};

static struct gpud_mem *mem(struct amdgpu_device *adev)
{
	return (struct gpud_mem *)adev->mman.gtt_mgr_priv;
}

int gpud_mem_create(struct amdgpu_device *adev, const struct gpud_device_info *info)
{
	struct gpud_mem *m = calloc(1, sizeof(*m));

	if (!m)
		return -ENOMEM;
	m->vram_cpu = info->vram;
	m->vram_cpu_size = info->vram_size;
	ranges_init(&m->kernel_vram, GPUD_KERNEL_VRAM_START, min_t(u64, GPUD_KERNEL_VRAM_END, info->vram_size));
	adev->mman.gtt_mgr_priv = m;
	return 0;
}

/* The CPU pointer of a VRAM offset in the BAR0 window, or NULL. */
void *gpud_vram_cpu(struct amdgpu_device *adev, u64 offset, u64 size)
{
	struct gpud_mem *m = mem(adev);

	return offset + size <= m->vram_cpu_size ? m->vram_cpu + offset : NULL;
}

void amdgpu_gart_invalidate_tlb(struct amdgpu_device *adev)
{
	if (!adev->gart.ptr)
		return;
	mb();
	amdgpu_device_flush_hdp(adev, NULL);
	adev->gmc.gmc_funcs->flush_gpu_tlb(adev, 0, AMDGPU_GFXHUB(0), 0);
}

static void gart_write(struct amdgpu_device *adev, u64 gart_offset, u64 iova, u64 size)
{
	u64 flags = AMDGPU_PTE_VALID | AMDGPU_PTE_READABLE | AMDGPU_PTE_WRITEABLE | AMDGPU_PTE_SYSTEM |
		    AMDGPU_PTE_SNOOPED;
	u64 t = gart_offset / AMDGPU_GPU_PAGE_SIZE;

	for (u64 off = 0; off < size; off += AMDGPU_GPU_PAGE_SIZE, t++)
		writeq(((iova + off) & 0x000000FFFFFFF000ULL) | flags, (u8 __iomem *)adev->gart.ptr + t * 8);
	amdgpu_gart_invalidate_tlb(adev);
}

int amdgpu_bo_create_kernel(struct amdgpu_device *adev, unsigned long size, int align, u32 domain,
			    struct amdgpu_bo **bo_ptr, u64 *gpu_addr, void **cpu_addr)
{
	struct gpud_mem *m = mem(adev);
	struct amdgpu_bo *bo;
	int r;

	if (!size) {
		amdgpu_bo_free_kernel(bo_ptr, gpu_addr, cpu_addr);
		return 0;
	}
	if (*bo_ptr)
		goto out;
	bo = calloc(1, sizeof(*bo));
	if (!bo)
		return -ENOMEM;
	size = ALIGN(size, PAGE_SIZE);
	align = max(align, (int)PAGE_SIZE);
	bo->size = size;
	if (domain & AMDGPU_GEM_DOMAIN_VRAM) {
		u64 off;

		r = ranges_alloc(&m->kernel_vram, size, align, &off);
		if (r == 0) {
			bo->domain = AMDGPU_GEM_DOMAIN_VRAM;
			bo->vram_offset = off;
			bo->gpu_addr = adev->gmc.vram_start + off;
			bo->cpu_ptr = m->vram_cpu + off;
			memset_io(bo->cpu_ptr, 0, size);
			goto done;
		}
		if (!(domain & AMDGPU_GEM_DOMAIN_GTT)) {
			free(bo);
			return r;
		}
	}
	if (!(domain & AMDGPU_GEM_DOMAIN_GTT) || !adev->gart.ptr) {
		dev_err(adev->dev, "kernel BO in domain %#x not supported before the GART exists\n", domain);
		free(bo);
		return -EINVAL;
	}
	r = lkpi_platform->dma_alloc(lkpi_platform->ctx, size, &bo->cpu_ptr, &bo->iova);
	if (r) {
		free(bo);
		return -ENOMEM;
	}
	{
		u64 off;

		r = ranges_alloc(&m->gart, size, align, &off);
		if (r) {
			lkpi_platform->dma_free(lkpi_platform->ctx, bo->cpu_ptr, bo->iova, size);
			free(bo);
			return r;
		}
		bo->domain = AMDGPU_GEM_DOMAIN_GTT;
		bo->gpu_addr = adev->gmc.gart_start + off;
		gart_write(adev, off, bo->iova, size);
	}
done:
	*bo_ptr = bo;
out:
	if (gpu_addr)
		*gpu_addr = (*bo_ptr)->gpu_addr;
	if (cpu_addr)
		*cpu_addr = (*bo_ptr)->cpu_ptr;
	return 0;
}

int amdgpu_bo_create_reserved(struct amdgpu_device *adev, unsigned long size, int align, u32 domain,
			      struct amdgpu_bo **bo_ptr, u64 *gpu_addr, void **cpu_addr)
{
	return amdgpu_bo_create_kernel(adev, size, align, domain, bo_ptr, gpu_addr, cpu_addr);
}

void amdgpu_bo_free_kernel(struct amdgpu_bo **bo_ptr, u64 *gpu_addr, void **cpu_addr)
{
	struct amdgpu_bo *bo = bo_ptr ? *bo_ptr : NULL;

	if (bo) {
		/* Kernel objects are freed only at teardown; their address ranges are not reused while the GPU may
		 * still reach them. */
		if (bo->domain == AMDGPU_GEM_DOMAIN_GTT)
			lkpi_platform->dma_free(lkpi_platform->ctx, bo->cpu_ptr, bo->iova, bo->size);
		free(bo);
		*bo_ptr = NULL;
	}
	if (gpu_addr)
		*gpu_addr = 0;
	if (cpu_addr)
		*cpu_addr = NULL;
}

u64 amdgpu_bo_gpu_offset(struct amdgpu_bo *bo) { return bo->gpu_addr; }

int amdgpu_bo_init(struct amdgpu_device *adev)
{
	dev_info(adev->dev, "%lluM of VRAM, %lluM CPU-visible, kernel objects in %lluM-%lluM\n",
		 (unsigned long long)(adev->gmc.real_vram_size >> 20),
		 (unsigned long long)(adev->gmc.visible_vram_size >> 20), GPUD_KERNEL_VRAM_START >> 20,
		 GPUD_KERNEL_VRAM_END >> 20);
	return 0;
}

void amdgpu_bo_fini(struct amdgpu_device *adev) { (void)adev; }

int gpud_mem_scratch_init(struct amdgpu_device *adev)
{
	return amdgpu_bo_create_kernel(adev, AMDGPU_GPU_PAGE_SIZE, PAGE_SIZE, AMDGPU_GEM_DOMAIN_VRAM,
				       &adev->mem_scratch.robj, &adev->mem_scratch.gpu_addr,
				       (void **)&adev->mem_scratch.ptr);
}

/* ---- GART (amdgpu_gart.c) ---- */
int amdgpu_gart_init(struct amdgpu_device *adev)
{
	void *cpu;
	u64 iova;

	if (adev->dummy_page_addr)
		return 0;
	/* The page unused GART entries point at (amdgpu_gart_dummy_page_init). */
	if (lkpi_platform->dma_alloc(lkpi_platform->ctx, PAGE_SIZE, &cpu, &iova))
		return -ENOMEM;
	adev->dummy_page_addr = iova;
	adev->gart.num_cpu_pages = adev->gmc.gart_size / PAGE_SIZE;
	adev->gart.num_gpu_pages = adev->gmc.gart_size / AMDGPU_GPU_PAGE_SIZE;
	ranges_init(&mem(adev)->gart, 0, adev->gmc.gart_size);
	dev_info(adev->dev, "GART: %u GPU pages, dummy page at IOVA %#llx\n", adev->gart.num_gpu_pages,
		 (unsigned long long)iova);
	return 0;
}

/* Linux 7.3 amdgpu_gart.c */
int amdgpu_gart_table_vram_alloc(struct amdgpu_device *adev)
{
	int r;

	if (adev->gart.bo != NULL)
		return 0;

	r = amdgpu_bo_create_kernel(adev, adev->gart.table_size, PAGE_SIZE, AMDGPU_GEM_DOMAIN_VRAM, &adev->gart.bo,
				    NULL, (void *)&adev->gart.ptr);
	if (r)
		return r;

	memset_io(adev->gart.ptr, adev->gart.gart_pte_flags, adev->gart.table_size);
	return 0;
}

void amdgpu_gart_table_vram_free(struct amdgpu_device *adev)
{
	amdgpu_bo_free_kernel(&adev->gart.bo, NULL, (void *)&adev->gart.ptr);
}

/* ---- GMC helpers (Linux 7.3 amdgpu_gmc.c; XGMI and the GMC hole don't exist on SI) ---- */
void amdgpu_gmc_vram_location(struct amdgpu_device *adev, struct amdgpu_gmc *mc, u64 base)
{
	mc->vram_start = base;
	mc->vram_end = mc->vram_start + mc->mc_vram_size - 1;
	if (mc->real_vram_size < mc->visible_vram_size)
		mc->visible_vram_size = mc->real_vram_size;
	mc->fb_start = mc->vram_start;
	mc->fb_end = mc->vram_end;
	dev_info(adev->dev, "VRAM: %lluM 0x%016llX - 0x%016llX (%lluM used)\n",
		 (unsigned long long)mc->mc_vram_size >> 20, (unsigned long long)mc->vram_start,
		 (unsigned long long)mc->vram_end, (unsigned long long)mc->real_vram_size >> 20);
}

void amdgpu_gmc_gart_location(struct amdgpu_device *adev, struct amdgpu_gmc *mc,
			      enum amdgpu_gart_placement gart_placement)
{
	const uint64_t four_gb = 0x100000000ULL;
	u64 size_af, size_bf;
	u64 max_mc_address = adev->gmc.mc_mask;

	size_bf = mc->fb_start;
	size_af = max_mc_address + 1 - ALIGN(mc->fb_end + 1, four_gb);

	if (mc->gart_size > max(size_bf, size_af)) {
		dev_warn(adev->dev, "limiting GART\n");
		mc->gart_size = max(size_bf, size_af);
	}

	switch (gart_placement) {
	case AMDGPU_GART_PLACEMENT_HIGH:
		mc->gart_start = max_mc_address - mc->gart_size + 1;
		break;
	case AMDGPU_GART_PLACEMENT_LOW:
		if (size_bf >= mc->gart_size)
			mc->gart_start = 0;
		else
			mc->gart_start = ALIGN(mc->fb_end, four_gb);
		break;
	case AMDGPU_GART_PLACEMENT_BEST_FIT:
	default:
		if ((size_bf >= mc->gart_size && size_bf < size_af) || (size_af < mc->gart_size))
			mc->gart_start = 0;
		else
			mc->gart_start = max_mc_address - mc->gart_size + 1;
		break;
	}

	mc->gart_start &= ~(four_gb - 1);
	mc->gart_end = mc->gart_start + mc->gart_size - 1;
	dev_info(adev->dev, "GART: %lluM 0x%016llX - 0x%016llX\n", (unsigned long long)mc->gart_size >> 20,
		 (unsigned long long)mc->gart_start, (unsigned long long)mc->gart_end);
}

void amdgpu_gmc_set_agp_default(struct amdgpu_device *adev, struct amdgpu_gmc *mc)
{
	(void)adev;
	mc->agp_start = 0xffffffffffff;
	mc->agp_end = 0;
	mc->agp_size = 0;
}

void amdgpu_gmc_set_gart_size(struct amdgpu_device *adev, u64 default_size)
{
	if (amdgpu_gart_size == -1)
		adev->gmc.gart_size = default_size;
	else
		adev->gmc.gart_size = (u64)amdgpu_gart_size << 20;
}

int amdgpu_gmc_vram_checking(struct amdgpu_device *adev)
{
	struct amdgpu_bo *vram_bo = NULL;
	uint64_t vram_gpu = 0;
	void *vram_ptr = NULL;
	int ret, size = 0x100000;
	uint8_t cptr[10];

	ret = amdgpu_bo_create_kernel(adev, size, PAGE_SIZE, AMDGPU_GEM_DOMAIN_VRAM, &vram_bo, &vram_gpu, &vram_ptr);
	if (ret)
		return ret;
	memset(vram_ptr, 0x86, size);
	memset(cptr, 0x86, 10);
	ret = memcmp(vram_ptr, cptr, 10) || memcmp((u8 *)vram_ptr + (size / 2), cptr, 10) ||
	      memcmp((u8 *)vram_ptr + size - 10, cptr, 10) ? -EIO : 0;
	amdgpu_bo_free_kernel(&vram_bo, &vram_gpu, &vram_ptr);
	return ret;
}

/* ---- VM sizing (Linux 7.3 amdgpu_vm.c) ---- */
static uint32_t amdgpu_vm_get_block_size(uint64_t vm_size)
{
	unsigned bits = ilog2(vm_size) + 18;

	if (vm_size <= 8)
		return (bits - 9);
	else
		return ((bits + 3) / 2);
}

/* amdgpu sizes the VM from system RAM (three times RAM, rounded up); gpud fixes it at 256 GB, what amdgpu picks
 * on a Mac Pro with 64 GB, so page table sizes match. */
void amdgpu_vm_adjust_size(struct amdgpu_device *adev, uint32_t min_vm_size, uint32_t fragment_size_default,
			   unsigned max_level, unsigned max_bits)
{
	unsigned int max_size = 1 << (max_bits - 30);
	unsigned int vm_size = amdgpu_vm_size != -1 ? (unsigned)amdgpu_vm_size : clamp(256u, min_vm_size, max_size);
	uint64_t tmp;

	adev->vm_manager.max_pfn = (uint64_t)vm_size << 18;
	tmp = roundup_pow_of_two(adev->vm_manager.max_pfn);
	if (amdgpu_vm_block_size != -1)
		tmp >>= amdgpu_vm_block_size - 9;
	tmp = DIV_ROUND_UP(fls64(tmp) - 1, 9) - 1;
	adev->vm_manager.num_level = min_t(unsigned int, max_level, tmp);
	if (amdgpu_vm_block_size != -1)
		adev->vm_manager.block_size = min((unsigned)amdgpu_vm_block_size,
						  max_bits - AMDGPU_GPU_PAGE_SHIFT - 9 * adev->vm_manager.num_level);
	else if (adev->vm_manager.num_level > 1)
		adev->vm_manager.block_size = 9;
	else
		adev->vm_manager.block_size = amdgpu_vm_get_block_size(tmp);
	adev->vm_manager.fragment_size = fragment_size_default;
	dev_info(adev->dev, "vm size is %u GB, %u levels, block size is %u-bit, fragment size is %u-bit\n", vm_size,
		 adev->vm_manager.num_level + 1, adev->vm_manager.block_size, adev->vm_manager.fragment_size);
}

void amdgpu_vm_manager_init(struct amdgpu_device *adev) { (void)adev; }
void amdgpu_vm_manager_fini(struct amdgpu_device *adev) { (void)adev; }
