// SPDX-License-Identifier: MIT
/*
 * Client address spaces: one GPU VM per client, in place of amdgpu_vm.c and amdgpu_vm_pt.c. SI's VM is two levels
 * (gmc_v6_0_gart_enable: VM_CONTEXT1 with PAGE_TABLE_DEPTH 1, block size from amdgpu_vm_adjust_size): a page
 * directory of 8-byte entries, each pointing at a page table of 2^block_size entries for 4 KiB pages. Tables live
 * in VRAM in the kernel carve-out and are written by the CPU through BAR0; nothing is ever moved or evicted.
 *
 * Entry formats follow amdgpu: a directory entry is the table's MC address | VALID (amdgpu_gmc_get_pde_for_bo);
 * a VRAM leaf entry is the offset into VRAM (vram_base_offset 0 on dGPUs) | flags; a system leaf entry is the
 * IOVA | SYSTEM | SNOOPED | flags (amdgpu_ttm_tt_pte_flags).
 */
#include "amdgpu.h"
#include <gpud_core.h>

void *gpud_vram_cpu(struct amdgpu_device *adev, u64 offset, u64 size);

struct gpud_vm {
	struct amdgpu_device *adev;
	unsigned vmid;
	unsigned block_size;
	u64 num_pde;
	struct amdgpu_bo *pd;
	u64 *pd_cpu;
	struct amdgpu_bo **pt;
	u64 **pt_cpu;
	bool dirty; /* entries changed since the last TLB flush */
	struct mutex lock;
};

struct gpud_vm *gpud_vm_create(struct amdgpu_device *adev, unsigned vmid)
{
	struct gpud_vm *vm = calloc(1, sizeof(*vm));
	u64 pd_gpu;
	int r;

	if (!vm || vmid == 0 || vmid >= AMDGPU_NUM_VMID)
		goto fail;
	vm->adev = adev;
	vm->vmid = vmid;
	vm->block_size = adev->vm_manager.block_size;
	vm->num_pde = adev->vm_manager.max_pfn >> vm->block_size;
	vm->pt = calloc(vm->num_pde, sizeof(*vm->pt));
	vm->pt_cpu = calloc(vm->num_pde, sizeof(*vm->pt_cpu));
	if (!vm->pt || !vm->pt_cpu)
		goto fail;
	r = amdgpu_bo_create_kernel(adev, vm->num_pde * 8, PAGE_SIZE, AMDGPU_GEM_DOMAIN_VRAM, &vm->pd, &pd_gpu,
				    (void **)&vm->pd_cpu);
	if (r)
		goto fail;
	mutex_init(&vm->lock);
	dev_info(adev->dev, "VM %u: %llu directory entries at %#llx, %u KiB tables\n", vmid,
		 (unsigned long long)vm->num_pde, (unsigned long long)pd_gpu, (8u << vm->block_size) >> 10);
	return vm;
fail:
	if (vm) {
		free(vm->pt);
		free(vm->pt_cpu);
	}
	free(vm);
	return NULL;
}

u64 gpud_vm_pd_addr(struct gpud_vm *vm) { return vm->pd->gpu_addr; }
unsigned gpud_vm_vmid(struct gpud_vm *vm) { return vm->vmid; }

static u64 *vm_pt(struct gpud_vm *vm, u64 pde, bool create)
{
	u64 gpu;

	if (vm->pt_cpu[pde] || !create)
		return vm->pt_cpu[pde];
	if (amdgpu_bo_create_kernel(vm->adev, 8ULL << vm->block_size, PAGE_SIZE, AMDGPU_GEM_DOMAIN_VRAM, &vm->pt[pde],
				    &gpu, (void **)&vm->pt_cpu[pde]))
		return NULL;
	writeq(gpu | AMDGPU_PTE_VALID, &vm->pd_cpu[pde]);
	return vm->pt_cpu[pde];
}

/*
 * Maps [va, va + size) to device memory starting at addr: a VRAM offset, or with AMDGPU_PTE_SYSTEM in flags, an
 * IOVA. flags carries VALID/READABLE/WRITEABLE (and SYSTEM|SNOOPED for host memory). Runs of 64 KiB or more that
 * are aligned in both spaces get the fragment field, so the TLB caches them as one entry (amdgpu_vm_update_range).
 */
int gpud_vm_map(struct gpud_vm *vm, u64 va, u64 size, u64 addr, u64 flags)
{
	u64 first = va >> AMDGPU_GPU_PAGE_SHIFT, n = size >> AMDGPU_GPU_PAGE_SHIFT;
	u64 mask = (1ULL << vm->block_size) - 1;
	int r = 0;

	if ((va | size | addr) & AMDGPU_GPU_PAGE_MASK || first + n > vm->adev->vm_manager.max_pfn)
		return -EINVAL;
	mutex_lock(&vm->lock);
	for (u64 i = 0; i < n;) {
		u64 pfn = first + i;
		u64 *pt = vm_pt(vm, pfn >> vm->block_size, true);
		/* largest fragment (2^frag pages) that is aligned at this page in both spaces and fits */
		unsigned frag = 0;

		if (!pt) {
			r = -ENOMEM;
			break;
		}
		while (frag < 9 && !((pfn | ((addr >> 12) + i)) & ((2ULL << frag) - 1)) && i + (2ULL << frag) <= n)
			frag++;
		for (u64 k = 0; k < (1ULL << frag); k++, i++)
			writeq(((addr + (i << 12)) & 0x000000FFFFFFF000ULL) | flags | AMDGPU_PTE_FRAG(frag),
			       &pt[(first + i) & mask]);
	}
	vm->dirty = true;
	mutex_unlock(&vm->lock);
	return r;
}

int gpud_vm_unmap(struct gpud_vm *vm, u64 va, u64 size)
{
	u64 first = va >> AMDGPU_GPU_PAGE_SHIFT, n = size >> AMDGPU_GPU_PAGE_SHIFT;
	u64 mask = (1ULL << vm->block_size) - 1;

	if ((va | size) & AMDGPU_GPU_PAGE_MASK || first + n > vm->adev->vm_manager.max_pfn)
		return -EINVAL;
	mutex_lock(&vm->lock);
	for (u64 i = 0; i < n; i++) {
		u64 *pt = vm_pt(vm, (first + i) >> vm->block_size, false);

		if (pt)
			writeq(0, &pt[(first + i) & mask]);
	}
	vm->dirty = true;
	mutex_unlock(&vm->lock);
	return 0;
}

/* ---- client VRAM: everything above the kernel carve-out ---- */
static struct {
	u64 start, end;
	struct { u64 start, end; } r[4096];
	int n;
	pthread_mutex_t lock;
} vram = { .lock = PTHREAD_MUTEX_INITIALIZER };

void gpud_vram_init(struct amdgpu_device *adev, u64 start)
{
	vram.r[0].start = start;
	vram.r[0].end = adev->gmc.real_vram_size;
	vram.n = 1;
}

int gpud_vram_alloc(struct amdgpu_device *adev, u64 size, u64 align, u64 *offset)
{
	(void)adev;
	align = max_t(u64, align, AMDGPU_GPU_PAGE_SIZE);
	size = ALIGN(size, AMDGPU_GPU_PAGE_SIZE);
	pthread_mutex_lock(&vram.lock);
	for (int i = 0; i < vram.n; i++) {
		u64 s = ALIGN(vram.r[i].start, align), e = s + size;

		if (e > vram.r[i].end)
			continue;
		if (s > vram.r[i].start && e < vram.r[i].end) {
			if (vram.n == (int)ARRAY_SIZE(vram.r))
				break;
			memmove(&vram.r[i + 2], &vram.r[i + 1], (vram.n - i - 1) * sizeof(vram.r[0]));
			vram.r[i + 1].start = e;
			vram.r[i + 1].end = vram.r[i].end;
			vram.r[i].end = s;
			vram.n++;
		} else if (s > vram.r[i].start) {
			vram.r[i].end = s;
		} else if (e < vram.r[i].end) {
			vram.r[i].start = e;
		} else {
			memmove(&vram.r[i], &vram.r[i + 1], (vram.n - i - 1) * sizeof(vram.r[0]));
			vram.n--;
		}
		*offset = s;
		pthread_mutex_unlock(&vram.lock);
		return 0;
	}
	pthread_mutex_unlock(&vram.lock);
	return -ENOMEM;
}

void gpud_vram_free(struct amdgpu_device *adev, u64 offset, u64 size)
{
	int i;

	(void)adev;
	size = ALIGN(size, AMDGPU_GPU_PAGE_SIZE);
	pthread_mutex_lock(&vram.lock);
	for (i = 0; i < vram.n && vram.r[i].start < offset; i++)
		;
	if (i > 0 && vram.r[i - 1].end == offset) {
		vram.r[i - 1].end += size;
		if (i < vram.n && vram.r[i].start == vram.r[i - 1].end) {
			vram.r[i - 1].end = vram.r[i].end;
			memmove(&vram.r[i], &vram.r[i + 1], (vram.n - i - 1) * sizeof(vram.r[0]));
			vram.n--;
		}
	} else if (i < vram.n && vram.r[i].start == offset + size) {
		vram.r[i].start = offset;
	} else if (vram.n < (int)ARRAY_SIZE(vram.r)) {
		memmove(&vram.r[i + 1], &vram.r[i], (vram.n - i) * sizeof(vram.r[0]));
		vram.r[i].start = offset;
		vram.r[i].end = offset + size;
		vram.n++;
	}
	pthread_mutex_unlock(&vram.lock);
}

u64 gpud_vram_free_bytes(void)
{
	u64 total = 0;

	pthread_mutex_lock(&vram.lock);
	for (int i = 0; i < vram.n; i++)
		total += vram.r[i].end - vram.r[i].start;
	pthread_mutex_unlock(&vram.lock);
	return total;
}

/* ---- submission ---- */
extern u32 gpud_fence_emit(struct amdgpu_ring *ring);

/* The ring for a DRM IP type and ring index (AMDGPU_HW_IP_GFX/COMPUTE/DMA), or -EINVAL. */
int gpud_ring_for(struct amdgpu_device *adev, u32 ip_type, u32 ring)
{
	struct amdgpu_ring *r = NULL;

	switch (ip_type) {
	case AMDGPU_HW_IP_GFX:
		if (ring < adev->gfx.num_gfx_rings)
			r = &adev->gfx.gfx_ring[ring];
		break;
	case AMDGPU_HW_IP_COMPUTE:
		if (ring < adev->gfx.num_compute_rings)
			r = &adev->gfx.compute_ring[ring];
		break;
	case AMDGPU_HW_IP_DMA:
		if ((int)ring < adev->sdma.num_instances)
			r = &adev->sdma.instance[ring].ring;
		break;
	}
	return r && r->sched.ready ? (int)r->idx : -EINVAL;
}

/*
 * Runs a client's IBs on one ring in its VM: VM flush (page directory base and TLB invalidate for the VMID), the
 * IBs, a fence with an interrupt. Returns the fence's sequence number in *seq. One submission at a time per device.
 */
static pthread_mutex_t submit_lock = PTHREAD_MUTEX_INITIALIZER;

int gpud_submit(struct amdgpu_device *adev, int ring_key, struct gpud_vm *vm, const struct gpud_ib *ibs, unsigned n,
		uint64_t *seq)
{
	struct amdgpu_ring *ring;
	struct amdgpu_ib ib[16];
	struct amdgpu_job job = { 0 };
	u32 status = AMDGPU_HAVE_CTX_SWITCH;
	int r;

	if (ring_key < 0 || (unsigned)ring_key >= adev->num_rings || n == 0 || n > ARRAY_SIZE(ib))
		return -EINVAL;
	ring = adev->rings[ring_key];
	if (!ring->sched.ready)
		return -ENODEV;
	if (vm) {
		job.vmid = vm->vmid;
		job.vm_pd_addr = vm->pd->gpu_addr;
	}
	for (unsigned i = 0; i < n; i++) {
		memset(&ib[i], 0, sizeof(ib[i]));
		ib[i].gpu_addr = ibs[i].va;
		ib[i].length_dw = ibs[i].size_dw;
		ib[i].flags = ibs[i].flags;
	}

	pthread_mutex_lock(&submit_lock);
	if (vm) {
		mutex_lock(&vm->lock);
		if (vm->dirty) {
			/* Page table writes went through BAR0: out of the HDP before the TLB refills from them. */
			amdgpu_device_flush_hdp(adev, NULL);
			vm->dirty = false;
		}
		mutex_unlock(&vm->lock);
	}
	r = amdgpu_ring_alloc(ring, ring->funcs->emit_frame_size + n * ring->funcs->emit_ib_size);
	if (r) {
		pthread_mutex_unlock(&submit_lock);
		return r;
	}
	if (vm) {
		bool sync = true;

		amdgpu_vm_flush(ring, &job, &sync, NULL, NULL);
	}
	if (ring->funcs->insert_start)
		ring->funcs->insert_start(ring);
	amdgpu_device_flush_hdp(adev, ring);
	if (ring->funcs->emit_cntxcntl)
		amdgpu_ring_emit_cntxcntl(ring, status);
	for (unsigned i = 0; i < n; i++) {
		amdgpu_ring_emit_ib(ring, vm ? &job : NULL, &ib[i], status);
		status &= ~AMDGPU_HAVE_CTX_SWITCH;
	}
	amdgpu_device_invalidate_hdp(adev, ring);
	*seq = gpud_fence_emit(ring);
	if (ring->funcs->insert_end)
		ring->funcs->insert_end(ring);
	amdgpu_ring_commit(ring);
	pthread_mutex_unlock(&submit_lock);
	return 0;
}

/* The last sequence number a ring has completed. */
uint64_t gpud_fence_read(struct amdgpu_device *adev, int ring_key)
{
	return le32_to_cpu(READ_ONCE(*adev->rings[ring_key]->fence_drv.cpu_addr));
}

/* 0 once the ring has passed seq, -ETIME after timeout_ns (negative: forever). Polls the fence slot. */
int gpud_fence_wait(struct amdgpu_device *adev, int ring_key, uint64_t seq, int64_t timeout_ns)
{
	u64 deadline = timeout_ns < 0 ? U64_MAX : lkpi_ktime_ns() + (u64)timeout_ns;
	unsigned spins = 0;

	if (ring_key < 0 || (unsigned)ring_key >= adev->num_rings)
		return -EINVAL;
	while ((int32_t)((u32)gpud_fence_read(adev, ring_key) - (u32)seq) < 0) {
		if (lkpi_ktime_ns() >= deadline)
			return -ETIME;
		/* spin briefly for short jobs, then back off */
		if (++spins < 2000)
			cpu_relax();
		else
			lkpi_udelay(spins < 20000 ? 5 : 50);
	}
	return 0;
}
