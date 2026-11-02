// SPDX-License-Identifier: MIT
/*
 * Rings, fences, IBs and interrupts for the SI hardware code, in place of amdgpu_ring.c, amdgpu_fence.c,
 * amdgpu_ib.c and amdgpu_irq.c without the GPU scheduler: rings are driven directly, a fence is a sequence number
 * the ring writes to memory, and interrupts are dispatched from gpud's interrupt thread. Function bodies marked
 * "Linux 7.3" are copied from the named file (Copyright Advanced Micro Devices, Inc., MIT license).
 */
#include "amdgpu.h"
#include "sid.h"
#include <gpud_core.h>

/* ---- rings (Linux 7.3 amdgpu_ring.c) ---- */
unsigned int amdgpu_ring_max_ibs(enum amdgpu_ring_type type)
{
	switch (type) {
	case AMDGPU_RING_TYPE_GFX:
		return 192;
	case AMDGPU_RING_TYPE_COMPUTE:
		return 125;
	default:
		return 49;
	}
}

int amdgpu_ring_alloc(struct amdgpu_ring *ring, unsigned int ndw)
{
	ndw = (ndw + ring->funcs->align_mask) & ~ring->funcs->align_mask;
	if (WARN_ON_ONCE(ndw > ring->max_dw))
		return -ENOMEM;
	ring->count_dw = ndw;
	ring->wptr_old = ring->wptr;
	if (ring->funcs->begin_use)
		ring->funcs->begin_use(ring);
	return 0;
}

void amdgpu_ring_insert_nop(struct amdgpu_ring *ring, uint32_t count)
{
	uint32_t occupied, chunk1, chunk2;

	occupied = ring->wptr & ring->buf_mask;
	chunk1 = ring->buf_mask + 1 - occupied;
	chunk1 = (chunk1 >= count) ? count : chunk1;
	chunk2 = count - chunk1;

	if (chunk1)
		memset32(&ring->ring[occupied], ring->funcs->nop, chunk1);
	if (chunk2)
		memset32(ring->ring, ring->funcs->nop, chunk2);

	ring->wptr += count;
	ring->wptr &= ring->ptr_mask;
	ring->count_dw -= count;
}

void amdgpu_ring_generic_pad_ib(struct amdgpu_ring *ring, struct amdgpu_ib *ib)
{
	u32 align_mask = ring->funcs->align_mask;
	u32 count = ib->length_dw & align_mask;

	if (count) {
		count = align_mask + 1 - count;
		memset32(&ib->ptr[ib->length_dw], ring->funcs->nop, count);
		ib->length_dw += count;
	}
}

void amdgpu_ring_commit(struct amdgpu_ring *ring)
{
	uint32_t count;

	if (ring->count_dw < 0)
		dev_err(ring->adev->dev, "writing more dwords to the ring than expected!\n");

	count = ring->funcs->align_mask + 1 - (ring->wptr & ring->funcs->align_mask);
	count &= ring->funcs->align_mask;
	if (count != 0)
		ring->funcs->insert_nop(ring, count);

	mb();
	if (ring->ring_obj && ring->ring_obj->domain == AMDGPU_GEM_DOMAIN_VRAM)
		amdgpu_device_flush_hdp(ring->adev, NULL);
	amdgpu_ring_set_wptr(ring);

	if (ring->funcs->end_use)
		ring->funcs->end_use(ring);
}

void amdgpu_ring_undo(struct amdgpu_ring *ring)
{
	ring->wptr = ring->wptr_old;
	if (ring->funcs->end_use)
		ring->funcs->end_use(ring);
}

static u64 ring_wb_gpu(struct amdgpu_ring *ring, unsigned offs) { return ring->adev->wb.gpu_addr + offs * 4; }
static u32 *ring_wb_cpu(struct amdgpu_ring *ring, unsigned offs) { return &ring->adev->wb.wb[offs]; }

int amdgpu_fence_driver_init_ring(struct amdgpu_ring *ring)
{
	ring->fence_drv.cpu_addr = NULL;
	ring->fence_drv.gpu_addr = 0;
	ring->fence_drv.sync_seq = 0;
	atomic_set(&ring->fence_drv.last_seq, 0);
	ring->fence_drv.initialized = false;
	spin_lock_init(&ring->fence_drv.lock);
	return 0;
}

int amdgpu_fence_driver_start_ring(struct amdgpu_ring *ring, struct amdgpu_irq_src *irq_src, unsigned int irq_type)
{
	ring->fence_drv.cpu_addr = ring->fence_cpu_addr;
	ring->fence_drv.gpu_addr = ring->fence_gpu_addr;
	*ring->fence_drv.cpu_addr = atomic_read(&ring->fence_drv.last_seq);
	ring->fence_drv.irq_src = irq_src;
	ring->fence_drv.irq_type = irq_type;
	ring->fence_drv.initialized = true;
	return 0;
}

int amdgpu_ring_init(struct amdgpu_device *adev, struct amdgpu_ring *ring, unsigned int max_dw,
		     struct amdgpu_irq_src *irq_src, unsigned int irq_type, unsigned int hw_prio,
		     atomic_t *sched_score)
{
	int sched_hw_submission = amdgpu_sched_hw_submission;
	unsigned int max_ibs_dw;
	int r;

	(void)sched_score;
	if (ring->adev == NULL) {
		if (adev->num_rings >= AMDGPU_MAX_RINGS)
			return -EINVAL;
		ring->adev = adev;
		ring->num_hw_submission = sched_hw_submission;
		ring->idx = adev->num_rings++;
		adev->rings[ring->idx] = ring;
		r = amdgpu_fence_driver_init_ring(ring);
		if (r)
			return r;
	}

	r = amdgpu_wb_get(adev, &ring->rptr_offs);
	if (!r)
		r = amdgpu_wb_get(adev, &ring->wptr_offs);
	if (!r)
		r = amdgpu_wb_get(adev, &ring->fence_offs);
	if (!r)
		r = amdgpu_wb_get(adev, &ring->trail_fence_offs);
	if (!r)
		r = amdgpu_wb_get(adev, &ring->cond_exe_offs);
	if (r) {
		dev_err(adev->dev, "(%d) ring write-back slot allocation failed\n", r);
		return r;
	}
	ring->fence_gpu_addr = ring_wb_gpu(ring, ring->fence_offs);
	ring->fence_cpu_addr = ring_wb_cpu(ring, ring->fence_offs);
	ring->rptr_gpu_addr = ring_wb_gpu(ring, ring->rptr_offs);
	ring->rptr_cpu_addr = ring_wb_cpu(ring, ring->rptr_offs);
	ring->wptr_gpu_addr = ring_wb_gpu(ring, ring->wptr_offs);
	ring->wptr_cpu_addr = ring_wb_cpu(ring, ring->wptr_offs);
	ring->trail_fence_gpu_addr = ring_wb_gpu(ring, ring->trail_fence_offs);
	ring->trail_fence_cpu_addr = ring_wb_cpu(ring, ring->trail_fence_offs);
	ring->cond_exe_gpu_addr = ring_wb_gpu(ring, ring->cond_exe_offs);
	ring->cond_exe_cpu_addr = ring_wb_cpu(ring, ring->cond_exe_offs);

	r = amdgpu_fence_driver_start_ring(ring, irq_src, irq_type);
	if (r)
		return r;
	max_ibs_dw = ring->funcs->emit_frame_size + amdgpu_ring_max_ibs(ring->funcs->type) * ring->funcs->emit_ib_size;
	max_ibs_dw = (max_ibs_dw + ring->funcs->align_mask) & ~ring->funcs->align_mask;
	if (max_ibs_dw > max_dw)
		max_dw = max_ibs_dw;
	ring->ring_size = roundup_pow_of_two(max_dw * 4 * sched_hw_submission);
	ring->buf_mask = (ring->ring_size / 4) - 1;
	ring->ptr_mask = ring->funcs->support_64bit_ptrs ? 0xffffffffffffffff : ring->buf_mask;
	ring->cached_rptr = 0;

	if (ring->ring_obj == NULL) {
		/* GPUD_RING_VRAM=1 puts the CP rings in VRAM (a diagnostic: the CP then fetches without the GART). */
		const char *vram = getenv("GPUD_RING_VRAM");
		u32 domain = vram && *vram == '1' && (ring->funcs->type == AMDGPU_RING_TYPE_GFX ||
						       ring->funcs->type == AMDGPU_RING_TYPE_COMPUTE) ?
				     AMDGPU_GEM_DOMAIN_VRAM : AMDGPU_GEM_DOMAIN_GTT;

		r = amdgpu_bo_create_kernel(adev, ring->ring_size + ring->funcs->extra_bytes, PAGE_SIZE,
					    domain, &ring->ring_obj, &ring->gpu_addr, (void **)&ring->ring);
		if (r) {
			dev_err(adev->dev, "(%d) ring create failed\n", r);
			return r;
		}
		amdgpu_ring_clear_ring(ring);
	}
	ring->max_dw = max_dw;
	ring->hw_prio = hw_prio;
	ring->vmid_wait = dma_fence_get_stub();
	dev_info(adev->dev, "ring %s: %u KiB at %#llx\n", ring->name, ring->ring_size >> 10,
		 (unsigned long long)ring->gpu_addr);
	return 0;
}

void amdgpu_ring_fini(struct amdgpu_ring *ring)
{
	if (!(ring->adev) || !(ring->adev->rings[ring->idx]))
		return;
	ring->sched.ready = false;
	amdgpu_wb_free(ring->adev, ring->rptr_offs);
	amdgpu_wb_free(ring->adev, ring->wptr_offs);
	amdgpu_wb_free(ring->adev, ring->cond_exe_offs);
	amdgpu_wb_free(ring->adev, ring->fence_offs);
	amdgpu_bo_free_kernel(&ring->ring_obj, &ring->gpu_addr, (void **)&ring->ring);
	dma_fence_put(ring->vmid_wait);
	ring->vmid_wait = NULL;
	ring->me = 0;
}

int amdgpu_ring_test_helper(struct amdgpu_ring *ring)
{
	struct amdgpu_device *adev = ring->adev;
	int r;

	r = amdgpu_ring_test_ring(ring);
	if (r)
		dev_err(adev->dev, "ring %s test failed (%d)\n", ring->name, r);
	else
		dev_info(adev->dev, "ring test on %s succeeded\n", ring->name);
	ring->sched.ready = !r;
	return r;
}

/* ---- fences: a ring's sequence number in its write-back slot ---- */
struct gpud_fence {
	struct dma_fence base;
	struct amdgpu_ring *ring;
};

static u32 fence_read(struct amdgpu_ring *ring) { return le32_to_cpu(READ_ONCE(*ring->fence_drv.cpu_addr)); }

static bool gpud_fence_signaled(struct dma_fence *f)
{
	struct gpud_fence *gf = container_of(f, struct gpud_fence, base);

	return (int32_t)(fence_read(gf->ring) - (u32)f->seqno) >= 0;
}

static void gpud_fence_release(struct dma_fence *f) { free(container_of(f, struct gpud_fence, base)); }

static const struct dma_fence_ops gpud_fence_ops = {
	.signaled = gpud_fence_signaled,
	.release = gpud_fence_release,
};

bool amdgpu_fence_process(struct amdgpu_ring *ring)
{
	u32 seq = fence_read(ring), last = atomic_xchg(&ring->fence_drv.last_seq, seq);

	return seq != last;
}

/* Emits a fence with an interrupt and returns its sequence number. */
u32 gpud_fence_emit(struct amdgpu_ring *ring)
{
	u32 seq = ++ring->fence_drv.sync_seq;

	amdgpu_ring_emit_fence(ring, ring->fence_drv.gpu_addr, seq, AMDGPU_FENCE_FLAG_INT);
	return seq;
}

static struct dma_fence *fence_create(struct amdgpu_ring *ring, u32 seq)
{
	struct gpud_fence *f = calloc(1, sizeof(*f));

	if (!f)
		return NULL;
	dma_fence_init(&f->base, &gpud_fence_ops, NULL, ring->idx, seq);
	f->ring = ring;
	return &f->base;
}

/* ---- the VM flush ahead of a job's IBs (amdgpu_vm_flush, for a job that always needs one) ---- */
void amdgpu_vm_flush(struct amdgpu_ring *ring, struct amdgpu_job *job, bool *need_pipe_sync, bool *emit_spm,
		     bool *emit_gds)
{
	(void)emit_spm;
	(void)emit_gds;
	if (!job || !job->vmid)
		return;
	if (ring->funcs->emit_pipeline_sync && (!need_pipe_sync || *need_pipe_sync))
		amdgpu_ring_emit_pipeline_sync(ring);
	if (need_pipe_sync)
		*need_pipe_sync = false;
	amdgpu_ring_emit_vm_flush(ring, job->vmid, job->vm_pd_addr);
}

/* ---- IBs: a 1 MiB GTT pool, handed out round-robin (only the driver's own tests use it) ---- */
static struct amdgpu_bo *ib_pool;
static u64 ib_pool_gpu;
static u8 *ib_pool_cpu;
static u64 ib_pool_next;
static pthread_mutex_t ib_pool_lock = PTHREAD_MUTEX_INITIALIZER;

int gpud_ib_pool_init(struct amdgpu_device *adev)
{
	int r = amdgpu_bo_create_kernel(adev, AMDGPU_IB_POOL_SIZE, PAGE_SIZE, AMDGPU_GEM_DOMAIN_GTT, &ib_pool,
					&ib_pool_gpu, (void **)&ib_pool_cpu);

	adev->ib_pool_ready = !r;
	return r;
}

int amdgpu_ib_get(struct amdgpu_device *adev, struct amdgpu_vm *vm, unsigned int size,
		  enum amdgpu_ib_pool_type pool_type, struct amdgpu_ib *ib)
{
	struct drm_suballoc *sa;

	(void)adev;
	(void)vm;
	(void)pool_type;
	if (!size)
		return 0;
	sa = calloc(1, sizeof(*sa));
	if (!sa)
		return -ENOMEM;
	size = ALIGN(size, 256);
	pthread_mutex_lock(&ib_pool_lock);
	if (ib_pool_next + size > AMDGPU_IB_POOL_SIZE)
		ib_pool_next = 0;
	sa->soffset = ib_pool_next;
	sa->eoffset = ib_pool_next + size;
	ib_pool_next += size;
	pthread_mutex_unlock(&ib_pool_lock);
	ib->sa_bo = sa;
	ib->ptr = (u32 *)(ib_pool_cpu + sa->soffset);
	ib->gpu_addr = ib_pool_gpu + sa->soffset;
	ib->flags = AMDGPU_IB_FLAG_EMIT_MEM_SYNC;
	return 0;
}

void amdgpu_ib_free(struct amdgpu_ib *ib, struct dma_fence *f)
{
	(void)f;
	free(ib->sa_bo);
	ib->sa_bo = NULL;
}

/* amdgpu_ib_schedule without the scheduler: the job's VM flush, the IBs, then a fence with an interrupt. */
int amdgpu_ib_schedule(struct amdgpu_ring *ring, unsigned int num_ibs, struct amdgpu_ib *ibs, struct amdgpu_job *job,
		       struct dma_fence **f)
{
	struct amdgpu_device *adev = ring->adev;
	bool need_pipe_sync = true;
	u32 status = 0, seq;
	unsigned int i;
	int r;

	if (num_ibs == 0)
		return -EINVAL;
	if (!ring->sched.ready) {
		dev_err(adev->dev, "couldn't schedule ib on ring <%s>\n", ring->name);
		return -EINVAL;
	}
	r = amdgpu_ring_alloc(ring, ring->funcs->emit_frame_size + num_ibs * ring->funcs->emit_ib_size);
	if (r) {
		dev_err(adev->dev, "scheduling IB failed (%d).\n", r);
		return r;
	}
	amdgpu_vm_flush(ring, job, &need_pipe_sync, NULL, NULL);
	if (ring->funcs->insert_start)
		ring->funcs->insert_start(ring);
	amdgpu_device_flush_hdp(adev, ring);
	if (job && ring->funcs->emit_cntxcntl) {
		status |= AMDGPU_HAVE_CTX_SWITCH;
		amdgpu_ring_emit_cntxcntl(ring, status);
	}
	for (i = 0; i < num_ibs; ++i) {
		amdgpu_ring_emit_ib(ring, job, &ibs[i], status);
		status &= ~AMDGPU_HAVE_CTX_SWITCH;
	}
	amdgpu_device_invalidate_hdp(adev, ring);
	seq = gpud_fence_emit(ring);
	if (ring->funcs->insert_end)
		ring->funcs->insert_end(ring);
	amdgpu_ring_commit(ring);
	if (f) {
		*f = fence_create(ring, seq);
		if (!*f)
			return -ENOMEM;
	}
	return 0;
}

int gpud_device_ib_tests(struct amdgpu_device *adev, long timeout_ms)
{
	int failed = 0;

	for (unsigned i = 0; i < adev->num_rings; i++) {
		struct amdgpu_ring *ring = adev->rings[i];
		u64 t0 = lkpi_ktime_ns();
		int r;

		if (!ring->sched.ready)
			continue;
		r = amdgpu_ring_test_ib(ring, timeout_ms);
		if (r) {
			dev_err(adev->dev, "IB test on %s failed (%d)\n", ring->name, r);
			ring->sched.ready = false;
			failed++;
		} else {
			dev_info(adev->dev, "IB test on %s succeeded in %llu us\n", ring->name,
				 (unsigned long long)(lkpi_ktime_ns() - t0) / 1000);
		}
	}
	return failed;
}

/* ---- interrupts (Linux 7.3 amdgpu_irq.c, without KFD, irq domains and request_irq) ---- */
int amdgpu_irq_init(struct amdgpu_device *adev)
{
	spin_lock_init(&adev->irq.lock);
	adev->irq.msi_enabled = true;
	adev->irq.installed = true;
	return 0;
}

void amdgpu_irq_fini_sw(struct amdgpu_device *adev)
{
	for (unsigned i = 0; i < AMDGPU_IRQ_CLIENTID_MAX; i++) {
		free(adev->irq.client[i].sources);
		adev->irq.client[i].sources = NULL;
	}
	adev->irq.installed = false;
}

int amdgpu_irq_add_id(struct amdgpu_device *adev, unsigned int client_id, unsigned int src_id,
		      struct amdgpu_irq_src *source)
{
	if (client_id >= AMDGPU_IRQ_CLIENTID_MAX || src_id >= AMDGPU_MAX_IRQ_SRC_ID || !source->funcs)
		return -EINVAL;
	if (!adev->irq.client[client_id].sources) {
		adev->irq.client[client_id].sources = calloc(AMDGPU_MAX_IRQ_SRC_ID, sizeof(struct amdgpu_irq_src *));
		if (!adev->irq.client[client_id].sources)
			return -ENOMEM;
	}
	if (adev->irq.client[client_id].sources[src_id] != NULL)
		return -EINVAL;
	if (source->num_types && !source->enabled_types) {
		source->enabled_types = calloc(source->num_types, sizeof(atomic_t));
		if (!source->enabled_types)
			return -ENOMEM;
	}
	adev->irq.client[client_id].sources[src_id] = source;
	return 0;
}

bool amdgpu_irq_enabled(struct amdgpu_device *adev, struct amdgpu_irq_src *src, unsigned int type)
{
	if (!adev->irq.installed || type >= src->num_types || !src->enabled_types || !src->funcs->set)
		return false;
	return !!atomic_read(&src->enabled_types[type]);
}

int amdgpu_irq_update(struct amdgpu_device *adev, struct amdgpu_irq_src *src, unsigned int type)
{
	enum amdgpu_interrupt_state state;
	unsigned long flags;
	int r;

	spin_lock_irqsave(&adev->irq.lock, flags);
	state = amdgpu_irq_enabled(adev, src, type) ? AMDGPU_IRQ_STATE_ENABLE : AMDGPU_IRQ_STATE_DISABLE;
	r = src->funcs->set(adev, src, type, state);
	spin_unlock_irqrestore(&adev->irq.lock, flags);
	return r;
}

int amdgpu_irq_get(struct amdgpu_device *adev, struct amdgpu_irq_src *src, unsigned int type)
{
	if (!adev->irq.installed)
		return -ENOENT;
	if (type >= src->num_types || !src->enabled_types || !src->funcs->set)
		return -EINVAL;
	if (atomic_inc_return(&src->enabled_types[type]) == 1)
		return amdgpu_irq_update(adev, src, type);
	return 0;
}

int amdgpu_irq_put(struct amdgpu_device *adev, struct amdgpu_irq_src *src, unsigned int type)
{
	if (!adev->irq.installed)
		return -ENOENT;
	if (type >= src->num_types || !src->enabled_types || !src->funcs->set)
		return -EINVAL;
	if (WARN_ON(!amdgpu_irq_enabled(adev, src, type)))
		return -EINVAL;
	if (atomic_dec_and_test(&src->enabled_types[type]))
		return amdgpu_irq_update(adev, src, type);
	return 0;
}

void amdgpu_irq_dispatch(struct amdgpu_device *adev, struct amdgpu_ih_ring *ih)
{
	u32 ring_index = ih->rptr >> 2;
	struct amdgpu_iv_entry entry;
	struct amdgpu_irq_src *src;
	int r;

	entry.ih = ih;
	entry.iv_entry = (const uint32_t *)&ih->ring[ring_index];
	entry.timestamp = 0;
	entry.timestamp_src = 0;
	amdgpu_ih_decode_iv(adev, &entry);

	if (entry.client_id >= AMDGPU_IRQ_CLIENTID_MAX || entry.src_id >= AMDGPU_MAX_IRQ_SRC_ID) {
		dev_dbg(adev->dev, "invalid IV client %u src %u\n", entry.client_id, entry.src_id);
	} else if (adev->irq.client[entry.client_id].sources &&
		   (src = adev->irq.client[entry.client_id].sources[entry.src_id])) {
		r = src->funcs->process(adev, src, &entry);
		if (r < 0)
			dev_err(adev->dev, "error processing interrupt (%d)\n", r);
	} else {
		dev_dbg(adev->dev, "unregistered interrupt client %u src %u\n", entry.client_id, entry.src_id);
	}
}

void amdgpu_irq_delegate(struct amdgpu_device *adev, struct amdgpu_iv_entry *entry, unsigned int num_dw)
{
	amdgpu_ih_ring_write(adev, &adev->irq.ih_soft, entry->iv_entry, num_dw);
	amdgpu_ih_process(adev, &adev->irq.ih_soft);
}

int gpud_irq_process(struct amdgpu_device *adev)
{
	return amdgpu_ih_process(adev, &adev->irq.ih);
}

/* ---- GFX helpers (Linux 7.3 amdgpu_gfx.c, amdgpu_rlc.c) ---- */
u32 amdgpu_gfx_create_bitmask(u32 bit_width) { return (u32)((1ULL << bit_width) - 1); }

void amdgpu_gfx_parse_disable_cu(struct amdgpu_device *adev, unsigned int *mask, unsigned int max_se,
				 unsigned int max_sh)
{
	(void)adev;
	memset(mask, 0, sizeof(*mask) * max_se * max_sh);
}

int amdgpu_gfx_get_num_kcq(struct amdgpu_device *adev)
{
	(void)adev;
	return amdgpu_num_kcq == -1 ? 8 : amdgpu_num_kcq;
}

u32 amdgpu_gfx_csb_preamble_start(u32 *buffer)
{
	u32 count = 0;

	buffer[count++] = cpu_to_le32(PACKET3(PACKET3_PREAMBLE_CNTL, 0));
	buffer[count++] = cpu_to_le32(PACKET3_PREAMBLE_BEGIN_CLEAR_STATE);
	buffer[count++] = cpu_to_le32(PACKET3(PACKET3_CONTEXT_CONTROL, 1));
	buffer[count++] = cpu_to_le32(0x80000000);
	buffer[count++] = cpu_to_le32(0x80000000);
	return count;
}

u32 amdgpu_gfx_csb_data_parser(struct amdgpu_device *adev, u32 *buffer, u32 count)
{
	const struct cs_section_def *sect = NULL;
	const struct cs_extent_def *ext = NULL;
	u32 i;

	for (sect = adev->gfx.rlc.cs_data; sect->section != NULL; ++sect) {
		for (ext = sect->section; ext->extent != NULL; ++ext) {
			if (sect->id == SECT_CONTEXT) {
				buffer[count++] = cpu_to_le32(PACKET3(PACKET3_SET_CONTEXT_REG, ext->reg_count));
				buffer[count++] = cpu_to_le32(ext->reg_index - PACKET3_SET_CONTEXT_REG_START);
				for (i = 0; i < ext->reg_count; i++)
					buffer[count++] = cpu_to_le32(ext->extent[i]);
			}
		}
	}
	return count;
}

void amdgpu_gfx_csb_preamble_end(u32 *buffer, u32 count)
{
	buffer[count++] = cpu_to_le32(PACKET3(PACKET3_PREAMBLE_CNTL, 0));
	buffer[count++] = cpu_to_le32(PACKET3_PREAMBLE_END_CLEAR_STATE);
	buffer[count++] = cpu_to_le32(PACKET3(PACKET3_CLEAR_STATE, 0));
	buffer[count++] = cpu_to_le32(0);
}

int amdgpu_gfx_rlc_init_sr(struct amdgpu_device *adev, u32 dws)
{
	const u32 *src_ptr;
	u32 *dst_ptr;
	u32 i;
	int r;

	r = amdgpu_bo_create_reserved(adev, dws * 4, PAGE_SIZE, AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GTT,
				      &adev->gfx.rlc.save_restore_obj, &adev->gfx.rlc.save_restore_gpu_addr,
				      (void **)&adev->gfx.rlc.sr_ptr);
	if (r) {
		dev_warn(adev->dev, "(%d) create RLC sr bo failed\n", r);
		amdgpu_gfx_rlc_fini(adev);
		return r;
	}
	src_ptr = adev->gfx.rlc.reg_list;
	dst_ptr = adev->gfx.rlc.sr_ptr;
	for (i = 0; i < adev->gfx.rlc.reg_list_size; i++)
		dst_ptr[i] = cpu_to_le32(src_ptr[i]);
	return 0;
}

void amdgpu_gfx_rlc_fini(struct amdgpu_device *adev)
{
	if (adev->gfx.rlc.save_restore_obj)
		amdgpu_bo_free_kernel(&adev->gfx.rlc.save_restore_obj, &adev->gfx.rlc.save_restore_gpu_addr,
				      (void **)&adev->gfx.rlc.sr_ptr);
	amdgpu_bo_free_kernel(&adev->gfx.rlc.clear_state_obj, &adev->gfx.rlc.clear_state_gpu_addr,
			      (void **)&adev->gfx.rlc.cs_ptr);
	amdgpu_bo_free_kernel(&adev->gfx.rlc.cp_table_obj, &adev->gfx.rlc.cp_table_gpu_addr,
			      (void **)&adev->gfx.rlc.cp_table_ptr);
}

/* ---- microcode (Linux 7.3 amdgpu_ucode.c) ---- */
int amdgpu_ucode_request(struct amdgpu_device *adev, const struct firmware **fw, enum amdgpu_ucode_required required,
			 const char *fmt, ...)
{
	const struct common_firmware_header *hdr;
	char fname[128];
	va_list ap;
	int r;

	(void)required;
	va_start(ap, fmt);
	r = vsnprintf(fname, sizeof(fname), fmt, ap);
	va_end(ap);
	if (r >= (int)sizeof(fname))
		return -EOVERFLOW;
	r = request_firmware(fw, fname, adev->dev);
	if (r)
		return -ENODEV;
	hdr = (const struct common_firmware_header *)(*fw)->data;
	if ((*fw)->size != le32_to_cpu(hdr->size_bytes)) {
		dev_err(adev->dev, "\"%s\" failed to validate\n", fname);
		return -EINVAL;
	}
	return 0;
}

void amdgpu_ucode_release(const struct firmware **fw)
{
	release_firmware(*fw);
	*fw = NULL;
}

void amdgpu_ucode_print_mc_hdr(const struct common_firmware_header *hdr) { (void)hdr; }
void amdgpu_ucode_print_gfx_hdr(const struct common_firmware_header *hdr) { (void)hdr; }
void amdgpu_ucode_print_rlc_hdr(const struct common_firmware_header *hdr) { (void)hdr; }
