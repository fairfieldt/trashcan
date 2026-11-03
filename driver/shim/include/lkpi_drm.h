/* SPDX-License-Identifier: MIT */
/*
 * The DRM-side types the imported files name: a dma_fence that can be waited on and signalled, and stand-ins for
 * the scheduler, suballocator and drm_device, which gpud doesn't use (rings are driven directly).
 */
#ifndef GPUD_LKPI_DRM_H
#define GPUD_LKPI_DRM_H

#include "lkpi.h"

struct drm_printer;

struct drm_device {
	struct device *dev;
};

#define drm_err(drm, fmt, ...) lkpi_log(LKPI_ERR, fmt, ##__VA_ARGS__)
#define drm_warn(drm, fmt, ...) lkpi_log(LKPI_WARN, fmt, ##__VA_ARGS__)
#define drm_info(drm, fmt, ...) lkpi_log(LKPI_INFO, fmt, ##__VA_ARGS__)
#define drm_dbg(drm, fmt, ...) lkpi_log(LKPI_DEBUG, fmt, ##__VA_ARGS__)
#define drm_dbg_driver drm_dbg
#define drm_err_once drm_err
#define drm_warn_once drm_warn
#define drm_info_once drm_info
#define drm_err_ratelimited drm_err
#define drm_WARN_ON(drm, c) WARN_ON(c)
#define drm_WARN_ON_ONCE(drm, c) WARN_ON(c)
#define drm_WARN(drm, c, fmt, ...) WARN(c, fmt, ##__VA_ARGS__)
#define DRM_ERROR(fmt, ...) lkpi_log(LKPI_ERR, fmt, ##__VA_ARGS__)
#define DRM_WARN(fmt, ...) lkpi_log(LKPI_WARN, fmt, ##__VA_ARGS__)
#define DRM_INFO(fmt, ...) lkpi_log(LKPI_INFO, fmt, ##__VA_ARGS__)
#define DRM_NOTE(fmt, ...) lkpi_log(LKPI_NOTICE, fmt, ##__VA_ARGS__)
#define DRM_DEBUG(fmt, ...) lkpi_log(LKPI_DEBUG, fmt, ##__VA_ARGS__)
#define DRM_DEBUG_DRIVER DRM_DEBUG
#define DRM_DEBUG_KMS DRM_DEBUG
#define DRM_ERROR_RATELIMITED DRM_ERROR
#define DRM_INFO_ONCE DRM_INFO
#define DRM_DEV_ERROR(dev, fmt, ...) lkpi_log(LKPI_ERR, fmt, ##__VA_ARGS__)
#define DRM_DEV_INFO(dev, fmt, ...) lkpi_log(LKPI_INFO, fmt, ##__VA_ARGS__)
#define DRM_DEV_DEBUG(dev, fmt, ...) lkpi_log(LKPI_DEBUG, fmt, ##__VA_ARGS__)
#define DRM_DEV_DEBUG_DRIVER DRM_DEV_DEBUG

static inline bool drm_need_swiotlb(int dma_bits) { (void)dma_bits; return false; }
static inline bool drm_dev_enter(struct drm_device *d, int *idx) { (void)d; *idx = 0; return true; }
static inline void drm_dev_exit(int idx) { (void)idx; }

/* ---- dma_fence: refcounted, signalled once, waitable with a timeout ---- */
struct dma_fence;
struct dma_fence_ops {
	const char *(*get_driver_name)(struct dma_fence *);
	const char *(*get_timeline_name)(struct dma_fence *);
	bool (*signaled)(struct dma_fence *);
	void (*release)(struct dma_fence *);
};
struct dma_fence {
	struct kref refcount;
	const struct dma_fence_ops *ops;
	u64 context;
	u64 seqno;
	int error;
	volatile bool signaled;
	ktime_t timestamp;
};
void dma_fence_init(struct dma_fence *f, const struct dma_fence_ops *ops, spinlock_t *lock, u64 context, u64 seqno);
struct dma_fence *dma_fence_get(struct dma_fence *f);
void dma_fence_put(struct dma_fence *f);
int dma_fence_signal(struct dma_fence *f);
bool dma_fence_is_signaled(struct dma_fence *f);
long dma_fence_wait_timeout(struct dma_fence *f, bool intr, long timeout_jiffies);
struct dma_fence *dma_fence_get_stub(void);
u64 dma_fence_context_alloc(unsigned int num);
static inline int dma_fence_wait(struct dma_fence *f, bool intr)
{
	long r = dma_fence_wait_timeout(f, intr, MAX_SCHEDULE_TIMEOUT);
	return r < 0 ? (int)r : 0;
}
static inline void dma_fence_set_error(struct dma_fence *f, int error) { f->error = error; }
#define dma_fence_get_rcu(f) dma_fence_get(f)

/* ---- the scheduler and suballocator: never run in gpud ---- */
struct drm_gpu_scheduler {
	bool ready;
	const char *name;
};
struct drm_sched_entity { int dummy; };
struct drm_sched_job { int dummy; };
struct drm_sched_fence { struct dma_fence finished; };
struct drm_sched_backend_ops { int dummy; };
static inline void drm_sched_fault(struct drm_gpu_scheduler *s) { (void)s; }
struct drm_suballoc { u64 soffset; u64 eoffset; void *manager; };
struct drm_suballoc_manager { int dummy; };
static inline u64 drm_suballoc_soffset(struct drm_suballoc *sa) { return sa->soffset; }

static inline bool drm_can_sleep(void) { return true; }

#endif
