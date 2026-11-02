// SPDX-License-Identifier: MIT
/*
 * The runtime behind lkpi.h: logging, time and delays, firmware files, PCI config space and DMA memory through the
 * gpud platform callbacks, and dma_fence.
 */
#include <lkpi.h>
#include <lkpi_drm.h>
#include <gpud_core.h>
#include <linux/pci.h>

#include <time.h>

/* One process serves one platform; the callbacks are set by gpud_device_create. */
const struct gpud_platform *lkpi_platform;
const char *lkpi_firmware_dir = "/lib/firmware";
int lkpi_log_level = LKPI_INFO;

void lkpi_log(int level, const char *fmt, ...)
{
	char buf[512];
	va_list ap;
	size_t n;

	if (level > lkpi_log_level)
		return;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	n = strlen(buf);
	if (n && buf[n - 1] == '\n')
		buf[n - 1] = 0;
	if (lkpi_platform && lkpi_platform->log)
		lkpi_platform->log(lkpi_platform->ctx, level, buf);
	else
		fprintf(stderr, "gpud[%d] %s\n", level, buf);
}

u64 lkpi_ktime_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (u64)ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

/* Delays below a millisecond spin (register polling expects them to be short and exact); longer ones sleep. */
void lkpi_udelay(unsigned long us)
{
	u64 end = lkpi_ktime_ns() + (u64)us * 1000;

	if (us >= 1000) {
		struct timespec ts = { .tv_sec = us / 1000000, .tv_nsec = (us % 1000000) * 1000 };

		nanosleep(&ts, NULL);
		return;
	}
	while (lkpi_ktime_ns() < end)
		__asm__ __volatile__("pause" ::: "memory");
}

void lkpi_msleep(unsigned long ms)
{
	struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000 };

	nanosleep(&ts, NULL);
}

size_t strscpy(char *dst, const char *src, size_t n)
{
	size_t len = strnlen(src, n);

	if (!n)
		return -E2BIG;
	if (len == n) {
		memcpy(dst, src, n - 1);
		dst[n - 1] = 0;
		return -E2BIG;
	}
	memcpy(dst, src, len + 1);
	return len;
}

int scnprintf(char *buf, size_t size, const char *fmt, ...)
{
	va_list ap;
	int n;

	if (!size)
		return 0;
	va_start(ap, fmt);
	n = vsnprintf(buf, size, fmt, ap);
	va_end(ap);
	return n < 0 ? 0 : (size_t)n >= size ? (int)size - 1 : n;
}

void init_waitqueue_head(wait_queue_head_t *q)
{
	pthread_mutex_init(&q->m, NULL);
	pthread_cond_init(&q->c, NULL);
}

void wake_up_all(wait_queue_head_t *q)
{
	pthread_mutex_lock(&q->m);
	pthread_cond_broadcast(&q->c);
	pthread_mutex_unlock(&q->m);
}

/* ---- firmware: files under the firmware directory ---- */
int request_firmware(const struct firmware **out, const char *name, struct device *dev)
{
	char path[512];
	struct firmware *fw;
	FILE *f;
	long n;
	u8 *data;

	(void)dev;
	*out = NULL;
	snprintf(path, sizeof(path), "%s/%s", lkpi_firmware_dir, name);
	f = fopen(path, "rb");
	if (!f) {
		lkpi_log(LKPI_ERR, "firmware %s: %s\n", path, strerror(errno));
		return -ENOENT;
	}
	fseek(f, 0, SEEK_END);
	n = ftell(f);
	fseek(f, 0, SEEK_SET);
	fw = calloc(1, sizeof(*fw));
	data = malloc(n > 0 ? n : 1);
	if (!fw || !data || fread(data, 1, n, f) != (size_t)n) {
		fclose(f);
		free(fw);
		free(data);
		return -EIO;
	}
	fclose(f);
	fw->size = n;
	fw->data = data;
	*out = fw;
	lkpi_log(LKPI_INFO, "firmware %s: %ld bytes\n", path, n);
	return 0;
}

void release_firmware(const struct firmware *fw)
{
	if (!fw)
		return;
	free((void *)fw->data);
	free((void *)fw);
}

/* ---- PCI config space through the platform ---- */
static int cfg_read(const struct pci_dev *dev, int where, int size, u32 *val)
{
	if (!dev->platform || !dev->platform->config_read)
		return -ENODEV;
	return dev->platform->config_read(dev->platform->ctx, where, size, val);
}

static int cfg_write(const struct pci_dev *dev, int where, int size, u32 val)
{
	if (!dev->platform || !dev->platform->config_write)
		return -ENODEV;
	return dev->platform->config_write(dev->platform->ctx, where, size, val);
}

int pci_read_config_byte(const struct pci_dev *dev, int where, u8 *val)
{
	u32 v = 0;
	int r = cfg_read(dev, where, 1, &v);

	*val = v;
	return r;
}

int pci_read_config_word(const struct pci_dev *dev, int where, u16 *val)
{
	u32 v = 0;
	int r = cfg_read(dev, where, 2, &v);

	*val = v;
	return r;
}

int pci_read_config_dword(const struct pci_dev *dev, int where, u32 *val) { return cfg_read(dev, where, 4, val); }
int pci_write_config_byte(const struct pci_dev *dev, int where, u8 val) { return cfg_write(dev, where, 1, val); }
int pci_write_config_word(const struct pci_dev *dev, int where, u16 val) { return cfg_write(dev, where, 2, val); }
int pci_write_config_dword(const struct pci_dev *dev, int where, u32 val) { return cfg_write(dev, where, 4, val); }

void pci_set_master(struct pci_dev *dev)
{
	u16 cmd;

	pci_read_config_word(dev, PCI_COMMAND, &cmd);
	if (!(cmd & PCI_COMMAND_MASTER))
		pci_write_config_word(dev, PCI_COMMAND, cmd | PCI_COMMAND_MASTER);
}

void pci_clear_master(struct pci_dev *dev)
{
	u16 cmd;

	pci_read_config_word(dev, PCI_COMMAND, &cmd);
	pci_write_config_word(dev, PCI_COMMAND, cmd & ~PCI_COMMAND_MASTER);
}

int pci_find_capability(struct pci_dev *dev, int cap)
{
	u8 pos = 0, id;
	u16 status;

	pci_read_config_word(dev, PCI_STATUS, &status);
	if (!(status & PCI_STATUS_CAP_LIST))
		return 0;
	pci_read_config_byte(dev, PCI_CAPABILITY_LIST, &pos);
	for (int i = 0; i < 48 && pos >= 0x40; i++) {
		pos &= ~3;
		pci_read_config_byte(dev, pos, &id);
		if (id == cap)
			return pos;
		pci_read_config_byte(dev, pos + 1, &pos);
	}
	return 0;
}

int pci_pcie_cap(struct pci_dev *dev) { return pci_find_capability(dev, PCI_CAP_ID_EXP); }

int pcie_capability_read_word(struct pci_dev *dev, int pos, u16 *val)
{
	int cap = pci_pcie_cap(dev);

	*val = 0;
	return cap ? pci_read_config_word(dev, cap + pos, val) : -EINVAL;
}

int pcie_capability_read_dword(struct pci_dev *dev, int pos, u32 *val)
{
	int cap = pci_pcie_cap(dev);

	*val = 0;
	return cap ? pci_read_config_dword(dev, cap + pos, val) : -EINVAL;
}

int pcie_capability_write_word(struct pci_dev *dev, int pos, u16 val)
{
	int cap = pci_pcie_cap(dev);

	return cap ? pci_write_config_word(dev, cap + pos, val) : -EINVAL;
}

int pcie_capability_clear_and_set_word_locked(struct pci_dev *dev, int pos, u16 clear, u16 set)
{
	u16 v;
	int r = pcie_capability_read_word(dev, pos, &v);

	if (r)
		return r;
	return pcie_capability_write_word(dev, pos, (v & ~clear) | set);
}

int pcie_get_readrq(struct pci_dev *dev)
{
	u16 ctl;

	pcie_capability_read_word(dev, PCI_EXP_DEVCTL, &ctl);
	return 128 << ((ctl & PCI_EXP_DEVCTL_READRQ) >> 12);
}

int pcie_set_readrq(struct pci_dev *dev, int rq)
{
	u16 v = (ffs(rq) - 8) << 12;

	if (rq < 128 || rq > 4096 || !is_power_of_2(rq))
		return -EINVAL;
	return pcie_capability_clear_and_set_word_locked(dev, PCI_EXP_DEVCTL, PCI_EXP_DEVCTL_READRQ, v);
}

/* ---- DMA memory through the platform ---- */
void *dma_alloc_coherent(struct device *dev, size_t size, dma_addr_t *dma, gfp_t gfp)
{
	void *cpu = NULL;
	u64 iova = 0;

	(void)dev;
	(void)gfp;
	size = ALIGN(size, PAGE_SIZE);
	if (!lkpi_platform || lkpi_platform->dma_alloc(lkpi_platform->ctx, size, &cpu, &iova))
		return NULL;
	*dma = iova;
	return cpu;
}

void dma_free_coherent(struct device *dev, size_t size, void *cpu, dma_addr_t dma)
{
	(void)dev;
	if (cpu && lkpi_platform)
		lkpi_platform->dma_free(lkpi_platform->ctx, cpu, dma, ALIGN(size, PAGE_SIZE));
}

/* ---- dma_fence ---- */
static pthread_mutex_t fence_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t fence_cond = PTHREAD_COND_INITIALIZER;

void dma_fence_init(struct dma_fence *f, const struct dma_fence_ops *ops, spinlock_t *lock, u64 context, u64 seqno)
{
	(void)lock;
	kref_init(&f->refcount);
	f->ops = ops;
	f->context = context;
	f->seqno = seqno;
	f->error = 0;
	f->signaled = false;
}

struct dma_fence *dma_fence_get(struct dma_fence *f)
{
	if (f)
		kref_get(&f->refcount);
	return f;
}

static void dma_fence_release(struct kref *k)
{
	struct dma_fence *f = container_of(k, struct dma_fence, refcount);

	if (f->ops && f->ops->release)
		f->ops->release(f);
	else
		free(f);
}

void dma_fence_put(struct dma_fence *f)
{
	if (f)
		kref_put(&f->refcount, dma_fence_release);
}

int dma_fence_signal(struct dma_fence *f)
{
	if (!f || f->signaled)
		return -EINVAL;
	pthread_mutex_lock(&fence_lock);
	f->timestamp = ktime_get();
	f->signaled = true;
	pthread_cond_broadcast(&fence_cond);
	pthread_mutex_unlock(&fence_lock);
	return 0;
}

bool dma_fence_is_signaled(struct dma_fence *f)
{
	if (f->signaled)
		return true;
	if (f->ops && f->ops->signaled && f->ops->signaled(f)) {
		dma_fence_signal(f);
		return true;
	}
	return false;
}

/* Fences that know how to check themselves (ops->signaled) are polled every 20 us; others wait for a signal. */
long dma_fence_wait_timeout(struct dma_fence *f, bool intr, long timeout)
{
	u64 deadline = timeout == MAX_SCHEDULE_TIMEOUT ? U64_MAX : lkpi_ktime_ns() + (u64)timeout * NSEC_PER_MSEC;

	(void)intr;
	while (!dma_fence_is_signaled(f)) {
		u64 now = lkpi_ktime_ns();

		if (now >= deadline)
			return 0;
		if (f->ops && f->ops->signaled) {
			lkpi_udelay(20);
		} else {
			struct timespec ts;

			clock_gettime(CLOCK_REALTIME, &ts);
			ts.tv_nsec += 1000000;
			if (ts.tv_nsec >= NSEC_PER_SEC) {
				ts.tv_sec++;
				ts.tv_nsec -= NSEC_PER_SEC;
			}
			pthread_mutex_lock(&fence_lock);
			if (!f->signaled)
				pthread_cond_timedwait(&fence_cond, &fence_lock, &ts);
			pthread_mutex_unlock(&fence_lock);
		}
	}
	if (timeout == MAX_SCHEDULE_TIMEOUT)
		return timeout;
	return max_t(long, 1, (long)((deadline - lkpi_ktime_ns()) / NSEC_PER_MSEC));
}

static struct dma_fence stub_fence = { .refcount = { { 1 << 30 } }, .signaled = true };

struct dma_fence *dma_fence_get_stub(void) { return dma_fence_get(&stub_fence); }

u64 dma_fence_context_alloc(unsigned int num)
{
	static atomic64_t next = { 1 };

	return atomic_add_return(num, &next) - num;
}
