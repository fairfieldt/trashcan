// SPDX-License-Identifier: MIT
/*
 * libgpud_drm.so: puts gpud behind an unmodified Mesa RADV and libdrm. Loaded with LD_PRELOAD, it
 *
 * - connects to every gpud in /run/gpud (one ADDRESS.sock per GPU, in PCI order) and answers libdrm's device
 *   enumeration with one AMD device per GPU, whose render node is the client's buffer file from that gpud (a memfd,
 *   opened as /proc/self/fd/N), so RADV opens it and later mmap()s host-visible buffers out of it;
 * - intercepts ioctl() on that file and implements the DRM and AMDGPU ioctls RADV uses (version, caps, GEM close,
 *   syncobjs, and AMDGPU info, GEM create/mmap/VA/metadata/op, context, CS, fence waits, VM, fence-to-handle), with
 *   buffer objects, address spaces and submission in gpud and syncobjs and contexts here.
 *
 * Everything a DRM file holds in the kernel (buffers, syncobjs, contexts) is per device here too. Every other fd and
 * call goes to the real libc and libdrm. GPUD_SOCKET (paths separated by ':') overrides the sockets; GPUD_DRM_DEBUG=1
 * logs each ioctl.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>

#include "gpud_proto.h"

/* ---- libdrm's device description (xf86drm.h, stable ABI) ---- */
#define DRM_NODE_PRIMARY 0
#define DRM_NODE_CONTROL 1
#define DRM_NODE_RENDER 2
#define DRM_NODE_MAX 3
#define DRM_BUS_PCI 0
typedef struct {
	uint16_t domain;
	uint8_t bus, dev, func;
} drmPciBusInfo;
typedef struct {
	uint16_t vendor_id, device_id, subvendor_id, subdevice_id;
	uint8_t revision_id;
} drmPciDeviceInfo;
typedef struct _drmDevice {
	char **nodes;
	int available_nodes;
	int bustype;
	union {
		drmPciBusInfo *pci;
		void *other;
	} businfo;
	union {
		drmPciDeviceInfo *pci;
		void *other;
	} deviceinfo;
} drmDevice, *drmDevicePtr;

/* Our drmDevice and everything it points to, in one allocation. */
struct gpud_device {
	drmDevice dev;
	char *nodes[DRM_NODE_MAX];
	drmPciBusInfo bus;
	drmPciDeviceInfo info;
	char path[64];
	uint32_t magic;
};
#define GPUD_DEVICE_MAGIC 0x67707564

struct bo;
struct syncobj;

/* One GPU: its gpud connection and what a DRM file would hold in the kernel. */
struct gdev {
	int sock;
	pthread_mutex_t conn_lock; /* one request at a time on the socket */
	int bo_fd;
	dev_t bo_dev;
	ino_t bo_ino;
	struct gpud_hello_rep hello;
	char bo_path[64];
	struct bo *bos;
	uint32_t nbos;
	pthread_mutex_t bo_lock;
	struct syncobj *syncobjs;
	uint32_t nsyncobjs;
	pthread_mutex_t sync_lock;
	pthread_cond_t sync_cond;
	uint32_t next_ctx;
};
#define GPUD_MAX_DEVICES 8
static struct gdev devs[GPUD_MAX_DEVICES];
static int ndevs;

static bool debug;

/* ioctl's request type: unsigned long in glibc, int in musl (dross runs Alpine's musl builds of Mesa). */
#ifdef __GLIBC__
typedef unsigned long ioctl_req;
#else
typedef int ioctl_req;
#endif

static int (*real_ioctl)(int, ioctl_req, ...);

#define LOG(...) do { if (debug) fprintf(stderr, "gpud_drm: " __VA_ARGS__); } while (0)

/* ---- the connection ---- */
static int rpc(struct gdev *g, uint32_t op, const void *req, uint32_t req_len, const void *req2, uint32_t req2_len,
	       void *rep, uint32_t rep_len, int *fd_out)
{
	char buf[GPUD_MAX_MSG];
	struct gpud_req *h = (struct gpud_req *)buf;
	struct gpud_rep *rh = (struct gpud_rep *)buf;
	char cbuf[CMSG_SPACE(sizeof(int))];
	struct iovec iov = { buf, sizeof(buf) };
	struct msghdr msg = { .msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf, .msg_controllen = sizeof(cbuf) };
	ssize_t n;

	if (sizeof(*h) + req_len + req2_len > sizeof(buf))
		return -E2BIG;
	h->op = op;
	h->len = req_len + req2_len;
	memcpy(buf + sizeof(*h), req, req_len);
	if (req2_len)
		memcpy(buf + sizeof(*h) + req_len, req2, req2_len);
	pthread_mutex_lock(&g->conn_lock);
	if (send(g->sock, buf, sizeof(*h) + h->len, MSG_NOSIGNAL) < 0) {
		pthread_mutex_unlock(&g->conn_lock);
		return -EIO;
	}
	n = recvmsg(g->sock, &msg, MSG_CMSG_CLOEXEC);
	pthread_mutex_unlock(&g->conn_lock);
	if (n < (ssize_t)sizeof(*rh))
		return -EIO;
	if (fd_out) {
		struct cmsghdr *c = CMSG_FIRSTHDR(&msg);

		*fd_out = c && c->cmsg_type == SCM_RIGHTS ? *(int *)CMSG_DATA(c) : -1;
	}
	if (rh->status == 0 && rep)
		memcpy(rep, buf + sizeof(*rh), rh->len < rep_len ? rh->len : rep_len);
	return rh->status;
}

static pthread_once_t connect_once = PTHREAD_ONCE_INIT;

static bool connect_one(struct gdev *g, const char *path)
{
	struct sockaddr_un addr = { .sun_family = AF_UNIX };
	struct stat st;
	int fd;

	snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);
	g->sock = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
	if (g->sock < 0 || connect(g->sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		fprintf(stderr, "gpud_drm: can't connect to gpud at %s: %s\n", path, strerror(errno));
		goto fail;
	}
	pthread_mutex_init(&g->conn_lock, NULL);
	pthread_mutex_init(&g->bo_lock, NULL);
	pthread_mutex_init(&g->sync_lock, NULL);
	pthread_cond_init(&g->sync_cond, NULL);
	g->next_ctx = 1;
	if (rpc(g, GPUD_HELLO, NULL, 0, NULL, 0, &g->hello, sizeof(g->hello), &fd) || fd < 0 ||
	    g->hello.version != GPUD_PROTO_VERSION) {
		fprintf(stderr, "gpud_drm: gpud handshake failed on %s\n", path);
		goto fail;
	}
	/* Out of the way of the application's own low fds. */
	g->bo_fd = fcntl(fd, F_DUPFD_CLOEXEC, 1000);
	close(fd);
	fstat(g->bo_fd, &st);
	g->bo_dev = st.st_dev;
	g->bo_ino = st.st_ino;
	snprintf(g->bo_path, sizeof(g->bo_path), "/proc/self/fd/%d", g->bo_fd);
	LOG("connected to %s: VMID %u, buffer file %s (%llu GiB)\n", path, g->hello.vmid, g->bo_path,
	    (unsigned long long)(g->hello.bo_file_size >> 30));
	return true;
fail:
	if (g->sock >= 0)
		close(g->sock);
	g->sock = -1;
	return false;
}

/* Every gpud socket, in PCI address order (their names), or GPUD_SOCKET's list. */
static void do_connect(void)
{
	const char *env = getenv("GPUD_SOCKET");
	glob_t gl = { 0 };

	debug = getenv("GPUD_DRM_DEBUG") && *getenv("GPUD_DRM_DEBUG") == '1';
	if (env) {
		char *list = strdup(env), *save = NULL;

		for (char *p = strtok_r(list, ":", &save); p && ndevs < GPUD_MAX_DEVICES; p = strtok_r(NULL, ":", &save))
			if (connect_one(&devs[ndevs], p))
				ndevs++;
		free(list);
		return;
	}
	if (glob(GPUD_SOCKET_DIR "/*.sock", 0, NULL, &gl) == 0)
		for (size_t i = 0; i < gl.gl_pathc && ndevs < GPUD_MAX_DEVICES; i++)
			if (connect_one(&devs[ndevs], gl.gl_pathv[i]))
				ndevs++;
	globfree(&gl);
	if (!ndevs)
		fprintf(stderr, "gpud_drm: no gpud in %s\n", GPUD_SOCKET_DIR);
}

static bool connected(void)
{
	pthread_once(&connect_once, do_connect);
	return ndevs > 0;
}

/* The device whose buffer file fd is, or NULL. */
static struct gdev *dev_of(int fd)
{
	struct stat st;

	if (fd < 0 || !ndevs || fstat(fd, &st) != 0)
		return NULL;
	for (int i = 0; i < ndevs; i++)
		if (st.st_ino == devs[i].bo_ino && st.st_dev == devs[i].bo_dev)
			return &devs[i];
	return NULL;
}

static int64_t now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

/* Waits for a fence in slices, so other threads can use the socket in between. 0, or -ETIME at deadline. */
static int fence_wait(struct gdev *g, uint64_t fence, int64_t deadline)
{
	for (;;) {
		int64_t left = deadline - now_ns();
		struct gpud_wait_req w = { .fence = fence, .timeout_ns = left < 0 ? 0 : left > 2000000 ? 2000000 : left };
		int r = rpc(g, GPUD_WAIT, &w, sizeof(w), NULL, 0, NULL, 0, NULL);

		if (r != -ETIME || left <= 0)
			return r;
	}
}

/* ---- buffer objects (client-side cache of what gpud reported) ---- */
struct bo {
	struct gpud_bo_rep rep;
	struct drm_amdgpu_gem_metadata meta;
	bool used;
};
static struct bo *bo_get(struct gdev *g, uint32_t handle)
{
	return handle < g->nbos && g->bos[handle].used ? &g->bos[handle] : NULL;
}

/* ---- g->syncobjs: a fence id, signalled, or nothing yet ---- */
struct syncobj {
	bool used, signaled, has_fence;
	uint64_t fence;
};
static int syncobj_new(struct gdev *g, bool signaled, uint64_t fence, bool has_fence, uint32_t *handle)
{
	uint32_t h;

	pthread_mutex_lock(&g->sync_lock);
	for (h = 1; h < g->nsyncobjs && g->syncobjs[h].used; h++)
		;
	if (h >= g->nsyncobjs) {
		uint32_t n = g->nsyncobjs ? g->nsyncobjs * 2 : 64;
		struct syncobj *s = realloc(g->syncobjs, n * sizeof(*s));

		if (!s) {
			pthread_mutex_unlock(&g->sync_lock);
			return -ENOMEM;
		}
		memset(s + g->nsyncobjs, 0, (n - g->nsyncobjs) * sizeof(*s));
		g->syncobjs = s;
		h = g->nsyncobjs ? g->nsyncobjs : 1;
		g->nsyncobjs = n;
	}
	g->syncobjs[h] = (struct syncobj){ .used = true, .signaled = signaled, .has_fence = has_fence, .fence = fence };
	pthread_mutex_unlock(&g->sync_lock);
	*handle = h;
	return 0;
}

static struct syncobj *syncobj_get(struct gdev *g, uint32_t h)
{
	return h < g->nsyncobjs && g->syncobjs[h].used ? &g->syncobjs[h] : NULL;
}

static void syncobj_set_fence(struct gdev *g, uint32_t h, uint64_t fence)
{
	pthread_mutex_lock(&g->sync_lock);
	struct syncobj *s = syncobj_get(g, h);

	if (s) {
		s->has_fence = true;
		s->signaled = false;
		s->fence = fence;
	}
	pthread_cond_broadcast(&g->sync_cond);
	pthread_mutex_unlock(&g->sync_lock);
}

/*
 * The state of one syncobj for a wait: 1 signalled, 0 pending on *fence, -EINVAL no fence and not waiting for one,
 * or -EAGAIN no fence yet (WAIT_FOR_SUBMIT).
 */
static int syncobj_state(struct gdev *g, uint32_t h, uint32_t flags, uint64_t *fence)
{
	struct syncobj *s = syncobj_get(g, h);

	if (!s)
		return -ENOENT;
	if (s->signaled)
		return 1;
	if (s->has_fence) {
		*fence = s->fence;
		return (flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE) ? 1 : 0;
	}
	return (flags & (DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE)) ? -EAGAIN
													       : -EINVAL;
}

static int syncobj_wait(struct gdev *g, const uint32_t *handles, uint32_t count, int64_t deadline, uint32_t flags,
			uint32_t *first)
{
	bool all = flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL;

	for (;;) {
		uint32_t done = 0;
		bool need_submit = false;

		for (uint32_t i = 0; i < count; i++) {
			uint64_t fence = 0;
			int st;

			pthread_mutex_lock(&g->sync_lock);
			st = syncobj_state(g, handles[i], flags, &fence);
			pthread_mutex_unlock(&g->sync_lock);
			if (st == -EAGAIN) {
				need_submit = true;
				continue;
			}
			if (st < 0)
				return st;
			if (st == 0) {
				struct gpud_wait_req w = { .fence = fence, .timeout_ns = 0 };

				st = rpc(g, GPUD_WAIT, &w, sizeof(w), NULL, 0, NULL, 0, NULL) == 0;
			}
			if (st) {
				done++;
				if (!all) {
					if (first)
						*first = i;
					return 0;
				}
			}
		}
		if (done == count)
			return 0;
		if (now_ns() >= deadline)
			return -ETIME;
		if (need_submit) {
			struct timespec ts;

			clock_gettime(CLOCK_REALTIME, &ts);
			ts.tv_nsec += 200000;
			if (ts.tv_nsec >= 1000000000) {
				ts.tv_sec++;
				ts.tv_nsec -= 1000000000;
			}
			pthread_mutex_lock(&g->sync_lock);
			pthread_cond_timedwait(&g->sync_cond, &g->sync_lock, &ts);
			pthread_mutex_unlock(&g->sync_lock);
		} else {
			usleep(20);
		}
	}
}

static int64_t abs_timeout(int64_t t)
{
	/* DRM timeouts are absolute CLOCK_MONOTONIC nanoseconds. */
	return t;
}

/* ---- the ioctls ---- */
static int drm_version(struct drm_version *v)
{
	static const char name[] = "amdgpu", date[] = "20150101", desc[] = "AMD GPU (gpud)";

	v->version_major = 3;
	v->version_minor = 64;
	v->version_patchlevel = 0;
#define COPY(field, s)                                                                \
	do {                                                                          \
		if (v->field && v->field##_len)                                       \
			memcpy(v->field, s, v->field##_len < sizeof(s) ? v->field##_len : sizeof(s)); \
		v->field##_len = sizeof(s) - 1;                                       \
	} while (0)
	COPY(name, name);
	COPY(date, date);
	COPY(desc, desc);
#undef COPY
	return 0;
}

static int get_cap(struct drm_get_cap *c)
{
	switch (c->capability) {
	case DRM_CAP_SYNCOBJ:
		c->value = 1;
		return 0;
	case DRM_CAP_SYNCOBJ_TIMELINE:
	case DRM_CAP_PRIME:
		c->value = 0;
		return 0;
	default:
		return -EINVAL;
	}
}

static int gem_create(struct gdev *g, union drm_amdgpu_gem_create *a)
{
	struct gpud_bo_create_req req = {
		.size = a->in.bo_size,
		.alignment = a->in.alignment,
		.domains = a->in.domains,
		.flags = a->in.domain_flags,
	};
	struct gpud_bo_rep rep;
	int r = rpc(g, GPUD_BO_CREATE, &req, sizeof(req), NULL, 0, &rep, sizeof(rep), NULL);

	if (r)
		return r;
	pthread_mutex_lock(&g->bo_lock);
	if (rep.handle >= g->nbos) {
		uint32_t n = rep.handle * 2 + 64;
		struct bo *b = realloc(g->bos, n * sizeof(*b));

		if (!b) {
			pthread_mutex_unlock(&g->bo_lock);
			return -ENOMEM;
		}
		memset(b + g->nbos, 0, (n - g->nbos) * sizeof(*b));
		g->bos = b;
		g->nbos = n;
	}
	g->bos[rep.handle] = (struct bo){ .rep = rep, .used = true };
	pthread_mutex_unlock(&g->bo_lock);
	LOG("BO %u: %llu bytes, domains %#llx flags %#llx -> %s\n", rep.handle, (unsigned long long)rep.size,
	    (unsigned long long)req.domains, (unsigned long long)req.flags, rep.domain == AMDGPU_GEM_DOMAIN_VRAM ? "VRAM" : "GTT");
	memset(a, 0, sizeof(*a));
	a->out.handle = rep.handle;
	return 0;
}

static int gem_close(struct gdev *g, struct drm_gem_close *c)
{
	struct gpud_bo_handle h = { .handle = c->handle };
	int r = rpc(g, GPUD_BO_FREE, &h, sizeof(h), NULL, 0, NULL, 0, NULL);

	pthread_mutex_lock(&g->bo_lock);
	if (bo_get(g, c->handle))
		g->bos[c->handle].used = false;
	pthread_mutex_unlock(&g->bo_lock);
	return r;
}

static int gem_mmap(struct gdev *g, union drm_amdgpu_gem_mmap *a)
{
	struct bo *b;
	uint64_t off;

	pthread_mutex_lock(&g->bo_lock);
	b = bo_get(g, a->in.handle);
	off = b ? b->rep.mmap_offset : GPUD_NO_MMAP;
	pthread_mutex_unlock(&g->bo_lock);
	if (off == GPUD_NO_MMAP)
		return -EINVAL;
	a->out.addr_ptr = off;
	return 0;
}

static int gem_va(struct gdev *g, struct drm_amdgpu_gem_va *a)
{
	struct gpud_va_req req = {
		.handle = a->handle,
		.operation = a->operation,
		.flags = a->flags,
		.va = a->va_address,
		.offset = a->offset_in_bo,
		.size = a->map_size,
	};

	return rpc(g, GPUD_VA, &req, sizeof(req), NULL, 0, NULL, 0, NULL);
}

static int gem_metadata(struct gdev *g, struct drm_amdgpu_gem_metadata *a)
{
	int r = 0;

	pthread_mutex_lock(&g->bo_lock);
	struct bo *b = bo_get(g, a->handle);

	if (!b)
		r = -ENOENT;
	else if (a->op == AMDGPU_GEM_METADATA_OP_SET_METADATA)
		b->meta = *a;
	else if (a->op == AMDGPU_GEM_METADATA_OP_GET_METADATA)
		a->data = b->meta.data;
	else
		r = -EINVAL;
	pthread_mutex_unlock(&g->bo_lock);
	return r;
}

static int gem_op(struct gdev *g, struct drm_amdgpu_gem_op *a)
{
	struct drm_amdgpu_gem_create_in info;
	int r = 0;

	if (a->op == AMDGPU_GEM_OP_SET_PLACEMENT)
		return 0;
	if (a->op != AMDGPU_GEM_OP_GET_GEM_CREATE_INFO)
		return -EINVAL;
	pthread_mutex_lock(&g->bo_lock);
	struct bo *b = bo_get(g, a->handle);

	if (b) {
		memset(&info, 0, sizeof(info));
		info.bo_size = b->rep.size;
		info.alignment = b->rep.alignment;
		info.domains = b->rep.domain;
		info.domain_flags = b->rep.flags;
		memcpy((void *)(uintptr_t)a->value, &info, sizeof(info));
	} else {
		r = -ENOENT;
	}
	pthread_mutex_unlock(&g->bo_lock);
	return r;
}

static int amdgpu_info(struct gdev *g, struct drm_amdgpu_info *a)
{
	struct gpud_info_req req;
	char out[GPUD_MAX_MSG - 64];

	if (!a->return_size || a->return_size > sizeof(out))
		return -EINVAL;
	memcpy(req.query, a, sizeof(req.query));
	int r = rpc(g, GPUD_INFO, &req, sizeof(req), NULL, 0, out, a->return_size, NULL);

	if (r == 0)
		memcpy((void *)(uintptr_t)a->return_pointer, out, a->return_size);
	return r;
}

static int ctx(struct gdev *g, union drm_amdgpu_ctx *a)
{
	switch (a->in.op) {
	case AMDGPU_CTX_OP_ALLOC_CTX:
		a->out.alloc.ctx_id = __atomic_fetch_add(&g->next_ctx, 1, __ATOMIC_SEQ_CST);
		return 0;
	case AMDGPU_CTX_OP_FREE_CTX:
		return 0;
	case AMDGPU_CTX_OP_QUERY_STATE:
	case AMDGPU_CTX_OP_QUERY_STATE2:
		memset(&a->out, 0, sizeof(a->out));
		return 0;
	case AMDGPU_CTX_OP_GET_STABLE_PSTATE:
	case AMDGPU_CTX_OP_SET_STABLE_PSTATE:
		memset(&a->out, 0, sizeof(a->out));
		return 0;
	default:
		return -EINVAL;
	}
}

static int ring_fence(struct gdev *g, uint32_t ip_type, uint32_t ring, uint64_t handle, uint64_t *fence)
{
	struct gpud_ring_req req = { .ip_type = ip_type, .ring = ring };
	struct gpud_ring_rep rep;
	int r = rpc(g, GPUD_RING, &req, sizeof(req), NULL, 0, &rep, sizeof(rep), NULL);

	if (r)
		return r;
	/* CS handles are already full fence ids; a bare sequence number gets this ring's index. */
	*fence = GPUD_FENCE_RING(handle) ? handle : GPUD_FENCE(rep.ring, handle);
	return 0;
}

static int cs(struct gdev *g, union drm_amdgpu_cs *a)
{
	uint64_t *chunk_ptrs = (uint64_t *)(uintptr_t)a->in.chunks;
	struct gpud_ib_desc ibs[16];
	uint64_t deps[256];
	uint32_t nibs = 0, ndeps = 0, outs[64], nouts = 0;
	struct gpud_submit_req req = { 0 };
	struct gpud_fence_rep rep;
	int r;

	for (uint32_t i = 0; i < a->in.num_chunks; i++) {
		struct drm_amdgpu_cs_chunk *c = (struct drm_amdgpu_cs_chunk *)(uintptr_t)chunk_ptrs[i];
		void *data = (void *)(uintptr_t)c->chunk_data;

		switch (c->chunk_id) {
		case AMDGPU_CHUNK_ID_IB: {
			struct drm_amdgpu_cs_chunk_ib *ib = data;

			if (nibs == 16)
				return -EINVAL;
			if (nibs && (ib->ip_type != req.ip_type || ib->ring != req.ring)) {
				LOG("CS with IBs for more than one ring\n");
				return -EINVAL;
			}
			req.ip_type = ib->ip_type;
			req.ring = ib->ring;
			ibs[nibs++] = (struct gpud_ib_desc){ .va = ib->va_start, .size_dw = ib->ib_bytes / 4,
							     .flags = ib->flags };
			break;
		}
		case AMDGPU_CHUNK_ID_SYNCOBJ_IN: {
			struct drm_amdgpu_cs_chunk_sem *s = data;
			uint32_t n = c->length_dw * 4 / sizeof(*s);

			/* Wait until each has a fence (or is signalled), then make the fences dependencies. */
			for (uint32_t k = 0; k < n; k++) {
				uint64_t fence = 0;
				int st;

				for (;;) {
					pthread_mutex_lock(&g->sync_lock);
					st = syncobj_state(g, s[k].handle, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, &fence);
					if (st == -EAGAIN) {
						struct timespec ts;

						clock_gettime(CLOCK_REALTIME, &ts);
						ts.tv_sec += 1;
						pthread_cond_timedwait(&g->sync_cond, &g->sync_lock, &ts);
					}
					pthread_mutex_unlock(&g->sync_lock);
					if (st != -EAGAIN)
						break;
				}
				if (st < 0)
					return st;
				if (st == 0 && ndeps < 256)
					deps[ndeps++] = fence;
			}
			break;
		}
		case AMDGPU_CHUNK_ID_SYNCOBJ_OUT: {
			struct drm_amdgpu_cs_chunk_sem *s = data;
			uint32_t n = c->length_dw * 4 / sizeof(*s);

			for (uint32_t k = 0; k < n && nouts < 64; k++)
				outs[nouts++] = s[k].handle;
			break;
		}
		case AMDGPU_CHUNK_ID_DEPENDENCIES:
		case AMDGPU_CHUNK_ID_SCHEDULED_DEPENDENCIES: {
			struct drm_amdgpu_cs_chunk_dep *d = data;
			uint32_t n = c->length_dw * 4 / sizeof(*d);

			for (uint32_t k = 0; k < n && ndeps < 256; k++) {
				uint64_t fence;

				if (d[k].handle && ring_fence(g, d[k].ip_type, d[k].ring, d[k].handle, &fence) == 0)
					deps[ndeps++] = fence;
			}
			break;
		}
		case AMDGPU_CHUNK_ID_BO_HANDLES:
		case AMDGPU_CHUNK_ID_FENCE:
			/* Every buffer stays resident; user fences aren't used by RADV. */
			break;
		default:
			LOG("CS chunk %u not supported\n", c->chunk_id);
			return -EINVAL;
		}
	}
	if (!nibs)
		return -EINVAL;
	req.num_ibs = nibs;
	req.num_deps = ndeps;
	char payload[sizeof(ibs) + sizeof(deps)];

	memcpy(payload, ibs, nibs * sizeof(ibs[0]));
	memcpy(payload + nibs * sizeof(ibs[0]), deps, ndeps * sizeof(deps[0]));
	r = rpc(g, GPUD_SUBMIT, &req, sizeof(req), payload, nibs * sizeof(ibs[0]) + ndeps * sizeof(deps[0]), &rep,
		sizeof(rep), NULL);
	if (r)
		return r;
	for (uint32_t k = 0; k < nouts; k++)
		syncobj_set_fence(g, outs[k], rep.fence);
	a->out.handle = rep.fence;
	LOG("CS ip %u ring %u: %u IBs, %u deps -> fence %#llx\n", req.ip_type, req.ring, nibs, ndeps,
	    (unsigned long long)rep.fence);
	return 0;
}

static int wait_cs(struct gdev *g, union drm_amdgpu_wait_cs *a)
{
	uint64_t fence;
	int r = ring_fence(g, a->in.ip_type, a->in.ring, a->in.handle, &fence);

	if (r)
		return r;
	r = fence_wait(g, fence, abs_timeout(a->in.timeout));
	memset(&a->out, 0, sizeof(a->out));
	if (r == -ETIME) {
		a->out.status = 1;
		return 0;
	}
	return r;
}

static int wait_fences(struct gdev *g, union drm_amdgpu_wait_fences *a)
{
	struct drm_amdgpu_fence *f = (struct drm_amdgpu_fence *)(uintptr_t)a->in.fences;
	int64_t deadline = abs_timeout(a->in.timeout_ns);

	for (;;) {
		uint32_t done = 0;

		for (uint32_t i = 0; i < a->in.fence_count; i++) {
			uint64_t fence;
			struct gpud_wait_req w = { .timeout_ns = 0 };

			if (ring_fence(g, f[i].ip_type, f[i].ring, f[i].seq_no, &fence))
				return -EINVAL;
			w.fence = fence;
			if (rpc(g, GPUD_WAIT, &w, sizeof(w), NULL, 0, NULL, 0, NULL) == 0) {
				done++;
				if (!a->in.wait_all) {
					a->out.status = 1;
					a->out.first_signaled = i;
					return 0;
				}
			}
		}
		if (done == a->in.fence_count) {
			a->out.status = 1;
			a->out.first_signaled = 0;
			return 0;
		}
		if (now_ns() >= deadline) {
			a->out.status = 0;
			return 0;
		}
		usleep(20);
	}
}

static int fence_to_handle(struct gdev *g, union drm_amdgpu_fence_to_handle *a)
{
	uint64_t fence;
	uint32_t h;
	int r = ring_fence(g, a->in.fence.ip_type, a->in.fence.ring, a->in.fence.seq_no, &fence);

	if (r)
		return r;
	if (a->in.what != AMDGPU_FENCE_TO_HANDLE_GET_SYNCOBJ)
		return -EOPNOTSUPP;
	r = syncobj_new(g, false, fence, true, &h);
	if (r)
		return r;
	a->out.handle = h;
	return 0;
}

static int syncobj_ioctl(struct gdev *g, unsigned long req, void *arg)
{
	switch (_IOC_NR(req)) {
	case 0xbf: /* SYNCOBJ_CREATE */ {
		struct drm_syncobj_create *c = arg;

		return syncobj_new(g, c->flags & DRM_SYNCOBJ_CREATE_SIGNALED, 0, false, &c->handle);
	}
	case 0xc0: /* SYNCOBJ_DESTROY */ {
		struct drm_syncobj_destroy *d = arg;

		pthread_mutex_lock(&g->sync_lock);
		struct syncobj *s = syncobj_get(g, d->handle);

		if (s)
			s->used = false;
		pthread_mutex_unlock(&g->sync_lock);
		return s ? 0 : -ENOENT;
	}
	case 0xc4: /* SYNCOBJ_RESET */
	case 0xc5: /* SYNCOBJ_SIGNAL */ {
		struct drm_syncobj_array *a = arg;
		uint32_t *h = (uint32_t *)(uintptr_t)a->handles;

		pthread_mutex_lock(&g->sync_lock);
		for (uint32_t i = 0; i < a->count_handles; i++) {
			struct syncobj *s = syncobj_get(g, h[i]);

			if (s) {
				s->signaled = _IOC_NR(req) == 0xc5;
				s->has_fence = false;
			}
		}
		pthread_cond_broadcast(&g->sync_cond);
		pthread_mutex_unlock(&g->sync_lock);
		return 0;
	}
	case 0xc3: /* SYNCOBJ_WAIT */ {
		struct drm_syncobj_wait *w = arg;

		return syncobj_wait(g, (uint32_t *)(uintptr_t)w->handles, w->count_handles, abs_timeout(w->timeout_nsec),
				    w->flags, &w->first_signaled);
	}
	case 0xcc: /* SYNCOBJ_TRANSFER */ {
		struct drm_syncobj_transfer *t = arg;

		if (t->src_point || t->dst_point)
			return -EINVAL;
		pthread_mutex_lock(&g->sync_lock);
		struct syncobj *src = syncobj_get(g, t->src_handle), *dst = syncobj_get(g, t->dst_handle);

		if (src && dst) {
			dst->signaled = src->signaled;
			dst->has_fence = src->has_fence;
			dst->fence = src->fence;
		}
		pthread_cond_broadcast(&g->sync_cond);
		pthread_mutex_unlock(&g->sync_lock);
		return src && dst ? 0 : -ENOENT;
	}
	default:
		/* sync files and opaque fds need kernel objects; timelines are emulated by Mesa */
		return -EOPNOTSUPP;
	}
}

/* Dispatches on the ioctl number, as the DRM core does: the encoded size and direction vary with the uapi
 * headers userspace was built against. */
static int gpud_ioctl(struct gdev *g, unsigned long req, void *arg)
{
	unsigned nr = _IOC_NR(req);

	if (_IOC_TYPE(req) != DRM_IOCTL_BASE)
		return -ENOTTY;
	switch (nr) {
	case 0x00: /* VERSION */
		return drm_version(arg);
	case 0x0c: /* GET_CAP */
		return get_cap(arg);
	case 0x0d: /* SET_CLIENT_CAP */
		return -EINVAL;
	case 0x09: /* GEM_CLOSE */
		return gem_close(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE:
		return gem_create(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_GEM_MMAP:
		return gem_mmap(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA:
		return gem_va(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_GEM_METADATA:
		return gem_metadata(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_GEM_OP:
		return gem_op(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_GEM_WAIT_IDLE: {
		union drm_amdgpu_gem_wait_idle *w = arg;

		memset(&w->out, 0, sizeof(w->out));
		return 0;
	}
	case DRM_COMMAND_BASE + DRM_AMDGPU_INFO:
		return amdgpu_info(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_CTX:
		return ctx(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_CS:
		return cs(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_CS:
		return wait_cs(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_FENCES:
		return wait_fences(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_FENCE_TO_HANDLE:
		return fence_to_handle(g, arg);
	case DRM_COMMAND_BASE + DRM_AMDGPU_VM:
		memset(&((union drm_amdgpu_vm *)arg)->out, 0, sizeof(((union drm_amdgpu_vm *)arg)->out));
		return 0;
	case DRM_COMMAND_BASE + DRM_AMDGPU_GEM_USERPTR:
		return -EINVAL;
	}
	if (nr >= 0xbf && nr <= 0xcf)
		return syncobj_ioctl(g, DRM_IOWR(nr, char), arg);
	LOG("ioctl %#lx (nr %#x) not supported\n", req, nr);
	return -EINVAL;
}

int ioctl(int fd, ioctl_req req, ...)
{
	va_list ap;
	void *arg;

	va_start(ap, req);
	arg = va_arg(ap, void *);
	va_end(ap);
	if (!real_ioctl)
		real_ioctl = dlsym(RTLD_NEXT, "ioctl");
	struct gdev *g = dev_of(fd);

	if (!g)
		return real_ioctl(fd, req, arg);
	/* The request's 32 bits, without musl's int sign-extending into the DRM_IOCTL_* constants. */
	unsigned long request = (unsigned int)req;
	/* Work on a zero-padded copy of the caller's struct, copied back up to the caller's size, as drm_ioctl does. */
	uint64_t kbuf[64] = { 0 };
	size_t usize = _IOC_SIZE(request);
	int r;

	if (usize > sizeof(kbuf)) {
		r = -EINVAL;
	} else {
		if (arg && (_IOC_DIR(request) & _IOC_WRITE))
			memcpy(kbuf, arg, usize);
		r = gpud_ioctl(g, request, kbuf);
		if (r == 0 && arg && (_IOC_DIR(request) & _IOC_READ))
			memcpy(arg, kbuf, usize);
	}

	LOG("ioctl nr %#x -> %d\n", (unsigned)_IOC_NR(request), r);
	if (r < 0) {
		errno = -r;
		return -1;
	}
	return 0;
}

/* ---- libdrm's device queries ---- */
static drmDevicePtr new_device(struct gdev *g)
{
	struct gpud_device *d = calloc(1, sizeof(*d));

	if (!d)
		return NULL;
	d->magic = GPUD_DEVICE_MAGIC;
	snprintf(d->path, sizeof(d->path), "%s", g->bo_path);
	d->nodes[DRM_NODE_RENDER] = d->path;
	d->dev.nodes = d->nodes;
	d->dev.available_nodes = 1 << DRM_NODE_RENDER;
	d->dev.bustype = DRM_BUS_PCI;
	d->bus = (drmPciBusInfo){ g->hello.pci_domain, g->hello.pci_bus, g->hello.pci_dev, g->hello.pci_func };
	d->info = (drmPciDeviceInfo){ g->hello.vendor_id, g->hello.device_id, g->hello.subvendor_id,
				      g->hello.subdevice_id, g->hello.revision };
	d->dev.businfo.pci = &d->bus;
	d->dev.deviceinfo.pci = &d->info;
	return &d->dev;
}

static bool ours(drmDevicePtr dev) { return dev && ((struct gpud_device *)dev)->magic == GPUD_DEVICE_MAGIC; }

int drmGetDevices2(uint32_t flags, drmDevicePtr devices[], int max_devices)
{
	(void)flags;
	if (!connected())
		return -ENODEV;
	for (int i = 0; devices && i < ndevs && i < max_devices; i++) {
		devices[i] = new_device(&devs[i]);
		if (!devices[i])
			return -ENOMEM;
	}
	return ndevs;
}

int drmGetDevices(drmDevicePtr devices[], int max_devices) { return drmGetDevices2(0, devices, max_devices); }

int drmGetDevice2(int fd, uint32_t flags, drmDevicePtr *device)
{
	static int (*real)(int, uint32_t, drmDevicePtr *);

	struct gdev *g = connected() ? dev_of(fd) : NULL;

	if (g) {
		*device = new_device(g);
		return *device ? 0 : -ENOMEM;
	}
	if (!real)
		real = dlsym(RTLD_NEXT, "drmGetDevice2");
	return real ? real(fd, flags, device) : -ENODEV;
}

int drmGetDevice(int fd, drmDevicePtr *device) { return drmGetDevice2(fd, 0, device); }

void drmFreeDevice(drmDevicePtr *device)
{
	static void (*real)(drmDevicePtr *);

	if (!device || !*device)
		return;
	if (ours(*device)) {
		free(*device);
		*device = NULL;
		return;
	}
	if (!real)
		real = dlsym(RTLD_NEXT, "drmFreeDevice");
	if (real)
		real(device);
}

void drmFreeDevices(drmDevicePtr devices[], int count)
{
	for (int i = 0; devices && i < count; i++)
		drmFreeDevice(&devices[i]);
}

int drmGetNodeTypeFromFd(int fd)
{
	static int (*real)(int);

	if (dev_of(fd))
		return DRM_NODE_RENDER;
	if (!real)
		real = dlsym(RTLD_NEXT, "drmGetNodeTypeFromFd");
	return real ? real(fd) : -1;
}

#define NAME_FROM_FD(fn)                                         \
	char *fn(int fd)                                         \
	{                                                        \
		static char *(*real)(int);                       \
		struct gdev *g = dev_of(fd);                     \
		if (g)                                           \
			return strdup(g->bo_path);               \
		if (!real)                                       \
			real = dlsym(RTLD_NEXT, #fn);            \
		return real ? real(fd) : NULL;                   \
	}
NAME_FROM_FD(drmGetPrimaryDeviceNameFromFd)
NAME_FROM_FD(drmGetRenderDeviceNameFromFd)
NAME_FROM_FD(drmGetDeviceNameFromFd2)
