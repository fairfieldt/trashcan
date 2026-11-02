/* SPDX-License-Identifier: MIT */
/*
 * gpud's client protocol: one request and one reply per message on a SOCK_SEQPACKET Unix socket. Requests start
 * with struct gpud_req, replies with struct gpud_rep (status 0 or -errno), each followed by the op's payload. The
 * HELLO reply carries one file descriptor (SCM_RIGHTS): the client's buffer file, a memfd whose pages back every
 * host-memory buffer object at the offset the server reports, so the client maps buffers with mmap(fd, offset).
 *
 * Fences are 64-bit ids: ring index in the top 8 bits, the ring's sequence number below.
 * The structs are mirrored in gpud's Rust server (driver/gpud/src/proto.rs) with the same layout.
 */
#ifndef GPUD_PROTO_H
#define GPUD_PROTO_H

#include <stdint.h>

#define GPUD_PROTO_VERSION 1
#define GPUD_MAX_MSG 65536
#define GPUD_SOCKET_DEFAULT "/run/gpud/gpud.sock"
#define GPUD_FENCE(ring, seq) (((uint64_t)(ring) << 56) | ((seq) & 0x00ffffffffffffffULL))
#define GPUD_FENCE_RING(f) ((int)((f) >> 56))
#define GPUD_FENCE_SEQ(f) ((f) & 0x00ffffffffffffffULL)
#define GPUD_NO_MMAP UINT64_MAX

enum gpud_op {
	GPUD_HELLO = 1,	 /* -> gpud_hello_rep + fd */
	GPUD_INFO,	 /* gpud_info_req -> query.return_size bytes */
	GPUD_BO_CREATE,	 /* gpud_bo_create_req -> gpud_bo_rep */
	GPUD_BO_FREE,	 /* gpud_bo_handle */
	GPUD_BO_QUERY,	 /* gpud_bo_handle -> gpud_bo_rep */
	GPUD_VA,	 /* gpud_va_req */
	GPUD_SUBMIT,	 /* gpud_submit_req, ibs[num_ibs], deps[num_deps] (fence ids) -> gpud_fence_rep */
	GPUD_WAIT,	 /* gpud_wait_req */
	GPUD_RING,	 /* gpud_ring_req -> gpud_ring_rep */
};

struct gpud_req {
	uint32_t op;
	uint32_t len; /* payload bytes */
};

struct gpud_rep {
	int32_t status;
	uint32_t len;
};

struct gpud_hello_rep {
	uint32_t version;
	uint32_t vmid;
	uint64_t bo_file_size;
	uint16_t pci_domain;
	uint8_t pci_bus, pci_dev, pci_func, revision;
	uint16_t vendor_id, device_id, subvendor_id, subdevice_id;
};

/* struct drm_amdgpu_info without its user pointer (return_pointer is ignored; return_size bytes come back). */
struct gpud_info_req {
	uint8_t query[32];
};

struct gpud_bo_create_req {
	uint64_t size;
	uint64_t alignment;
	uint64_t domains; /* AMDGPU_GEM_DOMAIN_* */
	uint64_t flags;	  /* AMDGPU_GEM_CREATE_* */
};

struct gpud_bo_handle {
	uint32_t handle;
	uint32_t pad;
};

struct gpud_bo_rep {
	uint32_t handle;
	uint32_t domain;      /* where it went */
	uint64_t size;
	uint64_t mmap_offset; /* in the buffer file, or GPUD_NO_MMAP */
	uint64_t alignment;
	uint64_t domains;     /* as requested */
	uint64_t flags;
};

struct gpud_va_req {
	uint32_t handle;
	uint32_t operation; /* AMDGPU_VA_OP_* */
	uint32_t flags;	    /* AMDGPU_VM_PAGE_* */
	uint32_t pad;
	uint64_t va;
	uint64_t offset; /* in the buffer */
	uint64_t size;
};

struct gpud_ib_desc {
	uint64_t va;
	uint32_t size_dw;
	uint32_t flags;
};

struct gpud_submit_req {
	uint32_t ip_type; /* AMDGPU_HW_IP_* */
	uint32_t ring;
	uint32_t num_ibs;
	uint32_t num_deps;
};

struct gpud_fence_rep {
	uint64_t fence;
};

struct gpud_wait_req {
	uint64_t fence;
	int64_t timeout_ns; /* negative: no limit */
};

struct gpud_ring_req {
	uint32_t ip_type;
	uint32_t ring;
};

struct gpud_ring_rep {
	int32_t ring; /* gpud's ring index */
	uint32_t pad;
	uint64_t last_fence; /* the ring's most recently completed fence */
};

#endif
