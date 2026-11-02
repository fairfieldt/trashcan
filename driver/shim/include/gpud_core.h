/* SPDX-License-Identifier: MIT */
/*
 * The C side of gpud as the Rust side sees it: create a Southern Islands device on a platform (MMIO and VRAM
 * mappings, DMA allocation, config space, logging), bring its IP blocks up with Linux's code, and drive the rings.
 * Nothing here knows about VFIO; the callbacks are implemented by gpud-platform.
 */
#ifndef GPUD_CORE_H
#define GPUD_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct amdgpu_device;

struct gpud_platform {
	void *ctx;
	/* level: 3 error, 4 warning, 5 notice, 6 info, 7 debug */
	void (*log)(void *ctx, int level, const char *msg);
	/* Zeroed, pinned host memory the device reaches at *iova; size is a multiple of 4096. */
	int (*dma_alloc)(void *ctx, size_t size, void **cpu, uint64_t *iova);
	void (*dma_free)(void *ctx, void *cpu, uint64_t iova, size_t size);
	/* PCI config space of the function: size 1, 2 or 4. */
	int (*config_read)(void *ctx, int offset, int size, uint32_t *value);
	int (*config_write)(void *ctx, int offset, int size, uint32_t value);
};

struct gpud_device_info {
	void *mmio;		/* BAR2 mapping */
	size_t mmio_size;
	void *vram;		/* BAR0 mapping: the CPU's window on the start of VRAM */
	size_t vram_size;
	uint64_t vram_bar_addr; /* BAR0 bus address, as config space reports it */
	uint16_t vendor, device, subsystem_vendor, subsystem_device;
	uint8_t revision;
	uint8_t bus, devfn;
	const char *firmware_dir; /* holds amdgpu/tahiti_*.bin */
};

struct amdgpu_device *gpud_device_create(const struct gpud_platform *platform, const struct gpud_device_info *info);
/* Runs every supported IP block's early, software, hardware and late init in amdgpu's order. 0 or -errno. */
int gpud_device_init(struct amdgpu_device *adev);
/* An IB test (a scratch write or a write-back slot) on every ready ring; the number of rings that failed. */
int gpud_device_ib_tests(struct amdgpu_device *adev, long timeout_ms);
/* Drains the interrupt ring; call when the MSI fires. Returns how many entries were processed. */
int gpud_irq_process(struct amdgpu_device *adev);
void gpud_device_fini(struct amdgpu_device *adev);

/* Rings by index, for status and submission. */
int gpud_ring_count(struct amdgpu_device *adev);
const char *gpud_ring_name(struct amdgpu_device *adev, int ring);
int gpud_ring_ready(struct amdgpu_device *adev, int ring);
int gpud_ring_type(struct amdgpu_device *adev, int ring); /* AMDGPU_RING_TYPE_* */

#ifdef __cplusplus
}
#endif
#endif
