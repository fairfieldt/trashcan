// SPDX-License-Identifier: MIT
/*
 * drminfo: what an amdgpu render node answers to the AMDGPU_INFO queries RADV and libdrm make at startup. Run against
 * amdgpu (the oracle) and against gpud (through libgpud_drm.so) to compare.
 * usage: drminfo /dev/dri/renderD129, or LD_PRELOAD=libgpud_drm.so drminfo gpud (the device libdrm enumerates)
 *
 * drminfo clockmon OUT samples every enumerated GPU's core and memory clock (MHz) and temperature (C) four times a
 * second into OUT until killed, in scripts/clockmon.sh's format: through gpud, which has no hwmon.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <drm/amdgpu_drm.h>
#include <drm/drm.h>

static int fd;

static int info(uint32_t query, void *out, uint32_t size, void (*fill)(struct drm_amdgpu_info *, const void *),
		const void *arg)
{
	struct drm_amdgpu_info req = { .return_pointer = (uintptr_t)out, .return_size = size, .query = query };

	if (fill)
		fill(&req, arg);
	return ioctl(fd, DRM_IOCTL_AMDGPU_INFO, &req) ? -errno : 0;
}

static void hw_ip(struct drm_amdgpu_info *r, const void *a) { r->query_hw_ip.type = *(const uint32_t *)a; }
static void fw(struct drm_amdgpu_info *r, const void *a) { r->query_fw.fw_type = *(const uint32_t *)a; }
static void mmr(struct drm_amdgpu_info *r, const void *a)
{
	r->read_mmr_reg.dword_offset = ((const uint32_t *)a)[0];
	r->read_mmr_reg.count = ((const uint32_t *)a)[1];
	r->read_mmr_reg.instance = ((const uint32_t *)a)[2];
}

static uint32_t sensor(uint32_t type)
{
	struct drm_amdgpu_info req = { .return_size = 4, .query = AMDGPU_INFO_SENSOR };
	uint32_t v = 0;

	req.return_pointer = (uintptr_t)&v;
	req.sensor_info.type = type;
	return ioctl(fd, DRM_IOCTL_AMDGPU_INFO, &req) ? 0 : v;
}

static int clockmon(const char *out)
{
	int (*get)(uint32_t, void **, int) = (int (*)(uint32_t, void **, int))dlsym(RTLD_DEFAULT, "drmGetDevices2");
	void *dev[8];
	int fds[8], n = get ? get(0, dev, 8) : 0;
	FILE *f = fopen(out, "w");

	if (n < 1 || !f) {
		fprintf(stderr, "clockmon: %s\n", n < 1 ? "no device enumerated" : strerror(errno));
		return 1;
	}
	for (int i = 0; i < n; i++)
		fds[i] = open((*(char ***)dev[i])[2], O_RDWR | O_CLOEXEC);
	fprintf(f, "t");
	for (int i = 0; i < n; i++)
		fprintf(f, " sclk%d mclk%d temp%d", i, i, i);
	fprintf(f, "\n");
	for (;;) {
		struct timespec ts;

		clock_gettime(CLOCK_REALTIME, &ts);
		fprintf(f, "%lld.%03ld", (long long)ts.tv_sec, ts.tv_nsec / 1000000);
		for (int i = 0; i < n; i++) {
			fd = fds[i];
			fprintf(f, " %u %u %u", sensor(AMDGPU_INFO_SENSOR_GFX_SCLK), sensor(AMDGPU_INFO_SENSOR_GFX_MCLK),
				sensor(AMDGPU_INFO_SENSOR_GPU_TEMP) / 1000);
		}
		fprintf(f, "\n");
		fflush(f);
		usleep(250000);
	}
}

int main(int argc, char **argv)
{
	struct drm_amdgpu_info_device d = { 0 };
	struct drm_amdgpu_memory_info m = { 0 };
	struct drm_amdgpu_info_vram_gtt vg = { 0 };
	char name[32] = { 0 };
	struct drm_version v = { .name = name, .name_len = sizeof(name) - 1 };
	uint32_t u32;
	uint64_t u64;
	int r;

	const char *path = argc > 1 ? argv[1] : "/dev/dri/renderD128";

	if (!strcmp(path, "clockmon") && argc > 2)
		return clockmon(argv[2]);

	if (!strcmp(path, "gpud")) {
		/* drmGetDevices2 from whichever library provides it (libgpud_drm.so when preloaded) */
		int (*get)(uint32_t, void **, int) = (int (*)(uint32_t, void **, int))dlsym(RTLD_DEFAULT, "drmGetDevices2");
		void *dev[4];

		if (!get || get(0, dev, 4) < 1) {
			fprintf(stderr, "no device enumerated\n");
			return 1;
		}
		path = (*(char ***)dev[0])[2]; /* drmDevice.nodes[DRM_NODE_RENDER] */
		printf("device %s\n", path);
	}
	fd = open(path, O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		perror("open");
		return 1;
	}
	r = ioctl(fd, DRM_IOCTL_VERSION, &v);
	printf("version %d: %s %d.%d.%d\n", r, name, v.version_major, v.version_minor, v.version_patchlevel);
	r = info(AMDGPU_INFO_ACCEL_WORKING, &u32, 4, NULL, NULL);
	printf("accel_working %d: %u\n", r, u32);
	r = info(AMDGPU_INFO_DEV_INFO, &d, sizeof(d), NULL, NULL);
	printf("dev_info %d: device_id %#x chip_rev %u external_rev %u pci_rev %u family %u se %u sh_per_se %u\n", r,
	       d.device_id, d.chip_rev, d.external_rev, d.pci_rev, d.family, d.num_shader_engines,
	       d.num_shader_arrays_per_engine);
	printf("  gpu_counter_freq %u max_engine_clock %llu max_memory_clock %llu min %llu/%llu\n", d.gpu_counter_freq,
	       (unsigned long long)d.max_engine_clock, (unsigned long long)d.max_memory_clock,
	       (unsigned long long)d.min_engine_clock, (unsigned long long)d.min_memory_clock);
	printf("  cu_active %u cu_ao_mask %#x rb_mask %#x num_rb %u hw_contexts %u ids_flags %#llx\n",
	       d.cu_active_number, d.cu_ao_mask, d.enabled_rb_pipes_mask, d.num_rb_pipes, d.num_hw_gfx_contexts,
	       (unsigned long long)d.ids_flags);
	printf("  va %#llx-%#llx align %u high %#llx-%#llx pte_frag %u gart_page %u ce_ram %u\n",
	       (unsigned long long)d.virtual_address_offset, (unsigned long long)d.virtual_address_max,
	       d.virtual_address_alignment, (unsigned long long)d.high_va_offset, (unsigned long long)d.high_va_max,
	       d.pte_fragment_size, d.gart_page_size, d.ce_ram_size);
	printf("  vram_type %u width %u wave %u vgprs %u cu_per_sh %u tcc %u gs_vgt %u gs_prim %u max_gs_waves %u\n",
	       d.vram_type, d.vram_bit_width, d.wave_front_size, d.num_shader_visible_vgprs, d.num_cu_per_sh,
	       d.num_tcc_blocks, d.gs_vgt_table_depth, d.gs_prim_buffer_depth, d.max_gs_waves_per_vgt);
	printf("  pcie gen %u lanes %u tcc_disabled %#llx double_offchip %u\n", d.pcie_gen, d.pcie_num_lanes,
	       (unsigned long long)d.tcc_disabled_mask, d.gc_double_offchip_lds_buf);
	printf("  cu_bitmap");
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			printf(" %x", d.cu_bitmap[i][j]);
	printf("\n");
	r = info(AMDGPU_INFO_VRAM_GTT, &vg, sizeof(vg), NULL, NULL);
	printf("vram_gtt %d: vram %llu vis %llu gtt %llu\n", r, (unsigned long long)vg.vram_size,
	       (unsigned long long)vg.vram_cpu_accessible_size, (unsigned long long)vg.gtt_size);
	r = info(AMDGPU_INFO_MEMORY, &m, sizeof(m), NULL, NULL);
	printf("memory %d: vram %llu/%llu max %llu vis %llu/%llu gtt %llu/%llu max %llu\n", r,
	       (unsigned long long)m.vram.total_heap_size, (unsigned long long)m.vram.usable_heap_size,
	       (unsigned long long)m.vram.max_allocation, (unsigned long long)m.cpu_accessible_vram.total_heap_size,
	       (unsigned long long)m.cpu_accessible_vram.usable_heap_size, (unsigned long long)m.gtt.total_heap_size,
	       (unsigned long long)m.gtt.usable_heap_size, (unsigned long long)m.gtt.max_allocation);
	for (uint32_t t = 0; t < 3; t++) {
		struct drm_amdgpu_info_hw_ip ip = { 0 };

		r = info(AMDGPU_INFO_HW_IP_INFO, &ip, sizeof(ip), hw_ip, &t);
		u32 = 0;
		int r2 = info(AMDGPU_INFO_HW_IP_COUNT, &u32, 4, hw_ip, &t);
		printf("hw_ip %u %d: version %u.%u caps %#llx rings %#x align %u/%u count(%d) %u\n", t, r,
		       ip.hw_ip_version_major, ip.hw_ip_version_minor, (unsigned long long)ip.capabilities_flags,
		       ip.available_rings, ip.ib_start_alignment, ip.ib_size_alignment, r2, u32);
	}
	uint32_t fws[] = { AMDGPU_INFO_FW_GFX_ME, AMDGPU_INFO_FW_GFX_PFP, AMDGPU_INFO_FW_GFX_CE, AMDGPU_INFO_FW_GFX_RLC,
			   AMDGPU_INFO_FW_GFX_MEC, AMDGPU_INFO_FW_SDMA, AMDGPU_INFO_FW_UVD, AMDGPU_INFO_FW_VCE };
	for (unsigned i = 0; i < sizeof(fws) / 4; i++) {
		struct drm_amdgpu_info_firmware f = { 0 };

		r = info(AMDGPU_INFO_FW_VERSION, &f, sizeof(f), fw, &fws[i]);
		printf("fw %u %d: ver %#x feature %u\n", fws[i], r, f.ver, f.feature);
	}
	uint32_t regs[][3] = { { 0x263d, 1, 0xffffffff }, { 0xa0d4, 1, 0xffffffff }, { 0x263e, 1, 0xffffffff },
			       { 0x9d8, 1, 0xffffffff } };
	for (unsigned i = 0; i < 4; i++) {
		r = info(AMDGPU_INFO_READ_MMR_REG, &u32, 4, mmr, regs[i]);
		printf("mmr %#x %d: %#x\n", regs[i][0], r, u32);
	}
	uint32_t tile[32] = { 0 }, tile_q[3] = { 0x2644, 32, 0xffffffff };
	r = info(AMDGPU_INFO_READ_MMR_REG, tile, sizeof(tile), mmr, tile_q);
	printf("tile_modes %d:", r);
	for (int i = 0; i < 32; i++)
		printf(" %x", tile[i]);
	printf("\n");
	r = info(AMDGPU_INFO_TIMESTAMP, &u64, 8, NULL, NULL);
	printf("timestamp %d: %llu\n", r, (unsigned long long)u64);
	return 0;
}
