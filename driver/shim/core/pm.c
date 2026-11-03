// SPDX-License-Identifier: MIT
/*
 * What Linux's SI power management (pm/legacy-dpm, amdgpu_atombios.c) calls outside the files gpud imports: the
 * display configuration DPM plans memory-clock switches around, VRAM the VBIOS reserves for itself, and the
 * display, i2c and atomfirmware entry points that amdgpu_atombios.c links against but gpud never calls.
 */
#include "amdgpu.h"
#include "amdgpu_atomfirmware.h"
#include "amdgpu_display.h"
#include "amdgpu_dpm_internal.h"
#include "amdgpu_i2c.h"
#include <gpud_core.h>

/*
 * amdgpu_dpm_internal.c with no display: no CRTC is active, so the vblank time is unbounded and memory clock
 * switches never have to wait for one.
 */
void amdgpu_dpm_get_display_cfg(struct amdgpu_device *adev)
{
	struct amd_pp_display_configuration *cfg = &adev->pm.pm_display_cfg;

	cfg->min_vblank_time = 0xffffffff;
	cfg->display_clk = adev->clock.default_dispclk;
	cfg->num_display = 0;
	gpud_display_pm(adev);
}

/* The VBIOS's own VRAM (amdgpu_atombios_allocate_fb_scratch): kept out of client VRAM by gpud_vram_init. */
u64 gpud_fw_vram_offset, gpud_fw_vram_size;

void amdgpu_ttm_init_vram_resv(struct amdgpu_device *adev, enum amdgpu_resv_region_id id, uint64_t offset,
			       uint64_t size, bool needs_cpu_map)
{
	(void)needs_cpu_map;
	if (id != AMDGPU_RESV_FW_VRAM_USAGE || !size)
		return;
	gpud_fw_vram_offset = offset;
	gpud_fw_vram_size = size;
	dev_info(adev->dev, "VBIOS VRAM: %llu KiB at %#llx\n", (unsigned long long)size >> 10,
		 (unsigned long long)offset);
}

/* No display, no i2c buses, no atomfirmware (GFX9+) VBIOS. */
u32 amdgpu_display_hpd_get_gpio_reg(struct amdgpu_device *adev) { (void)adev; return 0; }
void amdgpu_display_add_encoder(struct amdgpu_device *adev, uint32_t encoder_enum, uint32_t supported_device, u16 caps)
{
	(void)adev;
	(void)encoder_enum;
	(void)supported_device;
	(void)caps;
}
void amdgpu_display_add_connector(struct amdgpu_device *adev, uint32_t connector_id, uint32_t supported_device,
				  int connector_type, struct amdgpu_i2c_bus_rec *i2c_bus, uint16_t connector_object_id,
				  struct amdgpu_hpd *hpd, struct amdgpu_router *router)
{
	(void)adev;
	(void)connector_id;
	(void)supported_device;
	(void)connector_type;
	(void)i2c_bus;
	(void)connector_object_id;
	(void)hpd;
	(void)router;
}
void amdgpu_link_encoder_connector(struct drm_device *dev) { (void)dev; }
struct amdgpu_i2c_chan *amdgpu_i2c_create(struct drm_device *dev, const struct amdgpu_i2c_bus_rec *rec, const char *name)
{
	(void)dev;
	(void)rec;
	(void)name;
	return NULL;
}
struct amdgpu_i2c_chan *amdgpu_i2c_lookup(struct amdgpu_device *adev, const struct amdgpu_i2c_bus_rec *i2c_bus)
{
	(void)adev;
	(void)i2c_bus;
	return NULL;
}
uint32_t amdgpu_atomfirmware_query_firmware_capability(struct amdgpu_device *adev) { (void)adev; return 0; }
void amdgpu_atomfirmware_scratch_regs_init(struct amdgpu_device *adev) { (void)adev; }
int amdgpu_atomfirmware_allocate_fb_scratch(struct amdgpu_device *adev) { (void)adev; return 0; }
