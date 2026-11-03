/* SPDX-License-Identifier: MIT */
/*
 * gpud: what amdgpu_atombios.c uses from Linux's amdgpu_display.h. The display callbacks, which Linux dispatches
 * through adev->mode_info.funcs, are gpud functions that do nothing: gpud has no display.
 */
#ifndef GPUD_AMDGPU_DISPLAY_H
#define GPUD_AMDGPU_DISPLAY_H
#include "amdgpu.h"

u32 amdgpu_display_hpd_get_gpio_reg(struct amdgpu_device *adev);
void amdgpu_display_add_encoder(struct amdgpu_device *adev, uint32_t encoder_enum, uint32_t supported_device, u16 caps);
void amdgpu_display_add_connector(struct amdgpu_device *adev, uint32_t connector_id, uint32_t supported_device,
				  int connector_type, struct amdgpu_i2c_bus_rec *i2c_bus, uint16_t connector_object_id,
				  struct amdgpu_hpd *hpd, struct amdgpu_router *router);
void amdgpu_link_encoder_connector(struct drm_device *dev);
#endif
