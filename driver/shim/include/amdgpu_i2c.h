/* SPDX-License-Identifier: MIT */
/* gpud: Linux's amdgpu_i2c.h declarations amdgpu_atombios.c links against; gpud creates no i2c buses. */
#ifndef GPUD_AMDGPU_I2C_H
#define GPUD_AMDGPU_I2C_H
#include "amdgpu_display.h"
/* Linux's struct amdgpu_i2c_chan (amdgpu_mode.h) without its bit-banging and DP AUX members. */
struct amdgpu_i2c_chan {
	struct i2c_adapter adapter;
	struct drm_device *dev;
	struct amdgpu_i2c_bus_rec rec;
	struct mutex mutex;
};
struct amdgpu_i2c_chan *amdgpu_i2c_lookup(struct amdgpu_device *adev, const struct amdgpu_i2c_bus_rec *i2c_bus);
struct amdgpu_i2c_chan *amdgpu_i2c_create(struct drm_device *dev, const struct amdgpu_i2c_bus_rec *rec, const char *name);
#endif
