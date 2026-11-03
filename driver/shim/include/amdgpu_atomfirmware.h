/* SPDX-License-Identifier: MIT */
/* gpud: the atomfirmware (GFX9+ VBIOS) entry points amdgpu_atombios.c links against; SI's VBIOS is ATOM BIOS. */
#ifndef GPUD_AMDGPU_ATOMFIRMWARE_H
#define GPUD_AMDGPU_ATOMFIRMWARE_H
#include "amdgpu.h"
uint32_t amdgpu_atomfirmware_query_firmware_capability(struct amdgpu_device *adev);
void amdgpu_atomfirmware_scratch_regs_init(struct amdgpu_device *adev);
int amdgpu_atomfirmware_allocate_fb_scratch(struct amdgpu_device *adev);
#endif
