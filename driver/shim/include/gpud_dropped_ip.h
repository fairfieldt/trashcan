/* SPDX-License-Identifier: MIT */
/* IP blocks si.c names but gpud doesn't run (display, video, power for now): placeholders that are never started. */
#ifndef GPUD_DROPPED_IP_H
#define GPUD_DROPPED_IP_H
#include "amdgpu.h"
extern const struct amdgpu_ip_block_version dce_v6_0_ip_block;
extern const struct amdgpu_ip_block_version dce_v6_4_ip_block;
extern const struct amdgpu_ip_block_version dm_ip_block;
extern const struct amdgpu_ip_block_version amdgpu_vkms_ip_block;
extern const struct amdgpu_ip_block_version uvd_v3_1_ip_block;
extern const struct amdgpu_ip_block_version vce_v1_0_ip_block;
extern const struct amdgpu_ip_block_version si_smu_ip_block;
#endif
