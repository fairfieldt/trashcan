/* SPDX-License-Identifier: MIT */
/*
 * gpud's amdgpu_mode.h: the display records the VBIOS and power-management code use (copied from Linux 7.3-rc7
 * amdgpu_mode.h, Copyright Advanced Micro Devices, Inc., MIT license); gpud has no display.
 */
#ifndef GPUD_AMDGPU_MODE_H
#define GPUD_AMDGPU_MODE_H

enum amdgpu_hpd_id {
	AMDGPU_HPD_1 = 0,
	AMDGPU_HPD_2,
	AMDGPU_HPD_3,
	AMDGPU_HPD_4,
	AMDGPU_HPD_5,
	AMDGPU_HPD_6,
	AMDGPU_HPD_NONE = 0xff,
};

struct amdgpu_i2c_bus_rec {
	bool valid;
	/* id used by atom */
	uint8_t i2c_id;
	/* id used by atom */
	enum amdgpu_hpd_id hpd;
	/* can be used with hw i2c engine */
	bool hw_capable;
	/* uses multi-media i2c engine */
	bool mm_i2c;
	/* regs and bits */
	uint32_t mask_clk_reg;
	uint32_t mask_data_reg;
	uint32_t a_clk_reg;
	uint32_t a_data_reg;
	uint32_t en_clk_reg;
	uint32_t en_data_reg;
	uint32_t y_clk_reg;
	uint32_t y_data_reg;
	uint32_t mask_clk_mask;
	uint32_t mask_data_mask;
	uint32_t a_clk_mask;
	uint32_t a_data_mask;
	uint32_t en_clk_mask;
	uint32_t en_data_mask;
	uint32_t y_clk_mask;
	uint32_t y_data_mask;
};

struct amdgpu_gpio_rec {
	bool valid;
	u8 id;
	u32 reg;
	u32 mask;
	u32 shift;
};

struct amdgpu_hpd {
	enum amdgpu_hpd_id hpd;
	u8 plugged_state;
	struct amdgpu_gpio_rec gpio;
};

struct amdgpu_router {
	u32 router_id;
	struct amdgpu_i2c_bus_rec i2c_info;
	u8 i2c_addr;
	/* i2c mux */
	bool ddc_valid;
	u8 ddc_mux_type;
	u8 ddc_mux_control_pin;
	u8 ddc_mux_state;
	/* clock/data mux */
	bool cd_valid;
	u8 cd_mux_type;
	u8 cd_mux_control_pin;
	u8 cd_mux_state;
};

enum amdgpu_connector_audio {
	AMDGPU_AUDIO_DISABLE = 0,
	AMDGPU_AUDIO_ENABLE = 1,
	AMDGPU_AUDIO_AUTO = 2
};

struct amdgpu_atom_ss {
	uint16_t percentage;
	uint16_t percentage_divider;
	uint8_t type;
	uint16_t step;
	uint8_t delay;
	uint8_t range;
	uint8_t refdiv;
	/* asic_ss */
	uint16_t rate;
	uint16_t amount;
};

#endif
