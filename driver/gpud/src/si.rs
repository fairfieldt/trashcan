//! Southern Islands (Tahiti) facts the probe needs, from Linux 7.3's amdgpu. Register offsets are dword offsets
//! as in `drivers/gpu/drm/amd/include/asic_reg/`; BAR2 holds the registers on SI (`amdgpu_device_init`: BAR5 is
//! only for Bonaire and later).

pub const VENDOR_ATI: u16 = 0x1002;
/// Tahiti XT; the D700 is 1002:6798 with Apple subsystem IDs 106b:0127 and 106b:0128.
pub const DEVICE_TAHITI_XT: u16 = 0x6798;

pub const REGISTER_BAR: u8 = 2;
pub const VRAM_BAR: u8 = 0;

pub const MM_INDEX: u32 = 0x0000; // bif_3_0_d.h
pub const MM_DATA: u32 = 0x0001;
pub const MM_INDEX_HI: u32 = 0x0006;
pub const CONFIG_MEMSIZE: u32 = 0x150A; // bif_3_0_d.h: VRAM size in MiB
pub const MC_VM_FB_LOCATION: u32 = 0x0809; // gmc_6_0_d.h: VRAM base and top in the MC address space, 16 MiB units
pub const BIF_FB_EN: u32 = 0x1524; // bif_3_0_d.h: host access to VRAM through BAR0 (read and write enable bits)

/// Registers the probe reads, in the order and format of results' `oracle.sh` (amdgpu's debugfs view) so the two
/// can be diffed. All are read-only in effect: no read side effects on SI.
pub const PROBE_REGISTERS: &[(&str, u32)] = &[
    ("CONFIG_MEMSIZE", CONFIG_MEMSIZE),
    ("CONFIG_APER_SIZE", 0x150C), // bif_3_0_d.h
    ("MC_VM_FB_LOCATION", MC_VM_FB_LOCATION),
    ("MC_VM_FB_OFFSET", 0x081A), // gmc_6_0_d.h from here to MC_SEQ_SUP_CNTL
    ("MC_VM_AGP_TOP", 0x080A),
    ("MC_VM_AGP_BOT", 0x080B),
    ("MC_VM_AGP_BASE", 0x080C),
    ("MC_VM_SYSTEM_APERTURE_LOW_ADDR", 0x080D),
    ("MC_VM_SYSTEM_APERTURE_HIGH_ADDR", 0x080E),
    ("VM_L2_CNTL", 0x0500),
    ("VM_CONTEXT0_CNTL", 0x0504),
    ("MC_ARB_RAMCFG", 0x09D8),
    ("MC_SHARED_CHMAP", 0x0801),
    ("MC_SEQ_SUP_CNTL", 0x0A32),
    ("GRBM_STATUS", 0x2004), // gfx_6_0_d.h
    ("GRBM_STATUS2", 0x2002),
    ("SRBM_STATUS", 0x0394), // oss_1_0_d.h
    ("SRBM_STATUS2", 0x0393),
    ("CP_STAT", 0x21A0), // gfx_6_0_d.h
    ("CP_ME_CNTL", 0x21B6),
    ("RLC_CNTL", 0x30C0),
    ("IH_RB_CNTL", 0x0F80), // oss_1_0_d.h
    ("DMA_STATUS_REG", 0x340D),
    ("GB_ADDR_CONFIG", 0x263E), // gfx_6_0_d.h
    ("CC_GC_SHADER_ARRAY_CONFIG", 0x226F),
    ("BIOS_SCRATCH_0", 0x05C9), // bif_3_0_d.h
    ("BIOS_SCRATCH_7", 0x05D0),
    ("BIF_FB_EN", BIF_FB_EN),
    ("HDP_HOST_PATH_CNTL", 0x0B00), // gmc_6_0_d.h
    ("SCRATCH_REG0", 0x2140),       // gfx_6_0_d.h
];

/// VRAM base and size (bytes) in the MC address space from MC_VM_FB_LOCATION (`gmc_v6_0_vram_gtt_location`).
pub fn fb_location(v: u32) -> (u64, u64) {
    let base = ((v & 0xffff) as u64) << 24;
    let top = ((v >> 16) as u64) << 24;
    (base, top.saturating_sub(base) + (1 << 24))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_the_d700_framebuffer_location() {
        // amdgpu: "VRAM: 6144M 0x000000F400000000 - 0x000000F57FFFFFFF", MC_VM_FB_LOCATION 0xf57ff400
        assert_eq!(fb_location(0xf57f_f400), (0xF4_0000_0000, 6144 << 20));
    }
}
