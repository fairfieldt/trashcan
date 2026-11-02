# Linux dependencies of the amdgpu Southern Islands path

Audit of Linux 7.3-rc7 for hosting the Tahiti (GCN1, FirePro D700) hardware code in a userspace
driver server behind a Linux-KPI-style C shim. Every number below was measured from the built
objects or grepped from source on 2026-11-02; nothing was rebuilt.

## Method

- Build: x86-64, GCC, Mac Pro config: `CONFIG_DRM_AMDGPU=m`, `CONFIG_DRM_AMDGPU_SI=y`, `CONFIG_DRM_AMDGPU_CIK=y`,
  `CONFIG_DRM_AMD_DC=y`, `CONFIG_DRM_AMD_DC_SI=y`, `CONFIG_HSA_AMD=y`. FTRACE, DYNAMIC_DEBUG, retpoline/rethunk,
  stack protector and FORTIFY_SOURCE are on; KASAN, UBSAN, KCOV and lockdep are off.
- Objects: the 1,039 `.o` files listed in `drivers/gpu/drm/amd/amdgpu/amdgpu.mod`. Imports from
  `x86_64-linux-gnu-nm -u`, definitions from `nm --defined-only` (global symbols only).
- External symbols: exporting module from `Module.symvers`; source file from the `EXPORT_SYMBOL*` site (symbols
  exported from several files were resolved by checking which built `.o` defines them). Subsystem = source directory.
- Call sites: relocations naming the symbol in code sections (`.text*`, `.init.text`, `.exit.text`), from
  `readelf -rW`. Each inlined copy counts; calls through function pointers do not.
- Lines = `wc -l`. Code = non-blank lines after stripping C comments with a small script (approximate).
- Inline code is invisible to `nm`: MMIO `readl`/`writel`, atomics, `list_*`, `kref`, the `spin_lock` wrappers.
  Only their out-of-line callees show (`_raw_spin_lock`, `dma_fence_release` from `dma_fence_put`,
  `ttm_bo_move_to_lru_tail` from `amdgpu_bo_unreserve`).
- "ABI" symbols come from the compiler or config, not from the C code, and are left out of counts unless stated:
  `__fentry__`, `__x86_return_thunk`, `__x86_indirect_thunk_*`, `__stack_chk_fail`, `__ref_stack_chk_guard`,
  `__SCT__*` static calls, `__preempt_count`, `cpu_number`, `const_current_task`, `this_cpu_off`,
  `__per_cpu_offset`, `cpu_info`, `pv_ops`, `BUG_func`.
- Paths: `amdgpu/…`, `pm/…` and `display/…` mean `drivers/gpu/drm/amd/amdgpu/…`, `drivers/gpu/drm/amd/pm/…`
  and `drivers/gpu/drm/amd/display/…`. Other paths are relative to the tree root.

`amdgpu.ko` imports 1,136 symbols: vmlinux 689, drm 211, drm_kms_helper 69, drm_display_helper 64, ttm 47,
gpu-sched 25, drm_exec 7, drm_suballoc_helper 5, cec 4, drm_ttm_helper 4, amdxcp 3, acpi video 3, and one each
from crc16, i2c-algo-bit, drm_panel_backlight_quirks, drm_buddy, drm_client_lib.

## 1. Hardware modules (SI-specific)

Twelve objects form the SI hardware path without display or video. All twelve carry the AMD MIT/X11 permission
text ("Permission is hereby granted, free of charge, to any person obtaining a copy …") and no SPDX tag.
Columns (a)-(d) are distinct undefined symbols: (a) other hardware modules, (b) amdgpu core, (c) DRM, TTM,
dma-fence/dma-resv/dma-buf, gpu-sched, (d) Linux support.

| File | Lines | Code | `nm -u` | (a) | (b) | (c) | (d) | ABI |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| amdgpu/si.c | 2,742 | 2,304 | 50 | 13 | 16 | 1 | 15 | 5 |
| amdgpu/si_ih.c | 325 | 235 | 11 | 0 | 5 | 0 | 4 | 2 |
| amdgpu/si_dma.c | 851 | 579 | 28 | 0 | 17 | 2 | 4 | 5 |
| amdgpu/gmc_v6_0.c | 1,173 | 911 | 45 | 0 | 33 | 2 | 7 | 3 |
| amdgpu/gfx_v6_0.c | 3,843 | 3,224 | 51 | 0 | 27 | 6 | 13 | 5 |
| amdgpu/atom.c | 1,720 | 1,522 | 29 | 0 | 0 | 1 | 18 | 10 |
| amdgpu/amdgpu_atombios.c | 1,963 | 1,579 | 30 | 6 | 6 | 2 | 11 | 5 |
| amdgpu/amdgpu_bios.c | 644 | 448 | 37 | 0 | 4 | 1 | 25 | 7 |
| pm/legacy-dpm/si_dpm.c | 8,185 | 6,985 | 72 | 34 | 13 | 2 | 18 | 5 |
| pm/legacy-dpm/si_smc.c | 301 | 210 | 11 | 0 | 5 | 0 | 4 | 2 |
| pm/legacy-dpm/legacy_dpm.c | 1,010 | 894 | 22 | 4 | 1 | 1 | 11 | 5 |
| pm/amdgpu_dpm_internal.c | 95 | 51 | 3 | 0 | 0 | 1 | 0 | 2 |
| **12 files** | **22,852** | **18,942** | | | | | **76 distinct non-ABI external** | |

`si_set_ip_blocks()` (amdgpu/si.c:2690-2708) registers six IP blocks for Tahiti (si_common, gmc_v6_0, si_ih,
gfx_v6_0, si_dma, si_smu), then display (vkms, DC `dm_ip_block` or `dce_v6_0`) and video (uvd_v3_1, vce_v1_0).
Each of the six has an `amd_ip_funcs` table with 8-14 callbacks (75 in total). Register access goes through
amdgpu core functions, not inline MMIO: 1,010 call sites in the 12 files (`amdgpu_device_wreg` 521,
`amdgpu_device_rreg` 381, `amdgpu_reg_{smc,pcie,pciep}_{rd32,wr32}` 108), all defined in amdgpu/amdgpu_reg_access.c.

Undefined symbols per file. [drop] marks display/video code we plan to drop; "reg_access `rreg/wreg`" means
`amdgpu_device_rreg/wreg` in amdgpu/amdgpu_reg_access.c.

- **amdgpu/si.c**
  - (a) `gfx_v6_0_ip_block`, `gmc_v6_0_ip_block`, `si_dma_ip_block`, `si_ih_ip_block`, `si_smu_ip_block` (si_dpm.o),
    `amdgpu_read_bios` (amdgpu_bios.o), `amdgpu_atombios_scratch_regs_engine_hung`; [drop] `dce_v6_0_ip_block`,
    `dce_v6_4_ip_block` (dce_v6_0.o), `dm_ip_block` (display/amdgpu_dm/amdgpu_dm.o), `amdgpu_vkms_ip_block`,
    `uvd_v3_1_ip_block`, `vce_v1_0_ip_block`.
  - (b) amdgpu_device.c: `amdgpu_device_{has_dc_support,pci_config_reset,pci_reset,program_register_sequence,should_use_aspm}`;
    amdgpu_drv.c: `amdgpu_pcie_gen2`, `amdgpu_reset_method` (module parameters); amdgpu_ip.c: `amdgpu_device_ip_block_add`;
    amdgpu_reg_access.c: `amdgpu_device_{rreg,wreg}`, `amdgpu_reg_{pcie,pciep,smc}_{rd32,wr32}`.
  - (c) `__drm_err`.
  - (d) PCI: `pci_{set,clear}_master`, `pcie_capability_read_{word,dword}`, `pcie_capability_clear_and_set_word_locked`,
    `pcie_{get,set}_readrq`; locking: `mutex_{lock,unlock}`, `_raw_spin_lock_irqsave`, `_raw_spin_unlock_irqrestore`;
    delay: `__const_udelay`, `msleep`; log: `_dev_info`, `_dev_warn`.
- **amdgpu/si_ih.c**: (b) amdgpu_ih.c `amdgpu_ih_ring_init`; amdgpu_irq.c `amdgpu_irq_{init,fini_sw}`; reg_access
  `rreg/wreg`. (d) `pci_set_master`, `__const_udelay`, `_dev_info`, `_dev_warn`.
- **amdgpu/si_dma.c**: (b) amdgpu_ring.c `amdgpu_ring_{alloc,commit,init,fini,insert_nop,test_helper}`; amdgpu_ib.c
  `amdgpu_ib_{get,free,schedule}`; amdgpu_fence.c `amdgpu_fence_process`; amdgpu_irq.c `amdgpu_irq_add_id`;
  amdgpu_wb.c `amdgpu_wb_{get,free}`; amdgpu_ttm.c `amdgpu_sdma_set_buffer_funcs_scheds`; amdgpu_vm.c
  `amdgpu_sdma_set_vm_pte_scheds`; reg_access `rreg/wreg`. (c) `dma_fence_release`, `dma_fence_wait_timeout`.
  (d) `__const_udelay`, `refcount_warn_saturate`, `_dev_info`, `sprintf`.
- **amdgpu/gmc_v6_0.c**: (b) amdgpu_gmc.c `amdgpu_gmc_{gart_location,vram_location,set_gart_size,set_agp_default,vram_checking}`;
  amdgpu_gart.c `amdgpu_gart_init`, `amdgpu_gart_table_vram_{alloc,free}`; amdgpu_object.c `amdgpu_bo_{init,fini,gpu_offset}`;
  amdgpu_vm.c `amdgpu_vm_{manager_init,manager_fini,adjust_size,update_fault_cache}`; amdgpu_ids.c `amdgpu_vmid_reset_all`;
  amdgpu_gtt_mgr.c `amdgpu_gtt_mgr_recover`; amdgpu_gem.c `amdgpu_gem_force_release`; amdgpu_irq.c
  `amdgpu_irq_{add_id,get,put,delegate}`; amdgpu_ucode.c `amdgpu_ucode_{request,release,print_mc_hdr}`; amdgpu_ip.c
  `amdgpu_device_ip_get_ip_block`; amdgpu_device.c `amdgpu_device_resize_fb_bar`; amdgpu_drv.c `amdgpu_emu_mode`,
  `amdgpu_vm_fault_stop`; reg_access `rreg/wreg`, `amdgpu_reg_pcie_{rd32,wr32}`. (c) `__drm_dev_dbg`, `drm_need_swiotlb`.
  (d) `dma_set_mask`, `dma_set_coherent_mask`, `__const_udelay`, `__printk_ratelimit`, `_dev_{err,info,warn}`.
- **amdgpu/gfx_v6_0.c**: (b) amdgpu_ring.c (same six as si_dma), amdgpu_ib.c `amdgpu_ib_{get,free,schedule}`,
  amdgpu_fence.c `amdgpu_fence_process`, amdgpu_gfx.c `amdgpu_gfx_{csb_data_parser,csb_preamble_start,csb_preamble_end,get_num_kcq,parse_disable_cu}`,
  amdgpu_rlc.c `amdgpu_gfx_rlc_{init_sr,fini}`, amdgpu_object.c `amdgpu_bo_create_reserved`, `amdgpu_bo_kunmap`,
  amdgpu_ucode.c `amdgpu_ucode_{request,release,print_gfx_hdr,print_rlc_hdr}`, amdgpu_irq.c `amdgpu_irq_add_id`,
  amdgpu_device.c `amdgpu_get_soft_full_reset_mask`, reg_access `rreg/wreg`. (c) `dma_fence_release`,
  `dma_fence_wait_timeout`, `drm_sched_fault`, `ttm_bo_move_to_lru_tail`, `__drm_dev_dbg`, `__drm_err`.
  (d) `mutex_{lock,unlock}`, `ww_mutex_unlock`, `_raw_spin_{lock,unlock}`, `__const_udelay`, `__sw_hweight32`,
  `refcount_warn_saturate`, `_dev_{err,info,warn}`, `_printk`, `sprintf`.
- **amdgpu/atom.c** (AtomBIOS interpreter): (c) `__drm_err`. (d) kmalloc family + `kfree`, `mutex_{lock,unlock}`,
  `__const_udelay`, `__udelay`, `msleep`, `jiffies`, `memcpy`, `strncmp`, `strnlen`, `sized_strscpy`, `scnprintf`,
  `__fortify_panic`, `_dev_info`, `_printk`. No amdgpu core dependency.
- **amdgpu/amdgpu_atombios.c**: (a) atom.o `amdgpu_atom_{execute_table,parse,parse_cmd_header,parse_data_header}`;
  [drop] `amdgpu_i2c_create` (amdgpu_i2c.o), `amdgpu_link_encoder_connector` (amdgpu_encoders.o). (b)
  amdgpu_atomfirmware.c `amdgpu_atomfirmware_{allocate_fb_scratch,query_firmware_capability,scratch_regs_init}`;
  amdgpu_ttm.c `amdgpu_ttm_init_vram_resv`; reg_access `rreg/wreg`. (c) `__drm_dev_dbg`, `__drm_err`. (d) kmalloc
  family, `kfree`, `mutex_init_generic`, `memcpy`, `sprintf`, `strnlen`, `__fortify_panic`, `sysfs_emit`,
  `devm_device_add_group`.
- **amdgpu/amdgpu_bios.c** (VBIOS fetch): (b) amdgpu_device.c `amdgpu_device_need_post`; amdgpu_virt.c
  `amdgpu_virt_get_dynamic_data_info`; reg_access `rreg/wreg`. (c) `__drm_err`. (d) ACPI (VFCT table, ATRM):
  `acpi_{get_table,put_table,get_handle,evaluate_object,format_exception}`, `is_acpi_device_node`; PCI:
  `pci_{map_rom,unmap_rom,get_base_class,dev_put}`, `vga_default_device`; `ioremap`, `ioremap_wc`, `iounmap`,
  `memcpy_fromio`; kmalloc family, `kmemdup_noprof`, `kfree`; `__dynamic_dev_dbg`, `_dev_{err,info,notice}`,
  `dump_stack`; `memcpy`.
- **pm/legacy-dpm/si_dpm.c**: (a) 11 from amdgpu_atombios.o (`amdgpu_atombios_get_{asic_ss_info,clock_dividers,default_voltages,leakage_vddc_based_on_leakage_idx,max_vddc,memory_pll_dividers,svi2_info,voltage_table}`,
  `_init_mc_reg_table`, `_is_voltage_gpio`, `_set_engine_dram_timings`), 11 from si_smc.o (`amdgpu_si_*`), 11 from
  legacy_dpm.o (power-table parsing, thermal work handler, `amdgpu_legacy_dpm_compute_clocks`), `amdgpu_atom_parse_data_header`.
  (b) amdgpu_acpi.c `amdgpu_acpi_is_pcie_performance_request_supported`, `amdgpu_acpi_pcie_performance_request`;
  amdgpu_ip.c `amdgpu_device_ip_set_{clockgating,powergating}_state`; amdgpu_irq.c `amdgpu_irq_add_id`; amdgpu_ucode.c
  `amdgpu_ucode_{request,release}`; amdgpu_drv.c `amdgpu_dpm`; reg_access `rreg/wreg`, `amdgpu_reg_pciep_rd32`,
  `amdgpu_reg_smc_{rd32,wr32}`. (c) `__drm_dev_dbg`, `__drm_err`. (d) workqueue `queue_work_on`, `cancel_work_sync`,
  `flush_work`, `system_percpu_wq`; `mutex_{lock,unlock}`; kmalloc family, `kfree`; `memset`; `__sw_hweight32`;
  `seq_printf`; `__dynamic_dev_dbg`, `_dev_{err,info}`, `_printk`, `dump_stack`.
- **pm/legacy-dpm/si_smc.c**: (b) reg_access `rreg/wreg`, `amdgpu_reg_smc_{rd32,wr32}`; amdgpu_ucode.c
  `amdgpu_ucode_print_smc_hdr`. (d) `_raw_spin_lock_irqsave`, `_raw_spin_unlock_irqrestore`, `__const_udelay`, `_dev_warn`.
- **pm/legacy-dpm/legacy_dpm.c**: (a) `amdgpu_atombios_lookup_i2c_gpio`, `amdgpu_atom_parse_data_header`,
  `amdgpu_dpm_get_display_cfg` (amdgpu_dpm_internal.o); [drop] `amdgpu_i2c_lookup` (amdgpu_i2c.o). (b) amdgpu_drv.c
  `amdgpu_dpm`. (c) `__drm_dev_dbg`. (d) `i2c_new_client_device` (thermal sensor), `mutex_{lock,unlock}`, kmalloc
  family, `kfree`, `sized_strscpy`, `strnlen`, `__fortify_panic`, `_dev_info`.
- **pm/amdgpu_dpm_internal.c**: (c) `drm_mode_vrefresh` only. It walks CRTCs to build the display config that DPM
  uses for memory-clock switching, so DPM depends on KMS state.

Display and video modules (all [drop]; MIT/X11 text, `vce_v1_0.c` also `SPDX-License-Identifier: MIT`):

| File | Lines | Code | `nm -u` | (a) | (b) | (c) | (d) | Notable imports |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| amdgpu/dce_v6_0.c | 3,507 | 2,665 | 125 | 51 | 20 | 28 | 21 | 20 drm.ko KMS (vblank, EDID, CRTC), 3 drm_kms_helper, delayed work, timers |
| amdgpu/atombios_crtc.c | 885 | 736 | 14 | 9 | 0 | 1 | 0 | atom/atombios, amdgpu_pll |
| amdgpu/atombios_encoders.c | 2,126 | 1,813 | 40 | 14 | 3 | 6 | 12 | backlight, acpi video, EDID |
| amdgpu/atombios_dp.c | 770 | 582 | 30 | 7 | 0 | 14 | 5 | 12 `drm_dp_*` (drm_display_helper) |
| amdgpu/atombios_i2c.c | 176 | 126 | 11 | 2 | 0 | 2 | 3 | atom only |
| amdgpu/uvd_v3_1.c | 859 | 542 | 34 | 9 | 14 | 2 | 6 | 9 from amdgpu_uvd.c |
| amdgpu/vce_v1_0.c | 854 | 576 | 43 | 12 | 18 | 0 | 10 | 12 from amdgpu_vce.c, `memcpy_toio` |

The dependency between hardware and display code runs one way: display calls into atom/atombios, while the 12
hardware files reach display only through si.c's IP block symbols, `amdgpu_i2c_create`,
`amdgpu_link_encoder_connector`, `amdgpu_i2c_lookup` and `drm_mode_vrefresh`.

## 2. Firmware

Every SI firmware request goes through `amdgpu_ucode_request()` (amdgpu/amdgpu_ucode.c:1505), which calls
`request_firmware()` or `firmware_request_nowarn()`. si_ih and si_dma load no firmware; SI has no PSP, so every upload is done by the driver.

| Tahiti file | Requested by | Uploaded by | Path to hardware |
|---|---|---|---|
| `amdgpu/tahiti_mc.bin` (`amdgpu/si58_mc.bin` if `MC_SEQ_MISC0[31:24] == 0x58`) | gmc_v6_0 (`gmc_v6_0.c:134`) | `gmc_v6_0_mc_load_microcode`: `MC_SEQ_IO_DEBUG_INDEX/DATA` pairs, then `MC_SEQ_SUP_PGM`; skipped when `MC_SEQ_SUP_CNTL.RUN` is already set | MMIO |
| `amdgpu/tahiti_pfp.bin`, `_me.bin`, `_ce.bin` | gfx_v6_0 (`gfx_v6_0.c:362-380`) | `gfx_v6_0_cp_gfx_load_microcode`: `CP_PFP_UCODE_ADDR/DATA`, `CP_ME_RAM_WADDR/DATA`, `CP_CE_UCODE_ADDR/DATA` | MMIO |
| `amdgpu/tahiti_rlc.bin` | gfx_v6_0 (`gfx_v6_0.c:389`) | `gfx_v6_0_rlc_resume`: `RLC_UCODE_ADDR/DATA` | MMIO; the RLC also needs save/restore and clear-state buffers in VRAM (`amdgpu_gfx_rlc_init_sr`, `amdgpu_bo_create_reserved`) |
| `amdgpu/tahiti_smc.bin` | si_dpm (`si_dpm.c:7790`) | `si_upload_firmware` → `amdgpu_si_load_smc_ucode` (`si_smc.c:231`): `SMC_IND_INDEX_0`/`SMC_IND_DATA_0` into SMC SRAM | MMIO, indirect |
| `amdgpu/tahiti_uvd.bin` [drop] | `amdgpu_uvd.c:264` | `memcpy_toio` into a VRAM BO (`amdgpu_uvd.c:511`) | memory |
| `amdgpu/vce_1_0_0.bin` [drop] | `amdgpu_vce.c:176` | `memcpy_toio` into a VRAM BO (`amdgpu_vce.c:341`, `vce_v1_0.c:238-245`) | memory |

`MODULE_FIRMWARE` lines: `gfx_v6_0.c:63-66`, `gmc_v6_0.c:48` and `:53`, `si_dpm.c:79`, `amdgpu_uvd.c:112` (name defined
at `:60`), `amdgpu_vce.c:68` (name defined at `:45`).

## 3. amdgpu core files the SI blocks call into

Closures follow global-symbol references between the 1,039 objects, starting from the 12 hardware files.

- **Unpruned**: follow every reference. Reached 1,032 of 1,039 objects (all but `amdgpu_lockdep`, `vcn_sw_ring` and
  five DC files). Breadth-first, the first layer already includes display (31 objects: 24 core, 4 display, 3 SI
  display/video). `amdgpu/amdgpu_device.c` references 60 other objects and is referenced by 129.
- **Pruned**: follow references only into hardware or core objects. Excluded: DC and other display (593), video (35),
  other-ASIC IP (153), KFD (56), RAS (32), swsmu/powerplay (77).
- **Minimal**: pruned, and also not following 31 core files for hardware or features SI does not have: PSP
  (psp, psp_ta, rap, securedisplay, fw_attestation), MES and user queues (mes, mes_userqueue, userq, userq_fence,
  eviction_fence, umsch_mm, seq64), SR-IOV (virt, vf_error), XGMI, partitions (xcp), IP discovery, VPE, HMM userptr,
  PMU, coredump, debugfs, benchmark, ATPX, EEPROM/FRU, ring_mux, fdinfo, ioc32, trace_points, amdgpu_pm (sysfs/hwmon).
  This list is a judgement call, not a measurement. References into excluded files become stubs.

| Set | Objects | Lines | Code | External symbols (non-ABI) | Unresolved refs to stub |
|---|--:|--:|--:|--:|--:|
| 12 hardware files | 12 | 22,852 | 18,942 | 76 | - |
| + core files they reference directly | 36 (24 core) | 61,476 | 44,657 | 463 | - |
| Minimal closure | 58 (46 core) | 72,817 | 52,174 | 529 | 332 symbols (75 are amdgpu tracepoints) |
| Pruned closure | 89 (77 core) | 105,313 | 76,061 | 631 | 273 symbols |
| Unpruned closure | 1,032 | 1,015,162 | 773,890 | 1,106 | 0 |

Line counts cover `.c` files only. The 12 hardware objects also compile against 1,035 headers (369,052 lines,
from the `.o.cmd` dependency lists), of which 443 are `include/linux/*` and 189 are amd headers (122,249 lines,
including 30 DC and 18 pm headers). The minimal closure uses 1,167 headers (461,727 lines).

Core files of the minimal closure. DRM/TTM/sched/fence/Linux = distinct non-ABI undefined symbols;
"refs" = number of other amdgpu objects referenced (any group).

| File | Lines | Code | DRM | TTM | sched | fence | Linux | refs |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| amdgpu/amdgpu_device.c | 6,886 | 4,570 | 13 | 2 | 3 | 5 | 118 | 60 |
| amdgpu/amdgpu_drv.c | 3,251 | 2,192 | 21 | 0 | 0 | 0 | 45 | 30 |
| amdgpu/amdgpu_vm.c | 3,242 | 1,864 | 9 | 5 | 4 | 11 | 37 | 19 |
| amdgpu/amdgpu_gfx.c | 3,098 | 2,212 | 1 | 1 | 5 | 2 | 49 | 16 |
| amdgpu/amdgpu_ttm.c | 3,009 | 2,017 | 4 | 23 | 3 | 6 | 62 | 15 |
| pm/amdgpu_dpm.c | 2,137 | 1,577 | 0 | 0 | 0 | 0 | 9 | 5 |
| amdgpu/amdgpu_kms.c | 2,036 | 1,608 | 6 | 2 | 0 | 0 | 35 | 25 |
| amdgpu/amdgpu_cs.c | 1,834 | 1,338 | 15 | 3 | 4 | 6 | 24 | 16 |
| amdgpu/amdgpu_gmc.c | 1,794 | 1,223 | 2 | 1 | 0 | 2 | 27 | 22 |
| amdgpu/amdgpu_object.c | 1,709 | 974 | 8 | 9 | 0 | 9 | 23 | 11 |
| amdgpu/amdgpu_acpi.c | 1,619 | 1,023 | 2 | 0 | 0 | 0 | 42 | 2 |
| amdgpu/amdgpu_ucode.c | 1,551 | 1,405 | 2 | 0 | 0 | 0 | 15 | 2 |
| amdgpu/amdgpu_gem.c | 1,393 | 1,031 | 20 | 9 | 0 | 5 | 39 | 9 |
| amdgpu/amdgpu_ring.c | 1,155 | 719 | 2 | 0 | 4 | 4 | 28 | 9 |
| amdgpu/amdgpu_atomfirmware.c | 1,117 | 887 | 1 | 0 | 0 | 0 | 4 | 3 |
| amdgpu/amdgpu_fence.c | 1,050 | 583 | 3 | 0 | 1 | 5 | 34 | 3 |
| amdgpu/amdgpu_vram_mgr.c | 1,037 | 658 | 8 | 6 | 0 | 1 | 21 | 0 |
| amdgpu/amdgpu_reg_access.c | 982 | 713 | 0 | 0 | 0 | 0 | 11 | 4 |
| amdgpu/amdgpu_vm_pt.c | 967 | 499 | 2 | 2 | 0 | 0 | 6 | 4 |
| amdgpu/amdgpu_ctx.c | 949 | 712 | 4 | 0 | 9 | 3 | 23 | 6 |
| amdgpu/amdgpu_irq.c | 811 | 442 | 0 | 0 | 0 | 0 | 27 | 5 |
| amdgpu/amdgpu_ids.c | 706 | 377 | 0 | 0 | 0 | 6 | 22 | 2 |
| amdgpu/amdgpu_dma_buf.c | 655 | 362 | 8 | 3 | 0 | 2 | 16 | 7 |
| amdgpu/amdgpu_sdma.c | 625 | 455 | 1 | 0 | 2 | 0 | 12 | 9 |
| amdgpu/amdgpu_rlc.c | 624 | 463 | 0 | 1 | 0 | 0 | 6 | 3 |
| amdgpu/amdgpu_ib.c | 566 | 323 | 2 | 0 | 0 | 1 | 14 | 7 |
| amdgpu/amdgpu_gart.c | 538 | 282 | 2 | 2 | 0 | 0 | 22 | 4 |
| amdgpu/amdgpu_job.c | 530 | 385 | 3 | 0 | 8 | 3 | 17 | 12 |
| amdgpu/amdgpu_ip.c | 526 | 321 | 0 | 0 | 0 | 0 | 3 | 2 |
| amdgpu/amdgpu_sync.c | 514 | 275 | 0 | 0 | 2 | 9 | 7 | 2 |
| amdgpu/amdgpu_gtt_mgr.c | 423 | 225 | 5 | 5 | 0 | 1 | 7 | 1 |
| amdgpu/amdgpu_reset.c | 355 | 271 | 1 | 0 | 0 | 0 | 20 | 10 |
| amdgpu/amdgpu_ih.c | 317 | 178 | 0 | 0 | 0 | 0 | 12 | 3 |
| amdgpu/amdgpu_vm_sdma.c | 305 | 167 | 0 | 0 | 1 | 4 | 3 | 6 |
| amdgpu/amdgpu_bo_list.c | 259 | 180 | 2 | 0 | 0 | 0 | 13 | 3 |
| amdgpu/amdgpu_doorbell_mgr.c | 244 | 103 | 0 | 0 | 0 | 0 | 1 | 2 |
| amdgpu/amdgpu_preempt_mgr.c | 153 | 65 | 1 | 5 | 0 | 1 | 7 | 0 |
| amdgpu/amdgpu_vm_cpu.c | 143 | 66 | 0 | 0 | 0 | 1 | 4 | 6 |
| amdgpu/amdgpu_csa.c | 139 | 90 | 4 | 0 | 0 | 0 | 2 | 2 |
| amdgpu/amdgpu_wb.c | 129 | 54 | 0 | 0 | 0 | 0 | 5 | 1 |
| amdgpu/amdgpu_sched.c | 129 | 83 | 1 | 0 | 0 | 0 | 7 | 2 |
| amdgpu/amdgpu_sa.c | 121 | 61 | 7 | 0 | 0 | 0 | 2 | 1 |
| amdgpu/amdgpu_vm_tlb_fence.c | 111 | 66 | 0 | 0 | 0 | 5 | 6 | 1 |
| amdgpu/amdgpu_nbio.c | 96 | 60 | 0 | 0 | 0 | 0 | 1 | 3 |
| amdgpu/amdgpu_hdp.c | 84 | 53 | 0 | 0 | 0 | 0 | 1 | 2 |
| amdgpu/amdgpu_mmhub.c | 46 | 20 | 0 | 0 | 0 | 0 | 1 | 1 |
| **46 core files** | **49,965** | **33,232** | | | | | | |

The pruned closure adds 31 files (32,496 lines): amdgpu_pm.c 5,119, amdgpu_psp.c 5,075, amdgpu_discovery.c 3,760,
amdgpu_debugfs.c 2,380, amdgpu_virt.c 2,128, amdgpu_userq.c 1,576, amdgpu_xgmi.c 1,520, amdgpu_mes.c 1,171,
amdgpu_xcp.c 1,129, amdgpu_vpe.c 1,049, amdgpu_userq_fence.c 1,004, and 20 smaller files.

The 332 unresolved symbols of the minimal closure, by defining object: amdgpu_trace_points 75, amdgpu_ras 31,
amdgpu_smu 28, amdgpu_virt 23 (including `amdgpu_sriov_wreg`, called from amdgpu_reg_access.c), amdgpu_amdkfd 19,
amdgpu_userq 15, amdgpu_xcp 12, amdgpu_xgmi 11, amdgpu_mes 11, and 1-6 each in 46 more objects (55 in all; display:
`amdgpu_display_supported_domains` from amdgpu_gem.c and amdgpu_dma_buf.c, `amdgpu_i2c_*`; other ASICs: reset handlers of aldebaran,
sienna_cichlid, smu_v13_0_10).

External symbols by subsystem (distinct symbols; "sites" = code call sites in the minimal closure):

| Subsystem | 12 HW files | Minimal | Sites (minimal) | Pruned | All amdgpu.ko |
|---|--:|--:|--:|--:|--:|
| DRM core (drm.ko: GEM, PRIME, syncobj, drm_mm, file/ioctl, print) | 3 | 48 | 414 | 54 | 61 |
| DRM KMS/display (drm.ko KMS, kms_helper, display_helper, client) | 1 | 17 | 28 | 17 | 285 |
| DRM helpers (drm_exec, drm_suballoc, drm_ttm_helper, gpu_buddy) | 0 | 22 | 69 | 23 | 23 |
| TTM | 1 | 47 | 133 | 47 | 47 |
| gpu-sched | 1 | 23 | 83 | 23 | 25 |
| dma-fence | 2 | 14 | 180 | 17 | 18 |
| dma-resv | 0 | 10 | 29 | 11 | 12 |
| dma-buf / sync_file | 0 | 7 | 8 | 7 | 10 |
| PCI | 12 | 47 | 99 | 49 | 51 |
| locking (mutex, spinlock, ww_mutex, rwsem, semaphore) | 8 | 24 | 893 | 27 | 29 |
| memory: kmalloc, slab, pages, vmalloc | 6 | 17 | 398 | 17 | 25 |
| memory: user mm (mmap, HMM, notifiers, page tables) | 0 | 10 | 13 | 14 | 34 |
| lib: lists, rbtree, bitmap, IDR/xarray, sort | 1 | 25 | 312 | 32 | 67 |
| string/format | 8 | 19 | 123 | 26 | 33 |
| printk/log | 8 | 11 | 852 | 11 | 13 |
| time, timers, delays | 4 | 14 | 124 | 15 | 21 |
| workqueue, kthread | 4 | 13 | 57 | 15 | 25 |
| wait queues, completions | 0 | 5 | 5 | 5 | 15 |
| RCU/SRCU | 0 | 4 | 102 | 4 | 8 |
| DMA mapping, scatterlist, IOMMU | 2 | 16 | 36 | 16 | 16 |
| IRQ | 0 | 10 | 10 | 10 | 10 |
| firmware loading | 0 | 3 | 5 | 3 | 3 |
| arch: ioremap, PAT/WC, cpu, FPU | 5 | 21 | 72 | 25 | 40 |
| ACPI, DMI, platform | 6 | 17 | 35 | 17 | 26 |
| power management (runtime PM, suspend) | 0 | 18 | 50 | 18 | 21 |
| sysfs, debugfs, seq_file | 2 | 20 | 158 | 36 | 40 |
| device model, VFS (fd), module params | 1 | 22 | 47 | 35 | 61 |
| uaccess | 0 | 6 | 45 | 9 | 10 |
| tracing, perf | 0 | 4 | 28 | 24 | 26 |
| tasks, creds, CPU masks, misc | 0 | 6 | 34 | 8 | 21 |
| VGA, fbdev, backlight, I2C, CEC, hwmon | 1 | 8 | 9 | 13 | 26 |
| other modules (amdxcp, crc16) | 0 | 1 | 1 | 3 | 4 |
| **total non-ABI** | **76** | **529** | **4,452** | **631** | **1,106** |

All 47 TTM symbols that amdgpu imports are reachable from the minimal closure.

## 4. DRM ioctl surface

Sources grepped (not built or run): Mesa 25.0.7 (release tarball) and libdrm 2.4.124, the version Debian 13
ships as `libdrm-amdgpu1 2.4.124-2`. Files: `src/amd/vulkan/winsys/amdgpu/*.c`, `src/amd/common/ac_gpu_info.c`,
`src/amd/common/ac_linux_drm.c`, `src/vulkan/runtime/vk_drm_syncobj.c`, libdrm `amdgpu/*.c`. In Mesa 25.0.7 RADV
calls `ac_drm_*` wrappers in `ac_linux_drm.c`; these issue most amdgpu ioctls directly and use libdrm_amdgpu only for
device init, BO alloc/map/import/export/userptr and the userspace VA allocator.

amdgpu driver ioctls (`amdgpu_ioctls_kms[]`, amdgpu/amdgpu_drv.c:3094-3115): 21, of which 20 are allowed on render
nodes (`DRM_AUTH|DRM_RENDER_ALLOW`); `AMDGPU_SCHED` is `DRM_MASTER` only.

| ioctl | Handler | RADV (Mesa 25.0.7) |
|---|---|---|
| GEM_CREATE | amdgpu_gem.c | yes, via libdrm `amdgpu_bo_alloc` |
| GEM_MMAP | amdgpu_gem.c | yes, via libdrm `amdgpu_bo_cpu_map`, followed by `mmap()` on the DRM fd |
| GEM_VA | amdgpu_gem.c | yes: MAP, UNMAP, REPLACE, CLEAR |
| GEM_METADATA | amdgpu_gem.c | yes: set on export, get on import |
| GEM_OP | amdgpu_gem.c | yes: `GET_GEM_CREATE_INFO` in `ac_drm_bo_query_info` (imports) |
| GEM_USERPTR | amdgpu_gem.c | yes, via libdrm, for `VK_EXT_external_memory_host` |
| CTX | amdgpu_ctx.c | yes: alloc, free; stable-pstate get/set for profiling |
| CS | amdgpu_cs.c | yes: chunks IB, FENCE, BO_HANDLES, SYNCOBJ_IN/OUT, SYNCOBJ_TIMELINE_WAIT/SIGNAL |
| WAIT_CS | amdgpu_cs.c | yes (`ac_drm_cs_query_fence_status`) |
| INFO | amdgpu_kms.c | yes (see below) |
| VM | amdgpu_vm.c | only with RADV's reserve-VMID option (SQTT) |
| BO_LIST, FENCE_TO_HANDLE, WAIT_FENCES, GEM_WAIT_IDLE | bo_list.c, cs.c, cs.c, gem.c | no (libdrm has wrappers) |
| SCHED | amdgpu_sched.c | no |
| USERQ, USERQ_SIGNAL, USERQ_WAIT | userq.c, userq_fence.c | no (wrappers exist in ac_linux_drm.c, unused by RADV) |
| GEM_LIST_HANDLES, PROC_OPTIONS | amdgpu_gem.c, amdgpu_kms.c | no (not in Mesa 25.0.7) |

RADV uses 11 of the 21. Render-node core ioctls (`DRM_RENDER_ALLOW` in `drivers/gpu/drm/drm_ioctl.c`): 19.

| Core ioctl | RADV path |
|---|---|
| VERSION | yes (RADV requires name `amdgpu`; libdrm requires major version 3) |
| GET_CAP | yes: `DRM_CAP_SYNCOBJ`, `DRM_CAP_SYNCOBJ_TIMELINE` (and `ADDFB2_MODIFIERS`, a KMS cap) |
| GEM_CLOSE | yes (`drmCloseBufferHandle`) |
| PRIME_HANDLE_TO_FD, PRIME_FD_TO_HANDLE | yes (dma-buf export/import) |
| SYNCOBJ_CREATE, DESTROY, HANDLE_TO_FD, FD_TO_HANDLE, TRANSFER, WAIT, TIMELINE_WAIT, RESET, SIGNAL, TIMELINE_SIGNAL, QUERY | yes (`vk_drm_syncobj` plus `ac_linux_drm.c`; sync-file import/export use the HANDLE_TO_FD/FD_TO_HANDLE flags) |
| SET_CLIENT_NAME, SYNCOBJ_EVENTFD | no |
| GEM_CHANGE_HANDLE | no; mapped to `drm_invalid_op` in this tree |

Not ioctls but still needed: device enumeration (libdrm `drmGetDevices2` walks `/dev/dri` and reads the PCI identity
from sysfs; `drmGetNodeTypeFromFd` uses the char-device major/minor), and `mmap()` of BO offsets on the DRM fd.
GEM_FLINK, GEM_OPEN and GET_CLIENT appear in libdrm_amdgpu but are not reached from RADV on a render node.

`AMDGPU_INFO` handles 27 top-level queries (79 `case` labels including FW, SENSOR, VBIOS and VIDEO_CAPS subcodes).
RADV plus libdrm init issue 19: ACCEL_WORKING, DEV_INFO, MEMORY, MAX_IBS, FW_VERSION (GFX_ME/PFP/CE/MEC, UVD, VCE, VCN),
HW_IP_COUNT, HW_IP_INFO, VIDEO_CAPS (DECODE, ENCODE), READ_MMR_REG, VRAM_GTT, VRAM_USAGE, VIS_VRAM_USAGE, GTT_USAGE,
SENSOR (GFX_SCLK, GFX_MCLK, GPU_TEMP), TIMESTAMP, NUM_BYTES_MOVED, NUM_EVICTIONS, NUM_VRAM_CPU_PAGE_FAULTS,
GPUVM_FAULT. UQ_FW_AREAS is queried only on GFX11+ with `AMD_USERQ`. On SI, libdrm's `amdgpu_query_gpu_info_init`
reads registers through READ_MMR_REG: 0x263d (CC_RB_BACKEND_DISABLE) and 0xa0d4 (PA_SC_RASTER_CONFIG) per shader
engine, 0x263e (GB_ADDR_CONFIG), 0x2644 x32 (GB_TILE_MODE0-31), 0x9d8 (MC_ARB_RAMCFG). The kernel serves these from
`si_allowed_read_registers[]` (amdgpu/si.c:1120).

## 5. Assessment

Minimal path: MMIO init with no display or video (si_common, gmc_v6_0, si_ih, gfx_v6_0, si_dma, si_dpm) plus the
core needed to serve RADV's 11 ioctls. Call sites are code relocations in the 58-object minimal closure
(pruned-closure count in brackets where it differs). "Replace" means the shim can supply a simpler implementation
with the same call signature; "emulate" means callers depend on the semantics.

| Dependency | Call sites | Where | Strategy |
|---|--:|---|---|
| TTM: BO init, placement, validate/evict, LRU, pool, kmap, VM fault (`ttm_*`, 44 of 47 imported symbols have call sites) | 133 [156] | amdgpu_ttm.c 42, amdgpu_object.c 23, amdgpu_vm.c 10, amdgpu_gem.c 9, amdgpu_vram_mgr.c 9, gtt_mgr 7 | Kernel BOs (`amdgpu_bo_create_kernel/_reserved/_at`: 21 calls, 21 `amdgpu_bo_free_kernel`; 1 in the hardware files): **replace** with a static pinned VRAM/GTT carve-out. User BOs for RADV need eviction-free placement, CPU mapping and GART binding: **emulate** a subset. `ttm_bo_move_to_lru_tail` (30) comes from inline `amdgpu_bo_unreserve` and can be a no-op without eviction. TTM itself is 7,628 lines and imports 150 symbols (shmem, shrinkers, list_lru, x86 `set_pages_array_wc`, AGP). |
| gpu-sched (`drm_sched_*`, 23 symbols) | 83 [88] | amdgpu_ttm.c 14, amdgpu_vm.c 13, amdgpu_ctx.c 11, amdgpu_job.c 10, amdgpu_sync.c 7, amdgpu_ring.c 6 | Ring and IB tests bypass the scheduler (`amdgpu_ib_schedule`, 29 ring/IB/fence calls in the hardware files), so bring-up does not need it. User CS (entity, job, dependencies, timeout/fault) and the VM/TTM SDMA entities: **emulate**; gpu-sched is 2,682 lines and imports 85 symbols. `drm_sched_fault` in gfx_v6_0: stub. |
| dma-fence (14 symbols) | 180 [240] | `dma_fence_release` 101 (inline `dma_fence_put`), `dma_fence_signal` 25, `dma_fence_wait_timeout` 22 | **Emulate**: refcount, signal, callbacks, timed wait. `drivers/dma-buf/dma-fence*.c` (2,047 lines) is GPL-2.0-only: importing it makes gpud GPL-2.0 as a whole, which the license plan allows (README), but it assumes RCU and irq_work; a small rewrite of the subset used is the alternative. |
| dma-resv (10) + ww_mutex (4) + drm_exec (6) | 29 + 65 + 52 | BO reservation in object/vm/cs/gem | **Emulate**: wound/wait `-EDEADLK` backoff in `drm_exec` loops. With a single submission thread this can become one big lock. dma-resv.c is MIT (818 lines). |
| IP block framework | 34 sites into amdgpu_ip.c and 26 into amdgpu_device.c from the hardware files | `amdgpu_device_ip_block_add` (28, si.c), `ip_get_ip_block`, `set_{clock,power}gating_state` | **Replace**: call the six `amd_ip_funcs` tables in order. amdgpu_device.c (6,886 lines, 118 Linux symbols, refs to 60 objects, 129 referrers) is the hub that drags in PSP, discovery, KFD, RAS, SR-IOV and XGMI; taking it whole means stubbing most of the 332 unresolved symbols. |
| mutex / spinlock / rwsem+semaphore | 429 / 359 / 40 [710 / 474 / 54] | everywhere | **Replace** with pthread primitives. Spinlocks taken with IRQs off (`_irqsave`: 84) also guard state shared with the IH handler; that needs a defined interrupt-thread model, not a no-op. |
| RCU | 102 | mostly inline `dma_fence_get_rcu`/resv iteration | **Replace** read side with no-ops; `call_rcu` (3) and `rcu_barrier` (1) as deferred free. |
| workqueue; time and timers; udelay | 57; 58 (ktime/jiffies 38, timer API 9, msleep/usleep 11); 66 | delayed work for fence fallback, IH, ring idle, thermal; 58 of the udelay sites are in the hardware files | **Emulate** workqueue (a worker thread with delayed queueing and `cancel_*_sync`); delays and clocks: **replace** with busy-wait, sleep, `clock_gettime`. |
| request_firmware | 5 (3 symbols) | amdgpu_ucode.c only | **Replace**: read a file. Six Tahiti blobs (§2). |
| PCI (`pci_*`, `pcie_*`, `vga_*`) | 99 (47 symbols); 24 in hardware files | config space, power state, save/restore, ROM, BAR resize, link caps | **Replace** with VFIO or sysfs config access; most of the 47 can fail or no-op. si.c's PCIe gen2/ASPM code needs real config writes. |
| DMA mapping + scatterlist | 31 (14 symbols) | TTM pool, GART bind, dma-buf | **Replace** with a static pinned, IOMMU-mapped (VFIO) system-memory pool for GTT. |
| IRQ | 10 (10 symbols) | amdgpu_irq.c: `request_threaded_irq`, irq_domain | **Replace** with a VFIO eventfd or IH-ring polling thread. |
| ioremap / MMIO | 26 (`ioremap*`, `iounmap`, `memcpy_*io`, `memset_io`, `ioread8`); `readl`/`writel` are inline | `amdgpu_device_rreg/wreg` (902 sites in hardware files) | **Replace**: mmap BAR2 (SI's register BAR; BAR5 is Bonaire and later). Register access already funnels through amdgpu/amdgpu_reg_access.c. |
| uaccess, user mm, HMM | 45 / 13 | ioctl copy-in/out; GEM_USERPTR | Copy-in/out: **replace** with server-side buffers. Userptr (`VK_EXT_external_memory_host`) maps client pages: no simple stub; reject or **emulate** with shared memory. |
| ACPI, runtime PM, sysfs/debugfs/seq_file, tracing; module params | 35 / 50 / 158 / 28; 9 param symbols | amdgpu_acpi.c, amdgpu_drv.c, amdgpu_bios.c | **Stub**. VBIOS must still be fetched: amdgpu_bios.c tries ACPI VFCT, ATRM, then PCI ROM. |
| printk, kmalloc, string, lib | 852 / 398 / 123 / 312 | everywhere | **Replace**, mechanical (kmalloc to malloc; `list_*`, rbtree, xarray/IDR from FreeBSD-LinuxKPI-style headers). |

Hardest, in order:

1. **TTM + amdgpu_object/ttm/vram_mgr/gtt_mgr** (133 TTM call sites; 7,628 lines of TTM). Kernel BOs are easy
   to serve from a pinned carve-out. User BOs are not, because RADV needs CPU mapping into another process.
2. **dma-fence/dma-resv/ww_mutex/drm_exec and gpu-sched together** (180 + 29 + 65 + 52 + 83 sites). These are one
   synchronization model; the fence code is GPL-2.0-only (importable under gpud's GPL terms, but built on RCU).
3. **amdgpu_device.c as the IP framework**: 60 outgoing object references. Replacing it with a small sequencer over
   the six `amd_ip_funcs` tables avoids most of the 332 unresolved symbols.
4. **Header surface**: the 12 hardware files alone include 443 `include/linux` headers. A FreeBSD-style
   `linuxkpi` header tree is needed before the files compile at all.

Easy by measurement: firmware loading (5 sites, 1 file), IRQ (10 sites), MMIO (one funnel), and atom.c (no
amdgpu core dependency; kmalloc, mutex, delays and printk only).
