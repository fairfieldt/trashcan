#!/bin/bash
# Record amdgpu's view of a GPU before it moves to vfio-pci: register values through amdgpu's debugfs, the VBIOS
# amdgpu loaded, the VFCT table, and the PCI state. Run as root under the machine's gpuq lease.
# usage: oracle.sh BDF DRI_MINOR OUTDIR
set -euo pipefail
bdf=$1 minor=$2 out=$3
mkdir -p "$out"
dbg=/sys/kernel/debug/dri/$minor
grep -q "$bdf" "$dbg/name"
# dword register offsets from drivers/gpu/drm/amd/include/asic_reg/*/{gmc_6_0,gfx_6_0,bif_3_0,oss_1_0}_d.h
regs="CONFIG_MEMSIZE=0x150A CONFIG_APER_SIZE=0x150C MC_VM_FB_LOCATION=0x0809 MC_VM_FB_OFFSET=0x081A
MC_VM_AGP_TOP=0x080A MC_VM_AGP_BOT=0x080B MC_VM_AGP_BASE=0x080C MC_VM_SYSTEM_APERTURE_LOW_ADDR=0x080D
MC_VM_SYSTEM_APERTURE_HIGH_ADDR=0x080E VM_L2_CNTL=0x0500 VM_CONTEXT0_CNTL=0x0504 MC_ARB_RAMCFG=0x09D8
MC_SHARED_CHMAP=0x0801 MC_SEQ_SUP_CNTL=0x0A32 GRBM_STATUS=0x2004 GRBM_STATUS2=0x2002 SRBM_STATUS=0x0394
SRBM_STATUS2=0x0393 CP_STAT=0x21A0 CP_ME_CNTL=0x21B6 RLC_CNTL=0x30C0 IH_RB_CNTL=0x0F80 DMA_STATUS_REG=0x340D
GB_ADDR_CONFIG=0x263E CC_GC_SHADER_ARRAY_CONFIG=0x226F BIOS_SCRATCH_0=0x05C9 BIOS_SCRATCH_7=0x05D0 BIF_FB_EN=0x1524
HDP_HOST_PATH_CNTL=0x0B00 SCRATCH_REG0=0x2140"
for kv in $regs; do
    name=${kv%=*} off=$((${kv#*=} * 4))
    v=$(dd if="$dbg/amdgpu_regs" bs=4 count=1 skip=$((off / 4)) status=none | od -An -tx4 | tr -d ' ')
    printf '%-32s 0x%04X 0x%s\n' "$name" "$((off / 4))" "$v"
done > "$out/amdgpu-regs.txt"
cat "$out/amdgpu-regs.txt"
[ -e "$dbg/amdgpu_vbios" ] && cat "$dbg/amdgpu_vbios" > "$out/amdgpu-vbios.bin" || true
cat /sys/firmware/acpi/tables/VFCT > "$out/VFCT.bin"
cat /sys/firmware/acpi/tables/DMAR > "$out/DMAR.bin"
lspci -vvv -nn -s "$bdf" > "$out/lspci-vvv.txt"
dmesg > "$out/dmesg.txt"
sha256sum "$out"/*.bin
