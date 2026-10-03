#!/bin/bash
# Builds Debian's 6.12 kernel (linux-source-6.12) on incite with:
#   0001/0002  the D700 fixes (exact-match VFCT VBIOS selection; SI GPIO DC only on mobile parts),
#              the same changes as ~/D700macpro and the hand-built amdgpu.ko in /lib/modules/*/updates
#   0003       Ivy Bridge-EP (8086:0e00) in the PCI P2PDMA host bridge whitelist
# and config changes:
#   PCI_P2PDMA=y          peer-to-peer DMA between the two D700s' BARs
#   DMABUF_MOVE_NOTIFY=y  without it, every dma-buf map pins the exporter's BO into GTT
#                         (amdgpu_dma_buf_pin), so cross-GPU imports never stay in VRAM
#   no debug info / BTF   halves build time
# Output: ~/trashcan/kernel/linux-image-6.12.111-p2p_*.deb (and headers), installed separately.
# usage (on incite): build.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=$HOME/trashcan/kernel
base_config=/boot/config-6.12.111+deb13-amd64
src=$work/linux-source-6.12

mkdir -p "$work"
rm -rf "$src"
tar -C "$work" -xJf /usr/src/linux-source-6.12.tar.xz
cd "$src"
for p in "$here"/patches/*.patch; do
    echo "applying $(basename "$p")"
    patch -p1 --forward < "$p"
done

cp "$base_config" .config
scripts/config \
    --enable PCI_P2PDMA \
    --enable DMABUF_MOVE_NOTIFY \
    --disable DEBUG_INFO_DWARF5 --disable DEBUG_INFO_DWARF4 --disable DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT \
    --enable DEBUG_INFO_NONE --disable DEBUG_INFO --disable DEBUG_INFO_BTF \
    --set-str SYSTEM_TRUSTED_KEYS "" --set-str SYSTEM_REVOCATION_KEYS "" \
    --set-str LOCALVERSION "-p2p" --disable LOCALVERSION_AUTO
make olddefconfig
grep -E "^CONFIG_(PCI_P2PDMA|DMABUF_MOVE_NOTIFY|DEBUG_INFO_NONE|LOCALVERSION)=" .config

time make -j"$(nproc)" bindeb-pkg KDEB_PKGVERSION=6.12.111-1p2p
ls -la "$work"/*.deb
