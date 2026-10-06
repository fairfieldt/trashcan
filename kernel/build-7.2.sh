#!/bin/bash
# Builds incite's kernel, kernel.org Linux 7.2.9 as 7.2.9-p2p, with patches-7.2/ (see docs/kernel.md):
#   0001-0003  the D700 and P2P fixes (VFCT exact match in its upstream form, SI GPIO DC only on
#              mobile parts, Ivy Bridge-EP in the P2PDMA whitelist)
#   0004       thunderbolt_net ring interrupt throttling as a module parameter
#   0005       USB4STREAM without Tx-side E2E flow control (it wedges the Falcon Ridge NHI)
# and configs/6.12.111-p2p.config plus:
#   PCI_P2PDMA=y   peer-to-peer DMA between the D700s (7.2 has no DMABUF_MOVE_NOTIFY option; move
#                  notify is always on)
#   USB4_STREAM=m  USB4STREAM (/dev/tbstreamX)
#   no debug info / BTF
# Output: $WORK/linux-{image,headers}-7.2.9-p2p_7.2.9-${REV}p2p_amd64.deb, installed separately.
# usage: on the Mac, kernel/cross-build-7.2.sh (5 minutes); natively on incite, build-7.2.sh (~40).
# Non-x86_64 hosts cross-compile with ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-. The base config is
# configs/6.12.111-p2p.config (incite's /boot/config-6.12.111-p2p). Bump REV for each rebuild so dpkg
# upgrades in place. JOBS overrides make -j (default nproc).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ver=7.2.9
rev=${REV:-2}
sha=b4c5dfbe51a364a6c7f03869200f88c8e1f77403539005f14b7fc6bc91b8d8ba
work=${WORK:-$HOME/kernel/7.2}
base_config=${BASE_CONFIG:-$here/configs/6.12.111-p2p.config}
kmake=(make)
[ "$(uname -m)" = x86_64 ] || kmake=(make ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-)
src=$work/linux-$ver

mkdir -p "$work"
cd "$work"
[ -f linux-$ver.tar.xz ] || curl -sSLO https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-$ver.tar.xz
echo "$sha  linux-$ver.tar.xz" | sha256sum -c -

rm -rf "$src"
tar -xJf linux-$ver.tar.xz
cd "$src"
for p in "$here"/patches-7.2/*.patch; do
    echo "applying $(basename "$p")"
    patch -p1 --forward < "$p"
done

cp "$base_config" .config
scripts/config --file .config \
    --enable PCI_P2PDMA \
    --module USB4_STREAM \
    --disable DEBUG_INFO_DWARF5 --disable DEBUG_INFO_DWARF4 --disable DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT \
    --enable DEBUG_INFO_NONE --disable DEBUG_INFO --disable DEBUG_INFO_BTF \
    --set-str SYSTEM_TRUSTED_KEYS "" --set-str SYSTEM_REVOCATION_KEYS "" \
    --set-str LOCALVERSION "-p2p" --disable LOCALVERSION_AUTO
"${kmake[@]}" olddefconfig
grep -E "^CONFIG_(PCI_P2PDMA|USB4|USB4_NET|USB4_CONFIGFS|USB4_STREAM|DEBUG_INFO_NONE|LOCALVERSION)=" .config

rm -f "$work"/*.deb "$work"/*.buildinfo "$work"/*.changes
time "${kmake[@]}" -j"${JOBS:-$(nproc)}" bindeb-pkg KDEB_PKGVERSION=$ver-${rev}p2p
ls -la "$work"/*.deb
