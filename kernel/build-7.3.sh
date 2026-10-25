#!/bin/bash
# Builds the Mac Pro 6,1 kernel, Linux 7.3 (VER, default 7.3-rc7) as e.g. 7.3.0-rc7-p2p, with patches-7.3/
# (see docs/kernel.md):
#   0001  amdgpu: use the VFCT VBIOS image whose PCI bus matches exactly (each D700 gets its own VBIOS)
#   0003  P2PDMA allowlist: Ivy Bridge-EP host bridge (8086:0e00)
#   0004  thunderbolt_net ring interrupt throttling as a module parameter
#   0005  USB4STREAM without Tx-side E2E flow control (it wedges the Falcon Ridge NHI)
#   0006  USB4STREAM: a CLOSE that has been read no longer fails every later read and write (7.2's behaviour)
#   0007  USB4STREAM: a close no longer waits 500 ms for the RX ring, which never empties
#   0009  applesmc: the Mac Pro 6,1's port lights as LED class devices (optional)
#   0010  applesmc: port_lights_levels, all port lights in one write (optional)
# The engine needs 0001-0007. 0009 and 0010 have had little testing; remove them from patches-7.3/ to build without.
# The numbering skips 0002 (SI GPIO DC only on mobile parts, upstream in 7.3 as cd883d0a3937) and 0008.
# Config: configs/7.3.0-rc7-p2p.config (or BASE_CONFIG) through olddefconfig, with the settings below reasserted;
# the build stops if an option we rely on is missing.
# Output: $WORK/linux-{image,headers}-$KREL_$PKGVER_amd64.deb (and linux-libc-dev, which is never installed).
# WORK defaults to ~/kernel/7.3.
# usage: kernel/build-7.3.sh, natively on Debian 13 x86_64 (about 40 minutes on a Mac Pro 6,1). Other hosts
# cross-compile with ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu- (set automatically when the host is not x86_64;
# Debian's gcc-x86-64-linux-gnu).
# VER: 7.3-rcN comes from kernel.org's cgit snapshot (.tar.gz), 7.3 and 7.3.N from the CDN (.tar.xz). Each needs its
# sha256 in the table below (or SHA256=...). Bump REV for each rebuild of the same VER so dpkg upgrades in place.
# JOBS overrides make -j (default nproc). KLOCAL overrides the release suffix (default -p2p): a test build with its
# own suffix, e.g. -p2p-s (7.3.0-rc7-p2p-s), installs beside the running kernel and can be booted once.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ver=${VER:-7.3-rc7}
rev=${REV:-1}
case $ver in
    # snapshot of v7.3-rc7, commit 24a19b953e2a ("Linux 7.3-rc7"), as its pax header records
    7.3-rc7) sha=f789fc0d7b3713b6be23284f2a9c35bf41fb86785b24cfcc506c20d8ff712523 ;;
    *) sha=${SHA256:?no sha256 for $ver: add it to the table in build-7.3.sh or pass SHA256} ;;
esac
case $ver in
    *-rc*) tarball=linux-$ver.tar.gz; url=https://git.kernel.org/torvalds/t/$tarball ;;
    *) tarball=linux-$ver.tar.xz; url=https://cdn.kernel.org/pub/linux/kernel/v7.x/$tarball ;;
esac
work=${WORK:-$HOME/kernel/7.3}
base_config=${BASE_CONFIG:-$here/configs/7.3.0-rc7-p2p.config}
kmake=(make)
[ "$(uname -m)" = x86_64 ] || kmake=(make ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-)
src=$work/linux-$ver

mkdir -p "$work"
cd "$work"
[ -f "$tarball" ] || curl -sSfL -o "$tarball" "$url"
echo "$sha  $tarball" | sha256sum -c -

rm -rf "$src"
tar -xf "$tarball"
cd "$src"
for p in "$here"/patches-7.3/*.patch; do
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
    --set-str LOCALVERSION "${KLOCAL:--p2p}" --disable LOCALVERSION_AUTO
"${kmake[@]}" olddefconfig
missing=0
for o in PCI_P2PDMA=y DRM_AMDGPU=m DRM_AMDGPU_SI=y DRM_AMD_DC_SI=y DRM_VGEM=m NETCONSOLE=m NETCONSOLE_DYNAMIC=y \
         USB4=m USB4_NET=m USB4_CONFIGFS=m USB4_STREAM=m DEBUG_INFO_NONE=y; do
    grep -qx "CONFIG_$o" .config || { echo "config: CONFIG_$o missing" >&2; missing=1; }
done
[ $missing = 0 ]
krel=$("${kmake[@]}" -s kernelrelease)
kver=$("${kmake[@]}" -s kernelversion)
pkgver=$(echo "$kver" | sed 's/-rc/~rc/')-${rev}p2p    # 7.3.0-rc7 -> 7.3.0~rc7-1p2p, sorting before 7.3.0-1p2p
echo "kernel release $krel, package version $pkgver"
cp .config "$work/config-$krel"

rm -f "$work"/*.deb "$work"/*.buildinfo "$work"/*.changes
time "${kmake[@]}" -j"${JOBS:-$(nproc)}" bindeb-pkg KDEB_PKGVERSION="$pkgver"
ls -la "$work"/*.deb
