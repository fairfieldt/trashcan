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
# The engine needs 0001-0007. 0009 and 0010 have had little testing. KSET picks the patch set:
#   KSET=engine (default)  0001, 0003-0007          kernel release 7.3.0-rc7-p2p
#   KSET=lights            0001, 0003-0007, 0009-10 kernel release 7.3.0-rc7-p2p-lights
# The package version ends in +ps<ID>, ID the first 8 hex digits of the sha256 of `sha256sum` over the set's patch
# files in order, so `dpkg -l` and deploy/trashcan-cluster preflight tell exactly which patches a kernel has
# (deploy/manifest.toml lists the ids).
# The numbering skips 0002 (SI GPIO DC only on mobile parts, upstream in 7.3 as cd883d0a3937) and 0008.
# Config: configs/7.3.0-rc7-p2p.config (or BASE_CONFIG) through olddefconfig, with the settings below reasserted;
# the build stops if an option we rely on is missing.
# Output: $WORK/linux-{image,headers}-$KREL_$PKGVER_amd64.deb (and linux-libc-dev, which is never installed),
# $WORK/SHA256SUMS for the packages and $WORK/kernel-build-$KREL.txt recording the source, config, patches and
# packages (kernel/install-kernel.sh checks both).
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
kset=${KSET:-engine}
case $kset in
    engine) patches="0001 0003 0004 0005 0006 0007" klocal_default=-p2p ;;
    lights) patches="0001 0003 0004 0005 0006 0007 0009 0010" klocal_default=-p2p-lights ;;
    *) echo "KSET must be engine or lights" >&2; exit 1 ;;
esac
patch_files=()
for n in $patches; do
    f=$(ls "$here"/patches-7.3/$n-*.patch 2>/dev/null)
    [ -n "$f" ] || { echo "no patches-7.3/$n-*.patch" >&2; exit 1; }
    patch_files+=("$f")
done
psid=$( (cd "$here/patches-7.3" && sha256sum "${patch_files[@]##*/}") | sha256sum | cut -c1-8)
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
for p in "${patch_files[@]}"; do
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
    --set-str LOCALVERSION "${KLOCAL:-$klocal_default}" --disable LOCALVERSION_AUTO
"${kmake[@]}" olddefconfig
missing=0
for o in PCI_P2PDMA=y DRM_AMDGPU=m DRM_AMDGPU_SI=y DRM_AMD_DC_SI=y DRM_VGEM=m NETCONSOLE=m NETCONSOLE_DYNAMIC=y \
         USB4=m USB4_NET=m USB4_CONFIGFS=m USB4_STREAM=m DEBUG_INFO_NONE=y; do
    grep -qx "CONFIG_$o" .config || { echo "config: CONFIG_$o missing" >&2; missing=1; }
done
[ $missing = 0 ]
krel=$("${kmake[@]}" -s kernelrelease)
kver=$("${kmake[@]}" -s kernelversion)
# 7.3.0-rc7 -> 7.3.0~rc7-1p2p+ps<id>, sorting before 7.3.0-1p2p
pkgver=$(echo "$kver" | sed 's/-rc/~rc/')-${rev}p2p+ps$psid
echo "kernel release $krel, package version $pkgver, patch set $kset ($psid)"
cp .config "$work/config-$krel"

rm -f "$work"/*.deb "$work"/*.buildinfo "$work"/*.changes "$work"/SHA256SUMS
time "${kmake[@]}" -j"${JOBS:-$(nproc)}" bindeb-pkg KDEB_PKGVERSION="$pkgver"
cd "$work"
sha256sum linux-image-"${krel}"_*.deb linux-headers-"${krel}"_*.deb > SHA256SUMS
{
    echo "release $krel"
    echo "package_version $pkgver"
    echo "patchset $kset $psid"
    echo "source $tarball $sha"
    echo "base_config $(basename "$base_config") $(sha256sum < "$base_config" | cut -d' ' -f1)"
    echo "config config-$krel $(sha256sum < "config-$krel" | cut -d' ' -f1)"
    (cd "$here/patches-7.3" && sha256sum "${patch_files[@]##*/}") | sed 's/^/patch /'
    sed 's/^/package /' SHA256SUMS
    cc=gcc; [ "$(uname -m)" = x86_64 ] || cc=x86_64-linux-gnu-gcc
    echo "built $(date -u +%FT%TZ) on $(uname -m) with $($cc --version | head -1)"
} > "kernel-build-$krel.txt"
ls -la "$work"/*.deb
cat "kernel-build-$krel.txt"
