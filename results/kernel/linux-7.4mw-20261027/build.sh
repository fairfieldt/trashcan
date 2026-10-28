#!/bin/bash
# Test kernel for the 7.4 USB4STREAM CLOSE question (results/kernel/linux-7.4mw-20261027): mainline in the 7.4 merge
# window plus the upstream packages in kernel/upstream/ (applesmc port lights, P2PDMA Ivy Bridge-EP, VFCT exact match,
# USB4STREAM no Tx E2E and HopID release, thunderbolt_net ethtool coalescing); no 0006 (conflicts with mainline's
# CLOSE handling) and no 0007 (superseded). Release 7.3.0-mw8a36f53: installed beside the deployment kernel and booted
# once with grub-reboot. Run in the kbuild VM; config as kernel/build-7.3.sh.
# usage (VM): build.sh UPSTREAM_DIR CONFIG [EXTRA_PATCH...]   e.g. build.sh ~/a18/up27 ~/a18/7.2.9-p2p.config
set -euo pipefail
up=$1 base_config=$2; shift 2
base=8a36f53fd5ec
klocal=${KLOCAL:--mw8a36f53}
work=$HOME/a18/k74; src=$work/src
kmake=(make ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-)
linux=$HOME/a18/linux
mkdir -p "$work"; rm -rf "$src"; git -C "$linux" worktree prune
git -C "$linux" worktree add -q --detach "$src" "$base"
cd "$src"
for s in applesmc-port-lights p2pdma-ivy-bridge-ep vfct-exact-match thunderbolt-stream-tx-e2e thunderbolt-stream-hopid thunderbolt-net-coalesce; do
    for p in "$up/$s"/series/0*.patch; do
        case $p in *0000-cover-letter.patch) continue ;; esac
        git -c user.name=t -c user.email=t@t am -q "$p"; echo "applied $s/$(basename "$p")"
    done
done
for p in "$@"; do git -c user.name=t -c user.email=t@t am -q "$p"; echo "applied extra $(basename "$p")"; done
cp "$base_config" .config
scripts/config --file .config --enable PCI_P2PDMA --module USB4_STREAM \
    --disable DEBUG_INFO_DWARF5 --disable DEBUG_INFO_DWARF4 --disable DEBUG_INFO_DWARF_TOOLCHAIN_DEFAULT \
    --enable DEBUG_INFO_NONE --disable DEBUG_INFO --disable DEBUG_INFO_BTF \
    --set-str SYSTEM_TRUSTED_KEYS "" --set-str SYSTEM_REVOCATION_KEYS "" \
    --set-str LOCALVERSION "$klocal" --disable LOCALVERSION_AUTO
"${kmake[@]}" olddefconfig
for o in PCI_P2PDMA=y DRM_AMDGPU=m DRM_AMDGPU_SI=y DRM_AMD_DC_SI=y DRM_VGEM=m NETCONSOLE=m NETCONSOLE_DYNAMIC=y \
         USB4=m USB4_NET=m USB4_CONFIGFS=m USB4_STREAM=m DEBUG_INFO_NONE=y SENSORS_APPLESMC=m; do
    grep -qx "CONFIG_$o" .config || { echo "config: CONFIG_$o missing" >&2; exit 1; }
done
krel=$("${kmake[@]}" -s kernelrelease); echo "kernel release $krel"
cp .config "$work/config-$krel"
rm -f "$work"/*.deb "$work"/*.buildinfo "$work"/*.changes
"${kmake[@]}" -j"$(nproc)" bindeb-pkg KDEB_PKGVERSION="7.3.0+mw${base}-${REV:-1}" > "$work/build.log" 2>&1 || { tail -40 "$work/build.log"; exit 1; }
ls -la "$work"/*.deb
