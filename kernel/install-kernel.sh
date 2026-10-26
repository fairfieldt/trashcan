#!/bin/bash
# Install, boot-once, pin and verify a kernel built by kernel/build-7.3.sh, on the node (docs/kernel.md). It never
# reboots: it says when a reboot is needed, and the operator reboots.
#
#   sudo kernel/install-kernel.sh install DIR [--apply]   check DIR's SHA256SUMS and kernel-build-*.txt, install the
#                                                         image and headers, and boot that kernel once at the next
#                                                         reboot (GRUB's default stays the fallback)
#   sudo kernel/install-kernel.sh once RELEASE [--apply]  boot RELEASE once at the next reboot (e.g. back to Debian's)
#   sudo kernel/install-kernel.sh pin RELEASE [--apply]   make RELEASE GRUB's default, by entry id (after `verify`)
#   kernel/install-kernel.sh verify [DIR]                 read-only checks of the running kernel (against DIR's
#                                                         kernel-build-*.txt when given)
#   kernel/install-kernel.sh list                         installed kernels, GRUB's default and next boot
#
# Without --apply, install/once/pin only print what they would do.
set -euo pipefail
cmd=${1:-}; shift || true
apply=0 arg=""
for x in "$@"; do
    case $x in --apply) apply=1 ;; *) arg=$x ;; esac
done
run() { echo "  $*"; [ "$apply" = 0 ] || "$@"; }
need_root() { [ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }; }

# GRUB's "submenu>entry" id for a kernel release.
entry_id() {
    local rel=$1 sub ent
    sub=$(grep -o "submenu 'Advanced options[^{]*\$menuentry_id_option '[^']*'" /boot/grub/grub.cfg | sed "s/.*option '//; s/'\$//")
    ent=$(grep -o "menuentry '[^']*, with Linux $rel'[^{]*\$menuentry_id_option '[^']*'" /boot/grub/grub.cfg | sed "s/.*option '//; s/'\$//")
    [ -n "$sub" ] && [ -n "$ent" ] || { echo "no GRUB entry for $rel in /boot/grub/grub.cfg" >&2; return 1; }
    echo "$sub>$ent"
}

boot_once() {
    local rel=$1 id
    [ -e "/boot/vmlinuz-$rel" ] || { echo "/boot/vmlinuz-$rel is not installed" >&2; exit 1; }
    id=$(entry_id "$rel")
    echo "boot $rel once at the next reboot (GRUB entry $id); GRUB's default stays $(grep -h '^GRUB_DEFAULT=' /etc/default/grub /etc/default/grub.d/*.cfg 2>/dev/null | tail -1)"
    run grub-reboot "$id"
    if [ "$apply" = 1 ]; then
        echo "reboot when nothing runs on this node: sudo systemctl reboot"
        echo "Unplug Thunderbolt cables that loop this Mac's own controllers first, or the firmware hangs before the chime."
        echo "If $rel does not come up, power-cycle: the next boot takes GRUB's default again."
    fi
}

case $cmd in
install)
    need_root
    dir=${arg:?usage: install-kernel.sh install DIR [--apply]}
    cd "$dir"
    [ -f SHA256SUMS ] || { echo "$dir/SHA256SUMS missing (kernel/build-7.3.sh writes it)" >&2; exit 1; }
    sha256sum --quiet -c SHA256SUMS || { echo "checksum mismatch: do not install these packages" >&2; exit 1; }
    build=$(ls kernel-build-*.txt | head -1)
    rel=$(awk '$1 == "release" {print $2}' "$build")
    pkgver=$(awk '$1 == "package_version" {print $2}' "$build")
    read -r _ kset psid < <(grep '^patchset ' "$build")
    img=$(ls linux-image-"${rel}"_*.deb) hdr=$(ls linux-headers-"${rel}"_*.deb)
    [ "$(dpkg-deb -f "$img" Version)" = "$pkgver" ] || { echo "$img is not version $pkgver" >&2; exit 1; }
    echo "kernel $rel, package $pkgver, patch set $kset ($psid); packages match SHA256SUMS"
    grep -qs 'amdgpu.si_support=1' /etc/default/grub /etc/default/grub.d/*.cfg ||
        echo "warning: GRUB has no amdgpu.si_support=1: run nodes/macpro/install.sh --apply first"
    have=$(dpkg-query -W -f '${Version} ${Status},' "linux-image-$rel" "linux-headers-$rel" 2>/dev/null || true)
    if [ "$have" = "$pkgver install ok installed,$pkgver install ok installed," ]; then
        echo "already installed"
    else
        # --force-unsafe-io: no per-file fsync; an original Apple SSD takes up to seconds per cache flush.
        run dpkg --force-unsafe-io -i "$img" "$hdr"
    fi
    [ "$apply" = 0 ] || update-grub 2> /dev/null
    if [ "$(uname -r)" = "$rel" ]; then
        echo "$rel is running already"
    elif [ "$apply" = 1 ]; then
        boot_once "$rel"
    else
        echo "  grub-reboot <entry for $rel>   (boot it once at the next reboot)"
    fi
    [ "$apply" = 1 ] || echo "preview only: rerun with --apply"
    ;;
once)
    need_root
    boot_once "${arg:?usage: install-kernel.sh once RELEASE [--apply]}"
    [ "$apply" = 1 ] || echo "preview only: rerun with --apply"
    ;;
pin)
    need_root
    rel=${arg:?usage: install-kernel.sh pin RELEASE [--apply]}
    [ "$(uname -r)" = "$rel" ] || { echo "boot $rel and run verify before pinning it (running $(uname -r))" >&2; exit 1; }
    id=$(entry_id "$rel")
    echo "make $rel GRUB's default (entry $id) in /etc/default/grub.d/91-trashcan-default.cfg"
    if [ "$apply" = 1 ]; then
        echo "GRUB_DEFAULT=\"$id\"" > /etc/default/grub.d/91-trashcan-default.cfg
        update-grub 2> /dev/null
        echo "pinned; other kernels stay in GRUB's Advanced options as fallbacks"
    else
        echo "preview only: rerun with --apply"
    fi
    ;;
verify)
    rel=$(uname -r) fail=0
    ok() { echo "ok    $*"; }
    bad() { echo "FAIL  $*"; fail=1; }
    ver=$(dpkg-query -W -f '${Version}' "linux-image-$rel" 2>/dev/null || echo none)
    echo "kernel $rel, package $ver"
    if [ -n "$arg" ]; then
        build=$(ls "$arg"/kernel-build-*.txt | head -1)
        want=$(awk '$1 == "release" {print $2}' "$build")
        [ "$rel" = "$want" ] && ok "running the built release $want" || bad "running $rel, built $want"
        [ "$ver" = "$(awk '$1 == "package_version" {print $2}' "$build")" ] && ok "package version matches the build" || bad "package version $ver differs from the build"
        cfgsum=$(awk '$1 == "config" {print $3}' "$build")
        [ "$(sha256sum < "/boot/config-$rel" | cut -d' ' -f1)" = "$cfgsum" ] && ok "/boot/config-$rel is the built config" || bad "/boot/config-$rel differs from the build's"
    fi
    case $ver in *+ps*) ok "patch set id ${ver##*+ps} (deploy/manifest.toml)" ;; *) echo "note  no patch set id in the package version (built before +ps ids)" ;; esac
    for p in radeon.si_support=0 amdgpu.si_support=1; do
        grep -qw "$p" /proc/cmdline && ok "command line has $p" || bad "command line lacks $p"
    done
    grep -qx CONFIG_PCI_P2PDMA=y "/boot/config-$rel" && ok "CONFIG_PCI_P2PDMA=y" || bad "no CONFIG_PCI_P2PDMA"
    vb=() n=0
    for d in /sys/bus/pci/drivers/amdgpu/0000:*; do
        [ -e "$d/vbios_version" ] || continue
        n=$((n + 1)); vb+=("$(cat "$d/vbios_version")")
        echo "      $(basename "$d"): VBIOS $(cat "$d/vbios_version"), VRAM $(($(cat "$d/mem_info_vram_total") >> 20)) MiB, DPM $(cat "$d/power_dpm_force_performance_level")"
    done
    [ "$n" = 2 ] && ok "two GPUs on amdgpu" || bad "$n GPUs on amdgpu (want 2)"
    [ "$n" = 2 ] && [ "${vb[0]}" != "${vb[1]}" ] && ok "each GPU has its own VBIOS (patch 0001)" || bad "the GPUs report the same VBIOS: 0001 missing?"
    [ -e /sys/module/thunderbolt_net/parameters/throttling ] && ok "thunderbolt_net throttling=$(cat /sys/module/thunderbolt_net/parameters/throttling) (patch 0004)" ||
        echo "note  thunderbolt_net not loaded or without the throttling parameter"
    { [ -d /sys/module/thunderbolt_stream ] || find "/lib/modules/$rel" -name 'thunderbolt_stream.ko*' | grep -q .; } &&
        ok "thunderbolt_stream module available" || bad "no thunderbolt_stream module (CONFIG_USB4_STREAM)"
    if [ -e /sys/devices/platform/applesmc.768/port_lights_levels ]; then
        ok "applesmc port lights present (0009/0010, in the kernel or a module override)"
    fi
    echo "      GRUB default: $(grep -h '^GRUB_DEFAULT=' /etc/default/grub /etc/default/grub.d/*.cfg 2>/dev/null | tail -1); next boot: $(grub-editenv list 2>/dev/null | sed -n 's/^next_entry=//p')"
    echo "      fallback kernels: $(ls /boot/vmlinuz-* | sed 's#/boot/vmlinuz-##' | grep -vx "$rel" | xargs)"
    [ "$fail" = 0 ] && echo "verify: PASS" || echo "verify: FAIL"
    exit $fail
    ;;
list)
    dpkg-query -W -f '${Package} ${Version} ${Status}\n' 'linux-image-*' | grep ' installed$' | cut -d' ' -f1-2
    echo "running: $(uname -r)"
    echo "GRUB default: $(grep -h '^GRUB_DEFAULT=' /etc/default/grub /etc/default/grub.d/*.cfg 2>/dev/null | tail -1)"
    echo "next boot: $(grub-editenv list 2>/dev/null | sed -n 's/^next_entry=//p')"
    ;;
*)
    sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//; $d'
    exit 2
    ;;
esac
