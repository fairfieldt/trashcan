#!/bin/bash
# The port lights driver for an installed kernel, without building or booting a new one: applesmc with
# kernel/patches-7.3/0009 and 0010, built as a module and installed as /lib/modules/KREL/updates/applesmc.ko,
# which modprobe prefers to the kernel's own applesmc. A kernel built with 0009 and 0010 (kernel/build-7.3.sh)
# needs none of this.
#
#   driver.sh build [--kernel KREL] [--headers DIR] [--source FILE] [--host-tools TREE] [--out DIR]
#       Builds applesmc.ko for KREL (default: the running kernel) and prints its path. Runs on the Mac or on any
#       Linux host with gcc, make, patch, curl and KREL's headers (DIR, default /lib/modules/KREL/build; the
#       linux-headers-KREL package). The headers must be for exactly KREL. Off x86-64 it cross-compiles with
#       x86_64-linux-gnu-gcc; if the headers' host programs (scripts/, tools/) are x86-64 binaries, --host-tools
#       TREE takes them from a kernel source tree of the same version built on this host (into a copy of the
#       headers). applesmc.c for KREL's version comes from kernel.org, checked against a pinned sha256 (7.3-rc7
#       and 7.3 so far); for other versions pass the kernel's drivers/hwmon/applesmc.c as --source FILE.
#       DIR default lights/build/KREL. Needs no root.
#   driver.sh install [--dry-run] [--no-reload] KO
#       On the Mac, as root: checks that KO is applesmc with the port lights and was built for the running
#       kernel, installs it and reloads applesmc. fancurve and portlights-node are stopped for those seconds (the
#       SMC runs the fan meanwhile) and started again. If the new module does not load, the kernel's own applesmc
#       is put back. --no-reload: takes effect at the next boot.
#   driver.sh uninstall [--dry-run] [--no-reload]
#       Removes a port lights applesmc from /lib/modules/$(uname -r)/updates and reloads the kernel's own.
#   driver.sh status
#       The running kernel, the installed override, and whether the port lights are there.
#
# --dry-run prints what would change and changes nothing. Install and uninstall are idempotent. Nothing here
# reboots. Exit status: 0 done or nothing to do, 1 failed, 2 usage.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
patches=("$here"/../kernel/patches-7.3/0009-applesmc-*.patch "$here"/../kernel/patches-7.3/0010-applesmc-*.patch)
smc=/sys/devices/platform/applesmc.768
PATH=$PATH:/usr/sbin:/sbin

say() { echo "lights driver: $*"; }
die() { echo "lights driver: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^set -e/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0" >&2; exit 2; }

# applesmc.c by kernel version tag: sha256 (the file is the same in v7.3-rc7 and v7.3)
pinned_sha() {
    case $1 in
        v7.3-rc7 | v7.3) echo 3f930296eb6e808a420b8ab72cfa1926919e957ef36ebec60b5400a4c1f751fd ;;
        *) return 1 ;;
    esac
}

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# The vermagic string's kernel release, and whether the module has the port lights.
ko_krel() { tr '\0' '\n' <"$1" | sed -n 's/^vermagic=\([^ ]*\).*/\1/p' | head -1; }
ko_has_lights() { grep -aq port_lights_levels "$1"; }

lights_present() { [ -e $smc/port_lights_levels ]; }

build() {
    local krel="" headers="" source="" host_tools="" out=""
    while [ $# -gt 0 ]; do
        case $1 in
            --kernel) krel=$2; shift 2 ;;
            --headers) headers=$2; shift 2 ;;
            --source) source=$2; shift 2 ;;
            --host-tools) host_tools=$2; shift 2 ;;
            --out) out=$2; shift 2 ;;
            *) usage ;;
        esac
    done
    krel=${krel:-$(uname -r)}
    headers=${headers:-/lib/modules/$krel/build}
    out=${out:-$here/build/$krel}
    [ -f "$headers/include/config/kernel.release" ] ||
        die "no kernel headers for $krel in $headers (Debian: apt install linux-headers-$krel)"
    local rel
    rel=$(cat "$headers/include/config/kernel.release")
    [ "$rel" = "$krel" ] || die "the headers in $headers are for $rel, not $krel"
    for p in "${patches[@]}"; do [ -f "$p" ] || die "missing $p"; done

    rm -rf "$out"
    mkdir -p "$out/src/drivers/hwmon" "$out/mod"
    if [ -n "$source" ]; then
        cp "$source" "$out/src/drivers/hwmon/applesmc.c"
    else
        local mk v p s e tag sha
        mk=$headers/Makefile
        v=$(sed -n 's/^VERSION = //p' "$mk"); p=$(sed -n 's/^PATCHLEVEL = //p' "$mk")
        s=$(sed -n 's/^SUBLEVEL = //p' "$mk"); e=$(sed -n 's/^EXTRAVERSION = //p' "$mk")
        if [ "$s" = 0 ]; then tag=v$v.$p$e; else tag=v$v.$p.$s$e; fi
        sha=$(pinned_sha "$tag") ||
            die "no pinned applesmc.c for kernel $tag: pass the kernel's drivers/hwmon/applesmc.c as --source FILE"
        say "applesmc.c of $tag from kernel.org"
        curl -sSfL -o "$out/src/drivers/hwmon/applesmc.c" \
            "https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/drivers/hwmon/applesmc.c?h=$tag"
        [ "$(sha256 "$out/src/drivers/hwmon/applesmc.c")" = "$sha" ] || die "applesmc.c of $tag: sha256 mismatch"
    fi
    for p in "${patches[@]}"; do
        patch -s -d "$out/src" -p1 --no-backup-if-mismatch <"$p" ||
            die "$(basename "$p") does not apply to this applesmc.c"
    done
    cp "$out/src/drivers/hwmon/applesmc.c" "$out/mod/"
    echo 'obj-m := applesmc.o' >"$out/mod/Makefile"

    if [ -n "$host_tools" ]; then
        # a private copy of the headers with this host's scripts/ and tools/ programs
        mkdir -p "$out/headers"
        cp -a "$headers/." "$out/headers/"
        local f
        while IFS= read -r f; do
            if head -c 4 "$out/headers/$f" | grep -aq ELF && [ -x "$host_tools/$f" ]; then
                cp "$host_tools/$f" "$out/headers/$f"
            fi
        done < <(cd "$out/headers" && find scripts tools -type f -perm -u+x 2>/dev/null)
        headers=$out/headers
    fi
    local cross=()
    [ "$(uname -m)" = x86_64 ] || cross=(ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-)
    say "building for $krel"
    make -s -C "$headers" M="$out/mod" "${cross[@]}" modules
    local ko=$out/mod/applesmc.ko
    [ "$(ko_krel "$ko")" = "$krel" ] || die "built $ko for $(ko_krel "$ko"), not $krel"
    ko_has_lights "$ko" || die "built $ko without the port lights"
    echo "$ko"
}

# Reloads applesmc with fancurve and portlights-node stopped meanwhile. False if applesmc did not load.
reload() {
    local fan node ok=0
    fan=$(systemctl is-active fancurve 2>/dev/null || true)
    node=$(systemctl is-active portlights-node 2>/dev/null || true)
    [ "$node" = active ] && systemctl stop portlights-node
    [ "$fan" = active ] && systemctl stop fancurve
    modprobe -r applesmc 2>/dev/null || rmmod applesmc 2>/dev/null || true
    modprobe applesmc || ok=1
    [ "$fan" = active ] && systemctl start fancurve
    [ "$node" = active ] && systemctl start portlights-node
    return $ok
}

need_root() { [ "$(id -u)" = 0 ] || [ -n "$dry" ] || die "run as root"; }

install_ko() {
    local ko="" krel dest
    while [ $# -gt 0 ]; do
        case $1 in
            --dry-run) dry=1; shift ;;
            --no-reload) noreload=1; shift ;;
            -*) usage ;;
            *) ko=$1; shift ;;
        esac
    done
    [ -n "$ko" ] || usage
    [ -f "$ko" ] || die "no file $ko"
    krel=$(uname -r)
    dest=/lib/modules/$krel/updates/applesmc.ko
    [ -n "$(ko_krel "$ko")" ] || die "$ko is not a kernel module"
    [ "$(ko_krel "$ko")" = "$krel" ] || die "$ko is built for $(ko_krel "$ko"), this kernel is $krel"
    ko_has_lights "$ko" || die "$ko is not applesmc with the port lights"
    grep -q '/applesmc\.ko$' "/lib/modules/$krel/modules.builtin" 2>/dev/null &&
        die "applesmc is built into $krel and cannot be replaced by a module"
    [ "$(cat /sys/module/module/parameters/sig_enforce 2>/dev/null)" = Y ] &&
        die "this kernel only loads signed modules (Secure Boot or module.sig_enforce)"
    need_root
    if [ -f "$dest" ] && cmp -s "$ko" "$dest"; then
        say "$dest: unchanged"
        if lights_present || [ -n "$noreload" ]; then return 0; fi
        [ -n "$dry" ] && { say "would reload applesmc"; return 0; }
    else
        if [ -n "$dry" ]; then
            say "would install $ko as $dest, run depmod${noreload:+ (no reload)}${noreload:- and reload applesmc}"
            return 0
        fi
        install -D -m 0644 "$ko" "$dest"
        depmod "$krel"
        say "installed $dest"
    fi
    if [ -n "$noreload" ]; then
        say "applesmc not reloaded: the port lights come at the next boot"
        return 0
    fi
    if ! reload || ! lights_present; then
        echo "lights driver: the new applesmc did not load; putting the kernel's own back" >&2
        rm -f "$dest"
        depmod "$krel"
        reload || true
        exit 1
    fi
    say "applesmc reloaded: $(ls /sys/class/leds | grep -c '^smc::') port lights"
}

uninstall_ko() {
    local krel dest
    while [ $# -gt 0 ]; do
        case $1 in
            --dry-run) dry=1; shift ;;
            --no-reload) noreload=1; shift ;;
            *) usage ;;
        esac
    done
    krel=$(uname -r)
    dest=/lib/modules/$krel/updates/applesmc.ko
    if [ ! -f "$dest" ]; then
        say "no applesmc override for $krel"
        return 0
    fi
    ko_has_lights "$dest" || die "$dest is not a port lights applesmc; leaving it"
    need_root
    if [ -n "$dry" ]; then
        say "would remove $dest, run depmod${noreload:+ (no reload)}${noreload:- and reload applesmc}"
        return 0
    fi
    rm -f "$dest"
    depmod "$krel"
    say "removed $dest"
    if [ -n "$noreload" ]; then
        say "applesmc not reloaded: the override is gone at the next boot"
        return 0
    fi
    reload || die "applesmc did not load"
    if lights_present; then say "the port lights are still there (built into the kernel's applesmc)"
    else say "applesmc reloaded without the port lights"; fi
}

status() {
    local krel dest
    krel=$(uname -r)
    dest=/lib/modules/$krel/updates/applesmc.ko
    echo "kernel: $krel"
    if [ -f "$dest" ]; then echo "override: $dest (built for $(ko_krel "$dest"))"; else echo "override: none"; fi
    if lights_present; then echo "port lights: yes ($(ls /sys/class/leds | grep -c '^smc::') LEDs)"
    else echo "port lights: no"; fi
}

dry="" noreload=""
[ $# -gt 0 ] || usage
cmd=$1; shift
case $cmd in
    build) build "$@" ;;
    install) install_ko "$@" ;;
    uninstall) uninstall_ko "$@" ;;
    status) status ;;
    *) usage ;;
esac
