#!/bin/bash
# Install, check or remove the system configuration a trashcan node needs, on the node itself (as root). Without
# --apply it only shows what it would change. Repeatable: a second --apply changes nothing. It never reboots and
# never resets a Thunderbolt link; changes that need a reboot (kernel command line) or a module reload (the
# thunderbolt_net throttle) say so. See nodes/macpro/README.md.
#
# usage: sudo nodes/macpro/install.sh [--apply] [--user USER] [--skip C1,C2] [--registry NODES.TSV]
#                                     [--lights [--lights-args "ARGS"]]
#        sudo nodes/macpro/install.sh --uninstall [--apply]
#
# Components (all by default; --skip leaves one out):
#   tblink   Thunderbolt link addressing and USB4STREAM creation (tblink.service), stream device access (udev),
#            thunderbolt_net interrupt throttle; --registry installs the node table (deploy/trashcan-cluster registry)
#   fan      fan control (fancurve.service): the main fan at full speed while the GPUs work, slower when idle and cool
#   network  NetworkManager leaves thunderbolt* alone; with ifupdown, DHCP on every Ethernet port, so a cable in
#            either port works (the Debian installer configures only the port cabled at install time)
#   power    suspend masked (the Mac Pro does not resume)
#   gpu      amdgpu for the D700s on the kernel command line (radeon.si_support=0 amdgpu.si_support=1), the user in
#            render and video, and sudo for the engine's DPM level writes (render group, those files only)
#   logs     persistent journal
#   panic    reboot 10 s after a panic, oops or lockup; /usr/local/sbin/emergency-reboot
#   ssh      sshd drops sessions whose client went away within about a minute
#   disk     the write cache off on an original Apple SSD (APPLE SSD SM...), whose cache flushes stall all its I/O
#            for up to seconds under load (udev rule, applied at once)
#   lights   only with --lights: the port lights, by lights/install.sh (docs/lights.md) with --lights-args (e.g.
#            "--bin FILE --build-module"); needs a portlights build and a kernel with 0009/0010 or the module.
#            --uninstall removes them too (lights/uninstall.sh) when this installer installed them.
#
# State: /var/lib/trashcan-node (installed files, the originals they replaced, masks, groups), used by --uninstall.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
mode=preview uninstall=0 skip="" registry="" user=${SUDO_USER:-} lights=0 lights_args=""
while [ $# -gt 0 ]; do
    case $1 in
        --apply) mode=apply ;;
        --uninstall) uninstall=1 ;;
        --user) user=$2; shift ;;
        --skip) skip=,$2, ; shift ;;
        --registry) registry=$(realpath "$2"); shift ;;
        --lights) lights=1 ;;
        --lights-args) lights_args=$2; shift ;;
        -h|--help) sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//; $d'; exit 0 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)" >&2; exit 1; }
STATE=/var/lib/trashcan-node
if [ "$mode" = apply ]; then
    mkdir -p "$STATE/backup"
    touch "$STATE/files" "$STATE/masked" "$STATE/groups" "$STATE/enabled"
elif [ ! -d "$STATE" ]; then
    STATE=$(mktemp -d /run/trashcan-node-preview.XXXXXX)   # a preview writes nothing outside /run
    trap 'rm -rf "$STATE"' EXIT
    touch "$STATE/files" "$STATE/masked" "$STATE/groups" "$STATE/enabled"
fi
want() { [[ $skip != *,$1,* ]]; }
applying() { [ "$mode" = apply ]; }
changes=0 notes=()
stage=$(mktemp -d /run/trashcan-node.XXXXXX)
trap 'rm -rf "$stage"; [[ $STATE == /run/* ]] && rm -rf "$STATE"' EXIT
changed=()   # destination paths that differ from what is installed

note() { notes+=("$*"); }
act() {   # act DESCRIPTION CMD...: count a change; run it with --apply
    changes=$((changes + 1))
    echo "  $1"
    if applying; then shift; "$@"; fi
}

# replace SRC DEST MODE: write DEST as a new file and rename it into place, so a running script (tblink, fancurve)
# keeps reading the old one until its service restarts.
replace() {
    mkdir -p "$(dirname "$2")"
    cp "$1" "$2.trashcan-new" && chmod "$3" "$2.trashcan-new" && chown root:root "$2.trashcan-new" && mv -f "$2.trashcan-new" "$2"
}

# put SRC DEST MODE: install a file if its content or mode differs. The first time a file not installed by us is
# replaced, the original is kept for --uninstall.
put() {
    local src=$1 dest=$2 m=$3
    if [ -f "$dest" ] && cmp -s "$src" "$dest" && [ "$(stat -c %a "$dest")" = "$m" ]; then
        return 0   # already as wanted; recorded only if we wrote it, so --uninstall leaves files it found in place
    fi
    changed+=("$dest")
    changes=$((changes + 1))
    if [ -e "$dest" ]; then
        echo "  update $dest"
        diff -u "$dest" "$src" | sed -n '3,40p' | sed 's/^/      /' || true
    else
        echo "  create $dest"
    fi
    if applying; then
        if [ -e "$dest" ] && ! grep -qxF "$dest" "$STATE/files" && [ ! -e "$STATE/backup$dest" ]; then
            install -D -m 0644 /dev/null "$STATE/backup$dest" && cp -a "$dest" "$STATE/backup$dest"
        fi
        replace "$src" "$dest" "$m"
        grep -qxF "$dest" "$STATE/files" || echo "$dest" >> "$STATE/files"
    fi
}
is_changed() { local d; for d in "${changed[@]}"; do [[ $d == $1 ]] && return 0; done; return 1; }

enable_service() {   # enable_service NAME RESTART(0|1)
    local s=$1
    if ! systemctl -q is-enabled "$s" 2>/dev/null; then
        act "enable $s" systemctl enable -q "$s"
        applying && { grep -qxF "$s" "$STATE/enabled" || echo "$s" >> "$STATE/enabled"; }
    fi
    if ! systemctl -q is-active "$s"; then
        act "start $s" systemctl start "$s"
    elif [ "$2" = 1 ]; then
        act "restart $s (its files changed)" systemctl restart "$s"
    fi
}

R=$here/root
# apple_ssds: the scsi_disk directories of original Apple SSDs (vendor ATA, model APPLE SSD SM...).
apple_ssds() {
    local d
    for d in /sys/class/scsi_disk/*; do
        [ -e "$d/cache_type" ] || continue
        [[ $(cat "$d/device/vendor" 2>/dev/null) == ATA* && $(cat "$d/device/model" 2>/dev/null) == "APPLE SSD SM"* ]] && echo "$d"
    done
}
user_ok() { [ -n "$user" ] && id "$user" > /dev/null 2>&1; }

# ------------------------------------------------------------------------------------------------------- uninstall
if [ "$uninstall" = 1 ]; then
    echo "== uninstall ($mode)"
    touched=""
    while read -r f; do
        [ -n "$f" ] || continue
        if [ -e "$STATE/backup$f" ]; then
            act "restore $f (the original it replaced)" replace "$STATE/backup$f" "$f" "$(stat -c %a "$STATE/backup$f")"
            touched+=" $f"
        elif [ -e "$f" ]; then
            act "remove $f" rm -f "$f"
            touched+=" $f"
        fi
    done < "$STATE/files"
    while read -r s; do [ -n "$s" ] && act "disable and stop $s" systemctl disable -q --now "$s"; done < "$STATE/enabled"
    while read -r t; do [ -n "$t" ] && act "unmask $t" systemctl unmask -q "$t"; done < "$STATE/masked"
    while read -r ug; do [ -n "$ug" ] && act "remove ${ug% *} from group ${ug#* }" gpasswd -d ${ug% *} ${ug#* }; done < "$STATE/groups"
    has() { [[ " $touched " == *" $1"* ]]; }
    has /etc/systemd/system/ && act "systemctl daemon-reload" systemctl daemon-reload
    for s in tblink fancurve; do
        if { has /usr/local/sbin/$s || has /etc/systemd/system/$s.service; } && systemctl -q is-active $s.service; then
            act "restart $s.service (its files changed; links are kept)" systemctl restart $s.service
        fi
    done
    has /etc/udev/rules.d/70-tbstream.rules && act "reload udev rules" bash -c "udevadm control --reload && udevadm trigger --action=change --sysname-match='tbstream*'"
    if has /etc/udev/rules.d/60-trashcan-apple-ssd.rules; then
        act "reload udev rules" udevadm control --reload
        for d in $(apple_ssds); do
            act "turn the write cache of $(ls "$d/device/block") back on" sh -c "echo 'write back' > $d/cache_type"
        done
    fi
    has /etc/sysctl.d/ && act "reload sysctls" sysctl -q --system
    has /etc/systemd/journald.conf.d/ && act "restart journald" systemctl restart systemd-journald.service
    has /etc/ssh/sshd_config.d/ && act "reload sshd" bash -c 'sshd -t && systemctl reload ssh'
    has /etc/default/grub.d/ && act "update-grub (the original command line from the next boot)" bash -c 'update-grub 2> /dev/null'
    has /etc/NetworkManager/ && systemctl -q is-active NetworkManager && act "reload NetworkManager" systemctl reload NetworkManager
    if [ -e "$STATE/lights" ] && want lights; then
        echo "-- lights (lights/uninstall.sh)"
        if [ -f "$repo/lights/uninstall.sh" ]; then
            changes=$((changes + 1))
            bash "$repo/lights/uninstall.sh" $(applying || echo --dry-run) | sed 's/^/  /'
        else
            note "lights: portlights-node is installed but this checkout has no lights/uninstall.sh"
        fi
    fi
    if [ "$changes" = 0 ]; then
        echo "$(hostname): nothing installed"
    elif applying; then
        rm -rf "$STATE"
        echo "$(hostname): removed"
    else
        echo "$(hostname): $changes changes; rerun with --apply"
    fi
    exit 0
fi

echo "== $(hostname): $mode"

# ---------------------------------------------------------------------------------------------------------- tblink
if want tblink; then
    echo "-- tblink"
    put "$R/usr/local/sbin/tblink" /usr/local/sbin/tblink 755
    put "$R/etc/systemd/system/tblink.service" /etc/systemd/system/tblink.service 644
    put "$R/etc/udev/rules.d/70-tbstream.rules" /etc/udev/rules.d/70-tbstream.rules 644
    # Only a changed option (not a comment) needs the module reloaded.
    [ "$(grep -v '^#' "$R/etc/modprobe.d/thunderbolt-net.conf")" = "$(grep -sv '^#' /etc/modprobe.d/thunderbolt-net.conf)" ] && tbopt=same || tbopt=new
    put "$R/etc/modprobe.d/thunderbolt-net.conf" /etc/modprobe.d/thunderbolt-net.conf 644
    [ -z "$registry" ] || put "$registry" /etc/tblink/nodes.tsv 644
    [ -s /etc/tblink/nodes.tsv ] || [ -n "$registry" ] || note "tblink: no /etc/tblink/nodes.tsv yet, so no links are addressed (deploy/trashcan-cluster registry --push --apply)"
fi
# ------------------------------------------------------------------------------------------------------------- fan
if want fan; then
    echo "-- fan"
    put "$repo/scripts/fancurve.sh" /usr/local/sbin/fancurve.sh 755
    put "$repo/scripts/fancurve.service" /etc/systemd/system/fancurve.service 644
fi
# --------------------------------------------------------------------------------------------------------- network
if want network; then
    echo "-- network"
    if [ -d /etc/NetworkManager ]; then
        put "$R/etc/NetworkManager/conf.d/90-thunderbolt-unmanaged.conf" /etc/NetworkManager/conf.d/90-thunderbolt-unmanaged.conf 644
    fi
    if [ -f /etc/network/interfaces ] && ! systemctl -q is-active NetworkManager; then
        # Every wired port without a stanza in the other ifupdown files gets DHCP from ours.
        own=/etc/network/interfaces.d/trashcan-ethernet
        : > "$stage/ethernet"
        for d in /sys/class/net/*; do
            dev=$(basename "$d")
            [ -e "$d/device" ] && [ "$(cat "$d/type")" = 1 ] && [ ! -d "$d/wireless" ] || continue
            case $dev in thunderbolt*|tailscale*|wl*) continue ;; esac
            grep -qs "^iface $dev " /etc/network/interfaces $(ls /etc/network/interfaces.d/* 2>/dev/null | grep -vxF "$own") && continue
            printf 'allow-hotplug %s\niface %s inet dhcp\niface %s inet6 auto\n\n' "$dev" "$dev" "$dev" >> "$stage/ethernet"
        done
        if [ -s "$stage/ethernet" ]; then
            { echo "# DHCP on Ethernet ports the installer did not configure (nodes/macpro/install.sh)"; cat "$stage/ethernet"; } > "$stage/ethernet2"
            put "$stage/ethernet2" "$own" 644
            grep -qs '^source.*interfaces.d' /etc/network/interfaces || note "network: /etc/network/interfaces does not source interfaces.d; add 'source /etc/network/interfaces.d/*'"
        elif [ -e "$own" ]; then
            act "remove $own (every port is configured elsewhere)" rm -f "$own"
        fi
    fi
fi
# ----------------------------------------------------------------------------------------------------------- power
if want power; then
    echo "-- power"
    for t in sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target; do
        if [ "$(systemctl is-enabled "$t" 2>/dev/null)" != masked ]; then
            act "mask $t" systemctl mask -q "$t"
            applying && echo "$t" >> "$STATE/masked"
        fi
    done
fi
# ------------------------------------------------------------------------------------------------------------- gpu
if want gpu; then
    echo "-- gpu"
    # amdgpu, not radeon, drives the Southern Islands D700s. Already on the command line (from /etc/default/grub or
    # an earlier run) means nothing to add.
    current=$(bash -c '. /etc/default/grub 2>/dev/null; for f in /etc/default/grub.d/*.cfg; do [ -e "$f" ] && [ "$f" != /etc/default/grub.d/90-trashcan.cfg ] && . "$f"; done; echo "$GRUB_CMDLINE_LINUX_DEFAULT $GRUB_CMDLINE_LINUX"')
    missing=""
    for p in radeon.si_support=0 amdgpu.si_support=1; do [[ " $current " == *" $p "* ]] || missing+=" $p"; done
    if [ -n "$missing" ]; then
        echo "GRUB_CMDLINE_LINUX_DEFAULT=\"\$GRUB_CMDLINE_LINUX_DEFAULT$missing\"" > "$stage/grub.cfg"
        put "$stage/grub.cfg" /etc/default/grub.d/90-trashcan.cfg 644
    fi
    grep -q 'amdgpu.si_support=1' /proc/cmdline || note "gpu: the running kernel has no amdgpu.si_support=1; it takes effect at the next boot"
    if user_ok; then
        for g in render video; do
            if ! id -nG "$user" | tr ' ' '\n' | grep -qx "$g"; then
                act "add $user to $g (from the next login)" usermod -aG "$g" "$user"
                applying && echo "$user $g" >> "$STATE/groups"
            fi
        done
    else
        note "gpu: no user given (--user USER or run through sudo); add the account that runs the engine to render and video"
    fi
    # The engine raises its GPUs' DPM level while it works (engine/src/clocks.rs: `sudo -n tee FILE...`, files in
    # sorted order). Allow exactly those writes for render, for every card number amdgpu may get.
    {
        echo "# The trashcan engine sets its GPUs' DPM level (engine/src/clocks.rs). Installed by nodes/macpro/install.sh."
        f() { echo "/sys/class/drm/card$1/device/power_dpm_force_performance_level"; }
        cmds=()
        for i in 0 1 2 3; do cmds+=("/usr/bin/tee $(f $i)"); done
        for i in 0 1 2 3; do for j in 0 1 2 3; do [ $i -lt $j ] && cmds+=("/usr/bin/tee $(f $i) $(f $j)"); done; done
        printf '%%render ALL=(root) NOPASSWD: %s' "${cmds[0]}"
        printf ', \\\n    %s' "${cmds[@]:1}"
        echo
    } > "$stage/sudoers"
    visudo -cqf "$stage/sudoers" || { echo "generated sudoers rule does not parse" >&2; exit 1; }
    put "$stage/sudoers" /etc/sudoers.d/trashcan-amdgpu-dpm 440
fi
# ------------------------------------------------------------------------------------------------------ logs/panic
if want logs; then
    echo "-- logs"
    put "$R/etc/systemd/journald.conf.d/90-trashcan-crash.conf" /etc/systemd/journald.conf.d/90-trashcan-crash.conf 644
fi
if want panic; then
    echo "-- panic"
    put "$R/etc/sysctl.d/90-trashcan-panic.conf" /etc/sysctl.d/90-trashcan-panic.conf 644
    put "$R/usr/local/sbin/emergency-reboot" /usr/local/sbin/emergency-reboot 755
fi
if want ssh; then
    echo "-- ssh"
    put "$R/etc/ssh/sshd_config.d/50-keepalive.conf" /etc/ssh/sshd_config.d/50-keepalive.conf 644
fi
if want disk && [ -n "$(apple_ssds)" ]; then
    echo "-- disk"
    put "$R/etc/udev/rules.d/60-trashcan-apple-ssd.rules" /etc/udev/rules.d/60-trashcan-apple-ssd.rules 644
    # The rule acts when the disk appears (boot); set the running disk too.
    for d in $(apple_ssds); do
        [ "$(cat "$d/cache_type")" = "write through" ] || act "turn the write cache of $(ls "$d/device/block") off" sh -c "echo 'write through' > $d/cache_type"
    done
fi

# ------------------------------------------------------------------------------------- reload what changed, services
echo "-- services"
is_changed '/etc/systemd/system/*' && act "systemctl daemon-reload" systemctl daemon-reload
if want fan; then
    is_changed /usr/local/sbin/fancurve.sh || is_changed /etc/systemd/system/fancurve.service && r=1 || r=0
    if [ -d /sys/devices/platform/applesmc.768 ]; then enable_service fancurve.service $r
    else note "fan: no applesmc device (not a Mac?); fancurve.service not started"; fi
fi
if want tblink; then
    # A restart re-reads the script; it keeps every link, address and stream (it only re-adds missing ones).
    is_changed /usr/local/sbin/tblink || is_changed /etc/systemd/system/tblink.service || is_changed /etc/tblink/nodes.tsv && r=1 || r=0
    enable_service tblink.service $r
    if is_changed /etc/udev/rules.d/70-tbstream.rules; then
        act "reload udev rules for /dev/tbstream*" bash -c "udevadm control --reload && udevadm trigger --action=change --sysname-match='tbstream*'"
    fi
    if [ "$tbopt" = new ]; then
        note "tblink: the thunderbolt_net throttle applies when the module next loads (reboot, or tblink reload with no engine running)"
    fi
fi
if is_changed /etc/NetworkManager/conf.d/90-thunderbolt-unmanaged.conf && systemctl -q is-active NetworkManager; then
    act "reload NetworkManager" systemctl reload NetworkManager
fi
if is_changed /etc/network/interfaces.d/trashcan-ethernet; then
    for dev in $(awk '/^iface/ {print $2}' "$stage/ethernet" 2>/dev/null | sort -u); do
        # A port without a cable keeps waiting for carrier in the background.
        act "bring up $dev (DHCP)" bash -c "ifup --allow=hotplug $dev > /dev/null 2>&1 || true"
    done
fi
is_changed /etc/default/grub.d/90-trashcan.cfg && act "update-grub (kernel command line, next boot)" bash -c 'update-grub 2> /dev/null'
is_changed /etc/sysctl.d/90-trashcan-panic.conf && act "apply the panic sysctls" sysctl -q -p /etc/sysctl.d/90-trashcan-panic.conf
if is_changed /etc/systemd/journald.conf.d/90-trashcan-crash.conf; then
    act "restart journald (persistent journal)" bash -c 'systemctl restart systemd-journald.service && journalctl --flush'
fi
is_changed /etc/ssh/sshd_config.d/50-keepalive.conf && act "reload sshd" bash -c 'sshd -t && systemctl reload ssh'
is_changed /etc/udev/rules.d/60-trashcan-apple-ssd.rules && act "reload udev rules" udevadm control --reload

# ------------------------------------------------------------------------------------------------ prerequisites
grep -qs ' /sys/kernel/config configfs ' /proc/mounts || note "configfs is not mounted: tblink cannot create streams (links run over TCP)"
grep -qs ' /sys/kernel/debug debugfs ' /proc/mounts || note "debugfs is not mounted: the port lights cannot see GPU activity"

# ---------------------------------------------------------------------------------------------------------- lights
if [ "$lights" = 1 ] && want lights; then
    echo "-- lights (lights/install.sh)"
    if [ -f "$repo/lights/install.sh" ]; then
        # It has its own preview (--dry-run), state and messages; one line per step.
        out=$(bash "$repo/lights/install.sh" $(applying || echo --dry-run) $lights_args 2>&1) || { echo "$out" | sed 's/^/  /'; echo "lights/install.sh failed" >&2; exit 1; }
        echo "$out" | sed 's/^/  /'
        applying && touch "$STATE/lights"
    else
        note "lights: this checkout has no lights/install.sh"
    fi
fi

echo
for n in "${notes[@]}"; do echo "note: $n"; done
if [ "$changes" = 0 ]; then
    echo "$(hostname): nothing to change"
elif applying; then
    echo "$(hostname): $changes changes applied (state in $STATE; --uninstall reverts them)"
else
    echo "$(hostname): $changes changes; rerun with --apply"
fi
