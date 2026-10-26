#!/bin/bash
# Installs the port lights on this Mac Pro 6,1: portlights-node.service, which runs blinkenlights while the GPUs
# work and otherwise leaves the lights to the SMC.
#
#   sudo lights/install.sh [--dry-run] [--bin FILE] [--module KO | --build-module]
#                          [--conductor [ADDR:]PORT | --no-conductor] [--no-start]
#
# Contract (for installers that call this script):
# - Runs on the Mac, as root, from a trashcan source tree (it uses lights/ and kernel/patches-7.3/). Linux x86-64.
# - Installs /usr/local/bin/portlights (--bin FILE; default target/release/portlights or
#   target/x86_64-unknown-linux-gnu/release/portlights in this tree), /etc/systemd/system/portlights-node.service
#   and /etc/default/portlights (PORTLIGHTS_ARGS: node options). Enables portlights-node and restarts it if
#   anything changed or it is not running (--no-start: enable only).
# - Driver: if the running kernel already has the port lights (patches 0009 and 0010, built in or loaded), nothing
#   is done. --module KO installs a module built with lights/driver.sh build; --build-module builds it here first
#   (needs gcc, make, patch, curl and linux-headers-$(uname -r)). Otherwise it says so and the node waits for the
#   driver. Installing the module reloads applesmc: fancurve stops for a few seconds and the SMC runs the fan.
# - Conductor frames (the music visualizer) are off by default: --conductor PORT (e.g. 7465) or ADDR:PORT lets
#   the node take them on UDP; --no-conductor turns that off. Without either, an existing /etc/default/portlights
#   is kept as it is.
# - Idempotent: running it again changes nothing that is already in place. It prints one line per file or
#   step (installed, updated, unchanged). --dry-run prints the same and changes nothing.
# - Never reboots, never touches the network, links or GPUs.
# - Exit status: 0 installed (or already installed), 1 failed, 2 usage. lights/uninstall.sh undoes it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
bin_dest=/usr/local/bin/portlights
unit_dest=/etc/systemd/system/portlights-node.service
conf=/etc/default/portlights
smc=/sys/devices/platform/applesmc.768

say() { echo "portlights: $*"; }
die() { echo "portlights: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^set -e/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0" >&2; exit 2; }

dry="" bin="" module="" build_module="" listen="" set_listen="" no_start=""
while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) dry=1; shift ;;
        --bin) bin=${2:?}; shift 2 ;;
        --module) module=${2:?}; shift 2 ;;
        --build-module) build_module=1; shift ;;
        --conductor) listen=${2:?}; set_listen=1; shift 2 ;;
        --no-conductor) listen=""; set_listen=1; shift ;;
        --no-start) no_start=1; shift ;;
        *) usage ;;
    esac
done
[ -z "$module" ] || [ -z "$build_module" ] || usage

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "runs on the Mac Pro (Linux x86-64)"
[ -n "$dry" ] || [ "$(id -u)" = 0 ] || die "run as root (sudo)"
[ -d $smc ] || die "no applesmc device ($smc): not a Mac, or applesmc is not loaded"

# the binary
if [ -z "$bin" ]; then
    for b in "$root/target/release/portlights" "$root/target/x86_64-unknown-linux-gnu/release/portlights"; do
        if [ -f "$b" ]; then bin=$b; break; fi
    done
    [ -n "$bin" ] || die "no portlights binary: build it (cargo build --release -p portlights) or pass --bin FILE"
fi
[ -f "$bin" ] || die "no file $bin"
[ "$(od -An -tx1 -j18 -N2 "$bin" | tr -d ' ')" = 3e00 ] || die "$bin is not an x86-64 Linux binary"
runs=$("$bin" 2>&1 || true)
grep -q '^usage' <<<"$runs" || die "$bin does not run here"

changed=""
# put SRC at DEST (mode MODE) if it differs
place() {
    local src=$1 dest=$2 mode=$3 what
    if [ -f "$dest" ] && cmp -s "$src" "$dest"; then
        say "$dest: unchanged"
        return
    fi
    if [ -f "$dest" ]; then what=updated; else what=installed; fi
    changed=1
    if [ -n "$dry" ]; then say "$dest: would be $what"; return; fi
    install -D -m "$mode" "$src" "$dest"
    say "$dest: $what"
}

# the driver
if [ -e $smc/port_lights_levels ]; then
    say "driver: the port lights are there"
elif [ -n "$module" ] || [ -n "$build_module" ]; then
    if [ -n "$build_module" ]; then
        if [ -n "$dry" ]; then
            say "driver: would build applesmc for $(uname -r) (lights/driver.sh build)"
            module=""
        else
            module=$("$here/driver.sh" build | tail -1)
        fi
    fi
    if [ -n "$module" ]; then
        "$here/driver.sh" install ${dry:+--dry-run} "$module"
    elif [ -n "$dry" ]; then
        say "driver: would install it and reload applesmc"
    fi
else
    say "driver: missing; the node waits for it (lights/driver.sh, or --module / --build-module)"
fi

place "$bin" $bin_dest 0755
place "$here/portlights-node.service" $unit_dest 0644

# /etc/default/portlights
want=""
if [ -n "$set_listen" ] || [ ! -f $conf ]; then
    args=""
    [ -z "$listen" ] || args="--listen $listen"
    want=$(printf '%s\n' \
        "# Options for portlights node. --listen [ADDR:]PORT takes frames from a conductor (the music visualizer)." \
        "PORTLIGHTS_ARGS=\"$args\"")
fi
if [ -n "$want" ]; then
    if [ -f $conf ] && [ "$(cat $conf)" = "$want" ]; then
        say "$conf: unchanged"
    else
        if [ -f $conf ]; then what=updated; else what=installed; fi
        changed=1
        if [ -n "$dry" ]; then
            say "$conf: would be $what: $(tail -1 <<<"$want")"
        else
            printf '%s\n' "$want" >$conf
            chmod 0644 $conf
            say "$conf: $what: $(tail -1 <<<"$want")"
        fi
    fi
else
    say "$conf: kept"
fi

if [ -n "$dry" ]; then
    if [ -n "$changed" ] || ! systemctl is-active -q portlights-node; then
        if [ -n "$no_start" ]; then say "portlights-node: would be enabled"
        else say "portlights-node: would be enabled and restarted"; fi
    else
        say "portlights-node: running, unchanged"
    fi
    exit 0
fi
systemctl daemon-reload
systemctl enable -q portlights-node.service
if [ -n "$no_start" ]; then
    say "portlights-node: enabled, not started"
elif [ -n "$changed" ] || ! systemctl is-active -q portlights-node; then
    systemctl restart portlights-node.service
    sleep 1
    systemctl is-active -q portlights-node || die "portlights-node did not start: journalctl -u portlights-node"
    say "portlights-node: running ($(journalctl -u portlights-node -n 1 -o cat --no-pager 2>/dev/null))"
else
    say "portlights-node: running, unchanged"
fi
