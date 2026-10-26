#!/bin/bash
# Removes what lights/install.sh installed and gives the port lights back to the SMC.
#
#   sudo lights/uninstall.sh [--dry-run] [--keep-driver] [--keep-config]
#
# Contract (for installers that call this script):
# - Runs on the Mac, as root, from a trashcan source tree.
# - Stops and disables portlights-node (the lights go back to the SMC, the power light steady), and removes
#   /etc/systemd/system/portlights-node.service, /usr/local/bin/portlights and /etc/default/portlights
#   (--keep-config keeps the last). Removes a port lights applesmc installed by lights/driver.sh for the
#   running kernel and reloads the kernel's own applesmc (--keep-driver keeps it); fancurve stops for a few
#   seconds meanwhile. A kernel with the patches built in keeps its port lights.
# - Idempotent; prints one line per step; --dry-run prints the same and changes nothing. Never reboots.
# - Exit status: 0 removed (or nothing to remove), 1 failed, 2 usage.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
bin_dest=/usr/local/bin/portlights
unit_dest=/etc/systemd/system/portlights-node.service
conf=/etc/default/portlights

say() { echo "portlights: $*"; }
die() { echo "portlights: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^set -e/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0" >&2; exit 2; }

dry="" keep_driver="" keep_config=""
while [ $# -gt 0 ]; do
    case $1 in
        --dry-run) dry=1; shift ;;
        --keep-driver) keep_driver=1; shift ;;
        --keep-config) keep_config=1; shift ;;
        *) usage ;;
    esac
done
[ -n "$dry" ] || [ "$(id -u)" = 0 ] || die "run as root (sudo)"

if systemctl cat portlights-node.service >/dev/null 2>&1; then
    if [ -n "$dry" ]; then
        say "portlights-node: would be stopped and disabled"
    else
        systemctl disable -q --now portlights-node.service
        say "portlights-node: stopped and disabled"
    fi
else
    say "portlights-node: not installed"
fi

files=($unit_dest $bin_dest)
[ -n "$keep_config" ] || files+=($conf)
for f in "${files[@]}"; do
    if [ ! -e "$f" ]; then
        say "$f: not there"
    elif [ -n "$dry" ]; then
        say "$f: would be removed"
    else
        rm -f "$f"
        say "$f: removed"
    fi
done
[ -n "$dry" ] || systemctl daemon-reload

if [ -n "$keep_driver" ]; then
    say "driver: kept"
else
    "$here/driver.sh" uninstall ${dry:+--dry-run}
fi
