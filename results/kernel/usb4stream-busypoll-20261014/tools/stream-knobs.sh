#!/bin/bash
# On a Mac Pro with kernel/patches-7.3-diag/0008: set the experiment switches of tblink's `trashcan` stream on the link to
# PEER (only while nothing has the device open; used from its next open) and print them.
# usage: stream-knobs.sh PEER [ATTR=VALUE ...]     e.g. stream-knobs.sh enlight paths_last=1 poll_mode=0
set -uo pipefail
CFG=/sys/kernel/config/thunderbolt/stream
DEVS=/sys/bus/thunderbolt/devices
peer=$1
shift
xd=$(/usr/local/sbin/tblink links | awk -F'\t' -v p="$peer" '$2 == p {print $1}')
[ -n "$xd" ] || { echo "no link to $peer" >&2; exit 1; }
svc=
for s in "$DEVS/$xd/$xd".*; do [ "$(cat "$s/key" 2>/dev/null)" = stream ] && svc=$(basename "$s"); done
d=$CFG/$svc/trashcan
[ -f "$d/paths_last" ] || { echo "$d has no experiment switches (kernel $(uname -r))" >&2; exit 1; }
for kv in "$@"; do
    echo "${kv#*=}" | sudo -n tee "$d/${kv%%=*}" > /dev/null || { echo "cannot set $kv" >&2; exit 1; }
done
for a in rx_e2e paths_last poll_mode rx_repost rx_eof_close busy_poll throttling; do printf '%s=%s ' "$a" "$(cat "$d/$a")"; done
echo
