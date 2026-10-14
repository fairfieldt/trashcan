#!/bin/bash
# On a Mac Pro, for the hand-run tests before tblink.service carried the streams (window 1):
#   stream-ops.sh up        one pass of the new tblink (stream-tblink next to this script): streams on every working
#                           link; the devices get the udev rule's group and mode; prints peer, state, stream
#   stream-ops.sh down      remove the `trashcan` streams and their groups once empty (what `up` made)
#   stream-ops.sh dev PEER  this Mac's stream device on the link to PEER
#   stream-ops.sh state     modules, stream groups, devices, links (the "as found" record)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cfg=/sys/kernel/config/thunderbolt/stream
case ${1:-} in
    up)
        sudo timeout 4 "$here/stream-tblink" run > "$here/stream-tblink-pass.log" 2>&1
        sudo chgrp render /dev/tbstream* 2> /dev/null
        sudo chmod 660 /dev/tbstream* 2> /dev/null
        "$here/stream-tblink" links | tail -n +2 | cut -f2,6,10
        ;;
    down)
        for d in "$cfg"/*/trashcan; do [ -d "$d" ] && sudo rmdir "$d" && sudo rmdir "$(dirname "$d")" 2> /dev/null; done
        ls "$cfg"
        ;;
    dev)
        "$here/stream-tblink" links | awk -F'\t' -v p="$2" '$2 == p { print $10 }'
        ;;
    state)
        echo "$(hostname) $(date +%s.%N)"
        lsmod | grep -E '^thunderbolt' | awk '{print $1, $3}'
        find "$cfg" -mindepth 1 -maxdepth 2 2> /dev/null | sort
        ls -l /dev/tbstream* 2> /dev/null
        /usr/local/sbin/tblink links
        pgrep -af 'trashcan|link-test' | grep -v pgrep
        ;;
    *) echo "usage: stream-ops.sh up|down|dev PEER|state" >&2; exit 2 ;;
esac
