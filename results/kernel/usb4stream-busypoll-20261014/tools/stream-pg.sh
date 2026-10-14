#!/bin/bash
# On a Mac Pro, a probe stream beside tblink's `trashcan` (CPU-only probes on fresh streams, which carry no CLOSE of an
# earlier session):
#   stream-pg.sh up PEER NAME HOPID BUSY   create NAME on the link to PEER with input and output HopID HOPID and
#                                          busy_poll BUSY (0 or 1); prints its device
#   stream-pg.sh down NAME                 remove every stream NAME
set -uo pipefail
CFG=/sys/kernel/config/thunderbolt/stream
DEVS=/sys/bus/thunderbolt/devices
case ${1:-} in
    up)
        peer=$2 name=$3 hop=$4 busy=$5
        xd=$(/usr/local/sbin/tblink links | awk -F'\t' -v p="$peer" '$2 == p {print $1}')
        [ -n "$xd" ] || { echo "no link to $peer" >&2; exit 1; }
        svc=
        for s in "$DEVS/$xd/$xd".*; do [ "$(cat "$s/key" 2>/dev/null)" = stream ] && svc=$(basename "$s"); done
        [ -n "$svc" ] || { echo "$xd has no stream service" >&2; exit 1; }
        d=$CFG/$svc/$name
        sudo -n mkdir -p "$d" || exit 1
        for a in in_hopid out_hopid; do
            [ "$(cat "$d/$a")" = "$hop" ] || echo "$hop" | sudo -n tee "$d/$a" > /dev/null || { echo "cannot set $a $hop" >&2; exit 1; }
        done
        echo "$busy" | sudo -n tee "$d/busy_poll" > /dev/null || { echo "cannot set busy_poll" >&2; exit 1; }
        dev=/dev/tbstream$(cat "$d/index")
        for t in 1 2 3 4 5 6 7 8 9 10; do [ -e "$dev" ] && break; sleep 0.2; done
        sudo -n chgrp render "$dev" && sudo -n chmod 660 "$dev"
        echo "$dev busy_poll=$(cat "$d/busy_poll") hopids=$(cat "$d/in_hopid")/$(cat "$d/out_hopid")"
        ;;
    down)
        for d in "$CFG"/*/"$2"; do [ -d "$d" ] && sudo -n rmdir "$d"; done
        ls -d "$CFG"/*/* 2> /dev/null | tr '\n' ' '
        ;;
    *) echo "usage: stream-pg.sh up PEER NAME HOPID BUSY | down NAME" >&2; exit 2 ;;
esac
