#!/bin/bash
# Copy a window's results into this directory: the window's local output (steps, link-test JSON, driver checks,
# states) and, for the chain and TP runs, the drivers' JSON/err, the nodes' logs and every clock log (stream-* names)
# from the Macs' workspaces. Dumps of validation runs stay out (large); their comparison is in the window output.
# usage (from the repo): results/kernel/usb4stream-reliable-20261013/tools/collect.sh WINDOW_OUT NAME
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$1 name=$2
dst=$here/../$name
mkdir -p "$dst"
rsync -a --exclude '*.out' "$src/" "$dst/"
for h in incite indite inflight alight invite enlight; do
    for ws in trashcan-exp/flashnext trashcan-exp/stream-tp; do
        mkdir -p "$dst/$h"
        rsync -a --include 'stream-*.json' --include 'stream-*.err' --include 'node-stream-*.log' --include 'clocks/' \
            --include 'clocks/*stream-*' --exclude '*' "$h:$ws/results/" "$dst/$h/" 2>/dev/null || true
    done
    find "$dst/$h" -type d -empty -delete 2>/dev/null || true
done
du -sh "$dst"
