#!/bin/bash
# Set up ~/trashcan-exp/stream-tp on incite and NODE for tp4-pair.sh (files only): the run scripts from this checkout,
# main's binary from ~/trashcan-exp/flashnext and this branch's binary.
# usage (from the repo): results/kernel/usb4stream-reliable-20261013/tools/tp4-workspace.sh NODE MAIN_BINARY_NAME BINARY
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
node=$1 main=$2 bin=$3
for h in incite "$node"; do
    ssh "$h" "mkdir -p ~/trashcan-exp/stream-tp/scripts ~/trashcan-exp/stream-tp/results && cp ~/trashcan-exp/flashnext/$main ~/trashcan-exp/stream-tp/$main"
    rsync -a "$repo/scripts/with-clocks.sh" "$repo/scripts/clockmon.sh" "$repo/scripts/clocksummary.sh" "$repo/scripts/latency-run.py" \
        "$h:trashcan-exp/stream-tp/scripts/"
    rsync -a "$bin" "$h:trashcan-exp/stream-tp/trashcan-stream"
    echo "$h: $(ssh "$h" "cd ~/trashcan-exp/stream-tp && md5sum $main trashcan-stream | cut -c1-12 | tr '\n' ' '; ls scripts | tr '\n' ' '")"
done
