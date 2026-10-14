#!/bin/bash
# Copy this work's run outputs (names starting with host-) from every Mac's ~/trashcan-exp/flashnext/results into
# results/qwen4/decode-host-20261013/HOST/ (driver outputs, node logs, clock logs, traces). Arguments: the Macs to copy
# from (default all six).
cd "$(dirname "$0")/.." || exit 1
hosts=${*:-incite indite inflight alight invite enlight}
rc=0
for m in $hosts; do
    mkdir -p "$m/clocks"
    rsync -a --include='host-*' --include='node-host-*' --exclude='*' "$m:trashcan-exp/flashnext/results/" "$m/" &&
        rsync -a --include='host-*' --include='node-host-*' --exclude='*' "$m:trashcan-exp/flashnext/results/clocks/" "$m/clocks/" ||
        { echo "fetch: $m failed"; rc=1; }
done
exit $rc
