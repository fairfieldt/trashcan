#!/bin/bash
# Copy the test binary and the window helpers to every Mac's ~/trashcan-exp/flashnext (no lease: files only).
# usage (from the repo): results/kernel/usb4stream-reliable-20261013/tools/deploy-files.sh [BINARY]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
bin=${1:-$repo/target/trashcan-incite}
for h in incite indite inflight alight invite enlight; do
    rsync -a "$bin" "$h:trashcan-exp/flashnext/trashcan-stream"
    rsync -a "$repo/nodes/macpro/root/usr/local/sbin/tblink" "$h:trashcan-exp/flashnext/stream-tblink"
    rsync -a "$here/stream-ops.sh" "$h:trashcan-exp/flashnext/stream-ops.sh"
    rsync -a "$here/tbcheck.py" "$h:trashcan-exp/flashnext/stream-tbcheck.py"
    echo "$h: $(ssh "$h" 'md5sum ~/trashcan-exp/flashnext/trashcan-stream | cut -c1-12')"
done
