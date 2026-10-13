#!/bin/bash
# Wait for a one-Mac gpuq lease, run a workspace script under it, release with its last line.
# usage: run-lease.sh MACHINE TICKET SCRIPT
set -uo pipefail
m=$1 id=$2 script=$3
while :; do
    gpuq --on "$m" wait "$id" > /dev/null; rc=$?
    [ $rc -eq 2 ] && continue
    [ $rc -eq 0 ] && break
    echo "ticket $id ended before it was granted (rc $rc)"; exit 1
done
out=$(ssh "$m" "cd ~/trashcan-exp/flashnext && gpuq run $id -- ./$script" 2>&1)
echo "$out" | cut -c1-120
gpuq --on "$m" release "$id" -r "$(echo "$out" | tail -1 | cut -c1-100)"
