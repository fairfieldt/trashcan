#!/bin/bash
# Wait for a one-Mac gpuq ticket, run SCRIPT in ~/trashcan-exp/flashnext under it, release with its last line.
# usage: run1mac.sh MAC TICKET SCRIPT
m=$1 id=$2 script=$3
while :; do
    gpuq --on "$m" wait "$id" > /dev/null; rc=$?
    [ $rc -eq 2 ] && continue
    [ $rc -eq 0 ] && break
    echo "$m #$id ended before it was granted (rc $rc)"; exit 1
done
out=$(ssh "$m" "cd ~/trashcan-exp/flashnext && gpuq run $id -- ./$script" 2>&1)
echo "$out" | tail -12
gpuq --on "$m" release "$id" -r "$(echo "$out" | tail -1 | cut -c1-200)" > /dev/null 2>&1
echo "$m #$id released"
