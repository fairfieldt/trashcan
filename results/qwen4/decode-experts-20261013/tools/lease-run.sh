#!/bin/bash
# Wait for a gpuq ticket on a Mac (re-waiting on timeouts), then run a session script there under it, then release.
# usage: lease-run.sh HOST TICKET EXTEND_MIN SCRIPT RESULT_TEXT
host=$1 id=$2 ext=$3 script=$4 result=$5
while :; do
    gpuq --on "$host" wait "$id" > /dev/null; rc=$?
    [ $rc -eq 2 ] && continue
    [ $rc -eq 0 ] && break
    echo "ticket $id ended before it was granted (rc $rc)"; exit 1
done
echo "granted $(date +%T)"
[ "$ext" -gt 0 ] && gpuq --on "$host" extend "$id" "$ext" -r "longer session" > /dev/null 2>&1
ssh "$host" "cd ~/trashcan-exp/flashnext && gpuq run $id -- ./$script" 2>&1 | tail -30
gpuq --on "$host" release "$id" -r "$result" > /dev/null 2>&1
echo "released $(date +%T)"
