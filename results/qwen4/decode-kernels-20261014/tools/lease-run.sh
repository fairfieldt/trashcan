#!/bin/bash
# Wait for a gpuq ticket on a Mac (re-waiting on timeouts), run a session script there under it, then release.
# usage: lease-run.sh HOST TICKET SCRIPT RESULT_TEXT
host=$1 id=$2 script=$3 result=$4
while :; do
    gpuq --on "$host" wait "$id" > /dev/null
    rc=$?
    [ $rc -eq 2 ] && continue
    [ $rc -eq 0 ] && break
    echo "ticket $id ended before it was granted (rc $rc)"
    exit 1
done
echo "granted $(date +%T)"
ssh "$host" "cd ~/trashcan-exp/flashnext && gpuq run $id -- ./$script" 2>&1 | tail -40
gpuq --on "$host" release "$id" -r "$result" > /dev/null 2>&1
echo "released $(date +%T)"
