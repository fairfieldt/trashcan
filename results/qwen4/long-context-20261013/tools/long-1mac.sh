#!/bin/bash
# One single-Mac gpuq job: request a lease on MACHINE, wait for it, run CMD there under the lease (in
# ~/trashcan-exp/flashnext, through bash -c; no single quotes in CMD), then release with CMD's last output line.
# usage: long-1mac.sh MACHINE LABEL EST_MIN -- CMD...
set -uo pipefail
m=$1 label=$2 est=$3
shift 3
[ "${1:-}" = -- ] && shift
cmd="$*"
id=$(gpuq --on "$m" request "$label" --est "$est" --owner "Flash-Next long-context agent" \
    | sed -n 's/^Ticket #\([0-9]*\) created.*/\1/p')
[ -n "$id" ] || { echo "$m: request failed"; exit 1; }
trap 'gpuq --on "$m" release "$id" -r "stopped" > /dev/null 2>&1; exit 130' INT TERM
echo "$m: ticket $id queued $(date +%H:%M:%S)"
gpuq --on "$m" wait "$id" > /dev/null || { echo "$m: ticket $id ended while waiting"; exit 1; }
echo "$m: granted $(date +%H:%M:%S): $cmd"
out=$(ssh "$m" "cd trashcan-exp/flashnext && gpuq run $id -- bash -c '$cmd'" 2>&1)
rc=$?
echo "$out"
gpuq --on "$m" release "$id" -r "(rc $rc) $(echo "$out" | tail -1 | cut -c1-200)" > /dev/null 2>&1
echo "$m: released $(date +%H:%M:%S) rc=$rc"
exit $rc
