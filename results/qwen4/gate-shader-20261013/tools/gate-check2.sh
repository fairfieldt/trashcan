#!/bin/bash
# Round 2 of the bounded gate's single-GPU check on alight (one-Mac lease, --est 3): copies host-gate2-run.sh there and
# runs it under the lease. DRY=1: copy, show the script and the binary's checksum, run nothing on the GPU.
set -uo pipefail
h=alight
W=trashcan-exp/flashnext
src=/Users/fairfieldt/dev/claude-work/flashnext-20261012/host/host-gate2-run.sh
rsync -a "$src" "$h:$W/host-gate2-run.sh"
if [ -n "${DRY:-}" ]; then
    ssh -- "$h" "cd $W && md5sum trashcan-host-gate2 | cut -c1-12 && bash -n host-gate2-run.sh && echo syntax ok && ./trashcan-host-gate2 qwen4 gate-check --help | head -20"
    exit 0
fi
id=$(gpuq --on "$h" request "host gate-check 2" --est 3 --owner "Flash-Next decode host agent" | sed -n 's/^Ticket #\([0-9]*\) created.*/\1/p')
[ -n "$id" ] || exit 1
echo "ticket $h #$id"
gpuq --on "$h" wait "$id" || exit 1
ssh -- "$h" "gpuq run $id -- bash $W/host-gate2-run.sh"
rc=$?
gpuq --on "$h" release "$id" -r "host gate-check 2: rc $rc" > /dev/null
exit $rc
