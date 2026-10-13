#!/bin/bash
# The head candidate kernels' GPU check on both of incite's D700s, under a one-Mac lease (ticket ID on incite):
# qwen4_head_topk/qwen4_head_merge against qwen35's topk/merge, bit for bit, and their times.
# usage: topk-check.sh ID BINARY
set -uo pipefail
id=$1 bin=$2
while :; do
    gpuq --on incite wait "$id" > /dev/null; rc=$?
    [ $rc -eq 2 ] && continue
    [ $rc -eq 0 ] && break
    echo "ticket $id ended before it was granted (rc $rc)"; exit 1
done
for g in 0 1; do
    ssh incite "gpuq run $id -- bash -c 'cd ~/trashcan-exp/flashnext && ./$bin qwen4 head-topk-check --gpu $g > results/host-topk-check-$g.json 2>&1'"
    echo "gpu $g rc $?"
done
ssh incite 'cat ~/trashcan-exp/flashnext/results/host-topk-check-0.json ~/trashcan-exp/flashnext/results/host-topk-check-1.json'
gpuq --on incite release "$id" -r "head top-k check done" > /dev/null
echo "topk-check done"
