#!/bin/bash
# How a graph whose gate gave up skips its dispatches, on both GPUs of one Mac (gpuq run ID -- bash THIS, from
# ~/trashcan-exp/flashnext): conditional rendering against a flag every dispatch reads (`qwen4 gate-skip-check`).
# Every gate gives up after its first poll, so nothing waits.
cd ~/trashcan-exp/flashnext || exit 1
. ./dg-common.sh
B=${B:-./trashcan-dg1}
power
for g in 0 1; do
    step skip-gpu$g $B qwen4 gate-skip-check --gpu $g --dispatches 100 --reps 200
done
echo "skip check done; kernel log matches $(hits | wc -l) (baseline $base)"
