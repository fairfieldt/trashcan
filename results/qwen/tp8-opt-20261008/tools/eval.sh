#!/bin/bash
# The back-out binary (trashcan-eval: the record + W1 below prefill sizes) on the four-cable ring: against the record
# with defaults, W1's decode and MTP pairs, then W1's correctness gates.
set -euo pipefail
T=$(dirname "$0")
MTP=/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"
W1=TRASHCAN_TP_WIRE=f16
phase=${1:-all}
if [ $phase = all ] || [ $phase = bench ]; then
for i in 1 2; do
  $T/tp8.sh e-base-$i trashcan-base "$DECODE" ""
  $T/tp8.sh e-eval-$i trashcan-eval "$DECODE" ""
  $T/tp8.sh e-evalw-$i trashcan-eval "$DECODE" "$W1"
done
for i in 1 2; do
  $T/tp8.sh e-mtp-base-$i trashcan-base "$DECODE --mtp $MTP" ""
  $T/tp8.sh e-mtp-eval-$i trashcan-eval "$DECODE --mtp $MTP" ""
  $T/tp8.sh e-mtp-evalw-$i trashcan-eval "$DECODE --mtp $MTP" "$W1"
done
fi
if [ $phase = all ] || [ $phase = gates ]; then
$T/tp8.sh e-val-eval trashcan-eval "qwen-validate results/e-val-eval" ""
$T/tp8.sh e-val-evalw trashcan-eval "qwen-validate results/e-val-evalw" "$W1"
$T/tp8.sh e-check-mtp-evalw trashcan-eval "qwen-check --mtp $MTP" "$W1"
$T/tp8.sh e-check-v3-evalw trashcan-eval "qwen-check --verify 3" "$W1"
$T/tp8.sh e-check-v3-eval trashcan-eval "qwen-check --verify 3" ""
fi
