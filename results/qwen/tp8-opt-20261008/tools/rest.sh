#!/bin/bash
# The rest of the merged candidates' evaluation on the four-cable ring (runbook sections 4-6): MTP for W1 and A1,
# decode after 7680 tokens for A1, then the correctness gates for W1, A1 and the default.
set -euo pipefail
T=$(dirname "$0")
MTP=/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"
sw() { case $1 in w1) echo TRASHCAN_TP_WIRE=f16;; a1) echo TRASHCAN_QWEN_DECODE_ATTN=1;; esac; }
phase=${1:-all}
if [ $phase = all ] || [ $phase = mtp ]; then
for cfg in w1 a1; do
  for i in 1 2; do
    $T/tp8.sh mtp-default-$cfg-$i trashcan-opt "$DECODE --mtp $MTP" ""
    $T/tp8.sh mtp-$cfg-$i trashcan-opt "$DECODE --mtp $MTP" "$(sw $cfg)"
  done
done
fi
if [ $phase = all ] || [ $phase = long ]; then
for i in 1 2; do
  $T/tp8.sh long-default-$i trashcan-opt "qwen-bench --pp 7680 --tg 64" ""
  $T/tp8.sh long-a1-$i trashcan-opt "qwen-bench --pp 7680 --tg 64" "$(sw a1)"
done
fi
if [ $phase = all ] || [ $phase = gates ]; then
for cfg in w1 a1; do
  env=$(sw $cfg)
  $T/tp8.sh val-$cfg trashcan-opt "qwen-validate results/val-$cfg" "$env"
  $T/tp8.sh check-mtp-$cfg trashcan-opt "qwen-check --mtp $MTP" "$env"
  $T/tp8.sh check-v3-$cfg trashcan-opt "qwen-check --verify 3" "$env"
done
$T/tp8.sh val-default trashcan-opt "qwen-validate results/val-default" ""
$T/tp8.sh check-v3-default trashcan-opt "qwen-check --verify 3" ""
fi
