#!/bin/bash
# Each merged candidate against the default of the same binary, alternating, decode after pp512 (runbook section 5,
# without h1off, measured separately in h1-*).
set -euo pipefail
T=$(dirname "$0")
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"
sw() { case $1 in w1) echo TRASHCAN_TP_WIRE=f16;; a1) echo TRASHCAN_QWEN_DECODE_ATTN=1;; k1) echo TRASHCAN_QWEN_TOPK=wave;;
  f1) echo TRASHCAN_QWEN_FUSED_FFN=1;; g1) echo TRASHCAN_QWEN_FUSED_GDN=1;; m1) echo TRASHCAN_QWEN_SHORTK=16;; esac; }
for cfg in ${CFGS:-w1 a1 k1 f1 g1 m1}; do
  for i in 1 2; do
    $T/tp8.sh dec-default-$cfg-$i trashcan-opt "$DECODE" ""
    $T/tp8.sh dec-$cfg-$i trashcan-opt "$DECODE" "$(sw $cfg)"
  done
done
