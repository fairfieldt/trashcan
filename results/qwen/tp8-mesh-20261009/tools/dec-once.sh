#!/bin/bash
# Decode rows through the one-step path (the default) against the direct path (full:0), f16 payload, alternating.
set -uo pipefail
T=$(dirname "$0")
for i in 1 2 3; do
  $T/tp8full.sh m-dec1-once-$i trashcan-mesh "qwen-bench --pp 512 --tg 128 --reps 2" "TRASHCAN_TP_WIRE=f16 TRASHCAN_TP_MESH=full"
  $T/tp8full.sh m-dec1-direct-$i trashcan-mesh "qwen-bench --pp 512 --tg 128 --reps 2" "TRASHCAN_TP_WIRE=f16 TRASHCAN_TP_MESH=full:0"
done
