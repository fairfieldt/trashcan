#!/bin/bash
# Record vs merged default vs merged with H1 off, alternating, decode.
set -euo pipefail
T=$(dirname "$0")
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"
for i in 1 2; do
  $T/tp8.sh h1-old-$i trashcan-base "$DECODE" ""
  $T/tp8.sh h1-new-$i trashcan-opt "$DECODE" ""
  $T/tp8.sh h1-off-$i trashcan-opt "$DECODE" "TRASHCAN_TP_HOSTPASS=0"
done
