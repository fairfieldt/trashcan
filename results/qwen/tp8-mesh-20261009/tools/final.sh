#!/bin/bash
# The committed code (trashcan-final: full mesh chosen from the probe, f16 below prefill size by default): six cables
# against four, defaults only, then its correctness gates.
set -uo pipefail
T=$(dirname "$0")
MTP=/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"
for i in 1 2; do
  $T/tp8full.sh f-six-$i trashcan-final "$DECODE" ""
  $T/../../tp8-opt-20261008/tools/tp8.sh f-four-$i trashcan-final "$DECODE" ""
done
for i in 1 2; do
  $T/tp8full.sh f-six-mtp-$i trashcan-final "$DECODE --mtp $MTP" ""
  $T/../../tp8-opt-20261008/tools/tp8.sh f-four-mtp-$i trashcan-final "$DECODE --mtp $MTP" ""
done
$T/tp8full.sh f-val32 trashcan-final "qwen-validate results/f-val32" "TRASHCAN_TP_WIRE=f32"
$T/tp8full.sh f-check-mtp trashcan-final "qwen-check --mtp $MTP" ""
$T/tp8full.sh f-check-v3 trashcan-final "qwen-check --verify 3" ""
