#!/bin/bash
# Round 3 A/B on the six-cable TP8 cluster: trashcan-c3 (eabff7b plus qwen_q4c3_r5, five rows per wave for
# three-column shards past 2048 rows) against itself with the old kernel (TRASHCAN_QWEN_C3=r4), alternated.
# usage: TICKETS=incite=ID,invite=ID,enlight=ID,indite=ID c3.sh PHASE   PHASE: mtp | profile | gates | decode
set -uo pipefail
here=$(dirname "$0")
MTP=${MTP:-/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf}
run() { "$here/../../tp8-mesh-20261009/tools/tp8full.sh" "$@"; }
BENCH="qwen-bench --pp 512 --tg 128 --reps 2"
OLD=TRASHCAN_QWEN_C3=r4
case ${1:?phase} in
mtp)
  for i in 1 2; do
    run c3-mtp-r4-$i trashcan-c3 "$BENCH --mtp $MTP" "$OLD"
    run c3-mtp-r5-$i trashcan-c3 "$BENCH --mtp $MTP" ""
  done ;;
profile)
  run c3-prof-r4 trashcan-c3 "qwen-bench --pp 512 --tg 32 --reps 1 --mtp $MTP" "$OLD" --profile
  run c3-prof-r5 trashcan-c3 "qwen-bench --pp 512 --tg 32 --reps 1 --mtp $MTP" "" --profile
  ;;
gates)
  run c3-val32 trashcan-c3 "qwen-validate results/c3-val32" "TRASHCAN_TP_WIRE=f32"
  run c3-check-v3 trashcan-c3 "qwen-check --verify 3" ""
  run c3-check-mtp trashcan-c3 "qwen-check --mtp $MTP" ""
  ;;
decode)
  run c3-dec-r5 trashcan-c3 "$BENCH" ""
  ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
