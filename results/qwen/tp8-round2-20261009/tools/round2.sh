#!/bin/bash
# Round 2 A/B on the six-cable TP8 cluster: the record's binary (trashcan-final, a25cdd0) against trashcan-r2 (this
# round) with its switches: TRASHCAN_QWEN_MATVEC=base (kernels as before: host-side changes only), auto (two rows
# per wave, loads of two blocks ahead for long rows), and TRASHCAN_QWEN_MERGE=h (hoisted merge loads).
# usage: TICKETS=incite=ID,invite=ID,enlight=ID,indite=ID round2.sh PHASE   PHASE: decode | mtp | profile | gates | attn
set -uo pipefail
here=$(dirname "$0")
MTP=${MTP:-/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf}
run() { "$here/../../tp8-mesh-20261009/tools/tp8full.sh" "$@"; }
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"
BASE=TRASHCAN_QWEN_MATVEC=base
MERGE=TRASHCAN_QWEN_MERGE=h
case ${1:?phase} in
decode)
  for i in 1 2; do
    run r2-dec-final-$i trashcan-final "$DECODE" ""
    run r2-dec-base-$i trashcan-r2 "$DECODE" "$BASE"
    run r2-dec-auto-$i trashcan-r2 "$DECODE" ""
    run r2-dec-merge-$i trashcan-r2 "$DECODE" "$MERGE"
  done ;;
mtp)
  for i in 1 2; do
    run r2-mtp-final-$i trashcan-final "$DECODE --mtp $MTP" ""
    run r2-mtp-auto-$i trashcan-r2 "$DECODE --mtp $MTP" ""
    run r2-mtp-merge-$i trashcan-r2 "$DECODE --mtp $MTP" "$MERGE"
  done ;;
lanes)
  # trashcan-r3: trashcan-r2 with the norm pass as before; TRASHCAN_TP_LANE_THREADS=1 moves the lanes on worker
  # threads; trashcan-r2avx is trashcan-r2 built with -C target-cpu=ivybridge (AVX host passes).
  for i in 1 2; do
    run r2-lanes-auto-$i trashcan-r3 "$DECODE" ""
    run r2-lanes-thr-$i trashcan-r3 "$DECODE" "TRASHCAN_TP_LANE_THREADS=1"
    run r2-lanes-avx-$i trashcan-r2avx "$DECODE" ""
  done ;;
kernels)
  # trashcan-r4 (trashcan-r3 plus these switches): DeltaNet recurrence with 2 or 1 rows per wave, attention partials
  # walking keys two at a time.
  for i in 1 2; do
    run r2-k-auto-$i trashcan-r4 "$DECODE" ""
    run r2-k-gdn2-$i trashcan-r4 "$DECODE" "TRASHCAN_QWEN_GDN=r2"
    run r2-k-gdn1-$i trashcan-r4 "$DECODE" "TRASHCAN_QWEN_GDN=r1"
    run r2-k-p2-$i trashcan-r4 "$DECODE" "TRASHCAN_QWEN_PARTIAL=p2"
  done ;;
fused)
  # trashcan-r5 (trashcan-r4 plus the fused decode all-reduce passes, on by default): against the separate passes
  # (TRASHCAN_TP_FUSED=0) and without rounding the total (TRASHCAN_TP_ROUND=0, numerics change).
  for i in 1 2; do
    run r2-f-off-$i trashcan-r5 "$DECODE" "TRASHCAN_TP_FUSED=0"
    run r2-f-on-$i trashcan-r5 "$DECODE" ""
    run r2-f-noround-$i trashcan-r5 "$DECODE" "TRASHCAN_TP_ROUND=0"
  done ;;
final)
  # trashcan-r7: the code as committed (two-row matvec, one-row recurrence, two-key partial, unrolled merge, the
  # instrumentation), against the record's binary; and without rounding the decode totals (TRASHCAN_TP_ROUND=0).
  for i in 1 2; do
    run r2-final-rec-$i trashcan-final "$DECODE" ""
    run r2-final-new-$i trashcan-r7 "$DECODE" ""
    run r2-final-noround-$i trashcan-r7 "$DECODE" "TRASHCAN_TP_ROUND=0"
  done
  for i in 1 2; do
    run r2-final-mtp-rec-$i trashcan-final "$DECODE --mtp $MTP" ""
    run r2-final-mtp-new-$i trashcan-r7 "$DECODE --mtp $MTP" ""
    run r2-final-mtp-noround-$i trashcan-r7 "$DECODE --mtp $MTP" "TRASHCAN_TP_ROUND=0"
  done ;;
remaining)
  # The last two MTP runs of `final` (its fourth run's teardown hung on a flaky link from the driving Mac) and the gates.
  run r2-final-mtp-new-2 trashcan-r7 "$DECODE --mtp $MTP" ""
  run r2-final-mtp-noround-2 trashcan-r7 "$DECODE --mtp $MTP" "TRASHCAN_TP_ROUND=0"
  ;;
gates-noround)
  # The checks without rounding the decode totals, and the committed binary (trashcan-r8: trashcan-r7 with the
  # one-row recurrence under the qwen_gdn name) once each way.
  run r2-check-mtp-noround trashcan-r7 "qwen-check --mtp $MTP" "TRASHCAN_TP_ROUND=0"
  run r2-check-v3-noround trashcan-r7 "qwen-check --verify 3" "TRASHCAN_TP_ROUND=0"
  run r2-commit-dec trashcan-r8 "$DECODE" ""
  run r2-commit-mtp trashcan-r8 "$DECODE --mtp $MTP" ""
  ;;
profile)
  run r2-prof-dec trashcan-r2 "qwen-bench --pp 512 --tg 32 --reps 1" "" --profile
  run r2-prof-mtp trashcan-r2 "qwen-bench --pp 512 --tg 32 --reps 1 --mtp $MTP" "" --profile
  ;;
gates)
  ENV=${2:-}
  run r2-val32 trashcan-r7 "qwen-validate results/r2-val32" "TRASHCAN_TP_WIRE=f32 $ENV"
  run r2-check-mtp trashcan-r7 "qwen-check --mtp $MTP" "$ENV"
  run r2-check-v3 trashcan-r7 "qwen-check --verify 3" "$ENV"
  ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
