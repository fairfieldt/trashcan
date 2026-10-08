#!/bin/bash
# The measurement plan of results/qwen/tp8-opt-20261008/README.md, one phase at a time. Every run is one tp8.sh
# invocation (about 40 s); configurations alternate within a phase so session drift falls on both sides.
# usage: TICKETS=... [MTP=path] ab.sh PHASE      PHASE: gates | decode | mtp | prefill | sweep | long | profile | base
set -euo pipefail
here=$(dirname "$0")
MTP=${MTP:-/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf}
NEW=${NEW:-trashcan-opt}     # this branch
OLD=${OLD:-trashcan-base}    # 261d453, the TP8 record
run() { "$here/tp8.sh" "$@"; }

# One switch per candidate (docs/tp8-optimization.md); H1 is on by default, so its A/B turns it off. A function rather
# than an associative array: macOS still ships bash 3.2.
sw() {
  case $1 in
    w1) echo "TRASHCAN_TP_WIRE=f16" ;;
    w1v) echo "TRASHCAN_TP_WIRE=f16:15360" ;;
    h1off) echo "TRASHCAN_TP_HOSTPASS=0" ;;
    a1) echo "TRASHCAN_QWEN_DECODE_ATTN=1" ;;
    k1) echo "TRASHCAN_QWEN_TOPK=wave" ;;
    f1) echo "TRASHCAN_QWEN_FUSED_FFN=1" ;;
    g1) echo "TRASHCAN_QWEN_FUSED_GDN=1" ;;
    m1) echo "TRASHCAN_QWEN_SHORTK=16" ;;
    *) echo "unknown candidate $1" >&2; exit 2 ;;
  esac
}
KERNELS="$(sw a1) $(sw k1) $(sw f1) $(sw g1) $(sw m1)"
ALL="$(sw w1) $KERNELS"
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"

case ${1:?phase} in
base)
  # The new binary with every switch at its default must match the record binary: same tokens, same speed.
  for i in 1 2; do
    run base-old-$i $OLD "$DECODE" ""
    run base-new-$i $NEW "$DECODE" ""
  done ;;
gates)
  # Correctness, everything on and each candidate alone (W1 and A1 change sums; the others must not).
  for cfg in all w1 a1 k1 f1 g1 m1; do
    if [ $cfg = all ]; then env=$ALL; else env=$(sw $cfg); fi
    run val-$cfg $NEW "qwen-validate results/val-$cfg" "$env"
    run check-$cfg $NEW "qwen-check" "$env"
    run check-mtp-$cfg $NEW "qwen-check --mtp $MTP" "$env"
    run check-v3-$cfg $NEW "qwen-check --verify 3" "$env"
  done
  run val-default $NEW "qwen-validate results/val-default" "" ;;
decode)
  # Each candidate against the default, alternating, then everything together.
  for cfg in w1 h1off a1 k1 f1 g1 m1; do
    for i in 1 2; do
      run dec-default-$cfg-$i $NEW "$DECODE" ""
      run dec-$cfg-$i $NEW "$DECODE" "$(sw $cfg)"
    done
  done
  for i in 1 2 3; do
    run dec-default-all-$i $NEW "$DECODE" ""
    run dec-all-$i $NEW "$DECODE" "$ALL"
  done ;;
mtp)
  for cfg in w1 w1v a1 k1 f1 g1; do
    for i in 1 2; do
      run mtp-default-$cfg-$i $NEW "$DECODE --mtp $MTP" ""
      run mtp-$cfg-$i $NEW "$DECODE --mtp $MTP" "$(sw $cfg)"
    done
  done
  for i in 1 2 3; do
    run mtp-default-all-$i $NEW "$DECODE --mtp $MTP" ""
    run mtp-all-$i $NEW "$DECODE --mtp $MTP" "$ALL"
  done ;;
prefill)
  # Only W1 touches prefill; pp2048 also crosses micro-batch tails.
  for i in 1 2; do
    run pp-default-$i $NEW "qwen-bench --pp 512 --tg 16 --reps 2" ""
    run pp-w1-$i $NEW "qwen-bench --pp 512 --tg 16 --reps 2" "$(sw w1)"
    run pp2k-default-$i $NEW "qwen-bench --pp 2048 --tg 16 --reps 2" ""
    run pp2k-w1-$i $NEW "qwen-bench --pp 2048 --tg 16 --reps 2" "$(sw w1)"
  done ;;
sweep)
  # Partition multiplier with the new attention, payload threshold, rows per wave.
  for parts in 1 2 4; do
    run sweep-parts$parts $NEW "$DECODE" "$ALL TRASHCAN_TP_PARTS=$parts"
  done
  run sweep-w1v $NEW "$DECODE" "$(sw w1v) $KERNELS"
  run sweep-m1r8 $NEW "$DECODE" "$(sw w1) $(sw a1) $(sw k1) $(sw f1) $(sw g1) TRASHCAN_QWEN_SHORTK=8" ;;
long)
  # Decode after 7.5K tokens, where the old split-K attention grows with keys per partition.
  for i in 1 2; do
    run long-default-$i $NEW "qwen-bench --pp 7680 --tg 64" ""
    run long-a1-$i $NEW "qwen-bench --pp 7680 --tg 64" "$(sw a1)"
  done ;;
profile)
  # Dispatch timelines for the record (tools/tp_profile.py, tools/tp_timeline.py).
  run prof-default $NEW "qwen-bench --pp 512 --tg 32" "" --profile
  run prof-all $NEW "qwen-bench --pp 512 --tg 32" "$ALL" --profile ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
