#!/bin/bash
# Ring against full-mesh all-reduces (TRASHCAN_TP_MESH=full) on the six-cable cluster, alternating, same binary and
# sessions with every Mac's three cables open (tp8full.sh), f32 and with the f16 payload (TRASHCAN_TP_WIRE=f16).
# usage: TICKETS=... [BIN=trashcan-mesh] mesh.sh PHASE      PHASE: decode | mtp | prefill | direct | gates
# For the A/B binary (trashcan-mesh), which selected the full mesh with TRASHCAN_TP_MESH. The committed code takes the
# full mesh by itself when given all six cables: compare it with tp8.sh's four instead, as final.sh does.
set -euo pipefail
here=$(dirname "$0")
MTP=${MTP:-/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf}
BIN=${BIN:-trashcan-mesh}
run() { "$here/tp8full.sh" "$@"; }
DECODE="qwen-bench --pp 512 --tg 128 --reps 2"
FULL=TRASHCAN_TP_MESH=full
W1=TRASHCAN_TP_WIRE=f16
case ${1:?phase} in
decode)
  for i in 1 2 3; do
    run m-dec-ring-$i $BIN "$DECODE" ""
    run m-dec-full-$i $BIN "$DECODE" "$FULL"
    run m-dec-ringw-$i $BIN "$DECODE" "$W1"
    run m-dec-fullw-$i $BIN "$DECODE" "$W1 $FULL"
  done ;;
mtp)
  for i in 1 2; do
    run m-mtp-ring-$i $BIN "$DECODE --mtp $MTP" ""
    run m-mtp-full-$i $BIN "$DECODE --mtp $MTP" "$FULL"
    run m-mtp-ringw-$i $BIN "$DECODE --mtp $MTP" "$W1"
    run m-mtp-fullw-$i $BIN "$DECODE --mtp $MTP" "$W1 $FULL"
  done ;;
prefill)
  for i in 1 2; do
    run m-pp-ring-$i $BIN "qwen-bench --pp 512 --tg 16 --reps 2" ""
    run m-pp-full-$i $BIN "qwen-bench --pp 512 --tg 16 --reps 2" "$FULL"
    run m-pp2k-ring-$i $BIN "qwen-bench --pp 2048 --tg 16 --reps 1" ""
    run m-pp2k-full-$i $BIN "qwen-bench --pp 2048 --tg 16 --reps 1" "$FULL"
  done ;;
direct)
  # Decode rows through the direct path too, and verification batches through the one-step path.
  for i in 1 2; do
    run m-dir-full-$i $BIN "$DECODE --mtp $MTP" "$W1 $FULL"
    run m-dir-all-$i $BIN "$DECODE --mtp $MTP" "$W1 TRASHCAN_TP_MESH=full:0"
    run m-dir-none-$i $BIN "$DECODE --mtp $MTP" "$W1 TRASHCAN_TP_MESH=full:1000000000"
  done ;;
gates)
  # f32 full-mesh totals are the ring's bit for bit: the dumps must be identical.
  run m-val-ring $BIN "qwen-validate results/m-val-ring" ""
  run m-val-full $BIN "qwen-validate results/m-val-full" "$FULL"
  run m-val-fullw $BIN "qwen-validate results/m-val-fullw" "$W1 $FULL"
  for cfg in full fullw; do
    if [ $cfg = full ]; then env=$FULL; else env="$W1 $FULL"; fi
    run m-check-mtp-$cfg $BIN "qwen-check --mtp $MTP" "$env"
    run m-check-v3-$cfg $BIN "qwen-check --verify 3" "$env"
  done ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
