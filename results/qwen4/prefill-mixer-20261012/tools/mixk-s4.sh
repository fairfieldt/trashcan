#!/bin/bash
# Session 4 under lease $1: final binary (trashcan-mixk) against c55128f with clocks logged, alternating per chunk size,
# then per-dispatch timestamps of the final code (trashcan-mixk-prof: the same code with the temporary DISP output).
L=$1; R=~/trashcan-exp/flashnext/results/mixk
export CLK=1
for b in 12 16 32 64 128 256; do
  BIN=trashcan-main bash $R/mixk-x.sh $L s4 "$b" "base:X=0"
  bash $R/mixk-x.sh $L s4 "$b" "new:X=0"
done
unset CLK
BIN=trashcan-mixk-prof bash $R/mixk-x.sh $L s4p "128 256" "new:Q4_DISPATCH_PROFILE=1"
echo "session 4 done"
