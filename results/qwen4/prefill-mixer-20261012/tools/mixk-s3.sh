#!/bin/bash
# Session 3 under lease $1: the final mixer path (trashcan-mixk) against c55128f: dumps, chunk sizes, crossover.
L=$1; R=~/trashcan-exp/flashnext/results/mixk
BIN=trashcan-c55128f bash $R/mixk-dump.sh $L s3 "base:X=0"
bash $R/mixk-dump.sh $L s3 "new:X=0"
BIN=trashcan-c55128f bash $R/mixk-x.sh $L s3 "65 128 256" "base:X=0"
bash $R/mixk-x.sh $L s3 "65 128 256" "new:X=0"
bash $R/mixk-x.sh $L s3 "8 12 16 24 32 48 64" "fused:X=0 gemm:Q4X_FUSED_MAX=0"
bash $R/mixk-x.sh $L s3 "16 32 64" "gemm32:Q4X_MIX_SPLITS=32:Q4X_FUSED_MAX=0"
echo "session 3 done"
