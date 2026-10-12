#!/bin/bash
# Session 2 on invite under lease $1: dumps and timings of the v2 switches, then the fused/GEMM crossover.
L=$1; R=~/trashcan-exp/flashnext/results/mixk
bash $R/mixk-dump.sh $L v2 "old:Q4X_MIX_SPLITS=0 v1:Q4X_MIX_SPLITS=16 inj:Q4X_INJ_NORM=1 up:Q4X_MIX_UP=1 both:Q4X_INJ_NORM=1:Q4X_MIX_UP=1"
bash $R/mixk-x.sh $L v2 "128" "v1:Q4X_MIX_SPLITS=16 inj:Q4X_INJ_NORM=1 up:Q4X_MIX_UP=1 both:Q4X_INJ_NORM=1:Q4X_MIX_UP=1"
bash $R/mixk-x.sh $L v2p "128 256" "both:Q4X_INJ_NORM=1:Q4X_MIX_UP=1:Q4_DISPATCH_PROFILE=1"
bash $R/mixk-x.sh $L cross "8 12 16 24 32 48 64" "fused:Q4X_INJ_NORM=1:Q4X_MIX_UP=1 gemm:Q4X_FUSED_MAX=0:Q4X_INJ_NORM=1:Q4X_MIX_UP=1"
bash $R/mixk-x.sh $L cross "16 32 64" "gemm32:Q4X_MIX_SPLITS=32:Q4X_FUSED_MAX=0:Q4X_INJ_NORM=1:Q4X_MIX_UP=1"
