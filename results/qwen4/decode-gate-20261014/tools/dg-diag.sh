#!/bin/bash
# Decode gate diagnosis (gpuq run ID -- bash THIS, from ~/trashcan-exp/flashnext): why the second stage's gate missed
# its release in dg-correct.sh. One process on both GPUs, the model's first 6 layers, 6 greedy tokens: the gates
# sweeping the first layer's expert weights, then a buffer of their own (TRASHCAN_Q4_GATE_SWEEP=own). The startup check
# now tests each gate's own page; a gate that gives up logs what it read.
cd ~/trashcan-exp/flashnext || exit 1
. ./dg-common.sh
echo "kernel $(uname -r)"
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
B=${B:-./trashcan-dg3}
export TRASHCAN_Q4_LAYERS=6
C="--context 4096 --batch 128 --head-split 2 --cuts 3"
power
step diag-weights $B qwen4 validate $M "$R/diag-weights" --gpus 0,1 $C --greedy 6 --tokens-file p100.ids
TRASHCAN_Q4_GATE_SWEEP=own step diag-own $B qwen4 validate $M "$R/diag-own" --gpus 0,1 $C --greedy 6 --tokens-file p100.ids
for s in diag-weights diag-own; do
    echo "== $s"
    grep -E "gate|gave up" "$R/$s.err" | cut -c1-330 | head -12
done
echo "diag done; kernel log matches $(hits | wc -l) (baseline ${base:-})"
