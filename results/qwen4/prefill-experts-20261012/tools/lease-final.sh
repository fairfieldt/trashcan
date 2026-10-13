#!/bin/bash
# alight lease $1: final stage-bench, layers 0-7 on both GPUs (the task's command), main and exp, B = 16..256,
# with clock logs; then the p100 stage dump of the final binary.
set -uo pipefail
cd ~/trashcan-exp/flashnext
ID=$1
TAG=-final BINS="main exp" bash exp-unit.sh $ID bench "16 32 64 128 256"
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
for b in main exp; do
    gpuq run $ID -- ./trashcan-$b qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --head-rows 0:21504,21504:43008 \
        --context 32768 --batch 128 --tokens-file exp-p100.ids --out results/exp-dump-p100-$b-final > results/exp-dump-p100-$b-final.log 2>&1
    echo "dump $b rc $?"
done
