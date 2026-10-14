#!/bin/bash
# One QSA layer (layer 3, GPU 0) at 204,800 positions with a Q8 K/V cache: the 128-token chunk that ends a prefill of
# 204,672 tokens and 16 decode steps after it, with per-dispatch GPU timestamps and sustained clocks. Run under a gpuq
# lease in ~/trashcan-exp/flashnext.
# usage: long-200k.sh BINARY NAME
cd ~/trashcan-exp/flashnext || exit 1
bin=$1 name=$2
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
mkdir -p results/clocks
scripts/with-clocks.sh results/clocks/$name.txt sudo -n env TRASHCAN_Q4_PROFILE=1 python3 scripts/latency-run.py \
    ./$bin qwen4 stage-bench $M --ranges 3:4 --gpus 0 --context 204800 --kv q8 --batch 128 --prefill 1 --decode 16 \
    --start 204672 > results/$name.json 2> results/$name.err
echo "$name rc=$? $(tail -1 results/$name.err)"
