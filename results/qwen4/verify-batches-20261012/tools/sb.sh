#!/bin/bash
# Run on a Mac under a gpuq lease, from ~/trashcan-exp/flashnext: stage-bench (layers 0-7, both GPUs, 32K context,
# profile on) for each spec NAME=BINARY:BATCH[:ENV=VAL,ENV=VAL...]; START (decode position) and DEC steps first.
# usage: sb.sh START DEC PREFILL SPEC...
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
start=$1 dec=$2 pf=$3; shift 3
mkdir -p results/clocks
for spec in "$@"; do
    name=${spec%%=*}; rest=${spec#*=}
    bin=$(echo "$rest" | cut -d: -f1); b=$(echo "$rest" | cut -d: -f2); envs=$(echo "$rest" | cut -s -d: -f3- | tr , " ")
    env TRASHCAN_Q4_PROFILE=1 $envs scripts/with-clocks.sh results/clocks/$name.txt ./$bin qwen4 stage-bench $M \
        --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b --head-rows 0:21504,21504:43008 --decode $dec \
        --start $start --prefill $pf > results/$name.json 2> results/$name.err
    echo "$name rc=$? $(tail -1 results/$name.err | cut -c1-200)"
done
