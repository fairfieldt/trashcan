#!/bin/bash
# Run on a Mac under a gpuq lease, from ~/trashcan-exp/flashnext: stage-bench runs for each NAME=BINARY[:ENVVAR=VAL] given.
# usage: dec-sb.sh START DECODE NAME=BIN[:ENVVAR=VAL] ...
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
start=$1 dec=$2; shift 2
mkdir -p results/clocks
for spec in "$@"; do
    name=${spec%%=*}; rest=${spec#*=}; bin=${rest%%:*}; envs=""
    [ "$rest" != "$bin" ] && envs=$(echo "${rest#*:}" | tr , " ")
    env $envs scripts/with-clocks.sh results/clocks/$name.txt ./$bin qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 \
        --context 32768 --batch 128 --head-rows 0:21504,21504:43008 --decode $dec --start $start --prefill 1 \
        > results/$name.json 2> results/$name.err
    echo "$name rc=$? $(tail -1 results/$name.err)"
done
