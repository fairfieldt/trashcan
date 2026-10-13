#!/bin/bash
# Run on a Mac under a gpuq lease, from ~/trashcan-exp/flashnext: stage-bench (layers 0-7 on both GPUs) for each
# NAME=BINARY[:ENVVAR=VAL,...] and each batch, with per-dispatch profiling (PROFILE=0 to turn it off).
# usage: pf2-sb.sh "B1 B2 ..." NAME=BIN[:ENV=VAL,...] ...
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
batches=$1; shift
prof=""
[ "${PROFILE:-1}" = 1 ] && prof="TRASHCAN_Q4_PROFILE=1"
mkdir -p results/clocks
for b in $batches; do
for spec in "$@"; do
    name=${spec%%=*}; rest=${spec#*=}; bin=${rest%%:*}; envs=""
    [ "$rest" != "$bin" ] && envs=$(echo "${rest#*:}" | tr , " ")
    env $prof $envs scripts/with-clocks.sh results/clocks/pf2-$name-b$b.txt ./$bin qwen4 stage-bench $M \
        --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b --head-rows 0:21504,21504:43008 --decode 2 --start 64 \
        --prefill 5 > results/pf2-$name-b$b.json 2> results/pf2-$name-b$b.err
    echo "pf2-$name-b$b rc=$? $(tail -1 results/pf2-$name-b$b.err)"
done
done
