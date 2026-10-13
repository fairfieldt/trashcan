#!/bin/bash
# Run on a Mac under a gpuq lease, from ~/trashcan-exp/flashnext: stage-bench dumps of layers 0-7 (the last chunk's
# last token) for each spec NAME=BINARY:BATCH[:ENV=VAL,...] on a token file.
# usage: dump.sh TOKENS SPEC...
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
ids=$1; shift
for spec in "$@"; do
    name=${spec%%=*}; rest=${spec#*=}
    bin=$(echo "$rest" | cut -d: -f1); b=$(echo "$rest" | cut -d: -f2); envs=$(echo "$rest" | cut -s -d: -f3- | tr , " ")
    rm -rf results/$name
    env $envs ./$bin qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b \
        --head-rows 0:21504,21504:43008 --tokens-file $ids --out results/$name > results/$name.log 2>&1
    echo "$name rc=$? $(ls results/$name 2>/dev/null | wc -l) files"
done
