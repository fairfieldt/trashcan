#!/bin/bash
# Run on a Mac under a gpuq lease, from ~/trashcan-exp/flashnext: stage dumps of layers 0-7 (both GPUs) for a prompt,
# for each NAME=BINARY[:ENV=VAL,...] and chunk size: the last chunk's last-token activations in results/pf2-dump-*.
# usage: pf2-dump.sh IDS "B1 B2 ..." NAME=BIN[:ENV=VAL,...] ...
set -uo pipefail
cd ~/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
ids=$1 batches=$2; shift 2
for b in $batches; do
for spec in "$@"; do
    name=${spec%%=*}; rest=${spec#*=}; bin=${rest%%:*}; envs=""
    [ "$rest" != "$bin" ] && envs=$(echo "${rest#*:}" | tr , " ")
    out=results/pf2-dump-$name-$(basename $ids .ids)-b$b
    rm -rf $out
    env $envs ./$bin qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b \
        --head-rows 0:21504,21504:43008 --tokens-file $ids --out $out > $out.log 2>&1
    echo "$out rc=$? $(tail -1 $out.log)"
done
done
