#!/bin/bash
# On a Mac under lease $1: layers 0-7 on p100.ids in one 128-token chunk, dumped per config (name:ENV=V:ENV=V ...).
# usage: mixk-dump.sh LEASE TAG "CONFIGS"
cd ~/trashcan-exp/flashnext && mkdir -p results/mixk
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
L=$1 tag=$2
for cfg in $3; do
  name=${cfg%%:*}; envs=$(echo ${cfg#*:} | tr ':' ' ')
  gpuq run $L -- env $envs ./${BIN:-trashcan-mixk} qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch 128 \
    --head-rows 0:21504,21504:43008 --tokens-file results/mixk/p100.ids --out results/mixk/dump-$tag-$name > /dev/null 2> results/mixk/dump-$tag-$name.err
  echo dump $name $?
done
