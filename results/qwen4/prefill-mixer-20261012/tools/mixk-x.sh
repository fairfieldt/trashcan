#!/bin/bash
# On a Mac under lease $1: stage-bench layers 0-7 for each batch in $3 and each config in $4 (name:ENV=V:ENV=V ...).
# usage: mixk-x.sh LEASE TAG "BATCHES" "CONFIGS"
cd ~/trashcan-exp/flashnext && mkdir -p results/mixk
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
L=$1 tag=$2
for b in $3; do
  ctx=32768; [ $b -gt 128 ] && ctx=8192
  for cfg in $4; do
    name=${cfg%%:*}; envs=$(echo ${cfg#*:} | tr ':' ' ')
    clk=; [ -n "${CLK:-}" ] && clk="scripts/with-clocks.sh results/mixk/clocks/x-$tag-$name-b$b.txt"
    gpuq run $L -- env $envs $clk ./${BIN:-trashcan-mixk} qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context $ctx --batch $b \
      --head-rows 0:21504,21504:43008 --decode 2 --start 64 --prefill 3 > results/mixk/x-$tag-$name-b$b.json 2> results/mixk/x-$tag-$name-b$b.err
    python3 -c "
import json; d=json.load(open('results/mixk/x-$tag-$name-b$b.json')); ph={}
for st in d['prefill']:
  for l,n,ms in st['phases']: ph[n]=ph.get(n,0)+ms
print('b$b', '$name'.ljust(8), '${BIN:-trashcan-mixk}', 'wall', round(d['prefill_wall_ms'],1), 'mixer', round(ph['mixer'],2), 'experts', round(ph['experts'],1), 'deltanet', round(ph['deltanet'],1), 'decode', round(d['decode_wall_ms'],2))"
  done
done
