#!/bin/bash
# On incite under lease $1: stage-bench layers 0-7 at several chunk sizes with the threshold knobs.
cd ~/trashcan-exp/flashnext && mkdir -p results/thr
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
for b in 16 32 48 64 96 128; do
  for cfg in base:: fused:Q4X_FUSED_MAX=1000: slot:Q4X_GROUPED_MIN=1000: both:Q4X_FUSED_MAX=1000:Q4X_GROUPED_MIN=1000; do
    name=${cfg%%:*}; envs=$(echo ${cfg#*:} | tr ':' ' ')
    gpuq run $1 -- env $envs ./trashcan-x qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b \
      --head-rows 0:21504,21504:43008 --decode 2 --start 64 --prefill 3 > results/thr/$name-$b.json 2> results/thr/$name-$b.err
    python3 -c "
import json; d=json.load(open('results/thr/$name-$b.json')); ph={}
for s in d['prefill']:
  for l,n,ms in s['phases']: ph[n]=ph.get(n,0)+ms
print('$b', '$name'.ljust(6), 'wall', round(d['prefill_wall_ms'],1), 'mixer', round(ph['mixer'],1), 'experts', round(ph['experts'],1), 'deltanet', round(ph['deltanet'],1))"
  done
done
