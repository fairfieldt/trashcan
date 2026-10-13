#!/bin/bash
# On one Mac under lease $1: stage-bench of b5b79c7 (layers 0-7, both GPUs) at chunks of 16, 32, 64 and 128 tokens, to
# re-fit the per-chunk cost behind Chain::chunk_size.
cd ~/trashcan-exp/flashnext || exit 1
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
for b in 16 32 64 128; do
  gpuq run $1 -- scripts/with-clocks.sh results/clocks/refit-b$b.txt ./trashcan-b5b79c7 qwen4 stage-bench $M --ranges 0:4,4:8 \
    --gpus 0,1 --context 32768 --batch $b --head-rows 0:21504,21504:43008 --decode 2 --start 64 --prefill 5 \
    > results/refit-b$b.json 2> results/refit-b$b.err
  python3 -c "
import json; d=json.load(open('results/refit-b$b.json'))
g=[sum(s['layer_ms']) for s in d['prefill']]
print($b, 'wall', round(d['prefill_wall_ms'],1), 'per-GPU stage ms', [round(x,1) for x in g])"
done
