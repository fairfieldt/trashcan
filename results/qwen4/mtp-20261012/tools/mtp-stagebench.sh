#!/bin/bash
# Verification-batch cost on one Mac (run under a gpuq lease in ~/trashcan-exp/flashnext): stage-bench of the first
# Mac's stages (layers 0-3 with the n-gram module on GPU 0, 4-7 on GPU 1), single-token decode and 2-, 3- and
# 4-token chunks, G-iq4 at 32K context.
cd ~/trashcan-exp/flashnext || exit 1
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
mkdir -p results/clocks
for b in 2 3 4; do
    scripts/with-clocks.sh results/clocks/mtp-sb-b$b.txt sudo -n python3 scripts/latency-run.py \
        ./trashcan-mtp qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b \
        --prefill 32 --decode 32 --start 64 > results/mtp-sb-b$b.json 2> results/mtp-sb-b$b.err
    echo "b=$b rc=$?"
done
for b in 2 3 4; do
    python3 -c "
import json,sys
r=json.load(open('results/mtp-sb-b$b.json'))
print('b=$b', 'decode wall %.2f ms' % r['decode_wall_ms'], 'chunk wall %.2f ms' % r['prefill_wall_ms'],
      'decode gpu', [round(sum(s['layer_ms']),2) for s in r['decode']], 'chunk gpu', [round(sum(s['layer_ms']),2) for s in r['prefill']])
"
done
