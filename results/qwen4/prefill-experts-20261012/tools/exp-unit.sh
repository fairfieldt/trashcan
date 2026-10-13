#!/bin/bash
# On alight under gpuq lease $1: unit checks of the grouped expert kernels (exp) against main.
#   exp-unit.sh ID isa|dump|bench|all [BATCHES]
# Results in ~/trashcan-exp/flashnext/results/exp-*.
set -uo pipefail
cd ~/trashcan-exp/flashnext && mkdir -p results/clocks
ID=$1 what=$2 batches=${3:-"32 64 128 256"}
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
R="--ranges 0:4,4:8 --gpus 0,1 --head-rows 0:21504,21504:43008"
run() { gpuq run $ID -- "$@"; }
if [ $what = isa ] || [ $what = all ]; then
    run env MESA_SHADER_CACHE_DISABLE=true RADV_DEBUG=shaders,shaderstats ./trashcan-exp norm-bench --iterations 1 \
        > /dev/null 2> results/exp-isa.txt
    echo "isa: $(wc -c < results/exp-isa.txt) bytes"
fi
if [ $what = dump ] || [ $what = all ]; then
    for b in main exp; do
        run env Q4X_DUMP_IDS=1 ./trashcan-$b qwen4 stage-bench $M $R --context 32768 --batch 128 \
            --tokens-file exp-p100.ids --out results/exp-dump-p100-$b > results/exp-dump-p100-$b.log 2>&1
        echo "dump p100 $b rc $?"
    done
    run env Q4X_DUMP_IDS=1 ./trashcan-exp qwen4 stage-bench $M $R --context 32768 --batch 128 \
        --tokens-file exp-long128.ids --out results/exp-dump-l128-exp > results/exp-dump-l128-exp.log 2>&1
    echo "dump l128 rc $?"
    run env Q4X_DUMP_IDS=1 ./trashcan-exp qwen4 stage-bench $M $R --context 8192 --batch 256 \
        --tokens-file exp-long256.ids --out results/exp-dump-l256-exp > results/exp-dump-l256-exp.log 2>&1
    echo "dump l256 rc $?"
fi
if [ $what = bench ] || [ $what = all ]; then
    for B in $batches; do
        ctx=32768; [ $B -gt 128 ] && ctx=8192
        for b in ${BINS:-main exp}; do
            name=exp-sb-$b-$B${TAG:-}
            run env Q4X_MARK_GU=1 ${ENVS:-} scripts/with-clocks.sh results/clocks/$name.txt ./trashcan-$b qwen4 stage-bench $M $R \
                --context $ctx --batch $B --decode 2 --start 64 --prefill 3 > results/$name.json 2> results/$name.err
            python3 -c "
import json; d=json.load(open('results/$name.json')); ph={}
for s in d['prefill']:
  for l,n,ms in s['phases']: ph[n]=ph.get(n,0)+ms
print('$name', 'wall', round(d['prefill_wall_ms'],1), ' '.join(f'{k} {v:.1f}' for k,v in ph.items()))"
            tail -1 results/$name.err
        done
    done
fi
