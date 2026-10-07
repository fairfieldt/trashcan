#!/bin/bash
# Decode-formulation multi-column Q4_0 kernels (qwen_q4c*) against the multivec kernels on halved Qwen3.5-9B
# shapes; every qwen_q4c column is checked bitwise against the one-column decode kernel. Run from
# ~/trashcan-exp/qwen-dense inside ~/trashcan-exp/locked-run.sh.
set -euo pipefail
model=$HOME/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
out=results/micro-cols
mkdir -p $out
run() {
    bin/trashcan matvec-bench $model --tensor $1 --batch $2 --kernel $3 --iterations 200 > $out/$1-b$2-$3.txt 2>&1 || true
    python3 -c "
import json,sys
for l in open('$out/$1-b$2-$3.txt'):
    if l.startswith('{'):
        r=json.loads(l); print('$1', r['batch'], r['kernel'], 'gpu', r['gpu'], round(r['milliseconds']*1e3,1), 'us')
    elif 'differ' in l or 'rror' in l: print('$1 $2 $3', l.strip())
"
}
for b in 2 3 4; do
    for t in blk.0.ffn_gate.weight blk.0.ffn_down.weight; do
        run $t $b qwen_q4c$b
        [ $b = 4 ] && run $t $b multivec4 || run $t $b multivec4x$b
    done
    run blk.0.attn_qkv.weight $b qwen_q4c${b}n
    run blk.0.attn_qkv.weight $b qwen_q4c$b
done
