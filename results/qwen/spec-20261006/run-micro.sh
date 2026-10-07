#!/bin/bash
# Q4_0 matrix kernel cost by column count on Qwen3.5-9B shapes (the benchmark halves them as for Gemma's
# tensor-parallel shards). Run from ~/trashcan-exp/qwen-dense inside ~/trashcan-exp/locked-run.sh.
set -euo pipefail
model=$HOME/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
out=results/micro
mkdir -p $out
for t in blk.0.ffn_gate.weight blk.0.ffn_down.weight blk.0.attn_qkv.weight; do
    for b in 1 2 3 4; do
        bin/trashcan matvec-bench $model --tensor $t --batch $b --iterations 200 > $out/$t-b$b.txt 2>&1
        grep -iE "us|ms" $out/$t-b$b.txt | head -2 | sed "s/^/$t b$b: /"
    done
done
