#!/bin/bash
set -euo pipefail
cd ~/trashcan-exp/improvements-20261007
bin=${1:-./atomic-candidates}
model=/home/fairfieldt/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf
for n in 64 129 256; do
 scripts/run.sh attention-4096-$n "$bin" qwen-attention-bench --tokens 4096 --n "$n" --iterations 5
done
scripts/run.sh attention-8192-256 "$bin" qwen-attention-bench --tokens 8192 --n 256 --iterations 5
for shape in 4352:5120:blk.0.ffn_gate.weight 5120:4352:blk.0.ffn_down.weight 1536:5120:blk.0.attn_gate.weight; do
 IFS=: read -r rows cols tensor <<< "$shape"
 for kernel in matvecq4 matvecq4p qwen_q4c3 qwen_q4c3r2; do
  batch=1; [[ "$kernel" == qwen* ]] && batch=3
  scripts/run.sh mv-$rows-$cols-$kernel "$bin" matvec-bench "$model" --rows "$rows" --cols "$cols" --tensor "$tensor" --kernel "$kernel" --batch "$batch" --iterations 100 --warmup-ms 500
 done
done
