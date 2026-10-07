#!/bin/bash
# matvec-bench sweep over tensor-parallel shard shapes. usage: mvsweep.sh OUT.jsonl
out=$1; cd ~/trashcan-exp/decode
M27=~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf
M9=~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
run() { ./trashcan matvec-bench $1 --tensor $2 --rows $3 --cols $4 --kernel $5 --iterations 200 --warmup-ms 300 2>/dev/null >> $out || echo "fail $2 $3 $4 $5" >&2; }
for k in matvecq4 matvecq4r8; do
  run $M27 blk.0.ffn_gate.weight 4352 5120 $k
  run $M27 blk.0.ffn_gate.weight 17408 5120 $k
  run $M27 blk.0.ffn_down.weight 5120 4352 $k
  run $M27 blk.0.attn_qkv.weight 2560 5120 $k
  run $M27 blk.0.attn_gate.weight 1536 5120 $k
  run $M27 blk.0.ssm_out.weight 5120 1536 $k
  run $M27 blk.3.attn_q.weight 3072 5120 $k
  run $M27 blk.3.attn_k.weight 256 5120 $k
  run $M9 blk.0.ffn_gate.weight 6144 4096 $k
  run $M9 blk.0.ffn_down.weight 4096 6144 $k
  run $M9 blk.0.attn_qkv.weight 4096 4096 $k
  run $M9 blk.0.ssm_out.weight 4096 2048 $k
done
run $M27 blk.0.ffn_gate.weight 4352 5120 matvecq4narrow
run $M27 blk.0.attn_qkv.weight 2560 5120 matvecq4narrow
