#!/bin/bash
# Start the TP4 node (machine 1) on this Mac in the background, tagged $1; with $1 starting "gpud" it runs on gpud
# (libgpud_drm.so). Clocks are sampled into clocks-$1-node.txt (drminfo clockmon on gpud, clockmon.sh on amdgpu).
cd "$(dirname "$0")"
tag=$1
M=~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
pkill -f "trashcan qwen-tp-node"; pkill -f "clockmon"; sleep 1
if [[ $tag == gpud* ]]; then
  LD_PRELOAD=$PWD/libgpud_drm.so nohup ./drminfo clockmon clocks-$tag-node.txt > /dev/null 2>&1 &
  LD_PRELOAD=$PWD/libgpud_drm.so nohup ./trashcan qwen-tp-node $M --mtp $MTP > node-$tag.log 2>&1 &
else
  nohup scripts/clockmon.sh clocks-$tag-node.txt > /dev/null 2>&1 &
  nohup ./trashcan qwen-tp-node $M --mtp $MTP > node-$tag.log 2>&1 &
fi
for i in $(seq 1 240); do grep -qiE "listening|lanes|waiting|error" node-$tag.log && break; sleep 1; done
tail -3 node-$tag.log
