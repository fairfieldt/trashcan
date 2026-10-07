#!/bin/bash
# 9B TP2 MTP decode after a two-queue prefill (merged) against one queue per device (trashcan-x, TRASHCAN_X_QUEUES=1):
# per-segment GPU timestamps (TRASHCAN_TP_PROFILE) and the compute rings' fence counters around each run.
# usage (inside the lease, performance governor): ab9b3.sh
cd ~/trashcan-exp/merged
M=~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
rings() { for d in 0 1; do sudo -n grep -A1 "comp_" /sys/kernel/debug/dri/$d/amdgpu_fence_info | grep -o "0x[0-9a-f]*" | tr '\n' ' '; done; echo; }
for name in merged x-q1; do
    bin=./trashcan; envs="TRASHCAN_TP_PROFILE=results/prof-9b-$name.jsonl"
    if [ $name = x-q1 ]; then bin=./trashcan-x; envs="$envs TRASHCAN_X_QUEUES=1"; fi
    echo "$name rings before: $(rings)"
    ENV="$envs" ./run.sh prof-9b-$name-mtp $bin qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
    echo "$name rings after: $(rings)"
    ENV="${envs/prof-9b/prof-9b-tg}" ./run.sh prof-9b-$name-tg $bin qwen-bench $M --tp --pp 512 --tg 128
    echo "$name rings after tg: $(rings)"
done
for f in results/prof-9b-*-mtp.json results/prof-9b-*-tg.json; do
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); m=d.get('mtp') or {}; print(sys.argv[1], round(d['tg_tok_s'],2), round(m.get('verify_ms_per_round',0),3), round(d['tg_tp_ms']['wait'],3))" $f
done
