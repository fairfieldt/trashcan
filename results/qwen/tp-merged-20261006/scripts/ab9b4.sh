#!/bin/bash
# 9B TP2 MTP decode with the qwen_matf32 split: decode branch, merged (two-queue prefill), merged with one queue
# for the micro-batches (TRASHCAN_TP_QUEUES=1), and gated forwards on the second queue (trashcan-x,
# TRASHCAN_X_GATEQ=2); then the merged prefill. usage (inside the lease, performance governor): ab9b4.sh
cd ~/trashcan-exp/merged
M=~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
for round in 1 2; do
    ./run.sh ab4-9b-decode-mtp-$round ./trashcan-decode qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
    ./run.sh ab4-9b-merged-mtp-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
    ENV="TRASHCAN_TP_QUEUES=1" ./run.sh ab4-9b-merged-q1-mtp-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
    ENV="TRASHCAN_X_GATEQ=2" ./run.sh ab4-9b-x-gateq2-mtp-$round ./trashcan-x qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
done
for round in 1 2; do
    ./run.sh ab4-9b-merged-pp-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 32
done
for f in results/ab4-9b-*.json; do
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); m=d.get('mtp') or {}; print(sys.argv[1], round(d['pp_tok_s'],2), round(d['tg_tok_s'],2), round(m.get('verify_ms_per_round',0),3), round(m.get('draft_ms_per_round',0),3))" $f
done
