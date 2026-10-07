#!/bin/bash
# Second 9B TP2 A/B on incite: qwen_matf32 split into a scalar variant (more than 4 columns) and the 16-load one
# (trashcan-x default; TRASHCAN_X_MATF32=u16 is the merged build's shader everywhere), and for MTP decode one
# queue per device (TRASHCAN_X_QUEUES=1) and undoubled work buffers (TRASHCAN_X_HALVES=1, one micro-batch).
# usage (inside the lease, performance governor): ab9b2.sh
cd ~/trashcan-exp/merged
M=~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
for round in 1 2; do
    ./run.sh ab2-9b-prefill-$round ./trashcan-prefill qwen-bench $M --tp --pp 512 --tg 32
    ./run.sh ab2-9b-merged-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 32
    ./run.sh ab2-9b-x-split-$round ./trashcan-x qwen-bench $M --tp --pp 512 --tg 32
    ENV="TRASHCAN_X_MATF32=u16" ./run.sh ab2-9b-x-u16-$round ./trashcan-x qwen-bench $M --tp --pp 512 --tg 32
    ./run.sh ab2-9b-decode-mtp-$round ./trashcan-decode qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
    ./run.sh ab2-9b-merged-mtp-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
    ENV="TRASHCAN_X_QUEUES=1" ./run.sh ab2-9b-x-q1-mtp-$round ./trashcan-x qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
    ENV="TRASHCAN_X_HALVES=1 TRASHCAN_TP_MICRO=1" ./run.sh ab2-9b-x-h1-mtp-$round ./trashcan-x qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
done
for f in results/ab2-9b-*.json; do
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); m=d.get('mtp') or {}; print(sys.argv[1], round(d['pp_tok_s'],2), round(d['tg_tok_s'],2), round(d['pp_tp_ms']['wait'],1) if d.get('pp_tp_ms') else None, round(m.get('verify_ms_per_round',0),3))" $f
done
