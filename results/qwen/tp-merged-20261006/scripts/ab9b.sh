#!/bin/bash
# 9B TP2 A/B on incite, interleaved: prefill-branch, decode-branch and merged binaries, and the merged source with
# temporary switches (trashcan-x: TRASHCAN_X_ORDER=old, TRASHCAN_X_NORM=old for batches over 8 columns).
# usage (inside the lease, performance governor): ab9b.sh
cd ~/trashcan-exp/merged
M=~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
for round in 1 2; do
    ./run.sh ab-9b-prefill-$round ./trashcan-prefill qwen-bench $M --tp --pp 512 --tg 32
    ./run.sh ab-9b-merged-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 32
    ENV="TRASHCAN_X_ORDER=old" ./run.sh ab-9b-x-order-$round ./trashcan-x qwen-bench $M --tp --pp 512 --tg 32
    ENV="TRASHCAN_X_NORM=old" ./run.sh ab-9b-x-norm-$round ./trashcan-x qwen-bench $M --tp --pp 512 --tg 32
    ENV="TRASHCAN_X_ORDER=old TRASHCAN_X_NORM=old" ./run.sh ab-9b-x-both-$round ./trashcan-x qwen-bench $M --tp --pp 512 --tg 32
    ENV="TRASHCAN_TP_ARENA=0" ./run.sh ab-9b-merged-noarena-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 32
    ./run.sh ab-9b-decode-$round ./trashcan-decode qwen-bench $M --tp --pp 512 --tg 128
    ./run.sh ab-9b-merged-tg-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 128
    ./run.sh ab-9b-decode-mtp-$round ./trashcan-decode qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
    ./run.sh ab-9b-merged-mtp-$round ./trashcan qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
done
for f in results/ab-9b-*.json; do
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], round(d['pp_tok_s'],2), round(d['tg_tok_s'],2), round(d['pp_tp_ms']['wait'],1) if d.get('pp_tp_ms') else None)" $f
done
