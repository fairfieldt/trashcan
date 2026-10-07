#!/bin/bash
# 9B TP2 on incite with the final binary: benchmarks (two runs each) and one profiled run of the unified
# TRASHCAN_TP_PROFILE switch (per-dispatch stamps; prefill kernels in qwen-bench's pp_profile, per-step lines for
# tools/tp_profile.py). usage (inside the lease, performance governor): final9b.sh
cd ~/trashcan-exp/merged
M=~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf
MTP=~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
for r in a b; do
    ./run.sh final-9b-tp2-$r ./trashcan qwen-bench $M --tp --pp 512 --tg 128
done
for r in a b; do
    ./run.sh final-9b-tp2-mtp-$r ./trashcan qwen-bench $M --tp --pp 512 --tg 128 --mtp $MTP
done
for r in a b; do
    ./run.sh final-9b-tp2-pp2048-$r ./trashcan qwen-bench $M --tp --pp 2048 --tg 32
done
ENV="TRASHCAN_TP_PROFILE=results/prof-9b-final.jsonl TRASHCAN_TP_PROFILE_DISPATCH=1" \
    ./run.sh prof-9b-final ./trashcan qwen-bench $M --tp --pp 512 --tg 32 --mtp $MTP
for f in results/final-9b-tp2-*.json results/prof-9b-final.json; do
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); m=d.get('mtp') or {}; print(sys.argv[1], round(d['pp_tok_s'],2), round(d['tg_tok_s'],2), m and round(m['acceptance'],3), d['pp_top'][0], {k: round(v,2) for k,v in (d.get('tg_tp_ms') or {}).items() if k in ('wait','add','submit')})" $f
done
