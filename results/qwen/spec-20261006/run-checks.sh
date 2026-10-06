#!/bin/bash
# Long greedy continuation gate and a llama.cpp-comparable acceptance run (k=3, full draft head, the
# 48-token prompt, 200 tokens) for Qwen3.5-9B on incite; run from ~/trashcan-exp/qwen-dense inside
# ~/trashcan-exp/locked-run.sh.
set -euo pipefail
m=$HOME/models/qwen3.5-9b
out=results/final
mkdir -p $out
bin/trashcan qwen-check $m/Qwen3.5-9B-trashcan-Q4_0.gguf --mtp $m/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf \
    --tokens 600 --steps 256 > $out/check-256.json 2> $out/check-256.log
bin/trashcan qwen-check $m/Qwen3.5-9B-trashcan-Q4_0.gguf --mtp $m/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf \
    --draft-tokens 3 --draft-vocab 0 --tokens 48 --steps 200 > $out/check-llama-k3.json 2> $out/check-llama-k3.log
for f in check-256 check-llama-k3; do
    python3 -c "import json,sys; r=json.load(open('$out/$f.json')); s=r['speculative']; print('$f', r['pass'], r['restore_exact'], len(r['column_mismatches']), r['max_column_score_error'], s['identical'], s['stats']['proposed'], s['stats']['accepted'], round(s['stats']['acceptance'],3))"
done
