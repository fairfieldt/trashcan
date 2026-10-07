#!/bin/bash
# A/B of verification kernels on Qwen3.5-9B: bin/ (multivec for 2-4 columns) against bin-q4c/ (qwen_q4c*).
# Correctness check on the candidate, building-block costs and k=2 sweeps in ABBA order after a warm-up.
# Run from ~/trashcan-exp/qwen-dense inside ~/trashcan-exp/locked-run.sh.
set -euo pipefail
m=$HOME/models/qwen3.5-9b
model=$m/Qwen3.5-9B-trashcan-Q4_0.gguf
mtp=$m/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
out=results/ab-q4c
mkdir -p $out
lat() { sudo -n python3 "$HOME/trashcan/scripts/latency-run.py" "$@"; }
bin-q4c/trashcan qwen-check $model --mtp $mtp --tokens 600 --steps 128 > $out/check.json 2> $out/check.log
python3 -c "import json; r=json.load(open('$out/check.json')); s=r['speculative']; print('check', r['pass'], r['restore_exact'], len(r['column_mismatches']), r['max_column_score_error'], s['identical'])"
lat bin/trashcan qwen-bench $model --pp 64 --tg 64 > /dev/null 2>&1
for b in bin bin-q4c; do
    lat $b/trashcan qwen-spec-bench $model --mtp $mtp --draft-tokens 3 --tokens 600 --reps 20 > $out/spec-$b.json 2> /dev/null
    python3 -c "import json; r=json.load(open('$out/spec-$b.json')); print('$b', [(o['op'], round(o['ms'],1)) for o in r['ops'] if 'ms' in o])"
done
BIN=bin/trashcan scripts/qwen-spec-sweep.sh $model $mtp $out/a1 2
BIN=bin-q4c/trashcan scripts/qwen-spec-sweep.sh $model $mtp $out/b1 2
BIN=bin-q4c/trashcan scripts/qwen-spec-sweep.sh $model $mtp $out/b2 2
BIN=bin/trashcan scripts/qwen-spec-sweep.sh $model $mtp $out/a2 2
for d in b1 b2 a2; do cmp -s $out/a1/d2.out $out/$d/d2.out && echo "$d output identical to a1" || echo "$d output DIFFERS from a1"; done
