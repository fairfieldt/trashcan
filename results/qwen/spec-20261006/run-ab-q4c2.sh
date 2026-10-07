#!/bin/bash
# Two-column verification on Qwen3.5-9B: bin/ (multivec4x2) against bin-q4cw/ (wide qwen_q4c2 everywhere),
# building-block costs in ABBA order after a warm-up, plus a k=1 correctness check on the candidate.
# Run from ~/trashcan-exp/qwen-dense inside ~/trashcan-exp/locked-run.sh.
set -euo pipefail
m=$HOME/models/qwen3.5-9b
model=$m/Qwen3.5-9B-trashcan-Q4_0.gguf
mtp=$m/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
out=results/ab-q4c2
mkdir -p $out
lat() { sudo -n python3 "$HOME/trashcan/scripts/latency-run.py" "$@"; }
bin-q4cw/trashcan qwen-check $model --mtp $mtp --draft-tokens 1 --tokens 600 --steps 128 > $out/check.json 2> $out/check.log
python3 -c "import json; r=json.load(open('$out/check.json')); s=r['speculative']; print('check', r['pass'], r['restore_exact'], len(r['column_mismatches']), r['max_column_score_error'], s['identical'])"
lat bin/trashcan qwen-bench $model --pp 64 --tg 64 > /dev/null 2>&1
for b in bin bin-q4cw bin-q4cw bin; do
    lat $b/trashcan qwen-spec-bench $model --mtp $mtp --draft-tokens 3 --tokens 600 --reps 20 > $out/spec-$b.json 2> /dev/null
    python3 -c "import json; r=json.load(open('$out/spec-$b.json')); print('$b', [(o['op'], round(o['ms'],2)) for o in r['ops'] if o['op'].startswith(('decode','verify'))])"
done
