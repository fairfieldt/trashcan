#!/bin/bash
# Final Qwen3.5-9B speculative-decoding measurements on incite, run from ~/trashcan-exp/qwen-dense inside
# ~/trashcan-exp/locked-run.sh: a warm-up, plain vs two-draft greedy decoding in both orders, a temperature
# 0.7 depth sweep, the building-block costs and prefill with and without the MTP layer.
set -euo pipefail
m=$HOME/models/qwen3.5-9b
model=$m/Qwen3.5-9B-trashcan-Q4_0.gguf
mtp=$m/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
out=${OUT:-results/final}
mkdir -p $out
lat() { sudo -n python3 "$HOME/trashcan/scripts/latency-run.py" bin/trashcan "$@"; }
lat qwen-bench $model --pp 64 --tg 64 > /dev/null 2>&1
scripts/qwen-spec-sweep.sh $model $mtp $out/greedy-a 0 2
scripts/qwen-spec-sweep.sh $model $mtp $out/greedy-b 2 0
TEMP=0.7 scripts/qwen-spec-sweep.sh $model $mtp $out/t07 0 1 2 3
lat qwen-spec-bench $model --mtp $mtp --draft-tokens 3 --tokens 600 --reps 20 > $out/spec-bench.json 2> $out/spec-bench.log
for k in plain mtp; do
    args=()
    [ $k = mtp ] && args=(--mtp $mtp)
    lat qwen-bench $model --pp 512 --tg 64 "${args[@]}" > $out/bench-$k.json 2> $out/bench-$k.log
    cat $out/bench-$k.json | cut -c1-200
done
