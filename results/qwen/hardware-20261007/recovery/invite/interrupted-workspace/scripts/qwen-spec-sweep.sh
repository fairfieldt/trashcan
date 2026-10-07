#!/bin/bash
# Speculative-decoding depth sweep on incite: the three Gemma MTP benchmark prompts, 128 tokens each,
# thinking off, one process per depth (0 = no MTP). TEMP sets the temperature (default 0, greedy).
# Run under a gpuq lease (`gpuq run ID -- ...`) from a workspace holding bin/trashcan.
# Output: OUT/d{depth}.out (text) and OUT/d{depth}.log (rates and MTP stats).
# usage: scripts/qwen-spec-sweep.sh MODEL MTP OUT [DEPTH ...] [-- EXTRA ARGS]
set -euo pipefail
model=$1 mtp=$2 out=$3
shift 3
depths=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do depths+=("$1"); shift; done
[ "${1:-}" = -- ] && shift
[ ${#depths[@]} -gt 0 ] || depths=(0 1 2 3)
mkdir -p "$out"
prompts="Explain how a transformer works, step by step, in English.
/reset
Explain photosynthesis to a ten-year-old in three short paragraphs.
/reset
Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example."
for d in "${depths[@]}"; do
    args=(qwen-chat "$model" --temperature "${TEMP:-0}" --max-tokens 128 --checkpoints 0 "$@")
    [ "$d" -gt 0 ] && args+=(--mtp "$mtp" --draft-tokens "$d")
    printf '%s\n' "$prompts" | sudo -n python3 "$HOME/trashcan/scripts/latency-run.py" "${BIN:-bin/trashcan}" "${args[@]}" \
        > "$out/d$d.out" 2> "$out/d$d.log"
    grep -E "decode|MTP" "$out/d$d.log" | sed "s/^/d$d /"
done
for d in "${depths[@]}"; do
    cmp -s "$out/d${depths[0]}.out" "$out/d$d.out" && echo "d$d output identical to d${depths[0]}" || echo "d$d output DIFFERS from d${depths[0]}"
done
