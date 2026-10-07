#!/bin/bash
# Multi-turn chat with and without prefix reuse on incite: one user turn per line of TURNS, greedy, the
# same chat with --checkpoints 8 and 0. Run under a gpuq lease (`gpuq run ID -- ...`) from the workspace.
# Output: OUT/c{8,0}.out (text) and .log (per-turn prefill/decode rates, reused tokens).
# usage: scripts/qwen-reuse.sh MODEL TURNS OUT [EXTRA qwen-chat ARGS...]
set -euo pipefail
model=$1 turns=$2 out=$3
shift 3
mkdir -p "$out"
for c in ${ORDER:-8 0}; do
    sudo -n python3 "$HOME/trashcan/scripts/latency-run.py" bin/trashcan qwen-chat "$model" --temperature 0 \
        --checkpoints $c "$@" < "$turns" > "$out/c$c.out" 2> "$out/c$c.log"
    grep prefill "$out/c$c.log" | sed "s/^/c$c /"
done
cmp -s "$out/c8.out" "$out/c0.out" && echo "reused and cold text identical" || echo "reused and cold text DIFFER"
