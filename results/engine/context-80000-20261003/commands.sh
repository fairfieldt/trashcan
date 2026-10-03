#!/bin/bash
# Run from ~/trashcan on incite. Uses the unchanged model kernels from 5925623.
set -euo pipefail
exec 9>"$HOME/trashcan-exp/.lock"
flock 9
R=results/engine/context-80000-20261003
M=$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
scripts/clockmon.sh "$R/clocks.txt" & monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
for pp in 4096 80000; do
    sudo -n python3 scripts/latency-run.py target/release/trashcan bench "$M" \
        --pp "$pp" --tg 128 --batch 512 --progress --profile "$R/ctx$pp.tsv" \
        >"$R/ctx$pp.json" 2>"$R/ctx$pp.log"
done
# The actual run also saved source/binary hashes, start/end times and pre/post reset logs.
