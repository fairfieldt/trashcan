#!/bin/bash
# Time scoring one hidden state with head slices of several sizes on one D700 (G-iq4, layer 4 as the stage body).
out=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/head-slice-bench.log
: > "$out"
for r in 0:20736 0:82944 0:124160 0:248320; do
    echo "rows $r" >> "$out"
    ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=4 incite \
        "~/trashcan-exp/locked-run.sh qwen-flash ./bin/trashcan qwen4 stage-bench /home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf --ranges 4:5 --gpus 0 --head-rows $r --decode 4 --prefill 0 --batch 16" \
        2>&1 | grep -v absent >> "$out"
done
