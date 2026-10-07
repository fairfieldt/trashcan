#!/bin/bash
# Put ISTA's GSQ Q2_0 down projections into S-q2dn in place, saving the absmax placeholders first.
set -euo pipefail
Q=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash
I=/Volumes/Green/Qwen/flash-next/ista
args=()
for l in 1 5 8 12 16 20 24 30 33 36 40 45; do
    args+=("blk.$l.ffn_down_exps.weight=$I/blk.$l.ffn_down_exps.weight.q2_0")
done
python3 "$Q/gguf_splice.py" --save "$I/absmax" /Volumes/Green/Qwen/flash-next/cand/S-q2dn/S-q2dn-00001-of-00006.gguf \
    "${args[@]}"
