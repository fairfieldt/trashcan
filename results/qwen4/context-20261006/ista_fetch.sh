#!/bin/bash
# Fetch the ISTA GSQ-RCO Q2_0 down projections listed in ista-fetch.tsv (HTTP range requests), one file per tensor.
set -u
Q=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash
OUT=/Volumes/Green/Qwen/flash-next/ista
REPO=https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main
mkdir -p "$OUT"
while IFS=$'\t' read -r layer variant shard tensor start end size; do
    f="$OUT/$tensor.q2_0"
    for try in 1 2 3 4 5; do
        have=$(stat -f %z "$f" 2>/dev/null || echo 0)
        [ "$have" = "$size" ] && break
        # Resume a partial file from where it stopped.
        curl -sS -L --retry 3 -r "$((start + have))-$end" "$REPO/$shard" >> "$f" || sleep 5
    done
    have=$(stat -f %z "$f" 2>/dev/null || echo 0)
    echo "$tensor ($variant build): $have of $size bytes"
done < "$Q/ista-fetch.tsv"
echo "fetch done"
