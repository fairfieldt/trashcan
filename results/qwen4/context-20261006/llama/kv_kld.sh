#!/bin/bash
# KL divergence of 8-bit K/V caches against the f16 cache at long positions (llama.cpp, Metal, the Q8_0 GGUF).
# One window of N tokens from the start of I promessi sposi (383,792 tokens); llama-perplexity scores the second
# half of the window, so -c 65536 measures positions 32K-64K and -c 204800 positions 100K-200K.
BIN=/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin
M=/Users/fairfieldt/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
D=/Volumes/Green/Qwen/flash-next/longctx
L=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/longctx
mkdir -p "$D"
cd "$L" || exit 1
for n in 65536 204800; do
    start=$(date +%s)
    "$BIN/llama-perplexity" -m "$M" -f promessi_sposi.txt -c $n --chunks 1 -ngl 99 -t 8 -fa on -ctk f16 -ctv f16 \
        --kl-divergence-base "$D/base-f16-$n.kld" > "kld-base-f16-$n.log" 2>&1
    echo "base f16 $n: exit $? in $(( $(date +%s) - start )) s $(grep -E 'Final' kld-base-f16-$n.log)"
    for kv in "f16 q8_0" "q8_0 q8_0"; do
        set -- $kv
        start=$(date +%s)
        "$BIN/llama-perplexity" -m "$M" -f promessi_sposi.txt -c $n --chunks 1 -ngl 99 -t 8 -fa on -ctk "$1" \
            -ctv "$2" --kl-divergence-base "$D/base-f16-$n.kld" --kl-divergence > "kld-$1-$2-$n.log" 2>&1
        echo "K $1 V $2 $n: exit $? in $(( $(date +%s) - start )) s"
        grep -E "Mean PPL\(Q\)/PPL|Mean    KLD|Maximum KLD|99.9%   KLD|99.0%   KLD|Same top p|RMS" "kld-$1-$2-$n.log"
    done
done
echo "kv kld done"
