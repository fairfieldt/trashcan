#!/bin/bash
# Check that llama.cpp runs qwen4exp with each K/V cache type on Metal (4K context, one chunk).
BIN=/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin
M=/Users/fairfieldt/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
cd /Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/longctx || exit 1
for kv in "f16 f16" "f16 q8_0" "q8_0 q8_0"; do
    set -- $kv
    "$BIN/llama-perplexity" -m "$M" -f promessi_sposi.txt -c 4096 --chunks 1 -ngl 99 -t 8 -fa on -ctk "$1" -ctv "$2" \
        > "probe-$1-$2.log" 2>&1
    echo "K $1 V $2: exit $? $(grep -E 'Final|rror|failed|not supported' "probe-$1-$2.log" | head -2 | tr '\n' ' ')"
done
