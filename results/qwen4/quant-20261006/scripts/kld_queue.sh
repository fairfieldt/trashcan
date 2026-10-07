#!/bin/bash
# Quantize each candidate from the Q8_0 reference, measure KL divergence against the Q8_0 logits on the corpus,
# keep the log and delete the candidate. usage: kld_queue.sh NAME "QUANT ARGS" [NAME "QUANT ARGS" ...]
set -u
Q=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash
BIN=/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin
cd "$Q"
while [ $# -ge 2 ]; do
    name=$1; args=$2; shift 2
    if [ -s "kld/$name.log" ] && grep -q "Mean    KLD" "kld/$name.log"; then
        echo "$name: done already"; continue
    fi
    dir=/Volumes/Green/Qwen/flash-next/cand/$name
    if [ ! -s "$dir/$name-00006-of-00006.gguf" ]; then
        # shellcheck disable=SC2086
        ./quant.sh "$name" $args || { echo "$name: quantization failed"; continue; }
    fi
    start=$(date +%s)
    "$BIN/llama-perplexity" -m "$dir/$name-00001-of-00006.gguf" -f corpus.txt -c 2048 -t 24 -dev none -nr \
        --kl-divergence-base kld/base-q8.kld --kl-divergence > "kld/$name.log" 2>&1
    echo "$name: KLD in $(( $(date +%s) - start )) s"
    grep -E "Mean PPL|Mean    KLD|99.0%   KLD|Same top p|RMS Δp" "kld/$name.log"
    du -sh "$dir" > "kld/$name.size"
    ls -la "$dir" >> "kld/$name.size"
    [ -n "${KEEP:-}" ] || rm -rf "$dir"
done
