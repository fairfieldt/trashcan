#!/bin/bash
# KLD against the Q8_0 model (the 39K-token corpus) for S-q2dn with ISTA's GSQ Q2_0 down projections on 12 layers,
# then with ggml's absmax Q2_0 in the same places (the saved placeholders spliced back), then GSQ restored.
set -u
Q=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash
BIN=/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin
I=/Volumes/Green/Qwen/flash-next/ista
S=/Volumes/Green/Qwen/flash-next/cand/S-q2dn/S-q2dn-00001-of-00006.gguf
cd "$Q" || exit 1
kld() {
    start=$(date +%s)
    "$BIN/llama-perplexity" -m "$S" -f corpus.txt -c 2048 -t 24 -dev none -nr \
        --kl-divergence-base kld/base-q8.kld --kl-divergence > "kld/$1.log" 2>&1
    echo "$1: KLD in $(( $(date +%s) - start )) s"
    grep -E "Mean PPL\(Q\)/PPL|Mean    KLD|99.0%   KLD|Same top p|RMS Δp" "kld/$1.log"
}
splice() {
    args=()
    for l in 1 5 8 12 16 20 24 30 33 36 40 45; do args+=("blk.$l.ffn_down_exps.weight=$1/blk.$l.ffn_down_exps.weight$2"); done
    python3 "$Q/gguf_splice.py" "$S" "${args[@]}" > /dev/null
}
kld S-q2dn-gsq
splice "$I/absmax" ""
kld S-q2dn-absmax
splice "$I" ".q2_0"
echo "q2dn quality done"
