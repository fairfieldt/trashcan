#!/bin/bash
# Build a candidate from the Q8_0 reference with llama-quantize, keeping the six-shard layout, on the external
# disk. The n-gram shard (3) differs from the source only in header key order unless the table is requantized, so
# it is replaced by a symlink to the source shard (no extra space).
# usage: quant.sh NAME [--tensor-type T=Q ...]   (everything not overridden stays as in the Q8_0 source)
set -euo pipefail
name=$1; shift
SRC=/Users/fairfieldt/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0
DST_DIR=/Volumes/Green/Qwen/flash-next/cand/$name
BIN=/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin
mkdir -p "$DST_DIR"
log=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/quant-$name.log
start=$(date +%s)
"$BIN/llama-quantize" --allow-requantize --keep-split \
    --imatrix /Users/fairfieldt/models/qwen3.8-flash-next/imatrix_unsloth.gguf_file \
    --tensor-type ple_conv1d=f32 \
    "$@" "$SRC-00001-of-00006.gguf" "$DST_DIR/$name.gguf" Q8_0 ${THREADS:-24} > "$log" 2>&1
echo "quantized in $(( $(date +%s) - start )) s" | tee -a "$log"
case " $* " in
    *per_layer_token_embd*) ;;
    *) rm "$DST_DIR/$name-00003-of-00006.gguf"
       ln -s "$SRC-00003-of-00006.gguf" "$DST_DIR/$name-00003-of-00006.gguf"
       echo "n-gram shard linked to the source" | tee -a "$log" ;;
esac
grep -E "quant size|fallback" "$log" | tee -a "$log.summary"
