#!/bin/bash
# S-q2dn: G-iq4 with the routed and shared down projections of one layer per six-Mac GPU (RCO's least sensitive:
# 1 5 8 12 16 20 24 30 33 36 40 45) at Q2_0. llama-quantize copies every tensor already in its target type from G-iq4
# and quantizes those 24 tensors with ggml's Q2_0 (absmax); gguf_splice.py then puts ISTA's GSQ Q2_0 codes in place of
# the routed ones. The variant without the splice is kept as S-q2dn-absmax for comparison.
set -euo pipefail
Q=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash
BIN=/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin
SRC=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
NGRAM=/Users/fairfieldt/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf
L='(1|5|8|12|16|20|24|30|33|36|40|45)'
name=S-q2dn
DST=/Volumes/Green/Qwen/flash-next/cand/$name
mkdir -p "$DST"
start=$(date +%s)
"$BIN/llama-quantize" --allow-requantize --keep-split \
    --tensor-type "blk\.$L\.ffn_down_exps=q2_0" --tensor-type "blk\.$L\.ffn_down_shexp=q2_0" \
    --tensor-type ple_conv1d=f32 \
    --tensor-type ffn_gate_exps=iq4_xs --tensor-type ffn_up_exps=iq4_xs --tensor-type ffn_down_exps=iq4_nl \
    --tensor-type ffn_gate_shexp=iq4_xs --tensor-type ffn_up_shexp=iq4_xs --tensor-type ffn_down_shexp=iq4_nl \
    --output-tensor-type q6_k \
    "$SRC" "$DST/$name.gguf" Q8_0 ${THREADS:-16} > "$Q/quant-$name.log" 2>&1
echo "quantized in $(( $(date +%s) - start )) s" | tee -a "$Q/quant-$name.log"
rm "$DST/$name-00003-of-00006.gguf"
ln -s "$NGRAM" "$DST/$name-00003-of-00006.gguf"
grep -E "quant size" "$Q/quant-$name.log"
grep -cE "converting to q2_0" "$Q/quant-$name.log" || true
