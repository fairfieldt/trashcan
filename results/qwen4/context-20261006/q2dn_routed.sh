#!/bin/bash
# S-q2dn-r: like S-q2dn but only the routed down projections in Q2_0 (ISTA's GSQ codes); the shared experts keep
# IQ4_NL. Separates the routed tensors' cost from the absmax Q2_0 shared experts'. Runs after q2dn_quality.sh.
set -u
Q=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash
BIN=/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin
I=/Volumes/Green/Qwen/flash-next/ista
SRC=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
NGRAM=/Users/fairfieldt/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf
L='(1|5|8|12|16|20|24|30|33|36|40|45)'
name=S-q2dn-r
DST=/Volumes/Green/Qwen/flash-next/cand/$name
until grep -q "q2dn quality done" "$Q/q2dn-quality.out" 2>/dev/null; do sleep 15; done
mkdir -p "$DST"
"$BIN/llama-quantize" --allow-requantize --keep-split \
    --tensor-type "blk\.$L\.ffn_down_exps=q2_0" --tensor-type ple_conv1d=f32 \
    --tensor-type ffn_gate_exps=iq4_xs --tensor-type ffn_up_exps=iq4_xs --tensor-type ffn_down_exps=iq4_nl \
    --tensor-type ffn_gate_shexp=iq4_xs --tensor-type ffn_up_shexp=iq4_xs --tensor-type ffn_down_shexp=iq4_nl \
    --output-tensor-type q6_k "$SRC" "$DST/$name.gguf" Q8_0 16 > "$Q/quant-$name.log" 2>&1
rm "$DST/$name-00003-of-00006.gguf"
ln -s "$NGRAM" "$DST/$name-00003-of-00006.gguf"
args=()
for l in 1 5 8 12 16 20 24 30 33 36 40 45; do args+=("blk.$l.ffn_down_exps.weight=$I/blk.$l.ffn_down_exps.weight.q2_0"); done
python3 "$Q/gguf_splice.py" "$DST/$name-00001-of-00006.gguf" "${args[@]}" > /dev/null
cd "$Q" || exit 1
start=$(date +%s)
"$BIN/llama-perplexity" -m "$DST/$name-00001-of-00006.gguf" -f corpus.txt -c 2048 -t 24 -dev none -nr \
    --kl-divergence-base kld/base-q8.kld --kl-divergence > "kld/$name.log" 2>&1
echo "$name: KLD in $(( $(date +%s) - start )) s"
grep -E "Mean PPL\(Q\)/PPL|Mean    KLD|99.0%   KLD|Same top p|RMS Δp" "kld/$name.log"
echo "q2dn routed done"
