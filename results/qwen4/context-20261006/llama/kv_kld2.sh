#!/bin/bash
# Second pass: (1) a noise floor at 64K: f16 K/V without flash attention against the flash-attention f16 base;
# (2) 204,800-token windows (llama-perplexity wants twice the window in the file, so the novel is given twice; only
# the first 204,800 tokens, all novel, are evaluated): f16 base and q8_0 K/V. f16 K + q8_0 V is skipped at this length
# because llama.cpp runs that combination's attention on the CPU here (611 s at 64K against 129 s).
BIN=/Users/fairfieldt/dev/trashcan/llama.cpp/build-mac/bin
M=/Users/fairfieldt/models/qwen3.8-flash-next/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf
D=/Volumes/Green/Qwen/flash-next/longctx
L=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/longctx
cd "$L" || exit 1
cat promessi_sposi.txt promessi_sposi.txt > promessi_sposi_x2.txt
start=$(date +%s)
"$BIN/llama-perplexity" -m "$M" -f promessi_sposi.txt -c 65536 --chunks 1 -ngl 99 -t 8 -fa off -ctk f16 -ctv f16 \
    --kl-divergence-base "$D/base-f16-65536.kld" --kl-divergence > kld-f16-f16-nofa-65536.log 2>&1
echo "K f16 V f16 without flash attention 65536: exit $? in $(( $(date +%s) - start )) s"
grep -E "Mean PPL\(Q\)/PPL|Mean    KLD|Maximum KLD|99.9%   KLD|99.0%   KLD|Same top p|RMS" kld-f16-f16-nofa-65536.log
rm -f "$D/base-f16-204800.kld"
start=$(date +%s)
"$BIN/llama-perplexity" -m "$M" -f promessi_sposi_x2.txt -c 204800 --chunks 1 -ngl 99 -t 8 -fa on -ctk f16 -ctv f16 \
    --kl-divergence-base "$D/base-f16-204800.kld" > kld-base-f16-204800.log 2>&1
echo "base f16 204800: exit $? in $(( $(date +%s) - start )) s $(grep -E 'Final' kld-base-f16-204800.log)"
start=$(date +%s)
"$BIN/llama-perplexity" -m "$M" -f promessi_sposi_x2.txt -c 204800 --chunks 1 -ngl 99 -t 8 -fa on -ctk q8_0 \
    -ctv q8_0 --kl-divergence-base "$D/base-f16-204800.kld" --kl-divergence > kld-q8_0-q8_0-204800.log 2>&1
echo "K q8_0 V q8_0 204800: exit $? in $(( $(date +%s) - start )) s"
grep -E "Mean PPL\(Q\)/PPL|Mean    KLD|Maximum KLD|99.9%   KLD|99.0%   KLD|Same top p|RMS" kld-q8_0-q8_0-204800.log
echo "kv kld2 done"
