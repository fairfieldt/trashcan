#!/bin/bash
# llama.cpp runs f16 K with q8_0 V far slower on this Mac (flash attention with mixed K/V types falls back to the
# CPU), so skip that case at 204,800 tokens: stop it as soon as it starts, and kv_kld.sh moves on to q8_0 K/V.
L=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/longctx
until [ -e "$L/kld-f16-q8_0-204800.log" ]; do sleep 5; done
sleep 5
pkill -f "llama-perplexity.*-c 204800 .*-ctk f16 -ctv q8_0"
echo "skipped f16/q8_0 at 204800" >> "$L/kv-kld.out"
