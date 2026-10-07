# Qwen3.8-Flash-Next toward 200K context on six Macs (2026-10-06)

Three changes make room for a 204,800-position context on every D700 of six Macs: Q8 K/V caches for the QSA layers,
raw indexer keys kept only for the blocks in progress, and Q2_0 down projections on one layer per GPU. This folder
has their quality checks (llama.cpp on the M3 Ultra) and the GPU runs (incite).

## 8-bit K/V caches in llama.cpp

llama.cpp `b8e9ce8`, Metal, flash attention, the Q8_0 GGUF. Text: *I promessi sposi* (383,792 tokens; given twice for
the 204,800-token window because llama-perplexity wants twice the window in the file; only the first window, all
novel, is evaluated). llama-perplexity scores the second half of a window, so `-c 65536` measures positions
32,768-65,535 and `-c 204800` positions 102,400-204,799. Each cache type is compared with the f16 cache of the same
run length (`longctx/kv_kld.sh`, `kv_kld2.sh`; logs `kld-*.log`).

KV_TABLE

NEEDLE_TABLE

## GPU (incite)

GPU_TABLE
