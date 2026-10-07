# Qwen3.8-Flash-Next toward 200K context on six Macs (2026-10-06)

Three changes make room for 204,800 positions on every D700 of six Macs: Q8 K/V caches for the QSA layers, raw
indexer keys kept only for the blocks in progress, and Q2_0 down projections on one layer per GPU. This folder has
their quality checks (llama.cpp on the M3 Ultra) and the GPU runs (incite, 850/1370 MHz).

## 8-bit K/V caches in llama.cpp

llama.cpp `b8e9ce8`, Metal, flash attention, the Q8_0 GGUF, *I promessi sposi* (383,792 tokens of a real novel).
llama-perplexity scores the second half of a window; `-c 65536 --chunks 1` measures positions 32,768-65,535 against
the f16 cache of the same window (`llama/kv_kld.sh`, `llama/kv_kld2.sh`, logs in `llama/`; the base logits are on
the external disk).

| cache | mean KLD | median | p99 | same top token | PPL ratio |
|---|---|---|---|---|---|
| f16 K/V without flash attention (another kernel, same precision) | 0.0102 | 0.0037 | 0.099 | 95.48% | 1.0003 |
| f16 K, q8_0 V (attention falls back to the CPU here) | 0.0118 | 0.0041 | 0.114 | 95.20% | 1.0012 |
| q8_0 K/V | 0.0136 | 0.0049 | 0.126 | 94.84% | 1.0010 |

The text is hard for the model (PPL 7.4 at these positions; 8.0 at 100K-200K), so many positions are near-ties and
any perturbation moves the top token: merely running f16 attention through another kernel costs KLD 0.0102. Q8 K/V
adds 0.0034 to that. The 204,800-token window could not be scored: llama-perplexity keeps the logits of the whole
window (203 GB) next to the base log-probabilities (51 GB), and with the model wired by Metal the q8_0 run was killed
for memory after its forward pass; the f16 base pass alone completed (PPL 7.99, 524 s).

NEEDLE_TABLE

## Q2_0 down projections

S-q2dn is G-iq4 (the six-Mac recipe) with the down projections of layers 1, 5, 8, 12, 16, 20, 24, 30, 33, 36, 40 and
45 in Q2_0: one layer per six-Mac GPU, those ISTA's RCO allocation puts at Q2_0 first (`rco_down.py`). The routed
tensors are ISTA-DASLab's GSQ-RCO codes (`ista_fetch.sh`: 12 range requests, 2.8 GB, from the IQ3_S build where RCO
chose Q2_0 there, else the IQ3_XXS build); the shared experts' down projections are ggml's Q2_0 (absmax), because the
GPU stacks the shared expert with the routed ones. `make_s_q2dn.sh` builds the file from G-iq4 with llama-quantize
(placeholders of the right type), `splice_s_q2dn.sh` overwrites the routed tensors in place (`gguf_splice.py`).

KLD against the Q8_0 model on the 39K-token corpus (the harness of results/qwen4/quant-20261006, `q2dn_quality.sh`,
`q2dn_routed.sh`):

| GGUF | down projections of the 12 layers | mean KLD | same top token | PPL ratio |
|---|---|---|---|---|
| G-iq4 | IQ4_NL (routed and shared) | 0.0391 | 95.07% | 1.002 |
| S-q2dn-r | GSQ Q2_0 routed, IQ4_NL shared | 0.0776 | 93.11% | 1.001 |
| S-q2dn | GSQ Q2_0 routed, ggml Q2_0 shared (what the GPU runs) | 0.0984 | 92.19% | 1.010 |
| S-q2dn, absmax | ggml Q2_0 routed and shared | 0.1785 | 89.66% | 1.044 |

GSQ's codes halve the divergence of ggml's absmax Q2_0, but twelve Q2_0 layers still double G-iq4's.

## GPU runs (incite)

| file | run | result |
|---|---|---|
| `validate-giq4-kvq8.txt`, `validate-giq4-kvk16v8.txt` | G-iq4, `qwen4 validate --kv q8` / `k16v8`, layers 0-5 on the GPUs, Mac node rounding its K/V alike, against `qwen4 ref --kv` | same 10 tokens; QSA output rel 5.3e-4 / 4.6e-4 |
| `validate-sq2dn-layers-0-6.txt` | S-q2dn, 14-token prompt (per-slot expert kernels) | same tokens; Q2_0 layer outputs rel 1.1e-6 |
| `validate-sq2dn64-layers-0-6.txt` | S-q2dn, 64-token chunk (grouped expert kernels) | same tokens; Q2_0 layer outputs rel 4.0e-6 |
| `stagebench-sq2dn-0-6.json` | S-q2dn layers 0-5, `--prefill 2` | Q2_0 down: 0.29 ms per decode token (as IQ4_NL), 68 against 58 ms per 256-token chunk |
| `stagebench-sq2dn-0-4-204800-q8.json` | S-q2dn layers 0-3, `--context 204800 --kv q8 --batch 128 --head-rows 0:21504 --start 204672 --decode 16 --prefill 1` | 5780 of 5888 MiB (budget 5868); decode: DeltaNet 1.23-1.24 ms, QSA 3.65 ms (selection 1.92), head slice 1.0 ms; last chunk's QSA layer 202 ms |

The ring of raw indexer keys was checked on the 3061-token sparse prompt (G-q4_0, layers 0-5): all 111 captured
tensors bit-identical to the previous kernels.
