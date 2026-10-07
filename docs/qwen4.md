# Qwen3.8-Flash-Next (`qwen4exp`) on a chain of Mac Pros

Qwen3.8-Flash-Next is a 177B-parameter sparse MoE: 121B of routed experts (about 6B used per token), a 51B hashed
n-gram embedding read at one layer, and 4.6B of dense weights. This document covers the reference, the n-gram
table placement, quantization and VRAM for six and four Mac Pros, the GPU stages and projected chain speed. Code:
`engine/src/qwen4/`, shaders `engine/shaders/qwen4_*.comp`. The dense Qwen models are in [qwen.md](qwen.md).

Status: the CPU reference matches llama.cpp; GPU stages for every layer type run on incite's D700s and match the
reference with the six-Mac recipe's formats; a full six-Mac GPU stage (4 layers, 32K context) loads and runs on one
D700. Not done yet: spreading the head over GPUs (required for six Macs), a chain of real Mac Pros, MTP and a
chat command (see [What is left](#what-is-left)).

| | |
|---|---|
| layers | 48: 36 Gated DeltaNet, 12 Qwen Sparse Attention (every fourth), plus one MTP layer (not used yet) |
| residual | 4 hyper-connection streams of 2560 (`hc`), mixed before and written after every sublayer; no norms |
| DeltaNet | 16 key heads, 48 value heads of 128, sigmoid output gate |
| attention | 24 Q / 2 KV heads of 256, RoPE on 64 dims (base 1e7); indexer of 4 heads x 128 over 4-token blocks |
| MoE | 512 experts of width 640, top 10, plus a gated shared expert of 640 in every layer |
| n-gram | 16 rows of 160 per token from a 320M-row table, added at layer 1 |
| vocabulary | 248,320 (the qwen35 tokenizer), untied head |

## Layer math (`engine/src/qwen4/cpu.rs` is the executable reference)

Every stream starts as the token embedding. A mixer reads the residual `x` (4 x 2560 per token):

- `xn_s = RMS(x_s) * gamma_s` per stream; `lo = silu(W_down xn / 4)` (10240 -> 320);
  `g = sigmoid(W_up lo)` (320 -> 10240); the sublayer input is `u = mean_s(g_s * xn_s)`;
- the write weights are `w_s = 2 sigmoid(W_inject xn / 4)`, and after the sublayer `x_s += w_s * y`.

A layer is `[n-gram module at layer 1] -> mixer -> DeltaNet or QSA -> combine -> mixer -> MoE -> combine`; the
final mixer (no write weights) produces the head input. The GGUF stores the gammas already folded to `1 + w`.

- **N-gram module (layer 1).** Row ids come from hashing the token and its one and two predecessors with fixed
  u64 multipliers (`qwen4exp.ple.*`): 8 bigram and 8 trigram heads, each `hash % size_h + offset_h`. An
  end-of-sequence token cuts the context at and before it; a missing predecessor reads as end-of-sequence.
  The 16 gathered rows of 160 form `e` (2560). Then `k = RMS_s(W_key e) * gk`, `v = W_value e`, and per stream
  `a = (k_s . RMS(x_s) gq_s) / sqrt(2560)`, `gate_s = sigmoid(sign(a) sqrt(max(|a|, 1e-6)))`, `gv_s = gate_s v`.
  A depthwise causal convolution of `RMS(gv_s) * gc` with kernel 4 dilated by 3 (taps t-9, t-6, t-3, t) and
  SiLU gives `c`; `x += gv + c`. The convolution keeps 9 x 10240 values of history.
- **Gated DeltaNet:** as in qwen35 (value head `h` reads key head `h % 16`), except the gated norm multiplies by
  `sigmoid(z)` instead of `silu(z)`.
- **QSA:** gated attention as in qwen35 (24 Q, 2 KV heads, per-head Q/K norms, NeoX RoPE on 64 dims, scale 1/16,
  sigmoid output gate). A token at position `p` sees `nb = (p + 1) / 4` complete blocks. While `nb <= 512`
  (`p + 1 <= 2051`) it attends to every position. Beyond that it attends to the 512 blocks with the highest
  indexer score plus the incomplete tail ending at `p`. The score of block `b` is
  `sum_h relu(q_h . kb) / sqrt(128)`, with `q_h` the token's 4 indexer heads (RMS-normed, rotated at `p`) and `kb`
  the mean of the block's 4 raw indexer keys, RMS-normed and rotated to the block's first position `4b`. Ties go
  to the earlier block (llama.cpp's order there is arbitrary).
- **MoE:** softmax over 512 router logits, the 10 largest renormalized (sum clamped at 2^-14), SwiGLU experts;
  the shared expert's output is scaled by `sigmoid(w . u)`.

## Reference validation

`trashcan qwen4 ref` runs the CPU reference and writes `tools/dump-activations`-format dumps; `trashcan qwen4
layer-check` runs one layer at a time on llama.cpp's own inputs. Against llama.cpp `b8e9ce8` (CPU) on the Q8_0
GGUF ([results/qwen4/reference-20261006](../results/qwen4/reference-20261006/README.md)):

| check | result |
|---|---|
| 14-token prompt, greedy | identical 10 tokens |
| n-gram rows, initial residual | identical |
| logits (f32 reference) | rel 8.8e-2, cos 0.9963, top-10 overlap 10/10 |
| teacher-forced layers, ggml rounding emulated (`--emulate-ggml`) | update rel ~5e-7 in 10 of 48 layers; the rest differ only through Q8 rounding boundaries or near-tied experts |
| 3061-token prompt (765 blocks, sparse), greedy | identical 10 tokens; logits cos 0.9984 (emulated) |
| QSA layers 3, 23, 47 on that prompt, teacher-forced | the same 512 blocks as llama.cpp, scores within 7e-5, attention output within 1.3e-3 |

llama.cpp's CPU path quantizes activations to Q8_0 before every Q8_0 product (2-3% noise per expert output);
`--emulate-ggml` reproduces that rounding, the BF16 rounding of BF16 products, f16 K/V and indexer caches, and
the f32 BLAS products of batches of 32 or more. The tokenizer is qwen35's: vocabulary, merges and pre-tokenizer
are byte-identical, and the 37 qwen tokenizer fixtures pass. The Flash-Next template renders the reference prompt
to the same 14 ids as qwen35's.

The chain (`trashcan qwen4 validate`, `qwen4 node`) reproduces the single-shot reference exactly on the CPU path:
chunked prefill (batches of 4 and 5) and a two-process TCP chain give identical dumps and tokens.

## N-gram table placement

The table is read only at layer 1, so it stays with the driver, next to the token embedding: the driver hashes
each token with its two predecessors, gathers 16 rows of 160 and sends `n x 2560` floats to the stage owning
layer 1, which must be on the driver (`net::Chain` enforces it). `--ngram` picks the source: `map` (the GGUF
mapping), `ram-q4`/`ram-q8` (a resident copy, Q8_0 rows requantized to Q4_0 at load), `disk` (pread through the
page cache) or `direct` (O_DIRECT 4 KiB pages). Measured on incite's SSD (Intel 660p, PCIe 2.0 x4), 16 readers
([results/qwen4/ngram-20261006](../results/qwen4/ngram-20261006/README.md)):

| storage | RAM used | one token's rows (p50) | prefill |
|---|---|---|---|
| RAM, Q4_0 rows | 26.8 GiB | 0.07 ms | 0.8-0.95 M tok/s |
| SSD through the page cache | page cache only | 0.40 ms | 2.8k tok/s random tokens, 10k tok/s on real text |
| SSD, O_DIRECT | none | 0.72-0.74 ms | 2.7k tok/s |

**Recommendation: keep the table on the driver's SSD and read rows through the page cache.** A decode step costs
0.4 ms against tens of milliseconds of GPU work (and the reads can start as soon as the token is sampled, while
the GPU runs layer 0), and 2.7k-10k tokens per second is several times the measured prefill throughput of a GPU
stage. Common n-grams stay cached for free, and no RAM is committed: a resident Q4_0 copy takes 27 GiB of the
driver's 64 GB without a measurable end-to-end gain and costs quality (KLD 0.023, see below), and a resident Q8_0
copy (51 GiB) does not fit safely. On disk the rows keep the release's precision (Q8_0 here, 170 bytes; BF16 rows
would cost the same 4 KiB read).

## Quantization

Candidates are made with `llama-quantize --imatrix` (unsloth's imatrix) from the Q8_0 release, changing one group
of tensors at a time; everything else stays Q8_0. Quality is llama.cpp's KL divergence of the candidate's
next-token distribution from the Q8_0 model's over 39K tokens of mixed text (19 chunks of 2048, CPU), with the
share of positions where both pick the same top token
([results/qwen4/quant-20261006](../results/qwen4/quant-20261006/README.md), recipes included):

| candidate | changed from Q8_0 | mean KLD | same top token | PPL ratio |
|---|---|---|---|---|
| E-iq4xs | experts IQ4_XS gate/up, IQ4_NL down | 0.0357 | 95.3% | 1.003 |
| E-q4_0 | experts Q4_0 | 0.0402 | 94.9% | 1.003 |
| E-mxfp4 | experts MXFP4 | 0.0656 | 93.7% | 0.999 |
| E-iq2xxs | experts IQ2_XXS gate/up, IQ4_NL down | 0.2134 | 88.6% | 1.077 |
| N-q4_0 | every dense matrix Q4_0 (mixers, DeltaNet, attention, shared expert, n-gram projections) | 0.1961 | 88.7% | 1.065 |
| T-q4_0 | n-gram table Q4_0 | 0.0229 | 96.3% | 1.002 |

- **Dense matrices are the sensitive part.** 4.6B dense parameters at Q4_0 cost five times the divergence of
  121B expert parameters at Q4_0, and the hyper-connection mixers alone cost 0.119. The recipe keeps dense
  matrices in Q8_0, which the GPU stages read with `qwen4_q8_mv`, `qwen4_q8_gemm` and fused Q8_0 mixers: about
  34 MB more per DeltaNet layer than Q4_0 and 0.25 ms more per decode step of a layer.
- **IQ4_XS gate/up with IQ4_NL down is the best 4-bit expert format** and also the smallest (the down rows are
  640 wide, which rules out the 256-element formats there). MXFP4 is clearly worse on this model.
- **2-bit experts cost as much as 4-bit dense matrices** (KLD 0.21, the same top token 88.6% of the time against
  95.3%). Four Macs at 2 bits is a large quality step, not a small one.
- **The n-gram table should keep the release's precision.** A Q4_0 table costs KLD 0.023, which is why the
  table stays on the driver's SSD at Q8_0 rather than in RAM at Q4_0.

## VRAM

`tools/qwen4_budget.py` models what `gpu.rs` allocates per stage (the mixers' matrices, including the fused copy
used for decode, the f32 router, DeltaNet or QSA matrices and state, the stacked experts, K/V and indexer caches,
batch scratch, the head) and matches all 14 stages measured on incite within 0.2 MiB (`--check`). Stages print
their device-local heap usage and refuse to run past the driver's budget, because RADV silently places the excess
in system memory (a 6480 MiB stage ran its head 18x slower that way).

Six Macs give each D700 4 layers. At 32K context and 256-token prefill chunks, with IQ4_XS/IQ4_NL experts and
Q8_0 dense matrices:

| | MiB |
|---|---|
| layers 0-3 on one D700 (n-gram module, 3 DeltaNet, 1 sparse QSA), **measured** | 5843 of 5888 (budget 5869) |
| the head (Q6_K repacked to 10 bits per weight, final mixer, logits) | 765 |
| head spread over all 12 GPUs | 64 per GPU, 5907 per GPU: 19 over |
| with four savings not implemented yet (`--lean`: prefill reuses the fused mixer matrix, Q8_0 router and indexer projections, a raw indexer-key ring) | 5845 per GPU: fits by 43 |
| the same with 128-token prefill chunks | 5766 per GPU: fits by 122 |

Q4_0 experts do not fit six Macs (6043 MiB before any head share). Five Macs put 5 layers on most GPUs, where even
IQ3_XXS gate/up with IQ4_NL down is 106 MiB over (`--lean`, 128-token chunks). Four Macs give 6 layers per GPU
(`qwen4_budget.py`):

| four Macs, 6 layers per D700 | per GPU (32K, head over 8 GPUs, `--lean`, 128-token chunks) | quality |
|---|---|---|
| IQ4_XS/IQ4_NL experts | 8583 MiB: does not fit | KLD 0.036 |
| IQ2_XXS gate/up, MXFP4 down | 5802 MiB: fits by 86 | KLD 0.21 or worse; no GPU kernels for either format yet |
| IQ4 experts in host RAM, ~63% of each layer's experts cached in VRAM | fits by construction | KLD 0.036 |

**Recommendation.** Six Macs: IQ4_XS gate/up and IQ4_NL down for routed and shared experts, Q8_0 dense matrices,
Q6_K head spread over the GPUs, n-gram table at Q8_0 on the driver's SSD (the `G-iq4` recipe: 116 GiB, 51 GiB of
it the n-gram table). It needs the head spread and the `--lean` savings before it runs at 32K context. Four Macs:
the same weights, with each Mac's experts (15.3 GiB) in host RAM and a static VRAM cache of each layer's most used
~63%, which covers about 85% of routed activations by the imatrix counts (`expert_popularity.py` in the quant
results); misses (about 3.5 MB per layer per token) are uploaded at the measured 6.5 GB/s. The 2-bit in-VRAM
alternative fits only with the same savings and loses much more quality.

## GPU stages (`engine/src/qwen4/gpu.rs`)

A stage owns a contiguous layer range on one D700, records one command buffer per layer per batch size and
replays it with a new position, as the qwen35 stages do. The residual (4 x 2560 f32 per token) enters and
leaves through host buffers; that is also the payload between machines (40 KiB per token). Every mixer applies
the previous sublayer's write (`x_s += w_s y`) as it reads the residual, so a layer ends with a pending combine
that the next mixer, the head or the stage output resolves.

| part | kernels |
|---|---|
| mixer, up to 8 tokens | `qwen4_mix_down` (pending combine, the four stream norms folded into the 320 bottleneck and 4 write rows), `qwen4_mix_up` (read gates, stream mean, stores the combined residual) |
| mixer, prefill | `qwen4_hc_norm`, the dense matrix kernels, `qwen4_act`, `qwen4_hc_mix` |
| dense matrices | Q8_0: `qwen4_q8_mv` (1-8 tokens), `qwen4_q8_gemm` (64 x 64 tiles), f16 scales, 8.5 bpw; Q4_0: the Gemma kernels (f32 scales, 5 bpw) |
| n-gram module | key/value projections, `qwen4_ple_gate`, `qwen4_ple_conv` (dilated convolution, 9-step history) |
| DeltaNet | qwen35's `qwen_conv` and `qwen_gdn`, `qwen4_gated_norm` (sigmoid gate) |
| QSA, up to 2051 tokens | qwen35's `qwen_qk` and the split/unsplit attention kernels (dense is exact there) |
| QSA, longer contexts | `qwen4_idx_store`/`qwen4_idx_pool` (raw and pooled block keys), `qwen_qk` on the indexer queries, `qwen4_idx_score` (block scores, one wave per block for 8 tokens), `qwen4_idx_select` (exact radix top-512, ties to the earlier block), `qwen4_attention_sparse[_full]` |
| MoE | router (f32, with the shared gate as row 512), `qwen4_topk`; decode: `qwen4_moe_gu`, `qwen4_moe_down` per (token, slot); prefill (16+ tokens): `qwen4_moe_route` groups slots by expert, `qwen4_moe_{gu,down}_grouped` decode each expert's rows once per 4 slots, `qwen4_moe_sum` |
| experts | IQ4_XS (4.25 bpw), IQ4_NL or Q4_0 (f16 scale per 32, 4.5 bpw), at their GGUF size; the shared expert is stacked as expert 512 and must share the routed format |
| head | final mixer, qwen35's Q6 `headvec` (int8 with f32 scales per 16), top-64 |

`qwen4 stage-bench` times stages alone (prefill chunks, then decode steps, back to back so the clocks stay up) with
GPU timestamps around every layer and phase; `--start` prefills that many tokens first and also reports the last
of those chunks.

### Validation on the D700s

Layers 0-5 (n-gram module, DeltaNet, QSA and MoE) ran on incite's two GPUs with layers 6-47 and the head on a CPU
node on the M3 Ultra (`--cuts 3,6 --nodes MAC:7071`), compared with the CPU reference on the same GGUF
([results/qwen4/gpu-20261006](../results/qwen4/gpu-20261006/README.md)):

| GGUF | 14-token prompt, 10 greedy tokens | logits | DeltaNet, n-gram and MoE layers | QSA layer (f16 K/V) |
|---|---|---|---|---|
| all Q4_0 (fused and unfused mixers) | identical | rel 4.3e-5 | rel ~1e-6 | rel 4.4e-4 |
| Q8_0 dense, Q4_0 experts | identical | rel 2.5e-5 | rel ~1e-6 | rel 5.1e-4 |
| Q8_0 dense, IQ4_XS/IQ4_NL experts | identical | rel 1.7e-5 | rel 5e-7 to 1.1e-6 | rel 4.4e-4 |

On a 3061-token prompt in 256-token chunks (sparse QSA, grouped experts), the last chunk's QSA output is within
rel 6.6e-4 of the CPU reference and the indexer queries within 2.4e-5; moving block scoring into
`qwen4_idx_score` left all 111 captured tensors bit-identical.

### Measured stage times (incite, 850/1370 MHz)

Six-Mac recipe (Q8_0 dense, IQ4_XS/IQ4_NL experts), per layer:

| phase | decode, 64 tokens of context | decode at 32K | prefill, 256-token chunk at 0 | chunk at 32K |
|---|---|---|---|---|
| mixer (each of two) | 0.14-0.17 ms | 0.17 ms | 4.3-5.7 ms | 4.3-5.6 ms |
| DeltaNet | 0.49-0.50 ms | 0.49-0.50 ms | 19.2-19.4 ms | 19.3-19.4 ms |
| QSA | 0.43 ms | 1.40 ms (projections 0.22, block selection 0.61, attention 0.57) | 16.1 ms | 119.8 ms (selection 35.9, attention 75.9) |
| router and top-k | 0.10 ms | 0.10 ms | 3.7-3.9 ms | 3.8 ms |
| experts (10 routed + shared) | 0.29-0.30 ms | 0.29 ms | 57.4-58.2 ms | 57.4 ms |
| n-gram module (layer 1) | 0.43 ms | | 8.4 ms | |
| DeltaNet layer | 1.23-1.30 ms | 1.23-1.24 ms | 89-96 ms | 89-90 ms |
| QSA layer | 1.16 ms | 2.20 ms | 88-99 ms | 196 ms |
| head (final mixer, 248,320 rows, top-64) | 4.6 ms | | | |

Layer 1 adds the n-gram module to its DeltaNet layer (0.43 ms per decode token, 8.4 ms per chunk).

Formats compared on layers 0-5 (decode GPU time per DeltaNet layer; experts per decode token / per prefill chunk):

| GGUF | DeltaNet layer | experts | six layers, wall per decode step |
|---|---|---|---|
| all Q4_0, unfused mixers | 1.31-1.38 ms | 0.23 / 52.5 ms | 9.05 ms |
| all Q4_0 | 0.93-0.99 ms | 0.23 / 52.5 ms | 6.65 ms |
| Q8_0 dense, Q4_0 experts | 1.18-1.26 ms | 0.25 / 52.7 ms | 8.29 ms |
| Q8_0 dense, IQ4 experts | 1.23-1.30 ms | 0.29 / 57.7 ms | 8.62 ms |

Fusing the mixers into two dispatches each took decode from 9.05 to 6.65 ms for six layers: the mixers were half
of a decode layer and almost none of its bytes. Moving block scoring out of the single-workgroup select took the
QSA layer at 32K from 13.25 to 2.16 ms per decode token and the 256-token chunk there from 272 to 190 ms (G-q8d). In
prefill the grouped experts take 60% of a layer (0.5 TFLOP/s); 2 or 8 slots per pass instead of 4 were slower.
A decode layer reads about 110 MB (61 MB of DeltaNet matrices, 14 MB of mixers, 5 MB router, 29 MB of experts) in
1.23 ms, 34% of the D700's 263 GB/s.

## Projected chain performance

No chain longer than incite plus the Mac CPU node has run, so these are projections from the per-layer
measurements above: decode 1.24 ms per DeltaNet layer, 1.16 ms per QSA layer (2.20 ms at 32K), 0.43 ms for the
n-gram module, 4.6 ms for the head, 0.39 ms of host work per GPU stage (two stages: 8.62 ms wall against 7.84 ms of
GPU time), and per machine hop 40 KiB over Thunderbolt 2 IP (8.8 Gb/s, 128 us round trip:
[results/kernel/tb-loopback-20261006](../results/kernel/tb-loopback-20261006/README.md)), about 0.1 ms.

| | six Macs, IQ4 experts in VRAM | four Macs, IQ4 experts in host RAM |
|---|---|---|
| decode, short context | 36 x 1.24 + 12 x 1.16 + 0.43 + 4.6 + 12 stages x 0.39 + 5 hops = 69 ms: **14.5 tok/s** | the same GPU work with 8 stages and 3 hops (67 ms) + 48 x 0.53 ms of expert uploads = 93 ms: **10.8 tok/s** |
| decode at 32K | 81 ms: **12.3 tok/s** | 105 ms: **9.5 tok/s** |
| prefill (chunks pipelined across Macs, a Mac's two GPUs in turn) | 0.76 s per chunk per Mac: **~335 tok/s**; a 2K prompt in ~10 s | 1.2-2.0 s per chunk per Mac (expert uploads overlapped or not): **125-210 tok/s** |
| weight-bandwidth bound for decode | 6.0 GB per token at 263 GB/s: 23 ms, 44 tok/s | |

## What is left

1. **Spread the head over GPUs** by vocabulary rows (each machine on the return path, or the last machine's two
   GPUs), and store it compactly (Q6_K is 6.6 bits, the current repack 10). Six Macs need this.
2. **The `--lean` savings** (about 60 MiB per GPU) and a 128-token prefill chunk, for margin at 32K.
3. **A real chain**: only one Mac Pro (incite) was available, so multi-Mac decode and prefill are projections.
4. **Prefill**: an LDS-tiled grouped expert GEMM (58 ms of a 94 ms layer), chunked DeltaNet (19 ms), grouping
   the 12 query heads that share a KV head in sparse attention (76 ms per chunk at 32K), the f32 router (3.8 ms),
   and the two GPUs of a Mac working on different chunks.
5. **Decode**: fewer dispatches per layer (34% of bandwidth now) and a multi-workgroup radix select (0.6 ms at 32K).
6. **Four Macs**: the host-RAM expert cache (or 2-bit kernels, if that quality is acceptable).
7. **MTP** (a 49th layer with its own experts) and the rollback snapshots speculative decoding needs; a chat
   command and the Responses server for `qwen4exp` (today the CLI has `ref`, `validate`, `bench`, `node`).

## Commands

```sh
# reference and validation (Mac)
trashcan qwen4 ref MODEL OUT [--tokens-file IDS] [--emulate-ggml] [--greedy N]
trashcan qwen4 layer-check MODEL LLAMA_DUMP [--emulate-ggml]
# chain: every machine after the first, then the driver
trashcan qwen4 node MODEL --listen 0.0.0.0:7071 [--cpu | --gpus 0,1]
trashcan qwen4 validate MODEL OUT --gpus 0,1 --cuts 3,6 --nodes MAC:7071 --greedy 10
trashcan qwen4 bench MODEL --gpus 0,1 --nodes ... --pp 256 --tg 32
# GPU stages alone, and the n-gram table
trashcan qwen4 stage-bench MODEL --ranges 0:4 --gpus 0 --context 32768 --start 32512 --decode 16 --prefill 1
trashcan qwen4 ngram-bench META_SHARD TABLE_SHARD --mode ram-q4|direct|buffered-cold [--ids IDS]
python3 tools/qwen4_budget.py [--context N] [--batch N] [--head-split N] [--lean] [--check]
```
