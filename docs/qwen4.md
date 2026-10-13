# Qwen3.8-Flash-Next (`qwen4exp`) on a chain of Mac Pros

Qwen3.8-Flash-Next is a 177B-parameter sparse MoE: 121B of routed experts (about 6B used per token), a 51B hashed
n-gram embedding read at one layer, and 4.6B of dense weights. This document covers the reference, the n-gram
table placement, quantization and VRAM for six and four Mac Pros, the GPU stages, projected chain speed and the
changes for 200K context. Code: `engine/src/qwen4/`, shaders `engine/shaders/qwen4_*.comp`. The dense Qwen models
are in [qwen.md](qwen.md).

Status: the CPU reference matches llama.cpp; GPU stages for every layer type match the reference with the six-Mac
recipe's formats; with Q8 K/V caches and Q2_0 down projections on one layer per GPU a stage runs at 204,800
positions. The six-Mac chain runs (12 GPU stages over a Thunderbolt ring, 2026-10-12): the same greedy tokens as the
CPU reference, prefill 432 tok/s on a 512-token prompt, 633 on 2048 and 755 on 8192 with the two GPUs of a Mac on
consecutive chunks ([results/qwen4/prefill-overlap-20261012](../results/qwen4/prefill-overlap-20261012/README.md)),
the chunk size chosen by prompt length ([results/qwen4/prefill-chunks-20261012](../results/qwen4/prefill-chunks-20261012/README.md)),
split-K prefill mixers ([results/qwen4/prefill-mixer-20261012](../results/qwen4/prefill-mixer-20261012/README.md))
experts that decode each expert's rows once per tile of its tokens ([results/qwen4/prefill-experts-20261012](../results/qwen4/prefill-experts-20261012/README.md))
and the prefill kernels for 16-128-token chunks of [results/qwen4/prefill-kernels-20261013](../results/qwen4/prefill-kernels-20261013/README.md), decode 13.5 tok/s with the GPUs held at amdgpu's high clock
level while work arrives (`engine/src/clocks.rs`) and 8.1 without ([results/qwen4/chain-20261012](../results/qwen4/chain-20261012/README.md)), 14.7 with
every Mac scoring its head slices at once ([results/qwen4/head-parallel-20261012](../results/qwen4/head-parallel-20261012/README.md)), and 17.9 with
the decode kernels of [results/qwen4/decode-kernels-20261012](../results/qwen4/decode-kernels-20261012/README.md). MTP speculative
decoding (the block drafting on the driver's CPU, the chain verifying the pending token and its drafts in one batch,
each round drafting as many tokens as the recent acceptance pays for) takes greedy decoding of chat replies from 18.3
to 21.5-21.6 tok/s (+17-18%) with identical output, and costs at most 1.6% where drafts miss
([results/qwen4/mtp-20261012](../results/qwen4/mtp-20261012/README.md)). Not done yet: a cheaper head split and prefix
checkpoints (see [What is left](#what-is-left)). The Responses server runs on the chain ([Serving](#serving)).

| | |
|---|---|
| layers | 48: 36 Gated DeltaNet, 12 Qwen Sparse Attention (every fourth), plus one MTP block (a QSA layer with its own experts; drafts for speculative decoding) |
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
| **G-iq4** | **the six-Mac recipe: routed and shared experts IQ4_XS gate/up, IQ4_NL down; head Q6_K** | **0.0391** | **95.1%** | **1.002** |
| E-iq4xs | experts IQ4_XS gate/up, IQ4_NL down | 0.0357 | 95.3% | 1.003 |
| E-q4_0 | experts Q4_0 | 0.0402 | 94.9% | 1.003 |
| E-mxfp4 | experts MXFP4 | 0.0656 | 93.7% | 0.999 |
| E-iq3xxs | experts IQ3_XXS gate/up, IQ4_NL down | 0.0876 | 92.6% | 1.022 |
| E-iq2xxs | experts IQ2_XXS gate/up, IQ4_NL down | 0.2134 | 88.6% | 1.077 |
| N-q4_0 | every dense matrix Q4_0 (mixers, DeltaNet, attention, shared expert, n-gram projections) | 0.1961 | 88.7% | 1.065 |
| A-hc | hyper-connection mixers Q4_0 | 0.1190 | 91.4% | 1.026 |
| A-gdn | DeltaNet projections Q4_0 | 0.0613 | 94.0% | 1.012 |
| A-attn | QSA projections Q4_0 | 0.0669 | 93.4% | 1.011 |
| A-shexp | shared experts Q4_0 | 0.0191 | 96.3% | 1.003 |
| T-q4_0 | n-gram table Q4_0 | 0.0229 | 96.3% | 1.002 |

- **Dense matrices are the sensitive part.** 4.6B dense parameters at Q4_0 cost five times the divergence of
  121B expert parameters at Q4_0. Every group is sensitive on its own: the hyper-connection mixers 0.119, the QSA
  projections 0.067, the DeltaNet projections 0.061, even the shared experts 0.019. The recipe keeps dense
  matrices in Q8_0, which the GPU stages read with `qwen4_q8_mv`, `qwen4_q8_gemm` and fused Q8_0 mixers: about
  34 MB more per DeltaNet layer than Q4_0 and 0.25 ms more per decode step of a layer.
- **IQ4_XS gate/up with IQ4_NL down is the best 4-bit expert format** and also the smallest (the down rows are
  640 wide, which rules out the 256-element formats there). MXFP4 is clearly worse on this model. Quantizing the
  shared experts alike (the GPU stacks them with the routed ones) and the head to Q6_K adds only 0.003 (G-iq4).
- **2-bit experts cost as much as 4-bit dense matrices** (KLD 0.21, the same top token 88.6% of the time against
  95.3%). Four Macs at 2 bits is a large quality step, not a small one.
- **The n-gram table should keep the release's precision.** A Q4_0 table costs KLD 0.023, which is why the
  table stays on the driver's SSD at Q8_0 rather than in RAM at Q4_0.

## VRAM

`tools/qwen4_budget.py` models what `gpu.rs` allocates per stage (the mixers' matrices, including the fused copy
used for decode, the f32 router, DeltaNet or QSA matrices and state, the stacked experts, K/V and indexer caches,
batch scratch, the head) and matches all 15 stages measured on incite within 0.2 MiB (`--check`). Stages print
their device-local heap usage and refuse to run past the driver's budget, because RADV silently places the excess
in system memory (a 6480 MiB stage ran its head 18x slower that way).

Six Macs give each D700 4 layers. At 32K context and 256-token prefill chunks, with IQ4_XS/IQ4_NL experts and
Q8_0 dense matrices:

| | MiB |
|---|---|
| layers 0-3 on one D700 (n-gram module, 3 DeltaNet, 1 sparse QSA), **measured** | 5843 of 5888 (budget 5869) |
| the head (Q6_K repacked to 10 bits per weight, final mixer, logits) | 765 |
| head split over all 12 GPUs (`--head-split 12`) | 64 per GPU, 5907 per GPU: 19 over |
| the same with 128-token prefill chunks (`--batch 128`) | 5828 per GPU: fits by 60 |
| layers 0-3 and the first twelfth of the head (rows 0..21504), 128-token chunks, **measured** | 5830 of 5888 (budget 5869); decodes at 32K |
| with four savings not implemented yet (`--lean`: prefill reuses the fused mixer matrix, Q8_0 router and indexer projections, a raw indexer-key ring) | 5766 per GPU at 128-token chunks: fits by 122 |

Q4_0 experts do not fit six Macs (6043 MiB before any head share). Five Macs put 5 layers on most GPUs, where even
IQ3_XXS gate/up with IQ4_NL down is 106 MiB over (`--lean`, 128-token chunks). Four Macs give 6 layers per GPU
(`qwen4_budget.py`):

| four Macs, 6 layers per D700 | per GPU (32K, head over 8 GPUs, `--lean`, 128-token chunks) | quality |
|---|---|---|
| IQ4_XS/IQ4_NL experts | 8583 MiB: does not fit | KLD 0.036 |
| IQ2_XXS gate/up, MXFP4 down | 5802 MiB: fits by 86 | KLD 0.21 or worse; no GPU kernels for either format yet |
| IQ4 experts in host RAM, ~63% of each layer's experts cached in VRAM (not implemented) | fits by construction | KLD 0.036 |

`--head-split N` splits the head by vocabulary rows over the last N stages. The machine that ends the model sends the
final hidden state up the chain first; every machine passes it on and then scores its stages' rows, and the merged
candidates follow up the chain. Two CPU processes with the head in two halves reproduce the
reference's logits exactly; incite's two GPUs and the Mac node with the head in three give the same tokens and
logits within rel 3.4e-7 of the unsplit run. Scoring a slice on one D700 takes 1.12 ms for 1/12 of the vocabulary,
2.03 ms for 1/3, 2.62 ms for 1/2 and 4.60 ms for all of it (1,024 rows: 0.79 ms, so about 0.75 ms is fixed per
slice). A Mac's GPUs score at the same time (two twelfths: 1.12 ms together), and so do all six Macs: about 1.1 ms
plus the hops per token, against 4.6 ms for one GPU holding the whole head and 6.7 ms for twelve slices scored Mac
after Mac as the result travelled back, as before (decode 13.5 -> 14.7 tok/s,
[results/qwen4/head-parallel-20261012](../results/qwen4/head-parallel-20261012/README.md)).

**Recommendation.** Six Macs: IQ4_XS gate/up and IQ4_NL down for routed and shared experts, Q8_0 dense matrices,
Q6_K head split over the GPUs, n-gram table at Q8_0 on the driver's SSD (the `G-iq4` recipe: 116 GiB, 51 GiB of it
the n-gram table). At 32K context every GPU fits with `--head-split 12 --batch 128` (the first GPU's stage was
loaded and run on incite); the `--lean` savings would add margin. Four Macs (not implemented): the same weights,
with each Mac's experts (15.3 GiB) in host RAM and a static VRAM cache of each layer's most used ~63%, which covers
about 85% of routed activations by the imatrix counts (`expert_popularity.py` in the quant results); misses (about
3.5 MB per layer per token) are uploaded at the measured 6.5 GB/s. The 2-bit in-VRAM alternative fits only with the
same savings and loses much more quality.

## GPU stages (`engine/src/qwen4/gpu.rs`)

A stage owns a contiguous layer range on one D700, records one command buffer per layer per batch size (one for the
whole stage for batches of up to 8 tokens) and replays it with a new position, as the qwen35 stages do. The residual (4 x 2560 f32 per token) enters and
leaves through host buffers; that is also the payload between machines (40 KiB per token). Every mixer applies
the previous sublayer's write (`x_s += w_s y`) as it reads the residual, so a layer ends with a pending combine
that the next mixer, the head or the stage output resolves.

| part | kernels |
|---|---|
| mixer, up to 8 tokens | `qwen4_mix_down_sk` (pending combine, the four stream norms folded into the 320 bottleneck and 4 write rows; split over 8 chunks of the residual, the last workgroup to finish sums them; `_sk4`: four tokens per workgroup), `qwen4_mix_up` (read gates, stream mean, stores the combined residual) |
| mixer, prefill | `qwen4_hc_norm` (one workgroup per stream and token); Q8_0: the 320 bottleneck and 4 write rows (the fused matrix) as one GEMM split 16 ways along K (`qwen4_q8_gemm`, or up to 32 tokens `qwen4_q8_gemm_s8r1po`), `qwen4_mix_act` (sums the parts, activations), `qwen4_q8_gemm_mix` (the up projection with the read gates folded into the sublayer input; up to 16 tokens `qwen4_q8_gemm_s4r4om`); Q4_0: the dense matrix kernels, `qwen4_act`, `qwen4_hc_mix` |
| dense matrices | Q8_0: `qwen4_q8_mvw4_r4`/`_r2` (1-8 tokens: 8-byte loads, 4 or 2 rows per wave), `qwen4_q8_gemm_s16r1p` (9-32 tokens: a lane per row decoding each weight once for 16 tokens, whose inputs are scalar operands), `qwen4_q8_gemm` (64 x 64 tiles beyond), optionally split along K (the 2560-row out projections in 8 parts, summed by `qwen4_sum_parts`), f16 scales, 8.5 bpw; Q4_0: the Gemma kernels (f32 scales, 5 bpw); small f32 matrices (router, alpha/beta, indexer): `qwen4_matf32` up to 4 tokens |
| n-gram module | key/value projections, `qwen4_ple_sums` and `qwen4_ple_gate`, `qwen4_ple_conv` (dilated convolution, 9-step history) |
| DeltaNet | qwen35's `qwen_conv`, `qwen4_gdn` (up to 8 tokens), beyond: `qwen4_gdn_prep` (q, k norms and the gates once per token) and `qwen4_gdn_rows` (a state row in a lane's registers, q and k as LDS broadcasts), `qwen4_gated_norm` (sigmoid gate) |
| QSA, up to 2051 tokens | qwen35's `qwen_qk` and the split/unsplit attention kernels (dense is exact there) |
| QSA, longer contexts | `qwen4_idx_store`/`qwen4_idx_pool` (raw and pooled block keys), `qwen_qk` on the indexer queries, `qwen4_idx_score` (block scores, one wave per block for 8 tokens), `qwen4_idx_select` (exact radix top-512, ties to the earlier block), `qwen4_attention_sparse[_full]` |
| MoE | router (f32, with the shared gate as row 512), `qwen4_topk`; below 16 tokens `qwen4_moe_gu`, `qwen4_moe_down` per (token, slot); from 16 tokens `qwen4_moe_tiles` groups the slots by expert into tiles of up to 8, `qwen4_moe_gu_tiles`/`qwen4_moe_down_tiles` decode each expert's rows once per tile (a lane per row, whole 64-byte lines, the slots' inputs through scalar loads) and `qwen4_moe_sum` adds a token's slots (2.6x faster at 128 tokens, results/qwen4/prefill-experts-20261012); up to 8 tokens `qwen4_moe_down_w` for the down projections |
| experts | IQ4_XS (4.25 bpw), IQ4_NL or Q4_0 (f16 scale per 32, 4.5 bpw), at their GGUF size; the shared expert is stacked as expert 512 and must share the routed format |
| head | final mixer, qwen35's Q6 `headvec` (int8 with f32 scales per 16), top-64; any range of vocabulary rows (`--head-split`) |

`qwen4 stage-bench` times stages alone (prefill chunks, then decode steps, back to back so the clocks stay up) with
GPU timestamps around every layer and phase; `--start` prefills that many tokens first and also reports the last
of those chunks. With `TRASHCAN_Q4_PROFILE=1` it also times every dispatch of the decode steps and of one prefill
chunk (`dispatches`, `prefill_dispatches`).

### Validation on the D700s

Layers 0-5 (n-gram module, DeltaNet, QSA and MoE) ran on incite's two GPUs with layers 6-47 and the head on a CPU
node on the M3 Ultra (`--cuts 3,6 --nodes MAC:7071`), compared with the CPU reference on the same GGUF
([results/qwen4/gpu-20261006](../results/qwen4/gpu-20261006/README.md)):

| GGUF | 14-token prompt, 10 greedy tokens | logits | DeltaNet, n-gram and MoE layers | QSA layer (f16 K/V) |
|---|---|---|---|---|
| all Q4_0 (fused and unfused mixers) | identical | rel 4.3e-5 | rel ~1e-6 | rel 4.4e-4 |
| Q8_0 dense, Q4_0 experts | identical | rel 2.5e-5 | rel ~1e-6 | rel 5.1e-4 |
| Q8_0 dense, IQ4_XS/IQ4_NL experts | identical | rel 1.7e-5 | rel 5e-7 to 1.1e-6 | rel 4.4e-4 |

On a 3061-token prompt in 256-token chunks (sparse QSA, with the grouped expert kernels of the time), the last chunk's QSA output is within
rel 6.6e-4 of the CPU reference and the indexer queries within 2.4e-5; moving block scoring into
`qwen4_idx_score` left all 111 captured tensors bit-identical.

### Measured stage times (incite, 850/1370 MHz)

Six-Mac recipe (Q8_0 dense, IQ4_XS/IQ4_NL experts), per layer:

Decode columns: the kernels of results/qwen4/decode-kernels-20261012 (before them in brackets), context 32768 (QSA with the
indexer); prefill columns from the prefill rounds (the decode kernels left prefill unchanged).

| phase | decode, 64 tokens of context | decode at 32K | prefill, 256-token chunk at 0 | chunk at 32K |
|---|---|---|---|---|
| mixer (each of two) | 0.10-0.12 ms (0.14-0.17) | 0.10-0.12 ms (0.17) | 2.4 ms | 2.4 ms |
| DeltaNet | 0.41 ms (0.49-0.50) | 0.41 ms (0.49-0.50) | 19.2-19.4 ms | 19.3-19.4 ms |
| QSA | 0.48 ms (0.58) | 1.27 ms (projections 0.21, block selection 0.53, attention 0.53; before 1.40) | 16.1 ms | 119.8 ms (selection 35.9, attention 75.9) |
| router and top-k | 0.075 ms (0.10) | 0.075 ms (0.10) | 3.7-3.9 ms | 3.8 ms |
| experts (10 routed + shared) | 0.26 ms (0.29-0.30) | 0.26 ms (0.29) | 57.4-58.2 ms | 57.4 ms |
| n-gram module (layer 1) | 0.24 ms (0.43) | | 8.4 ms | |
| DeltaNet layer | 0.95-0.98 ms (1.22-1.30) | 0.95-0.98 ms (1.22-1.24) | 89-96 ms | 89-90 ms |
| QSA layer | 1.04 ms (1.38) | 1.83 ms (2.18) | 88-99 ms | 196 ms |
| head (final mixer, 248,320 rows, top-64) | 4.6 ms | | | |

Layer 1 adds the n-gram module to its DeltaNet layer (0.24 ms per decode token, 8.4 ms per chunk). The prefill mixer
time is from results/qwen4/prefill-mixer-20261012 (4.3-5.7 ms before the split-K bottleneck; it does not depend on the
position).

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
prefill the per-(token, slot) expert kernels took 50-60% of a layer; kernels that decode each expert's rows once per
tile of its slots took the 8 layers' experts at 128 tokens from 221 to 84 ms (results/qwen4/prefill-experts-20261012).
Splitting the mixer bottleneck GEMM along K (its 324 rows gave 5-6 workgroups) took 16 prefill mixers from 63 to
20 ms per 128-token chunk and from 77 to 38 ms per 256-token chunk; the fused mixers now win only up to 8 tokens
(64-token chunks: 53 -> 11 ms; results/qwen4/prefill-mixer-20261012).
A decode layer reads about 110 MB (61 MB of DeltaNet matrices, 14 MB of mixers, 5 MB router, 29 MB of experts) in
0.96 ms, 44% of the D700's 263 GB/s (1.23 ms, 34%, before the decode kernels of results/qwen4/decode-kernels-20261012:
split mixer bottleneck, 8-byte Q8_0 loads, loads issued before use, one command buffer per decode step).

## Chain performance: measured, ceilings and targets

Six Macs, 12 stages (2026-10-12, [results/qwen4/chain-20261012](../results/qwen4/chain-20261012/README.md),
[results/qwen4/prefill-batch-20261012](../results/qwen4/prefill-batch-20261012/README.md),
[results/qwen4/prefill-overlap-20261012](../results/qwen4/prefill-overlap-20261012/README.md)). **Ceiling**: what the
hardware allows with this placement. **Now**: measured where marked, otherwise modeled from stage-bench times.
**Realistic**: with good but not heroic GCN kernels and the work listed below the tables (about ±30%).

Inputs: a D700 reads 263 GB/s and does 3.48 TFLOP/s FP32 (no packed FP16, no int8 dot products, so quantized weights
are dequantized to FP32). A token uses about 6.0B parameters, 12 GFLOP, and a decode token reads about 6.0 GB:
36 DeltaNet layers of ~110 MB (62 MB DeltaNet, 14 MB mixers, 5 MB router, 29 MB for 11 experts) plus 6.3 MB of
recurrent state, 12 QSA layers of ~104 MB, the n-gram module (35 MB) and the head (0.8 GB at 10 bits per weight,
spread over the 12 GPUs). QSA reads at most ~2,051 positions per token (4 MB of f16 K/V, 2.2 MB of Q8) whatever the
context, plus 128 bytes per position of pooled indexer keys (50 MB over 12 layers at 32K, 315 MB at 200K). Contexts
past 32K need the S-q2dn build and a Q8 K/V cache ([200K context](#200k-context-on-six-macs)).

**Decode** (one stream; in the pipeline one GPU works at a time, so the ceiling is all bytes at 263 GB/s):

| context | ceiling | now | realistic |
|---|---|---|---|
| 512 | 21.2 ms, 47 tok/s | 56 ms, **17.9 tok/s** (measured) | ~35 ms, ~28 tok/s |
| 2K-8K | 21.3-21.4 ms, 47 tok/s | 62 ms, 16.1 tok/s (measured at 2K) | ~37 ms, ~27 tok/s |
| 32K (G-iq4 maximum) | 21.6 ms, 46 tok/s | ~66 ms, ~15 tok/s | ~40 ms, ~25 tok/s |
| 128K (S-q2dn, Q8 K/V) | 22.0 ms, 45 tok/s | ~77 ms, ~13 tok/s | ~43 ms, ~23 tok/s |
| 200K | 22.2 ms, 45 tok/s | ~84 ms, ~12 tok/s | ~46 ms, ~22 tok/s |
| 262K (model maximum) | 22.5 ms, 44 tok/s | ~90 ms, ~11 tok/s | ~48 ms, ~21 tok/s |

The 56 ms of today: about 50 ms of GPU layers (0.95-1.11 ms each against 0.42 ms of bytes: 44% of the bandwidth;
13 ms less since the decode kernels of results/qwen4/decode-kernels-20261012), ~1 ms of head slices (all Macs at once since 2026-10-12, which saved 5.9 ms),
~3-4 ms of host work (0.39 ms per stage) and ~1 ms of Thunderbolt hops. Toward ~28 tok/s: fused decode kernels at
60-70% of the bandwidth (~28 ms), less host work per stage (~3 ms); at long contexts a multi-workgroup block selection
(0.6 ms per QSA layer at 32K, 1.9 ms at 200K today).

**Prefill** (whole prompt; the ceiling is the FP32 roofline of all 12 GPUs, ~3,300 tok/s in steady state, and for
short prompts the time to fill 12 stages):

| prompt | ceiling | now | realistic |
|---|---|---|---|
| 512 | ~0.25 s | 1.19 s, **432 tok/s** (measured; 1024 tokens: 543 tok/s) | ~1.2 s, ~430 tok/s |
| 2048 | ~0.8 s | 3.2 s, **633 tok/s** (measured) | ~3 s, ~660 tok/s |
| 4K | ~1.5 s, 2,800 tok/s | 6.0 s, **686 tok/s** (measured) | ~5 s, ~820 tok/s |
| 8K | ~2.7 s, 3,000 tok/s | 10.8 s, **755 tok/s** (measured) | ~9 s, ~920 tok/s |
| 32K | ~11 s, 3,150 tok/s | ~40 s, ~820 tok/s | ~35 s, ~950 tok/s |
| 128K | ~42 s | ~3.4 min, ~650 tok/s | ~2.5 min, ~900 tok/s |
| 200K | ~65 s | ~5.8 min, ~590 tok/s | ~4 min, ~850 tok/s |
| 262K | ~83 s | ~7.8 min, ~560 tok/s | ~5.5 min, ~820 tok/s |

Today: the experts take 42-45% of a chunk's time at 20-39% of FP32 peak (each expert's rows decoded once per tile of
up to 8 of its tokens; most tiles of 16-64-token chunks hold 1-3), the Q8_0 projections run at 27-45% of the peak up
to 32 tokens and 50-63% beyond, and the DeltaNet recurrence is 2.7% of a chunk
([results/qwen4/prefill-kernels-20261013](../results/qwen4/prefill-kernels-20261013/README.md)). Every GPU is a
pipeline stage (a Mac's two GPUs work on consecutive chunks), so a prompt takes about the sum of its chunks' stage times
plus 11 times the largest, and the pipeline fill is most of a short prompt's time. Toward the realistic column: the
expert tiles, faster block selection and sparse attention at long contexts, the f32 router, and the stage output copy
to host memory (~2 GB/s: 1.3 ms per GPU per 64-token chunk).

**Beyond these ceilings.** MTP speculative decoding (measured, [results/qwen4/mtp-20261012](../results/qwen4/mtp-20261012/README.md)):
the block (1.49 GB) runs on the driver's CPU (5.7 ms per draft step with the draft head's first rows on the driver's
GPU head slices); greedy drafts are accepted 90%, 85% and 84% of the time (first, second given the first, third given
both), so a round of 3 drafts yields ~3 tokens for 3 draft steps and a 4-token verification. With the decode kernels
a 2 / 3 / 4-token verification costs 1.69x / 2.0x / 2.37x a 53.9 ms decode step, and chat replies run at 21.5-21.6
tok/s against 18.3 (+39% before those kernels, when verification cost 1.43x / 1.69x / 2.0x); ~39-41 tok/s is
projected with ~35 ms decode steps and verification of 2 / 3 / 4 tokens at 1.2x / 1.4x / 1.65x. Tensor or
expert parallelism over all 12 GPUs at once: ~2 ms of bytes per token, but ~96 cross-Mac all-reduces per token at
>= 100 us each over the ring bound it near 50-80 tok/s, at a large cost in engineering.

Four Macs (not built: each Mac's experts in host RAM with ~63% of each layer's experts cached in VRAM) were projected
at ~10.8 tok/s decode and 125-210 tok/s prefill.

## 200K context on six Macs

At 204,800 positions one QSA layer's f16 K/V cache is 400 MiB, and every GPU of six Macs holds one QSA layer, so the
recipe that fits at 32K is 302 MiB over at 200K (`qwen4_budget.py --grid --context 204800`). Implemented so far:

| change | per GPU at 204,800 | quality |
|---|---|---|
| Q8 K/V cache (`--kv q8`: int8 with an f16 scale per 32, 272 bytes per head and position instead of 512) | −188 MiB | KLD 0.0136 against the f16 cache at positions 32K-64K, where merely running f16 attention through another kernel gives 0.0102 (llama.cpp); NEEDLE_SHORT |
| raw indexer keys in a ring of the batch plus a block (only pooling reads them) | −100 MiB | exact: 111 tensors bit-identical |
| Q2_0 down projections on one layer per GPU (1, 5, 8, 12, 16, 20, 24, 30, 33, 36, 40, 45): ISTA's GSQ codes for the routed experts, ggml's Q2_0 for the shared expert (the GPU stacks it with the routed ones) | −225 MiB | KLD 0.098 against Q8_0, against 0.039 without them |

The Q2_0 layers are expensive in quality (results/qwen4/quant-20261006): on the 39K-token corpus the routed GSQ
tensors alone take KLD from 0.039 to 0.078 (same top token 93.1% against 95.1%), the shared experts at ggml's Q2_0 add
0.021, and ggml's own (absmax) Q2_0 codes for the routed experts would make it 0.179; GSQ's codes are worth having.

Headroom on the fullest GPU (IQ4_XS/IQ4_NL experts, Q8_0 dense, head split over 12 GPUs). `--lean` and `--more` model
six savings that are not implemented: prefill reusing the fused mixer matrix, Q8_0 router and indexer projections,
the head kept at Q6_K size, f16 pooled indexer keys and block scores in tiles:

| K/V cache | Q2_0 down | current, 128-token chunks | `--lean` | `--lean --more` | `--lean --more`, 64-token chunks |
|---|---|---|---|---|---|
| f16 | none | −302 | −256 | −199 | −161 |
| f16 | 1 of 4 layers | −77 | −31 | +26 | +64 |
| f16 K, Q8 V | 1 of 4 layers | +17 | +62 | +120 | +158 |
| Q8 | none | −115 | −69 | −12 | **+26** |
| Q8 | 1 of 4 layers | **+110** | +156 | +213 | +252 |

Two routes to 200K follow:

- **Q8 K/V and the Q2_0 layers** run today with 110 MiB to spare (up to 286,720 positions, past the model's
  262,144), at KLD 0.098, or 0.078 once the shared expert gets its own format next to Q2_0 routed experts.
- **Q8 K/V and the six savings with 64-token prefill chunks** keep the 32K recipe's quality (KLD 0.039 plus the Q8 K/V
  effect) with 26 MiB to spare, but need all six changes and a slower prefill. With the current code, Q8 K/V alone
  reaches 114,688 positions (147,456 with `--lean` or with 64-token chunks).

**Measured on incite:** layers 0-3 of the Q2_0 build (Q2_0 down on layer 1), the first twelfth of the head, a Q8 K/V
cache for 204,800 positions and 128-token chunks allocate 5780 of 5888 MiB (budget 5868, as modeled) and run at
850/1370 MHz. Decode at position 204,672: 1.23-1.24 ms per DeltaNet layer, 3.65 ms for the QSA layer (projections
0.26, block selection 1.92, attention 0.66), 1.0 ms for the head slice. A 128-token chunk takes 51-62 ms per layer at
position 0 and 202 ms for the QSA layer at 204,544 (selection 101, attention 54). The Q2_0 down projections cost the
same 0.29 ms per decode token as IQ4_NL and 6 ms more per chunk (38 against 32 ms, measured with the grouped expert
kernels since removed), since 640-wide rows gave that kernel only 10 of its 16 lanes' blocks.

Projected six-Mac decode at 200K: 36 x 1.24 + 12 x 3.65 + 0.43 + 6.7 + 12 x 0.39 + 0.5 = 101 ms, **9.9 tok/s**
(14.1 at short context). Prefill of a 200K prompt averages about 0.57 s per 128-token chunk per Mac (6 DeltaNet
layers at 52 ms, 2 QSA layers averaging 129 ms): **about 225 tok/s, 15 minutes** for the whole prompt.

NEEDLE

**Past 262K: K/V in host RAM (estimate).** Each decode token reads only the ~2,051 selected positions of a QSA layer:
2.2 MB of Q8 K/V. Kept in host RAM and gathered over PCIe after block selection (scattered 4.3 KB block reads at an
assumed 3-5 GB/s; 6.5 GB/s measured for bulk uploads) that is about 0.5-0.7 ms per QSA layer, **+6-8 ms per token**
(8%). Prefill chunks select most blocks between them, so a chunk would stream the layer's whole K/V: 435 MB at 400K
positions, 67 ms at 6.5 GB/s, overlappable with the chunk's 200-400 ms of QSA work. Only the pooled indexer keys (f32
125 MiB per QSA layer at 1M positions, half as f16) and a staging buffer stay in VRAM, and host RAM needs about 1.1 KB
per position per QSA layer (2.2 GB per Mac at 1M). Without Q2_0 this would also free the 212 MiB per GPU that the Q8
K/V cache takes at 200K.

**MTP at long contexts.** The block runs on the driver's CPU with its K/V in host RAM (f32, 4 KiB per position,
0.8 GB at 200K). Its sparse attention reads at most ~2,051 positions per draft, but the attention part of a draft step
grows from ~1.6 ms at position 512 to ~5 ms at 2048-4096 (`draft-bench`; block scoring and selection run on one
thread per token), so long contexts need that threaded. Not measured past 8K.

## What is left

1. **A cheaper head split**: the Macs score their slices at once, so about 1.1 ms per token follows the last layer;
   cutting the 0.75 ms fixed cost per slice (single-workgroup top-k passes) would shorten it, and storing the head
   compactly (Q6_K is 6.6 bits, the repack 10) would free VRAM.
2. **The `--lean` savings** (about 60 MiB per GPU), for margin at 32K and room for 256-token prefill chunks. Q8_0
   mixers with write rows no longer read their separate down copy or f32 write rows (prefill uses the fused
   matrix too), so those uploads can go (~28 MiB per GPU).
3. **Clock levels** (done): in decode each GPU idles ~90% of a token and drops to its lowest clocks on `auto`;
   stages now hold the high level while work arrives and return to `auto` after 1 s idle (~35 W per idle GPU).
4. **Prefill**: experts that reuse weights across the tokens routed to an expert (done: 221 -> 84 ms for 8 layers
   at 128 tokens, 2048 tokens 434 -> 598 tok/s, results/qwen4/prefill-experts-20261012); a lane-per-row Q8_0 GEMM up
   to 32 tokens, split-K out projections and the DeltaNet recurrence in registers (done: 512 tokens 369 -> 432 tok/s,
   8192 702 -> 755, results/qwen4/prefill-kernels-20261013); the expert tiles (42-45% of a chunk; occupancy splits,
   prefetch, vector-loaded inputs and unrolled lines all lost there), the Q8_0 GEMMs at 64-128 tokens (58-63% of the
   FMA peak, LDS-bound), grouping the 12 query heads that share a KV head in sparse attention (76 ms per chunk at
   32K), the f32 router (8% of a chunk), and smaller chunks for short prompts, whose time is mostly the 12-stage
   pipeline fill since the two GPUs of a Mac work on consecutive chunks (done: 2048 tokens 210 -> 324 tok/s,
   results/qwen4/prefill-overlap-20261012).
5. **Decode**: a DeltaNet layer takes ~1.0 ms per token (about 110 GB/s; results/qwen4/decode-kernels-20261012):
   the expert gate/up kernel (0.17 ms, neither its VALU nor its loads alone explain it), the out projections at
   ~145 GB/s, the mixer's second half (0.05 ms each), single-wave top-k (0.03 ms); RADV writes back and invalidates
   the L2 at every barrier on GFX6. Batches of 2-4 tokens (MTP verification) re-read each token's 11 experts. And a
   multi-workgroup radix select (block selection takes 0.6 ms of a QSA layer at 32K and 1.9 ms at 200K).
6. **Long prefill**: block selection (101 ms) and sparse attention (54 ms) of a 128-token chunk at 200K.
7. **Four Macs**: the host-RAM expert cache (or 2-bit kernels, if that quality is acceptable).
8. **Past 262K**: QSA K/V in host RAM (estimated above).
9. **MTP** (done: chat replies 18.3 -> 21.5-21.6 tok/s with adaptive rounds, results/qwen4/mtp-20261012): cheaper
   2-4-token verification batches (1.69x / 2.0x / 2.37x a decode step since the decode kernels), a plain step that
   returns the residual row (k = 0 rounds cost ~1.1 ms more than a decode step), a shorter CPU draft step, and more of
   the prompt for the block (it reads the last prefill chunk: 11% fewer tokens per round on a 3K-token document,
   where drafting breaks even).
10. **Prefix checkpoints**: the server continues only from the live state, so a prompt that is not an extension of
    everything processed so far starts again at position 0.

## Serving

`trashcan serve` on a qwen4exp GGUF runs the Responses server as the chain's driver (the qwen35 tokenizer, template,
reasoning and tool-call parsing; see [server.md](server.md)). Start a node on every other Mac first, in chain order
along the Thunderbolt ring, then the server on the first Mac (incite):

```sh
# indite, inflight, alight, invite, enlight
trashcan qwen4 node G-iq4-00001-of-00006.gguf --listen 0.0.0.0:7071 --gpus 0,1
# incite
trashcan serve G-iq4-00001-of-00006.gguf --host 0.0.0.0 --port 8000 \
    --nodes indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071 \
    --head-split 12 --context 32768 --batch 128 --ngram disk
```

Each `--nodes` address is resolved on the previous machine of the chain (`<peer>.tb<bus>` names from `tblink`).
A request continues from the live state when everything processed so far (the previous prompt and its reply) is a
prefix of its prompt, as in a growing conversation; any other prompt starts again at position 0, because the DeltaNet
and n-gram states only run forward. `--no-prefix-cache` always starts at 0. `--mtp mtp-G-iq4.gguf [--draft-tokens 3]
[--draft-vocab 65536]` drafts with the MTP block on the driver's CPU, up to 3 tokens per round, as many as the recent
acceptance pays for at the measured costs (above temperature 0 a draft is kept when the sampled token equals it, so
sampling follows the target distribution with the same draws). No `--tp`.
`scripts/test-server.py` passes against it, and a second chat turn reuses the first
([results/qwen4/chain-20261012](../results/qwen4/chain-20261012/README.md#server)).

## Commands

```sh
# reference and validation (Mac)
trashcan qwen4 ref MODEL OUT [--tokens-file IDS] [--emulate-ggml] [--greedy N]
trashcan qwen4 layer-check MODEL LLAMA_DUMP [--emulate-ggml]
# chain: every machine after the first, then the driver
trashcan qwen4 node MODEL --listen 0.0.0.0:7071 [--cpu | --gpus 0,1]
trashcan qwen4 validate MODEL OUT --gpus 0,1 --cuts 3,6 --nodes MAC:7071 [--head-split 3] [--kv q8] --greedy 10
trashcan qwen4 bench MODEL --gpus 0,1 --nodes ... --pp 256 --tg 32
# MTP: CPU reference, acceptance, the CPU drafter, speculative decoding on the chain (greedy with and without drafts)
trashcan qwen4 mtp ref MODEL MTP OUT --chain 3
trashcan qwen4 mtp accept MODEL MTP OUT.json --tokens-file IDS --steps 256 --depth 3
trashcan qwen4 mtp replay MODEL MTP RUN.json OUT.json --vocab 65536 [--rounds]
trashcan qwen4 mtp draft-check MODEL MTP; trashcan qwen4 mtp draft-bench MODEL MTP --start 512 --threads 12
trashcan qwen4 mtp spec MODEL --gpus 0,1 --nodes ... --mtp MTP --draft-tokens 3 --tg 192 --runs 3 PROMPT.ids...
# six Macs (not run yet: every other Mac runs `qwen4 node MODEL --listen 0.0.0.0:7071 --gpus 0,1`), 32K with G-iq4,
# 200K with the Q2_0 build and Q8 K/V
trashcan qwen4 bench G-iq4.gguf --gpus 0,1 --nodes MAC2:7071,MAC3:7071,MAC4:7071,MAC5:7071,MAC6:7071 \
    --head-split 12 --context 32768 --batch 128 --ngram disk --pp 2048 --tg 64
trashcan qwen4 bench S-q2dn.gguf --gpus 0,1 --nodes MAC2:7071,MAC3:7071,MAC4:7071,MAC5:7071,MAC6:7071 \
    --head-split 12 --context 204800 --batch 128 --kv q8 --ngram disk --pp 2048 --tg 64
# GPU stages alone, and the n-gram table
trashcan qwen4 stage-bench MODEL --ranges 0:4 --gpus 0 --context 32768 --start 32512 --decode 16 --prefill 1
trashcan qwen4 stage-bench S-q2dn.gguf --ranges 0:4 --gpus 0 --context 204800 --kv q8 --batch 128 \
    --head-rows 0:21504 --start 204672 --decode 16 --prefill 1
trashcan qwen4 stage-bench MODEL --ranges 4:5 --gpus 0 --head-rows 0:20736 --decode 4 --prefill 0 --batch 16
trashcan qwen4 ngram-bench META_SHARD TABLE_SHARD --mode ram-q4|direct|buffered-cold [--ids IDS]
python3 tools/qwen4_budget.py [--context N] [--batch N] [--head-split N] [--lean] [--kv q8] [--q2-down rco]
python3 tools/qwen4_budget.py --grid --context 204800   # six Macs, long-context options
python3 tools/qwen4_budget.py --check
```
