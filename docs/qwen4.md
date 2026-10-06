# Qwen3.8-Flash-Next (`qwen4exp`) on a chain of Mac Pros

Qwen3.8-Flash-Next is a 177B-parameter sparse MoE: 121B of routed experts (about 6B used per token), a 51B hashed
n-gram embedding read at one layer, and 4.6B of dense weights. This document covers the reference, the n-gram
table placement, the quantization recipes for six and four Mac Pros, and the GPU stages. Code: `engine/src/qwen4/`,
shaders `engine/shaders/qwen4_*.comp`. The dense Qwen models are in [qwen.md](qwen.md).

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
driver's 64 GB without a measurable end-to-end gain, and a resident Q8_0 copy (51 GiB) does not fit safely.
On disk the rows can keep the release's precision (Q8_0 here, 170 bytes; BF16 rows cost the same 4 KiB read).

## Quantization and VRAM

TBD

## GPU stages

TBD

## Commands

```sh
# reference and validation (Mac)
trashcan qwen4 ref MODEL OUT [--tokens-file IDS] [--emulate-ggml]
trashcan qwen4 layer-check MODEL LLAMA_DUMP [--emulate-ggml]
# chain: every machine after the first, then the driver
trashcan qwen4 node MODEL --listen 0.0.0.0:7071 [--cpu | --gpus 0,1]
trashcan qwen4 validate MODEL OUT --gpus 0,1 --cuts 3,6 --nodes MAC:7071
trashcan qwen4 bench MODEL --gpus 0,1 --nodes ... --pp 256 --tg 32
trashcan qwen4 ngram-bench META_SHARD TABLE_SHARD --mode ram-q4|direct|buffered-cold [--ids IDS]
```
