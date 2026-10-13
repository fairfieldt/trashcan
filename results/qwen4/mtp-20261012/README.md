# Flash-Next MTP speculative decoding (2026-10-12)

Qwen3.8-Flash-Next ships one MTP block. The trashcan engine now drafts with it on the chain driver's CPU (incite) and
verifies the pending token and its drafts as one batch through the 12 GPU stages, rolling the recurrent state back to
the accepted prefix. On the six-Mac chain, greedy decoding of three chat replies goes from **14.85 to 20.86 tok/s
(+40%) with three drafts per round** (20.54 with two), output identical to plain greedy decoding; the Responses server
passes `scripts/test-server.py` with `--mtp`. On a 3K-token document summary it does not pay (-5%: fewer accepted drafts
and a costlier verification at that context).

## Weights

The unsloth GGUFs have no MTP block. The original repository `Qwen/Qwen3.8-Flash-Next` (public, revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`) has 31 `mtp.*` tensors: `pre_fc_norm_embedding`, `pre_fc_norm_hidden`
(per hyper-connection stream, 10240), `fc_embedding` and `fc_hidden` (2560 x 2560 each), one QSA layer with both
hyper-connection mixers and its own 512 experts, and a final mixer (`hyper_connection_mixer`). 5.2 GB in BF16.

```sh
tools/fetch-mtp.py Qwen/Qwen3.8-Flash-Next hf-mtp             # HTTP ranges: the mtp.* tensors, embedding and head (7.2 GiB)
python llama.cpp/convert_hf_to_gguf.py --mtp --outtype bf16 --outfile mtp-Qwen3.8-Flash-Next-BF16.gguf hf-mtp
llama-quantize --tensor-type ffn_gate_exps=iq4_xs --tensor-type ffn_up_exps=iq4_xs --tensor-type ffn_down_exps=iq4_nl \
    --tensor-type ffn_gate_shexp=iq4_xs --tensor-type ffn_up_shexp=iq4_xs --tensor-type ffn_down_shexp=iq4_nl \
    --output-tensor-type q6_k --token-embedding-type q8_0 mtp-Qwen3.8-Flash-Next-BF16.gguf mtp-G-iq4.gguf Q8_0 24
```

llama.cpp `b8e9ce8` converts the block to `blk.48.*` (`nextn.eh_proj` = `[fc_embedding | fc_hidden]`, gammas folded to
`1 + w`). unsloth's importance matrix has no entries for block 48, so the experts are plain IQ4_XS/IQ4_NL (the six-Mac
recipe without imatrix), dense matrices Q8_0: 2536 MiB, of which the block is 1.49 GB (1.40 GB experts); the file's
embedding and head copies are unused (the engine takes the trunk's). Files: `/Volumes/Green/Qwen/flash-next/mtp/` on
the Mac, `~/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf` on incite (the only machine that needs it).

## The block (`engine/src/qwen4/mtp.rs`)

As llama.cpp's `graph_mtp` (the only implementation available: Hugging Face transformers 5.19 ignores `mtp.*`):
position p pairs the token at p with the trunk's hc-wide residual at p - 1 (all four streams, before the final mixer;
zeros before position 0): `x_s = W_e RMS(e) enorm + W_h RMS(h_s) hnorm_s` for each stream s, then one QSA layer
(mixers, sparse attention with its own K/V and indexer caches, 512 experts and the shared expert), the block's own
final mixer and the trunk's LM head. A chained draft feeds the block its own output residual as `h`.

Validation (`qwen4 mtp ref`, CPU reference, G-iq4 trunk): on the 14-token reference prompt the block's prediction at
the last prompt position is the trunk's next token (9419) and three chained drafts are the trunk's greedy continuation
(0 2500 628). A logits-level comparison with llama.cpp's MTP context was not possible (driving it through the C API
crashed in `ggml_gallocr_alloc_graph`), so the comparison with llama.cpp is acceptance: llama-server `--spec-type
draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0` on the same GGUFs and prompts (CPU) against our reference (both
greedy; the texts agree for the first tens of tokens, then drift apart at near-ties):

| prompt (tokens) | llama.cpp, tokens per round (k = 3) | CPU reference, tokens per round (k = 3) |
|---|---|---|
| water (23) | 3.11 | 3.05 |
| csv (39) | 3.76 | 3.66 |
| transformer (25) | 2.98 | 3.16 |
| long (3061) | 2.77 (120 tokens) | 2.83 (136 tokens) |

## Acceptance (CPU reference)

`qwen4 mtp accept` decodes greedily on the CPU reference and drafts 3 tokens with the block before every step;
`accept.py` gives the conditional acceptance and the exact tokens per round of greedy speculative decoding. Prompts
(`acceptance/`): three chat questions (water, csv, transformer; thinking off) and the 3061-token document summary
used before (long), 256 steps or until the reply ends.

| prompt | a1 | a2 \| a1 | a3 \| a2 | tokens/round k=1 | k=2 | k=3 | generated ids below 32K / 64K |
|---|---|---|---|---|---|---|---|
| water | 0.883 | 0.840 | 0.815 | 1.90 | 2.61 | 3.05 | 86.7% / 94.1% |
| csv | 0.961 | 0.927 | 0.921 | 1.97 | 2.84 | 3.66 | 94.1% / 98.8% |
| transformer | 0.898 | 0.860 | 0.811 | 1.88 | 2.64 | 3.16 | 89.5% / 94.5% |
| long (136 steps) | 0.824 | 0.703 | 0.701 | 1.79 | 2.39 | 2.83 | 90.4% / 97.1% |
| pooled | 0.900 | 0.853 | 0.836 | 1.90 | 2.64 | 3.19 | |

The same runs replayed with the engine's CPU drafter (`qwen4 mtp replay`: the saved tokens and trunk residuals; the
block fed the last 128 prompt tokens), pooled over the four prompts:

| drafter | tokens/round k=1 | k=2 | k=3 |
|---|---|---|---|
| reference block, full vocabulary (water and csv only) | 1.93 | 2.72 | 3.32 |
| CPU drafter, full vocabulary | 1.89 | 2.62 | 3.14 |
| draft vocabulary 128K, 4-bit dense matrices and head | 1.88 | 2.59 | 3.17 |
| 64K, Q8 | 1.85 | 2.50 | 3.06 |
| **64K, 4-bit dense matrices (the engine's drafter)** | **1.84** | **2.53** | **3.05** |
| same, accepted drafts keep their chained rows (no rewrite, the engine's rounds) | 1.84 | 2.50 | 2.95 |
| 32K | 1.81 | 2.43 | 2.87 |
| 64K, 2-bit head | 1.82 | 2.46 | 2.90 |
| 96K, 2-bit head with its best 32 rows rescored at Q8 | 1.88 | 2.61 | 3.17 |

The drafter's int8 arithmetic and 4-bit dense matrices cost nothing measurable; the draft vocabulary does (tokens
above 64K are 1-6% of the replies: markdown, code). The block reads only the end of the prompt: on the long prompt
2.52 tokens per round (k = 3) with the last 128 prompt tokens and 2.39 with the last 16, against 2.83 with the whole
prompt; the chat prompts fit in the window.

## Drafter on the driver's CPU (`engine/src/qwen4/draft.rs`)

Each D700 has 40-80 MiB free at 32K context, so the block (1.49 GB) cannot go on a GPU; the driver's CPU (Xeon
E5-2697 v2, 12 cores / 24 threads, AVX without FMA, ~45 GB/s) runs it. Matrices are repacked at load: dense matrices
as 4-bit codes (IQ4_NL values, f32 scale per 32), the experts as the GGUF's IQ4 codes, the draft head (the trunk's Q6_K
rows below the draft vocabulary) as Q8_0; activations are quantized to Q8_0 per 32 and multiplied in int8 with SSSE3.
Checked on incite against the reference block (`qwen4 mtp draft-check`): draft logits cos 0.9996 with Q8 matrices,
0.98-0.99 with 4-bit ones, same argmax. One draft step at position 512 (`qwen4 mtp draft-bench`, p50 of 64 steps):

| configuration | weights read | ms per step |
|---|---|---|
| 64K head rows Q8, Q8 dense, 12 threads | 312 MiB | 8.6-10.5 (head 4.2) |
| 64K head rows Q8, 4-bit dense, 12 threads | 273 MiB | 9.2-11.1 |
| 64K head rows Q8, 4-bit dense, 24 threads | 273 MiB | 8.2 (head 4.1, attention 1.6, experts 1.2) |
| **same, rows 0-43007 on the GPUs (CPU ranks 43008-65535), 24 threads** | 155 MiB | **5.9** (head 1.7) |
| 64K head rows 4-bit, 12 threads | 193 MiB | 8.2-10.1 (head 3.7) |
| 128K head rows Q8, 12 threads | 492 MiB | 11.4-12.3 |
| full vocabulary Q8, 12 threads | 814 MiB | 18.4 |

The Q8 head streams at ~45 GB/s; the 4-bit and 2-bit kernels are compute bound on this CPU (no AVX2), so a smaller
head format saves little. The rest of a step runs well below the bandwidth (about a dozen short parallel loops).
Spinning or pinned worker threads, four-block reductions and decoding a weight block once for several tokens did
not change the step time measurably and were removed. The driver's two GPUs already hold the head rows 0-43007 (two
twelfths of `--head-split 12`): the drafter scores those there while the CPU ranks the rest of the draft vocabulary
(`Speculator::use_local_head`); on the chain a draft step took 5.7 ms (session 2) against 9.9 ms with the whole head
on the CPU (session 1).

## Verification on the chain

A round drafts k tokens, sends the pending token and the drafts as one batch (`forward` with `verify`), and every
stage keeps what a rollback needs: the DeltaNet states and convolution windows after every token but the last (qwen35's
`qwen_conv`/`qwen_gdn` snapshots, k x 3.1 MiB per DeltaNet layer: +28 MiB per GPU at k = 3, which fits at 32K with
`--batch 128`, the fullest GPU at 5830 of 5865 MiB) and, on the driver's layer 1, the n-gram convolution history before
the batch (the new history is the old one followed by the kept tokens' normed inputs). QSA K/V, the indexer's raw-key
ring and pooled block keys are addressed by position (a block completed by a rejected token is pooled again when it
completes). The stage that ends the model returns the final hidden state of every row and the batch's residual rows
(the block's input); every head slice scores all rows (row by row, then one top-k and merge pass over all of them) and
the candidates come back per row. `{"rollback": keep}` follows a round with a rejected draft. The block's rows at
accepted drafts keep the chained drafts' values: rewriting them with the trunk's rows raised acceptance by ~3% at
k = 3 but cost a 3-token draft step per round (session 1: 18.2 against 19.2 tok/s), so the engine does not.

Verification cost (`qwen4 stage-bench`, inflight, layers 0-3 with the n-gram module on GPU 0 and 4-7 on GPU 1, GPU ms
per stage at 850/1370 MHz, `stage-bench/`):

| batch | GPU 0 | GPU 1 | relative to one token |
|---|---|---|---|
| 1 token (decode at 64) | 5.57 | 5.15 | 1 |
| 2 tokens | 7.69 | 7.11 | 1.38 |
| 3 tokens | 9.01 | 8.49 | 1.64 |
| 4 tokens | 10.68 | 10.13 | 1.94 |

The experts grow with every token (per-(token, slot) kernels: 1.17 -> 2.91 ms per stage for 3 tokens), the mixers and
DeltaNet by ~45% (the fused mixers read their matrices per token). On the chain a 4-token verification takes 133 ms
against 67.3 ms per plain token (1.98x), a 3-token one 113 ms (1.67x); rollback 0.1-0.5 ms.

Correctness: `qwen4 mtp spec` decodes greedily without and with drafts and requires identical ids. It passed on the CPU
path (one process and a two-process TCP chain), on incite's GPUs for layers 0-5 with the M3 Ultra's CPU for 6-47
(`hybrid/`: the reference prompt and two chat prompts; also with the head split over the three stages, which drafts
on the GPU head slices), and on the six-Mac chain in every run below (the reference prompt: 9419 0 2500 628 353 1438
488 3242 30 248046). Each column of a verification batch runs the same kernels as a single-token decode.

## End to end (six Macs)

`qwen4 mtp spec MODEL --gpus 0,1 --nodes indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
--head-split 12 --context 32768 --batch 128 --ngram disk --mtp mtp-G-iq4.gguf --draft-tokens K --tg 192 --runs 3
water.ids csv.ids transformer.ids`: for each prompt plain greedy decoding, then speculative, three rounds alternating;
rates exclude the prompt. Run through `chain6.sh` and `tools/q4run.py` (nodes under gpuq leases, clocks logged:
850/1370 MHz on every GPU while working, `chain*/*.txt`), commands in `chain*/commands.log`.

**Session 2** (`chain2/`, commit `c4ac1fe`, the engine's drafter: 4-bit dense, Q8 head rows 43008-65535 on the CPU and
0-43007 on the driver's GPUs, 24 threads, no rewrite):

| prompt | plain tok/s | MTP k=3 | gain | tokens/round | MTP k=2 | gain | tokens/round |
|---|---|---|---|---|---|---|---|
| water | 14.83 | 19.66 | +32.5% | 2.95 | 19.50 | +31.1% | 2.42 |
| csv | 14.84 | 23.43 | +57.9% | 3.54 | 22.60 | +53.0% | 2.81 |
| transformer | 14.87 | 19.48 | +31.0% | 2.94 | 19.52 | +31.8% | 2.44 |
| **mean** | **14.85** | **20.86** | **+40.5%** | | **20.54** | **+38.7%** | |

Per round: k = 3: 17.1 ms of drafting (5.7 ms per step), 133.3 ms of verification, 0.1-0.25 ms of rollback; k = 2:
11.6 ms, 112.6 ms. The 3061-token document (96 tokens, one run): plain 13.31 tok/s, MTP 12.59 (2.21 tokens per round,
verification 155 ms at that context); its prefill took 10.41 s against 8.35 s, because the last chunk ran alone on
every machine to keep its residual rows (fixed in session 3).

Server (`tools/q4serve-test.sh` with `--mtp mtp-G-iq4.gguf --draft-tokens 3`, `--context 16384`): every
`scripts/test-server.py` check passes (parity, seeded sampling, token limits, tool calls); `q4-turns.py`: turn 1, 57
tokens in 3.20 s (17.8 tok/s streamed, against 13.9 without MTP in results/qwen4/chain-20261012), turn 2 (78 of 99
prompt tokens reused) 64 tokens in 2.73 s (23.4 tok/s).

**Session 1** (`chain1/`, commit `ff50ba6`: the whole draft head on the CPU, 12 threads; the variants compared):

| variant | k | plain tok/s | MTP tok/s | gain | draft ms/round | verify ms/round |
|---|---|---|---|---|---|---|
| rewrite of accepted rows, Q8 drafter | 3 | 14.58 | 18.18 | +24.7% | 29.6 | 147.0 (15 ms rewrite) |
| no rewrite, 4-bit drafter | 3 | 14.55 | 19.17 | +31.7% | 29.4 | 135.1 |
| no rewrite, 4-bit drafter | 2 | 14.62 | 19.04 | +30.2% | 20.1 | 114.1 |
| no rewrite, 96K 2-bit head rescored | 3 | 14.60 | 18.66 | +27.8% | 38.9 | 135.2 |

SESSION3

## Projections

`project.py`: tok/s = tokens per round / (k draft steps + a (k + 1)-token verification), with the pooled tokens per
round of the engine's rounds (1.84 / 2.50 / 2.95 for k = 1 / 2 / 3, the long prompt included; the chat prompts alone
give ~3.1 at k = 3). Today's chain (67.3 ms per token, verification 1.67x and 1.98x) with 6 ms draft steps gives
20.1 tok/s at k = 2 and 19.5 at k = 3 (measured on the chat prompts: 20.5 and 20.9). With the decode kernels being
worked on (35 ms per token, 3 tokens at 1.4x and 4 at ~1.65x):

| draft step | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| 10 ms | 35.4 | 36.2 | 33.5 |
| 6 ms (today's drafter with the GPU head rows) | 38.3 | 41.0 | 38.8 |
| 4 ms | 40.0 | 43.9 | 42.1 |

~40 tok/s needs the faster decode and verification kernels; with them k = 2 is the better depth unless 4-token
verification becomes relatively cheaper.

Drafting ahead during verification (the CPU idles ~110 ms per round: continue the chain and send the next k chained
drafts at once when every draft and the bonus token matched) was simulated on 9-deep chains (`ahead.py`): 0-5% at best
(deep drafts are accepted less often than fresh ones); not implemented.

## Reproduction

```sh
# Mac: weights (above), CPU reference and acceptance
trashcan qwen4 mtp ref G-iq4-00001-of-00006.gguf mtp-G-iq4.gguf out --chain 3
trashcan qwen4 mtp accept G-iq4-00001-of-00006.gguf mtp-G-iq4.gguf acceptance/water.json --tokens-file water.ids --steps 256 --depth 3
python3 accept.py acceptance/{water,csv,transformer,long}.json [--vocab 65536]
trashcan qwen4 mtp replay G-iq4-00001-of-00006.gguf mtp-G-iq4.gguf acceptance/water.json out.json --vocab 65536 [--depth 9] [--rounds]
tools/llama_server.sh mtp-G-iq4.gguf 8091 log; python3 tools/llama_accept.py http://127.0.0.1:8091 out.json 256 3 *.ids
# incite (under a lease): the drafter, and the hybrid check with the M3 Ultra as the last stage
tools/mtp-cpu6.sh; tools/mtp-hybrid.sh
# one Mac (under a lease): verification cost
tools/mtp-stagebench.sh
# six Macs
chain6.sh mtp-chain2 40 OWNER -- tools/mtp-chain2.sh; python3 summary.py chain2/mtp2-k3.json chain2/mtp2-k2.json
python3 project.py
```

## What is left

- Faster 2-4 token verification batches (a 4-token batch costs 1.98 decode steps: the per-slot expert kernels and
  the fused mixers grow with every token), the largest lever toward ~40 tok/s.
- A shorter draft step: the dense half of a step runs well below the CPU's bandwidth (about a dozen short parallel
  loops); the block's attention grows with the context (~5 ms at positions 2048-4096 against ~1.6 at 512) and needs
  threaded block scoring past 2K.
- Long prompts: the block reads at most the last 64 tokens of the prompt's last chunk (11% fewer tokens per round on
  the 3K-token document than with the whole prompt), and drafting does not pay on that document today; an adaptive
  depth (fewer drafts when acceptance drops) was not tried.
- Above temperature 0 the server keeps a draft when the sampled token equals it (the output follows the target's
  distribution with the same draws as plain decoding); acceptance there was not measured.
