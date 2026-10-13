# Flash-Next MTP speculative decoding (2026-10-12)

Qwen3.8-Flash-Next ships one MTP block. The trashcan engine now drafts with it on the chain driver's CPU (incite) and
verifies the pending token and its drafts as one batch through the 12 GPU stages, rolling the recurrent state back to
the accepted prefix. On the six-Mac chain, greedy decoding of real text goes from **14.6 to 19.2 tok/s (+32%, three
drafts per round)** with output identical to plain greedy decoding (session 1 below; session 2 with the final drafter
is in [End to end](#end-to-end-six-macs)).

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
`1 + w`). unsloth's importance matrix has no entries for block 48, so the experts are plain IQ4_XS/IQ4_NL (the
six-Mac recipe without imatrix), dense matrices Q8_0: 2536 MiB, of which the block is 1.49 GB (1.40 GB experts); the
embedding and head copies in the file are unused (the engine takes the trunk's). Files:
`/Volumes/Green/Qwen/flash-next/mtp/` on the Mac, `~/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf` on incite.

## The block (`engine/src/qwen4/mtp.rs`)

As llama.cpp's `graph_mtp` (the only implementation available: Hugging Face transformers 5.19 ignores `mtp.*`):
position p pairs the token at p with the trunk's hc-wide residual at p - 1 (all four streams, before the final mixer;
zeros before position 0): `x_s = W_e RMS(e) enorm + W_h RMS(h_s) hnorm_s` for each stream s, then one QSA layer
(mixers, sparse attention with its own K/V and indexer caches, 512 experts and the shared expert), the block's own
final mixer and the trunk's LM head. A chained draft feeds the block its own output residual as `h`.

Validation (`qwen4 mtp ref`, CPU reference on the M3 Ultra, G-iq4 trunk): on the 14-token reference prompt the block's
prediction at the last prompt position is the trunk's next token (9419) and three chained drafts are the trunk's
greedy continuation (0 2500 628). A logits-level comparison with llama.cpp's MTP context was not possible (driving it
through the C API crashed in `ggml_gallocr_alloc_graph`), so the check against llama.cpp is acceptance: llama-server
`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0` on the same GGUFs and prompts (CPU) against our
reference (both greedy; the texts are the same for the first tens of tokens and then drift apart at near-ties):

| prompt (tokens) | llama.cpp, tokens per round (k = 3) | CPU reference, tokens per round (k = 3) |
|---|---|---|
| water (23) | 3.11 | 3.05 |
| csv (39) | 3.76 | 3.66 |
| transformer (25) | 2.98 | 3.16 |
| long (3061) | 2.77 (120 tokens) | 2.83 (136 tokens) |

## Acceptance (CPU reference)

`qwen4 mtp accept` decodes greedily on the CPU reference and drafts 3 tokens with the block before every step;
`accept.py` turns that into conditional acceptance and the exact tokens per round of greedy speculative decoding
(greedy output is the same with or without drafts). Prompts (in `acceptance/`): three chat questions (water, csv,
transformer; thinking off) and the 3061-token document summary used before (`long`), 256 steps or until the reply ends.

| prompt | a1 | a2 \| a1 | a3 \| a2 | tokens/round k=1 | k=2 | k=3 | generated ids below 32K / 64K |
|---|---|---|---|---|---|---|---|
| water | 0.883 | 0.840 | 0.815 | 1.90 | 2.61 | 3.05 | 86.7% / 94.1% |
| csv | 0.961 | 0.927 | 0.921 | 1.97 | 2.84 | 3.66 | 94.1% / 98.8% |
| transformer | 0.898 | 0.860 | 0.811 | 1.88 | 2.64 | 3.16 | 89.5% / 94.5% |
| long (136 steps) | 0.824 | 0.703 | 0.701 | 1.79 | 2.39 | 2.83 | 90.4% / 97.1% |
| pooled | 0.900 | 0.853 | 0.836 | 1.90 | 2.64 | 3.19 | |

The same runs replayed with the engine's CPU drafter (`qwen4 mtp replay`: same tokens and trunk residuals; the block
fed the last 128 prompt tokens, as the chain does), pooled over the four prompts:

| drafter | tokens/round k=1 | k=2 | k=3 |
|---|---|---|---|
| reference block, full vocabulary (water + csv only) | 1.93 | 2.72 | 3.32 |
| fast drafter, full vocabulary | 1.89 | 2.62 | 3.14 |
| draft head rows below 128K, 4-bit dense matrices and head | 1.88 | 2.59 | 3.17 |
| rows below 64K, Q8 | 1.85 | 2.50 | 3.06 |
| **rows below 64K, 4-bit dense matrices** | **1.84** | **2.53** | **3.05** |
| rows below 32K | 1.81 | 2.43 | 2.87 |
| rows below 64K, 2-bit head | 1.82 | 2.46 | 2.90 |
| rows below 96K, 2-bit head with its best 32 rows rescored at Q8 | 1.88 | 2.61 | 3.17 |
| rows below 64K, 4-bit, accepted drafts keep the chained rows (no rewrite) | 1.84 | 2.50 | 2.95 |

The drafter's int8 arithmetic and 4-bit dense matrices cost nothing measurable; the draft vocabulary does (tokens
above 64K are 1-6% of the replies, markdown and code among them). The block reads only the last prompt chunk: on the
long prompt that costs 11% of the tokens per round at k = 3 (2.52 with the last 128 or 16 prompt tokens against 2.83
with the whole prompt; the chat prompts fit in one chunk).

## Drafter on the driver's CPU (`engine/src/qwen4/draft.rs`)

Each D700 has 40-80 MiB free at 32K context, so the block (1.49 GB) cannot go on a GPU; the driver's CPU (Xeon
E5-2697 v2, 12 cores, AVX without FMA, ~45 GB/s) runs it. Matrices are repacked at load: dense matrices as 4-bit codes
(IQ4_NL values, f32 scale per 32), the experts as the GGUF's IQ4 codes, the draft head (the trunk's Q6_K rows below the
draft vocabulary) as Q8_0; activations are quantized to Q8_0 per 32 and multiplied in int8 with SSSE3. Checked on incite
against the reference block on the reference prompt (`qwen4 mtp draft-check`): draft logits cos 0.9996 (Q8 matrices),
0.98-0.99 (4-bit), same argmax. One draft step at position 512 (`qwen4 mtp draft-bench`, p50 of 64, 12 threads):

| draft head | weights read | ms per step | of which head |
|---|---|---|---|
| 32K rows Q8_0 | 222 MiB | 7.5 | 2.7 |
| 64K rows Q8_0 | 312 MiB | 8.6-10.5 | 4.2 |
| 64K rows Q8_0, 4-bit dense | 273 MiB | 8.9-9.2 (8.2 with 24 threads) | 4.1 |
| 64K rows 4-bit | 193 MiB | 8.2-10.1 | 3.7 |
| 128K rows Q8_0 | 492 MiB | 11.4-12.3 | 8.0 |
| full vocabulary Q8_0 | 814 MiB | 18.4 | |

The Q8 head streams at ~45 GB/s; the 4-bit and 2-bit kernels are compute bound on this CPU (no AVX2), so a smaller
head format saves little time. The rest of a step (attention 1.6-2.4 ms, experts 1.2-1.9, mixers 0.4-0.5 each, input
projection 0.5-0.8) runs well below the memory bandwidth: about a dozen parallel loops per step, each short. Spinning
or pinned worker threads, decoding a weight block once for several tokens and four-block reductions did not change
the step time measurably and were removed.

The driver's two GPUs already hold the head rows 0-43007 (two twelfths): the final drafter scores those there while the
CPU ranks rows 43008-65535 (`Speculator::use_local_head`): DRAFT_GPU_HEAD.

## Verification on the chain

A round drafts k tokens, sends the pending token and the drafts as one batch (`forward` with `keep: verify`), and every
stage keeps what a rollback needs: the DeltaNet states and convolution windows after every token but the last (qwen35's
`qwen_conv`/`qwen_gdn` snapshots, (k) x 3.1 MiB per DeltaNet layer) and, on the driver's layer 1, the n-gram
convolution history before the batch (its new history is the old one followed by the kept tokens' normed inputs). QSA
K/V, the indexer's raw-key ring and pooled block keys are addressed by position: a block completed by a rejected token
is pooled again when it completes. The stage that ends the model returns the final hidden state of every row and keeps
the batch's residual rows (the block's input); every head slice scores all rows (row by row, then one top-k and merge
pass over all of them) and the candidates come back per row. `{"rollback": keep}` follows the round when a draft was
rejected. The batch's kernels compute each column as a single-token decode does, so the verified tokens equal greedy
decoding's (checked below).

Verification cost (`qwen4 stage-bench`, inflight, layers 0-3 with the n-gram module on GPU 0 and 4-7 on GPU 1, GPU ms
per stage at 850/1370 MHz):

| batch | GPU 0 | GPU 1 | relative to one token |
|---|---|---|---|
| 1 token (decode at 64) | 5.57 | 5.15 | 1 |
| 2 tokens | 7.69 | 7.11 | 1.38 |
| 3 tokens | 9.01 | 8.49 | 1.64 |
| 4 tokens | 10.68 | 10.13 | 1.94 |

The experts grow with every token (the per-(token, slot) kernels: 1.17 -> 2.91 ms per stage for 3 tokens), the mixers
and DeltaNet by ~45% (fused mixers read their matrices per token). On the chain a 4-token verification took 135 ms
against 68.6 ms per plain token (1.97x), a 3-token one 114 ms (1.66x).

Correctness: `qwen4 mtp spec` decodes greedily without and with drafts and requires identical ids. It passed on the CPU
path (one process and a two-process TCP chain), on incite's GPUs for layers 0-5 with the M3 Ultra's CPU for 6-47
(`mtp-hybrid`: the reference prompt and two chat prompts, 24-32 tokens), and on the six-Mac chain for every run below
(the reference prompt: 9419 0 2500 628 353 1438 488 3242 30 248046).

## End to end (six Macs)

`qwen4 mtp spec MODEL --gpus 0,1 --nodes indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
--head-split 12 --context 32768 --batch 128 --ngram disk --mtp mtp-G-iq4.gguf --draft-tokens K --draft-threads 12
--tg 192 --runs 3 water.ids csv.ids transformer.ids`: for each prompt, plain greedy decoding then speculative, three
rounds alternating; rates exclude the prompt. Run through `chain6.sh` and `tools/q4run.py` (nodes under gpuq leases,
clocks logged: 850/1370 MHz on every GPU during the runs, `chain1/*.txt`). Session 1 (`chain1/`, binary 0f-era drafter:
spinning pool, 4-bit or Q8 head on the CPU):

| run | k | plain tok/s | MTP tok/s | gain | tokens/round | draft ms/round | verify ms/round |
|---|---|---|---|---|---|---|---|
| water / csv / transformer, rewrite of accepted rows, Q8 drafter | 3 | 14.58 | 18.18 | +24.7% | 2.94 / 3.67 / 3.03 | 29.6 | 147.0 (incl. 15 ms rewrite) |
| no rewrite, 4-bit drafter | 3 | 14.55 | **19.17** | **+31.7%** | 2.98 / 3.60 / 2.88 | 29.4 | 135.1 |
| no rewrite, 4-bit drafter | 2 | 14.62 | 19.04 | +30.2% | 2.44 / 2.80 / 2.44 | 20.1 | 114.1 |
| no rewrite, 2-bit head of 96K rows rescored | 3 | 14.60 | 18.66 | +27.8% | 3.06 / 3.65 / 3.03 | 38.9 | 135.2 |

Per prompt (k = 3, no rewrite): water 18.07, csv 21.83, transformer 17.60 tok/s. Rollback 0.1-0.4 ms per round.

SESSION2

## Projections

`project.py`: tok/s = tokens per round / (k draft steps + verification of k + 1 tokens). With today's chain (68.6 ms per
token, verification 1.66x and 1.97x for 3 and 4 tokens) and a 10 ms draft step this reproduces the measured 19 tok/s.
With the fused decode kernels being worked on (35 ms per token, 3 tokens at 1.4x, 4 at ~1.65x) and the pooled tokens per
round of the final drafter (2.53 at k = 2, 3.05 at k = 3):

| draft step | k = 2 | k = 3 |
|---|---|---|
| 10 ms (today's CPU drafter) | 37 tok/s | 36 tok/s |
| 7 ms | 40 tok/s | 39 tok/s |
| 4 ms | 44 tok/s | 44 tok/s |

So ~40 tok/s needs both the faster decode/verification kernels and a draft step of ~7 ms or less.

Drafting ahead during verification (the CPU idles ~100 ms per round: continue the chain, and send the next k chained
drafts at once when every draft and the bonus token matched) was simulated on 9-deep chains (`ahead.py`): 0-5% at best
(the deep drafts are accepted less often than fresh ones), not implemented.

## Reproduction

```sh
# Mac: weights (above), CPU reference and acceptance
trashcan qwen4 mtp ref G-iq4-00001-of-00006.gguf mtp-G-iq4.gguf out --chain 3
trashcan qwen4 mtp accept G-iq4-00001-of-00006.gguf mtp-G-iq4.gguf acceptance/water.json --tokens-file water.ids --steps 256 --depth 3
python3 accept.py acceptance/*.json [--vocab 65536]
trashcan qwen4 mtp replay G-iq4-00001-of-00006.gguf mtp-G-iq4.gguf acceptance/water.json out.json --vocab 65536 [--depth 9] [--rounds]
# incite (under a lease): drafter
trashcan qwen4 mtp draft-check MODEL MTP --threads 12
trashcan qwen4 mtp draft-bench MODEL MTP --context 8192 --start 512 --steps 64 --batch 1,2 --threads 12 --vocab 65536
# one Mac: verification cost
trashcan qwen4 stage-bench MODEL --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch 3 --prefill 32 --decode 32 --start 64
# six Macs
chain6.sh mtp-chain2 40 OWNER -- mtp-chain2.sh
python3 summary.py chain2/mtp2-k3.json
```

## What is left

- Faster verification batches (the experts and mixers of 2-4 tokens; the decode-kernel work), the largest lever: a
  4-token batch costs 1.97 decode steps.
- A shorter draft step: the dense half of a step runs at a fraction of the CPU's bandwidth (a dozen short parallel
  loops); fewer, larger loops or a GPU home for the dense matrices once VRAM is freed.
- The block reads only the last prompt chunk (11% fewer tokens per round on the 3K-token document); feeding it every
  prompt row needs the residual rows of all chunks back at the driver during prefill.
- Above temperature 0 the server keeps a draft when the sampled token equals it (the output follows the target's
  distribution with the same draws); acceptance there was not measured.
