# Flash-Next MTP speculative decoding (2026-10-12)

Qwen3.8-Flash-Next ships one MTP block. The trashcan engine drafts with it on the chain driver's CPU (incite) and
verifies the pending token and its drafts as one batch through the 12 GPU stages, rolling the recurrent state back to
the accepted prefix; every round drafts as many tokens (0 to 3) as the recent acceptance pays for at the measured
costs. On the six-Mac chain with main's decode kernels (`ef4ce87`, 18.3 tok/s plain), greedy decoding of three chat
replies runs at **21.45-21.62 tok/s (+17-18%)**, output identical to plain greedy decoding; on a reply that drafting
cannot predict it costs 1.2-1.6% (fixed 3-draft rounds lost 57% there), and on a 3K-token document summary it breaks
even (+0.2-0.8%, fixed rounds -16%). Before the decode kernels the same rounds gave +39% (14.86 -> 20.66 tok/s):
those kernels made single tokens 20% cheaper but 2-4-token verification batches only 5-7%, which now cost 1.69x /
2.0x / 2.37x a plain token. The Responses server passes `scripts/test-server.py` with `--mtp`.

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
prompt. Since main sizes prefill chunks by prompt length (`Chain::chunk_size`, 16 tokens up to ~350-token prompts),
the block gets the last chunk's rows only: 7, 7, 9 and 53 rows for water, csv, transformer and long. Replayed
(`tools/replay_window.sh`, k = 3, the engine's rounds): pooled 2.935 tokens per round against 2.954 with the whole
chat prompts and 64 rows of the long one (-0.6%; water and csv unchanged, transformer 2.88 against 2.94), so the
engine does not keep rows across chunks.

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

## Adaptive rounds (`spec::Policy`)

Fixed rounds lose when drafts miss: on the reference prompt (the reply ends and the trunk repeats one token, which the
block never drafts) three drafts per round took greedy decoding from 14.6 to 6.8 tok/s (session 1) and 8.7 (session 2,
1.29 tokens per round). Every round now picks its number of drafts k (0 to `--draft-tokens`) for the most expected
tokens per millisecond:

- acceptance `a`: the share of recent drafts that matched (all depths pooled, each draft's weight decaying by 0.94
  per newer one, so about the last 16 drafts; 0.8 worth 2 drafts at the start of each sequence); a round of k drafts
  yields 1 + a + ... + a^k tokens on average;
- costs: running averages (weight 0.2 for the newest) of the measured draft step and of each verification batch size
  including its rollback, taken from the rounds themselves. Each size is run once before the first choice (k = 3, 2,
  1, 0, the first rounds after loading) and again once its cost is 256 rounds old, so sizes the choice avoids stay
  current; no cost constants, so faster decode or verification kernels move the break-even with them;
- k = 0 is a plain step through the verification path (a 1-token batch, which returns the residual row) while the
  driver runs the block on the pending token anyway once its own stages are done and the rest of the chain works
  (`Chain::verify_with`; the shadow draft: the whole 64K draft vocabulary on the CPU). Its draft is compared with the next token, so the
  acceptance stays measured without probe rounds, and the block's keys stay complete for when drafting resumes.

With session 2's costs (67.3 ms per token, verification 1.67x/1.98x, 6 ms draft steps) the best choice is k = 0
below a = 0.43, k = 1 to 0.71, k = 2 to 0.81 and k = 3 above (`project.py`); with 35 ms tokens and 1.4x/1.65x
verification: 0.37, 0.67 and 0.86.

`policy_sim.py` replays the rounds on the CPU reference's acceptance traces with session 3's costs
(`policy_sim.txt`; session 4's in `policy_sim-session4.txt`): a pooled rate against one per depth or first and
later drafts apart, memories of ~16, ~10 and ~5 drafts, priors worth 4 or 2
drafts. Every variant lands within about ±1.5% of the others on the chat and long traces (decisions near a break-even
flip on small differences), and within 1% of the best fixed depth on their mean; the reference prompt prefers short
memories and a weak prior. The engine keeps one pooled rate, a memory of ~16 drafts and a prior worth 2.

Session 5 compared that pooled rate with separate rates for first drafts and for later drafts (chained drafts are
accepted less often, and session 4 showed water running plain steps at 88% first-draft acceptance), in the same runs
with fixed 3-draft rounds (tok/s; plain 18.28 chat mean, 18.52 reference, 16.07 long):

| | reference | water | csv | transformer | chat mean | long |
|---|---|---|---|---|---|---|
| pooled rate (the engine) | **18.30** | 20.23 | 24.66 | 19.97 | **21.62** | 16.10 |
| first and later drafts apart | 17.64 | 19.30 | 24.49 | 21.31 | 21.70 | 16.29 |
| fixed 3 drafts | 7.88 | 21.13 | 24.70 | 19.78 | 21.87 | 13.53 |

The split helped transformer and lost on water, where the later-draft rate dropped and was never tested again at one
draft per round (304 rounds of k = 1); the chat mean moved 0.4% and the reference prompt went from -1.2% to -4.8%.
It was removed. Adaptive rounds stay within 1-2% of fixed 3-draft rounds on chat replies (session 4 -1.7%, session 5 -1.1%)
and far ahead on low acceptance.

## End to end (six Macs)

`qwen4 mtp spec MODEL --gpus 0,1 --nodes indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
--head-split 12 --context 32768 --batch 128 --ngram disk --mtp mtp-G-iq4.gguf --draft-tokens K --tg 192 --runs 3
water.ids csv.ids transformer.ids`: for each prompt plain greedy decoding, then speculative, three rounds alternating;
rates exclude the prompt. Run through `chain6.sh` and `tools/q4run.py` (nodes under gpuq leases, clocks logged:
850/1370 MHz on every GPU while working, `chain*/*.txt`), commands in `chain*/commands.log`.

**Session 5** (`chain5/`, commit `e0fe684` on main `ef4ce87`): the estimator comparison in
[Adaptive rounds](#adaptive-rounds-specpolicy); its pooled-rate pass is the engine as committed (session 4's
policy). Costs as in session 4: k = 0 55-57 ms, verification with rollback 89-94 / 107-114 / 126-131 ms for 2 / 3 / 4
tokens, draft steps 5.7-5.9 ms. Clocks 850/1370 MHz on every GPU while working.

**Session 4** (`chain4/`, commit `9377f34` on main `ef4ce87`, the decode kernels: single-token decode 14.7 -> 17.9
tok/s; adaptive rounds with one pooled acceptance rate, the shadow draft on the driver's main thread, the prior worth 2
drafts). Same runs as session 3:

| prompt | plain tok/s | adaptive | gain | rounds k = 0/1/2/3 | fixed 3 drafts | gain |
|---|---|---|---|---|---|---|
| reference (64 tokens, 3 runs) | 18.56 | 18.26 | -1.6% | 159/1/8/4 | 7.90 | -57.5% |
| water | 18.38 | 20.03 | +9.0% | 49/4/137/63 | 21.30 | +15.9% |
| csv | 18.35 | 24.47 | +33.3% | 0/3/5/156 | 24.31 | +32.5% |
| transformer | 18.24 | 19.85 | +8.8% | 79/2/150/44 | 19.89 | +9.0% |
| **chat mean** (192 tokens, 3 runs) | **18.32** | **21.45** | **+17.1%** | | **21.83** | **+19.1%** |
| long (3061-token prompt, 96 tokens, 2 runs) | 16.13 | **16.26** | **+0.8%** | 61/1/53/3 | 13.47 | -16.5% |

The decode kernels made a single token 20% cheaper (53.9 ms) but 2-4-token batches only 5-7%: verification with
rollback now takes 91 / 108 / 128 ms for 2 / 3 / 4 tokens (1.69x / 2.0x / 2.37x of a plain token, against 1.43x /
1.69x / 2.0x in session 3), so a round must keep more drafts to pay and the gain on chat replies falls from +39% to
+17%. A k = 0 round costs 55 ms (+1.1 ms over a plain step with the shadow draft on the main thread, about what it
cost on its own thread in session 3: the overhead is the verification path itself). On water the rounds ran 49 plain
steps while 88% of their shadow drafts were right: one rate pooled over first and chained drafts sat near the higher
break-even (chained drafts are accepted less often than first ones), which session 5 tested. Server: every
`scripts/test-server.py` check passes; `q4-turns.py` 57 tokens in 2.83 s, then 64 in 2.53 s.

**Session 3** (`chain3/`, commit `d04a8e1`, run as `85ab9a4` on main `b5b79c7`: adaptive prefill chunks, split-K prefill mixers,
expert prefill tiles; adaptive rounds with the shadow draft on its own thread and a prior worth 4 drafts). For every
prompt and run: plain greedy decoding, adaptive rounds (up to 3 drafts) and fixed 3-draft rounds (an experiment switch
for this comparison), all three with identical ids; clocks 850/1370 MHz on every GPU while working:

| prompt | plain tok/s | adaptive | gain | rounds k = 0/1/2/3 | fixed 3 drafts | gain |
|---|---|---|---|---|---|---|
| reference (64 tokens, 3 runs) | 14.89 | 14.32 | -3.8% | 148/1/19/4 | 7.51 | -49.6% |
| water | 14.84 | 19.46 | +31.2% | 2/4/180/42 | 20.34 | +37.1% |
| csv | 14.89 | 22.89 | +53.7% | 1/3/12/152 | 23.42 | +57.3% |
| transformer | 14.86 | 19.63 | +32.1% | 2/5/177/43 | 18.98 | +27.7% |
| **chat mean** (192 tokens, 3 runs) | **14.86** | **20.66** | **+39.0%** | | **20.91** | **+40.7%** |
| long (3061-token prompt, 96 tokens, 2 runs) | 13.29 | **14.33** | **+7.8%** | 27/4/73/2 | 12.98 | -2.4% |

Measured round costs (verification with rollback): k = 0 (a plain step through the verification path, with the
shadow draft) 68-71 ms against 67.2 ms per plain token, k = 1 95-98 ms, k = 2 113-114 ms, k = 3 134-135 ms, plus
5.7-5.8 ms per draft step (at the long prompt's ~3.1K positions: 77/110/129/160 ms and 6.5 ms). On the reference
prompt the reply's first 9 tokens are drafted right, then the trunk repeats a token the block never drafts (1 of 148
shadow drafts right): adaptive rounds lose 3.8% there (the k = 0 rounds' 0.8-1.2 ms over a plain step and the few
missed 2-3-draft rounds before the acceptance falls under break-even) where fixed rounds lost half. On the chat
prompts adaptive rounds are within 1.2% of fixed 3-draft rounds: water's acceptance sits where 2 and 3 drafts are
estimated within 1% of each other and it mostly drafted 2 (-4.3% against fixed), transformer +3.4%. On the long prompt
(2.26 tokens per 3-draft round, under the 3-draft break-even) adaptive rounds gain 7.8% where fixed rounds lose. Its
prefill: 5.12 and 5.06 s with MTP against 4.88 and 4.84 s plain (+0.23 s: the tail fetch and the block over the last
chunk's 53 rows on the CPU; session 2's kept last chunk cost 2.1 s).

After session 3 (measured in session 4): the shadow draft runs on the driver's main thread once its own stages are
done (`Chain::verify_with`) instead of on a thread competing with them, and the acceptance prior is worth 2 drafts
(`policy_sim.py`: reference 14.41 -> 14.60 tok/s simulated, text unchanged).

Server (`tools/q4serve-test.sh ... --mtp mtp-G-iq4.gguf --draft-tokens 3`, adaptive rounds): every
`scripts/test-server.py` check passes; `q4-turns.py` turn 1 57 tokens in 2.98 s, turn 2 (78 of 99 prompt tokens
reused) 64 tokens in 2.82 s.

**Session 2** (`chain2/`, commit `1ce750b`, run as `c4ac1fe` on main `90cb371`, the engine's drafter: 4-bit dense, Q8 head rows 43008-65535 on the CPU and
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

**Session 1** (`chain1/`, commit `64f6870`, run as `ff50ba6` on main `da28c66`: the whole draft head on the CPU, 12 threads; the variants compared):

| variant | k | plain tok/s | MTP tok/s | gain | draft ms/round | verify ms/round |
|---|---|---|---|---|---|---|
| rewrite of accepted rows, Q8 drafter | 3 | 14.58 | 18.18 | +24.7% | 29.6 | 147.0 (15 ms rewrite) |
| no rewrite, 4-bit drafter | 3 | 14.55 | 19.17 | +31.7% | 29.4 | 135.1 |
| no rewrite, 4-bit drafter | 2 | 14.62 | 19.04 | +30.2% | 20.1 | 114.1 |
| no rewrite, 96K 2-bit head rescored | 3 | 14.60 | 18.66 | +27.8% | 38.9 | 135.2 |


## Projections

`project.py`: tok/s = tokens per round / (k draft steps + a (k + 1)-token verification), with the pooled tokens per
round of the engine's rounds (1.84 / 2.50 / 2.95 for k = 1 / 2 / 3, the long prompt included; the chat prompts alone
give ~3.1 at k = 3), 6 ms draft steps:

| chain | plain | k = 1 | k = 2 | k = 3 |
|---|---|---|---|---|
| session 2 (67.3 ms per token, verification 1.34x / 1.67x / 1.98x) | 14.9 | 19.2 | 20.1 | 19.5 |
| session 4, decode kernels (53.9 ms, 1.69x / 2.0x / 2.37x) | 18.6 | 19.0 | 20.8 | 20.2 |
| 35 ms per token, 2 / 3 / 4 tokens at 1.2x / 1.4x / 1.65x | 28.6 | 38.3 | 41.0 | 38.8 |
| 35 ms per token, 4 ms draft steps | 28.6 | 40.0 | 43.9 | 42.1 |

~40 tok/s needs single tokens near 35 ms and verification batches that grow by ~20% of a token per extra token
(today: 37 ms for the second token, 17-20 ms for the third and fourth). The per-acceptance break-even moves with the
costs: with session 4's, drafting pays above a per-draft acceptance of 0.72 (2 drafts from there, 3 above 0.81; one
draft never pays); with the 35 ms row, above 0.37.

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
# six Macs (sessions 3-5: plain, adaptive and fixed rounds per prompt; tools/fetch_session.sh copies the results)
chain6.sh mtp-chain2 40 OWNER -- tools/mtp-chain2.sh; python3 summary.py chain2/mtp2-k3.json chain2/mtp2-k2.json
chain6.sh mtp-chain4 35 OWNER -- tools/mtp-chain4.sh; python3 summary.py chain4/mtp4-{gate,k3,long}.json
python3 project.py
# Mac: the block's window under main's prefill chunks; the policy on the acceptance traces
tools/replay_window.sh
python3 policy_sim.py --gate chain3/mtp3-gate.json acceptance/{water,csv,transformer,long}.json
VERIFY=55,91,108,128 DRAFT=5.7 PLAIN=53.9 python3 policy_sim.py --gate chain4/mtp4-gate.json acceptance/{water,csv,transformer,long}.json
```

## What is left

- Cheaper 2-4-token verification batches (1.69x / 2.0x / 2.37x a plain token since the decode kernels): the largest
  lever; another agent works on those kernels, and the adaptive rounds pick the costs up as measured.
- A k = 0 round costs ~1.1 ms more than a plain step (the verification path's residual rows and copies); a plain step
  that also returns the residual row would remove it.
- A shorter draft step: the dense half runs well below the CPU's bandwidth (about a dozen short parallel loops); the
  block's attention grows with the context (~5 ms at positions 2048-4096 against ~1.6 at 512) and needs threaded
  block scoring past 2K.
- The block reads only the prompt's last chunk (16-64 rows under main's chunk sizes; -0.6% tokens per round on the
  four prompts, 11% fewer on the 3K-token document than with its whole prompt).
- Above temperature 0 the server keeps a draft when the sampled token equals it (the output follows the target's
  distribution with the same draws as plain decoding); acceptance there was not measured.
