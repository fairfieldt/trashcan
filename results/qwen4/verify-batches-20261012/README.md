# Flash-Next 1-4-token batches: mixers and Q8_0 matrix kernels (2026-10-12/13)

**Result.** Against main 4307169 on the six-Mac chain (session 2), MTP speculative decoding of the chat prompts
(water, csv, transformer; `qwen4 mtp spec --draft-tokens 3`, one run per prompt and binary, alternated) goes from
**23.00 to 26.97 tok/s (+17.3%)**, its gain over plain decoding from +19.0% to +33.4%, with output identical to plain
greedy decoding (main's suite for 4307169, results/qwen4/main-suite-20261013: 23.11). Verifying 2 / 3 / 4 tokens
takes 66-69 / 83-84 / 102-105 ms instead of 86 / 101 / 119-121: 1.36x / 1.69x / 2.09x of a plain decode step
(49.5 ms) instead of 1.66x / 1.95x / 2.32x (51.7 ms). Plain decode (`qwen4 bench --pp 512 --tg 128`, one
alternated pair) goes from 18.89 to 19.96 tok/s (main's suite: 19.05), incite's GPU time per token from 7.75 to
7.32 ms; prefill is unchanged (452 and 458 tok/s on 512 tokens; main's suite 458). The final tree, rebased onto main
2293550, repeats this (session 4): MTP 23.42 -> 27.08 tok/s (+15.6%), plain greedy 19.50 -> 20.32 tok/s, 2 / 3 / 4
tokens verified in 67 / 82-83 / 104-108 ms against 85-86 / 100 / 118-120, the 14-token validation bit-identical to
session 2's. Session 1, against main 4c1346d with three runs per prompt and binary (before the out projections' K
split): MTP 21.73 -> 25.11 tok/s, plain decode 18.11 -> 18.63.

One Mac (invite, layers 0-7, session 9), GPU ms of the 8 layers:

| tokens | main 4307169 | this | main, / one token | this, / one token |
|---|---|---|---|---|
| 1 (decode at 64) | 7.91 | **7.56** | 1 | 1 |
| 2 | 12.60 | **9.80** | 1.59 | **1.30** |
| 3 | 15.13 | **12.23** | 1.91 | **1.62** |
| 4 | 17.89 | **15.07** | 2.26 | **1.99** |

The targets (at most 1.15x for 2 tokens, 1.3x for 3) hold for the parts this work changed, not for whole batches:
the experts, which every token still reads for itself (the small-batch experts work), add 1.35-1.37 ms per extra token
to the 8 layers (1.83 ms for one token, 5.89 for four: 39% of a 4-token batch). Without the experts' growth a batch
of 2 / 3 / 4 tokens costs 1.12x / 1.26x / 1.46x of one token (main 1.42x / 1.57x / 1.75x).

Four kernel changes for Q8_0 batches of 1-8 tokens, each about reading a matrix once for every token of a batch and
not computing columns that are not there:

- the mixer's second half reads its 10240 x 320 matrix once per batch instead of once per token
  (`qwen4_mix_rows_up`; one token 42 against 51 µs);
- the mixer's first half is split by stream, with 8-byte loads and exact column counts (`qwen4_mix_down_skw`; one
  token 48 against 51 µs, 2-4 tokens 32-35% faster);
- the Q8_0 matrix-vector kernels get exact 2- and 3-column variants that decode each row's weights once for all
  columns (`qwen4_q8_mvw4_r4`/`_r2` `_c2`, `_c3`);
- the 2560 x 6144 out projections split each row's K over two waves (`qwen4_q8_mvk2`/`_mvk4`; one token 102 against
  121 µs, 2-4 tokens 122-192 against 202-249).

The branch had also aligned repacked Q4_0 and IQ4_XS blocks (for the lane-per-row expert kernels) and arena buffers
(for `qwen4_mix_rows_up`'s line loads) to 64 bytes; without them the kernels give the same bits in the same time
(session 10), so the final tree leaves both out.

## What the extra tokens cost (main, one Mac)

Single-Mac `stage-bench` on invite: layers 0-3 with the n-gram module on GPU 0 and 4-7 on GPU 1, 32K context, the
B-token chunk at position 0 (the verification-shaped batch, as in results/qwen4/mtp-20261012) against single tokens
decoded at position 64, `TRASHCAN_Q4_PROFILE=1` for per-dispatch times (`tools/sb.sh`; `tools/prof.py`, `tools/cmp.py`
for the tables). GPU ms of the 8 layers on main 4307169 (session 9):

| batch | main | / one token | mixers | DeltaNet | router | experts |
|---|---|---|---|---|---|---|
| 1 | 7.91 | 1 | 1.70 | 2.61 | 0.48 | 1.83 |
| 2 | 12.60 | 1.59 | 3.06 | 3.91 | 0.52 | 3.21 |
| 3 | 15.13 | 1.91 | 3.99 | 4.11 | 0.52 | 4.54 |
| 4 | 17.89 | 2.26 | 4.87 | 4.48 | 0.53 | 5.88 |

Per dispatch (DeltaNet layer 5; µs for 1 / 2 / 3 / 4 tokens), the growth came from:

| dispatch (main) | 1 | 2 | 3 | 4 | why it grew |
|---|---|---|---|---|---|
| mixer first half `qwen4_mix_down_sk_q8`, 2-8 tokens `_sk4_q8` (each of two) | 51 | 95-98 | 122-127 | 154-156 | the matrix is read once, but each 4-row wave re-reads and recombines the residual for every token (8 lanes per block, 4 weights each), and VALU and loads grow with the tokens |
| mixer second half `qwen4_mix_up_q8` (each of two) | 51 | 91-92 | 123 | 148-149 | a workgroup row per token: the 3.5 MB matrix is read once per token |
| qkv (10240 x 2560) and z (6144 x 2560), one group (the qkv dispatch's time spans both) | 242 | 366 | 373 | 380 | `_c4` kernels: 2 and 3 tokens cost 4 columns |
| out projection (2560 x 6144) | 121 | 202 | 215 | 249 | the same, and 2 rows per wave: per weight byte a lane loads 4 input bytes per column, against 2 at 4 rows |
| experts (gate/up, down) | 141, 85 | 248, 153 | 356, 211 | 463, 273 | per (token, slot) (not part of this work) |
| `qwen4_gdn` | 48 | 57 | 78 | 99 | the recurrence walks the tokens (not part of this work) |

## Changes

| kernel | what it does | layer 5, µs (1 / 2 / 3 / 4 tokens), main -> this |
|---|---|---|
| `qwen4_mix_rows_up` (new: the mixer's second half for Q8_0 with a bottleneck of 320 and 4 streams; 1-4 tokens per workgroup) | a lane per row: one workgroup per 16 elements of every stream (64 rows), wave w of 5 takes line w (64 columns) of the rows, so the workgroup reads 64 whole consecutive rows (20 KB) once for all its tokens; the bottleneck values through scalar loads; the waves' sums meet in LDS; then `qwen4_mix_up`'s epilogue (the pending combine stored in place, read gates, stream mean) with its inputs loaded together with the weights | 51 -> 41-42 / 91-92 -> 45-48 / 123 -> 50-52 / 148-149 -> 56-58 |
| `qwen4_mix_down_skw` (new: the mixer's first half for Q8_0 with 4 streams, 1-4 tokens per workgroup, 1-8 tokens) | `_sk`'s split with a finishing workgroup, split by stream instead (4 chunks of 80 blocks) with four lanes per block loading 8 weights (16 blocks of a row per load); exact column counts. One layout at every batch size, so a token's results are the same bits in a batch of any size (dumps below) | 51 -> 47-48 / 95-98 -> 65 / 122-127 -> 82-83 / 154-156 -> 102 |
| `qwen4_q8_mvw4_r4`/`_r2` with 2 and 3 columns (`_c2`, `_c3`; the column count is a specialization constant) | exact column counts; with several columns the inputs are loaded first and each row's weights decoded once for all of them (64 VGPRs for 3 columns against 84 for the old loop order). Single tokens keep the old loop | qkv and z group: 242 (unchanged) / 366 -> 263 / 373 -> 298 / 380 -> 362 |
| `qwen4_q8_mvk2`/`_mvk4` (new: matrices of 2048-4095 rows of at least 4096, i.e. the 2560 x 6144 out projections of the DeltaNet and QSA layers, 1-8 tokens) | a workgroup of 4 waves: 2 groups of rows (2 rows per wave for one token, 4 from 2 tokens, whose input loads grow with the columns) times 2 halves of K, the halves added through LDS; loads and arithmetic per lane as `qwen4_q8_mvw4`'s. Twice the waves of `_mvw4_r2`, half as long (rows of 6144 are 6.5 KB). Used at every batch size, so a token's results stay the same bits in batches of 1, 2 and 4, as on main; at 4 tokens it is slower than `qwen4_q8_mvw4_r4_c4` would be (190 against 173 µs, session 7) | 121 -> 102 / 202 -> 122 / 215 -> 151 / 249 -> 192 |

QSA layer 7: the q group (12288 x 2560, with the k and v projections of 512 x 2560) 196 (unchanged) / 329 -> 212 /
328 -> 256 / 331 -> 309 µs, the k and v projections themselves 42 -> 26 and 32 / 41 -> 28 and 36 / 42 -> 34 and 42,
the out projection 120 -> 102 / 204 -> 124 / 213 -> 145 / 252 -> 190.

Phases (8 layers, GPU ms; [profile-final.txt](profile-final.txt)): at 3 tokens mixers 3.99 -> 2.21, DeltaNet (with
its projections) 4.11 -> 3.30, QSA projections 0.69 -> 0.55, attention (with the out projection) 0.50 -> 0.38,
n-gram module 0.38 -> 0.32, router and experts unchanged; for one token mixers 1.70 -> 1.51, DeltaNet 2.61 -> 2.49,
attention 0.36 -> 0.32.

## Correctness

| check | result |
|---|---|
| one Mac, layers 0-7, the 96-token prompt (`stage-bench --tokens-file --out`, `tools/dump.sh`): this in chunks of 1 and of 4 against main's single tokens (session 9, main 4307169) | every tensor within rel 5.7e-6 ([compare-p96-chunks-b1-vs-main.txt](compare-p96-chunks-b1-vs-main.txt), [compare-p96-chunks-b4-vs-main.txt](compare-p96-chunks-b4-vs-main.txt)) |
| the same, chunks of 4 against chunks of 1 (a verification batch against the same tokens one at a time) | the same bits (sessions 8, 9 and 10; chunks of 2 as well in session 7, with the K split behind its switch) |
| chunks of 3 against chunks of 1 (session 8, the same kernels on main 7da6f93) | within 2.8e-6, from QSA alone (the attention partition count follows the chunk's end position: main's chunks of 3 differ the same way, 3.9e-6 in session 4, [compare-p96-main-chunks-b3-vs-main-chunks-b1.txt](compare-p96-main-chunks-b3-vs-main-chunks-b1.txt)); 5.4e-6 against main's single tokens |
| six Macs, `qwen4 validate`, the 14-token reference prompt in chunks of 4 (sessions 2 and 4) | greedy 9419 0 2500 628 353 1438 488 3242 30 248046 (the reference's); logits rel 2.82e-4 to the CPU reference (main 2.78e-4), 2.24e-5 to main; session 4 (the final tree on main 2293550) the same bits as session 2 (`trashcan-vfy10` on 4307169) |
| six Macs, p100 (sessions 2 and 4) | the same greedy tokens as main; logits rel 3.48e-2 to the CPU reference (main the same), 4.6e-6 / 6.2e-6 to main |
| six Macs, `qwen4 mtp spec` (sessions 1, 2 and 4) | greedy output with drafts identical to plain greedy decoding in every run (`all identical: True`) |

[compare-chain4-validate.txt](compare-chain4-validate.txt), [compare-chain2-validate.txt](compare-chain2-validate.txt),
[compare-chain1-validate.txt](compare-chain1-validate.txt) (session 1: the 14-token prompt as one chunk and in chunks
of 3, logits 2.84e-4 / 2.78e-4 to the CPU reference; the "worst relative error ... 9.38e-01" lines against the CPU
references are the same for main).

## End to end (six Macs)

The sessions ran under chain6.sh with results/qwen4/chain-20261012/tools/q4run.py (chain runs only), `--gpus 0,1
--nodes indite,inflight,alight,invite,enlight --head-split 12 --context 32768 --batch 128 --ngram disk`, the driver
on incite; 850/1370 MHz on every GPU while computing, at most 71 C (`chain*/*/clocks/`, `tools/clocksum.py`).

Session 4 ([chain4/](chain4/), `tools/chain4.sh`, q4run.py of main cca44358): `trashcan-vfy13` (the final tree:
this branch on main 2293550, sha256 8dd6060c...) against `trashcan-2293550`, alternated, one run each: validations of
both binaries and one MTP pass per binary (the plain greedy pass of each MTP run is the decode check). Every node
bound its port and loaded its layers (`tools/post4.sh`).

| | main 2293550 | this |
|---|---|---|
| MTP chat, plain greedy pass (192 tokens per prompt) | 19.50 tok/s | 20.32 (+4.2%) |
| **MTP chat, adaptive rounds (k up to 3)** | **23.42 tok/s** (+20.1%) | **27.08 tok/s** (+33.3%) |
| verification ms (with rollback) for k = 0 / 1 / 2 / 3 drafts | 52-54 / 85-86 / 100 / 118-120 | 53 / 67 / 82-83 / 104-108 |
| rounds by k (water; csv; transformer) | 3/2/48/21; 0/1/1/52; 16/0/57/12 | 1/12/66/1; 0/0/66/1; 0/33/56/0 |

| prompt | main plain | main MTP | this plain | this MTP |
|---|---|---|---|---|
| water | 19.52 | 22.30 (+14.2%) | 20.30 | 26.37 (+29.9%) |
| csv | 19.48 | 26.37 (+35.4%) | 20.28 | 30.03 (+48.1%) |
| transformer | 19.50 | 21.58 (+10.7%) | 20.37 | 24.85 (+22.0%) |

Session 3 ([chain3/](chain3/)) is not a measurement: another session's stalled run had left `trashcan-2293550` nodes
listening on indite, inflight, invite and enlight, so this branch's nodes there exited with "Address already in use"
and its driver ran against main's nodes (`chain3/*/node-*.log`); its base runs did not start. q4run.py now refuses to
start over a stray engine or an open node port (main cca44358).

Session 2 ([chain2/](chain2/), `tools/chain2.sh`): `trashcan-vfy10` (this branch on main 4307169, sha256 55f68c0a...;
the final tree without the alignment changes gives the same bits in the same time, session 10) against
`trashcan-4307169` (d3ea125b...), alternated, one run each as the testing policy asks for a change that targets MTP:
validations of both binaries, one plain-decode pair (a no-regression check), one MTP pass per binary.

| | main 4307169 | this |
|---|---|---|
| plain decode, `bench --pp 512 --tg 128` (long.ids) | 18.89 tok/s | **19.96** (+5.7%) |
| prefill pp512 | 452.0 tok/s | 458.0 |
| incite's GPU ms per decode token (two stages) | 4.00 + 3.75 | 3.78 + 3.54 |
| MTP chat, plain greedy pass (192 tokens per prompt) | 19.34 tok/s | 20.22 |
| **MTP chat, adaptive rounds (k up to 3)** | **23.00 tok/s** (+19.0%) | **26.97 tok/s** (+33.4%) |
| verification ms (with rollback) for k = 0 / 1 / 2 / 3 drafts | 53-54 / 86 / 101 / 119-121 | 49-51 / 66-69 / 83-84 / 102-105 |
| rounds by k (water; csv; transformer) | 2/2/49/21; 0/1/0/53; 19/0/55/14 | 1/31/44/9; 0/1/32/29; 2/12/63/2 |

| prompt | main plain | main MTP | this plain | this MTP |
|---|---|---|---|---|
| water | 19.42 | 22.02 (+13.4%) | 20.08 | 25.15 (+25.2%) |
| csv | 19.31 | 26.12 (+35.2%) | 20.32 | 29.10 (+43.2%) |
| transformer | 19.28 | 20.88 (+8.3%) | 20.26 | 26.66 (+31.6%) |

The policy (`spec::Policy`, measured costs) now runs 1- and 2-draft rounds where it ran plain steps or 3 drafts
before (transformer: 19 plain rounds -> 2; csv: 53 three-draft rounds -> 32 two-draft and 29 three-draft ones).

Session 1 ([chain1/](chain1/), `tools/chain1.sh`): `trashcan-vfy5` (this branch's first three kernels on main
4c1346d, sha256 3d378c79...) against `trashcan-4c1346d`, three runs each.

| | main 4c1346d | this |
|---|---|---|
| plain decode, `bench --pp 512 --tg 128` (long.ids) | 18.07, 18.15 tok/s (18.11; a third run did not start: the enlight node refused the connection) | 18.66, 18.61, 18.61 (**18.63**, +2.9%) |
| prefill pp512 | 370.5 tok/s | 369.5 |
| incite's GPU ms per decode token (two stages) | 4.22 + 4.01 | 4.10 + 3.88 |
| MTP chat, plain greedy pass (192 tokens, 3 runs x 3 prompts) | 18.34 tok/s | 18.79 |
| **MTP chat, adaptive rounds (k up to 3)** | **21.73 tok/s** (+18.5%) | **25.11 tok/s** (+33.7%) |
| verification ms (with rollback) for k = 0 / 1 / 2 / 3 drafts | 55 / 90-91 / 108-110 / 127-129 | 54-55 / 73-75 / 91-94 / 109-111 |

## Sessions

One-Mac sessions ran on invite under one-Mac gpuq leases (`tools/run-lease.sh MACHINE TICKET SCRIPT`), 850/1370 MHz
while computing ([invite/clocks](invite/clocks)): `tools/sb.sh 64 16 8 NAME=BINARY:BATCH[:ENV=VALUE]` runs
`TRASHCAN_Q4_PROFILE=1 scripts/with-clocks.sh ... trashcan qwen4 stage-bench G-iq4 --ranges 0:4,4:8 --gpus 0,1
--context 32768 --batch B --head-rows 0:21504,21504:43008 --decode 16 --start 64 --prefill 8`, and `tools/dump.sh`
the same with `--tokens-file vfy-p96.ids --out DIR` (no profile).

| session | what |
|---|---|
| 1 (`tools/s1.sh`) | main 4c1346d profiled at 1-4 tokens; lane-per-row Q8_0 and expert kernels (removed) |
| 2, 3 | mixer and dense variants behind temporary switches |
| 4, 5 | the cleaned-up kernels; `skl`, `w0` (removed); session 5 the kernels of chain session 1 |
| 6, 7 | a separate finishing pass, 2 rows per wave for the first half (removed); the out projections' K split |
| 8 | this branch with the K split on main 7da6f93 (`trashcan-vfy9`), dumps of chunks of 1, 3 and 4 |
| 9 | this branch on main 4307169 (`trashcan-vfy10`) against `trashcan-4307169`: the tables above, dumps |
| 10 | the final tree's alignment: `trashcan-vfy11b` (no alignment change) and `trashcan-vfy11a` (arena buffers on 64-byte boundaries) against `trashcan-vfy10` (those and Q4_0 / IQ4_XS repacks padded to 64 bytes), alternated twice at 3 tokens, and dumps ([compare-s10-padding.txt](compare-s10-padding.txt)) |

## Tried and removed

Temporary switches on stage-bench; sources of the removed kernels in [removed/](removed/) (the experiment patch of
session 6 as `vfy7-experiments.patch`), per-dispatch numbers in `invite/vfy-s1..s7-*.json` and `invite/vfy-s10-*.json`.

| candidate | result |
|---|---|
| lane-per-row Q8_0 matrix kernels for 1-4 tokens (`qwen4_q8_rows*`: a lane per row walking 64-byte lines over a K segment, segments over the waves of a workgroup summed in LDS, inputs through scalar loads) | much slower: one token 400 against 242 µs (qkv group), 3 tokens 411-440 against 215-370; rows 2.5 KB apart, a different 64-byte line per lane, stream at ~70 GB/s (session 1) |
| lane-per-row experts (`qwen4_moe_gu_rows`, `qwen4_moe_down_rows`) | slower: 3 tokens 690 / 347 against 451 / 216 µs (session 1) |
| `qwen4_mix_prep` (combine, gamma * x, sums of squares) + lane-per-row first half `qwen4_mix_rows_down` | first half 73 / 105 / 137 / 159 µs against 50 / 96 / 123 / 150 (session 1) |
| `qwen4_gdn` loading the next token's inputs while computing one | no change (session 1) |
| Q8_0 kernels loading the next iteration before computing the current one (`_p`) | not faster (one token 2-3% slower, 4 columns 15% slower: 128 VGPRs), and its results were wrong (not pursued) |
| 8 rows per wave for 2-4 columns; 2 rows per wave for the large matrices at 2-4 columns | slower: qkv group at 3 columns 355 / 393 against 296 µs |
| single-token kernels with every iteration's loads issued first (1, 2 or 4 rows per wave) | 2 rows: qkv group 237 against 242 µs, z 108 against 115; 1 and 4 rows and the out projection slower or equal; not kept for 2% of one group |
| `qwen4_mix_rows_upf`: a lane per whole row (320 weights in 20 loads), a wave per stream | 65 µs against 41 for one token (128 VGPRs, 160 waves) |
| first half with chunks of 16 blocks (8 or 4 lanes per block) | 69-71 µs for one token: the finishing pass grows with the chunks (20 partials per row, 420 workgroups on the counter) |
| first half without a finishing pass (diagnostic) | 43 against 50 µs: the pass and its counter cost ~7 µs |
| `qwen4_mix_down_skl`: the four streams in one workgroup, sums through LDS, no finishing pass | the same for one token, slower for 2-4 (128 VGPRs): 11.05 / 14.22 / 16.86 against 10.86 / 13.97 / 16.53 ms (session 4) |
| `qwen4_mix_rows_up` loading the epilogue's inputs in the first wave only | 0.08 ms slower per 8 layers (session 4) |
| first half with 2 rows per wave (more, shorter waves) | 57 / 109 against 48 / 84 µs for 1 / 3 tokens: slower (session 6) |
| first half's finishing pass as its own dispatch (`qwen4_mix_down_fin`) | 42 + 7.4 against 48 µs for one token; 68 + 8 against 84 for three (the in-kernel pass walks the tokens in one workgroup): the same sums, 8 µs per mixer only for 2-4 tokens, ~1% of a verification; not kept (session 6) |
| 2 rows per wave with a separate finishing pass | 51 + 7.4 / 92 + 8 µs: slower (session 6) |
| out projections with 2 rows per wave at every batch size (`k22`) | 142.5 / 201 / 255 against 124 / 151 / 190.5 µs for 2 / 3 / 4 tokens (session 7) |
| `qwen4_q8_mvw4_r4_c4` for the out projections at 4 tokens | 173 against 190.5 µs (0.1 ms per Mac per 4-token batch), but a token's results would differ between batches of 1 and 4; not taken |
| 64-byte alignment of repacked Q4_0 and IQ4_XS blocks (for the lane-per-row experts) and of arena buffers (for `qwen4_mix_rows_up`'s line loads) | no difference: 8 layers 12.21-12.23 ms at 3 tokens and 7.54-7.56 for one token with both, with the arena alignment alone and with neither, `qwen4_mix_rows_up` 41-42 µs for one token in all three, the same bits (session 10) |

## What is left

- The experts (39% of a 4-token batch, +1.35-1.37 ms per extra token and 8 layers) and `qwen4_gdn` (+20 µs per token and
  DeltaNet layer) grow with every token; the DeltaNet alpha/beta projections and the indexer (`qwen4_matf32_c4`)
  compute 4 columns for 2 and 3 tokens (alpha/beta 29 -> 35-41 µs; not part of this work).
- The head scores the rows of a verification batch one after another (`score_head`, ~0.3 ms per row per slice).
- The mixers are still far from the bandwidth for one token: 47-48 + 41-42 µs per mixer for 7 MB (~27 µs at
  263 GB/s), 1.5 of the 7.6 ms of 8 layers. Both halves are latency-bound (336 and 800 short waves, a finishing
  pass).
- Single-token Q8_0 kernels reach ~184 GB/s on the qkv and z group (44.6 MB in 242 µs) and ~164 GB/s on the out
  projections (138 before).

## Files

- `invite/`: single-Mac stage-bench JSON and stderr per session (`vfy-s1-*` ... `vfy-s10-*`; `new` / `base` this
  branch and main), `clocks/`.
- `chain1/` ... `chain4/`: six-Mac driver outputs, node logs and clocks per Mac, `commands.log` (`chain3/` invalid,
  above).
- `compare-*.txt`: activation comparisons (above); `profile-final.txt` (session 9: walls, phases and per-dispatch
  tables at 2, 3 and 4 tokens), `profile-s1.txt` (main 4c1346d and the session-1 kernels).
- `removed/`: removed kernels and the session-6 experiment patch.
- `tools/`: `s1.sh` ... `s10.sh` (single-Mac sessions), `chain1.sh` ... `chain4.sh` (six-Mac sessions),
  `precheck.sh` (binaries and inputs on all six Macs), `sb.sh`, `dump.sh`, `run-lease.sh`, `deploy.sh`, `gather.sh`,
  `post3.sh`, `post4.sh`, `leftovers.sh`, `procs.sh` (runs, copies, stray processes), `prof.py`, `cmp.py`,
  `compact.py`, `benchsum.py`, `clocksum.py`, `valcmp.sh` ... `valcmp4.sh`, `cmpdumps.sh`, `compares.sh`, `s10cmp.sh`
  (tables; the MTP tables from results/qwen4/mtp-20261012/summary.py), `vmstats.sh`, `stats.py`, `isa.py`,
  `loopmix.py` (RADV shader statistics and programs offline: the kbuild VM has Mesa 25.0.7 as the Macs, and
  `RADV_FORCE_FAMILY=tahiti` compiles every pipeline for the D700 without a GPU).
