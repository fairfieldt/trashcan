# Qwen3.8-27B TP8, round 2 (2026-10-09): matvec geometry, exchange instrumentation, host passes

**Result.** Qwen3.8-27B TP8 over the six-cable cluster decodes at 40.7 tok/s (record 37.1 in the same session, +9.6%)
and 51.7 with MTP (50.5, +2.4%), with the same tokens, from three kernel changes that keep every bit: two rows per
wave in the one-column projections (`matvecq4_r2`, `matvecq4_r2u2`), one state row per wave in the DeltaNet
recurrence, and a two-key attention partial with a 256-thread unrolled merge. GPU time per decode token 19.4 → 17.3 ms.
Without rounding the all-reduced decode totals (`TRASHCAN_TP_ROUND=0`) decode reaches 42.2 (+13.9%), at the cost of
decode rows no longer matching verification columns bit for bit (the checks still pass, with a column score error of 3.2e-3 instead of 0; the switch stays, rounding is the default). The all-reduce's parts are now
measured; a worker thread per cable, an AVX build, fused host passes, other wave geometries and signed-nibble
kernels did not help and are gone. The three-column verification kernels (64 VGPRs, 4 waves per SIMD) bound MTP.

## Setup

- **Source.** This round's binaries are built from the working tree on top of a25cdd0 (`main`); the record's binary is
  `trashcan-final` (a25cdd0's engine, results/qwen/tp8-mesh-20261009, sha256 `d13b69ff…`). Cross-built with
  `scripts/cross-build.sh` into `~/trashcan-exp/tp8opt/` on incite, invite, enlight and indite:
  `trashcan-r2` (kernel switches, exchange instrumentation, fused decode of the f16 total), `trashcan-r3` (r2 with the
  norm pass as before), `trashcan-r2avx` (r2 built with `-C target-cpu=ivybridge`), `trashcan-r4` (DeltaNet and
  attention-partial switches), `trashcan-r5` (fused decode all-reduce passes), `trashcan-r6` (signed-nibble kernel
  switch).
- **Machines and cables.** The six-cable cluster of results/qwen/tp8-mesh-20261009 (full-mesh all-reduces, f16
  payload below prefill size), kernel 7.2.9-p2p, streams as in that round's `streams.txt`. gpuq leases on all four
  (incite #133, invite #121, enlight #9, indite #7), the performance governor and the 0 µs latency request on all four
  (`tools/round2.sh`, which runs `../tp8-mesh-20261009/tools/tp8full.sh`); every card at 850/1370 MHz while computing
  (`incite/clocks/`). One smoke run (`r2-smoke`, not counted) came first.
- **Method.** Alternating single runs of `qwen-bench --pp 512 --tg 128 --reps 2` (with `--mtp` for MTP), two runs a
  side; tok/s per repetition; GPU wait and the exchange parts are incite's milliseconds per generated token
  (`tools/phases.py`, `../tp8-opt-20261008/tools/pairs.py`). The microbenchmarks ran on enlight's GPU 0 under their
  own leases (`microbench/`).

## Microbenchmarks on the D700 (`trashcan qwen-matvec-bench`, enlight)

Random Q4_0 weights at the 27B's TP8 rank shapes, 50 dispatches per timing, median of five; every variant keeps
`matvecq4`'s lane-to-block mapping, bias, FMA chain and reduction per row, and every output was bit for bit the
reference's. `rN` is N rows per wave, `wM` M waves per workgroup (matvecq4: 4 and 4, 16 rows per workgroup), `u2`
issues the loads of two blocks before using them. "g2" times two dispatches sharing one barrier, as a layer's
independent projections run.


**one column, one dispatch per barrier** (`matvec-n1-g1.jsonl`), µs per dispatch group:

| shape | matvecq4 | matvecq4_r2 | matvecq4_r2w2 | matvecq4_r4w1 | matvecq4_r1 | matvecq4_r4u2 | matvecq4_r2u2 |
|---|---|---|---|---|---|---|---|
| 2176x5120 | 51.9 | 46.4 (-11%) | 48.6 (-6%) | 49.5 (-4%) | 49.5 (-4%) | 50.1 (-3%) | 44.6 (-14%) |
| 5120x2176 | 51.4 | 46.4 (-10%) | 46.6 (-9%) | 50.8 (-1%) | 51.7 (+0%) | 56.6 (+10%) | 47.5 (-8%) |
| 1280x5120 | 35.8 | 29.2 (-18%) | 31.9 (-11%) | 35.8 (+0%) | 33.2 (-7%) | 36.3 (+1%) | 28.2 (-21%) |
| 768x5120 | 28.3 | 22.7 (-20%) | 24.5 (-13%) | 28.2 (-0%) | 22.1 (-22%) | 28.8 (+2%) | 21.5 (-24%) |
| 5120x768 | 26.9 | 23.7 (-12%) | 24.3 (-10%) | 26.4 (-2%) | 26.7 (-1%) | 37.9 (+41%) | 31.4 (+17%) |
| 1536x5120 | 38.7 | 33.8 (-13%) | 34.5 (-11%) | 37.7 (-3%) | 36.8 (-5%) | 37.5 (-3%) | 31.9 (-18%) |
| 256x5120 | 23.4 | 16.9 (-28%) | 17.5 (-25%) | 23.3 (-0%) | 15.2 (-35%) | 24.2 (+4%) | 15.7 (-33%) |

**one column, two dispatches per barrier (a layer's independent projections)** (`matvec-n1-g2.jsonl`), µs per dispatch group:

| shape | matvecq4 | matvecq4_r2 | matvecq4_r2w2 | matvecq4_r4w1 | matvecq4_r1 | matvecq4_r4u2 | matvecq4_r2u2 |
|---|---|---|---|---|---|---|---|
| 2176x5120 | 86.1 | 78.5 (-9%) | 80.3 (-7%) | 85.7 (-0%) | 95.8 (+11%) | 82.9 (-4%) | 77.4 (-10%) |
| 5120x2176 | 85.9 | 82.4 (-4%) | 83.2 (-3%) | 85.7 (-0%) | 95.9 (+12%) | 93.6 (+9%) | 83.0 (-3%) |
| 1280x5120 | 53.8 | 50.7 (-6%) | 52.9 (-2%) | 53.3 (-1%) | 57.0 (+6%) | 51.7 (-4%) | 49.5 (-8%) |
| 768x5120 | 38.5 | 33.6 (-13%) | 34.3 (-11%) | 37.8 (-2%) | 36.4 (-6%) | 37.9 (-2%) | 31.7 (-18%) |
| 5120x768 | 42.1 | 40.0 (-5%) | 40.6 (-3%) | 41.1 (-2%) | 46.8 (+11%) | 60.1 (+43%) | 53.9 (+28%) |
| 1536x5120 | 62.4 | 58.7 (-6%) | 60.1 (-4%) | 61.7 (-1%) | 68.3 (+9%) | 67.8 (+9%) | 57.3 (-8%) |
| 256x5120 | 24.6 | 19.5 (-21%) | 19.6 (-20%) | 24.7 (+0%) | 18.1 (-26%) | 24.9 (+1%) | 18.0 (-27%) |

**three columns (verification), one dispatch per barrier** (`matvec-n3-geometry-g1.jsonl`), µs per dispatch group:

| shape | qwen_q4c3 | qwen_q4c3_r2 | qwen_q4c3_r4w2 | qwen_q4c3_r4w1 | qwen_q4c3_r8w2 | qwen_q4c3_r8w4 | qwen_q4c3_r2w8 |
|---|---|---|---|---|---|---|---|
| 2176x5120 | 94.6 | 94.3 (-0%) | 91.7 (-3%) | 86.8 (-8%) | 102.0 (+8%) | 103.1 (+9%) | 97.3 (+3%) |
| 5120x2176 | 87.9 | 96.9 (+10%) | 87.8 (-0%) | 87.1 (-1%) | 106.3 (+21%) | 106.2 (+21%) | 99.4 (+13%) |
| 1280x5120 | 59.8 | 57.2 (-4%) | 59.6 (-0%) | 58.9 (-2%) | 80.0 (+34%) | 81.0 (+35%) | 61.5 (+3%) |
| 768x5120 | 48.4 | 42.5 (-12%) | 47.4 (-2%) | 46.9 (-3%) | 79.0 (+63%) | 76.1 (+57%) | 43.4 (-10%) |
| 5120x768 | 59.8 | 55.9 (-7%) | 60.2 (+1%) | 59.0 (-1%) | 85.6 (+43%) | 86.4 (+44%) | 60.4 (+1%) |
| 1536x5120 | 60.6 | 72.6 (+20%) | 70.7 (+17%) | 58.9 (-3%) | 80.7 (+33%) | 81.4 (+34%) | 63.1 (+4%) |
| 256x5120 | 43.8 | 33.0 (-25%) | 45.0 (+3%) | 44.9 (+3%) | 75.3 (+72%) | 75.9 (+73%) | 29.9 (-32%) |

**three columns, two dispatches per barrier** (`matvec-n3-geometry-g2.jsonl`), µs per dispatch group:

| shape | qwen_q4c3 | qwen_q4c3_r2 | qwen_q4c3_r4w2 | qwen_q4c3_r4w1 | qwen_q4c3_r8w2 | qwen_q4c3_r8w4 | qwen_q4c3_r2w8 |
|---|---|---|---|---|---|---|---|
| 2176x5120 | 136.6 | 183.3 (+34%) | 135.2 (-1%) | 132.3 (-3%) | 154.2 (+13%) | 155.2 (+14%) | 191.1 (+40%) |
| 5120x2176 | 154.9 | 181.3 (+17%) | 154.9 (-0%) | 154.4 (-0%) | 174.2 (+12%) | 174.1 (+12%) | 193.5 (+25%) |
| 1280x5120 | 97.9 | 109.7 (+12%) | 96.7 (-1%) | 93.4 (-5%) | 103.8 (+6%) | 103.7 (+6%) | 123.0 (+26%) |
| 768x5120 | 60.7 | 72.3 (+19%) | 69.9 (+15%) | 58.7 (-3%) | 80.7 (+33%) | 81.6 (+34%) | 83.3 (+37%) |
| 5120x768 | 97.4 | 98.5 (+1%) | 95.5 (-2%) | 93.6 (-4%) | 125.4 (+29%) | 126.2 (+30%) | 112.8 (+16%) |
| 1536x5120 | 97.8 | 130.4 (+33%) | 98.3 (+1%) | 96.0 (-2%) | 105.3 (+8%) | 106.3 (+9%) | 124.9 (+28%) |
| 256x5120 | 45.1 | 34.9 (-23%) | 45.9 (+2%) | 44.8 (-1%) | 76.0 (+69%) | 75.9 (+68%) | 55.1 (+22%) |

The two-row kernels win at every one-column shape (10-35%, most where a shard has few workgroups: a 768-row
projection is 48 workgroups of matvecq4, 1.5 per CU); the unrolled loads help rows of 5120 and hurt rows of 768. At
three columns (`qwen_q4c3`, verification) no geometry gains more than 3%: those kernels are bound by arithmetic and
activation re-reads, not by wave count (see Signed kernels below).

## Results on the cluster

All runs generated the same 128 tokens as the record (`token_ids` of every `.json`), and every prefill the same
candidates. Columns: tok/s per repetition (mean and range), prefill tok/s, then incite's milliseconds per generated
token: `wait` for the GPUs, `exchange` (everything between the local partial sum and the normed rows), the local
`sum`, the `total` (residual add, RMS norm, both GPUs' input rows), and inside the exchange the payload `encode` and
`decode`, the lane exchanges (`wire`), of which the time until every send was issued (`sent`) and until the first
byte came back (`first`).

### Decode (`qwen-bench --pp 512 --tg 128 --reps 2`, pairs alternated)

Runs by phase: `dec` (the record's binary `final`; `base`: trashcan-r2 with the old kernels, so the host-side
changes alone; `auto`: two rows per wave; `merge`: plus the hoisted-load merge), `lanes` (trashcan-r3; `thr`: a
worker thread per cable, `avx`: the AVX build), `k` (trashcan-r4; `gdn2`/`gdn1`: two or one state rows per wave in
the DeltaNet recurrence; `p2`: the two-key attention partial), `f` (trashcan-r5; `on`: the fused host passes,
`noround`: fused and without rounding the decode totals).

| configuration | n | tok/s | pp512 | wait | exchange | sum | total | encode | decode | wire | sent | first |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2-dec-final | 4 | 37.41 (36.99-37.68) | 209 | 19.42 | 5.88 | 0.43 | 0.77 | - | - | - | - | - |
| r2-dec-base | 4 | 37.48 (37.17-37.81) | 207 | 19.41 | 5.86 | 0.31 | 0.87 | 0.23 | 0.96 | 4.40 | 1.29 | 2.18 |
| r2-dec-auto | 4 | 39.79 (39.29-40.17) | 209 | 18.08 | 5.68 | 0.31 | 0.83 | 0.22 | 0.92 | 4.31 | 1.27 | 2.24 |
| r2-dec-merge | 4 | 39.76 (39.56-39.96) | 208 | 18.05 | 5.50 | 0.32 | 1.05 | 0.22 | 0.87 | 4.17 | 1.28 | 2.17 |
| r2-lanes-auto | 4 | 39.59 (38.86-39.87) | 210 | 18.06 | 5.60 | 0.31 | 1.05 | 0.22 | 0.89 | 4.24 | 1.34 | 2.05 |
| r2-lanes-thr | 4 | 38.30 (35.10-39.93) | 208 | 18.07 | 6.83 | 0.29 | 0.75 | 0.24 | 1.02 | 5.36 | 0.56 | 1.68 |
| r2-lanes-avx | 4 | 39.84 (39.70-39.95) | 210 | 18.08 | 5.57 | 0.36 | 0.85 | 0.23 | 0.86 | 4.26 | 1.28 | 2.17 |
| r2-k-auto | 4 | 39.95 (39.82-40.06) | 209 | 18.07 | 5.64 | 0.31 | 0.77 | 0.22 | 0.92 | 4.28 | 1.27 | 2.21 |
| r2-k-gdn2 | 4 | 40.41 (40.16-40.57) | 212 | 17.79 | 5.66 | 0.30 | 0.77 | 0.23 | 0.94 | 4.25 | 1.29 | 2.23 |
| r2-k-gdn1 | 4 | 40.86 (40.53-41.19) | 212 | 17.61 | 5.56 | 0.30 | 0.77 | 0.22 | 0.93 | 4.19 | 1.28 | 2.19 |
| r2-k-p2 | 4 | 39.91 (38.59-40.40) | 208 | 17.89 | 5.87 | 0.31 | 0.77 | 0.22 | 0.93 | 4.48 | 1.29 | 2.23 |
| r2-f-off | 4 | 39.88 (39.78-39.97) | 210 | 18.07 | 5.69 | 0.32 | 0.76 | 0.22 | 0.94 | 4.29 | 1.31 | 2.23 |
| r2-f-on | 4 | 39.29 (37.51-40.18) | 209 | 18.08 | 6.69 | 0.00 | 0.47 | 0.55 | 1.45 | 4.46 | 1.27 | 2.03 |
| r2-f-noround | 4 | 40.51 (40.45-40.60) | 210 | 18.07 | 5.91 | 0.00 | 0.47 | 0.54 | 0.84 | 4.31 | 1.26 | 2.17 |

| candidate | n | default tok/s | candidate tok/s | change | GPU wait ms/token | exchange ms/token | pp512 tok/s |
|---|---|---|---|---|---|---|---|
| two-row matvec (r2 auto) vs record | 4+4 | 37.41 (36.99-37.68) | 39.79 (39.29-40.17) | +6.4% | 19.42 → 18.08 | 5.88 → 5.68 | 209 → 209 |
| one-row recurrence vs r4 defaults | 4+4 | 39.95 (39.82-40.06) | 40.86 (40.53-41.19) | +2.3% | 18.07 → 17.61 | 5.64 → 5.56 | 209 → 212 |
| committed defaults (r7) vs record | 4+4 | 37.10 (35.84-37.62) | 40.65 (39.92-41.36) | +9.6% | 19.43 → 17.28 | 6.12 → 6.00 | 211 → 212 |
| r7 without rounding vs r7 | 4+4 | 40.65 (39.92-41.36) | 42.24 (42.07-42.35) | +3.9% | 17.28 → 17.27 | 6.00 → 4.88 | 212 → 212 |

- **Two rows per wave** (`matvecq4_r2`, `matvecq4_r2u2` for rows of 5120): GPU time 19.42 → 18.08 ms per token,
  +6.4%. The host-side changes alone (`base`) are neutral: the one-pass local sum saves 0.12 ms, the fused rounding
  decode is within noise, the norm pass written for both GPUs in one loop was 0.1 ms slower and went back.
- **One state row per wave** in the recurrence: 48 dispatches of 23 µs become 13 µs, +2.3%; two rows +1.2%.
- **Hoisted merge loads**: nothing end to end (the merge at 38.5 µs is not load-latency bound; see the attention
  microbenchmark below). **Two-key partial**: GPU time -0.18 ms, tok/s within noise (one stalled run).
- **A worker thread per cable**: the sends go out in 4 µs instead of 10 and the first byte arrives 4 µs sooner,
  but the exchanges end no sooner (one run stalled at 35 tok/s): the exchange waits for the slowest machine's
  segment, not for our syscalls. Removed. **AVX build** (`-C target-cpu=ivybridge`): the host passes are unchanged,
  so they are memory-bound. Not adopted.
- **Fused host passes** (the two partials summed as they are encoded, the total added into the residual with the
  squares as it is decoded): slower (`decode` 0.94 → 1.45 ms), removed. The same measurement shows the rounding of
  the decode total costs about 0.6 ms per token.
- **Without rounding the decode totals** (`TRASHCAN_TP_ROUND=0`): +3.9% on top of the committed defaults, and the
  exchange shrinks more than the decode pass (6.00 → 4.88 ms): every machine's host pass is on the chain to the next
  all-reduce. The cost is that a decode row and the same column of a verification batch no longer agree bit for bit
  (the verification total is rounded by the all-gather); see the gates.

### MTP (k = 2, 64K draft head)

| configuration | n | tok/s | pp512 | wait | exchange | sum | total | encode | decode | wire | sent | first |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r2-mtp-final verify 46.2 ms/round | 4 | 50.19 (49.79-50.55) | 207 | 13.96 | 4.04 | 0.57 | 0.92 | - | - | - | - | - |
| r2-mtp-auto verify 46.1 ms/round | 4 | 50.27 (49.49-50.85) | 206 | 13.94 | 4.02 | 0.42 | 1.07 | 0.55 | 0.20 | 3.09 | 0.98 | 1.69 |
| r2-mtp-merge verify 46.5 ms/round | 4 | 49.92 (48.31-50.54) | 206 | 13.92 | 4.16 | 0.42 | 1.09 | 0.57 | 0.23 | 3.17 | 1.07 | 1.71 |

| configuration | n | tok/s | verification ms/round | tokens/round |
|---|---|---|---|---|
| record (`r2-final-mtp-rec`) | 2 | 50.53 (50.15-50.91) | 45.8 | 2.47 |
| committed defaults (`r2-final-mtp-new`) | 4 | 51.72 (51.02-52.21) | 44.8 | 2.47 |
| without rounding (`r2-final-mtp-noround`) | 4 | 51.58 (50.71-52.15) | 44.9 | 2.47 |

The second record run's teardown hung on a flaky link from the driving Mac (tailnet timeouts for a few minutes; the
Macs were fine), so the record has two runs and the others four (`remaining` phase).

The row geometry does not reach the three-column kernels, which dominate a verification batch (`profile` below),
so MTP gains only from the recurrence, the attention kernels and the draft steps: +2.9%. Verification batches take
the direct reduce-scatter/all-gather, whose totals are rounded on the wire either way, so `TRASHCAN_TP_ROUND=0`
touches only the draft steps there (within noise).

### Where the time goes now (`profile`, trashcan-r4 with dispatch stamps, `incite/profile-r2-prof-*.jsonl`)

Decode (29 steps, medians): GPU 17.87 ms per token (was 19.23), launch 0.53, host between segments 6.67. Per
segment: DeltaNet 111 µs (qkv/z/alpha-beta projections 44 concurrent, conv 5, recurrence 23 (13 with one row
per wave), gated norm 7.7, out-projection 22.7), FFN 129 (gate and up 66 each, concurrent, down 45, SwiGLU 3.3),
attention 186 (q/k/v 45, qk 14, partial 56, merge 38, gate 4, out 23), head 1199 (headvec 980, top-k 214).
Verification of 3 columns (9 steps): GPU 31.5 ms, host between segments 13.1; per segment DeltaNet 220 (qkv
92, z 52, alpha-beta 52, recurrence 39, out 60), FFN 234 (gate and up 126 each, down 88), attention 257, head 1885.
The three-column kernels take about twice the one-column time at every shape: `qwen_q4c3` needs 64 VGPRs
(RADV shader stats), so 4 waves per SIMD, 512 resident waves for a dispatch of 544 (2176 rows): 2048 rows take 71 µs
and 2176 rows 94 (`microbench/matvec-occupancy.jsonl`). The one-column `matvecq4_r2` needs 24 VGPRs (10 waves per SIMD),
`matvecq4_r2u2` 36 (7).

### Attention kernels on the D700 (`trashcan qwen-decode-attn-bench`, enlight, `microbench/attn-bench*.jsonl`)

3 query heads over one KV head of 256, 64 partitions, 50 dispatches per timing: at 520 keys the partial takes 56.8 µs
(`attention_partial64`) or 48.5 (`attention_partial_p2_64`, keys two at a time; with 3 columns 79.7: its 64 VGPRs
cost occupancy when there are 576 workgroups), the merge 38.5 (`attention_merge64`), 36.9 with 16 loads hoisted
(`attention_merge_h64`, removed) and 28.9 as 256 threads with every loop unrolled (`attention_merge_u64`); at 2000
keys the merge is the same; at 100 keys and 16 partitions 10.2 either way. All bit for bit the reference's.

### Correctness gates (`gates`, trashcan-r7)

| gate | committed defaults | without rounding (`TRASHCAN_TP_ROUND=0`) |
|---|---|---|
| `qwen-validate` (14-token prompt, `TRASHCAN_TP_WIRE=f32`) | every one of the 258 tensors, the logits and the 10 greedy tokens identical to the record's dump (`f-val32`, results/qwen/tp8-mesh-20261009): the new kernels give the old ones' bits | the same (the prompt's all-reduces are prefill-sized, f32) |
| `qwen-check --mtp` (600-token prompt, 64 steps) | pass: restore exact, speculative output identical, 83% of drafts accepted, column score error 0 | pass: restore exact, speculative output identical, 83% accepted, column score error 3.2e-3 (a decode row's total is the exact f32 sum of the four rounded contributions; a verification column's is rounded once more by the all-gather) |
| `qwen-check --verify 3` | pass: no column mismatches, column score error 0 | pass: no column mismatches, column score error 3.2e-3 |

### The committed binary (`trashcan-r8`: trashcan-r7 with the one-row recurrence under the `qwen_gdn` name)

One run each way after the gates, the same tokens: decode 39.2 and 41.5 tok/s (GPU wait 17.28 ms per token, as
trashcan-r7's; the first repetition hit an exchange stall, as single repetitions did a few times in this session and
the last), MTP 47.9 and 52.6 (verification 48.4 and 44.1 ms per round, the first with the stall).

## What changed in the code

- `matvecq4_r2`, `matvecq4_r2u2` (`matvecq4_body.glsl`): matvecq4 with 2 rows per wave in 256-thread workgroups,
  the latter loading two blocks ahead; tensor-parallel ranks take them for one column (rows of at least 4096 values
  the unrolled one). `qwen_matvec-bench` times decode and verification kernels at given shapes against the first
  bit for bit (`--group N` dispatches per barrier).
- `qwen_gdn`: one state row per wave (was four); `qwen_q4cols_body.glsl` takes its workgroup rows from ROWS and WAVES.
- `attention_partial_p2`, `attention_merge_u`: Qwen's one-column attention steps and head dimensions up to 256
  (`vulkan::qwen_attention_shaders`; `TRASHCAN_QWEN_ATTN_KERNELS=old` restores the reference pair). `qwen-decode-attn-bench`
  times partial and merge kernels at a rank's shape.
- The all-reduce phases in `tg_tp_ms`/`pp_tp_ms`: `encode`, `decode`, `wire`, `sent`, `first` (`link::Trip`,
  `tp::Stats`); the local sum of two partials in one pass; `Mesh::once` rounds its total block by block while
  decoding (`wire::decode_tree_round`, the same bits as encoding and decoding it) and skips the rounding with
  `TRASHCAN_TP_ROUND=0`.
- Removed after measurement: the other geometries, the signed-nibble kernels, the hoisted-load merge, the worker
  thread per cable, the fused host passes.

## Files

- `tools/round2.sh`: the phases (`decode`, `mtp`, `lanes`, `kernels`, `fused`, `profile`, `gates`);
  `tools/phases.py`: the tables above.
- `microbench/`: the `qwen-matvec-bench` output (JSON lines).
- `incite/`: every run's driver `.json`/`.err`, `commands.log`, the dispatch profiles (`profile-r2-prof-*.jsonl.gz`)
  and `clocks/`; `invite/`, `enlight/`, `indite/`: their node logs and `clocks/`. The validation dump (`r2-val32`, 261
  files) stays on incite under `~/trashcan-exp/tp8opt/results/`, beside the record's `f-val32`.
