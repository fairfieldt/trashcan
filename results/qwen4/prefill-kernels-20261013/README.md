# Flash-Next prefill kernels for 16-128-token chunks (2026-10-13)

**Result.** Six Macs, real text, alternating with main (6f6635a): prefill **369 -> 432 tok/s on 512 tokens (+17%)**,
508 -> 543 on 1024 (+7%), **595 -> 633 on 2048 (+6%)**, 649 -> 686 on 4096 (+6%), **702 -> 755 on 8192 (+8%)**;
decode unchanged (16-18 tok/s after the prompt). The greedy tokens are main's and the CPU reference's. One Mac's
8 layers per chunk: 76.9 -> 48.6 ms at 16 tokens (-37%), 96.8 -> 77.1 at 32 (-20%), 134.5 -> 118.0 at 64 (-12%),
240.9 -> 216.6 at 128 (-10%). Four changes: a lane-per-row Q8_0 GEMM for chunks of up to 32 tokens (the 64-token tiles
computed 64 tokens whatever the chunk), the 2560-row out projections split along K, the DeltaNet recurrence with a
state row in a lane's registers, and the prefill mixer norm per stream. The expert kernels, now 42-45% of a chunk, are
unchanged: every variant tried (occupancy split, prefetch, vector-loaded inputs, unrolled lines) lost.

Tested source: this branch at 583d2dc (the changes on main 6f6635a; the single-Mac rounds ran the same kernels on
4c1346d), binary `trashcan-pf2` (sha256 92062327…) on every Mac; baselines `trashcan-4c1346d` (single Mac) and
`trashcan-6f6635a` (six Macs). G-iq4: Q8_0 dense matrices, IQ4_XS gate/up and IQ4_NL down experts. Every GPU ran at
850/1370 MHz while working, on alight and on all six Macs (the 300/150 samples are loading and idle;
[chain/c1-clocks.txt](chain/c1-clocks.txt), the `[clocks]` lines of the `.err` files).

## Where the time went (main at 4c1346d)

`TRASHCAN_Q4_PROFILE=1 trashcan qwen4 stage-bench G-iq4-00001-of-00006.gguf --ranges 0:4,4:8 --gpus 0,1 --context 32768
--batch B --head-rows 0:21504,21504:43008 --decode 2 --start 64 --prefill 5` on alight (layers 0-7 on both D700s;
`tools/pf2-sb.sh`), one profiled chunk of B tokens at position 0. ms summed over the 8 layers and both GPUs; a
deferred group (dispatches without a barrier between them) is timed as one span; in brackets the share of the FMA peak
(850 MHz x 2048 lanes) for the matrix work ([stage-bench/](stage-bench), `tools/prof.py`):

| phase (8 layers, both GPUs) | 16 tokens | 32 | 64 | 128 |
|---|---|---|---|---|
| chunk wall (mean of 5) | 76.9 | 96.8 | 134.5 | 240.9 |
| experts gate/up (`qwen4_moe_gu_tiles`) | 13.4 (20%) | 21.3 (25%) | 32.8 (32%) | 54.6 (39%) |
| experts down (`qwen4_moe_down_tiles`, 3 passes) | 6.6 (20%) | 10.7 (25%) | 17.1 (31%) | 28.2 (38%) |
| DeltaNet qkv + z + alpha/beta (one group) | 14.5 (16%) | 14.7 (32%) | 15.2 (61%) | 29.6 (63%) |
| DeltaNet out projection 2560 x 6144 | 10.9 (8%) | 11.0 (16%) | 11.9 (29%) | 15.3 (45%) |
| DeltaNet recurrence (`qwen_gdn`) | 3.0 | 5.9 | 11.6 | 23.2 |
| mixers (16) | 9.2 (11%) | 9.4 (21%) | 9.9 (39%) | 17.6 (44%) |
| QSA q/k/v, out | 4.3, 3.7 | 4.3, 3.7 | 4.5, 4.0 | 8.0, 5.2 |
| n-gram module (layer 1) | 2.2 | 2.3 | 2.5 | 4.5 |
| router (f32; not this round) | 3.7 | 5.8 | 9.7 | 16.9 |

- The Q8_0 GEMMs cost the same from 16 to 64 tokens: `qwen4_q8_gemm`'s 64 x 64 tiles compute 64 tokens whatever the
  chunk, so a 16-token chunk runs them at 8-16% of the FMA peak; at 64-128 tokens they reach 56-63%.
- The 2560-row out projections give the tiled GEMM 40 workgroups for 32 CUs (8-45%).
- The DeltaNet recurrence is latency-bound: one wave per state row and four cross-lane reductions per token, 3.9 ms
  per layer at 128 tokens for 0.4 GFLOP.
- The output copy of a stage's residual to the host buffer takes 1.3 ms (64 tokens) and 3.0 ms (128 tokens) per GPU
  (~2 GB/s; the input copy runs at ~9.5 GB/s). Host path, not changed here.

## The changes

- **Lane-per-row Q8_0 GEMM up to 32 tokens** (`qwen4_q8_gemm_s.comp`): a 64-lane workgroup owns 64 rows (R rows per
  lane) and T tokens; each lane walks its rows along K, loads a whole 64-byte line (two blocks) per row one step ahead,
  decodes every weight once for the T tokens, and takes the tokens' inputs as scalar loads (SGPR operands of the
  multiply-adds), so neither operand passes through LDS and a 16-token chunk computes 16 tokens, not 64. The decode is
  exact (`(q + 128)` as an unsigned byte, `fma(q + 128, d, -128 d) = q d`) and the multiply-adds run in
  `qwen4_q8_gemm`'s order, so the outputs are the same bits (`qwen-gemm-bench`: 0 mismatches at every shape and batch).
  Used for every Q8_0 prefill projection up to 32 tokens (T = 16, `_s16r1p`: DeltaNet qkv/z, QSA q/k/v, the n-gram
  key/value, the out projections), the mixer bottleneck (T = 8, the row blocks first, `_s8r1po`) and, up to 16 tokens,
  the mixer up projection (R = 4: the four streams' rows of one element in a lane, which stores
  `u = mean_s xn_s sigmoid(g_s)` as `qwen4_q8_gemm_mix` does; `_s4r4om`).
- **Out projections split along K** (DeltaNet and QSA, 2560 rows): 8 parts (4 when the partial sums would not fit
  `hgate`, i.e. above half the stage's batch), partial sums in `hgate` (free during the sublayer), then
  `qwen4_sum_parts` adds them in part order.
- **DeltaNet prefill recurrence**: `qwen4_gdn_prep` normalizes q (times 1/sqrt(128)) and k and computes decay and beta
  once per token (qwen_gdn's arithmetic); `qwen4_gdn_rows` keeps one state row in a lane's registers (128 values),
  loads each token's k and q one token ahead into LDS and reads them as broadcasts, and sums the dot products in key
  order within the lane, so no cross-lane reductions remain. Same math; the dot products' summation order differs.
  The kernel uses 256 VGPRs and spills some to scratch, and is still the fastest variant measured (below).
- **`qwen4_hc_norm`**: one workgroup per (stream, token) instead of per token, values kept in registers between the
  combine and the norm (the same bits).
- `quant::repack_q8h` aligns the Q8_0 blocks to 64 bytes, so a pair of blocks is one cache line.
- `qwen-gemm-bench` takes Q8_0 kernels (`qwen4_q8_*`), `KERNEL:PARTS` K splits and the mixer up projections.

Decode and verification batches (up to 8 tokens) run the same kernels as before.

## Per-dispatch before and after (alight, 8 layers, both GPUs)

Main 4c1346d against this branch's kernels (`stage-bench/pf2-base-b*.json`, `stage-bench/final-alight/pf2-c1-b*.json`;
GPU ms of one profiled chunk, the FMA-peak share in brackets):

| phase | 16 tokens | 32 | 64 | 128 |
|---|---|---|---|---|
| DeltaNet qkv + z + alpha/beta | 14.5 -> 5.8 (40%) | 14.7 -> 10.2 (45%) | 15.2 -> 15.2 | 29.6 -> 29.6 |
| DeltaNet out projection | 10.9 -> 3.2 (27%) | 11.0 -> 5.7 (31%) | 11.9 -> 7.0 (50%) | 15.3 -> 13.0 (54%) |
| QSA q/k/v | 4.3 -> 1.8 | 4.3 -> 3.0 | 4.5 -> 4.5 | 8.0 -> 8.0 |
| QSA out projection | 3.7 -> 1.0 | 3.7 -> 1.8 | 4.0 -> 2.3 | 5.2 -> 4.3 |
| mixers (16) | 9.2 -> 5.1 | 9.4 -> 7.8 | 9.9 -> 9.5 | 17.6 -> 17.0 |
| of which `qwen4_hc_norm` | 1.31 -> 0.47 | 1.36 -> 0.57 | 1.55 -> 1.09 | 2.38 -> 1.78 |
| n-gram module | 2.2 -> 1.0 | 2.3 -> 1.5 | 2.5 -> 2.5 | 4.5 -> 4.5 |
| DeltaNet recurrence (prep + rows) | 3.0 -> 1.05 | 5.9 -> 1.60 | 11.6 -> 2.82 | 23.2 -> 5.33 |
| experts (gate/up + down), unchanged | 20.0 -> 20.1 | 31.9 -> 32.0 | 49.9 -> 49.8 | 82.8 -> 82.7 |
| **chunk wall** | **76.9 -> 48.6** | **96.8 -> 77.1** | **134.5 -> 118.0** | **240.9 -> 216.6** |
| slower GPU of the Mac (4 layers + n-gram) | 39.1 -> 24.3 | 48.9 -> 38.9 | 65.7 -> 57.8 | 112.9 -> 101.9 |

After this round the experts take 43-45% of a 16-64-token chunk and 42% of a 128-token chunk; the projection GEMMs run
at 27-45% of the FMA peak up to 32 tokens and 50-63% beyond.

## GEMM microbenchmark

`trashcan qwen-gemm-bench --shapes ROWSxK --n N --kernels A,B,... --iterations 20` on alight GPU 0 (random Q8_0
weights; every kernel compared bit for bit with the first; `tools/pf2-gemm.sh`, [micro/](micro)), µs per GEMM and share
of the FMA peak:

| shape, tokens | `qwen4_q8_gemm` (64 x 64 tiles) | lane-per-row | |
|---|---|---|---|
| 10240 x 2560, 16 | 1578 (15%) | 830 (29%) `_s16r1p` | 1.9x |
| 10240 x 2560, 32 | 1599 (30%) | 1064 (45%) `_s16r1p` | 1.5x |
| 10240 x 2560, 64 | 1662 (58%) | 1766 (55%) `_s16r1p` | 0.94x |
| 10240 x 2560, 128 | 3125 (62%) | 5554 (35%) `_s16r1p`; best 3107 (62%) `_s16r2p` | 1.0x |
| 2560 x 6144 (out), 16 | 1798 (8%) unsplit; 954 (15%) 8 parts | 471 (31%) `_s16r1p`, 8 parts | 3.8x |
| 2560 x 6144, 32 | 1842 (16%); 978 (30%) 8 parts | 914 (32%), 8 parts | 2.0x |
| 2560 x 6144, 64 | 1955 (30%); 1056 (55%) 8 parts | 1643 (35%), 8 parts | 1.85x (tiles, 8 parts) |
| 2560 x 6144, 128 | 2553 (45%); 2127 (54%) 4 parts | 3387 (34%), 4 parts | 1.2x (tiles, 4 parts) |
| mixer bottleneck 324(384) x 10240, 16 parts, 16 | 265 (14%) | 136 (27%) `_s8r1po` | 1.9x |
| same, 32 | 269 (27%) | 213 (34%) | 1.26x |
| mixer up 10240 x 320 + read gates, 16 | 217 (14%) `_mix` | 151 (20%) `_s4r4om` | 1.44x |
| same, 32 | 221 (27%) | 215 (28%) | (tie: kept `_mix`) |

All lane-per-row results were bit-identical to `qwen4_q8_gemm` with the same K split. The lane-per-row kernel tops out
near 55%: each multiply-add instruction takes its input from an SGPR, so a wave issues one scalar load (and one or two
address instructions) per four multiply-adds per row, and waits for all outstanding scalar loads at once (they return
out of order). A diagnostic with every input in the scalar cache (wrong results) ran 30% faster (16 tokens: 29% -> 38%
of the peak; 128: 42% -> 50%), so scalar-cache misses cost about a quarter.

## Tried and dropped

Measured with the same stage-bench (8 layers, ms per profiled chunk, the same Mac for each comparison) or the GEMM
microbenchmark; the code and its temporary switches are removed (patches of the experiment states in [tools/](tools)).

| candidate | result |
|---|---|
| lane-per-row GEMM beyond 32 tokens; 32 tokens per workgroup; 2 or 4 rows per lane; the row blocks first | at 64 tokens 54-55% against the tiles' 58%, at 128 35-62% against 62%; 32-token workgroups need 128 SGPRs of inputs and wait on every few scalar loads (12-37%) |
| register-blocked LDS GEMM (`qwen4_q8_gemm_t`: 8 x 4, 4 x 8 and 8 x 8 outputs per thread, two-k LDS reads) | 10240 x 2560 at 128 tokens 65% against 62%, at 64 tokens 52% against 58%, out projection worse; the 8 x 8 variants 20-43% |
| expert gate/up tiles split over two pipelines (1-4 slots at 84 VGPRs, 3 waves; 5-8 slots at 128) | gate/up 6-8% slower at every chunk size (16: 13.4 -> 14.4 ms, 64: 32.8 -> 35.4, 128: 54.6 -> 57.7); with a line prefetch in the 1-4 pipeline still 4% slower |
| gate/up next-line prefetch for tiles of 1-4 slots in the single pipeline (fits its 128 VGPRs) | 3-6% slower (16: 13.4 -> 14.2, 128: 54.6 -> 56.1) |
| gate/up slot inputs through vector loads (tiles of 1-4 slots; in-order counter instead of scalar-cache drains) | 20% slower (32: 22.2 -> 26.9) |
| gate/up and down walks with the line's four blocks unrolled (no rotation of the loaded words: 17-21% of the small tiles' VALU instructions are `v_mov`) | gate/up 3-29% slower (16: 13.5 -> 16.3, 32: 21.3 -> 27.5, 128: 54.5 -> 56.3), down 2-10%; unrolled only up to 4 slots: 2% slower. The four-fold loop bodies presumably overflow the instruction cache shared by four CUs |
| down tiles next-line prefetch | 11% slower (32: 10.7 -> 12.5) |
| down passes sized by the batch (one pass up to 55 tokens, two at 64 instead of three) | no change (32: 78.6 vs 78.8 ms wall; 64: 118.5 vs 119.0) |
| DeltaNet rows with q and k as scalar loads instead of LDS broadcasts | 2.13 against 1.56 ms (32 tokens), 5.91 against 5.17 (128) |
| DeltaNet rows split over two lanes (64 values each, no spills, one swizzle per sum) | 1.59 -> 1.65 ms (32 tokens), 5.19 -> 5.78 (128) on inflight |

## Correctness

| check | result |
|---|---|
| GEMM microbenchmark, every lane-per-row variant and K split against `qwen4_q8_gemm` | 0 mismatched bits at 16-128 tokens |
| layers 0-7 on alight (`stage-bench --tokens-file p100.ids --out`), one 100-token chunk, against main | 114 tensors within rel 3.6e-5 (the QSA output of layer 7; layer 0's DeltaNet output 1.9e-7) ([compare-stage-p100-b128-g-vs-main.txt](compare-stage-p100-b128-g-vs-main.txt)); the per-stream norm and the LDS DeltaNet rows changed no bits against the variants before them |
| the same in 16-token chunks | within 1.9e-5 ([compare-stage-p100-b16-g-vs-main.txt](compare-stage-p100-b16-g-vs-main.txt)) |
| six Macs, 14-token reference prompt (one 14-token chunk) | greedy 9419 0 2500 628 353 1438 488 3242 30 248046 (the reference's); logits rel 2.9e-4 to the CPU dump (main 2.85e-4) ([compare-c1-val14-vs-cpu.txt](compare-c1-val14-vs-cpu.txt)) |
| six Macs, `p100.ids` (16-token chunks), against main 6f6635a | the same 10 greedy tokens; logits rel 5.7e-5, residual after layer 47 rel 1.1e-4 ([compare-c1-val100-vs-main.txt](compare-c1-val100-vs-main.txt)) |
| the same against the CPU reference `ref-p100` | the same 10 greedy tokens and top-10; logits rel 3.48e-2 for both binaries, every residual the same distance ([new](compare-c1-val100-vs-cpu.txt), [main](compare-c1-val100-main-vs-cpu.txt)) |
| six Macs, the first 300 ids of `long.ids` (32-token chunks), against main | the same 10 greedy tokens and top-10; logits rel 7.8e-3: within 1e-5 through layer 9, then growing from layer 10 ([compare-c1-val300-vs-main.txt](compare-c1-val300-vs-main.txt)) |
| the same against the CPU reference of the 300 ids (`ref-p300`) | the same greedy tokens for both; logits rel 9.9e-3 (this branch) and 1.35e-2 (main), residual after layer 47 rel 1.8e-2 and 2.3e-2 ([new](compare-c1-val300-vs-cpu.txt), [main](compare-c1-val300-main-vs-cpu.txt)) |

The 300-token prompt's divergence from main is the amplification of summation-order differences (the split-K out
projections, the DeltaNet dot products) through 48 layers of near-tied expert choices; this branch ends closer to the
CPU reference than main does.

## End to end (six Macs)

`qwen4 bench --head-split 12 --context 32768 --batch 128 --ngram disk --tg 16 --tokens-file long.ids` (4096: `long4k.ids`,
8192: `long8k.ids`), main `trashcan-6f6635a` and this branch alternating, two runs each at 512, 2048 and 8192 tokens and
one at 1024 and 4096 (`tools/pf2-chain.sh` under chain6.sh; driver outputs in [chain/](chain)). The chunk size is main's
rule (32 tokens for 512, 64 for 1024-4096, 128 for 8192):

| prompt | main 6f6635a prefill tok/s (decode) | this branch | change |
|---|---|---|---|
| 512 | 369.5, 369.2 (17.94, 17.70) | 432.2, 432.2 (18.01, 18.09) | +17.0% |
| 1024 | 508.3 (17.46) | 543.2 (17.86) | +6.9% |
| 2048 | 595.6, 595.0 (16.02, 16.15) | 632.2, 634.3 (16.23, 16.19) | +6.4% |
| 4096 | 648.7 (15.72) | 686.2 (16.15) | +5.8% |
| 8192 | 704.4, 699.5 (16.11, 15.86) | 754.2, 756.6 (16.08, 16.32) | +7.6% |

The chain gains less than the stage times (the first GPU's 4 layers: -21% at 32 tokens, -12% at 64, -10% at 128): the
transfers and host work between stages do not shrink.

## What is left

- **The experts** (now 42-45% of a chunk): a tile decodes each IQ4 weight with two VALU instructions and an LDS table
  read, then one multiply-add per slot, and most tiles of 16-64-token chunks hold 1-3 slots; the kernels run at about
  half of that VALU bound. Occupancy splits, prefetching, vector-loaded inputs and unrolled lines all lost (above), so
  a different structure is needed rather than tuning.
- **Q8_0 GEMMs at 64-128 tokens** stay at 58-63% of the FMA peak: the 64 x 64 tiles read one LDS operand per two
  multiply-adds (the LDS bandwidth), the register-blocked variants gained at most 5% at 128 tokens, and scalar-operand
  kernels wait on scalar-cache misses.
- **The router** (f32, 8% of a chunk; the experts agent) and **the stage output copy** to host memory at ~2 GB/s
  (1.3 ms per GPU at 64 tokens, 3 ms at 128; the host path).
- The n-gram module keeps incite's first GPU ~2.3 ms per 64-token chunk slower than every other stage.

## Files

- `stage-bench/`: `pf2-base-b*` (main 4c1346d), `pf2-b`/`e`/`f`/`flop`/`g`/`ivx`/`idp`/`idpvx`/`j-b*` (session 1-2
  variants on alight: split-K out projections and the gate/up split, LDS DeltaNet rows, per-stream norm, prefetch,
  vector inputs, down prefetch, pass sizing), `final-alight/` (this branch `c1` against the unrolled walks `u`),
  `variants-inflight/` (`c1`, `u4`, `hr` on inflight); clock logs (`.txt`) and `.err` beside each.
- `micro/`: `qwen-gemm-bench` JSON lines (`v1`-`v3` lane-per-row variants, `out`, `mixdn`, `mixup`, `tiled`).
- `chain/`: the six-Mac driver outputs (`pf2-c1-*`) and [c1-clocks.txt](chain/c1-clocks.txt) (every node's and the
  driver's clock summary: 850/1370 MHz while working).
- `compare-*.txt`: dump comparisons.
- `tools/`: `pf2-sb.sh` (stage-bench per binary and chunk size), `pf2-gemm.sh`, `pf2-dump.sh`, `pf2-s1.sh`/`pf2-s2.sh`
  (alight sessions), `pf2-s3a.sh`/`pf2-s3b.sh` (one-Mac jobs), `pf2-chain.sh` (the six-Mac session), `run1mac.sh`,
  `deploy.sh`, `collect-clocks.sh`, `prof.py` (categories and FMA-peak shares), `disp.py`, `cmp.py`, `stages.py`,
  `gemm.py`, `dcmp.py`, `vsref.py`, `ab.py`, `isa.py`, `opcount.py`, `session1-experiments.patch` and
  `session2-experiments.patch` (the experiment states).
