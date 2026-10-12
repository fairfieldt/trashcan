# Flash-Next prefill mixers: split-K bottleneck GEMM (2026-10-12)

**Result.** The hyper-connection mixers' prefill path took 3.7-4.6 ms per mixer almost whatever the chunk because its
10240 -> 320 bottleneck ran as a 64 x 64 tiled GEMM on 5 row tiles (10 workgroups at 128 tokens on a 32-CU D700), each
looping over K = 10240. The bottleneck and the 4 write rows now run as one Q8_0 GEMM split 16 ways along K, a small kernel
sums the parts and applies the activations, and the up projection computes the sublayer input in its epilogue.
16 mixers (layers 0-7): 62.8 -> 19.6 ms per 128-token chunk, 76.9 -> 38.2 per 256-token chunk, and the new path
beats the fused small-batch kernels from 12 tokens on (`FUSED_MAX` 64 -> 8), which matters now that prompts are cut
into 16-64-token chunks: a 64-token chunk takes 17% less stage time, a 32-token chunk 12% less. Six Macs, real text,
against main (88def0d): **prefill 260 -> 298 tok/s on 512 tokens, 364 -> 415-434 on 2048**, decode unchanged
(13.4-14.7 tok/s); the same greedy tokens as main and the CPU reference.

## Where the time went

Per-dispatch GPU timestamps (a temporary build: `tools/dispatch-profile.patch` stamps the start and end of every
dispatch and prints the sums per kernel), stage-bench on invite, layers 0-3 on GPU0 and 4-7 on GPU1, 16 mixers, ms per
mixer ([stage-bench/profile-base-b*.err](stage-bench)):

| dispatch (c55128f) | 65 tokens | 128 | 256 |
|---|---|---|---|
| `qwen4_hc_norm` (combine, stream norms) | 0.095 | 0.157 | 0.371 |
| `qwen4_q8_gemm` 320 x 10240 (bottleneck) | 2.375 | 2.434 | 2.618 |
| `qwen4_act` (both) | 0.007 | 0.007 | 0.009 |
| `qwen4_q8_gemm` 10240 x 320 (up) | 0.406 | 0.419 | 0.787 |
| `qwen4_hc_mix` | 0.033 | 0.062 | 0.122 |
| `qwen_matf32` 4 x 10240 (write rows, f32) | 0.729 | 0.729 | 0.729 |
| sum | 3.65 | 3.81 | 4.64 |

The bottleneck GEMM did 419 M FMAs in 2.4 ms at 128 tokens (10% of the D700's FMA rate) while the up projection, the
same kernel with 320 workgroups, ran at ~57%. The write rows (one wave per row and 8 tokens, K = 10240) were latency
bound: 0.73 ms at every size.

## The change

- `qwen4_q8_gemm` takes an optional K split (`p.m` parts): workgroup x covers row tile x % tiles of part x / tiles
  and writes its partial sums to `[part][t][rows]`. Unsplit callers are unchanged.
- The Q8_0 prefill mixer runs the fused matrix (320 bottleneck rows, then the 4 write rows: the matrix the small-batch
  kernels already use) through it with 16 parts: 6 row tiles x 16 parts = 96 workgroups per 64 tokens. The partial
  sums go to `hgate`, which is free until the up projection.
- `qwen4_mix_act` sums the 16 parts in order and applies silu(v / 4) (bottleneck) or 2 sigmoid(v / 4) (write
  weights). This replaces the f32 write-row projection and both `qwen4_act` dispatches.
- `qwen4_q8_gemm_mix` (a specialization of `qwen4_q8_gemm`) is the up projection with its rows permuted so that each
  thread's four accumulators are the four streams' read gates of one element; it stores
  u = mean_s xn_s sigmoid(g_s) directly (the `qwen4_hc_mix` arithmetic) instead of the 10240 gates.
- The final mixer (no write rows) takes the same path with the 320-row down matrix above 8 tokens; at 8 or fewer
  (decode) it keeps the matrix-vector kernels.
- `FUSED_MAX` 64 -> 8 for Q8_0 mixers. Q4_0 mixers keep the old prefill path and `FUSED_MAX` 64.

Same math as before: each product is the same f32 value (Q8_0 dequantizes exactly to f32), only the summation order of
the bottleneck and write rows differs (16 partial sums); the up projection and sublayer input are bit-identical to the
previous path's GEMM + `qwen4_hc_mix` (dump comparison of the two, rel 0).

Per dispatch afterwards (the profiling build of this change on 90cb371, [stage-bench/x-s4p-*](stage-bench)), ms per
mixer:

| dispatch | 128 tokens | 256 |
|---|---|---|
| `qwen4_hc_norm` | 0.157 | 0.371 |
| `qwen4_q8_gemm` 324 x 10240, 16 K parts | 0.519 | 0.983 |
| `qwen4_mix_act` | 0.018 | 0.033 |
| `qwen4_q8_gemm_mix` 10240 x 320 | 0.431 | 0.816 |
| sum | 1.13 (3.81 before) | 2.20 (4.64) |

Both GEMMs now run at 0.97-1.03 T FMA/s, 56-59% of the D700's 1.74 at 850 MHz (counting the sixth row tile's 60
empty rows); the fused epilogue saves the gate store and `qwen4_hc_mix` (0.05 ms per mixer at 128 tokens).

## Stage times

stage-bench, layers 0-7 (`--ranges 0:4,4:8 --gpus 0,1 --context 32768 --head-rows 0:21504,21504:43008 --decode 2
--start 64 --prefill 3`, context 8192 for 256), G-iq4, invite, baseline `trashcan-main` (main at 90cb371) and this
change rebased on it (before the two guards that only touch Q4_0 mixers and the final mixer, neither in layers 0-7),
alternating per chunk size, under `scripts/with-clocks.sh`: both D700s at 850/1370 MHz while
working (the 300/150 samples are the model load; [stage-bench/clocks](stage-bench/clocks)). Wall = the 8 layers of
one chunk (GPU0 then GPU1), mean of 3 chunks; GPU = layer times per GPU ([stage-bench/x-s4-*](stage-bench)):

| chunk | wall, main | wall, new | | 16 mixers, main | new | GPU0 / GPU1, main | new |
|---|---|---|---|---|---|---|---|
| 12 | 77.3 | 74.9 | -3.1% | 11.5 | 9.5 | 39.1 / 36.8 | 38.1 / 35.6 |
| 16 | 90.8 | 85.2 | -6.2% | 15.0 | 9.5 | 45.6 / 43.4 | 43.0 / 40.6 |
| 32 | 138.7 | 121.4 | -12.5% | 27.7 | 9.9 | 69.4 / 67.1 | 60.5 / 58.1 |
| 64 | 238.3 | 196.8 | -17.4% | 53.1 | 10.9 | 117.7 / 115.6 | 96.7 / 94.3 |
| 128 | 422.4 | 376.3 | -10.9% | 62.8 | 19.6 | 204.6 / 200.5 | 183.0 / 178.3 |
| 256 | 789.1 | 747.8 | -5.2% | 76.8 | 38.2 | 382.2 / 373.6 | 362.7 / 354.3 |

Up to 64 tokens main runs the fused kernels; from 65 the old prefill path (65 tokens: 59.3 -> 17.2 ms of mixers, wall
267.7 -> 225.9 against c55128f, [stage-bench/x-s3-*-b65](stage-bench)). Decode (1 token) runs the fused kernels as
before.

Fused kernels against the new path below 64 tokens (this change on c55128f with temporary `FUSED_MAX` and K-split
overrides, invite,
[stage-bench/x-s3-*](stage-bench)), 16 mixers in ms (wall of the 8 layers in brackets):

| chunk | fused | new path |
|---|---|---|
| 8 | 8.66 (41.2) | 9.37 (41.8) |
| 12 | 11.52 (77.3) | 9.45 (75.2) |
| 16 | 14.95 (90.5) | 9.51 (84.9) |
| 24 | 21.37 (114.6) | 9.69 (102.7) |
| 32 | 27.69 (138.7) | 9.90 (121.2) |
| 48 | 40.49 (188.9) | 10.46 (158.6) |
| 64 | 53.14 (238.4) | 10.90 (195.0) |

The fused kernels cost ~0.7 ms per token for 16 mixers, the new path ~9.4 ms up to 64 tokens (one 64-token tile), so
they tie at about 9 tokens: `FUSED_MAX` = 8, which is also where the other projections switch from the matrix-vector
kernels to the GEMM. 32 K parts instead of 16 measured the same (9.52, 9.93, 11.01 ms at 16, 32, 64 tokens), as did 8
at 128 tokens.

## Correctness

Six Macs (`qwen4 validate`, `--head-split 12 --context 32768 --batch 128 --ngram disk`; on 88def0d validate cuts
prompts like the server, so the 100-token prompt runs as six 16-token chunks through the new path and a 4-token tail
through the fused kernels):

| check | result |
|---|---|
| 14-token reference prompt (one 14-token chunk, new path), against the CPU reference | greedy 9419 0 2500 628 353 1438 488 3242 30 248046, identical; logits rel 2.87e-4, residual after layer 47 rel 1.8e-4 ([compare-val14-new-vs-cpu.txt](compare-val14-new-vs-cpu.txt); main measured 2.85e-4) |
| `p100.ids`, new against main (88def0d) | the same 10 greedy tokens, logits rel 5.9e-5, residual after layer 47 rel 1.1e-4, largest tensor rel 2.2e-4 ([compare-val100-new2-vs-old.txt](compare-val100-new2-vs-old.txt)) |
| the same, against the CPU reference | the same 10 greedy tokens and top-10, logits rel 3.48e-2, cos 0.99945, as main ([compare-val100-new2-vs-cpu.txt](compare-val100-new2-vs-cpu.txt), [compare-val100-old-vs-cpu.txt](compare-val100-old-vs-cpu.txt)) |
| layers 0-7 alone, `p100.ids` as one 100-token chunk (stage dump on invite), new against c55128f | largest tensor rel 1.6e-5 (QSA output), mixer outputs 5-8e-6 ([compare-stage0-7-new-vs-base.txt](compare-stage0-7-new-vs-base.txt)) |

New against main differs more at 16 tokens (fused kernels against the GEMM path, which fold the stream norms in
differently) than at 100 tokens (GEMM against GEMM), and stays at the level of summation order. `val14-new` and
`val100-new` ran with the binary before the last change (the final mixer at 8 tokens or fewer back on the
matrix-vector kernels); `val100-new2` is the committed code, and its dump is within rel 1.4e-7 of `val100-new`.

## End to end (six Macs)

`qwen4 bench --tokens-file long.ids` (a 3061-token document), `--head-split 12 --context 32768 --batch 128 --ngram
disk`, alternating baseline `trashcan-88def0d` (main) and this commit's code (`trashcan-mixk`), two runs each
(`tools/mixk-chain.sh` under `chain6.sh`; driver outputs in [chain/incite](chain/incite), node logs and clocks in
[chain/nodes](chain/nodes)). Every GPU at 850/1370 MHz while working (the engine holds the high DPM level; the
300/150 samples are loading and idle). Main picks 32-token chunks for 512 tokens and 64-token chunks for 2048:

| run | prefill tok/s | decode tok/s | incite GPU0 / GPU1 per chunk |
|---|---|---|---|
| main, pp512 | 259.6, 260.7 | 14.70, 14.63 | 71.1 / 68.9 ms |
| new, pp512 | 297.5, 297.8 | 14.67, 14.70 | 62.5 / 60.0 ms |
| main, pp2048 | 364.9, 364.1 | 13.45, 13.37 | 136.0 / 133.6 ms |
| new, pp2048 | 415.6, 434.3 | 13.49, 13.42 | 115.2 / 112.5 ms |

+14.4% at 512 tokens and +14-19% at 2048. The per-chunk stage times match stage-bench (-12% at 32 tokens, -15% at 64).

## Tried and dropped

- **Write rows in the norm kernel** (`qwen4_hc_norm` computing W_inject xn from the f32 write rows while it normalizes,
  so the GEMM has 5 row tiles instead of 6): 16 mixers 21.95 ms at 128 tokens against 20.45 without; the norm went
  from 0.16 to 0.32 ms per mixer, more than the sixth row tile cost ([stage-bench/x-v2-*](stage-bench): `v1` split-K
  only, `inj`, `up` the fused up projection, 19.58, and `both`).

## Files

- `stage-bench/`: `profile-base-b*` (c55128f with the per-dispatch timestamps), `x-s3-*` (final path and c55128f,
  and the crossover), `x-v2-*` (variants at 128 tokens), `x-s4-*` and `clocks/` (final stage-bench against
  90cb371), `x-s4p-*` (per-dispatch timestamps of this change).
- `chain/`: the six-Mac runs (`incite/` driver outputs, `commands.log` and clocks; `nodes/` node logs and clocks).
- `compare-*.txt`: dump comparisons.
- `tools/`: `mixk-x.sh` (stage-bench per chunk size and configuration), `mixk-dump.sh`, the session scripts
  (`mixk-s2.sh` ... `mixk-s4.sh`, `mixk-chain.sh` under `~/dev/claude-work/flashnext-20261012/chain6.sh`), `agg.py` (sums
  the per-dispatch timestamps), `dispatch-profile.patch` (the temporary timestamp code; applies on top of this commit, `Q4_DISPATCH_PROFILE=1`
  prints `DISP` lines).
