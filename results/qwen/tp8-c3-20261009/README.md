# Qwen3.8-27B TP8, round 3 (2026-10-09): the three-column verification kernel

**Result.** No faster three-column kernel was found under the bitwise contract, and nothing from this round is
kept in the engine. `qwen_q4c3` (4 rows per wave, 64 VGPRs, 4 waves per SIMD) is bound by latency hiding on the
D700, and every way of raising its occupancy costs more than it gains: activations staged in shared memory (three
designs, 5 waves per SIMD) run 30-50% slower, a prefetch of the next blocks' weights 8-17% slower, two rows per
wave only wins on shards of at most 1280 rows (-6 to -9%), and five rows per wave only on the 2176-row FFN
shard (-10% alone, -2% when gate and up run together, as they do). On the cluster the five-row kernel gave
51.3 tok/s with MTP against 51.9 (GPU time per token 13.50 against 13.54 ms, the same tokens), so it was removed
after the gates. The measurements say where the time goes: at 7 waves per SIMD the arithmetic is issued 98% of the
time, at 4 about 65%, and with the same instruction stream capped to 1 wave the kernel takes 88% longer.

## Setup

- **Source.** The working tree on top of eabff7b (`main`, the round-2 commit); the microbenchmark binaries
  (`trashcan-c3a` … `trashcan-c3i`) and the cluster binary `trashcan-c3` (sha256 `3d88724c…`) were cross-built with
  `scripts/cross-build.sh` into `~/trashcan-exp/tp8opt/` (enlight for the microbenchmarks; all four Macs for
  `trashcan-c3`). `trashcan-c3` is eabff7b plus `qwen_q4c3_r5` (below), chosen in `Rec::matrix` for three columns
  and `2048 < rows <= 2560`, with `TRASHCAN_QWEN_C3=r4` restoring `qwen_q4c3` for the A/B. The experimental kernel
  sources are under `kernels/`.
- **Microbenchmarks.** enlight's GPU 0 under lease #10, `trashcan qwen-matvec-bench` (50 dispatches per timing,
  median of five, random Q4_0 weights at the 27B's TP8 shard shapes, every kernel checked bit for bit against
  `qwen_q4c3`, or against `matvecq4_r2u2` for one column), `RADV_DEBUG=shaderstats,nocache` for the register counts
  (`tools/shaderstats_map.py` names the pipelines in creation order), `tools/table.py` for the tables.
- **Cluster.** The six-cable cluster of results/qwen/tp8-mesh-20261009 (full-mesh all-reduces), kernel 7.2.9-p2p,
  gpuq leases on all four (incite #134, invite #122, enlight #11, indite #8), the performance governor on all four
  during the session (schedutil restored after) and the 0 µs latency request; `tools/c3.sh` runs
  `../tp8-mesh-20261009/tools/tp8full.sh`. Every card at 850/1370 MHz while computing (`incite/clocks/`). Runs:
  `qwen-bench --pp 512 --tg 128 --reps 2 --mtp MTP`, two alternated runs a side, then one profiled run a side and
  the gates.

## Where the time of `qwen_q4c3` goes (microbenchmarks, enlight)

| kernel | VGPRs | waves/SIMD | VALU per shader | 2048x5120 | 2176x5120 |
|---|---|---|---|---|---|
| `qwen_q4c3` | 64 | 4 | 377 | 70 µs | 90 µs |
| same, half the dot products (`dnohi`, wrong results) | 64 | 4 | 48 fewer | 64 (-8%) | 71 (-22%) |
| same, every column reading column 0's activations (`dx0`, wrong) | 64 | 4 | 377 | 68 (-3%) | 90 (-2%) |
| activations loaded once and shared by the columns (`dx1`, wrong) | 36 | 7 | 277 | 53 (-24%) | 59 (-35%) |
| `dx1` capped to 2 waves per SIMD by 16 KB of shared memory | 128 | 2 | 282 | 72 (+2%) | 89 (-3%) |
| `qwen_q4c3` capped to 1 wave per SIMD by 21 or 32 KB of shared memory | 256 | 1 | 382 | 134 (+88%) | 158 (+73%) |

(`microbench/c3d-diag.jsonl`, `c3f-diag.jsonl`, `c3g-diag.jsonl`; the shared-memory padding changes the compiler's
occupancy target, hence the VGPR counts.) The kernel's loop is about 240 VALU instructions per 16 blocks of four
rows and three columns (12 FMA chains of 11, four nibble conversions of 11, the three bias sums, 30 of address
arithmetic); at 100% issue the 2048-row shard would take 45 µs. The `dx1` variant runs at 98% of its VALU floor
with 7 waves per SIMD, `qwen_q4c3` at about 65% with 4, and capping either to fewer waves costs the difference, so
the kernel waits for memory and lacks waves to cover the wait. Reading a third of the activation bytes (`dx0`)
changes nothing, so L2 bandwidth is not the limit. The budget that sets the occupancy is the register file: the
three columns' activations (24 VGPRs), the 12 accumulators, four rows' scales and nibbles (8), one row's
converted bytes (8) and the bias sums (3) leave no way under the 48 VGPRs of a fifth wave without extra work.

## Kernels tried (all bit for bit `qwen_q4c3`'s results)

µs per dispatch group, one dispatch per barrier unless "g2" (two independent dispatches sharing a barrier, as the
FFN gate and up projections run); `microbench/c3b…c3i-*.jsonl`, register counts from the `*-shaderstats.txt` files.

**Row geometry** (`qwen_q4cols_body.glsl` with other ROWS/WAVES; `c3b-n3-g1.jsonl`, the two-row column from `c3e-n3-g1.jsonl`, whose percentages are against that run's `qwen_q4c3`):

| shape | `qwen_q4c3` | 5 rows (`_r5`, 64 VGPRs) | 5 rows, 2 waves/WG | 6 rows | 2 rows (48 VGPRs, 5 waves) |
|---|---|---|---|---|---|
| 2048x5120 | 71.2 | 82.2 (+15%) | 82.7 (+16%) | 83.2 (+17%) | 90.3 (+27%) |
| 2176x5120 | 90.9 | 82.5 (-9%) | 83.0 (-9%) | 83.7 (-8%) | 93.8 (+4%) |
| 5120x2176 | 88.0 | 89.3 (+2%) | 89.1 (+1%) | 95.7 (+9%) | 97.9 (+11%) |
| 1280x5120 | 59.6 | 57.6 (-3%) | 57.8 (-3%) | 67.2 (+13%) | 57.9 (-2%) |
| 768x5120 | 48.3 | 55.5 (+15%) | 54.6 (+13%) | 64.2 (+33%) | 44.3 (-9%) |
| 5120x768 | 60.0 | 61.5 (+2%) | 60.6 (+1%) | 71.6 (+19%) | 56.6 (-6%) |
| 1536x5120 | 59.8 | 70.5 (+18%) | 68.8 (+15%) | 68.2 (+14%) | 74.4 (+25%) |
| 256x5120 | 43.8 | 51.8 (+18%) | 53.4 (+22%) | 62.4 (+42%) | 33.5 (-24%) |

Five rows per wave dispatch 436 waves for 2176 rows instead of 544 (the D700 holds 512 at 4 per SIMD), so that
shard runs in one round; everywhere else the fewer waves hide less latency. With gate and up concurrent (g2) the
2176-row pair takes 130.3 µs instead of 132.6 (`c3i-n3-g2.jsonl`): the second dispatch already fills the slots
the first one's tail leaves. Two rows per wave (48 VGPRs, 5 waves) wins on the small shards and loses on the
large ones; a per-shape choice of 2 rows up to 1280 and 5 rows past 2048 would be worth about 1% of a verification
round and was not pursued.

**Activations in shared memory** (`kernels/qwen_q4cols_lds_body.glsl`, `kernels/qwen_q4cols_ldsx_body.glsl`): a
workgroup loads the 16 blocks of each column that one iteration covers (6 KB) cooperatively; lanes read their
two vec4 of a column from the tile right before use.

| shape | `qwen_q4c3` | columns outer, nibbles converted per column (48 VGPRs, 5 waves) | same, double-buffered tile (128 VGPRs) | rows outer, tile read per row (64 VGPRs) | rows outer, 3 rows (48 VGPRs) | first design (bytes merged by the compiler, 64 VGPRs) |
|---|---|---|---|---|---|---|
| 2048x5120 | 69.4 | 101.4 (+46%) | 115.9 (+67%) | 107.0 (+54%) | 120.2 (+73%) | 90.1 (+25%) |
| 2176x5120 | 88.9 | 115.2 (+30%) | 142.1 (+60%) | 132.3 (+47%) | 120.5 (+34%) | 119.0 (+27%) |
| 5120x2176 | 87.6 | 114.4 (+31%) | 164.7 (+88%) | 130.5 (+49%) | 128.7 (+47%) | 119.6 (+36%) |
| 768x5120 | 47.5 | 67.1 (+41%) | 64.0 (+35%) | 68.7 (+45%) | 57.3 (+21%) | 63.3 (+31%) |

(each column's percentages are against `qwen_q4c3` in the same run: `c3c-n3-g1.jsonl` for the first two,
`c3h-n3-g1.jsonl` for the next two, `c3a-n3-g1.jsonl` for the last.)

What went wrong, from the ISA (`RADV_DEBUG=shaders`): the compiler merges identical nibble conversions across
the columns and then holds every row's converted bytes (32 VGPRs) across the column loop, back at 64 VGPRs (first
design); masking each column's conversion with a push constant that is zero stops the merge and reaches 48 VGPRs,
but the three conversions cost 88 more VALU per iteration (+33%); the weight loads are sunk below the barrier to
their first use (nir_opt_sink), serializing their DRAM latency with the tile's, unless read through a `volatile`
view; the tile's `vmcnt(0)` wait covers every earlier load; and GFX6 has no `ds_read_b128`, so the tile reads are
twice as many `ds_read_b64`. With all of that fixed the 48-VGPR kernel still issues its VALU only about 60% of the
time: the two barriers per iteration couple the waves of a workgroup, which sit on four SIMDs shared with other
workgroups' waves. Reading the tile per row (no extra conversion, 24 reads per lane per iteration) is hoisted by
the compiler back into 64 VGPRs even with per-row opaque offsets.

**Other variants** (`c3b-n3-g1-k5120.jsonl`, `c3h-n3-g1.jsonl`, `c3e-*.jsonl`, `c3i-*.jsonl`):

| variant | 2048x5120 | 2176x5120 | 5120x768 | note |
|---|---|---|---|---|
| row length compiled in (`KFIX 5120`: constant row offsets) | 69.1 (-3%) | 90.7 (+0%) | - | the scale loads fold into immediate offsets, 3 VALU fewer |
| row 0's indices hoisted, other rows one add each (`ADDR`) | 70.9 (+2%) | 91.2 (+1%) | 61.2 (+2%) | 3 VALU fewer |
| all 12 reductions before the stores (`EPI`) | 70.6 (-1%) | 92.3 (+2%) | 61.5 (+2%) | same code size; the compiler did not interleave them |
| next iteration's weights loaded after row 2 through a volatile view (`PREF`) | 78.7 (+13%) | 103.4 (+17%) | 64.4 (+8%) | 64 VGPRs, 30 VALU more; also slower for 2 and 4 columns and for the decode kernels (`c3i-n1-*.jsonl`, `c3i-n2-g1.jsonl`, `c3i-n4-g1.jsonl`) |
| three rows per wave in the one-column decode kernels (`matvecq4_r3`, `_r3u2`) | - | 45.4 / 44.6 vs 45.3 (`r2u2`) | 24.7 / 34.9 vs 23.7 (`r2`) | no gain (`c3f-n1-*.jsonl`) |

Reference points at the same shapes (`c3d-n1.jsonl`, `c3d-n2.jsonl`, `c3d-n4.jsonl`): one column `matvecq4_r2u2`
43.8 µs at 2048x5120 (36 VGPRs, 7 waves), two columns `qwen_q4c2` 58.9 (48 VGPRs, 5 waves), four columns
`qwen_q4c4` 103.2 (84 VGPRs, 3 waves).

## The five-row kernel on the cluster

`qwen_q4c3_r5`: `qwen_q4cols_body.glsl` with `COLS 3, ROWS 5, WAVES 4` and `CLAMP_ROWS` (a shard that is not a
multiple of 20 rows clamps its phantom rows' weight reads to the last row; their results are not stored), 20 rows
per workgroup, taken for three columns when `2048 < rows <= 2560` (the FFN gate and up shards, 2176 rows).

| configuration | n | tok/s | pp512 | GPU wait | exchange | sum | total | encode | decode | wire | sent | first |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen_q4c3` (c3-mtp-r4) verify 44.6 ms/round | 4 | 51.94 (51.76-52.24) | 208 | 13.54 | 3.93 | 0.42 | 0.92 | 0.55 | 0.20 | 3.00 | 1.00 | 1.65 |
| `qwen_q4c3_r5` past 2048 rows (c3-mtp-r5) verify 45.2 ms/round | 4 | 51.30 (49.64-52.43) | 208 | 13.50 | 4.21 | 0.42 | 0.93 | 0.56 | 0.21 | 3.26 | 1.02 | 1.66 |

(tok/s per repetition, mean and range; the other columns are incite's milliseconds per generated token,
`tools/phases.py`.) All eight repetitions generated the record's 128 tokens (`token_ids` of every `.json`, the
same as round 2's). The GPU time per token is the same within 0.3%; the two slower `r5` repetitions are exchange
stalls (`wire` 3.26 against 3.00 ms), which the previous rounds also saw a few times per session. The profiled
runs (`incite/profile-c3-prof-*.jsonl.gz`, `tools/dispatches.py`, medians over 256 dispatches on both local ranks)
put the 2176x5120 projection at 117.1 µs with the five-row kernel against 121.5, and the gate-and-up pair at
137.3 against 138.4 µs; every other dispatch is unchanged (1280x5120 92.8 against 93.1, 5120x2176 88.0 against
88.0). A verification round has 64 such pairs: 0.07 ms of 44.6. Not kept.

### Gates (`trashcan-c3` with the five-row kernel active)

| gate | result |
|---|---|
| `qwen-validate` (14-token prompt, `TRASHCAN_TP_WIRE=f32`), `c3-val32` against the record's `f-val32` (`tools/compare_activations.py`, both on incite) | every one of the 258 tensors, the logits and the 10 greedy tokens identical (relative error 0) |
| `qwen-check --verify 3` | pass: no column mismatches, column score error 0, restore exact |
| `qwen-check --mtp` (600-token prompt, 64 steps) | pass: restore exact, speculative output identical, 83% of drafts accepted, column score error 0 |

## What this means for the next attempt

The verification projections are about 23 ms of a 44.6 ms round; the kernel is about 65% VALU-busy at 4 waves per SIMD
and would need 7 for full issue, which the register budget of three columns times four rows does not allow, and
the latency it waits for is the memory system's under load (a single wave alone needs about 2 µs per 16 blocks, of
which 1.1 is arithmetic). What was not tried, because it changes more than this kernel: a weight layout that
interleaves a wave's 16 blocks of four rows (one 1.3 KB stream per wave-iteration instead of eight 64-256 B
pieces, for every Q4_0 kernel and `repack_q4`), the bias partial sums precomputed by the host pass that writes the
normed rows (21 VALU per iteration, 9%, with the bit-exactness of the host's add order to prove), and per-shape
geometry (2 rows up to 1280 rows, 5 past 2048), worth about 1% of a round together.

## Round 3b: the three next steps, microbenchmarked (enlight lease #12, `trashcan-c3j`/`trashcan-c3k`)

Everything bit for bit `qwen_q4c3`'s (or `matvecq4_r2u2`'s) results; `microbench/c3k-*.jsonl` (`c3j-*` is the
first form of the tiled kernels, whose per-row index helpers cost 33 VALU per iteration and lost 5-21%).

**Tiled weight layout** (`kernels/tiled-layout-and-bias.patch`): groups of four rows contiguous, the blocks in
tiles of 16 per row, [tile][row of 4][block][word], so a wave's 16 blocks of four rows are one 256-byte run of
scales and one 1 KB run of nibbles, with uniform tile bases and constant row offsets for full tiles. The loop
bodies lose the per-row address arithmetic (three columns: 212 VALU per iteration instead of 229; two-row decode,
two tiles: 114 instead of 126).

| shape | `qwen_q4c3` | tiled | g2: `qwen_q4c3` | tiled | `qwen_q4c2` | tiled | `matvecq4_r2u2` | tiled | `matvecq4_r2` | tiled |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048x5120 | 70.7 | 66.9 (-5%) | 121.9 | 120.0 (-2%) | - | - | - | - | - | - |
| 2176x5120 | 91.3 | 87.7 (-4%) | 133.0 | 129.8 (-2%) | 66.8 | 65.9 (-1%) | 45.2 | 46.5 (+3%) | 47.3 | 49.7 (+10%) |
| 5120x2176 | 88.0 | 89.8 (+2%) | 155.1 | 153.2 (-1%) | 65.9 | 66.6 (+1%) | 47.2 | 47.6 (+1%) | 45.7 | 45.7 (-3%) |
| 1280x5120 | 59.0 | 55.7 (-6%) | 94.1 | 90.7 (-4%) | 48.8 | 46.8 (-4%) | 28.4 | 29.6 (+4%) | 30.7 | 31.5 (+11%) |
| 768x5120 | 47.8 | 45.7 (-5%) | 58.5 | 56.5 (-3%) | 38.7 | 36.8 (-5%) | 21.4 | 22.4 (+5%) | 23.6 | 24.6 (+15%) |
| 5120x768 | 60.4 | 67.0 (+11%) | 97.6 | 103.8 (+6%) | 41.7 | 46.5 (+11%) | 31.6 | 26.9 (-15%) | 23.1 | 25.5 (-19%) |
| 1536x5120 | 59.7 | 56.6 (-5%) | 95.5 | 92.3 (-3%) | - | - | 32.2 | 32.6 (+1%) | 35.7 | 36.0 (+12%) |

The three-column kernels gain 4-6% on the 5120-long rows (2-4% with two dispatches overlapping, as gate and up
run) and lose 11% on the 768-long rows (one full tile and a partial one); the decode kernels do not gain (the
engine uses `matvecq4_r2u2` at 5120 and `matvecq4_r2` at 2176 and 768). A verification round would save about
1% and a decode token nothing, for a per-matrix layout flag and tiled variants of every Q4_0 kernel the model
touches (`matvecq4_r2u2`, `qwen_q4c2/3/4`, `multivec4`, `gemm2`, `gemm`), whose prefill GEMM behaviour is
unmeasured. Not integrated.

**Zero-point biases precomputed by the producer of x** (`BIAS` in the patch: the kernel reads 8 x the sum of a
lane's eight activations, in its own order of additions, from buffer d instead of summing them): 364 VALU instead
of 377 for the shader, three more loads per iteration; 2048x5120 73.8 µs (+4%), 2176x5120 87.7 (-4%), 5120x2176
89.1 (+1%), 1280x5120 59.0 (0%), 768x5120 49.0 (+2%), 5120x768 60.1 (0%); with two dispatches overlapping +0 to
+9%. The loads cost more than the 13 VALU they replace. Not integrated (it would also need the host norm pass
and the gated-norm and SwiGLU kernels to write the biases).

**Per-shape row geometry**: measured above (two rows per wave: 768x5120 -9%, 5120x768 -6%, 1280x5120 -2%; five
rows at 2176 rows). The out-projections (5120x768, 64 per round) would save about 0.2 ms of a 44.6 ms round and
the five-row kernel 0.07 ms (measured on the cluster); the 768-row z projection runs concurrently with the
1280-row qkv projection, where two-row waves would exceed the 640 slots of 5 waves per SIMD. Below what an A/B
on the cluster resolves (the same configuration repeats within about 1%); not pursued.

## Files

- `tools/c3.sh`: the cluster phases (`mtp`, `profile`, `gates`); `tools/phases.py` (from round 2): the run table;
  `tools/dispatches.py`: per-dispatch medians of a profiled run; `tools/shaderstats_map.py`: names the pipelines of
  a `RADV_DEBUG=shaderstats` dump; `tools/table.py`: microbenchmark tables.
- `kernels/`: the shared-memory kernel bodies, the define-based variants of `qwen_q4cols_body.glsl` and
  `matvecq4_body.glsl` as they were measured, and the tiled-layout and bias patch (none in the engine).
- `microbench/`: every `qwen-matvec-bench` run (`c3a` … `c3i`, JSON lines) and the shader statistics of each
  binary.
- `incite/`: the runs' driver `.json`/`.err`, `commands.log`, the dispatch profiles, `clocks/`; `invite/`,
  `enlight/`, `indite/`: node logs and `clocks/`. The validation dump (`c3-val32`, 261 files) stays on incite under
  `~/trashcan-exp/tp8opt/results/`, beside the record's `f-val32`.
