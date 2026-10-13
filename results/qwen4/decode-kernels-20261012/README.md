# Flash-Next decode kernels on the D700 (2026-10-12)

**Result.** Six-Mac decode (`qwen4 bench --pp 512 --tg 128`, real text) goes from **14.70 to 17.92 tok/s (+22%)**
against main at 9e0a420, three alternated runs each (14.64-14.74 against 17.81-18.03), and from 14.62 to 18.11 against
90cb371 (the round's first base); prefill is unchanged (pp512 298.0 / 298.5 tok/s, pp2048 430.3 / 434.3). On the chain
a DeltaNet layer takes 0.95-0.98 ms per decode token instead of 1.20-1.22, layer 1 (with the n-gram module) 1.19 instead
of 1.62, a QSA layer 1.11 instead of 1.45. The greedy tokens are the CPU reference's and main's; the decode path's
logits are within rel 1.7e-5 of main's. Batches of 2-4 tokens (MTP verification) are 7-8% faster than main, far from
the 1.3-1.4x-of-one-token target (now 1.56x, 1.84x, 2.21x for 2, 3, 4 tokens): each token still reads its own 11
experts. Rebased on b5b79c7 (main's grouped prefill experts), one confirmation run: 14.72
against 18.08 tok/s, the same validation results.

## Where a decode layer's time went

`TRASHCAN_Q4_PROFILE=1 trashcan qwen4 stage-bench ...` (new in this round) writes a GPU timestamp before and after
every dispatch of the decode graphs and prints each dispatch's time and the gap before it (`dispatches` in the JSON;
one prefill chunk's in `prefill_dispatches`). Layers 4-7 (the second GPU's stage of the first Mac), mean over 32
decode steps after 64 tokens; main's kernels from [stage-bench/dec-s1-prof.json](stage-bench/dec-s1-prof.json)
(enlight), this round's from [incite/dec-r3-sb-newp.json](incite/dec-r3-sb-newp.json). The dispatches of a deferred
group run concurrently, so a group's time is its last member's:

| dispatch | main µs | bytes | main GB/s | this round µs |
|---|---|---|---|---|
| mixer first half `qwen4_mix_down_q8` (324 x 10240), each of two | 92-117 | 3.5 MB | 30-38 | 50 (`qwen4_mix_down_sk`) |
| mixer second half `qwen4_mix_up_q8` (10240 x 320), each of two | 51 | 3.5 MB | 69 | 50 |
| qkv 10240 x 2560 + z 6144 x 2560 + alpha/beta f32 96 x 2560 (one group) | 273 | 45.5 MB | 167 | 230 |
| `qwen_conv`, recurrence, `qwen4_gated_norm` | 7 + 52 + 8 | 6.3 MB state | | 7 + 46 + 5 |
| out projection 2560 x 6144 | 143 | 16.7 MB | 117 | 116 |
| router f32 513 x 2560 | 58 | 5.3 MB | 91 | 39 |
| `qwen4_topk` (one wave) | 38 | | | 34 |
| `qwen4_moe_gu_iq4xs` (10 + 1 experts) | 175 | 19.1 MB | 109 | 173 |
| `qwen4_moe_down_iq4nl` | 113 | 9.9 MB | 88 | 84 (`qwen4_moe_down_w`) |
| QSA q/k/v group (12288 + 512 + 512 rows) | 204 | 36 MB | 177 | 187 |
| indexer f32 projections 128 x 2560, 512 x 2560 | 50, 58 | 1.3, 5.2 MB | | 19, 36 |
| stage end: pending combine (`qwen4_hc_norm`, one workgroup) | 49 | | | 6 (`qwen4_combine`) |
| layer 1: n-gram gate (`qwen4_ple_gate`, one workgroup per token) | 171 | | | 43 (`qwen4_ple_sums` + `qwen4_ple_gate`) |
| gap at a layer boundary (a submission per layer) | 11 | | | 1.4 (one command buffer per decode step) |
| gap before a dispatch (its barrier) | 1.0-1.4 | | | same |

A DeltaNet layer read its ~110 MB at about 90 GB/s, a third of the D700's 263 GB/s. The time went to kernels with too
few waves for the memory latency (the mixer's 324-row bottleneck as 21 workgroups; a combine and an n-gram gate in one
workgroup per token), to loads that wait for each other, and to serial single-wave work; the barriers themselves cost
about 1.2 µs each, and the 11 µs between layers came from submitting every layer as its own command buffer.

**What the instruction streams showed** (`RADV_DEBUG=shaders,nocache`, excerpts in [stage-bench/](stage-bench/)):

- **A conditional load costs a memory round trip.** ACO compiles `v = cond ? load : c` and loads under `if (z < cols)`
  into a branch around the load with an `s_waitcnt vmcnt(0)` at the join: `qwen4_topk` loaded its 8 logits in four
  serial round trips, `qwen_matf32` its activations and weights in two per iteration. Loading unconditionally (clamped
  index) and selecting afterwards issues them together: router 58 -> 39 µs, indexer projections 50 / 58 -> 19 / 36.
- **A load after a store to a buffer that may alias it waits for the store.** The gated norm, the mixer's second half
  and the combine read the next element after storing the previous one; loading everything first took the gated norm
  from 8 to 5 µs.
- **A single wave is slow at everything**: each subgroup reduction is five `ds_swizzle_b32` with a wait after each and
  two `v_readlane_b32`, and reading back weights it stored was a round trip each (top-k 38 -> 34 µs with them in shared
  memory). Unrolling top-k's ten picks made it 61 µs (larger code for one wave that fetches it cold after the barrier).
- `qwen4_moe_gu_iq4xs` issues about 370 VALU per loop iteration (8 nibble words, 64 IQ4 table lookups, 8 IQ4_XS scale
  decodes) at 6 waves per SIMD, yet halving the table lookups did not make it faster (below).

## The changes

- **Mixer first half split over the residual** (`qwen4_mix_down_sk`; `_sk4` reads the weights once for four tokens,
  2-8 tokens; main's fused mixers serve only batches of up to 8 since 9e0a420): workgroup (16 rows, 1280 residual
  columns) writes partial dot products and the chunk's sum of squares;
  the last workgroup of the token group to finish (an atomic counter it resets) sums them, applies the stream norms and
  the activations and writes `qwen4_mix_down`'s output, so `qwen4_mix_up` is unchanged. 168 workgroups instead of 21:
  92-117 -> 50 µs per mixer.
- **Q8_0 matrix-vector kernels with 8-byte loads** (`qwen4_q8_mvw4_r4`, `_r2`, `_c4` for 2-8 tokens, replacing
  `qwen4_q8_mv`): four lanes per 32-byte block, each loading 8 weights as one uvec2; 4 rows per wave for matrices of at
  least 4096 rows, 2 for the rest (the 2560-row out projections get twice the waves).
- **Decode projection groups start the small matrices first**: the last dispatch of a deferred group starts its waves
  last, and the 96-row alpha/beta matrix dispatched after qkv and z was the group's tail.
- **`qwen4_gdn`** (up to 8 tokens): the recurrence with lane l holding key dims 2l and 2l + 1 (one vec2 per lane for
  state, q and k) and two state rows per wave: 52 -> 46 µs (one and four rows: 52 and 51).
- **`qwen4_moe_down_w`** (up to 8 tokens): two lanes per block with uvec2 loads, and the (slot, block) pairs of a token
  flattened over the lane pairs, so ff = 640 (20 blocks) no longer leaves 3/8 of the lanes idle in a second pass:
  113 -> 84 µs.
- **`qwen4_matf32`** (1 and 2-4 tokens): f32 projections with every load issued before the arithmetic, in
  `qwen_matf32`'s per-lane order (the same bits).
- **`qwen4_topk`**: unconditional logit loads, the picked weights in shared memory (same selection and sums).
- **Loads before stores** in `qwen4_gated_norm`, `qwen4_mix_up`'s epilogue and `qwen4_combine`.
- **`qwen4_combine`** for the pending combine without a norm (stage end, before the n-gram module): 49 -> 6 µs.
- **`qwen4_ple_sums` + `qwen4_ple_gate`**: the n-gram gate's 13 sums over 10 workgroups per token, then the gate over
  10 workgroups: 171 -> 43 µs (layer 1).
- **One command buffer per decode step** (batches of up to 8; prefill keeps one per layer for amdgpu's job timeout).

Numerics: summation order only (the f32 projections, top-k selection, IQ4 decoding unchanged). Decode results stay the
same bits at every batch size (chunks of 1, 3 and 5 give identical dumps), as main's do.

## Tried and removed

Temporary switches on the stage benchmark (per-dispatch µs, [stage-bench/](stage-bench/)):

| candidate | result |
|---|---|
| the convolution folded into the qkv projection's epilogue (a dispatch and a barrier less) | qkv +14 µs, the group no faster |
| Q8 kernels with dword loads and 2 or 1 rows per wave; 16-byte (uvec4) loads | 2 rows: group 260, out 133 µs; 16-byte loads: group 269 |
| the mixer's second half summing the split partials itself in every workgroup | 69 µs (97 with 8 lanes per row) |
| 2 rows per wave in the split first half | 39-45 against 42 µs |
| top-k ranking the candidates at or above the 10th largest lane maximum | 44 µs (shared-memory loops and barriers in one wave) |
| a one-token f32 kernel with x in shared memory | router 72 against 56 µs |
| IQ4 values from a table of both nibbles of a byte (4 lookups per word), precomputed row indices, per-block scale shifts | gate/up 171 against 173 µs; down 92 against 84 |
| gate/up with uvec2 loads | 184 against 183 µs |
| 2-4 tokens: each expert's rows read once for every token routed to it (`gu4`, `down4`, distinct experts in shared memory) | 3 tokens: 829 + 564 µs against 3 x (172 + 88) |
| the mixer's second half for four tokens with 8 lanes per row | 52 µs for one token, no faster |
| `attention_merge_u` for QSA | 11 µs either way |

## Correctness

| check | result |
|---|---|
| layers 0-7 on one Mac (`stage-bench --tokens-file`), against main | the 14-token prompt in chunks of 1, 3 and 5: every tensor within rel 6.2e-6, and chunks of 1, 3 and 5 give identical bits; the 100-token prompt in chunks of 16: within 2.0e-5 |
| six Macs, `qwen4 validate`, the 14 tokens one at a time (`--batch 1`, every decode kernel) | greedy 9419 0 2500 628 353 1438 488 3242 30 248046 (the reference's); logits rel 2.78e-4 to the CPU reference (main 2.81e-4), 1.7e-5 to main |
| six Macs, the 14 tokens as one chunk | the same tokens; logits rel 2.84e-4 to the CPU reference (main's chain 2.85e-4) |
| six Macs, the 100-token prompt (16-token chunks on 9e0a420) | the reference's greedy tokens; logits rel 2.1e-5 to main's (the same 3.48e-2 from the CPU reference as main) |

[compare-r5.txt](compare-r5.txt) (on b5b79c7: the 14-token and 100-token prompts as above, 100 tokens within rel 6.7e-5
of main's logits), [compare-r4.txt](compare-r4.txt) (on 9e0a420), [compare-r3.txt](compare-r3.txt), [compare-r2.txt](compare-r2.txt) (on
90cb371; the final binary's 100-token dump there was bit-identical to the first's).

## End to end (six Macs)

`tools/q4run.py` with `--head-split 12 --context 32768 --batch 128 --ngram disk --tokens-file long.ids`, alternated
with main's binary on every Mac. r5: this commit's tree (on b5b79c7, `trashcan-decb`, sha256 11a88088…) against
`trashcan-b5b79c7`. r4: this branch rebased on 9e0a420 (`trashcan-dec`, sha256 bb4fc14e…) against
`trashcan-9e0a420`, prefill chunks chosen by prompt length (32 tokens for 512, 64 for 2048). r3 and r2: on 90cb371
(128-token chunks); r3 the same kernels, r2 before the loads-before-stores change and with the decode projection order
also used in prefill:

| | main tok/s (pp / tg) | this round tok/s (pp / tg) |
|---|---|---|
| r4 pp512 / tg128 (3 runs) | 298.0 / **14.70** (14.64-14.74) | 298.5 / **17.92** (17.81-18.03) |
| r4 pp2048 / tg32 (2 runs) | 430.3 / 13.48 | 434.3 / 16.06 |
| r3 pp512 / tg128 (3 runs) | 157.6 / 14.62 (14.61-14.64) | 157.5 / 18.11 (18.10-18.13) |
| r3 pp2048 / tg32 (2 runs) | 322.3 / 13.35 | 323.4 / 16.10 |
| r2 pp512 / tg128 (3 runs) | 158.0 / 14.65 (14.61-14.70) | 156.9 / 18.02 (17.92-18.12) |
| r2 pp2048 / tg32 | 324.0 / 13.21 | 320.2 / 16.16 |
| r5 on b5b79c7, pp512 / tg128 (1 run) | 368.7 / 14.72 | 372.1 / 18.08 |

incite's 8 layers per decode token (r4, GPU ms, `decode_layer_ms`): 1.21 1.62 1.22 1.45 1.20 1.21 1.22 1.45 against
0.98 1.19 0.95 1.11 0.97 0.95 0.95 1.12. Clocks 850/1370 MHz while computing on every Mac (`*/clocks/`).

## Single Mac (incite, layers 0-7 on two D700s, stage-bench, r4: against 9e0a420)

| | main | this round |
|---|---|---|
| decode at 64 tokens: wall per token / GPU ms of the 8 layers | 11.28 / 10.45 | 9.06 / 8.07 |
| per phase (DeltaNet layer): mixer, DeltaNet, mixer, router, experts | 0.17, 0.49, 0.17, 0.10, 0.29 | 0.10-0.12, 0.41, 0.10, 0.075, 0.26 |
| QSA layer / layer 1 (n-gram) | 1.38 / 1.62 | 1.04 / 1.19 |
| decode at 32K (r3, against 90cb371): GPU ms of the 8 layers; QSA layer | 12.07; 2.18 | 9.64; 1.83 |
| prefill chunk of 16 / 32 / 64 / 128 tokens, wall | 84.7 / 120.6 / 193.9 / 375.0 | 84.4 / 120.4 / 193.8 / 373.9 |

## 2-4 token batches (MTP verification)

Chunk of N tokens at position 0 (`--batch N --prefill 5`), wall for the 8 layers against the one-token decode wall
(r4, against 9e0a420):

| tokens | main wall | this round wall | ratio to one token (main / this round) | this round: mixers, DeltaNet, router, experts (GPU ms, 8 layers) |
|---|---|---|---|---|
| 1 | 11.28 | 9.06 | 1 / 1 | 1.7, 2.5, 0.6, 2.1 |
| 2 | 15.26 | 14.15 | 1.35 / 1.56 | 3.0, 3.9, 0.8, 3.7 |
| 3 | 18.13 | 16.70 | 1.61 / 1.84 | 3.9, 4.1, 0.8, 5.2 |
| 4 | 21.47 | 20.05 | 1.90 / 2.21 | 4.8, 4.4, 0.8, 6.7 |

The dense projections read their weights once per batch and so does the mixers' first half, but DeltaNet still grows
from 2.5 to 4.4 ms for 4 tokens (the multi-token matrix kernels hold 84 registers, 3 waves per SIMD, and the recurrence
walks the tokens in turn); the experts (each token's 10 routed experts and the shared one, 2.1 ms per token) and the
mixers' second half (`qwen4_mix_up`, once per token) grow with the batch. Reading an expert once for the tokens that
share it saves little at these sizes (routed experts rarely coincide; the kernels that did it were slower, above); a
four-token second mixer half would save about 0.1 ms per mixer at 4 tokens. From 16 tokens main's grouped expert tiles
(b5b79c7) take over.

## Files

- `incite/`: driver outputs of the six-Mac runs (`dec-r2-*`, `dec-r3-*` on 90cb371, `dec-r4-*` on 9e0a420, `dec-r5-*`
  on b5b79c7), the single-Mac runs on incite during them (`*-sb-*`, `*-pf*`), `commands.log`; `indite/` ... `enlight/`: node logs; `*/clocks/`.
- `stage-bench/`: single-Mac stage-bench JSON of the candidate rounds (`s1`, `s2` enlight, `s3` invite, `s4` indite),
  shader statistics and ISA excerpts.
- `tools/`: `summary.py` (the A/B table), `chain1.sh` ... `chain4.sh` (the six-Mac sessions, under chain6.sh), `sb.sh`
  (stage-bench runs), `disp.py`, `cmp.py`, `pf.py`, `wall.py`, `sbsum.py` (profile tables), `stats.py`, `isa.py`
  (RADV statistics and disassembly by pipeline), `valcmp.sh`, `valcmp3.sh`, `valcmp4.sh`, `gather.sh`.
