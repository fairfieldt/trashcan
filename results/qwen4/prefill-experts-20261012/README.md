# Flash-Next prefill: expert kernels that reuse each expert's weights across its tokens (2026-10-12)

From 16 tokens a batch's expert slots are grouped by expert and each expert's rows are decoded once per tile of up to
8 slots, instead of once per (token, slot). Below 16 tokens (decode, MTP verification) the per-slot kernels
`qwen4_moe_gu`/`qwen4_moe_down` run unchanged.

Tested source: this branch, measured in three rounds as main moved: on 90cb371 (the two GPUs of a Mac on consecutive
chunks; stage times and round 1), rebased on 88def0d (chunk size chosen by prompt length; round 2) and on 9e0a420
(split-K prefill mixers; round 3, the combined code to merge). Baselines: `trashcan-main` (90cb371),
`trashcan-88def0d`, `trashcan-9e0a420`, on every Mac. G-iq4 (IQ4_XS gate/up, IQ4_NL down, the shared expert stacked
as expert 512).

## The work

Per token and layer the 10 routed experts and the shared one take 11 x (1280 x 2560 + 2560 x 640) = 54M weights,
108 MFLOP; 0.87 GFLOP per token for a Mac's 8 layers, 111 GFLOP per 128-token chunk: 32 ms at the D700's 3.48 TFLOP/s
for 8 layers run one GPU after the other (stage-bench sums both GPUs). The per-slot kernels read 2.66 MB of weights
per slot (3.7 GB per layer at 128 tokens, ~52% of 263 GB/s in the 221 ms measured for 8 layers).

Real text routes far fewer distinct experts than a random-token stream would ([route-stats.txt](route-stats.txt),
first 128 and 256 tokens of `long.ids`, the top-k ids of layers 0-7 dumped by a temporary capture): at 128 tokens 185-335 routed experts
per layer get a token (mean 3.8-6.9 slots each, up to 64), at 256 tokens 236-397 (mean 6.5-10.9, up to 163). Tiles
of 8 slots: 292-365 per layer at 128 tokens plus 16 for the shared expert, against 1408 per-slot workgroups.

## Kernels

| kernel | what it does |
|---|---|
| `qwen4_moe_tiles` | one workgroup: counts each expert's slots (LDS atomics), cuts them into tiles of 8 and a remainder, scans, writes the tiles fullest first (all full tiles, then remainders of 7, 6, ... 1) and the slots grouped by expert |
| `qwen4_moe_gu_tiles` | one workgroup per 128 gate/up row pairs of a tile; a lane owns gate row i and up row i and walks all 2560 columns, loading 4 blocks (one 64-byte line) of each row at once and decoding every IQ4 weight once (LDS table) for all of the tile's slots; the slots' inputs are the same for every lane, so they come through scalar loads and feed `v_mac_f32` as SGPR operands; one copy of the walk per slot count (1-8); SwiGLU into `act` as before |
| `qwen4_moe_down_tiles` | the same for the down rows (one row per lane, 640 columns), writing each slot's output times its routing weight for a range of rows into `big` (the DeltaNet/QSA scratch, free during the experts: no new VRAM) |
| `qwen4_moe_sum` | adds a token's 11 slot outputs in slot order; three passes of rows (1024, 1024, 512) fit `big` |

No sums cross lanes, so a tile costs one decode per weight (2 VALU and an LDS read) plus one FMA per weight and slot.
Layers with Q2_0 down projections (the 200K build) group the gate/up projections and keep the per-slot down kernel,
which reads the same `act` layout. New buffers: the tile list and the grouped slot list (17 KiB per GPU at 128-token
chunks).

### How it got there (4 layers on GPU 0 of alight, `--ranges 0:4 --gpus 0`, synthetic stream, 128-token chunk, GPU ms)

The variants were temporary shader switches, since removed; their stage-bench outputs are in `variants/`
(`tools/exp-var.sh`).

| variant | gate/up (incl. tiles) | down + sums |
|---|---|---|
| per-slot kernels (main) | 110.6 (both) | |
| first version: a lane per row, one 16-byte block per load | 66.1 | 39.7 |
| same, decoding only (no inputs, no FMAs; diagnostic) | 62.3 | |
| same, constant inputs (no scalar loads; diagnostic) | 53.1 | |
| 4 blocks (a whole line) per row per load | 40.7 | 21.1 |
| same with coalesced addresses (wrong data; diagnostic) | 40.1 | |
| same, staged through LDS with coalesced loads (64- / 128-lane workgroups) | 39.8 / 49.2 | |
| whole lines and one copy per slot count | 29.4 | 19.4 |
| + tiles fullest first (**kept**) | **27.4** | **14.4** |
| gate pass then up pass per line (84 VGPRs instead of 128; the compiler unrolled it to 140 KB of code) | 43.0 | |
| tiles of 4 slots (84 VGPRs) | 31.9 | 15.2 |

The decoding-only diagnostic costing as much as the full kernel showed the first version was bound by its loads: rows
are 1280 bytes apart (320 for down), so with one 16-byte block per load each 64-byte line was evicted before the
lane came back for its next block and was fetched up to four times. Reading whole lines fixed that; coalescing itself
did not matter once lines were used whole. The exact-count copies remove the per-slot branches and their serialized
scalar loads. The gate/up kernel holds 128 VGPRs (2 waves per SIMD, 29.5 KB of code), the down kernel 48 (5 waves);
both lower-register variants above were slower.

## Stage times (alight, layers 0-7 on both GPUs, 850/1370 MHz, on 90cb371)

```sh
trashcan qwen4 stage-bench G-iq4-00001-of-00006.gguf --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch B \
    --head-rows 0:21504,21504:43008 --decode 2 --start 64 --prefill 3      # B = 256: --context 8192
```

GPU ms per chunk, phases summed over the 8 layers (`alight/exp-sb-*-final.json`; clock logs in the `.err` files):

| chunk | experts, main | experts, new | | wall, main | wall, new |
|---|---|---|---|---|---|
| 16 | 28.2 | 20.5 | 1.38x | 90.0 | 82.2 |
| 32 | 55.8 | 32.5 | 1.72x | 137.7 | 114.8 |
| 64 | 110.9 | 50.6 | 2.19x | 236.4 | 175.1 |
| 128 | 221.2 | 84.2 | 2.63x | 419.2 | 279.8 |
| 256 (context 8192) | 441.7 | 145.3 | 3.04x | 783.5 | 483.4 |

At 8 tokens the two paths tie (4 layers: 7.2 ms both), so `GROUPED_MIN` is 16. At 128 tokens the experts now run at
38% of the FP32 roofline (32 ms) counting only their multiply-adds; the per-slot kernels ran at 14%. Stage-bench feeds
a fixed pseudo-random token stream; with the first 128 tokens of `long.ids` instead (4 layers, a temporary switch) the
experts took 43.0 against 110.6 ms, the same as on the synthetic stream (41.8). VRAM: +17 KiB per GPU.

## Correctness (six Macs, `qwen4 validate`, G-iq4)

| check | result |
|---|---|
| `p100.ids`, round 1 (one 100-token chunk), new against main | the same 10 greedy tokens and top-10; logits rel 4.5e-5, residual after layer 47 rel 7.4e-5, `ffn_out-0` rel 1.6e-7 ([compare-val100-exp-vs-main-r1.txt](compare-val100-exp-vs-main-r1.txt)) |
| the same against the CPU reference (`ref-p100`) | the same 10 greedy tokens and top-10; logits rel 3.48e-2 for both binaries, every captured tensor's error within 8% of main's ([new](compare-val100-exp-vs-cpu-r1.txt), [main](compare-val100-main-vs-cpu-r1.txt)) |
| first 300 ids of `long.ids`, round 1 (chunks of 128, 128, 44), new against main | the same 10 greedy tokens and top-10; logits rel 1.2e-3: within 5e-5 through layer 19, then growing at layers 20 and 24-27 ([compare-val300-exp-vs-main-r1.txt](compare-val300-exp-vs-main-r1.txt)) |
| the same against a CPU reference of the 300 ids (`qwen4 ref` on the M3 Ultra, 44 s) | the same 10 greedy tokens for both; logits rel 1.34e-2 (new) and 1.31e-2 (main), residual after layer 47 rel 2.30e-2 for both ([new](compare-val300-exp-vs-cpu-r1.txt), [main](compare-val300-main-vs-cpu-r1.txt)) |
| round 2 (16-token chunks; the last chunk of 4 or 12 tokens takes the per-slot kernels), new against main | p100: the same 10 greedy tokens, logits rel 5.6e-5; p300: the same tokens, logits rel 1.0e-4; against the CPU references both binaries at the same distance as in round 1 ([r2 files](compare-val100-exp-vs-main-r2.txt)) |
| round 3 (combined with the split-K mixers), p100 | the same 10 greedy tokens as the CPU reference, logits rel 3.48e-2 as for `trashcan-9e0a420`; new against it rel 6.7e-5 ([compare-val100-exp-vs-cpu-r3.txt](compare-val100-exp-vs-cpu-r3.txt), [compare-val100-exp-vs-main-r3.txt](compare-val100-exp-vs-main-r3.txt)) |
| stage dump of `p100.ids` on alight (layers 0-7, one chunk), new against main | `ffn_out` rel 1.6e-7 at layer 0 (identical inputs: summation order only), at most 1.4e-5 by layer 7 ([compare-stage-p100-exp-vs-main.txt](compare-stage-p100-exp-vs-main.txt)) |

The 300-token prompt's divergence in round 1 starts where earlier tokens' near-tied expert choices can flip (layers
20, 24-27, the pattern docs/qwen4.md describes against the CPU reference), and it is a tenth of either binary's
distance from the reference.

## End to end (six Macs, real text)

`qwen4 bench --head-split 12 --context 32768 --batch 128 --ngram disk --tokens-file long.ids` (8K: `long.ids`
repeated), alternating baseline and new; driver outputs in `incite/` (`*-c1` round 1, `*-c3` round 2, `*-c4`
round 3), node clock summaries in `nodes/` (`tools/exp-chain.sh` through chain6.sh; every GPU at 850/1370 MHz while
working). Prefill tok/s:

| prompt | round 1: 90cb371 / new (128-token chunks) | round 2: 88def0d / new | round 3: 9e0a420 / new |
|---|---|---|---|
| 512 (`--tg 64`) | 157.4, 158.0 / 231.8, 232.7 (+47%) | 259.6, 260.4 / 315.6, 316.0 (+21%; 32-token chunks) | 299.1 / 369.3 (+23%) |
| 2048 (`--tg 32`) | 323.1, 324.5 / 459.7 (+42%) | 363.9, 363.5 / 473.3, 472.3 (+30%; 64-token chunks) | 433.7 / 598.0 (+38%) |
| 8192 (`--tg 32`) | 447.9 / 618.9 (+38%) | 446.5 / 618.2 (+38%; 128-token chunks) | |

Decode is unchanged (13.2-13.4 tok/s after 2048 and 8192 tokens, 14.6-14.8 after 512, both binaries). Round 1's
second new run at 2048 tokens did not start (indite's node was not up when the driver connected). The gain grows with
the chunk size, as the stage times do: the adaptive chunks of 88def0d give 512-token prompts 32-token chunks, where the
experts are 1.7x faster, against 2.6x at 128.

**Chunk size.** 88def0d sizes chunks as sqrt(16 n / 11) because a chunk cost about as much as 16 more tokens. A line
through the 8-layer stage times above (on 90cb371, before the split-K mixers, which shrink the fixed part) gives
44 ms + 2.94 ms per token for main and 58 ms + 1.75 ms per token with the grouped experts: a chunk then costs about as
much as 33 more tokens, which would move the best chunk to about sqrt(33 n / 11) (the same powers of two for 512, 2048
and 8192 tokens, 64 instead of 32 for 1024). Worth re-fitting on the combined code.

## What is left

- The gate/up kernel holds 128 VGPRs for its 8-slot copy, which sets 2 waves per SIMD for every tile; the variants
  that used fewer registers lost more elsewhere. Splitting the counts over two pipelines (1-4 and 5-8 slots) would
  give small tiles more waves; not tried.
- Decoding an IQ4 weight costs 2 VALU and an LDS read (the table lookup), as much as two of its multiply-adds; tiles
  of many slots amortize it, but most of a 16-64-token chunk's tiles hold 1-3 slots.
- The router (f32, 2.1 ms per layer at 128 tokens) is now a fifth of the experts phase.
