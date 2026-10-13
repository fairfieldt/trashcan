# Flash-Next small-batch experts, router and top-k on the D700 (2026-10-13)

**Result.** Six-Mac decode (`qwen4 bench --pp 512 --tg 128`, real text) goes from **18.13 to 19.04 tok/s (+5.0%)**
against main at 7da6f93, three alternated runs each (18.11-18.17 against 18.91-19.21), and MTP decoding of the chat
prompts from **21.50 to 22.94 tok/s (+6.7%)** with the same tokens; prefill gains too (512 tokens 428 -> 454 tok/s,
2048 633 -> 673), from the router of the prompt chunks. Per layer and token the MoE dispatches take 272 instead of 330
us (one token) and 444 / 603 / 760 instead of 557 / 754 / 945 us for 2 / 3 / 4 tokens (MTP verification); the
router of a 128-token chunk takes 0.24 instead of 2.1 ms. Same math: at 1 and 3 tokens every captured tensor of 8
layers has main's bits; the greedy tokens are the CPU reference's; prompt chunks differ from main only in the router's
summation order (logits rel 4.5e-5 after 100 tokens on the chain).

The MoE phase of a layer for batches of 1-15 tokens (plain decode, MTP verification of 2-4 tokens, the last chunk of a
prompt): the router (f32 513 x 2560, at every batch size), `qwen4_topk`, the gate/up kernel `qwen4_moe_gu` (IQ4_XS
for G-iq4), the down kernel `qwen4_moe_down_w` (IQ4_NL), the shared expert (stacked as expert 512), and their
dispatch in `engine/src/qwen4/gpu.rs`.

Tested source: the single-Mac sessions 4-7 ran this branch on main 6f6635a (`trashcan-moe2`, sha256 00defb12...;
session 4 the version before the shorter top-k) against `trashcan-6f6635a` (fc3beaee...), sessions 1-3 on main
4c1346d against `trashcan-4c1346d`; the six-Mac session and session 8 ran it rebased on main 7da6f93 (`trashcan-moe3`,
f1a7c632...) against `trashcan-7da6f93` (89f495e0...).

## Single Mac

enlight, layers 0-7 on both D700s, 850/1370 MHz while computing (`enlight/s*/clocks/`):

```sh
TRASHCAN_Q4_PROFILE=1 scripts/with-clocks.sh LOG trashcan qwen4 stage-bench G-iq4-00001-of-00006.gguf \
    --ranges 0:4,4:8 --gpus 0,1 --context 32768 --head-rows 0:21504,21504:43008 --batch N --decode 32 --start 64 --prefill 8
```

Microseconds per dispatch, mean over layers 5-7 of the second GPU (`tools/table.py`): decode steps (one token at 64)
and the profiled chunk of N tokens at position 0 (the shape of an MTP verification batch). Session 7 ran main's and
the final binary alternately:

| one token | main | this round |
|---|---|---|
| router f32 513 x 2560 | 38.1 (`qwen4_matf32`) | 32.1 (`qwen4_router`) |
| `qwen4_topk` | 34.3 | 23.5 |
| `qwen4_moe_gu_iq4xs` (10 routed + shared) | 173.1 | 136.3 |
| `qwen4_moe_down_w_iq4nl` | 84.1 | 79.8 |
| MoE dispatches | 329.6 | 271.7 (-18%) |

| chunk of N | main: router, top-k, gate/up, down (sum) | this round | | 8 layers' wall, ms |
|---|---|---|---|---|
| 2 | 65.5, 34.0, 310.9, 146.6 (557.0) | 39.5, 23.3, 240.6, 140.2 (443.6) | -20% | 14.09 -> 13.04 |
| 3 | 67.9, 33.9, 448.3, 204.3 (754.4) | 40.1, 22.9, 342.5, 197.1 (602.6) | -20% | 17.09 -> 15.89 |
| 4 | 69.4, 34.0, 580.3, 261.2 (944.9) | 40.6, 23.1, 443.9, 252.0 (759.5) | -20% | 19.91 -> 18.48 |
| 16 | 460.7, 33.6 (router and top-k) | 125.2, 34.0 (`qwen4_router_gemm`) | MoE -11% | 76.78 -> 74.14 |

Each token after the first adds about 165 us of MoE per layer (main 205-227). A 128-token prefill chunk's router
takes 235 instead of 2149 us per layer (session 3; the 8 layers' chunk wall 238.8 -> 224.6 ms).

### Correctness

Stage dumps of layers 0-7 (`stage-bench --tokens-file --out`, every captured tensor of the last token) against main's
(`compare/`): the 14-token prompt in chunks of 1 and 3 gives **the same bits for all 98 tensors** (router logits,
routing weights, expert outputs and everything after them); in chunks of 5 (the router's tiles from 5 tokens) the
router logits differ in summation order (rel 4e-8 - 3e-7), the worst tensor (layer 7's attention output) by rel
1.3e-5; the 100-token prompt in chunks of 16 the same (worst 1.45e-5). No expert choice changed (routing weights
within rel 1.5e-7). The shorter top-k left all four dumps (chunks of 1, 3, 5, 16) bit-identical
(`compare/stage-topk-short.txt`). Rebased on 7da6f93 (session 8, `compare/stage-7da6f93.txt`): chunks of 1 and 3
again give main's bits for all 98 tensors, chunks of 16 differ only through the router's summation order (worst
tensor rel 1.5e-5); MoE dispatches 327.9 -> 271.9 us at one token, 756.2 -> 601.5 at three.

## Six Macs

Session c1 (`tools/chain-ab.sh c1 val tg mtp pp2048` under the six-Mac leases): `trashcan-moe3` (this branch on
7da6f93) against `trashcan-7da6f93`, G-iq4, 12 GPU stages, `--head-split 12 --context 32768 --ngram disk --batch 128`,
every GPU at 850/1370 MHz while computing (`*/clocks/`, the node logs). Results in `incite/`, node logs in the other
Macs' directories; `tools/chain_summary.py`, `tools/valcmp.sh`, `results/qwen4/mtp-20261012/summary.py`.

| run | main 7da6f93 | this branch | |
|---|---|---|---|
| decode, `bench --pp 512 --tg 128` on `long.ids` (3 alternated runs) | 18.13 tok/s (18.11-18.17) | 19.04 (18.91-19.21) | +5.0% |
| prefill of the same 512 tokens | 428.0 tok/s (425.4-429.8) | 453.5 (447.1-458.1) | +6.0% |
| `--pp 2048 --tg 32` (one pair) | 633.4 / 16.28 tok/s | 673.0 / 17.03 | +6.3% / +4.6% |
| MTP, `qwen4 mtp spec --draft-tokens 3 --tg 192 --runs 3`, mean of the three chat prompts | 21.50 tok/s (plain 18.37) | 22.94 (plain 19.29) | +6.7% |
| water / csv / transformer | 20.51 / 24.25 / 19.74 | 21.85 / 25.96 / 21.01 | |
| verification of 2 / 3 / 4 tokens | 91-93 / 108-109 / 128 ms | 85-87 / 100-101 / 118-120 ms | |

Both binaries generated the same 192 tokens for each chat prompt (`tools/mtp_same.py`), every run the plain decode's.

Validation (`compare/chain-c1-validate.txt`): the 14-token prompt one token at a time and as one chunk gives the
reference's greedy tokens (9419 0 2500 628 353 1438 488 3242 30 248046) with logits within rel 2.78e-4 of the CPU
reference (main: 2.78-2.81e-4 in results/qwen4/decode-kernels-20261012); the 100-token prompt (one chunk, so the
router in tiles) is within rel 4.49e-5 of main's logits (worst of 583 tensors 1.27e-4) and gives main's greedy
tokens; against the CPU reference main and this branch are both at rel 3.48e-2, as before.

## The changes

- **`qwen4_moe_gu`** keeps main's lane layout (lane = (block, word) over 5 steps of 16 blocks; 4 gate and 4 up rows per
  wave), so each lane's partial sums and the wave sums are main's bits, with ~35% fewer instructions per weight (3.7
  VALU instead of 5.8): the 4 lanes of a block each decode the IQ4_XS scale of one row pair and share it by quad
  broadcast (main decoded all 8 rows' scales in every lane: 2 header loads and ~12 VALU per row and word), and the
  rows' offsets are constants of the 2560-column shape instead of per-row address arithmetic. 48 VGPRs, 5 waves per
  SIMD.
- **`qwen4_moe_down_w`** keeps main's lanes (two per 32-weight block, the token's (slot, block) pairs flattened over the
  lane pairs, so the same bits), walks the pairs incrementally (main divided and multiplied 32-bit integers in the loop)
  with the rows' offsets constants of 640-wide rows, and now serves every small batch (main used the per-slot kernel
  from 9 tokens). Q2_0 down projections keep the per-slot `qwen4_moe_down`; 640-wide rows cannot be IQ4_XS, so its
  `_iq4xs` variant went.
- **`qwen4_router`** (1 and 2-4 tokens): `qwen4_matf32`'s per-lane order (the same bits) without clamped indices, so the
  loads take their offsets from the instruction; 48 VGPRs, so all 513 rows' waves are resident at once
  (`qwen4_matf32` held 64, and the 129th workgroup waited for a second round). 2-4 tokens: 66-69 -> 40 us.
- **`qwen4_router_gemm`** (5 tokens and more): 64 x 64 tiles as `qwen4_q8_gemm` on the f32 rows, the K range split in
  16 / ceil(N / 64) parts so the 9 row tiles fill the GPU; `qwen4_topk` adds the parts (kept in `big`, which the
  experts only use later) in order.
- **`qwen4_topk`**: the 10th largest lane maximum (a radix select over one key per lane, counted with ballots) bounds the
  picks from below; the candidates (usually 10-14) go to shared memory in id order and lanes rank them by (probability,
  id). The same probabilities, picks, order, sum and weights as main's ten rounds of wave-wide maximum and minimum.
  One wave runs it, and its time follows the code it runs (probes below): with the 32-step select kept as a loop,
  the compaction without branches and 8 keys per lane fixed (512 experts), it has 540 instructions instead of 919
  (`compare/topk-isa-blocks.txt`) and takes 23.5 instead of 30.8 us.
- **The batch as a push constant** for these kernels, which started with a dependent load of it: 4.3 us per layer.
- The GPU stage now checks the shapes these kernels assume (2560-wide residual streams, 640-wide experts, 512
  experts, at most 64 picks).

## Tried and removed

Sources in `removed/`. One token unless noted; gate/up and down in microseconds per layer (main 173 and 84):

| candidate | result | why |
|---|---|---|
| a lane per half block of every row, the 16 inputs in registers, sums through shared memory (`qwen4_moe_guh`, `_downh`; 2 x 160 or 448 lanes per workgroup) | gate/up 171-201, down 97-118 | 2-3 waves per SIMD (shared memory and registers), so latency-bound despite 4.2-4.8 VALU per weight |
| a lane per whole block (uvec4) of 4 row pairs | gate/up 236 | the inputs went through L1 four times per weight byte |
| prefetching the next step: everything (64 / 84 VGPRs) or only the words (48) | gate/up 144 / 149, down 102 / 101 | fewer waves; without prefetch 136 / 80 |
| the shared expert's gate/up dispatched alongside top-k | no gain | its 40 workgroups alone took 42 us |
| 2 gate and 2 up rows per wave (40 VGPRs, 6 waves) | 128 at 1 token, 347 against 342 at 3 | not kept: 7 us per layer at one token only |
| 2 rows per wave in down | 87 against 80 | twice the input loads |
| the next pair's expert id loaded early in down | 80.5 / 198.6 against 79.7 / 195.9 (1 / 3 tokens) | |
| 2-5 tokens: an expert's rows decoded once for the tokens that share it, in pairs from a grouping pass (`qwen4_moe_pairs`, `qwen4_moe_gu3s`); before that, grouped in-kernel (`gu3m`) | gate/up 313 / 468 / 559 against 253 / 342 / 444 at 2 / 3 / 4 tokens | 64-84 VGPRs; only 14-25% of a verification batch's routed slots share an expert (`tools/share_stats.py`) |
| grid order putting the workgroups of all slots of a row range together (L2 reuse of shared experts) | 721 against 729 us of MoE at 3 tokens | |
| top-k fused into the router's last workgroup (an atomic counter) | 60.1 against 31.5 + 30.8 at 1 token, 72.7 against 69.0 at 3 | 3 us per layer, not worth a second copy of top-k |
| top-k by a 32-step radix select over all 512 keys | 36.5 | |
| LUT offsets of four nibbles from one masked word (`(q << 2) & 0x3c3c3c3c`, a byte extract per nibble) | 250 against 234 VALU per step | the compiler re-masks every byte (it cannot see the low bits are zero) |

**Top-k probes** (session 5, `removed/topk-probes-x1.diff`: the first version returning after a stage): after the
loads 4.5 us, after the maximum 8.2, after the probabilities 11.4, after the threshold 17.4, the whole kernel 31.6.
Two top-k dispatches back to back took 30 us each (session 4), but a single workgroup need not run on the same compute
unit twice, so that did not rule out cold instruction fetch; the shorter code did cut 7 us. A first step
(`removed/qwen4_topk3.comp`: 8 keys per lane fixed, compaction without branches, the loop still unrolled) gave 28.7
against 31.6 us.

## What is left

Against this round's targets: one token's MoE dispatches take 272 us (target ~180), each token after the first
~165 us per layer (target ~110).

- The gate/up kernel runs at about 60% of the VALU rate (3.7 VALU per weight: 2 per nibble to address the IQ4 table in
  shared memory, 1.25 for the multiply-adds); its 19.1 MB would take ~85 us at a realistic bandwidth. Two gate and two
  up rows per wave (6 waves) gave 7 us at one token and lost at three.
- Down (80 us for 9.9 MB) is one round of 640 waves walking 7 dependent steps each.
- Top-k (23 us) is still one wave running about 400 instructions once (the candidates' compaction alone 180).
- 2-4-token batches: each token still reads its own experts (about 165 us per extra token and layer).

## Files

- `enlight/s1` ... `s8`: stage-bench JSON, stderr and clock logs of the single-Mac sessions (`s1`, `s2`: the
  half-block and v3 candidates; `s3`: prefetch, top-k and the prefill router; `s4`: the pair kernels and the last
  candidates; `s5`: top-k probes, 2 and 4 tokens; `s6`: the shorter top-k; `s7`: the final binary against main at
  1-4 and 16 tokens; `s8`: rebased on 7da6f93).
- `incite/`: the six-Mac session's driver output (bench and MTP JSON, stderr, clock logs, `commands-c1.log`);
  `indite/`, `inflight/`, `alight/`, `invite/`, `enlight/`: its node and clock logs.
- `compare/`: stage-dump comparisons, `topk-isa-blocks.txt` (top-k's instructions per basic block, before and after).
- `tools/`: session scripts (`sb.sh`, `s1.sh` ... `s8.sh`, `chain-ab.sh`, `gather.sh`, `valcmp.sh`), `table.py`,
  `prof.py`, `cmpdump.py`, `chain_summary.py`, `mtp_same.py`, `share_stats.py`, `iq4stats.py`, `vmstats.sh` (offline RADV statistics
  in the kbuild VM), `isablocks.py`.
- `removed/`: the candidate kernels that were removed.
