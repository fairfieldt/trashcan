# Flash-Next decode kernels, round 2 (2026-10-14)

**Result.** On six Macs (all on Linux 7.3, the links on USB4STREAM), against main 9baedca9: decode after a 512-token
prompt 21.46 -> 22.57 tok/s (+5.1%), after 2048 tokens 20.73 -> 21.36 (+3.0%), after 8192 20.23 -> 21.46 (+6.1%),
prefill unchanged (one pair each, every link on its stream in both arms, after the stream fix); MTP chat 30.05 ->
31.28 tok/s (+4.1%) with the same tokens; validation bit-identical to main.

One D700 (indite GPU 0, layers 4-7: three DeltaNet layers and one QSA layer, decode after 512 tokens, s10): the stage
takes 3.645 ms on main 9baedca9 and 3.430 ms with this round's changes (-5.9%; 3.577 ms with the two kernel changes
alone). Three changes, all with the same arithmetic as main (every captured tensor the same bits, below):

- **Light barriers in decode-sized graphs** (up to 8 tokens): a dispatch ends with a barrier from shader writes to
  nothing instead of to shader reads. On GFX6, RADV adds a scalar-cache invalidation to every barrier whose
  destination is a shader read; without it the next layers' kernels find their descriptors and constants in the
  scalar cache (weight streaming never evicts it), and most kernels after the first layer of a submission run 1-10
  us faster: -4.1% of the stage. Decode kernels read nothing an earlier dispatch wrote through scalar loads (checked
  in the compiled programs), and the engine checks the barrier at startup.
- **Gate/up with 2 + 2 rows per wave for one token** (`qwen4_moe_gu2`): 141 -> 131 us per layer.
- **The QSA layer's post-projection kernels in one group**: ~44 -> ~15 us per QSA layer.

## Tested source

- Branch `worktree-agent-a70283bc249c19df0` on main 20899d88 (the engine as 9baedca9). Candidate binary
  `trashcan-dk2-c2` (sha256 0747bff75e0eb78f..., built from the engine tree of commit 79006eba), against
  `trashcan-9baedca` (6787fd3911fc8a86..., main's binary on all six Macs).
- Mesa 25.0.7-2+deb13u1 (RADV/ACO) everywhere. The six-Mac sessions dk2c2 (13:40-13:45) and dk2c2r (13:48-13:50,
  stopped) ran on Linux 7.3.0-rc7-p2p on all six Macs, dk2c2p (18:20, stopped after one run; 18:23-18:24) on
  7.3.0-rc7-p2p-s2. Single-Mac sessions on indite GPU 0 (`stage-bench --ranges 4:8
  --context 32768 --batch 128 --start 512 --decode 64`): s1-s7 on 7.2.9-p2p against 15e5457 (the measurements and
  kernel candidates), s8-s11 on 7.3.0-rc7-p2p (s9-s11 against 9baedca); the stress checks on indite (c1, 7.2.9) and
  enlight (c2, 7.3.0-rc7). Every timed run held 850/1370 MHz while computing (clock logs next to the JSON); A/B arms
  alternate within a session.
- An earlier six-Mac session (dk2c1, 12:57-12:59, mixed kernels: inflight, alight and indite on 7.3.0-rc7, the others
  on 7.2.9) compared the two kernel changes alone (c1 on e155a4f7) with `trashcan-15e5457` over TCP: decode after 512
  tokens 21.16 -> 21.29 tok/s (+0.6%; pairs +1.4%, -0.2%), after 2048 +2.4%, after 8192 +1.2%, MTP chat 29.77 ->
  29.88 (+0.4%), the same MTP tokens; validation bit-identical to main's dumps (`chain/dk2c1`).

## Where a decode layer's time goes (main 15e5457)

Practical ceiling first. `trashcan qwen4 barrier-bench` (new, a diagnostic) streams 256 MiB through a D700 with a
grid-stride loop: 203-205 GB/s with uvec2 or uvec4 loads from 320-1280 workgroups (191 at 2560), so **about 205 GB/s,
78% of the 263 GB/s peak, is the practical read rate**. Matrix rows read as the decode kernels read them (4 rows of 512
bytes per wave and step, as the Q8_0 kernels; 8 rows of 256, as gate/up) reach 198 / 193 GB/s, and the same bytes laid
out so that a wave's step is contiguous 202 / 195 GB/s: the weights' layout costs about 2%.

Per dispatch, `TRASHCAN_Q4_PROFILE=1`, mean of 64 decode steps after 512 tokens (s1, `indite/s1/main-p.json`;
`tools/prof.py`). A deferred group's members overlap (its time is its last member's); the gap is the time between the
previous dispatch's end and this one's start (the barrier and the launch). Bytes are the weights and the recurrent
state a kernel streams; code is the final program's size (RADV, Tahiti):

| DeltaNet layer 5 | us | gap | MB | GB/s | code |
|---|---|---|---|---|---|
| mixer 1, first half `qwen4_mix_down_skw_c1` | 48.1 | 0.3 | 3.53 | 73 | 3152 |
| mixer 1, second half `qwen4_mix_rows_up_c1` | 42.6 | 1.1 | 3.48 | 82 | 2192 |
| alpha/beta f32 + z + qkv Q8_0 (one group) | 241.8 | 1.4 | 45.5 | 188 | 1612, 1740 |
| `qwen_conv` | 7.3 | 1.0 | | | 456 |
| `qwen4_gdn` (state read and written) | 47.4 | 1.9 | 6.29 | 133 | 1740 |
| `qwen4_gated_norm` | 5.2 | 3.4 | | | 380 |
| out projection `qwen4_q8_mvk2_c1` | 102.8 | 1.1 | 16.71 | 163 | 1120 |
| mixer 2 (two halves) | 47.1 + 41.7 | 2.4 | 7.0 | 79 | |
| router f32 `qwen4_router` | 33.6 | 1.4 | 5.25 | 156 | 828 |
| `qwen4_topk` (one wave) | 23.2 | 1.0 | | | 2700 |
| gate/up `qwen4_moe_gu_iq4xs` (10 routed + shared) | 140.2 | 1.3 | 19.15 | 137 | 3652 |
| down `qwen4_moe_down_w_iq4nl` | 84.9 | 1.0 | 10.14 | 119 | 3044 |
| **layer** | **884.7** | **18.7 (14 barriers)** | **117.1** | **132** | |

The same layer at the practical rate would take 571 us; the 314 us over it are mostly the mixers (180 us for 14 MB:
113 us over), the experts (225 us for 29.3 MB: 82 over), the small kernels around the recurrence (conv, gated norm,
top-k: 36 us) and the barriers' gaps (19 us). The projections (91% of the practical rate), the out projection (80%)
and the router (76%) are closer to it.

The QSA layer 7 (970.6 us; 22 barriers, 26.3 us of gaps) has the same mixers and MoE, the q/k/v and indexer
projections as one group (230.6 us for 42.8 MB, 186 GB/s), then q/k/v norms 15.4, indexer key store 5.0 + 1.3,
pooling 7.8 + 1.3, indexer query norm 12.7 + 1.0, block scores 4.2, selection 5.5, attention
`qwen4_attention_grouped_dec32` 71.7, merge 13.9, gate 4.4, out projection 102.6, the stage-end combine 5.5.

**Code size.** The decode kernels are 0.3-3.7 KB each, ~23 KB per DeltaNet layer. Recording every dispatch twice
(`TRASHCAN_Q4_DIAG=twice`, a temporary switch) showed no gain for the second copy with main's barriers (topk 23.3 / 22.7
us, gate/up 140.0 / 140.3, mixer halves within 1.5 us): code fetch is not a cost of its own.

## Barriers

Main ends every dispatch with a pipeline barrier from shader writes to shader reads and writes. RADV turns it into a
wait for idle plus an L2 write-back and invalidation (the shader-write source on GFX6), an L1 invalidation and a
scalar-cache invalidation (any shader-read destination: ACO may load storage buffers through the scalar cache).
`barrier-bench` times a one-workgroup kernel followed by each kind, and its cache probe reads a buffer from 1280
workgroups through the vector and scalar caches, overwrites it from one workgroup after an execution-only barrier, then
runs the barrier under test and reads again (s10, `indite/s10/barrier.json`):

| barrier after each dispatch | us per dispatch | stale vector reads (16 KB / 1 KB probe) | stale scalar reads |
|---|---|---|---|
| main's: shader write -> shader read and write | 3.48 | 0 / 0 | 0 |
| execution only | 1.43 | 0 / 1240 of 1280 | 1280 |
| none -> uniform read ("lite": L1 and scalar cache) | 2.35 | 0 / 0 | 0 |
| none -> indirect-command read (scalar cache only) | 2.09 | 0 / 1240 | 0 |
| **shader write -> none ("light": L2 and L1, not the scalar cache)** | **2.58** | **0 / 0** | 1280 |
| none (dispatches overlap) | 0.33 | | |

(With 16 KB every workgroup reads its own 1 KB, and a CU rarely has the lines it reads again; with 1 KB every CU has
them, and only the 1 KB probe sees what an execution-only barrier leaves in the L1.)

Decode of layers 4-7 with c1's kernels and each barrier after every decode dispatch (s9, a temporary switch; GPU ms,
two runs each):

| barrier | stage | layer 4 / 5 / 6 / 7 |
|---|---|---|
| main's | 3.580 | 0.890 / 0.875 / 0.877 / 0.938 |
| execution only (stale data: a diagnostic) | 3.358 (-6.2%) | 0.870 / 0.807 / 0.802 / 0.880 |
| lite: L1 and scalar cache | 3.570 (-0.3%) | |
| scalar cache only | 3.560 (-0.6%) | |
| light: L2 and L1, not the scalar cache | 3.420 (-4.5%) | 0.882 / 0.821 / 0.816 / 0.899 |

So the cost is the scalar-cache invalidation, not the barrier operation itself: the gain is inside the kernels of the
second and later layers of a submission (the first layer's kernels run once per submission, which starts with a full
invalidation). Layer 5 with light barriers against main's (s10 profiles, c2 against c1): the f32 alpha/beta matrix
30.5 -> 21.3 us, `qwen4_gdn` 47.3 -> 40.0, `qwen_conv` 7.2 -> 5.3, down 85.1 -> 75.0, gate/up 130.7 -> 124.9, the out
projection 103.8 -> 98.0, the router 33.3 -> 30.5, the mixer halves 1-4 us each; the gaps between dispatches stay at
1.0-1.4 us. A kernel's start loads its descriptors, push constants and constants through the scalar cache; weight
streaming goes through the vector caches and leaves the scalar cache alone, so with the light barrier those loads hit
from the previous layer on, while after main's barrier they wait for memory (the L2 is invalidated too). The repeat
probe agrees: `qwen4_topk` dispatched back to back takes 23.6 us with main's barriers, 15.4 with lite ones (L2 kept)
and 12.7 with light ones (scalar cache kept).

This corrects the first reading of step 1 (that the execution-only gain came from stale L1 lines): with every
activation binding `coherent` (glc loads, no scalar loads of activations, dumps identical), execution-only barriers
gained the same 236 us per stage with the same per-layer shape (layer 4 -23 us, layers 5-7 -67 to -75 us, s2), but
glc loads bypass the L1 that the matrix kernels use for their input vectors (the out projection 103 -> 148 us, the
mixer's first half 48 -> 68, down 85 -> 109), so that build was 8.7% slower than main.

## What changed

- **Light barriers** (`Device::compute_barrier` mode 3, `Device::light_barriers`). `GpuStage::record` records graphs
  of up to 8 tokens (decode, MTP verification; one command buffer per stage) with a barrier from `SHADER_WRITE` to no
  access after each dispatch or deferred group. Copies, host transfers, inter-GPU copies and the start of every
  command buffer keep full barriers, and the recurrent state written in one step is read in the next submission. By
  the Vulkan memory model this barrier makes the writes available but not visible; on GFX6 RADV writes the L2 back,
  invalidates it and invalidates the L1 for the shader-write source, so later vector loads see the writes and only
  scalar loads may not. The conditions:
  - **No scalar loads of what an earlier dispatch wrote.** The kernels that may run after a light barrier are listed in
    `SCALAR_SAFE` (engine/src/vulkan.rs); any other kernel gets a full barrier first. `tools/coherence_audit.py --rule
    smem` traces every buffer access in RADV's compiled programs (Tahiti, `tools/vmstats.sh`) to its descriptor set
    and binding: set 2 holds everything a dispatch writes (its params binding is written only by the graph's leading
    copy, behind a full barrier), set 0 the weights. All 82 listed kernels pass (`audit/c2-scalar-safe.txt`; every
    pipeline: `audit/c2-all-pipelines.txt`, the 24 that fail are prefill or other models' kernels, which run behind
    full barriers). Main's decode kernels loaded through the scalar cache the decode attention's queries (16 loads per
    head), the 64 bottleneck values per token of the mixer's second half, and single values in seven kernels (expert
    ids, routing weights, gates): these now go through `coherent` views (glc vector loads, common.glsl) or, for the
    bulk uniform reads, one vector load per lane and `subgroupBroadcast` (`v_readlane`): the same values in the same
    arithmetic.
  - **A startup self-test** (`Device::light_barriers_ok`, ~1 ms once per device): the 1 KB cache probe with a light
    barrier. A stale vector read is logged ("light barriers: N of 1280 workgroups read stale data in the self-test;
    full barriers") and the device keeps full barriers. It detects a missing L1 invalidation (1240 of 1280 stale with
    an execution-only barrier, above); it does not check scalar loads, which the audit covers.
  - The audit and the stress check are to be rerun after any Mesa upgrade and after changes to the listed shaders
    (docs/qwen4.md, AGENTS.md).
- **`qwen4_moe_gu2`** (gate/up, one token, IQ4_XS): 2 gate and 2 up rows per wave and 8 rows per workgroup instead of
  4 + 4 and 16, so twice the waves with 40 registers (6 waves per SIMD instead of 5); the same per-lane sums and wave
  sums (the same bits). 140.8 -> 130.7 us per layer (s3, s4). 2-4 tokens keep `qwen4_moe_gu` (the 4-row kernel reads
  each token's inputs half as often; at 3 tokens the 2-row one was slower: results/qwen4/decode-experts-20261013).
  This variant was measured in that round (128 against 136 us) but not kept for 7 us; on main it gives 10 us per layer.
- **QSA: one group from the projections to the block scores.** `qwen4_idx_store`, the indexer query norm (`qwen_qk`) and
  `qwen4_idx_pool` join the attention's q/k/v norm group (the v norm defers its barrier when the cache is f16 and the
  indexer follows); `qwen4_idx_pool` reads the positions of the current batch from the indexer key projection instead
  of the ring (the values `qwen4_idx_store` copies there in the same group), so nothing in the group depends on
  another member. Four groups and barriers become one: ~44 -> ~15 us per QSA layer (s7). Q8 K/V caches keep the
  barrier before `qwen4_kv_store`.
- **`trashcan qwen4 barrier-bench`** (diagnostic, with `qwen4_stream_probe` and `qwen4_cache_probe`): the streaming
  and matrix-row read rates, the cost of each barrier kind, and which caches a barrier leaves stale.

One D700 (s10, layers 4-7, GPU ms; two alternated runs each, then profiled):

| | main 9baedca | c1 (the two kernel changes) | c2 (c1 + light barriers) |
|---|---|---|---|
| decode after 512 tokens | 3.642, 3.647 | 3.577, 3.577 (-1.9%) | 3.432, 3.427 (-5.9%) |
| per layer (4 / 5 / 6 / 7) | 0.898 / 0.883 / 0.884 / 0.977 | 0.889 / 0.874 / 0.875 / 0.939 | 0.883 / 0.826 / 0.823 / 0.900 |
| decode after 4096 tokens (profiled) | | 3.743 | 3.609 (-3.6%) |
| 4-token chunks after 508 tokens (verification-shaped) | | 7.344 | 7.147 (-2.7%) |

The decode attention pays 1.7 us for broadcasting its queries (73.6 against 71.9 us); everything else is faster.

## Tried and removed

Per dispatch on indite GPU 0 (us; main in brackets). Sources in `removed/`.

| candidate | result | why |
|---|---|---|
| MoE down for one token, 5 rows per wave (512 waves, one round), the next step's weights loaded during this step's products (a copy through `xor p.flags` at the top of the loop) | 87.6 (75.0, with light barriers; same bits) | 64 VGPRs, 4 waves per SIMD; ACO still copies the prefetched registers at the loop's end and waits for them there |
| the same, the 7 steps unrolled without branches | not run | ACO waits for every step's loads before the next step's (no overlap at 48 VGPRs), and the code is 19 KB |
| mixer first half, one token: each step's weights and inputs loaded before the previous step's products | 52.8 (48.1) | ACO waits for every load at the end of the loop body to copy the prefetched registers into the next iteration's (84 VGPRs, 3 waves per SIMD) |
| mixer second half: combine and norm before the products, xn in LDS | 59.3 (42.5) | the weight loads moved below the epilogue's branch: three round trips before the weights were requested |
| mixer second half: bottleneck values through LDS instead of 16 scalar loads per token | 44-45 (42-43) | 64 VGPRs, 4 waves per SIMD: two rounds of workgroups |
| mixer first half, one step per wave, the steps' partial sums combined in LDS in the original order (same bits) | 66 (48) | 84 VGPRs, 1620 waves in 5 rounds |
| both mixer halves in one dispatch, the second half's workgroups loading their weights while the first half runs and waiting for a flag (bounded spin, same bits) | 89.5 (92.1 for both halves and the barrier) | without the wait 71.6, with the second half only loading 67: the second half's loads lengthen the first half's latency chain instead of filling idle bandwidth |
| down: expert ids and weights staged in LDS | 86.4 (85.0) | |
| decode attention: V rows four positions at a time | 74.1 (71.8) | |
| decode attention: queries through LDS (one load per thread for the 12 heads) instead of 24 dependent scalar loads | 72.2 (71.9) at 512 positions, 198.4 (127.7) at 4096 | 28 KB of LDS: ACO allocates 256 VGPRs, one wave per SIMD |
| gdn: state loaded unconditionally, then selected | 49.4 (47.5) | |
| out projections with 4 rows per wave (`qwen4_q8_mvk4` for one token) | 106.4 (101.5) | |
| layout probes (timing only): gate/up and down weights of a wave's step contiguous | 139.3 (140.8), 101 (86) | the weights' layout is not what limits the experts |
| execution-only barriers, with or without coherent bindings; lite barriers | above | |

## Correctness

| check | result |
|---|---|
| stage dumps, layers 4-7, c2 against main 9baedca (s10): the 14-token prompt one token at a time and in chunks of 4, p100 in chunks of 16, 2101 tokens one at a time (sparse attention past 2051) | every captured tensor the same bits (49 of 49 each) |
| stage dumps of c1 against main 15e5457 (s7, s8), including Q8 K/V caches (14 and 100 tokens) | the same bits |
| stress check of c2 (enlight, 7.3.0-rc7, both GPUs, layers 0-7 with the n-gram module and two QSA layers): the first 520, 1030, 2060 and 2600 tokens of a real text one token at a time and the 2600 in chunks of 4, every captured tensor of the last token, twice each, alternating with main 9baedca | 100 of 100 tensors identical in all 20 comparisons; each binary's repetitions identical |
| the same stress check of c1 against 15e5457 (indite, 7.2.9) | identical |
| every dump comparison of the round (`dumps-compared.txt`, `tools/dump-cmps.sh`) | 63 comparisons, all identical |
| ISA audit of the 82 kernels that run after light barriers | no scalar loads of set-2 data other than params (`audit/c2-scalar-safe.txt`) |
| startup self-test | passed on every run (no "light barriers" line in any stderr) |
| `cargo test` (kbuild VM) | 95 passed |
| six Macs, val14 | logits rel 2.79e-4 against the CPU reference, top-10 10/10, greedy 10/10; all 691 dumped tensors the same bits as main's (results/qwen4/main-suite-20261013, ms-15e5457; `chain/dk2c2/validation.txt`) |
| six Macs, p100 | logits rel 3.48e-2, top-10 10/10, greedy 10/10; all 596 tensors the same bits as main's |
| six Macs, MTP chat | the same 192 tokens per prompt in both arms, and each arm's speculative output identical to its plain greedy output |

## Six Macs

After the stream fix (7.3.0-rc7-p2p-s2 with the USB4STREAM patches 0006 and 0007 on all six), session dk2c2p
(18:23-18:24, `chain/dk2c2p/summary.txt`, `tools/dk2-pairs.sh`) ran each pair once, arms alternated, and checked every
run's links right after it, ready to stop at the first link-end off its stream; none was:

| run | main 9baedca | c2 | change | links (both arms) |
|---|---|---|---|---|
| decode after a 512-token prompt (128 tokens) | 21.46 tok/s | 22.57 | **+5.1%** | all 10 link-ends on their streams |
| prefill 2048 tokens | 889.3 tok/s | 886.0 | -0.4% | all 10 on their streams |
| decode after 2048 (32 tokens) | 20.73 tok/s | 21.36 | **+3.0%** | |
| prefill 8192 tokens | 1122.4 tok/s | 1129.2 | +0.6% | all 10 on their streams |
| decode after 8192 (32 tokens) | 20.23 tok/s | 21.46 | **+6.1%** | |

Prefill runs graphs of more than 8 tokens with full barriers, as on main: no change expected, none measured. Every node
held 850/1370 MHz (invite GPU 0 up to 79 C).

**Closed at end.** A node logs "stream down (the other side closed its stream); continuing over TCP" on its upstream
link when the driver (or the previous node) closes its stream as it exits after the run; earlier runs rarely showed it
because the nodes were usually stopped before they logged the close. `tools/transport.py` counts a link-end as on its
stream under this name only when all of these hold: the down is logged by the node on its upstream link (to the driver
or the previous node) and says "the other side closed its stream", it is that link-end's only down or failure and its
last line, after an up; the other end of the link logged an up and no down, failed epoch or "not opened"; the run's rc
is 0. Anything else stops the session. The case that made the rule: the first attempt of this session (18:20,
`chain/dk2c2p-attempt1`) stopped after its first run (main's arm, decode after 512 tokens, rc 0, 21.28 tok/s), where
inflight's log for its link to the driver read

```
link incite (10.99.21.1:37132): stream up (0-3.1/trashcan /dev/tbstream0), epoch 2144576b
link incite (10.99.21.1:37132): stream down (the other side closed its stream); continuing over TCP
```

and the driver's for the same link only `link inflight (inflight.tb0:7071): stream up (0-3.1/trashcan /dev/tbstream0),
epoch 2144576b`. In dk2c2p, 4 link-ends were closed at end (the 512 c2 run: inflight's link to incite; the 2048 main
run: inflight's to incite and alight's to inflight; the 2048 c2 run: inflight's to incite), every one printed with its
lines by the check.

Before the fix, session dk2c2 (`chain/dk2c2/summary.txt`), main 9baedca against c2, arms alternated. A pair counts only if both arms
ran every link on its stream (`tools/transport.py`, from the "link ... stream up" lines of the driver's and the nodes'
logs):

| run | main 9baedca | c2 | change | links |
|---|---|---|---|---|
| MTP chat (3 prompts, 192 tokens, one pass each) | 30.05 tok/s | 31.28 | **+4.1%** | all 10 link-ends on streams in both arms |
| water / csv / transformer | 28.72 / 32.28 / 29.15 | 30.23 / 33.32 / 30.29 | +5.3% / +3.2% / +3.9% | |
| plain decode in the same runs | 20.98 tok/s | 22.04 | **+5.1%** | |
| verification of 1 / 2 / 3 / 4 tokens (ms per round) | 48-49 / 64-65 / 78-79 / 106 | 45 / 61-62 / 74-75 / 94-98 | | |
| decode after 512 tokens, two pairs | 20.91, 20.98 | 22.36, 22.23 | +6.4% | not valid: one link on TCP in each main run |
| decode after 2048 / 8192 tokens | 20.73 / 20.25 | 21.78 / 20.91 | +5.1% / +3.2% | not valid: different links on TCP |
| prefill 2048 / 8192 tokens | 888.8 / 1024.5 | 889.0 / 1128.8 | 0.0% / +10.2% | not valid; prefill graphs keep full barriers, so the 8192 difference is the links (8 link-ends on TCP against 4) |

Session dk2c2r (`chain/dk2c2r/summary.txt`) repeated the decode and prefill pairs, each run up to four times until all
of its links were on streams. Both decode pairs after 512 tokens ran on streams at their first attempt: 21.28 -> 22.29
and 21.32 -> 22.38 tok/s (+4.7%, +5.0%). The main arm of the 2048 pair failed four times (2-6 of 10 link-ends on
streams), and the session was stopped there: repeating runs until the links come up hides the link bug; the pairs ran
once the stream fix was rolled out (dk2c2p, above). (Prefill of the 512-token prompt varies between 355
and 522 tok/s from run to run in both sessions; it runs full barriers in both arms.) Every node held 850/1370 MHz while
computing (invite GPU 0 up to 78 C).

## What is left

- **The rest of a decode layer.** A DeltaNet layer now takes 0.83 ms for 117 MB (142 GB/s against the practical 205,
  0.57 ms): the mixers (4 dispatches, 170 us for 14 MB), the experts (gate/up 125 us, down 75 us for 29 MB), the small
  kernels around the recurrence (conv, gated norm, top-k: 33 us) and the barriers' gaps (still 1.0-1.4 us each: the
  light barrier keeps the wait for idle and the L2 write-back). The mixers, down and gate/up are one round of waves
  that wait for memory and then compute; overlapping the two needs a prefetch that ACO does not turn into register
  copies at the loop's end (two attempts this round, `removed/`). Fewer dispatches (conv and the gated norm into the
  recurrence, top-k into the router) remain open.
- **Verification batches** (MTP, 2-4 tokens) re-read each token's 11 experts: 0.165 ms per extra token and layer.
- **Mesa upgrades**: the light barriers rest on RADV's GFX6 barrier handling; rerun the audit and the stress check
  (AGENTS.md).

## Files

- `indite/s1` ... `s11`: stage-bench JSON, stderr and clock logs per single-Mac session (the session scripts are
  `tools/dk2-s*.sh`); `indite/c1-stress`, `enlight/c2-stress`: the stress checks' logs.
- `chain/dk2c1`, `chain/dk2c2`, `chain/dk2c2r`, `chain/dk2c2p` (and its stopped first attempt `dk2c2p-attempt1`): the
  six-Mac sessions' driver outputs and clock logs, each node's log and clock log, and `summary.txt`.
- `audit/`: `c2-scalar-safe.txt` (the kept rule on the kept build), `c2-all-pipelines.txt`; `coherent-build.txt` and
  `main-build.txt` (the `--rule coherent` audit of the coherent-bindings build and of main, step 1).
- `dumps-compared.txt`: every stage-dump comparison.
- `stats/`: RADV statistics (registers, waves per SIMD, code size, instructions) of the kernels in `SCALAR_SAFE` on
  main e155a4f7 (`main-e155a4f7.txt`) and on this branch (`c2.txt`).
- `tools/`: `prof.py` (per-dispatch tables), `walls.py`, `twice.py`, `cmpdump.py`, `stress-cmp.sh`, `dump-cmps.sh`,
  `coherence_audit.py` and `audit-decode.sh` (the ISA audit), `vmstats.sh` (RADV statistics and programs offline in
  the kbuild VM, `RADV_FORCE_FAMILY=tahiti`), `isa.py`, `isablocks.py`, `loopmix.py`, `stats.py`, `smemcheck.py`; the
  session scripts `dk2-s1.sh` ... `dk2-s11.sh`, `dk2-stress.sh`, `dk2-chain.sh` and `dk2-pairs.sh` (under chain6.sh),
  `lease-run.sh`,
  `check-macs.sh`, `archive-removed.sh`, `copy-results.sh`; the six-Mac summaries `chain_sum.py`, `mtp_summary.py`,
  `mtp_same.py`, `valcmp.sh`, `nodeclocks.py`, `transport.py`.
- `removed/`: the removed kernels.

## Reproduce

```
# ISA audit (kbuild VM, Mesa 25.0.7 as on the Macs): every pipeline's program, then the scalar-safe kernels
tools/vmstats.sh TAG && tools/audit-decode.sh engine/src/vulkan.rs dk2/stats/TAG.txt
# one Mac (under a gpuq lease): barrier costs and cache probe; decode A/B and dumps; the stress check
trashcan qwen4 barrier-bench --gpu 0
tools/dk2-s10.sh; tools/dk2-stress.sh TAG CANDIDATE trashcan-9baedca; tools/stress-cmp.sh DIR
# six Macs (chain6.sh holds all six): validation, decode/MTP/prefill pairs; then the summaries
chain6.sh LABEL 15 OWNER -- tools/dk2-chain.sh
tools/chain_sum.py chain/dk2c2 dk2c2; tools/transport.py chain/dk2c2 dk2c2; tools/nodeclocks.py chain/dk2c2
```
