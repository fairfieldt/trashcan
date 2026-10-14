# Flash-Next on long prompts and at long context (2026-10-13)

**Result (single Mac, per stage).** The QSA layer's prefill chunk at 8K-30K goes from 69-80 ms to 27-31 ms, and a
stage's 128-token chunk (layers 0-3 with the n-gram module) from 137-151 ms to 94-101 ms, flat in the position (95.9 ms
at position 0); the decode step's QSA layer from 1.63-1.79 ms to 1.17-1.21 ms at 8K-30K and from 1.03 to 1.01 ms at
128 positions. At 204,800 positions (Q8 K/V) one QSA layer's chunk goes from 176 to 38 ms.

**Six Macs** (against main eb4dbeb): prefill of an 8K prompt 806 -> 1127 tok/s (+40%), 16K 846 -> 1211 (+43%), and
decode after them 17.4 -> 19.7 and 17.1 -> 19.6 tok/s (+13%, +15%); a 512-token prompt 487 -> 518 tok/s (+6.5%) and the
decode after it 20.1 -> 20.6 tok/s (+2.4%). The same greedy tokens as main on the 14-token, 100-token, 3061-token and
8K prompts. 30K and 100K tokens ran to the end, but invite's GPU 0 overheated and throttled in those runs, so their
speed is still to be measured (see Clocks).

## Tested source

- Code: a2cc2421 on main eb4dbeb. Built as r24 (md5 c27931e7), not run itself: it is r23 without the temporary
  `TRASHCAN_Q4_CHUNK` switch that the chunk sweep used.
- long-s5, the six-Mac session:
  - r23 (ebf8bab5), this round on eb4dbeb;
  - r23x (9176f59a), r23 without `qwen4_matf32_t`;
  - r23s (b179ae7b), r23 with main's order in `Local::step`;
  - base: main's `trashcan-eb4dbeb` (c0857f9d).
- long-sv: r21 (cc6e4628), this round on main 5e48d7e; r21s (47ce8236); base `trashcan-2293550` (f3b8926a).
- One Mac, on main 4307169:
  - r19 (e5e9b341);
  - r20 (ef99e4d9), whose decode kernels are the final ones;
  - main `trashcan-4307169` (13faba5b).
- r17 and r18 on main 7da6f93: the decode-kernel A/B.
- r15 and r15f on main 6f6635a: the old-against-new kernel switch and the fused-leaf score variant.
- The work-in-progress history, r4 to r20, is in `logs/wip-commits-on-5e48d7e.txt`.
- Data:
  - `runs/` holds the chain runs (driver JSON and stderr, node logs, per-Mac clock logs) of long-s1 (main 4c1346d, the
    baseline timeline) and long-s5;
  - `sb/` holds the single-Mac stage-bench JSON and clock logs;
  - `logs/` holds the session logs and the comparison outputs;
  - `tools/` holds the scripts.

## Where the time went on long prompts (main 4c1346d)

Six Macs, `qwen4 bench --tokens-file long-32k.ids --pp N --tg 32 --head-split 12 --context 32768 --batch 128 --ngram
disk`; the prompt is the repository's docs and results READMEs as one user turn (`tools/mkcorpus.sh`: 777 KB, 261,027
tokens; `long-32k.ids` holds its first 32,768 ids, a prompt of N tokens is its first N):

| prompt | prefill | decode after it |
|---|---|---|
| 8192 | 697.7 tok/s (11.74 s) | 16.00 tok/s |
| 16384 | 742.1 tok/s (22.08 s) | 16.09 tok/s |
| 30000 | 752.0 tok/s (39.89 s) | 15.48 tok/s |

**Chain timeline.** `TRASHCAN_Q4_TIMELINE=1` (with `TRASHCAN_TP_PROFILE=1` for calibrated GPU timestamps) makes every
machine print a line per prompt-chunk event: the driver's embedding and n-gram rows, each stage's submit, wait and
GPU start and end, the node's wait for its input, the send of the output (`tools/timeline.py`). At 8192 tokens
(`long-s1-tl-pp8192`, 64 chunks of 128):

| Mac | GPU 0 busy | GPU 1 busy | per chunk |
|---|---|---|---|
| incite (driver; layers 0-7, n-gram module on GPU 0) | 98% (154.1 ms per chunk) | 93% | 157.3 ms |
| indite | 95% | 95% | 157.3 ms |
| inflight | 94% | 95% | 157.3 ms |
| alight | 91% | 90% | 157.4 ms |
| invite | 94% | 93% | 157.4 ms |
| enlight | 91% | 89% | 157.1 ms |

The whole chain moves at the pace of incite's first GPU (layers 0-3 plus the n-gram module), which idles only 3.2 ms
per chunk: reading its 5 MiB output (1.9 ms after the GPU ends) and writing the next input. The other GPUs wait for
their input. The driver's embedding and n-gram rows take 2.4 ms per chunk (`--ngram disk`, page cache) and a send 2-6
ms, both while the GPUs work; every node's reader thread had the next chunk ready or waited for the previous Mac. At
30,000 tokens the same: 162.5 ms per chunk, 159.5 of them busy on incite's GPU 0. So the host and the transfers cost
about 2%.

**Stage times at the chunk's position.** stage-bench measures chunks at position 0 (116 ms for layers 0-3, the basis
of the ~930 tok/s expected at 8K). On inflight (layers 0-3 on GPU 0, 4-7 on GPU 1, `TRASHCAN_Q4_PROFILE=1`,
`stage-bench --start S --prefill 1 --decode 16`, the late chunk is the last 128 of S tokens):

| chunk at | GPU 0, ms | QSA layer | its attention | its block scoring | selection |
|---|---|---|---|---|---|
| 0 | 116.0 | 29.0 | 3.8 | | |
| 2048 | 154.5 | 68.8 | 42.4 (`qwen4_attention_sparse_full` 39.6) | 1.7 | 0.33 |
| 8192 | 158.1 | 71.9 | 42.5 (39.8) | 5.1 | 0.32 |
| 16384 | 162.6 | 77.0 | 42.8 (40.1) | 8.6 | 0.38 |
| 29952 | 168.2 | 81.9 | 42.4 (39.7) | 14.3 | 0.71 |

From position 2048 on every token attends to ~2051 positions, and the sparse kernel (one wave per (head, token)
walking them one at a time, a cross-lane sum per position, the 12 query heads of a KV head each reading its rows)
takes 40 ms of a ~155 ms chunk; block scoring (a subgroupAdd per (block, token, head)) grows with the position. With
these times the pipeline model (sum of the chunks' times plus 11 times the last) gives 11.5 s for 8192 tokens against
11.6 measured: the 25% between ~930 and 707 tok/s was the QSA layer at the chunks' positions, not the chain.

Decode had the same kernels: the QSA layer of a decode step took 1.63-1.79 ms at 2K-30K against ~0.9-1.2 ms for a
DeltaNet layer, 0.36 ms of it the sparse attention and up to 0.29 ms the block selection.

## The changes

- **Prefill attention grouped by KV head** (`qwen4_attention_grouped`, `_q8`, `_k16v8`): a workgroup per (KV head,
  token) serves the 12 query heads that share it, so each visible K/V row is read once instead of 12 times. The four
  waves take turns of 64 visible positions; in a turn a lane takes a position, scores its K row against the 12
  queries (wave-uniform, scalar loads), one cross-lane maximum per head per turn keeps the online softmax (the sums
  stay per lane until the end), then the lanes take 4 dims each and add the 64 V rows times the 12 heads'
  probabilities. The waves' partial results merge in shared memory.
- **Decode and verification attention grouped by KV head** (`qwen4_attention_grouped_dec{8,16,32,64}`, Q8 variants;
  up to 8 tokens): a workgroup per (KV head, 64 visible positions, token). Each wave scores a quarter of the K row
  (64 dims) for all 12 heads, the quarters meet in shared memory, each wave turns 3 heads' scores into probabilities
  and adds 16 positions' V rows for all 12 heads, and the waves' sums merge in shared memory.
  `qwen4_attention_merge{8..64}` merges only the partitions that hold the token's positions; each thread loads its
  dim of all of them at once and every wave weighs them itself (one wait for memory, no shared memory).
- **Block scores, a lane per block** (`qwen4_idx_score`; `qwen4_idx_score_lds` for up to 8 tokens, the queries in
  shared memory): each lane computes every head's dot product of its block's pooled key with up to 8 tokens' indexer
  queries, in the order of the wave reduction it replaces (per lane two rounded products and their sum, then the
  balanced tree of a GFX6 `subgroupAdd`, `precise` so nothing is fused or reassociated). The pooled keys are stored
  `[dim / 4][block]` so a wave's lanes read consecutive blocks (`qwen4_idx_pool`).
- **Block selection** (`qwen4_idx_select`, 1024 threads for up to 8 tokens; `qwen4_idx_select256` for prefill
  chunks): a radix select with digits of 11, 11 and 10 bits over a 2048-bin shared histogram, up to 8 keys per thread
  kept in registers between the passes, the group and the bin holding the remaining rank found with inclusive scans
  by one wave, and the selected blocks written in block order with ballots; ties to the earlier block as before.
- **f32 projections of prefill chunks** (`qwen4_matf32_t`, here the indexer key and query; it also takes the DeltaNet
  alpha/beta rows): four rows and 16 tokens per wave instead of one row and 8 tokens, the same per-lane FMA order and
  wave sum, so the same bits. The indexer projections also join the attention projections' group (no barrier between
  them). Six Macs, two alternated pairs at 2048 tokens: prefill 873.8 and 877.1 tok/s against 863.7 and 863.4 without
  it (+1.4%); one Mac, the chunk at position 0: 95.9 against 99.1 ms on GPU 0.
- **The chain**: a stage's next prompt chunk is queued behind the running one (two submission slots: host inputs,
  fence and timestamps per slot, graphs recorded per slot), so a GPU does not wait for the host between chunks; the
  finished chunk's output is read after the next one was submitted (`Local::step`, `GpuStage::collect` = `wait` +
  `output`). Six Macs, two alternated pairs at 8192 tokens: prefill 1127.1 and 1126.4 tok/s against 1083.5 and 1083.9
  with main's order (wait for the running chunk, then submit the next), +4.0%; decode within noise.
- **Diagnostics**: `TRASHCAN_Q4_TIMELINE=1` prints the chunk events above (pipeline.rs `tl`); stage-bench reports the
  late chunk's dispatches (`late_dispatches`).

**Decode kernels pay for their code.** A decode step runs each kernel's code about once per wave, and every dispatch
fetches its instructions anew. The decode attention fully unrolled (12.5 KB of code) took 169 us at 2049 positions;
with the queries through lane broadcasts and the V rows loaded together (fewer waits for memory, 19.6 KB) 218 us; the
same arithmetic with its loops kept (4.4 KB) takes 131 us, and 67 instead of 137 us at 129 positions. Block scoring
for up to 8 tokens went from 55 to 33 us per decode step at 8K (48 KB of code, about a sixth of it run per token, to
3.1 KB). That is roughly 5-7 us per KB of code a wave runs once, the same order as the work itself in these kernels.

Tried and removed: the stage output written by a shader instead of the transfer copy (2.22 against 2.40 ms per chunk,
within noise); six query heads per workgroup in prefill attention (three waves per SIMD instead of two, but every K/V
row read and converted twice: 9.1-10.4 against 4.9-5.5 ms); the queries in shared memory for prefill block scores
(2.87 against 1.76 ms); a block selection with serial per-lane walks (the scans replaced it); the decode attention's
queries through lane broadcasts with the V rows loaded together (r18, slower: above); the score leaves as one fused
multiply-add (the bits still differ from the replaced kernel's).

## One Mac: stage times by position

`tools/long-sb.sh BINARY TAG S...` under a single-Mac lease (`stage-bench --ranges 0:4,4:8 --gpus 0,1 --context
32768 --batch 128 --prefill 1 --decode 16 --start S`, `TRASHCAN_Q4_PROFILE=1`): GPU 0 holds layers 0-3 and the n-gram
module, the late chunk is the 128-token chunk ending at S, the decode steps follow it. Main 4307169 against this round
(r20: prefill as r19, the decode kernels of r20), each pair on one Mac at 850/1370 MHz:

| positions | Mac | chunk, GPU 0 | QSA layer | its projections, scoring, selection | its attention | decode step, GPU 0 | QSA layer | its projections, scoring, selection | its attention |
|---|---|---|---|---|---|---|---|---|---|
| 128 | invite | 98.5 -> 96.0 | 26.9 -> 24.3 | 6.94 -> 5.13 | 3.52 -> 3.02 | 4.10 -> 4.07 | 1.03 -> 1.01 | 0.313 -> 0.292 | 0.200 -> 0.205 |
| 1024 | invite | 113.6 -> 95.1 | 43.6 -> 25.1 | 6.93 -> 5.17 | 21.58 -> 4.63 | 4.27 -> 4.08 | 1.20 -> 1.03 | 0.312 -> 0.291 | 0.371 -> 0.219 |
| 2176 | invite | 137.2 -> 98.4 | 66.6 -> 28.0 | 9.29 -> 5.45 | 41.73 -> 6.92 | 4.71 -> 4.21 | 1.64 -> 1.16 | 0.589 -> 0.362 | 0.535 -> 0.279 |
| 8320 | incite | 137.1 -> 95.0 | 69.5 -> 27.4 | 12.22 -> 5.77 | 41.46 -> 6.95 | 4.59 -> 4.09 | 1.63 -> 1.13 | 0.594 -> 0.360 | 0.527 -> 0.272 |
| 16512 | invite | 143.3 -> 99.6 | 72.8 -> 29.2 | 15.64 -> 6.17 | 41.95 -> 7.79 | 4.81 -> 4.24 | 1.74 -> 1.19 | 0.685 -> 0.387 | 0.536 -> 0.286 |
| 30080 | incite | 145.7 -> 97.6 | 78.3 -> 30.3 | 21.64 -> 6.78 | 41.67 -> 7.49 | 4.74 -> 4.13 | 1.77 -> 1.18 | 0.727 -> 0.398 | 0.534 -> 0.276 |

(ms; "projections, scoring, selection" is the qkv and index phases together, since the indexer projections moved from
the second to the first.) The chunk now costs about the same at every position: 96 ms at 0, 98-101 ms from 2K to 30K,
against 98.5 and 137-151 ms on main.

**204,800 positions.** `tools/long-200k.sh` on enlight: layer 3 alone on GPU 0 with a Q8 K/V cache for 204,800
positions (G-iq4; the 200K configuration of docs/qwen4.md is S-q2dn), the 128-token chunk ending at 204,672 and 16
decode steps after it:

| | main 4307169 | r20 |
|---|---|---|
| prefill chunk, QSA layer | 175.9 ms (block scoring and selection 101.4, attention 53.9) | 38.4 ms (10.2, 6.9) |
| decode step, QSA layer | 3.26 ms (1.85, 0.65) | 1.59 ms (0.54, 0.26) |

There the block scoring of a prefill chunk (9.3 ms for 51,168 blocks, 128 tokens and 4 heads of 128 dims) is the
largest QSA cost left, and in a decode step the block selection (0.32 ms: one workgroup over 51,168 keys) and scoring
(0.18 ms).

## Correctness

| check | result |
|---|---|
| kernels before and after this round in one binary (r15, `TRASHCAN_Q4_QSA=old` against the default), every token's rows of the last chunk (`tools/long-unit.sh`): f16 K/V at 2052, 2101, 8196, 30000 and 30081 tokens, Q8 K/V at 8196 and 30000 | the same selected blocks for all 162 tokens in both QSA layers; block scores: 22-33% differ in the last bits (with a fused multiply-add in the leaves about the same share: 2095 of 8193 and 117,902 of 359,700); attention output rel 2.2e-7 to 9.6e-7 in layer 3 and 5.5e-6 to 9.4e-5 in layer 7 (its input carries layer 3's differences); stage output rel 3.0e-6 to 4.7e-5 |
| r17 against r18, r19 against r20 (the decode kernels' code changed, not their arithmetic): last-token dumps at 8193, 8196, 30000 and 30081 tokens, Q8 K/V at 8196 | every tensor identical |
| r19 against main 4307169, last-token dumps at 8196 and 30000 tokens (`tools/long-equiv.sh`) | layers 0-2 identical; attention output rel 2.6e-7 / 8.5e-7 (layer 3) and 8.3e-6 / 7.2e-6 (layer 7); stage output rel 5.1e-6 / 3.4e-6 |
| bounds: `tools/boundscheck.py`, a CPU model of every new kernel's address arithmetic for each chunk and decode step, with the block selection simulated as the shader does it on adversarial scores (random, all equal, few values, NaN, infinities, signed zeros, denormals) | every index inside its buffer, and the selection the exact top-512 with ties to the earlier block, for 2049, 2052, 3061 (chunks of 64 and 128), 8192, 30000 and (context 114,688) 100,000 tokens |
| six Macs, r21: the 14-token reference prompt against the CPU reference | the reference's 10 greedy tokens; logits rel 2.81e-4 (main 2.78e-4), top-10 10/10 |
| six Macs, r21: p100 against the CPU reference `ref-p100` | the reference's 10 greedy tokens; logits rel 3.48e-2 (main the same), top-10 10/10 |
| six Macs, r23 against main eb4dbeb: the 3061-token and 8K prompts (`qwen4 validate --greedy 10`) | the same 10 greedy tokens; logits rel 3.3e-2 and 1.5e-2 (cos 0.99947, 0.99989), top-10 10/10. Long prompts amplify any difference: on alight two builds of this round that differ only in main's capacity-dependent router path already differ by 8.5e-4 after layer 39 (above) |
| six Macs, r21 (main 5e48d7e) and r21s (main's order in `Local::step`) on the 3061-token prompt, the run that hung invite, after its power cycle (long-sv) | both the same 10 greedy tokens as main 2293550; r21 and r21s logits bit-identical |

**invite, 15:35.** The first six-Mac attempt (long-s4, r21 on main 5e48d7e) ran the 14-token and p100 validations
(above) and was 20 s into the 3061-token prompt when invite's GPU 0 stopped answering on PCIe (completion timeout) and
the machine hung; its GPUs had been reset at 15:27 by another session's stall test (results/qwen4/invite-hang-20261013).
What this round does differently at that level: each GPU stage queues the next prompt chunk behind the running one
(two submissions in flight; each starts with a full barrier before it touches the shared device buffers) and has two
host-visible input slots, so the host writes the next chunk's inputs and reads the finished chunk's output while the
GPU works. No peer-to-peer copies, no new memory types, no submission near amdgpu's 2 s job timeout (a chunk takes
60-100 ms). Every new kernel's storage buffers are bound with their sizes, so an index out of range is clamped by the
hardware, and no loop bound depends on data (`tools/boundscheck.py` above). On alight after its reboot
(`tools/long-chk3061.sh`, `tools/chk-compare.sh`): the 3061-token prompt through invite's layers (32:36, 36:40) in
64-token chunks, queued two deep through `Local::push` as on a node (`tools/edit-r21q.py`, a diagnostic build) gives
the same bits as one chunk at a time (114 tensors), and against main 5e48d7e layers 32-34 are identical, layer 35's
attention output within 1.1e-6; the difference then grows through DeltaNet layers 36-38 to 8.4e-4 at the stage output,
as it does between two runs of this round that differ only in main's capacity-dependent router path. invite's GPU 1
had a compute-ring timeout at 16:05 running main's own code.

## Six Macs

`tools/s5.sh` under `chain6.sh` (long-s5; base main eb4dbeb, `trashcan-eb4dbeb`; this round r23, the same code as the
final commit but for the temporary `TRASHCAN_Q4_CHUNK` switch the sweep used): `qwen4 bench --head-split 12 --context
32768 --batch 128 --ngram disk --tokens-file long-32k.ids --pp N --tg 32` (512: `--tg 128` on `long.ids`, the
standard decode bench), one alternated pair per size, two where a difference was under 3%.

| prompt | prefill, main | this round | decode after it, main | this round |
|---|---|---|---|---|
| 512 (two pairs) | 486.6, 486.7 tok/s | 520.4, 515.8 (+6.5%) | 20.19, 20.05 tok/s | 20.68, 20.53 (+2.4%) |
| 8192 | 806.1 | 1126.7 (+39.8%) | 17.37 | 19.67 (+13.2%) |
| 16384 | 846.4 | 1210.5 (+43.0%) | 17.10 | 19.62 (+14.7%) |
| 30000 | 849.2 | (1049.7, invalid: invite GPU 0 at 450 MHz) | 17.01 | (18.12, invalid) |

Chunk size: 3072 tokens 958.2 tok/s in chunks of 64 (the band's choice) against 915.1 in chunks of 128, 4096 tokens
994.9 against 987.1, so the bands stay (64 up to ~4950 tokens).

**Clocks.** Every GPU held 850/1370 MHz in these runs except invite's GPU 0: it reached 85-89 C in the 8K and 16K runs
and 95 C in the 30K run of this round, and from the middle of that run (57 of 129 samples) it ran at 450/1370 MHz, and
stayed there through the timeline and 100K runs (at 81-82 C), which therefore measure invite, not the round. All other
GPUs stayed at 61-82 C. The single-Mac profiles above held 850/1370 throughout (`sb/*.txt` clock logs).

**Timeline at 8K** (`long-s5-tl-pp8192-new`; invite's GPU 0 at 450 MHz, so only the stages before it are
representative): incite's GPU 0, the stage that paced the chain before, is busy 100% of its 6.16 s with no idle time
between chunks (98% and 3.2 ms per chunk before), 96.8 ms per 128-token chunk (154.1 before); each chunk is submitted
about 85 ms before its GPU starts it, behind the running one. indite's GPUs are busy 96%.

**100K tokens** (`--context 114688 --kv q8`, `long-100k.ids`, this round only): runs to the end (rc 0); 784.6 tok/s
prefill and 17.49 tok/s decode with invite's GPU 0 at 450 MHz for the whole run, so not a speed result. Every
submission stayed far below amdgpu's 2 s job timeout (a layer of a chunk is one submission: under 40 ms at 204,800
positions on one Mac).

Invalid for speed, with the clock logs fetched next to them (`runs/NAME/clocks-invite.txt`): `long-s5-pp30000-new-1`
(invite GPU 0 at 850 MHz for 72 of 129 samples, then 450; max 95 C), `long-s5-tl-pp8192-new` (450 MHz for all 57 busy
samples, max 82 C) and `long-s5-pp100k-new` (450 MHz for all 496 samples, max 81 C). All six Macs share one fan curve
(2350 rpm); invite's GPU 0 runs 10-15 C hotter than the others, a cooling fault on that Mac to be fixed before the 30K
and 100K runs are repeated.

## What is left

- **Prefill attention** (`qwen4_attention_grouped`): 5-8 ms of a 128-token chunk from 2K on, about a third of the FP32
  peak. For each 16-dim slice of its K rows a wave loads the slice and the 12 queries' 192 values (scalar loads) and
  waits for them, at two waves per SIMD; its registers are full (128), so prefetching more needs a different split.
- **Block scoring of prefill chunks** at very long contexts: 9.3 ms of a chunk at 200K (about 20% of the peak).
- **Decode**: the QSA layer still costs ~0.2 ms more than a DeltaNet layer at 2K-30K: the decode attention (126-134
  us at 2K+ positions), block selection (48-85 us, one workgroup, 3 passes) and scoring (32-37 us). Decode kernels pay
  for their code size (above); other decode kernels may too.
- **The stage output copy** to host memory: 2.2-2.4 ms of a 128-token chunk (5 MiB at ~2.3 GB/s), the same with a
  shader copy; on a separate queue it could overlap the next chunk.
- **incite's GPU 0** (layers 0-3 and the n-gram module, ~4 ms of a 128-token chunk) is the slowest stage, so the chain
  runs at its pace.
