# Qwen3.8-27B TP8: analysis and optimization plan

Proposal of 2026-10-08, written from the source at 261d453 and the measurements in
[results/qwen/tp8-20261007](../results/qwen/tp8-20261007/README.md), without access to the Macs. Every gain
below is an estimate from those measurements; none is demonstrated until the runbook in
[results/qwen/tp8-opt-20261008](../results/qwen/tp8-opt-20261008/README.md) has been run on the four Macs.
Following AGENTS.md, each candidate ships behind a switch with its current behaviour as the default, and is
kept (switch removed) or deleted after its end-to-end A/B.

**Status (measured on the Macs 2026-10-09, [results](../results/qwen/tp8-opt-20261008/README.md#results-on-the-macs-2026-10-09)).**
W1 is kept for all-reduces below prefill size and is the default: decode 35.0 to 36.2 tok/s and MTP 46.0 to 47.0. On
prefill-sized all-reduces it made prefill 5% slower, and its f16 reduce-scatter/all-gather is gone. Every kernel (A1,
K1, F1, G1, M1) produced the bits it claimed on the D700s but ran slower than the kernel it replaced, except A1 at
8000 keys; none was faster end to end, and A1 could not be measured at long context. H1 cost 0.55 ms per token.
All of them, and `qwen-kernel-check`, are removed. X3, the sixth cable, is done: full-mesh all-reduces give 37.2 tok/s
decode, 50.1 with MTP and 211 prefill ([results/qwen/tp8-mesh-20261009](../results/qwen/tp8-mesh-20261009/README.md)).
Round 2 ([results/qwen/tp8-round2-20261009](../results/qwen/tp8-round2-20261009/README.md)) found the opposite of M1
to be right for the one-column projections (two rows per wave, 10-35% faster per dispatch, decode +6.4%), one state
row per wave in the recurrence (+2.3%), a two-key attention partial and an unrolled merge: 40.7 tok/s decode and 52.0
with MTP, 42.2 decode without rounding the decode totals (`TRASHCAN_TP_ROUND=0`). The exchange's parts are now
measured (`sent`, `first`, `wire`, `encode`, `decode` in `tg_tp_ms`); a thread per cable, an AVX build and fused host
passes did not help. Round 3 ([results/qwen/tp8-c3-20261009](../results/qwen/tp8-c3-20261009/README.md)) went after
the three-column verification kernel (64 VGPRs, 4 waves per SIMD) and found no faster one under the bitwise
contract: it is bound by latency hiding (98% VALU issue at 7 waves per SIMD, about 65% at 4), and activations in
shared memory (three designs), a weight prefetch, compiled-in row lengths, hoisted indices and a reordered
epilogue all run slower; five rows per wave only helps the 2176-row FFN shard (-10% alone, -2% with gate and up
concurrent) and was 0.3% of GPU time on the cluster, so it was removed. The candidates beyond that kernel were
microbenchmarked too: a weight layout interleaving a wave's 16 blocks of four rows gains 4-6% on the three-column
kernel for 5120-long rows and nothing on decode (about 1% of a verification round for tiled variants of every
Q4_0 kernel), precomputed zero-point biases gain nothing, per-shape row geometry about 0.5%. None integrated.
The estimates below are kept as written.

## Baseline

Qwen3.8-27B Q4_0 (Q6_K head), TP8 over incite, invite, enlight and indite, one rank per D700, 850/1370 MHz:

| workload | rate | per unit |
|---|---|---|
| decode (tg128 after pp512) | 35.0 tok/s | 28.4 ms per token |
| decode with MTP (k = 2, 64K draft head) | 46.0 tok/s | 53.9 ms per round of 2.47 tokens |
| prefill pp512 / pp2048 | 204 / 197 tok/s | 2.53 s per 512-token chunk |

### Decode step: 28.4 ms

From `tg_tp_ms` of `incite/rs-on-*.json` and the segment profile (`dprof-incite.txt`, `prof-*.txt`):

| part | ms per token | share |
|---|---:|---:|
| GPU segments (129 per token) | 19.2 | 68% |
| cross-machine exchanges (128 all-reduces, 58.6 µs each) | 7.5 | 26% |
| host: local partial sum (3 µs per all-reduce) | 0.38 | 1.3% |
| host: remote add, residual, RMS norm, both GPUs' input rows (7 µs) | 0.90 | 3.2% |
| release to GPU start (3.2 µs per segment) and first-segment launch | 0.52 | 1.8% |

GPU time of one decode token on rank 0 (dispatch profile `incite/profile-rsag-dprof.jsonl.gz`, 29 steps,
medians; concurrent dispatches overlap):

| segment (count) | GPU µs | critical path |
|---|---:|---|
| DeltaNet mixer (48) | 121 | z/qkv/alpha-beta projections 50, `qwen_conv` 5.0, `qwen_gdn` 23.0, `qwen_gated_norm` 7.7, out-projection 26.1, gaps |
| FFN (64) | 141 | gate and up 81-100 (concurrent), `qwen_swiglu` 3.3, down 49.9 |
| attention (16) | 192 | k/v/q projections 44, `qwen_qk` 13.8 (three concurrent dispatches), `attention_partial64` **55.7**, `attention_merge64` **38.5**, `qwen_gate` 4.4, out-projection 26.2 |
| head (1) | 1194 | `headvec` 975, three `qwen_topk` passes **72 + 74 + 68** |

Weight bytes a rank streams per decode token (Q4_0 repacked at 20 bytes per 32 weights, the head expanded to
1.25 bytes per weight): FFN 1337 MB, DeltaNet 444 MB, attention 144 MB, head 199 MB, **2124 MB in all**. At the
263 GB/s physical peak that is 8.1 ms; at the 199 GB/s `matvecq4` reaches on a whole 17408 x 5120 matrix
(docs/findings.md) 10.7 ms. The 19.2 ms measured splits into about 14.2 ms of Q4_0 projections at 130-140 GB/s
(small TP8 shards: 95 GB/s for the 5120 x 768 out-projections) and about 5 ms of small kernels, barriers and
latency chains.

### MTP round: 53.9 ms for 2.47 tokens

Verification of 3 columns takes 50.6 ms, drafting 2.9 ms (two draft steps) and rollback 0.4 ms
(`incite/split-mtp.json`). A verification all-reduce moves 60 KB instead of 20 KB and costs about 110 µs, so
exchanges take about 14 ms of every round; the GPUs about 35 ms including the two draft steps (three-column
kernels, DeltaNet over three columns with snapshots, the three-column head).

### Prefill: 2.53 s per 512 tokens

Each GPU is busy 2.42 s of the 2.53 s (`pp_profile` of `incite/rsag-dprof.json`, two micro-batches on two queues),
and the host spends 2.07 s in exchanges at about 0.5 GB/s each way per cable. The GEMMs are 85% of the GPU time,
`qwen_gdn` 187 ms, `qwen_add_norm_j20` (reading the totals from host memory) 160 ms. GPU work and cable time are
balanced, so halving either one alone gains only a few percent; prefill needs both.

## Architecture as it stands

- **Layout** (`gpu::Split`): each of the 8 ranks holds 1/8 of every layer: 2 DeltaNet key heads with their 6 value
  heads, 3 query heads with a shared KV head (two ranks hold each KV head), 2176 FFN columns and 31040 vocabulary
  rows. The hidden state is replicated.
- **Segments**: a forward is 2 x 64 + 1 = 129 segments. Each mixer and FFN ends with a partial output projection
  written to host memory; the host adds its two GPUs' partials, all-reduces the sum with the other three machines
  and writes the next segment's normed input rows (`hostnorm`), which the GPUs read from host memory.
- **Gating**: decode, verification and draft steps are one submission per rank whose segments wait on a VkEvent
  the host sets (the command processor polls host memory, 3 µs) and set a done event the host polls.
- **All-reduce** (`tp::Mesh`): recursive doubling over the ring of cables 0-1, 0-2, 1-3, 2-3. With two rounds the
  vector is cut in two slices, slice j going to the partner of round (i + j) mod 2, so both cables of a machine
  carry half the vector in each round. Prefill-sized vectors use reduce-scatter and all-gather. Every element is
  added in the same pairs everywhere, so all machines keep bitwise-identical residuals.
- **Lanes** (`link.rs`): non-blocking `read`/`write` on `/dev/tbstreamN`, polled by the calling thread. A 64 B
  exchange takes 13.5 µs, a 20 KiB one over three streams 21.8 µs (results/qwen/tp4-viability-20261006); the
  receive side of USB4STREAM on 7.2.9 is interrupt driven (7.3, in use since 2026-10-14, adds busy polling).

The data path of one decode all-reduce:

```
GPU r: out-projection -> partial (20 KB f32, host memory) -> done event
host:  notice (0.6-3 µs) -> sum of the two local partials (3 µs: the reads miss the CPU caches)
       -> round 0: half to partner A, half to partner B (10 KB per cable each way) -> add
       -> round 1: the halves swap partners -> add -> residual += total -> RMS norm -> normed rows
          into each local GPU's host input buffer (7 µs for all of this) -> release event
GPU r: next segment starts 3.2 µs later and reads the normed rows over PCIe (cached in L2)
```

Platform facts that shape the kernels (docs/findings.md): RADV writes back and invalidates the whole L2 at every
compute barrier on GFX6; the command processor starts a dispatch's workgroups only after the previous dispatch's
have all started; subgroup reductions on GCN 1.0 go through LDS swizzles (no DPP), several hundred cycles each;
invite's and indite's GPUs stream 5-7% slower, so incite and enlight wait for them at every exchange.

## What this analysis found

1. **Decode attention is latency bound per key.** At a 520-token context the 64-partition split gives each of
   192 workgroups about 9 keys, yet `attention_partial64` takes 55.7 µs at every position measured: about 5-6 µs
   per key. The kernel loops over the head dimension with a runtime bound (`p.dim`), so its per-lane arrays are
   not register-resident, and each key is one subgroup reduction plus dependent K and V loads with nothing in
   flight. The three query heads of a rank each re-read the same K/V row. The merge takes 38.5 µs for 64
   partitions: 64 dependent loads of about 0.6 µs. The cost grows linearly with keys per partition, so at an 8K
   context (128 partitions of 64 keys) the partial alone should take about 0.3 ms per layer, 4-5 ms per token.
2. **Head selection is barrier bound.** Each `qwen_topk` pass is a 1024-element bitonic sort of 55 steps, each
   followed by a workgroup barrier, about 1.3 µs per step. 45 of the 55 steps (every step with stride at most 64)
   only touch elements owned by the executing wave.
3. **Exchanges are two thirds latency, one third bytes at decode sizes**, and mostly bytes at verification and
   prefill sizes: 10 KB more per cable costs about 12 µs.
4. **The host makes five passes** over the vector after the exchange (remote add, residual add, squares, normed
   write for one GPU, copy for the other).
5. **Small kernels cost 3-14 µs regardless of their work**: dispatch, a barrier with a full L2 flush, and cold
   loads. Each DeltaNet layer has three of them in a row (`qwen_conv`, `qwen_gdn`, `qwen_gated_norm`), each FFN one
   (`qwen_swiglu`), each attention layer four.
6. **Prefill is balanced** between GPUs and cables (above).

## Opportunities, ranked

Estimated savings at the baseline workloads; "bitwise" means the outputs are unchanged bit for bit.

| id | change | decode ms/token | MTP ms/round | prefill | numerics | risk | in this round |
|---|---|---:|---:|---|---|---|---|
| W1 | f16 all-reduce payload with a power-of-two exponent per 256 values (range safe), negotiated at session setup; multi-column all-reduces sliced per column | -1.3 | -4.2 | exchange bytes halved: -3% alone | two f16 roundings per element per all-reduce; identical on every machine | medium | yes |
| A1 | decode attention: head dim compiled in, every query head of a KV head in one wave (K/V read once), keys in batches of 8 with all loads issued first, partition count as a push constant; merge with loads issued together and the output gate fused | -1.2 (much more at long context) | -1.4 | none | different summation order than the old split-K kernels | medium | yes |
| H1 | one fused host pass for the remote add, residual, RMS norm and both GPUs' input rows | -0.5 | -1.0 | none | bitwise | low | yes |
| F1 | FFN gate and up projections and SwiGLU in one kernel (one to four columns) | -0.28 | -0.3 | none | bitwise | low-medium | yes |
| G1 | DeltaNet convolution folded into `qwen_gdn` for decode and verification (up to 8 columns) | -0.25 | -0.3 | none | bitwise | medium | yes |
| K1 | `qwen_topk` with wave-local barriers where a step stays in one wave | -0.14 | -0.35 | none | bitwise | low | yes |
| T1 | `qwen-kernel-check`: every new kernel against the kernels it replaces on random inputs (bitwise or tolerance) and timed, on the D700s; the same checks run on lavapipe against CPU references | | | | | | yes |
| X1 | USB4STREAM receive busy polling (backport from linux-next) | -1 (est.) | -2 | small | none | kernel work | no: kernel |
| X2 | RADV: skip the L2 writeback/invalidate for compute-to-compute buffer barriers on GFX6 (patched Mesa, env-gated) | -1 to -3? | | some | none | driver work, needs ISA-level validation | no: driver |
| X3 | a sixth cable (incite bus 0 - indite bus 0): one-round decode all-reduce | -2.5 | -6 | -25% exchange | none | hardware | no |
| M1 | short-K decode matvec (the 64 out-projections at 95 GB/s, K = 768): 8 or 16 rows per wave with `matvecq4`'s per-row arithmetic | 0 to -0.8 | 0 to -0.8 | none | bitwise | low | yes (`TRASHCAN_QWEN_SHORTK`); other lane mappings would need the D700 |
| B1 | uneven FFN split to balance the 5-7% slower GPUs | up to -0.6 | | small | none | medium | no: needs per-machine calibration |
| G2 | gated norm folded into the DeltaNet out-projection | -0.3 | | | bitwise possible | medium | no |
| P1 | chunked (WY) DeltaNet for prefill; f16 totals into `qwen_add_norm` | | | -5 to -10% GPU | | high | no |
| S1 | pre-submit the next decode submission (double-buffered gated graphs) | -0.1 | | | | medium | no |

If the estimates hold, W1, A1, H1, F1, G1 and K1 together take a decode token from 28.4 to about 24.7 ms
(about 40 tok/s, +15%) and an MTP round from 53.9 to about 46 ms (about 53 tok/s, +15%), with prefill a few
percent faster. They are independent and can be accepted or rejected one by one.

## Design of the implemented items

### W1: compact all-reduce payload (`TRASHCAN_TP_WIRE=f16`)

Each slice sent in an all-reduce round becomes f16 values plus one exponent byte per 256 values: a block whose
largest magnitude is at least 2^15 is scaled by 2^-e (exact) so it stays far from f16's 65504 limit; Qwen's
massive activations therefore cannot overflow. Conversion is round to nearest even (F16C when present, the
same bits in software otherwise). To keep every machine bitwise identical, each partner of a round continues
from the values it sent plus those it received (decoded in one pass), so both add the same two rounded values;
reduce-scatter adds received rounded halves on the one machine that keeps them, and all-gather rounds a finished
segment on its owner once and relays the owner's bytes, never encoding a value twice (a second encoding could pick
a different shift). Reduce-scatter in f16 needs segments of whole blocks and otherwise falls back to the rounds. The
driver chooses the format and passes it to the other machines in the session setup, so the lanes never disagree on
message sizes; `f16:N` limits it to all-reduces of at least N values. In this format a multi-column all-reduce is
sliced per column (slice j of every column, on block boundaries), so every element of a verification batch is added
in the same pairs as in a one-column decode step.

### A1: decode attention (`TRASHCAN_QWEN_DECODE_ATTN=1`)

`qwen_attn_decode` takes one workgroup of one wave per (KV head, partition, column). The wave holds the
queries of all G query heads that read the KV head (G = 3 at TP8, 6 at TP4 and TP2) and walks its keys eight
at a time: it issues all K and V loads of the batch, then G x 8 independent subgroup reductions, then one
online-softmax update per head. Head dimension 256, G and the lane count are specialization constants, so the
per-lane arrays stay in registers (and the kernel also runs on 8-lane subgroups for testing). Partials are
written as `[max, denominator, 2 pad, 256 sums]`, vec4 aligned. `qwen_attn_merge` loads the partition headers
together, computes each partition's weight once, issues 16 partition loads at a time per output element and
applies `sigmoid(gate)`, replacing both `attention_merge*` and `qwen_gate` (dumps run the merge ungated, capture
the output and gate it with `qwen_gate`, as before). Partition counts follow the existing buckets
(`TRASHCAN_TP_PARTS` still scales them). Prefill keeps its kernels.

### H1: fused host pass (`TRASHCAN_TP_HOSTPASS=0` restores the separate passes)

One pass adds the last round's remote half and the total to the residual and accumulates the squares (eight
running sums, the same order as before); a second writes the normed row into every local GPU's input buffer.
Same arithmetic as before, so the results are bitwise unchanged.

### F1: fused FFN (`TRASHCAN_QWEN_FUSED_FFN=1`)

`qwen_ffn_gu{1,2,3,4}` reads the gate matrix through the weight binding and the up matrix through a storage
binding, keeps `matvecq4`'s (and `qwen_q4cN`'s) lane-to-block mapping, bias, FMA chain and reduction for both,
and writes `silu(g) * u` with `qwen_swiglu`'s expression: one dispatch instead of three, the same bits.

### G1: convolution in the recurrence (`TRASHCAN_QWEN_FUSED_GDN=1`)

`qwen_gdn_conv` computes the width-4 convolution and SiLU of the q, k and v channels it reads from the raw
projection and the convolution window, with `qwen_conv`'s expression, then runs `qwen_gdn`'s recurrence. A v
channel is read by one wave, which writes its window (and verification snapshots) back. Every wave of a key head
reads the q and k windows, and a wave of another workgroup may start after one has finished, so no wave may
overwrite them: `qwen_gated_norm_conv`, the gated norm that follows behind a barrier anyway, writes the q and k
windows and their snapshots. Up to 8 columns; prefill and activation dumps (which capture the convolution output)
keep the separate kernels.

### M1: many rows per wave (`TRASHCAN_QWEN_SHORTK=8|16`)

`qwen_matvec_rows8/16` is `matvecq4` with 8 or 16 rows per wave: the same lane-to-block mapping and FMA chain per row
(so the same bits), with each step's activation loads and the final reductions shared by more rows and a quarter
or half as many waves. Used for one-column projections whose rows hold at most 2048 values: the out-projections of
TP8 and TP4 ranks.

### K1: wave-local bitonic steps (`TRASHCAN_QWEN_TOPK=wave`)

A thread's elements are assigned so that a wave owns a contiguous run of 2 x subgroup-size elements in each
half of the 1024-element list; a bitonic step of stride at most the subgroup size then compares only elements
of the executing wave and needs a subgroup barrier instead of a workgroup barrier. 10 of the 55 steps keep a
workgroup barrier. The comparisons are the old kernel's, so the candidates are identical.

## Follow-ups that need the machines, the kernel or the driver

- **X1, USB4STREAM busy polling.** 7.3 (in use since 2026-10-14, `docs/kernel.md`) has busy polling: `busy_poll`
  in a stream's configfs group, set before it is opened, runs its rings without interrupts (`RING_FLAG_NO_INTERRUPT`).
  It should remove the interrupt and wake-up from each of the 256 one-way trips per decode token. Measure
  `tbstream-bench pingpong` with and without it: a 64 B round trip is 25 µs with interrupts.
- **X2, GFX6 barrier flushes.** RADV's `can_skip_buffer_l2_flushes` is false before GFX9, so every compute barrier
  writes back and invalidates the 768 KB L2. A patched `libvulkan_radeon.so` (loaded through `VK_ICD_FILENAMES`,
  nothing installed) that treats compute-to-compute buffer barriers as GFX9 does (partial flush and L1
  invalidate) while keeping the flushes around host and transfer access would measure how much of the
  5 ms of small-kernel time is the flushes. Correctness gates: `qwen-validate`, `qwen-check` and
  `qwen-kernel-check`.
- **M1, other mappings.** If 16 rows per wave do not lift the 5120 x 768 out-projections enough, a variant with 8
  lanes per row reduces in 3 swizzle steps instead of 6; to keep verification columns equal to decode, `qwen_q4cN`
  would need the same mapping.
- **B1, balance.** Give the slower GPUs (invite, indite) 32-64 fewer FFN columns; needs a `Split` with per-rank
  widths and a per-machine calibration.
- **Syscall cost.** Ivy Bridge runs with KPTI, so each non-blocking `read` that finds nothing costs about a
  microsecond. Count reads per exchange before considering `mitigations=off` on these dedicated nodes (a
  security decision for the owner).
