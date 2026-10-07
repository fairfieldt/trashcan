# Qwen3.8-27B TP4 improvements, 2026-10-07

Historical status below was superseded by [successful resumed validation](../verification-20261007/resumed/README.md).
Tiled attention remains; KV-only MTP was later [removed](../cleanup-20261007/README.md) because its component
saving did not establish a generation-throughput benefit. Original measurements are preserved below.

Work on main from `3fe97ca`, incite + invite, four D700s, three USB4STREAM lanes. Builds use the kbuild VM;
GPU commands use paired gpuq leases and `scripts/with-clocks.sh`. Both machines use the performance governor
and a scoped 0-us CPU latency request during measurements; governors are restored after each lease.

**Validation is incomplete.** The final mixed-prompt MTP run completed seven cases, then invite's second GPU
hit a compute-ring timeout/reset at the start of the sustained case. Invite subsequently became unreachable
over LAN and Tailscale. The cause is unresolved. The new attention and KV-only paths remain opt-in in main
until this failure can be investigated and the sustained MTP and server checks repeated. No throughput result
is claimed for the failed case. All leases were released; invite's final governor restoration could not run
while the machine was unreachable and must be checked when it returns.

## Retained implementation

- `TRASHCAN_QWEN_ATTN=tiled`: Qwen prefill attention for batches of at least 64 tokens, with GQA ratios including
  six. Scores occupy a bounded 64-query slab per queue, reused after the value pass. Graph keys include a visible-key
  bucket, preventing reuse of a short-prefix launch at later positions. Full-chunk bucket graphs are recorded at load.
  Scalar attention remains the default and covers smaller batches.
- `TRASHCAN_TP_KV_ONLY=1`: target forwards extend the MTP cache using K/V projections and K normalization/cache
  stores, omitting Q, attention, gating and output projection. Full attention remains available for A/B and captures.
- `qwen-bench --corpus FILE --reps N`: JSONL cases with tools, thinking, sampling and seed settings; resident weights,
  fresh recurrent state per repetition; token IDs, exact wall-clock windows, both hosts' CPU settings, MTP statistics.
  Prefill profile tallies reset between repetitions while trace identities remain continuous.
- `qwen-attention-bench`: deterministic tiled-vs-scalar comparisons on both local GPUs, including padded query tiles,
  later positions and tails.

These flags must be applied to **both** driver and node. Experimental results below used builds with the new
paths selected by flags or enabled by default during development. The committed code gates them explicitly.

## Paired prefill results

Same `tiled-buckets` binary, old scalar path vs tiled attention, context 12288, 512-token chunks, `tg32`, no MTP.
Each configuration loaded afresh, starting after both Macs cooled to at most 82 C. These are individual paired
measurements, not confidence intervals. Short-prompt changes are comparable to the observed session variability.

| Prompt | Scalar tok/s | Tiled tok/s | Change | Both GPUs on both Macs at 850/1370 during prefill |
|---|---:|---:|---:|---|
| 512 | 150.46 | 153.72 | +2.2% | scalar all samples; tiled invite 12/13 samples |
| 2048 | 139.21 | 143.90 | +3.4% | all samples, both runs |
| 4096 | 124.86 | 135.59 | +8.6% | scalar invite 117/118; tiled all samples |
| 8192 | 106.93 | 130.61 | +22.1% | all samples, both runs |

All 32 generated token IDs and all eight prefill candidate IDs agree for each pair. Largest score difference
among the eight candidates: 0.00149 (512), 0.00106 (2048), 0.000849 (4096), 0.000828 (8192). Tiled attention changes
floating-point summation order; its prefill scores are not bit-identical to scalar attention.

Kernel comparison, 256 queries, six query heads / one KV head, dim 256, five iterations (incite):

| Visible context | Scalar ms | Tiled ms | Largest absolute output error |
|---|---:|---:|---:|
| 4096 | 38.13 / 38.12 | 6.87 / 6.87 | 1.53e-7 |
| 8192 | 77.26 / 77.18 | 13.23 / 13.18 | 1.48e-7 |

Both GPUs also passed at 64 and 129 queries (the latter exercises a one-query final slab). All measured
iterations held 850/1370. These are kernel/application measurements, not hardware rooflines.

## MTP cache update

Three-column verification profiles (`profiles/*-verify.txt`): the final segment includes head selection and
MTP cache extension. Its GPU time fell from 3014 to 2733 us on incite rank 0, and 3149 to 2863 us on invite rank 2,
about 9%; trunk segment times remained essentially unchanged. This saves about 0.28 ms per verification round.
There is no demonstrated material whole-generation speedup: cooled unprofiled runs remain around 39 tok/s.
`final-mtp` gives 39.37 at full clocks; its second repetition, 37.65, includes a low-clock sample on invite.
Profiling runs are not used as throughput benchmarks.

## Experiments not retained

1. **Persistent reductions.** A barrier-based pool and a condition-variable pool with atomic completion counters
   preserve ordered sums and pass an uneven-chunk/repeated-job unit check and the 700-token GPU rollback check.
   They eliminate about 2560 thread creations per Mac per 512-token chunk. The atomic pool reduced the driver's
   measured reduction phase (241.6/278.9 ms vs 295.8/290.5), but did not improve total prefill: 143.1/141.8 vs
   144.8/142.6 tok/s in the cooled set. The original scoped-thread path is retained.
2. **CPU affinity.** Workers on separate physical cores, main thread on CPU 0, made full-clock decode slower:
   27.2 vs approximately 28.8 tok/s. Not retained. CPU sibling pairs are adjacent logical CPUs on both Macs.
3. **Projection kernels.** Loading one Q4 block ahead is slower (4352x5120: 89 vs 84 us on incite); two rows per
   wave for three-column verification is also slower (176 vs 133 us). Both match the reference bit for bit and
   pass CPU row checks; neither is retained. Other tested shapes: 5120x4352 and 1536x5120.
4. **Compact Q4 scales.** Original f16 scales reduce allocated memory by 371.17 MiB per GPU with MTP. Dynamic
   half selection gives 37.44/37.56 tok/s with MTP; direct 16-bit loads recover 38.65/38.81, against approximately
   39.1 for f32 scales. Plain decode is 28.59/28.68, with no clear improvement. Not retained.

The saved candidate shaders, worker implementation, integration diff and source archives permit follow-up
without adding these unsuccessful paths to the engine.

## Hardware effects and failed stress run

Standalone full-clock FFN matvecs on invite are still about 5% slower than incite (4352x5120:
88.2/88.9 vs 84.4/84.1 us). Invite's second GPU also throttles during verification near 90 C: one microbenchmark
ran at 450 MHz and took 182.5 vs 133.8 us. During the longer resident set, clocks fell to 450 MHz and MTP throughput
to 27-29 tok/s. Fan control was active at 1900 RPM on both Macs; CPU throttle counters were zero when checked.
Do not attribute these throttled results to code changes or infer the cause of the full-clock difference.

The mixed MTP run's last completed case was sampled generation. The kernel log then reported:
`ring comp_1.0.0 timeout`, reset of `0000:06:00.0`, lost VRAM, and `device wedged`; all three Thunderbolt links
subsequently disconnected. See `invite-gpu-reset-netconsole.log`. Last gpuq telemetry showed 79 C and a zero
clock reading on card 1; this does not establish a thermal cause. The missing node log/clock tail and previous-boot journal must be collected after recovery; the device
coredump was inaccessible while the machine was offline. Main's new paths remain gated pending diagnosis.

## Correctness and benchmark coverage

- `cargo test --release`: 42 passed. `cargo clippy --release`: 19 existing warnings, none added.
- Initial worker/KV candidate: 700-token prompt, 256 steps; restore exact, no verification score error,
  speculative output identical.
- Tiled bucket build: 4300-token prompt, 128 steps; 85 verification rounds, restore exact, score error 0,
  speculative output identical.
- Final candidate: 700 tokens / 256 steps, same checks pass with score error 0.
- Profile totals reset correctly: two successive profiled prefills have identical per-kernel call counts.
- Plain corpus completed eight cases, including a sustained tutorial generation. MTP completed code, Spanish,
  Chinese, native tool request/result, thinking and seeded sampling before the sustained-case failure.
  All completed greedy cases' IDs match plain decoding; sampled outputs differ by design.

The repeated water-cycle benchmark's 72% acceptance does not generalize: completed mixed cases range from
31% (Chinese) to 89% (code). The per-case data are in `incite/final-corpus-*.json`; partial MTP output is deliberately
preserved. Do not pool it as though all eight cases completed.

| Case | Plain tok/s | MTP tok/s | Accepted | Greedy IDs |
|---|---:|---:|---:|---|
| code | 29.51 | 45.34 | 89.1% | match |
| spanish | 29.19 | 33.08 | 49.2% | match |
| chinese | 29.30 | 26.54 | 31.0% | match |
| tool-request | 28.63 | 40.05 | 80.0% | match |
| tool-result | 28.73 | 35.36 | 72.2% | match |
| thinking | 28.21 | 38.39 | 74.5% | match |
| sampled | 29.53 | 32.16 | 59.3% | sampled outputs differ |
| sustained | 27.77 | failed | unavailable | unavailable |

These are single mixed-session runs, including changing contexts and warm-up. Plain sustained generation
completed 2048 tokens. The missing MTP sustained case is a GPU failure, not a zero throughput result.

## Source and reproduction

`builds.sha256` identifies every deployed binary. `sources/*.tar.gz` preserve the kbuild source snapshots for
measured candidates; `final-candidate` is `ac7b4140…` and has both new paths enabled by default. The subsequent
`reviewable` build gates them explicitly and adds flags to benchmark provenance. Its optimized GPU math is the same.
Baseline is main `3fe97ca` (`bd4cb096…`). Early candidate hashes also cover intermediate metadata/experimental changes;
use the archives rather than assuming every binary is the final source. Workspaces are
`~/trashcan-exp/improvements-20261007` on each Mac; `~/trashcan` was not modified.

```sh
# Cross-build (no GPU lease required).
HOSTS="incite invite" scripts/cross-build.sh trashcan-exp/improvements-20261007/trashcan
# Both commands below must be inside their machine's gpuq lease and with clock logs.
# Include the performance governor and latency-run.py on BOTH machines for comparable benchmarks.
TRASHCAN_QWEN_ATTN=tiled TRASHCAN_TP_KV_ONLY=1 trashcan qwen-tp-node MODEL --mtp MTP \
  --stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2
TRASHCAN_QWEN_ATTN=tiled TRASHCAN_TP_KV_ONLY=1 trashcan qwen-bench MODEL --mtp MTP \
  --tp-stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 --pp 8192 --tg 32 --context 12288
trashcan qwen-bench MODEL --tp-stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 \
  --corpus tests/fixtures/qwen-bench-corpus.jsonl --tg 128 --reps 2 [--mtp MTP]
```

`scripts/cooled-sweep.py` uses the JSON plans, waits for cooling without holding leases, then requests/waits,
sets/restores governors and releases both leases per command. `scripts/cooled.py` wraps driver/node GPU commands
with gpuq, clocks and latency requests. `scripts/summarize.py` joins exact timing windows to both machines' clock
samples (`summary.json`). Per-host `commands.log`, outputs, errors and clocks are preserved. Raw profiles are gzipped;
`profiles/` contains the summaries. Check exit status and the report's failure record before treating a run as complete.
