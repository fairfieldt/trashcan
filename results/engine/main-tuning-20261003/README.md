# Main-branch performance experiments — 2026-10-03

Incite, two FirePro D700s, kernel `6.12.111-p2p`, Mesa 25.0.7. The performance branch was merged
and work continues directly on `main`. All timed commands use the temporary zero-CPU-latency
request in `scripts/latency-run.py`; source hashes, commands, raw outputs and clocks are preserved.

**Final fixed benchmark: 23.415 tok/s decode, with pp512 near 160 tok/s.** The last fresh paired
iteration improves pooled tg256 **22.779 → 23.415 tok/s (+2.79%)** and decode after a 4096-token
prefill **20.552 → 21.359 tok/s (+3.93%)**. The fixed benchmark has two repetitions; the long and
chat comparisons are single pairs. All twelve ordinary/MTP chat outputs match exactly.

The complete sequence moves fixed decode from 21.970 to 23.415 tok/s (+6.58%) and long-context
decode from 17.903 to 21.359 tok/s (+19.31%). These comparisons span separate measurement phases;
the fresh paired results below are the stronger evidence for each individual change. Prefill is
not materially faster. Short ordinary chats, repeated-prefix reuse and the resident graph-cache
fix have separate measurements, rather than a single universal speedup.

The [ceiling audit](../ceiling-audit-20261003/README.md) counts the actual GGUF tensors and distinguishes
the 65.10 tok/s weight-only bound from the 51.57 tok/s bandwidth/transport scenario. Neither is a
practical prediction. M3's 40–50 tok/s stretch target and M5's typical-chat 1.5× target remain open.
[Remaining opportunities](remaining-opportunities.md) uses measured profiles and records the
unsuccessful kernel families; no absolute maximum is claimed.

## Final fixed-dimension attention iteration

[fixed-dim/build.json](fixed-dim/build.json) records all 41 source hashes for the integrated build,
which matches source commit `01bcbfe`. Its binary SHA-256 is `5b62aa4082717a95eae881219fc258a90879ea9b35b546c149153af3e6bb0b6e`.
Its fresh baseline is commit `cc3f50f`, binary `325b087f…`, from the preceding shared-KV phase.
[summary-fixed-dim.json](summary-fixed-dim.json) confirms all micro, application and final resident
acceptance suites are complete. [delivery-build.json](delivery-build.json) links the committed source,
measured binary, accepted summary and final clock log.

The selected shaders now compile dimensions 256/512 into loops, strides and accumulator arrays.
Their supported eight-query-head mappings need one K/V plane per workgroup, reducing shared
storage from 4/8 KiB to 2/4 KiB for sliding/global shapes. Arithmetic order, partition selection,
scratch layout and scalar fallbacks are preserved.

The isolated [probe](fixed-dim-probe/probe.json) passes **64 reference rows**: four token/partition
cases × two builds × two repetitions × two dimensions × two GPUs. CPU GLSL compilation and
SPIR-V validation also pass. Means across both repetitions/cards are:

| Selected shape | Prior shared µs | Fixed-dimension µs | Time reduction |
| --- | ---: | ---: | ---: |
| Sliding256, 128 keys / 16 partitions | 86.000 | 76.613 | 10.91% |
| Sliding256, 768 keys / 32 partitions | 185.476 | 164.185 | 11.48% |
| Global512, 128 keys / 16 partitions | 124.723 | 111.396 | 10.69% |
| Global512, 768 keys / 32 partitions | 262.398 | 226.772 | 13.58% |
| Global512, 1536 keys / 64 partitions | 411.317 | 386.191 | 6.11% |
| Global512, 8192 keys / 64 partitions | 871.484 | 746.394 | 14.35% |

Dimension256/64-partition cases are unchanged scalar controls and synthetic unlimited-window
inputs, not production sliding-layer gains. The probe restores the original source and binary
exactly. Its candidate binary `81c3ea31…` precedes the integrated build's comment-only correction
in `model.rs`; the integrated build has its own complete application/reference validation.

| Fresh pp512/tg256 benchmark | pp512 tok/s | tg256 tok/s |
| --- | ---: | ---: |
| Shared 1 | 161.170 | 22.844 |
| Shared 2 | 161.199 | 22.714 |
| Fixed-dimension 1 | 158.983 | 23.227 |
| Fixed-dimension 2 | 160.916 | 23.606 |
| Shared pooled, total tokens/time | 161.185 | 22.779 |
| Fixed-dimension pooled | 159.944 | 23.415 |
| Change | −0.77% | +2.79% |

Both decode repetitions improve. The prefill path is unchanged; these small prefill differences
are not a claimed gain. The fresh long pair is **117.519 → 117.344 tok/s prefill** and
**20.552 → 21.359 tok/s decode**. Two fixed repetitions and one long pair do not establish a
precise confidence interval. [Clock alignment](fixed-dim-clock-analysis.md) finds both cards
at 850/1370 MHz in every sampled row of all six reconstructed decode windows. The windows
are approximate, and the report preserves timestamp and sampling uncertainty.

| 128-token greedy chat | Ordinary shared → fixed tok/s | MTP shared → fixed tok/s |
| --- | ---: | ---: |
| Transformer | 25.10 → 25.62 | 29.01 → 29.69 |
| Photosynthesis | 25.35 → 25.67 | 24.23 → 24.80 |
| Csv | 24.93 → 25.92 | 36.32 → 36.96 |

These paired ordinary rates improve 1.3–4.0%, MTP 1.8–2.4%. CLI rates are rounded and each
entry is a single run. All twelve texts equal the corresponding shared ordinary output;
[chat-parity.json](fixed-dim/chat-parity.json) preserves the checks. Combined MTP versus ordinary
on these three final chats is about 1.15×, below M5.

The integrated build passes all **six reference paths**, each checking 132 required tensors,
logits and ten greedy tokens, plus 30 CPU tests, 30 tokenizer fixtures, four chat templates and
10,000 changing P2P rounds in both directions. Local formatting and strict Clippy pass.
Both final boundary checks pass: 250 tokens/batch128 and 2300 tokens/batch512, each with 64/64
serial/chunked greedy tokens and top-64 overlap. Relative logit errors are 0.00009830 and
0.00135167, respectively; every verification column and speculative acceptance/rollback passes.
The numerical values match the preceding build's checks.

The final binary also passes **36 resident requests, 170 harness checks, eight exact repeat pairs,
18 exact cross-configuration comparisons and both complete protocol suites**, using the default
batch512 with and without MTP. Both scoped cleanups report no remaining processes. The full run is
[20261003T115400Z-447949](fixed-dim/batch-defaults/20261003T115400Z-447949-summary.json).

| Final default server, HTTP wall seconds | Ordinary | MTP |
| --- | ---: | ---: |
| 512-token input, one output token | 3.180 | 3.120 |
| 1763-token input, first 256-token answer | 22.364 | 17.526 |
| Identical long request repeated | 11.638 | 6.869 |
| Full-history two-token continuation | 2.796 | 2.862 |
| Short greedy repeat after long history | 1.171 | 0.783 |

The long first request reuses 467 prompt tokens. Its repeat reuses 1763 ordinary or 1762 MTP
positions; both long-history continuations reuse 1759. Peak observed visible VRAM remains
237.250/244.574 MiB ordinary and 241.594/248.824 MiB with MTP, leaving at least 7.176 MiB in
these request-boundary snapshots. The final workload includes sampler changes, eight prompt
shapes, SSE/JSON and tool/protocol checks without the earlier resident stall. These are observed
8K-context results, not continuous allocation maxima or validation of 32K contexts.

## Shared-KV phase: bounded graphs and batch512

[shared-kv/build.json](shared-kv/build.json) records all 41 tested source hashes, the benchmark
driver hash and the two binary identities. All 41 selected source hashes match commit `cc3f50f`;
the manifest was captured before that commit. [summary-final.json](summary-final.json) aggregates only complete saved evidence.

| Role in this comparison | Identity | Binary SHA-256 |
| --- | --- | --- |
| Fresh core baseline | Checkpoint `63290a5`; the same binary as the historical core final build | `926542ab49ac851b6836d0af9dd4113be82c0167dfff2e1a0e6e3896a0e93809` |
| Selected shared-KV/LRU4/batch512 build | Exact source map in `shared-kv/build.json` | `325b087f565ea4d2b0394892735ab06f9b9e1ffe3b8a6f190bbd2f9e81ea719e` |

The target and assistant reuse each K/V load across two query heads for sliding attention with
16/32 partitions, and eight query heads for global attention with 16/32/64 partitions. The
eight-partition short path stays scalar. Dispatch counts account for heads per workgroup; large
prefill retains the existing unsplit attention shader. The target command cache retains at most
four graph pairs, evicting before replacement recording. These changes preserve the adaptive
partition thresholds established in the core phase.

The fresh scalar/selected microbenchmark uses this same selected binary and explicitly selects
each kernel with `--kernel scalar|selected`. All **48 recorded checks** pass: six token/partition
cases × two kernels × two dimensions × two GPUs. Mean times across the two cards are:

| Attention shape | Scalar µs | Selected µs | Kernel speedup |
| --- | ---: | ---: | ---: |
| Sliding dimension256, 128 tokens / 16 partitions | 92.624 | 85.964 | 1.08× |
| Sliding dimension256, 768 tokens / 32 partitions | 209.481 | 185.601 | 1.13× |
| Global dimension512, 128 tokens / 16 partitions | 160.119 | 124.540 | 1.29× |
| Global dimension512, 768 tokens / 32 partitions | 371.379 | 262.305 | 1.42× |
| Global dimension512, 1536 tokens / 64 partitions | 507.998 | 411.273 | 1.24× |
| Global dimension512, 8192 tokens / 64 partitions | 1396.381 | 871.124 | 1.60× |

The earlier `attention-shared-*` sweep tested heads2/4/8 and motivated this selection; its
**96 recorded reference checks** pass: 24 JSONL files × four GPU/dimension rows. The table above
instead measures the integrated dispatch.
Dimension256 cases beyond 1024 tokens are synthetic unlimited-visibility tests, not production
sliding layers. Repeating identical micro inputs keeps caches warm; application timings bound
the useful inference claim.

| Fresh fixed benchmark, explicit batch512 | pp512 tok/s | tg256 tok/s |
| --- | ---: | ---: |
| Core 1 | 158.742 | 22.007 |
| Core 2 | 161.545 | 22.049 |
| Core pooled by total tokens/time | 160.131 | 22.028 |
| Selected 1 | 158.886 | 22.592 |
| Selected 2 | 161.069 | 22.782 |
| Selected pooled by total tokens/time | 159.970 | 22.687 |
| Selected change from fresh core | −0.10% | +2.99% |

Both decode pairs improve, but two repetitions are insufficient for a precise confidence interval.
The separate fresh batch512/pp4096/tg128 pair records prefill **117.248 → 117.471 tok/s (+0.19%)**
and decode **19.154 → 20.370 tok/s (+6.35%)** in `shared-kv/long-{core,shared}.json`. Prefill is
effectively unchanged in both workloads. The shared-KV phase rate is 13.78% higher than the initial
merged-baseline long decode of 17.903 tok/s; that comparison spans different measurement phases
and is not another fresh paired result.

Fresh CLI chats use explicit batch128, temperature zero and 128 output tokens. Each entry is one
pair, with rates rounded to two decimals by the CLI:

| Prompt | Ordinary core → selected tok/s | Change | MTP core → selected tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Transformer, 26 input tokens | 25.06 → 25.10 | +0.16% | 29.28 → 29.36 | +0.27% |
| Photosynthesis, 27 input tokens | 24.90 → 24.91 | +0.04% | 24.34 → 24.48 | +0.58% |
| CSV function, 41 input tokens | 25.34 → 25.21 | −0.51% | 36.00 → 37.24 | +3.44% |

Ordinary short chats are effectively unchanged. The MTP CSV result improves, while the other two
MTP differences are small; single pairs do not establish a general MTP percentage. All 12 saved
texts match their corresponding core ordinary output byte for byte in
[shared-kv/chat-parity.json](shared-kv/chat-parity.json).

[shared-kv/validation.log](shared-kv/validation.log) passes 30 unfiltered Rust tests, 30 tokenizer
fixtures, four chat-template fixtures, 10,000 changing P2P visibility rounds of 3840 elements in
both directions, and all six activation/logit/greedy paths. Each GPU reference path checks 132
required tensors, logits and ten greedy tokens; the maximum reported relative tensor error is
0.000995. The log records these completed checks; the driver does not emit a separate validation
process exit/window file.

Both shared-KV boundary/ring-wrap checks also complete with `pass: true` and command exit zero.
These are fresh `325b087f` measurements, separate from the historical core/pre-assistant checks:

| Latest-build check | Prompt / batch | Greedy common prefix | Top64 overlap | Relative logit error | Accepted/proposed drafts |
| --- | --- | ---: | ---: | ---: | ---: |
| [check-250.json](shared-kv/check-250.json) | 250 / 128 | 64/64 | 64/64 | 0.0000983036 | 42/146 |
| [check-2300.json](shared-kv/check-2300.json) | 2300 / 512 | 64/64 | 64/64 | 0.00135167 | 42/139 |

The 250-token check crosses the short attention boundaries; the 2300-token check exercises sliding
ring wrap and speculative rollback. Their `.command.json`, `.window.json` and `.log` files retain
the exact commands, completion status and output.

`shared-kv/*.command.json` preserves exact commands, and `*.window.json` records whole-command
start/end times and exit status, including model loading and cleanup. Those windows do not delimit
inference, so no sustained-clock percentage for this phase is inferred from them. The historical
clock reconstruction below applies only to its named earlier measurements.

### Resident command-memory diagnosis and the LRU bound

The original instrumented full cold sequence reproduced the late greedy slowdown despite about
35.6 ms of shader work per forward: requests 10/11 took **4.915/5.979 seconds**, with GPU0 mean
inter-segment gaps of **97.8/142.0 ms**. A controlled diagnostic that freed cached command buffers
on candidate-count changes reduced the same requests to **1.666/1.701 seconds** and gaps to
**3.13/4.08 ms**. Both 19-request runs passed visible-text equality. See
[diagnostic-cold-full/client.json](diagnostic-cold-full/client.json),
[diagnostic-cache-clear-full/client.json](diagnostic-cache-clear-full/client.json), and their
`request-0010.json`/`request-0011.json` profiles. The diagnostic source and binary are identified in
[server-cache-clear-base.json](server-cache-clear-base.json); this is separate from production
benchmark evidence.

The diagnostic cleared four graph keys / 776 command buffers before stochastic requests in
1.209 ms, then two keys / 388 buffers before greedy requests in 0.632 ms. This supports command
storage/submission pressure as the source of the large gaps rather than increased shader work.
Visible VRAM stayed at 222.250/229.574 MiB across these clears, then grew in 9.094 MiB/card steps
as new shapes were recorded, reaching 256.000/255.996 MiB by request 16. The subsequent same-process
short replay cleared 11 graph pairs / 2134 commands, yet stochastic requests still took
5.129/5.331 seconds with GPU0 gaps of 112.7/119.4 ms. See the memory snapshots in the full client
report, [short-client.json](diagnostic-cache-clear-full/short-client.json) and
[server.log](diagnostic-cache-clear-full/server.log). Persistent driver backing storage/pooling is
a plausible explanation, not directly traced: samples bracket whole requests, including clear
and re-recording. Clearing only on sampling-mode changes was insufficient in this diagnostic.

The selected implementation therefore evicts the least recently used graph pair **before**
recording a replacement. Four times the observed batch128 graph increment is 36.375 MiB/card;
this measures the command-storage increment for that diagnostic configuration. It does not bound
all driver memory or establish a universally safe cache size. The later fixed-dimension source
corrects the original comment that rounded this to 38 MiB and attributed it to batch512.
In [model.rs](../../../engine/src/model.rs), `GraphCache::make_room` drops the old graphs before
either new GPU graph is recorded; `Graph::drop` frees their command buffers. Completed-forward
fences guard eviction, and partial
record failures release owned command buffers. The two cache tests cover recency, dropping
before replacement and retrying a failed replacement. The completed mixed-shape HTTP trials
below avoid the prior residency/latency cliff.

### Default batch512 and resident acceptance

The [default-batch audit](batch-defaults-analysis.md) records **72 requests, 338 harness checks,
16 exact repeated-output pairs, and two complete protocol suites**, all passing. All 108
cross-configuration comparisons match output and token counts. Batch512 and 512-MTP omit
`--batch` to test the default; all four servers stop cleanly after testing.

| Request, HTTP wall seconds | Batch32 | Batch128 | Batch512 default | Batch512 + MTP |
| --- | ---: | ---: | ---: | ---: |
| 512-token input, one output token | 7.865 | 3.954 | 3.194 | 3.156 |
| 1763-token input, first 256-token answer | 39.766 | 26.302 | 22.639 | 17.636 |
| Same input and answer repeated | 46.462 | 29.441 | 11.989 | 7.104 |
| Full-history continuation, two output tokens | 41.198 | 20.442 | 2.783 | 2.846 |

The first long request reuses 467 tokens from the preceding medium input, so it is not cold.
The long repeat safely misses with batch32/128 after the 256-token answer, while batch512
retains 1763 tokens (1762 with MTP's final-token replay). These are matched request-latency
comparisons including caching, not GPU decode-rate measurements. Short ordinary greedy
repeats remain near 1.2 seconds after long history and sampling changes.

Peak observed visible VRAM is 237.250/244.574 MiB with ordinary default512 and
241.594/248.824 MiB with MTP, below each card's 256 MiB capacity. The smallest observed
headroom is **7.176 MiB**. Device counters include display/other clients; the 20 snapshots per
configuration are not continuous maxima. Full target buffer allocations at 8K/batch512 are
4556.095 MiB/GPU, with approximately 415 MiB more for the assistant, excluding driver storage.
32K execution remains unvalidated. The audit preserves full VRAM, visible VRAM, GTT and RSS
measurements and their limits.

## Historical core phase: measurement and source identity

The remaining core tables compare the merged `ee9bea2` baseline with the completed `63290a5`
checkpoint. They precede the shared-KV/LRU4/default512 build and retain their original provenance.

Commands use `sudo -n python3 scripts/latency-run.py ...`, which requests zero CPU latency for the
command's lifetime. [core-build.json](core-build.json) maps the completed evidence to these builds:

| Evidence | Source identity | Binary SHA-256 |
| --- | --- | --- |
| Baseline kernel/application runs | `baseline-source.sha256`, all 36 entries match `ee9bea2` | `2e8d8c66990913d70e820adc61671fa93f0674b37cf22ab22a3e106074d90c40` |
| Core final reference suite, application benchmarks, MTP HTTP | `final-source.sha256`, all 37 entries match `63290a5` | `926542ab49ac851b6836d0af9dd4113be82c0167dfff2e1a0e6e3896a0e93809` |
| Ordinary HTTP and `long-check.*` | `pre-assistant-topk/final-source.sha256` | `7d6ffde33b58614d7a6d82ef5e77aec6710a9550ea5e665ec41126d2581ab820` |

The [pre-assistant snapshot](pre-assistant-topk/README.md) preserves the previous assistant/server
source and evidence. Its only engine-source differences from the core final manifest are
`assistant.rs` and `server.rs`, before resident assistant graphs became keyed by candidate count
and greedy requests selected one candidate. Ordinary HTTP and `long-check.*` in this directory are
byte-identical copies of that earlier evidence, **not reruns of the final binary**. The final binary
has its own six-path reference suite, application runs, 250-token boundary/MTP check and MTP HTTP
reports. `selected.json` is an earlier Q4-only experiment, before adaptive attention, and is kept
separate from final results. Trial sources live in `shader-candidates/`.

[measurement-windows.json](measurement-windows.json) preserves copied file timestamps and hashes
before Git loses mtimes. It reconstructs approximate inference windows by anchoring benchmark JSON
mtime to the end of decode and subtracting reported phase durations. In those approximate decode
windows, both cards are at **850 MHz core / 1370 MHz memory in every sampled row**: 43 samples for
each fixed run, 27 for long baseline and 24 for long final. Combined prefill/decode windows have
94.5–98.2% of samples with both cards at those clocks. Final fixed-run temperature peaks are 71/70 °C;
the long final window reaches 84/82 °C.

These are reconstructed windows, not exact start/stop instrumentation. Copied mtimes commonly have
whole-second granularity; stdout buffering, profile writes, cleanup and copying can shift the
anchor. The reconstruction assumes preserved incite timestamps on the same clock as `clocks.txt`.
The approximately 267 ms sample interval can miss brief changes, and sample fractions are neither
time-weighted duty cycle nor utilization. No CLI chat windows are inferred from rounded rates.
The clock log also covers uploads, idle periods and later experiments; its whole-log maxima must
not be attributed to a particular core benchmark. The captured window manifest hashes an earlier
clock-log snapshot, while the live log and regenerated summary can extend beyond it.

Matrix experiments warm each card for 500 ms, then time repeated GPU dispatches using Vulkan
timestamps. Decode candidates use 1000 iterations; batched/GEMM candidates use 100. Selected rows
and every input column are checked against a CPU dot product with nonuniform inputs. Attention
experiments warm for 300 ms and check six output elements against a CPU f64 softmax/reference.
All means below average the two GPU measurements; a single card's surviving row is not a complete pair.

Repetition keeps inputs, instructions and translations warm. Repacked matrix payload sizes are 17.58 MiB
for a local FFN projection, 4.69 MiB for a sliding Q/O projection, 2.34 MiB for sliding K and 1.17 MiB
for global K. These exceed Tahiti's 768 KiB shared L2, so the FFN result is not an entirely cache-resident
weight test. Smaller attention workloads can benefit substantially from cache. AMD lists cache sizes
in section 2.10 of its [OpenCL optimization guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/programmer-references/AMD_OpenCL_Programming_Optimization_Guide2.pdf).
Reported GB/s is effective weight payload divided by time, not a hardware DRAM counter. Q4 original
bytes use 18 bytes/block; the implemented repack uses 20. Application results remain the deciding evidence.

## Historical core: decode Q4 experiments

The retained core kernel decodes unsigned nibbles through `unpackUnorm4x8`, shares
the `8 × sum(input)` correction across output rows, then applies each block's scale. Wide FFN shapes
retain four output rows per wave and 256 threads per workgroup. Narrow matrices with both output
rows and input columns at most 4096 use two output rows per wave and 512 threads. Both dispatches
still produce 16 output rows per workgroup. Weight format is unchanged.

| Projection | Baseline mean µs | Selected mean µs | Selected geometry |
| --- | ---: | ---: | --- |
| Sliding FFN gate | 108.214 | 104.897 | unsigned, four rows/wave |
| Sliding FFN down | 103.468 | 100.288 | unsigned, four rows/wave |
| Sliding attention Q | 37.740 | 34.719 | unsigned, two rows/wave |
| Sliding attention O | 47.976 | 41.125 | unsigned, two rows/wave |
| Global attention K | 23.444 | 16.677 | unsigned, two rows/wave |
| Global attention Q | 63.438 | 62.669 | unsigned, two rows/wave |
| Global attention O | 106.274 | 89.328 | unsigned, two rows/wave |

The first four comparisons use `baseline-*` and `matvecq4-unsigned*` files. Global comparisons use
`global-main-baseline-*` and `global-q4-selected-*`. There is no recorded `baseline-attn_k.jsonl`
for the sliding K projection, so no baseline gain is invented for it.

Rejected alternatives establish why the selector is shape-dependent:

| Candidate family | Recorded result | Decision |
| --- | --- | --- |
| Signed row geometry: one, two or eight rows/wave | One-row FFN gate 124.140 µs; eight-row 130.417 µs vs 108.214 baseline. Two-row helps narrow projections but FFN down grows from 103.468 to 106.293 µs. | No universal signed geometry replacement. |
| Two-block unroll | FFN gate/down 110.052/105.995 µs; attention O 56.074 µs vs 47.976. | Rejected. |
| Half-block mapping | FFN gate/down 111.539/112.501 µs. | Rejected. |
| Whole-block mapping | FFN gate/down 136.912/145.183 µs. | Rejected. |
| Unsigned two-row everywhere | FFN down 104.828 µs vs 100.288 with unsigned four-row. | Keep two-row only for narrow shapes. |
| Unsigned eight-row | FFN gate 126.083 µs; attention O 65.734 µs. | Rejected. |

Moving the zero-point correction changes floating-point association. The synthetic micro inputs
are multiples of 1/16, which makes this correction unusually exact; those checks alone do not
establish real-activation fidelity. The completed six-path final reference suite passes all
132 required tensors, logits and ten greedy tokens; worst relative tensor error is 0.000995.
Long-ring, verification and continuation checks are recorded separately below.

## Historical core: batched verification and prefill experiments

The existing four-column verification kernel and 64×64×32 GEMM remain selected. Faster individual
shapes did not justify broad replacements that slowed the much larger FFN work.

| Trial | Comparison | Decision |
| --- | --- | --- |
| Small-batch row-major accumulation | FFN gate batch 4: 0.293071 → 0.403324 ms; down batch 4: 0.282119 → 0.449485 ms. Attention Q batch 2 improves 4.5%, but most tested shapes regress. | Rejected. |
| Small-batch unsigned decode | All nine tested shape/batch pairs regress about 0.9–5.2%; gate batch 8: 0.545407 → 0.564099 ms. | Rejected; unsigned remains decode-only. |
| GEMM transposed K=32 shared tiles | Gate batch 512: 13.561607 → 14.389148 ms; down: 13.890380 → 14.733061 ms. | Rejected. |
| GEMM transposed K=64 shared tiles | Gate batch 512: 13.561607 → 16.970555 ms; down: 13.890380 → 17.257108 ms. | Rejected. |

Both transpose layouts improve attention Q at batch 64 by about 3–5%, but lose at larger attention
batches and on all tested FFN shapes. Files use `multi-*` and `gemm-*` prefixes and retain batches
2/4/8 and 64/128/512, respectively. These tests did not establish a new prefill speedup.

## Historical core: attention partitions and packed loads

The new microbenchmark sweeps 8, 16, 32, 64 and 128 independent key partitions, then merges partial
softmax results. Too many partitions waste work at short prefixes; too few serialize long global
attention. Specialization constants permit the same shader source with different partition counts.

The retained application policy is **8 at ≤32 visible tokens, 16 at ≤256, 32 at ≤1024, and 64 above
1024 for global layers**. Sliding attention caps visibility at 1024 and therefore never selects 64.
Decode/small-batch target graphs are keyed by the chosen counts; assistant graphs use matching
partition layouts and target-cache visibility. Large prefill still uses the existing unsplit shader.
Boundaries are empirical policy choices; they are not a claim of an exact optimum at every length.

| Visible tokens | Dimension | Fixed 32 partitions, µs | Candidate partitions | Candidate µs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 96.233 | 8 | 43.684 |
| 1 | 512 | 167.406 | 8 | 63.721 |
| 128 | 256 | 111.137 | 16 | 92.596 |
| 128 | 512 | 194.867 | 16 | 160.299 |
| 256 | 256 | 132.319 | 16 | 132.101 |
| 256 | 512 | 230.795 | 16 | 232.453 |
| 1024 | 512 | 440.622 | 32 | 440.622 |
| 1536 | 512 | 582.913 | 64 | 508.932 |
| 8192 | 512 | 2407.273 | 64 | 1396.939 |

These are `attention-specialized-*` measurements. The 256-token global result is effectively a
tie, slightly favoring 32; the policy keeps a common short-prefix boundary. At 8192, 128 partitions
are slower than 64 (1736.243 versus 1396.939 µs for dimension 512), so 128 is not selected. Dimension
256 measurements above 1024 use synthetic unlimited visibility and do not represent production
sliding layers.

Packed f16-pair loads were also tested in `attention-packed-*`. They are generally worse at short
prefixes and inconsistent at long ones: at 128 tokens/16 partitions, dimension 256 grows from
92.596 to 96.226 µs; at 1536 tokens/64 partitions, global attention grows from 508.932 to 545.667 µs.
Some fixed-32 global cases improve, but this does not support replacing the then-selected loads under
the adaptive policy. Packed loads are rejected; specialization and adaptive partition counts are
retained with the short-chat and long-context evidence below.

## Historical core: completed application results

| Build/run | pp512 tok/s | tg256 tok/s |
| --- | ---: | ---: |
| Baseline 1 | 160.986 | 22.002 |
| Baseline 2 | 161.474 | 21.938 |
| Baseline pooled by total tokens/time | 161.230 | 21.970 |
| Preliminary selected Q4, before adaptive attention | 161.213 | 22.121 |
| Core final 1 | 160.904 | 21.741 |
| Core final 2 | 161.096 | 21.997 |
| Core final pooled by total tokens/time | 161.000 | 21.869 |
| Core final change from pooled baseline | −0.14% | −0.46% |

The preliminary selected decode difference is only about **+0.69%**, with one candidate run.
It did not establish a final application gain. The two final repetitions show no gain on this
workload; the measurements are not enough to estimate a confidence interval or a precise small
regression. pp512/tg256 decodes at visible lengths 513–768, where the adaptive policy stays at
32 partitions. It does not exercise the short-prefix or long-global partition choices.

All CLI chats use batch128, temperature zero and 128 output tokens. The table uses rates printed
to two decimal places in `chat-{baseline,final}-*.log`; each pair is one run.

| Prompt | Ordinary baseline → final tok/s | Change | MTP baseline → final tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Transformer, 26 input tokens | 24.17 → 25.25 | +4.47% | 28.87 → 29.17 | +1.04% |
| Photosynthesis, 27 input tokens | 24.14 → 24.88 | +3.07% | 23.99 → 24.50 | +2.13% |
| CSV function, 41 input tokens | 24.30 → 25.27 | +3.99% | 33.99 → 37.09 | +9.12% |

`chat-parity.json` covers all 12 saved outputs; every final and MTP output equals its corresponding
ordinary baseline text. MTP accepted/proposed counts are unchanged within each pair: transformer
82/138, photosynthesis 72/165, CSV 90/111. MTP is workload-dependent: it remains slightly slower
than ordinary decode for photosynthesis. Short-prompt prefill rates fluctuate in both directions
and do not establish a prefill improvement.

For `long-{baseline,final}.json`, batch512/pp4096/tg128 prefill stays **117.458 → 117.401 tok/s
(−0.05%)**, while decode rises **17.903 → 19.083 tok/s (+6.59%)**. This single pair exercises the
64-partition global path. It supports a long-context gain for the combined core change, without
isolating every change's contribution or establishing a universal long-context percentage.

The CLI now emits an already sampled token before computing its successor and skips the final
unused ordinary forward. This improves first output latency and saves one forward per finite
output budget (one of 128 forwards in these ordinary chats); that behavioral saving must be
distinguished from shader throughput changes. The fixed `bench` workload retains its existing
forward-count convention, so the CLI saving does not explain a synthetic benchmark speedup.

Target allocations at 8K/batch512 rise from 4552.079 to **4556.095 MiB/GPU**, matching the extra
4.016 MiB needed for 64-partition attention scratch. The assistant adds approximately 415 MiB.
These buffer totals exclude driver-managed command buffers and are not a residency guarantee.

## Historical core: prefix checks and output equality

Resident HTTP prefix reuse retains exact token prefixes only while required sliding history
survives, tracks rejected speculative writes, and either reuses compatible saved logits or replays
the final prompt token. `--no-prefix-cache` provides the cold comparison. Reused input tokens are
reported explicitly.

`prefix-{cold,cached,cold-mtp,cached-mtp}.json` each contains all **19 expected requests**, complete
bodies, outputs, timing and cache accounting at batch128. All four report `passed: true` and
`repeat_output_exact: true`; both cached reports also have `cold_comparison_exact: true`. The
summary independently verifies request/output equality and output-token counts for all 19 pairs
in each mode. Protocol/cache-boundary success and output equality are separate checks:
**`passed: true` alone does not imply exact parity**. Replaying a drafter's final prompt token can
change floating-point association, so future output differences must remain visible and be
excluded from equal-work latency claims.

| Case | Ordinary cold → cached seconds | Speedup | MTP cold → cached seconds | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Repeated 1403-token prompt, 128 output | 19.313 → 6.255 | 3.09× | 16.673 → 16.760 | 0.99×; cache miss |
| Repeated 1403-token prompt, 32 output | 14.601 → 1.532 | 9.53× | 13.946 → 1.265 | 11.02× |
| Full-history continuation, 1457 input / 2 output | 14.366 → 1.080 | 13.30× | 14.412 → 1.131 | 12.75× |
| Tool continuation, 110 input / 22 output | 4.666 → 3.794 | 1.23× | 2.318 → 1.819 | 1.27× |

These are resident request durations for exact matched outputs, not model loading time or a
general service-level latency promise. The complete scripted sequence has pooled cold/cached
latency ratios of **1.66× ordinary** and **1.17× MTP**; those ratios depend on its particular mix
of repeated, changed, stochastic and tool requests.

Ordinary 128/129-output repeats retain the full 1403-token prompt; 130 output crosses the batch128
rewind allowance and correctly falls back to zero cached tokens. MTP's additional verification
writes and final-token replay make the 128/129/130-output repeats cold at this capacity. Its
32-output repeat retains 1402 tokens. Full-history continuation reuses 1399 tokens in both modes.
Changed system messages and shorter prompts correctly invalidate reuse.

Caching is not uniformly faster: the final MTP stochastic repeat is **1.205 → 1.972 seconds**
despite exact output and 46/47 cached input tokens; ordinary tool repetition is also slightly
slower. The instrumented `diagnostic-{cold,cached,cold-full}/` runs preserve request and segment
evidence for the short-request timing anomaly. Their simplified cold/cached sequences and full
cold replay pass visible-text equality, but instrumented timings are separate evidence and do
not replace production reports. The full cold replay reproduces the greedy requests at
4.915/5.979 seconds: average GPU kernel work remains about 35.6 ms per decode forward, while
inter-segment gaps grow to about 98/142 ms on GPU0, versus approximately 3 ms on the preceding
stochastic requests. This motivated the controlled graph-cache diagnostic and bounded cache described in the shared-KV
phase above; these original timings do not validate the later production fix.

## Historical core: completed correctness evidence

- `final-validation.log`: 26 Rust tests, 30 tokenizer fixtures, four chat-template fixtures and
  10,000 changing P2P visibility rounds in both directions pass on incite. All six reference paths
  pass: ordinary batches 1/2/4/15, CPU relay, and batch8 MTP. Each compares 132 required tensors,
  logits and ten greedy tokens; maximum reported relative tensor error is 0.000995.
- `long-check.json`: the preserved pre-assistant build passes 2300-token/batch512 ring wrap and
  speculative rollback, with all 64 compared greedy tokens identical, top64 overlap 64/64,
  relative logit error 0.00135167 and 42/139 accepted draft proposals.
- `boundary-check.json`: the core final build passes 250-token/batch128 execution across short
  attention boundaries and MTP, with 64/64 greedy-prefix and top64 agreement, relative logit error
  0.0000983036 and 42/146 accepted proposals.
- All four prefix suites are complete as described above. `protocol-cached.log` and
  `protocol-cached-mtp.log` pass the Responses integration suite: JSON/SSE, persistent HTTP,
  candidate-count switching, seeded parity, limits, reasoning, invalid requests, native tools,
  namespaces, custom tools and client tool search. Ordinary protocol evidence uses the preserved
  pre-assistant build; MTP protocol evidence uses the core final build.
- `local-tests.log` and `local-clippy.log` pass their CPU checks. Their explicit `glslc missing`
  warning means those local runs do not validate shaders; GPU evidence comes from incite above.

## Reproduction

From the repository root, summarize each measurement phase separately:

```sh
python3 results/engine/main-tuning-20261003/summarize.py > results/engine/main-tuning-20261003/summary.json
python3 results/engine/main-tuning-20261003/summarize-final.py > results/engine/main-tuning-20261003/summary-final.json
```

The historical summarizer reports missing/in-progress files, two-GPU micro pairs, weighted
application throughput, per-request cold/cache equality and eligible latency ratios, provenance
files, and clock samples.
It also reads the 12 CLI chat metrics and saved outputs, `chat-parity.json`, both long-context
measurements and the boundary check. Default prefix inventory requires four completed reports
with all 19 named requests each. It does not infer GPU reference acceptance from benchmark JSON;
that evidence is in the validation logs. The saved summary is a snapshot and may include baseline
micro rows for later experiments without implying those candidates completed.

Candidate replay on incite uses `sweep-shaders.py` with names from `shader-candidates/`, or
`sweep-batched.py multi|gemm` with candidate filenames. These scripts temporarily replace shader
source, build and measure sequentially, then restore source in `finally`; rebuild the restored
source before any later application run. Their `baseline` label means the source present when the
script starts, so reproduce the saved baseline source before running against arbitrary new code.
`bench-final.py` records exact application commands: fixed pp512/tg256 repeats, three ordinary/MTP
chats, pp4096/tg128 and the 250-token boundary check. `check-servers.py` runs cold/cached ordinary
and MTP servers sequentially with the prefix harness and protocol checks. Reproduce the source
manifest corresponding to the requested evidence; do not relabel preserved ordinary HTTP or
long-check files as fresh measurements of a later binary.

`bench-shared-kv.py` records its phase's exact commands, whole-command windows, source manifest and
binary hashes. `summarize-final.py` requires complete micro inventories, reference checks, fixed
and long-context measurements, chat parity and boundary checks before marking the shared inventory
complete. Resident batch reports must match the selected binary and all overlapping declared
source hashes; it selects one invocation rather than assembling passing cases from different runs.
Its `shared_inventory_complete` field is separate from resident default-batch acceptance. Missing
or copied-in-progress files must never be treated as passed or replaced with older-build results.


`bench-fixed-dim.py` performs the final full-model phase against the saved `cc3f50f` binary.
`sweep-fixed-dim.py` records the isolated probe and restores its pinned baseline source/binary.
`summarize-fixed-dim.py` checks all 64 probe rows, fresh full-model measurements, source/command
identity and the final 512/512-MTP resident reports independently of the earlier summaries:

```sh
python3 results/engine/main-tuning-20261003/summarize-fixed-dim.py > results/engine/main-tuning-20261003/summary-fixed-dim.json
```

The final HTTP commands use `bench-batch-defaults.py --labels 512 512-mtp --use-default-512`
with `--output-dir results/engine/main-tuning-20261003/fixed-dim/batch-defaults`. Saved commands
and manifests remain authoritative; do not overwrite earlier runs or relabel them as a later build.
