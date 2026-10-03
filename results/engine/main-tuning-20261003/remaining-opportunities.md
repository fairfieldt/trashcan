# Remaining measured opportunities

This is a stopping-point assessment, not a claim of maximum performance. The preceding profile
reference is checkpoint `cc3f50f`, binary
`325b087f565ea4d2b0394892735ab06f9b9e1ffe3b8a6f190bbd2f9e81ea719e`.
The final fixed-dimension phase uses binary
`5b62aa4082717a95eae881219fc258a90879ea9b35b546c149153af3e6bb0b6e`, identified by
[`fixed-dim/build.json`](fixed-dim/build.json). Source commit `01bcbfe` matches all 41 declared source
hashes. All six reference-validation paths, both boundary checks and both HTTP protocol suites pass;
all twelve chat outputs match exactly. The final HTTP rerun independently passes 36 requests,
170 harness checks, eight exact repeat pairs and 18 exact cross-mode comparisons. Both server
cleanup reports have no remaining tracked processes, and memory peaks agree with the raw snapshots.

## Where the time remains

These are the last actual forward's two-GPU **mean** timestamp totals, in milliseconds. The GPUs run
concurrently; their times must not be added. These profiles describe one forward, not the entire run.

Preceding `cc3f50f` phase:

| Saved profile | Attention segments | FFN segments | Final head segment | Inter-segment gaps | Timeline total |
|---|---:|---:|---:|---:|---:|
| `shared-kv/chat-shared-transformer.tsv` | 15.077 | 18.221 | 2.993 | 2.597 | 38.888 |
| `shared-kv/decode-shared-2.tsv` | 19.781 | 18.214 | 2.994 | 4.311 | 45.300 |
| `shared-kv/long-shared.tsv` | 23.620 | 18.241 | 2.993 | 4.068 | 48.921 |
| `shared-kv/decode-shared-2.prefill.tsv` | 1012.710 | 2076.749 | 3.316 | 19.015 | 3111.790 |

Final fixed-dimension phase (`5b62aa40`):

| Saved profile | Attention segments | FFN segments | Final head segment | Inter-segment gaps | Timeline total |
|---|---:|---:|---:|---:|---:|
| `fixed-dim/chat-fixed-transformer.tsv` | 14.502 | 18.151 | 2.991 | 2.701 | 38.345 |
| `fixed-dim/decode-fixed-2.tsv` | 18.693 | 18.147 | 3.000 | 2.531 | 42.370 |
| `fixed-dim/long-fixed.tsv` | 22.058 | 18.118 | 2.996 | 2.601 | 45.773 |
| `fixed-dim/decode-fixed-2.prefill.tsv` | 1015.007 | 2076.925 | 3.319 | 19.603 | 3114.854 |

The separate application timings compare the new build with fresh shared-build runs: pooled tg256
is 22.779 → 23.415 tok/s, and tg128 after pp4096 is 20.552 → 21.359 tok/s. Long prefill is
117.519 → 117.344 tok/s, with no demonstrated prefill improvement. These whole-run measurements
are distinct from the last-forward budgets above; the single-profile gap difference is not a
measured synchronization optimization.

“Attention” includes Q/K/V and output projections, normalization, RoPE, KV work and attention kernels.
It is not a measurement of softmax or KV loading alone. FFN includes its projections, activation,
normalization and communication work. The final head segment includes normalization, vocabulary
projection, softcap/top-k and merge. Gaps include peer dependencies and scheduling, not just CPU cost.

## Most credible next work

1. **FFN and verification kernels have the largest broad budget.** FFN accounts for 40–47% of the
   final-build decode timelines and 67% of its shown 512-token prefill chunk. A hypothetical 20% reduction in the
   whole FFN category would improve these decode timelines by about 8.6–10.5%, and that prefill chunk
   by about 15%. Those are conditional Amdahl calculations, not measured gains. The next useful kernel
   hypothesis would need to reduce actual weight/dequantization or activation traffic—such as measured
   gate/up reuse—rather than repeat already-losing row geometries. The retained Q4 shaders already
   share activation loads and zero-point correction across rows.

2. **Adapt MTP to acceptance and measured verification cost.** The final-build chat logs give
   photosynthesis 72/165 accepted proposals and 24.80 tok/s, versus 25.67 ordinary; CSV accepts 90/111
   and reaches 36.96 versus 25.92. Transformer reaches 29.69 versus 25.62. Roughly 81–82 ms of verification
   plus three 4 ms draft proposals requires about 2.4 committed tokens per round to beat a 39 ms
   ordinary token. Photosynthesis averages about 2.31, CSV about 3.43. A bounded adaptive-depth or
   ordinary-decoding fallback is therefore more credible than a universal larger draft count. It still
   needs exact greedy and stochastic-law validation; these three paired chats do not establish a
   general MTP percentage. See `fixed-dim/chat-fixed-*.log` and its exact chat parity file.

3. **Long-context attention remains worth isolated profiling.** Its category grows while FFN/head
   remain almost constant, but the table cannot identify how much is projection versus inner attention.
   The fixed-dimension change has completed boundary and HTTP validation; a further attention change
   needs a new isolated measurement and matched-output application comparison.
   Synthetic unlimited-visibility dimension-256 results above 1024 tokens are not production sliding
   attention measurements. More head sharing or partitions is not automatically better.

4. **Head and scheduling work have smaller limits.** `headvec.comp` already shares each activation
   vector across four vocabulary rows per wave. Its unchanged 629,145,600-byte payload per card takes
   at least 2.392 ms at the audited 263.04 GB/s bandwidth, before other overhead. Against the sampled
   approximately 3 ms head category, this leaves only about 0.6 ms, or roughly 1.3–1.6% of the final decode
   timeline, as an optimistic saving. A 1.5 ms head would violate this unchanged-traffic bound;
   representation changes are outside this comparison. Removing every observed gap—an unattainable assumption
   with real dependencies—would give roughly 6–8% across the final short/fixed/long examples.
   These are bounded candidates for head specialization or carefully measured synchronization work,
   not reasons to relax correctness or grow cached graphs. Final-build HTTP memory snapshots confirm
   only 7.176 MiB of visible-VRAM headroom on the fuller card with MTP.

## Experiments that already lost

| Family | Representative measured result | Implication |
|---|---|---|
| Q4 universal row geometry / more unrolling | Two-block FFN gate 110.052 µs versus 108.214 baseline; unsigned two-row FFN down 104.828 µs versus retained four-row 100.288 | Keep the measured shape selector. |
| Q4 half-/whole-block mappings | FFN gate 111.539 / 136.912 µs versus 108.214 baseline | Reorganizing lanes alone did not improve throughput. |
| Verification row-major accumulation | Batch-4 FFN gate 0.293071 → 0.403324 ms; down 0.282119 → 0.449485 ms | A small attention-Q win did not offset the larger FFN regression. |
| Verification unsigned unpack | All nine tested shape/batch pairs regressed about 0.9–5.2% | Decode's unsigned improvement does not transfer automatically to verification. |
| Transposed GEMM tiles | Batch-512 FFN gate 13.561607 → 14.389148 / 16.970555 ms | Retain the current prefill tile until a new hypothesis beats it end to end. |
| Packed attention loads / more partitions | Global 1536-token attention 508.932 → 545.667 µs with packed loads; 8192-token 128 partitions slower than 64 | Keep measured context/shape selection. |
| Unbounded resident graph caching | Visible-VRAM exhaustion caused 98–142 ms gaps with unchanged shader work | Preserve eviction before new recording; late clearing is insufficient. |

Raw evidence is in the `matvecq4-*`, `multi-*`, `gemm-*`, `attention-packed-*` and
`attention-specialized-*` files, with the [experiment summary](README.md) and
[resident diagnostic analysis](diagnostic-analysis.md) recording qualifications. The recorded reference,
application, boundary and HTTP checks complete this pass. Larger future gains require a new measured
hypothesis; no global
hardware or practical maximum has been demonstrated.
