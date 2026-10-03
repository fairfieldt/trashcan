# Performance tuning — 2026-10-03

Incite, two FirePro D700s, kernel `6.12.111-p2p`, Mesa 25.0.7. The full-model baseline is commit
`f63fe89`. Measurements use the existing command-scoped CPU latency request of 0; clock logs retain
idle/upload periods as well as 850/1370 MHz under sustained load. The baseline and final directories
record source/binary hashes, profiles and all three chat outputs.

## Implemented changes

- **Q4 decode and verification:** XOR each packed nibble's sign bit, then use signed bitfield extraction
  to obtain `q-8` directly. Weight layout and accumulation order remain the same. This removes the
  separate integer subtraction from each unpacked value.
- **Five-to-eight-column verification:** use two four-column Q4 tiles with the existing four-row
  kernel. Update the host Y dispatch width to four columns, including tail tiles. The Q6 head keeps
  its previous eight-column kernel. This final selector adjustment is recorded separately below.
- **Candidate selection:** reduce each wave's candidates before combining four wave winners. The final
  merge consumes the sorted per-chunk lists by caching their heads and advancing only the winning list,
  instead of scanning all `128 × k` candidates for every output rank. Token-ID tie order is preserved.
- **Greedy HTTP requests:** cache separate one-candidate and 64-candidate target graphs. Greedy requests
  use the former, and stochastic requests use the latter. Assistant candidate capacity stays at 64.
- **Measurement:** warm each GPU for 500 ms, check nonuniform vectors against CPU dots, measure multiple
  projection shapes, and validate ranking against a CPU sort with ties, softcap and suppression.

## Measurements

Warmed matrix measurements use 1000 iterations, and are averaged across the two GPUs. Bandwidth counts
original Q4 bytes; repacked-byte bandwidth is recorded separately in each JSONL file.

| Q4 projection | baseline GB/s | final GB/s | gain |
| --- | ---: | ---: | ---: |
| FFN gate | 145.12 | 153.00 | 5.4% |
| FFN down | 148.37 | 160.41 | 8.1% |
| Attention query | 108.37 | 117.22 | 8.2% |
| Attention output | 81.97 | 91.87 | 12.1% |

Q4 unpacking alone improves FFN verification by 2.5%, 3.0% and 3.3% at batches 2, 4 and 8. The final
selector adjustment improves batches 5, 6, 7 and 8 by another 20%, 37%, 53% and 67%, respectively,
relative to the previous optimized eight-column kernel. Batch 8 takes 0.546 ms versus the original
0.945 ms, about **1.73× faster** overall. One-column top-64 selection
(including merge, softcap and suppression) falls from about 2.20 ms to 0.51 ms per GPU: **4.3× faster**.

| full model | baseline tok/s | final tok/s |
| --- | ---: | ---: |
| pp512 | 161.05 | 160.87 |
| tg128 after pp512 | 21.33 | 22.14 |

Decode improves **3.8%**; prefill is effectively unchanged. A separate pp512/tg256 pair confirms the
gain: **21.26 → 21.88 tok/s (+2.9%)**, with prefill **160.94 → 161.05 tok/s**. The final-token profile records 43.24–43.26 ms
of GPU segment work before versus 41.74–41.77 ms afterward. Median semaphore gaps are 24–28 us in these
runs, so the improvement comes from GPU work. Individual profile samples are not averaged over tokens.

| 128-token chat | ordinary before / after | MTP before / after |
| --- | ---: | ---: |
| Transformer | 23.62 / 24.25 | 28.46 / 29.05 |
| Photosynthesis | 23.28 / 23.82 | 23.13 / 23.68 |
| Python CSV | 23.01 / 24.01 | 34.70 / 34.89 |

All six final outputs equal the baseline ordinary output byte for byte. MTP changes are small, and the
CSV MTP difference is within plausible run-to-run variation. The combined MTP gain remains about 1.2×.

## Resident HTTP server

Each row is total wall time for the same 128-token greedy Responses request, excluding model load.
Final ordinary and MTP text equals the baseline ordinary text exactly.

| prompt | ordinary before / after (s) | MTP before / after (s) |
| --- | ---: | ---: |
| Transformer | 6.261 / 5.800 | 5.476 / 4.994 |
| Photosynthesis | 6.206 / 5.645 | 6.386 / 5.829 |
| Python CSV | 6.186 / 5.697 | 4.490 / 4.079 |

Ordinary HTTP latency falls **7.4–9.0%**; MTP latency falls **8.7–9.2%**. These end-to-end timings include
prefill, graph recording and response serialization, and measure one run per request/variant.
The server changes help more than the CLI changes because the previous server always ranked 64
candidates, even for greedy requests. Servers run sequentially on separate loopback ports.

## Validation

All six reference paths pass 132 activation tensors, softcapped logits and ten greedy tokens: batches
1, 2, 4 and 15, CPU relay, and MTP. Worst relative activation error is 0.000982. All 15 Rust tests,
30 tokenizer fixtures, four chat fixtures and 10,000 bidirectional changing P2P payloads pass.
Local Clippy passes with warnings denied. Responses integration checks pass with and without the
assistant, including alternating greedy/stochastic requests, seeded reproducibility, JSON/SSE parity,
reasoning, token limits, namespaces/custom tools, tool search and full-history continuations.

The 1700-token, batch-512 check passes sliding-cache wrap, every verification column, the 64-token
continuation and MTP acceptance/rollback. It reports 63/64 top-64 overlap and relative logit error
0.00176062; serial and chunked prefill share all 64 greedy continuation tokens.

The final selector adjustment independently passes all 132 reference activations, logits and ten
greedy tokens at batch 8. An 80-token, batch-8 check passes every verification column, the 64-token
continuation and MTP rollback, with **zero** logit error and **64/64** top-token overlap.

The main benchmark/server tables and their hashes predate this last selector change. They use
single-token decode, three-draft verification (at most four columns), and large prefill GEMM, whose
kernel selection remains the same. `selector-source.sha256` and `selector-binary.sha256` identify
this final build; `selector-*` files retain its focused checks and the seven-draft chat comparison.
That 128-token chat improves from **14.36 to 20.91 tok/s (+45.6%)**, with identical text, acceptance
(91/259 proposals) and 37 rounds. Verification falls from **211.42 to 136.13 ms/round**. The default
stays at three drafts, which still reaches 29.05 tok/s on this prompt.

## Reproduction and evidence

`remeasure.py` builds the saved baseline shaders with the current benchmark harness, measures baseline
and final kernels sequentially, and restores the current shader sources in a `finally` block. It uses
model assets in the documented incite locations. `baseline-shaders/` contains the exact five original
files. The original full-model executable must be built from `f63fe89` to repeat the HTTP comparison;
binaries and weights are intentionally omitted from this repository.

`server-check.py --baseline PATH --optimized PATH` starts resident servers sequentially, compares
identical 128-token requests and validates the Responses endpoint with ordinary and MTP generation.
`summarize.py` generates `summary.json` from the retained outputs after all checks complete.

An early warmup harness incorrectly reused command buffers freed by `Device::run`, causing a driver
reset in a small-batch measurement. Commit `bbaf641` adds a reusable execution path. All microbenchmark
comparisons retained here were rerun with that fix; the early measurements were discarded. The
`baseline/matvec.json` output predates warmup and is excluded from the bandwidth comparison.

The remaining bottlenecks are Q4 streaming/compute throughput, small-batch verification and repeated
full-history prefill in the HTTP server. M3's ≥180 GB/s and 40–50 tok/s targets, and M5's ≥1.5× typical-chat
speculative gain, remain open. Prefix caching needs separate KV lifetime and sliding-window validation;
no gain from that change was established in this run.
