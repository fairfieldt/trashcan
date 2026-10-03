# Bounded tiled long-context prefill — 2026-10-03

This round follows MTP commit `91c1672` and targets incite's dual D700s.
The candidate changes bulk prefill attention. Matrix, decode and MTP
verification kernels retain that measured source.

## Measured results

Two repetitions per phase, with reversed phase order for repetition two:

| Prompt tokens | Baseline prefill, tok/s | Tiled prefill, tok/s | Gain | Baseline decode, tok/s | Tiled decode, tok/s |
|---|---:|---:|---:|---:|---:|
| 512 | 160.60 | 174.40 | 8.6% | 28.72 | 28.26 |
| 4096 | 117.60 | 168.89 | 43.6% | 26.16 | 26.32 |

Rates pool tokens divided by elapsed time. Every chunk/decode top-token ID
matches its paired control; maximum top-score differences are 0.002264 at512
and 0.000965 at4096. Decode varies -1.6%/+0.6%; no decode gain is established.
These are measured workload rates, not hardware upper bounds.

The user stopped the 80K run after **73,216 tokens / 143 full chunks**. At that
position, the candidate takes **667.206 s (11m7s), 109.74 tok/s**, versus
**2365.500 s (39m25.5s), 30.95 tok/s** in the saved baseline: **3.55× faster**.
The last completed chunk falls from **29.714 to 5.584 s** (17.23 ->91.69 tok/s).
All 143 checkpoint top-token IDs match; maximum top-score difference is 0.000801.
This is a matched-prefix comparison, **not a completed 80K result**. The new
80K decode measurement was not reached. The saved baseline's full 80K numbers
remain 28.75 prefill /12.35 decode tok/s.

Eight primary attention cases cover empty/37/4096/80000 prefixes, 64/127/512
queries, both global512/one-KV-head and sliding256/four-KV-head shapes. Each has
both GPUs, 300-ms warmup and two timed command-buffer replays. At 80K/batch512,
global attention averages **3575.7 ->359.3 ms (9.95×)**. Full-window sliding
attention averages **24.1 ->4.08 ms (5.92×)**. These are isolated attention
operations, rather than whole-layer or application rates.

## Implementation

Bulk batches of at least 64 queries use three GPU passes: tiled Q/K score
multiplication, causal/window softmax, and tiled probability/V multiplication.
Both matrix passes reuse data in LDS across 64 query-head rows and 64 output
rows, stepping 32 dimensions/keys at a time. Every producer and consumer phase
has a workgroup barrier. Normal device dependency barriers remain between
passes and before scratch is reused.

The engine processes at most **64 tokens per score slab**, retaining its usual
512-token FFN/projection GEMMs. Scratch scales as 64 ×8 local query heads ×
rounded context ×f32, rather than batch512 ×8 ×context. At capacity80129 this
adds **156.7 MiB/GPU**, versus about 1.22 GiB for the recovered full-matrix
prototype. Sliding layers use a bounded 1024+31-key span.

Query-block offsets are push constants; full Q/output buffer addresses remain
fixed. The graph stays reusable at every position and keeps the four-entry
cache. Partial blocks mask invalid queries, future keys and window-expired
keys. The ring reserves window+batch slots. Padded V loads become zero before
multiplication, avoiding 0×NaN. Small batches and decode keep their existing
attention path. `TRASHCAN_PREFILL_ATTENTION=scalar` provides the old bulk path
as a control within the same source.

Global score dispatches cover capacity and uniformly reject future key tiles.
An indirect dispatch driven by the current prefix could remove that launch
work, especially early in an 80K run; it is an unmeasured follow-up. At 80K,
logical score/value math divided by measured kernel time is about 1.87 TFLOP/s
per GPU. This is an effective workload rate, compared with the D700's
3.48-TFLOP/s theoretical fp32 ceiling and the previously measured 2.2-TFLOP/s
best isolated FFN GEMM. Those bounds do not establish an application target.

## Correctness and limits

The micro compares every output element with the scalar GPU path (absolute
tolerance0.0003), checks 18 selected elements against an independent f64 stable
softmax (tolerance0.0002), and requires every output finite. Inputs use
xorshift values; K/V are f16. Unwritten cache slots, initial scores and outputs
contain NaNs to expose masking, padding or missing-store mistakes. Partial
127-query blocks and nonaligned sliding-ring positions exercise tails and wrap.
Additional smoke cases cover nine queries and a 128-query ring wrap.

Completed 1700-token batch64/128/512 comparisons preserve all **64 greedy
continuation tokens** against serial execution. Relative logit errors are
0.002566/0.002566/0.002559; top64 overlap is63/64 in each case. Thirty CPU tests
also pass. The tiled reduction order changes, so output bits are not promised
identical to scalar attention.

The final reference/assistant rollback, 250-token assistant boundary and
4096-token serial checks were **queued and cancelled at the user's request**.
No final candidate acceptance is claimed. The preceding MTP commit's complete
acceptance is in its own report. The saved 15-token activation fixture is
below the tiled cutover and does not validate the new path.

The interrupted 80K repeated-token benchmark has finite per-chunk top scores
and matches the saved IDs through 73,216 tokens. It is not an independent 80K
reference or retrieval-quality evaluation. No new long-context decode steps ran.

## Clocks, memory and reproduction

Slow intervals near40K and65K include one card at450 MHz; both subsequently
return to850/1370 MHz. VRAM/GTT counters stay steady through those intervals,
and no new kernel fault/reset is logged. Their cause is unresolved. The reported
matched-prefix rate includes both. Temperatures peak at100/97°C; the longest
continuous both-card850/1370 streak is978 samples, about4m20s at the observed
sampling cadence. Loading and idle intervals are also included in clock files.

Peak physical VRAM is **5889.33/5873.47 MiB of6144** per card. GTT peaks during
loading/initial recording at66.73/605.15 MiB, then stays near66.7/42.7 MiB during
long prefill. The new decode graph's memory footprint was not measured.

All GPU work holds `~/trashcan-exp/.lock`. Timed runs use
`sudo -n python3 scripts/latency-run.py` for a scoped 0-us CPU latency request.
`long-memory.csv` samples physical/visible/GTT usage; journals audit resets.
Source sync uses `rsync -ac`, followed by touching shaders/Rust before builds.

`run-short.py`, `run-long.sh`, `run-acceptance.sh` and `commands.sh` record the
workflow. All 47 candidate source files match local source; all 43 baseline
files match `91c1672`. The saved 80K baseline is from `013e343`;
`historical-baseline-kernel-check.txt` verifies that its ordinary batch512/128
and decode shaders are identical to `91c1672`. This long-prefix comparison is
one historical baseline/partial-candidate pair, with fresh reversed-order short
controls. `summarize.py` rebuilds `summary.json` and requires the completed
source, micro, short, continuation and interrupted-run evidence.

| Phase | Binary SHA256 |
|---|---|
| Baseline,91c1672 | `7a636984471a4a4dabf005a177a700104d758d2ea26abe440f098450c5287efb` |
| Candidate | `ca0c38baf63b96a445f5eaee21418dfa6b40498148162e2528057e30d39843e2` |

Pinned binaries remain on incite in `bin/` and are not committed.
`user-stop.txt` records the request, and `long-end.txt` records actual exit.
The queued acceptance job was cancelled before GPU work. Benchmark/monitor
processes exited and released the lock. `ctx80000.out` is empty because CLI
final totals are emitted only on completion; the partial trace is retained
in `ctx80000.log`. The original investigation directory remains untouched.
