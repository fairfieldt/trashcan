# Independently validated norm and attention-merge follow-up — 2026-10-03

This continues the investigation at `82e559f` on main. All five unfinished
Claude lens transcripts and their raw results were recovered and reviewed;
[recovered-lenses.md](recovered-lenses.md) records their identities, evidence,
contaminated measurements and unaccepted leads. The pre-existing untracked
investigation directory was preserved.

## Measured application improvement

Incite, two FirePro D700s, the existing kernel/Mesa installation. Timed commands
use a scoped 0-us CPU latency request. Both GPUs sustain 850/1370 MHz during
compute; complete clock logs include loading and idle intervals.

| Phase | pp512, pooled two runs | tg256 after pp512, pooled two runs | Decode gain vs baseline |
|---|---:|---:|---:|
| Fresh `82e559f` baseline | 160.828 tok/s | 23.323 tok/s | — |
| Scalar load-hoisted fused norm only | 160.883 tok/s | 26.062 tok/s | +11.74% |
| Norm plus wide attention merge | 160.761 tok/s | **28.723 tok/s** | **+23.16%** |

Rates pool tokens divided by total measured time, rather than averaging rates.
The individual final tg256 runs are 28.830 and 28.617 tok/s. Prefill is unchanged
within variation. Single paired pp4096/tg128 runs give **21.149 -> 26.389 tok/s
(+24.78%)** decode and 117.469 -> 117.687 tok/s prefill. That long-context
percentage has one pair, not repeated-run confidence.

The last-token traces explain the change. Mean per-GPU kernel work over the two
saved traces is about 39.86 -> 35.21 -> 31.62 ms. The two GPUs run concurrently;
do not add their times. Norm removes about 4.65 ms, merge another 3.57 ms.
Semaphore gaps vary between ~2.6 and ~4.4 ms in these final-token samples.
These are timestamp samples, not the complete run's average token duration.
The full application measurements above are authoritative for throughput.

## Accepted implementation

- `norm_fused.comp` specializes only the fully fused, weighted, one-head
  D=3840 path (flags=7). Independent inputs, peer partials, residuals and weights
  are loaded before the dependent arithmetic. It retains the original
  15-element per-lane FMA chains, subgroup reduction and four-wave summation
  order, and keeps all three outputs. Generic norm handles the other paths,
  including the assistant and CPU-reduction path.
- `attention_merge.comp` loads partition headers in parallel into shared
  memory and assigns one output element to each of 512 threads. Maximum,
  denominator and FMA accumulation keep the original partition order. Its
  supported model head dimensions remain 256/512, with 8/16/32/64/128 partitions.
- `norm-bench` checks mixed-scale randomized activations against the original
  generic GPU norm and an f64 CPU reference. It restores residual inputs before
  every replay, excludes that copy from kernel timestamps, warms each shader
  and collects two sets of 32 individual dispatch timings. Buffers remain alive
  to avoid the investigation harness's recycled-handle descriptor bug.
- `attention-bench` now also compares **every merge output bit** against a
  retained serial reference, in addition to the existing CPU softmax check.

The faster vector/960-thread norms from the investigation were not ported: they
change reduction order, and their synthetic-input equality did not establish
real-activation equality. The accepted scalar norm is independently measured
and checked here.

## Micro measurements and selector sweep

`norm-micro.jsonl`: 64 passing records, two GPUs, four random seeds, one/four
columns, two repetitions per shader. All three vectors are bit-identical to the
generic norm. One-column mean-of-median timings are **65.80 -> 19.17 us**;
four-column timings are **65.90 -> 19.81 us**. Worst CPU scaled error is below
1.2e-6. The input/weight buffers are warm; application gains are measured above.

The attention sweep alternates pinned baseline/final binaries for each case,
two repetitions, both GPUs, lengths 1/256/768/1536/8192 and every supported
partition count: 400 passing records, including 200 final all-element bit
comparisons. At 768 keys/32 partitions, total partial+merge time is about
**164.2 -> 102.1 us sliding** and **226.5 -> 100.7 us global**. The partial
kernel is unchanged, so the difference isolates the merge implementation.

The sweep confirms the current long global 64-partition choice and supports
further short-prefix tuning: at 256 keys, 32 partitions now beat 16 (55.1 vs
67.8 us sliding; 56.2 vs 70.6 us global). At 768 keys, global 64 is only ~1.8 us
ahead of 32. The production partition thresholds are unchanged in this commit;
changing their boundaries needs a finer sweep and its own application comparison.
The sliding 8192-key micro is synthetic; real sliding layers cap visibility at 1024.

## Correctness and acceptance

The final binary passes the complete `TRASHCAN_LONG_CHECK=1 scripts/test-engine.sh`
suite: 30 CPU tests, 30 tokenizer cases, four chat fixtures, 10,000 changing
bidirectional P2P payloads, and all six reference paths (132 tensors, logits and
ten greedy tokens each). Fresh baseline/norm/final b1 runs compare all 133
activation/logit files and 32 greedy tokens byte for byte. The ordinary
128-token transformer chat is also identical (25.62 -> 30.85 tok/s, one pair).

With the assistant enabled, the 1700-token batch512 test crosses the sliding
ring wrap and passes verification/rollback: top-64 overlap 64, relative logit
error 0.0012545, and all 64 serial/chunked continuation tokens identical. The
250-token batch128 partition-boundary test also passes, with relative error
0.0000983 and an identical 64-token continuation. `acceptance-binary.sha256`
matches the timed final binary. `reset-audit.txt` shows no new GPU reset; the
last recorded reset is the preceding investigation's 10:27:55 event.

Ordinary and MTP output bytes match across both binaries for all three short
prompts below. MTP uses the unchanged three-proposal policy and preserves each
prompt's accepted/proposed counts (82/138, 72/165, 90/111). These are single
paired chat runs, with 128 generated tokens each:

| Prompt | Ordinary baseline -> final, tok/s | MTP baseline -> final, tok/s |
|---|---:|---:|
| Transformer | 25.62 -> 30.85 | 29.34 -> 31.75 |
| Photosynthesis | 25.50 -> 30.28 | 24.70 -> 26.44 |
| CSV function | 25.59 -> 30.88 | 37.15 -> 39.09 |

Across these prompts, the MTP/ordinary speedup pooled by time shrinks from
**1.156x to 1.031x** as ordinary decode becomes faster. Four-column verification
still costs roughly 76 ms per round; the verification leads need their own
validated kernels and live policy measurements. M5 remains open. The separate
1788-token water-cycle prompt also has identical ordinary output, with decode
21.73 -> 27.59 tok/s and prefill 134.20 -> 133.94 tok/s (one pair).

## Source and reproduction

| Phase | Binary SHA256 |
|---|---|
| Baseline | `5b62aa4082717a95eae881219fc258a90879ea9b35b546c149153af3e6bb0b6e` |
| Norm only | `80cf62f0a32f5115f290c5abc9730e2c730b5561e12afe856a5d464a0db2ebd1` |
| Norm + merge / final | `2682f29e7969210a14d416a05aea7267f670fde0ddcc7a4ed532b841f12a2a3f` |

The baseline manifest's 40 files all match committed `82e559f`, checked against
Git content rather than mtimes. Each phase has `*-source.sha256`,
`*-binary.sha256` and a build log. Sources were copied with `rsync -ac` and
explicitly touched before remote builds to avoid the earlier stale-build hazard.
Pinned binaries remain on incite under this directory's `bin/`; binaries are
excluded from the committed result set. No experimental service or GPU schedule
was changed. Builds and GPU runs held the existing `~/trashcan-exp/.lock`.
The five norm-only changed files relative to `82e559f` are preserved under
`tested-source/norm/` with their project-relative paths; `norm-source-check.txt`
confirms they reproduce every hash in that intermediate phase's manifest.

[commands.sh](commands.sh) records the commands. [summarize.py](summarize.py)
rebuilds `summary.json` and checks that completed acceptance and timing evidence
agree with the final binary. Raw JSON, profiles, clocks, chat outputs and logs
are alongside this report, including full fresh-baseline/norm/final b1 activation
dumps in `{baseline,norm,merge}-b1/`, their SHA256 manifests and exact parity
results. Hardware weight-streaming bounds are
unchanged; 28.723 tok/s is a measured workload rate, not a hardware ceiling.
