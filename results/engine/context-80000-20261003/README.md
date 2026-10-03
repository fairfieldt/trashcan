# 80,000-token context experiment — 2026-10-03

Incite, two 6 GiB FirePro D700s, main at `5925623` plus optional benchmark
progress logging. Model, Vulkan backend and all shaders are unchanged from that
validated commit. This is a single fresh 4K/80K comparison, not repeated-run
confidence or a performance improvement to the engine.

## Prefill and decode

| Input tokens | Prefill time | Average prefill | 128 decode steps | Average decode |
|---|---:|---:|---:|---:|
| 4,096 | 34.840 s | 117.57 tok/s | 4.789 s | 26.73 tok/s |
| **80,000** | **2,782.499 s / 46m 22.5s** | **28.75 tok/s** | **10.361 s** | **12.35 tok/s** |

These are the same synthetic benchmark inputs as the earlier context comparison:
token ID 9259 repeated for the entire prefix, batch512, followed by 128 forced
greedy decode forwards without stopping on end-of-turn tokens. Model loading is
outside the timers. The 80K run allocates capacity for 80,129 positions and ends
at position 80,128. Global attention sees the full causal history; sliding layers
retain a 1,024-token attention window. The final prefill chunk has 128 tokens.

Prefill gets slower as the prefix grows. The last full 512-token chunk takes
32.281 s (**15.86 tok/s**); the final 128-token chunk takes 12.726 s. Averaging
the entire pass gives a higher rate because earlier queries have shorter history.

| Prefill interval, positions | Tokens in interval | Time | Rate |
|---|---:|---:|---:|
| 0 -> 16,384 | 16,384 | 208.051 s | 78.75 tok/s |
| 16,384 -> 32,768 | 16,384 | 389.785 s | 42.03 tok/s |
| 32,768 -> 49,152 | 16,384 | 576.097 s | 28.44 tok/s |
| 49,152 -> 65,536 | 16,384 | 765.669 s | 21.40 tok/s |
| 65,536 -> 80,000 | 14,464 | 842.860 s | 17.16 tok/s |

The bulk-prefill path uses `attention.comp`: each query scans its causal keys
with an online softmax. The measured chunk cost grows roughly linearly with
prefix length, so total global-attention prefill cost grows roughly quadratically
in this range. The accepted norm/merge work does not replace this prefill path.

## Decode profile and allocation pressure

The last-step GPU0 trace separates the additional work:

| Component | 4K history | 80K history |
|---|---:|---:|
| Eight global-attention segments combined | 4.009 ms | 36.980 ms |
| Remaining GPU work | 30.034 ms | 30.022 ms |
| Total GPU work | 34.043 ms | 67.002 ms |
| Semaphore gaps | 2.579 ms | 4.560 ms |

Global-attention segments include their projections/norms. They account for
essentially the entire increase in GPU work. GPU1 gives comparable totals
(34.056 -> 66.905 ms); the two GPUs run concurrently and their times must not
be added. These are last-step samples, not the complete run's average duration.

The first 80K decode step takes 153.7 ms; excluding it gives **12.44 tok/s**,
so first-step setup does not explain the whole average. The last 64 steps give
**13.81 tok/s**, while earlier 16-step blocks range roughly 76.5–96.3 ms/step.
The full 128-step rate, **12.35 tok/s**, is the primary result.

The engine's buffer allocation counter is **5,680.11 MiB per GPU**, versus
4,556.09 MiB in the 4K run. Allocated KV storage is 1,252.02 MiB global plus
240 MiB sliding per GPU; the sliding ring includes space for the incoming batch.
Late physical VRAM snapshots peak at 5,750.59 / 5,721.83 MiB, so the workload
fits the cards' 6,144 MiB. This does not establish the largest workable context.

The prefill-to-decode transition also exposes command-allocation pressure:
card1's visible aperture grows from 250.80 to **256.00 MiB**, while GTT allocation
grows from 42.66 to 235.65 MiB. Card0 visible/GTT usage also grows. The sequence
records graphs for batch512, the final batch128 chunk and single-token decode.
This is relevant to the earlier graph-pool cliff, but this run alone does not
isolate its contribution to the transient decode timing. `late-memory.csv`
samples these counters every second from around 64K through process exit;
earlier snapshots are in `memory-snapshots.txt`.

## Checks and reproduction

- All **157 prefill chunks and 128 decode steps** at 80K report finite top-1
  scores and the expected sequential positions. The 4K control also passes.
- The model kernels match `5925623`, which passed the full reference suite.
  No independent **80K reference-logit or retrieval-quality check** was run;
  this experiment establishes benchmark execution and throughput.
- Pre/post kernel reset logs are identical: no new GPU reset.
- Whole-command clocks contain 10,628 samples. Both cards sustain 850/1370 MHz
  during compute; loading/idle samples are also retained. Peak sampled
  temperatures are 97 / 95 C, with no sustained clock reduction.
- `cargo check`, 30 CPU tests on the Mac, the incite release build, formatting
  and diff checks pass.

Tested binary SHA256:
`b73b45e585402b2fd65233c5f6e807e3cc09b3bdc605be6412162efa80acb930`.
`source.sha256` pins all 43 input source files; `source-check.txt` confirms only
`engine/src/main.rs` differs from `5925623`. The new `bench --progress` flag
reports per-chunk/step timing and checks finite top scores; default execution
retains the original prefill path. Both runs use the same binary, exclusive
`~/trashcan-exp/.lock` and scoped 0-us CPU latency request. No service was changed.

[commands.sh](commands.sh) records the commands. [summarize.py](summarize.py)
audits complete positions, finite scores, source/binary identity and reset logs,
and regenerates `summary.json` and the per-step TSVs. Raw logs, last-prefill and
last-decode profiles, clock samples and memory counters are saved alongside.
These are measured workload rates, not hardware upper bounds.
