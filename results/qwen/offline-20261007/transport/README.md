# TP transport allocation cleanup, local validation, 2026-10-07

Historical experiment: exchange-buffer reuse was subsequently removed because TP4 measurements did not
establish a model-throughput benefit. See the [cleanup record](../../cleanup-20261007/README.md).
The measurements below describe the archived candidate; reproduction uses its source archive.

This work ran only on `pinkerton`, an Apple M3 Ultra (32 logical CPUs), macOS 27 / Darwin 27.0.0, using rustc 1.92.0. No Mac Pro, GPU, CUDA, or remote resources were used. GPU clocks are not applicable.

## Source and change

Baseline: `b0a2e6884d06ac8798df70f7057910ff1ddff3fb`. Only `engine/src/link.rs` is changed by this work. The prior allocator-based exchange remains under `cfg(test)` as the reference peer and paired benchmark baseline. Candidate SHA-256: `7944f936431093f4dcf1862ee1b9b5a7f5eedf2ef84adac0d79f30261dc7d0bb`. The patch is `transport.patch`.

Each connection now owns one reusable progress entry per lane. An exchange updates its offsets in place; the byte partition, lane visitation/write-then-read order, 256 KiB USB4STREAM rounds, and stall behavior are unchanged. No model arithmetic changes. The old code allocated five vectors per nonempty one-round exchange, or `2 + 3 * rounds` for a stream exchange; the new hot path allocates none. For three lanes this removes 640 allocation/free pairs per 128-reduction decode token and 23 pairs per 5 MiB prefill exchange.

## Correctness

`cargo test --release link::tests -- --nocapture`: 2 passed, 2 benchmark tests ignored. The new endpoint exchanges against the unchanged algorithm over real loopback TCP with 1, 3, and 5 lanes. Tests cover empty/unequal lane slices, 20 KiB decode, 60 KiB verification, exact/uneven round boundaries, 5 MiB prefill, changing sizes in the same session, and interleaved control messages. Both sides check all payload bytes. Invalid lengths/alignment, stalled peer timeout, and closed peer errors are checked.

The stream-round mode uses TCP sockets with the stream round limit selected; it validates framing and round compatibility but does not simulate the USB4STREAM receive ring or hardware timing.

## Isolated bookkeeping measurements

`progress_benchmark` uses the production start/advance helpers and the old vector setup, marks each round complete without I/O, and checks completion. Seven alternating old/new samples, 500,000 exchanges per sample. The buffers/offsets are passed through `black_box`. This isolates vector management; it is not an inference or transport throughput benchmark.

| Lanes | Payload | Old ns/exchange | Reused ns/exchange |
|---:|---:|---:|---:|
| 1 | 20480 | 95.51 | 3.21 |
| 1 | 61440 | 97.27 | 3.27 |
| 1 | 5242880 | 1260.05 | 57.68 |
| 3 | 20480 | 108.74 | 8.50 |
| 3 | 61440 | 110.18 | 8.69 |
| 3 | 5242880 | 536.00 | 45.45 |

The three-lane decode bookkeeping difference is about 0.10 microseconds per exchange, or 12.8 microseconds per 128 exchanges on this CPU. Even if that transferred directly, it is a small fraction of the previously measured roughly 35 ms TP4 decode token; no Mac speedup is claimed.

## Full loopback measurements

Seven alternating old/new samples per case, 20 warmups each; 2,000 measured exchanges for decode/verify and 40 for prefill. The peer uses the same implementation within each pair. Both endpoints verify the final payload. Median microseconds per exchange follow; all samples and ranges are retained in the logs and `summary.json`.

| Lanes | Stream rounds | Payload | Old us | Reused us |
|---:|:---:|---:|---:|---:|
| 1 | False | 20480 | 14.411 | 14.580 |
| 1 | False | 61440 | 20.442 | 17.296 |
| 1 | False | 5242880 | 784.779 | 829.176 |
| 1 | True | 20480 | 12.075 | 11.116 |
| 1 | True | 61440 | 18.103 | 17.741 |
| 1 | True | 5242880 | 1142.246 | 961.826 |
| 3 | False | 20480 | 28.758 | 30.299 |
| 3 | False | 61440 | 35.242 | 34.215 |
| 3 | False | 5242880 | 877.201 | 839.643 |
| 3 | True | 20480 | 32.745 | 28.433 |
| 3 | True | 61440 | 32.452 | 33.379 |
| 3 | True | 5242880 | 961.393 | 912.459 |

TCP scheduling/syscall variation is larger than the isolated bookkeeping saving, and paired cases move in both directions. These results establish no consistent loopback throughput gain and no regression bound for the Mac Pros. The retained change is the measured allocator/bookkeeping cleanup. Recheck the three USB4STREAM lanes and both machines under leases and clocks after recovery before claiming any prefill or decode throughput improvement.

## Reproduction

```sh
bash results/qwen/offline-20261007/test-candidate.sh link::tests -- --nocapture
bash results/qwen/offline-20261007/test-candidate.sh link::tests::progress_benchmark -- --ignored --nocapture
bash results/qwen/offline-20261007/test-candidate.sh link::tests::exchange_benchmark -- --ignored --nocapture
```

`loopback.log` is the first loopback run; `loopback-final.log` measures the final source after extracting the shared production bookkeeping helpers. `progress.log`, `tests.log`, and `clippy.log` preserve the other command outputs. Concurrent team work can change the total number of filtered tests and unrelated diagnostics.
