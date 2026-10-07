# Tensor-parallel decode and prefill in one build (2026-10-06/07)

The decode branch (`worktree-agent-aa34a3d7cd8816359`, [../tp-decode-20261006](../tp-decode-20261006/)) and the
prefill branch (`worktree-agent-ad218b8d68d6edc6c`, [../tp-prefill-20261006](../tp-prefill-20261006/)) both start at
933a991. This branch merges them, splits `qwen_matf32` into two pipelines (the merge made prefill 4% slower), fixes
`qwen-tp-node` ending its session after 30 s without a request, and corrects the first TP4 record
([../tp-20261006](../tp-20261006/)).

| Qwen3.8-27B, TP4 over incite + invite, USB4STREAM × 3 | merged | decode branch | prefill branch |
|---|---|---|---|
| pp512 tok/s | **150.41, 151.26** | 90.8, 91.0 | 148.5, 148.7 |
| tg128 tok/s | **28.44, 28.24** | 28.48, 28.63 | 16.2-17.0 (node without the latency request) |
| tg128 with MTP | **39.48, 39.63** (72.1% accepted) | 39.51, 39.63 | 26.35, 26.15 (node without the latency request) |
| pp2048 tok/s | **134.12, 133.64** | | 140.5 (node without the latency request) |

| Qwen3.5-9B, TP2 on incite | merged | decode branch | prefill branch |
|---|---|---|---|
| pp512 tok/s | **269.00, 268.95** | 208.5, 206.8 | 264.5, 263.2 |
| tg128 tok/s | **45.87, 45.90** | 45.95, 45.96 | 32.17, 32.03 |
| tg128 with MTP | **66.11, 66.09** (70.8% accepted) | 66.15, 66.13 | |
| pp2048 tok/s | **251.10, 250.46** | | 248.6 |

Merged: `qwen-bench` with the performance governor and a 0 µs CPU latency request (`scripts/latency-run.py`) on both
Macs, the node on invite included; branch columns are the numbers in those branches' READMEs. The generated text of
every run equals the branches' (decode and MTP), and the prefill's candidates (`pp_top`) are bit for bit the prefill
branch's. GPU clocks 850/1370 MHz while computing in every run (`incite/clocks`, `invite/clocks`; the other samples
are model loading and idle time); incite's cards reached 75 C, invite's 87 C.

Regressions against the branches' numbers, explained by back-to-back runs of the branches' own binaries:

- 27B prefill: it is paced by the slower Mac's host work and varies by session for the same binary, 142-151 tok/s
  (the cross-Mac wait is 1.0-1.5 s per 512-token chunk). Alternating with the prefill branch's binary in one session:
  prefill branch 142.1, 143.7, merged 143.9, 146.3 (`abq-*`); in another, prefill branch 142.5, 143.1 and pp2048 131.7
  against merged 142.5, 141.5 and 132.0, 130.8 (`abp-*`, `bench-*`). The prefill branch measured its 148.5 with the
  node lacking the latency request; without it the merged binary gives 149.9, 150.9, 150.7, 150.0 (`abn-*`).
- 27B MTP: the first merged pair was 38.14, 37.63 (`bench-27b-tp4-mtp-*`); alternating with the decode branch's
  binary, merged 38.90, 39.33 against 38.98, 39.29, 39.15 (`abm-*`, `abd-*`), then 39.48, 39.63 in the final set.
- 9B MTP: with one `qwen_matf32` pipeline the merged build verified 1.7% slower (65.2 against 66.1 tok/s); fixed
  (below).

## Tested source and commands

Commits on `worktree-agent-a6fd691eeb422b40a`: 8bff390 (merge), 264fafd (`qwen_matf32` split), 2e5b7b6 (idle fix),
3d8a079 (the correction), then these results. The runs used three cross-builds (`HOSTS="incite invite"
scripts/cross-build.sh trashcan-exp/merged/trashcan`):

| binary | source | runs |
|---|---|---|
| f71f25cf | the merge with the idle fix, one `qwen_matf32` pipeline | `smoke-*`, `gemma-test.log`, `val-9b-pipe`, `val-9b-tp2`, the 9B checks without `-final`, `bench-9b-tp2-*`, `ab-9b-merged-*`, `ab2-9b-merged-*`, `prof-9b-merged-*` |
| c0375f6c | 2e5b7b6 except two comments | `ab4-9b-merged-*`, every merged 27B run, `test-server-*`, `final-*`, `prof-9b-final` |
| f84d0889 | 2e5b7b6 | `gemma-test-final.log`, the `-final` checks and validates |

f84d0889 and c0375f6c have the same `.text`, `.rodata` and `.data.rel.ro`: the comment edits changed only debug
information. `trashcan-decode` and `trashcan-prefill` (`*-decode-*`, `*-prefill-*` runs) are cross-builds of b4f051d
and 35abb48. `trashcan-x` (`*-x-*` runs) is the merge with temporary A/B switches (`TRASHCAN_X_*`, see
`scripts/ab9b*.sh`), not committed; its second and third builds have the `qwen_matf32` split.

```sh
# incite and invite: ~/trashcan-exp/merged, inside gpuq leases; scripts/ holds the wrappers used
./gov.sh performance                                   # both Macs, benchmarks only; schedutil restored after
./node.sh node-27b-final MODEL27 --mtp MTP27           # invite: qwen-tp-node under latency-run.py, with clocks
./tp4check.sh                                          # incite: 27B validate and qwen-check runs
./tp4bench.sh final                                    # incite: 27B benchmarks, two runs each
./final9b.sh                                           # incite: 9B benchmarks and the profiled run
./final-checks-incite.sh                               # incite: Gemma checks, 9B validates and checks
scripts/ab27b.sh LEASE LEASE NAME BIN lat|nolat ARGS   # Mac: 27B A/B, the node restarted with the driver's binary
./run.sh NAME CMD...                                   # one run: clocks, latency request, ENV switches
```

`commands.log` on each Mac lists every command with its time.

## The merge

- **Three ways to run a forward.** Decode, verification and draft steps (up to 8 columns) are one event-gated
  submission per rank with the residual kept on the host (decode branch). A prefill batch of at least 128 tokens is
  two micro-batches on two queues (prefill branch). Batches in between, such as the 88-token tail of a 600-token
  prompt, keep one submission per segment released by a timeline semaphore. A forward is gated only as a single
  micro-batch, so a short last micro-batch (129 tokens: 128 + 1) is not; graph keys carry the micro-batch's params
  slot, first row, whether it ends with the head and whether it is gated. The timelines are per micro-batch; a
  gated forward signals the first one when its submission ends.
- **Kernels.** The decode branch's TP choices (small projections first, `qwen_add_norm_wide`, `qwen_topk`, `matvecq4`)
  apply to the micro-batches too. Reverting the projection order or the norm for prefill made it slower (9B pp512
  255.7, 254.7 and 256.5, 256.9 against 259.1, 260.2; `ab-9b-x-*`). The device arena does not change prefill (260.5,
  260.1 without it).
- **`qwen_matf32`.** The decode branch made it take 16 loads per lane for up to four columns, in the same shader as
  the plain loop. Compiled in, that path (80 more values per lane) made the prefill 4% slower although prefill batches
  never take it: 9B pp512 259.2, 259.4 against 269.6, 269.5 with the paths in two pipelines (`ab2-9b-x-*`). It is now
  a specialization constant: `qwen_matf32_u16` for decode and verification, `qwen_matf32` (U = 1) otherwise; the sums
  are unchanged. With the single pipeline, the merged build's MTP verification batches also ran 1.7% slower than the
  decode branch's (one pipeline too, but one queue and no micro-batches): 31.8 against 31.3 ms per 9B round, in the
  GPU segments themselves (`profiles/9b-verify-*.txt`). With the split they match (`ab4-*`: 66.11, 66.08 against
  66.14, 66.16 tok/s, also with both micro-batches on one queue). The cause was not found.
- **One profiler.** Both branches used `TRASHCAN_TP_PROFILE`: the decode branch for a JSON line per step with segment
  (and dispatch) timestamps aligned to host time, the prefill branch for GPU milliseconds per kernel in
  `qwen-bench`'s `pp_profile`. Now `TRASHCAN_TP_PROFILE=FILE` stamps every tensor-parallel graph and writes one line
  per local rank and graph (micro-batch) for each forward and draft step, each with the host times of that graph's
  segments; `TRASHCAN_TP_PROFILE_DISPATCH=1` adds every dispatch and copy, a matrix labelled with its rows and row
  length (`gemm2 6144x4096`). `tools/tp_profile.py FILE --n N` reads any step width (256 for a 512-token chunk's
  micro-batches, `profiles/9b-final-prefill-microbatches.txt`), and `qwen-bench` sums the prefill's dispatches per
  kernel in `pp_profile` (`incite/prof-9b-final.json`, raw lines in `profiles/prof-9b-final.jsonl.gz`).

## `qwen-tp-node` and idle time

The node read every request with the lanes' 30 s stall timeout, and `follow()` took any error for a disconnect: a
node without a request for 30 s ended its session and waited for a new handshake, while the driver (a server between
requests) kept its lanes and failed on the next request. Now the wait for the next request (`recv_message_idle`) has
no deadline: it polls lane 0 without pausing for 50 ms, then every millisecond. The rest of the message, the messages
inside an operation and every exchange keep the stall check (2 s inside a gated forward). End of file or a reset
ends the session with the reason in the node's log, as does a lane handshake from a new driver. The driver needs no
change: it does not touch the lanes between requests.

| check (27B TP4 with MTP, `trashcan serve --tp-stream ... --mtp ...`, node with `--mtp`) | result |
|---|---|
| `scripts/test-server.py` | pass (`incite/test-server-1.txt`) |
| no requests for 2 min 51 s | node CPU 0.85 s over 158 s (0.5% of a core), server 0; the session stays open |
| `scripts/test-server.py` again | pass (`incite/test-server-2-after-idle.txt`) |
| SIGTERM to the server | the node logs `session ended: the other machine closed the lanes: unexpected end of file` within seconds and waits for the next driver (`invite/node-27b-serve.log`) |

## The latency request on the node

| 27B TP4, merged binary | pp512 | tg128 | decode exchange per token |
|---|---|---|---|
| 0 µs request on both Macs (final set) | 150.41, 151.26 | 28.44, 28.24 | 4.2, 4.1 ms |
| node without the request (`abn-*`) | 150.71, 150.04 | 25.99, 26.12 | 7.6, 7.3 ms |

The first TP4 record ([../tp-20261006](../tp-20261006/), 16.6 tok/s and 26.0 with MTP) was measured with the node
lacking the request, which made each cross-Mac exchange cost about 94 µs instead of about 40; with the request on both
Macs that binary gives 18.7 and 28.6 ([../tp-decode-20261006](../tp-decode-20261006/)). A correction note is in that
README and in `docs/findings.md`.

## Correctness

| check | result |
|---|---|
| `cargo test --release` (Mac) | 41 passed |
| `cargo clippy` | the 19 warnings of 933a991, none in the files touched |
| Gemma, `scripts/test-engine.sh`'s checks from this workspace (`scripts/gemma-test.sh`) | tokenizer 30 and chat 4 fixtures, P2P 10,000 rounds, validate at batch 1, 2, 4, 15, CPU relay and MTP all PASS, worst 9.95e-4 (`incite/gemma-test-final.log`) |
| 9B pipeline over both GPUs (`--gpus 0,1`) vs our CPU reference / llama.cpp | logits 2.36e-4 / 2.14e-2, greedy 10/10; bit for bit the bring-up dump (`val-9b-pipe-final-vs-*.txt`; the 9.38e-1 "worst" is `Qcur`, stored pre-scaled by 1/16 as before) |
| 9B TP2 `qwen-validate` vs our CPU reference / llama.cpp | worst 3.74e-4 over 474 tensors, logits 2.35e-4 / 2.14e-2, greedy 10/10; bit for bit the earlier TP2 dumps |
| 27B TP4 `qwen-validate` vs our CPU reference / llama.cpp | worst 4.11e-4 over 946 tensors, logits 4.11e-4 / 3.28e-2, greedy 10/10; bit for bit the earlier TP4 dump |
| `qwen-check --verify 3 --tokens 600 --steps 128` | 9B TP2 and 27B TP4 pass: restore exact, 85 rounds, 0 column mismatches, score error 0 |
| `qwen-check --mtp --tokens 600 --steps 256` | pass, speculative output identical; 144 of 222 drafts accepted (9B), 148 of 216 (27B), as in both branches |
| 700-token prompt (last chunk 188 tokens: micro-batches of 128 and 60), with and without MTP | pass on both models; MTP 149 of 212 (9B), 145 of 220 (27B) |
| 4300-token prompt (8 chunks of 512, then 204: 128 + 76), verify 3 | 9B (128 steps) and 27B (64 steps, 42 rounds) pass |
| generated text | every check and benchmark equals both branches' |

## Files

- `incite/`, `invite/`: `NAME.json` (stdout) and `NAME.err` of every run, clock logs, node and server logs,
  `commands.log`. Prefixes: `final-` (benchmarks), `check*`, `val-*`, `test-server-*`, `ab*` (A/B runs; `abd`/`abp`
  are the decode and prefill branches' binaries, `abm`/`abq` the merged one alternating with them, `abn` the node
  without the latency request), `bench-` (the first 27B set, of c0375f6c, and the 9B set of f71f25cf), `prof-`
  (profiled runs).
- `val-*-vs-*.txt`: `tools/compare_activations.py` against `~/dev/claude-work/trashcan-20261006/ref/` and the earlier
  TP dumps; the dumps themselves are in `~/dev/claude-work/trashcan-20261006/merged/incite/` on the Mac.
- `profiles/`: `tools/tp_profile.py` summaries and the final profiled run's lines.
- `scripts/`: the run wrappers and A/B scripts.
