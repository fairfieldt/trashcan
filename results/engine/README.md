# Engine validation and measurements

These are recorded outputs from incite. Model weights and build binaries are not stored here.

- `validated-followup-20261003/` independently validates the investigation's load-hoisted norm
  and wide attention merge. Repeated tg256 rises from 23.32 to 28.72 tok/s; pp512 is unchanged.
  Source/binary hashes, clocks, reference and greedy parity evidence, and a recovery summary
  of all five unfinished Claude lenses are preserved alongside its README.
- `main-tuning-20261003/` contains the preceding main-branch Q4/attention and resident-prefix experiments,
  repeated benchmarks, matching chat outputs, long-context checks and complete validation evidence.
  Its README distinguishes measured gains, unchanged workloads and rejected candidates.
- `ceiling-audit-20261003/` validates exact tensor costs and separates theoretical bounds from practical
  targets. The roughly 52 tok/s bandwidth/transport scenario is not a demonstrated ceiling.
- `performance-20261003/` contains the preceding performance-branch tuning comparisons, warmed microbenchmarks,
  seven-draft selector measurements, repeated decode runs, reference/cache checks and Responses tests.
  Its README explains which source hashes belong to each measurement phase.
Run instructions and acceptance status are in `docs/implementation.md`.

- `summary.json` collects measured performance and validation status.
- `delivered-benchmark/` contains pp512/tg128, the Q4 microbenchmark, three paired ordinary/MTP
  chats, GPU clocks, hardware details, and the source/binary hashes used for those measurements.
- `final-source.sha256` and `final-binary.sha256` identify the later two-column selector adjustment.
  Its dedicated reference and continuation checks are `two-column-validation.log` and
  `photo-mtp-k1.{log,txt}`. The recorded three-chat timing table predates this adjustment.
- `delivered-validation-suite.log` records five reference paths, tokenizer/chat fixtures, CPU tests,
  and P2P visibility. `validation-b15/` preserves one engine activation dump for independent comparison.
  Its `.f32` files are little-endian f32 arrays, described by `index.tsv`.
- `stream-ring-check.{json,log}` records the final 1700-token streamed-P2P cache-wrap and rollback check.
- `interactive-*` records conversation history and reset behavior; `stochastic-chat*` records seeded
  sampling checks. `comparator-negative.log` records rejection of a nonfinite activation.
- `latency-comparison.json`, `final-profile.tsv` and `qos-profile.tsv` record the earlier PM QoS comparison.
- `relay-benchmark/`, `grouped-*`, `norm-pair-*`, `qos-*`, `stream-bench.json`, and the earlier ring-check
  files retain intermediate measurements behind the transport and kernel decisions.
- `assistant-manifest.json` pins the downloaded assistant's revision, sizes and checksums.
- `source/` contains the licensed Mesa source snapshot used to investigate RADV's global buffer list.
