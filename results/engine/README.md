# Engine validation and measurements

These are recorded outputs from incite. Model weights and build binaries are not stored here.
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
