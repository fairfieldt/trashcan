# Retired TP4 experiments, October 7

Removed unsuccessful performance paths from inference code. Being faster in an isolated component was
insufficient when controlled model runs did not show a benefit.

| Removed | Measurement or reason |
| --- | --- |
| Prepared GDN, its shaders and CLI benchmark | No reliable model-prefill gain. |
| Direct local sum | CPU phase improved; model throughput did not. |
| Reusable exchange progress storage | Extra state and code; no meaningful TP4 throughput gain. |
| 256-query attention slabs | Recovered 143.65 tok/s fell between 145.57 and 139.39 baseline runs. |
| KV-only MTP update | About 0.28 ms saved in a segment; no demonstrated generation gain. |
| Three/four prefill micro-batches | One queue: 125.9/119.5 tok/s versus about 133 with two. |
| Per-segment submit override | 9B: 30.65 versus 31.95 tok/s with submission ahead. |
| No-reduction and VRAM-only inference switches | Deliberately incorrect output for timing estimates. |

The corresponding environment selectors and obsolete metadata are gone. Prefill uses one or two
micro-batches and 64-query tiled attention when selected. Ordered copy-plus-add reductions and the
original exchange bookkeeping are restored. Cold-peer readiness and shutdown handling remain.

Measured improvements remain: output-budget capping, workload-specific adaptive MTP, tiled attention,
GEMM2, event gating, host normalization, memory arenas, balanced draft heads, and two-queue prefill.
Scalar attention, reference kernels, profiling and transport fallbacks still serve correctness checks
or supported configurations. Other rejected Qwen4, projection, compact-scale and reduction-pool
experiments were already absent from the engine.

Evidence: [hardware campaign](../hardware-20261007/README.md),
[KV-only and rejected kernels](../improvements-20261007/README.md),
[prefill scheduling](../tp-prefill-20261006/README.md), and
[decode scheduling](../tp-decode-20261006/README.md).
Historical source archives and raw measurements remain unchanged. The
[offline reproductions](../offline-20261007/README.md) now use archived source rather than deleted runtime files.

## Validation

- 58 CPU tests pass; Clippy succeeds with the same 19 existing warnings.
- Four restored shaders compile and pass SPIR-V validation. The original GDN shader is byte-identical
  to its earlier compiled version.
- The x86 Linux build passes through `BUILD_ONLY=1 scripts/cross-build.sh` in kbuild.
- TP4: 4,300 prompt tokens, 32 decode steps, context 8,192. Checkpoint restoration is exact,
  verification score error is zero, and speculative output is identical. Text matches the prior build.
- Serving: 148 requests, 37 each for plain, fixed MTP, adaptive MTP and uncached plain. Output limits
  1–8, cached follow-ups, edited prefixes, replays and EOS checks pass. Outputs, status and usage match
  the prior build; cache counts match in cached modes and are zero when caching is disabled.
- Both hosts restored their CPU governors, completed GPU teardown and released every lease.

`build.json`, `source.tar.gz` and `source.diff` identify the tested source, based on `b4a6f4c`.
The tested binary SHA-256 is `82473c90238dcb286f4f77d2bd71847bcf44114c9bdd590d1a6a1c34f715f804`.
It is installed as `trashcan-clean` in both experiment workspaces. `commands.json` records the validation
commands; `plans/` retains exact commands, outputs and clocks for both hosts. `audit.json` verifies source,
correctness, clock captures, governors and lease completion. This cleanup makes no new throughput claim.
