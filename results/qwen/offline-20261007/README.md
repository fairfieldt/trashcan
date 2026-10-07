# Qwen3.8-27B TP4 offline optimization work, October 7

This preserves the initial offline investigation. The subsequent [hardware record](../hardware-20261007/README.md) contains the target-host measurements, fan deployment, decisions, and final-source validation; pending checks below describe the original plan.

Starting source: `b0a2e68` on main. Three sub-agents investigated prefill, decode/MTP,
and transport; the coordinating agent tested host reductions and integrated the work.
The Mac Pros were unavailable, and no connection, GPU lease, deployment, or benchmark
was attempted on incite or invite. All new measurements below are CPU work on the
Apple M3 Ultra workstation. They are **not D700 or TP4 throughput measurements**.

## Prior evidence used

The merged TP4 record is approximately 150 tok/s prefill at 512 tokens, 28.4 tok/s
decode, and 39.5 with MTP from a cool start. Existing tiled attention improves long
prefills; the later 8K paired check measured 110.34 -> 134.43 tok/s. The 256-query
slab has measured standalone kernel gains but its distributed comparison was
interrupted by the breaker trip. These experiments were not repeated here.

Invite's warm MTP run fell to approximately 30 tok/s with clock reductions; cooling
restored 39.52 in a 2048-token repeat. Temperature/clock effects must be separated
from software effects. See [the recovered verification record](../verification-20261007/resumed/README.md).
Persistent reduction pools, extra micro-batches, compact Q4 scales, and earlier
projection variants already lacked measured wins and were not reintroduced.

## Changes

| Change | Selection | Evidence and remaining limit |
| --- | --- | --- |
| Reuse per-lane exchange progress buffers | Default | Removes five allocations per small exchange; interoperates with the previous wire implementation in local multi-lane tests. Local TCP throughput is noisy. |
| Cap speculative work to the remaining output budget | Default | Reserves the final sampled token without evaluating it; CPU tests cover every supported draft depth and acceptance boundary. Needs model-level greedy/rollback checks. |
| Prepare DeltaNet Q/K normalization and gates once | `TRASHCAN_QWEN_GDN=prepared` on both nodes | CPU math/ownership tests and compiled shader checks; no GPU speed or equivalence claim. Only used above eight columns. |
| Direct two-local-rank sum | `TRASHCAN_TP_SUM_DIRECT=1` on both nodes | One output pass instead of copy then add; bitwise ordered-sum tests and local CPU measurements. Mapped D700 memory remains unmeasured. |
| Choose MTP or serial decoding from completed-round costs | `TRASHCAN_QWEN_ADAPTIVE_MTP=1` on driver | Opt-in controller with warmup, periodic probes, and hysteresis. Offline replay is a cost-model illustration, not a throughput result. |

The GPU and adaptive performance candidates remain opt-in. Benchmark records include
their flags; both nodes report per-process prefill/reduction flags in host metadata.
Existing scalar/tiled attention, query slab, and KV-only defaults are unchanged.

Later hardware follow-up narrowed `TRASHCAN_TP_SUM_DIRECT=1` in the runtime to one column with exactly
two local ranks, still off by default. Paired full-clock TP4 measurements reduced the decode CPU sum
phase by 27–31% (0.107–0.125 ms per token), without establishing a model-throughput gain. Prefill showed
no gain. This offline record and its all-shape `host-bench.rs` reproduction describe the original broader
candidate; the general helper remains available to that benchmark. See the
[hardware record](../hardware-20261007/README.md) for the measured source and subsequent validation.

Short-budget sampling can consume a different random-number sequence because unused
drafts are omitted. Adaptive mode also makes its choices from timing, so identical
seeds need not produce identical sampled outputs across runs with different timing.
The acceptance algorithm is unchanged; hardware greedy equivalence remains to be checked.

## Local host measurements

`host-bench.rs` compares the previous copy-plus-add with the exact new sum helper,
using ordinary RAM, one thread, 5120 hidden dimensions, nine alternating-order samples.
The medians from `host-bench-final.csv` are:

| Rows | Previous sum, us | Direct sum, us | Less time |
| --- | ---: | ---: | ---: |
| 1 (decode) | 0.505 | 0.284 | 43.9% |
| 3 (verification) | 2.778 | 0.878 | 68.4% |
| 256 (prefill microbatch) | 200.177 | 135.039 | 32.5% |

These are tiny CPU components. They do not include PCIe, GPU work, network exchange,
or prefill's six host threads. They cannot be converted into model tok/s gains.

An attempted fused remote-sum/residual/RMS pass is **not integrated**. Early local
builds were slower; timings changed materially with benchmark code organization.
`rejected-host-fusion.rs` and the exploratory CSVs preserve that experiment, without
promoting a compiler-sensitive result into the runtime.

Reproduction:

```sh
rustc --edition=2021 -O results/qwen/offline-20261007/host-bench.rs -o target/tp-host-bench
target/tp-host-bench
PATH=/opt/homebrew/bin:$PATH cargo test --release
PATH=/opt/homebrew/bin:$PATH cargo clippy --release --tests
BUILD_ONLY=1 scripts/cross-build.sh
```

See the [transport record](transport/README.md) for protocol tests, raw measurements,
and the distinction between isolated allocation bookkeeping and TCP timings.
The [prefill](prefill/README.md) and [decode](decode/README.md) reports provide their
separate numerical checks and replay evidence.

## Integrated validation and source

- macOS release CPU tests: **53 passed**, two manual transport benchmarks ignored.
- Linux aarch64 CPU tests in the local kbuild VM: **53 passed**, the same two ignored.
- Clippy including tests: succeeds with the same 19 existing warnings; none added.
- All shaders compiled during the macOS and x86-64 Linux builds. The three GDN
  shaders also passed explicit SPIR-V validation. The default GDN shader remains
  byte-identical to the starting source with the same local compiler.
- `BUILD_ONLY=1 scripts/cross-build.sh` produced `target/trashcan-incite`, SHA-256
  `0bb31aab9a45eaa602ce6856623d8ca8884810c7f4926da2c6b17e4e271bca5a`.
  Deployment was skipped. The VM was initially stopped and was started locally
  for the build; neither Mac Pro was contacted.

`source-sha256.json` identifies the integrated source; `source.tar.gz` preserves it.
`build.json`, `linux-toolchain.log`, `cross-build.log`, `tests.log`, `linux-tests.log`,
and `clippy.log` record the build and checks. The Linux tests ran natively on the
ARM VM; the x86-64 binary was cross-compiled, not executed on a D700 machine.

## Checks after the user reports the Macs are back

First recover the interrupted slab results and verify saved governors, boot state,
recorders, and all three Thunderbolt streams using the existing
[slab recovery checklist](../prefill-slab-20261007/README.md). Do not infer permission
to resume from SSH reachability alone.

Build/deploy the recorded source to a new `~/trashcan-exp/` workspace, obtain and wait
for paired gpuq leases, and wrap each node's commands with `scripts/with-clocks.sh`.
Hold the performance governor and scoped zero-us CPU latency request on both nodes.
Release both leases promptly. Keep individual submissions below the amdgpu timeout.

1. Run `qwen-gdn-bench` at TP4 heads 4/12, token counts 9, 64, 128, 188, 256, and
   positions 0 and 700. Compare complete output/state; include prepare+barrier in timing.
2. Run `qwen-check` with 700 and 4300 prompt tokens, with/without MTP, each candidate
   separately. Include output limits 1..8, early stop, rollback, cached continuation,
   and the server checks. Compare greedy IDs to the recorded baseline.
3. Pair same-binary baseline/candidate runs at pp512/2048/4096/8192, tg128, with at
   least three resident repetitions. Test direct sum and prepared GDN independently
   before combining them. Record source/binary hashes, flags, exact timing windows,
   correctness, and both hosts' sustained clocks.
4. Compare fixed MTP, serial, and adaptive MTP on the mixed corpus and sustained cases,
   from comparable temperatures. Report probe overhead and per-case results. Do not
   pool throttled timings or require sampled token equality across decoding modes.

No new default GPU path or adaptive policy should be promoted on these offline results.
