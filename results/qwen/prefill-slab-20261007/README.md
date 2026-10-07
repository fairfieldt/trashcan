# Prefill query-slab experiment, October 7

Historical status below was superseded by recovery: the completed candidate measured 143.654 tok/s,
between the two baseline runs. The 256-query option was subsequently [removed](../cleanup-20261007/README.md).
The measured 64-query tiled implementation remains. Original interruption notes and evidence follow.

**Validation incomplete:** the user confirmed that the shared circuit breaker tripped
while the first distributed candidate run was active. The user subsequently ran Apple Diagnostics on both Macs and reported no issues
on either. Their return after the circuit change is pending the user’s notification. The local harness is no longer running;
incite ticket 68 and invite ticket 56 were released. No model-level speedup is claimed.
Retrieve the candidate JSONL, error and clock logs from the Macs after they return.
CPU governors were scoped to the benchmark and saved in `slab-governors.before`;
the abrupt power loss prevented confirmation of the final restoration. Check them on return.

## Candidate

Tiled attention currently processes 64 query tokens between score, softmax and value
barriers. The candidate processes 256, reducing dispatch/barrier count and increasing
parallelism without changing each output’s arithmetic. Select both
`TRASHCAN_QWEN_ATTN=tiled` and `TRASHCAN_QWEN_ATTN_SLAB=256` on both nodes.
The existing 64-query path remains the default. This performance change is committed as an opt-in experiment. End-to-end throughput
validation remains incomplete, so the default stays at 64 queries.
At context 8192, measured allocation rises from 4365.7 to 4437.7 MiB per GPU: +72 MiB.

## Completed checks

- incite, both D700s, seven standalone attention shapes, 8 iterations each: scalar
  tolerance passes and 256-query outputs are bit-identical to 64-query outputs.
  Shapes include 64/129/256/257 queries, 4096/8192 keys and 1/2/6 query heads per KV head.
- At 6 heads, 256 queries, 8192 keys: GPU0 13.131 -> 9.339 ms, GPU1 13.189 -> 9.327 ms,
  approximately 29% less attention time. These are kernel measurements, not end-to-end
  prefill throughput. Full raw data and sampled clocks are retained.
- 43 CPU tests pass on the integrated working source; local Clippy has 19 existing warnings.
- Distributed baseline: Qwen3.8 27B, TP4, all three two-lane Thunderbolt streams,
  context 8192, 4096 prompt tokens, 8 generated tokens, two resident repetitions.
  Prefill 145.57 and 139.39 tok/s. Every prefill sample on all four GPUs is 850/1370 MHz.
  Both repetitions have identical top-candidate scores and generated token IDs.
- First candidate configuration was interrupted by the breaker trip before its results
  were copied to the workstation. Remaining candidate/baseline configurations were not run.

The A/B plan uses the same binary, model, prompt, CPU performance governor and scoped
0-us latency request; only slab size differs. `source.tar.gz` and `build.sha256` identify
that exact binary’s source. The archive also includes the prior unfinished cancellation
work, shared by both configurations. It is not a claim that those drafts are complete.
The separately verified commit source excludes the pending cancellation patches;
`committed-tests.log` and `committed-linux-tests.log` record 42 passing CPU tests
on macOS and Linux and `committed-clippy.log` records
19 existing warnings. `committed-build.log` and `committed-build.sha256` identify the
Linux cross-build of that exact source. `committed-source.tar.gz` preserves it.
The seven-shape GPU measurements and interrupted TP4 run used the integrated
experimental binary identified by `source.tar.gz` and `build.sha256`; these are
separate from the commit-only cross-build, which has not run on the offline Macs.
Build: `HOSTS="incite invite" scripts/cross-build.sh trashcan-exp/prefill-lds-20261007/trashcan`.
Use `BUILD_ONLY=1 scripts/cross-build.sh` to build in kbuild without contacting the Macs.
GPU commands use paired gpuq leases and `scripts/with-clocks.sh`; reproduced by the
saved `scripts/cooled-sweep.py`, `scripts/cooled.py` and `scripts/plan.json`.

## Tracer and this power loss

Both machines had active platform tracing, crash telemetry and the external receiver.
A harmless PCIe slot-power-status read delivered a real call stack from each Mac to
local disk and the external workstation. Startup trigger detection was corrected to
ignore the comments listing available triggers. `tracer-self-check.log` is the externally
received proof. No slot power was toggled.

The external log ends with invite telemetry at 12:10:53 and incite at 12:10:54 PDT.
The final-minute records are in `breaker-external-tail.log.gz`. This incident has a
user-observed external power cause; it does not establish the cause of the earlier
unexplained shutdowns. No deliberate shutdown was requested. Once power is restored,
collect the disk traces and candidate output before another benchmark.

## Resume checklist

On the user’s notification that the Macs are back, recover the interrupted candidate
JSONL, clock logs, kernel journals and platform traces. Check both boot IDs, restore
saved CPU governors if necessary, confirm both recorders and the external receiver,
and verify all three Thunderbolt streams. Then repeat the controlled prefill comparison
under fresh paired leases. Do not resume GPU work merely because SSH becomes reachable.
