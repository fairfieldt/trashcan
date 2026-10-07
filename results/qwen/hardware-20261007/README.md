# Qwen3.8-27B TP4 hardware validation, October 7

This is the historical test record for the binaries listed below. Prepared GDN, direct sum, and exchange-buffer
reuse were subsequently removed because they did not establish a model-throughput benefit. Larger attention
slabs and KV-only MTP were also retired. See the [cleanup record](../cleanup-20261007/README.md) for current
code and validation; the measurements and source archives here are unchanged.

This follows the offline implementation in `7ef122b` and the user's confirmation
that incite and invite were available for tests. Each Mac supplies two D700 ranks.
GPU work is queued separately on both hosts through gpuq. Experiment files and binaries are in
`/home/fairfieldt/trashcan-exp/tp4-offline-20261007`, except the narrowly scoped
Thunderbolt device setup and the user-authorized system fan service. The other incite checkout was not used.

## Source and reproduction

| Binary | Source | SHA-256 |
| --- | --- | --- |
| `baseline` | `b0a2e6884d06ac8798df70f7057910ff1ddff3fb` | `22e8874f5af3cd89f7758483a07b74bff469c6f07871e5f977b2882a74deb96d` |
| `trashcan` | `7ef122b` offline optimization candidate | `0bb31aab9a45eaa602ce6856623d8ca8884810c7f4926da2c6b17e4e271bca5a` |
| `trashcan-ready` | candidate plus bounded model-readiness wait | `3ba897a39c05794f5ece7228bbdf0456c9e9edee26a6cf3e23ef6706d85d07cb` |
| `trashcan-final` | `a1e90ea` plus `builds/final/source.diff` | `8f9eefc47d7bf01a9072ecf8567b4312d8eacfdab3e65069d258cd93be9706bf` |

The x86 Linux binaries were cross-built in the local kbuild VM with
`BUILD_ONLY=1 scripts/cross-build.sh`, not on the Mac Pros. Source archives, hashes,
commands and build logs are in `builds/baseline`, `../offline-20261007`, and
`cold-readiness`. Performance comparisons use the unchanged candidate binary;
the later readiness fix and final one-column reduction restriction have separate binary names and provenance. `builds/final/source.tar.gz`, `build.json`, and `source.diff` identify the final validation build, with 61 passing CPU tests.

The reproducible plans are JSON lists consumed by `scripts/cooled-plan.py`.
For example:

```sh
python3 results/qwen/hardware-20261007/scripts/cooled-plan.py \
  results/qwen/hardware-20261007/prefill-plan.json
python3 results/qwen/hardware-20261007/refresh-analysis.py
```

Each plan takes and waits for paired leases, runs a single resident repetition,
confirms teardown and governor restoration, then releases both leases before
cooling. The prefill plan requires all four GPUs at most 72°C before requesting and after
obtaining leases; subsequent plans lower this to 65°C to allow more cooling. Start temperatures, tickets, exact commands and every result are retained
in `plans/PLAN/`. The scoped runner sets all CPU governors to `performance` and
holds a zero-microsecond `/dev/cpu_dma_latency` request only for its child command;
it restores the original `schedutil` governors on exit. Both commands are wrapped
in `scripts/with-clocks.sh`. A timeout preserves active jobs and leases for inspection.

All main comparisons use the same three USB4STREAM links, in paired endpoint order
`/dev/tbstream0,/dev/tbstream1,/dev/tbstream2`. Their physical buses are 0, 2, 1.
Bus 1 returned on both hosts after the user replugged top-left; every link reported
two physical lanes. There is no TCP inference fallback in these runs.

## Measurement rules

`analyze.py` compares complete greedy token IDs and all recorded prefill top-eight
scores. Prepared GDN and direct-sum comparisons require exact scores. Different
attention implementations may use an explicit absolute tolerance of 0.002.
Correctness is checked independently of performance qualification.

GPU clocks are matched to the driver's recorded prefill/decode Unix windows.
Both hosts reported synchronized NTP, with measured offsets of -11.979 ms and
-3.927 ms. Clock sampling is approximately 250 ms; it cannot resolve very short
kernel or decode intervals. Performance-qualified windows require at least 90% coverage, at least two actual
in-window samples per host, and at least 99% of covered time at 850/1370 MHz on every GPU. The raw
results retain throttled or insufficiently sampled windows rather than dropping
them silently. Opposite run order and cooling reduce, but cannot eliminate, noise.

`pp_tp_ms` and `tg_tp_ms` measure driver-host phases. Exchange includes peer and
transport synchronization, while wait is host waiting for GPU completion. They
are not isolated kernel times. `add` is `sum + total`; do not count it again.

## Prepared DeltaNet kernel

Both hosts ran token counts 9, 64, 129, 188, 256 and 257 at positions 0 and 700,
with four key heads and twelve value heads per rank. Each mode used twelve timed
iterations after its warmup. All 48 old/prepared GPU comparisons matched every
output and final recurrent-state value bit for bit (maximum absolute error 0).

At 256 columns the original kernel took approximately 1.947–1.950 ms and prepared
normalization/gates plus recurrence took 1.522–1.527 ms, about 21.7–21.8% less time.
GPU timestamps include preparation and its barrier, excluding input/state resets.
Modes ran in a fixed order, so these are component measurements, not TP4 throughput.
The process clock logs establish full sampled clocks for 44 of 48 pairs; the first
9-column process on each host includes idle/startup samples. Individual modes have
no Unix windows, so millisecond kernel timestamps cannot be aligned to those logs.
See `micro-summary.json` and the host `gdn-*.jsonl`, stderr and clock files.

## Recovery and rejected measurements

The initial two-stream bring-up encountered missing device access, followed by a
cold-peer readiness timeout. `check-prepared-700` and its first retry are excluded
from performance comparisons. The warm retry passed exact checkpoint restoration,
every verification column, and 32 speculative-versus-serial continuation IDs at
700 prompt tokens, with prepared GDN and direct sum together.

The exploratory `base-pp2048-a` used two streams and overlapped the user's third-link
replug. Its three resident repetitions warmed invite GPU 0 to 96°C; they are not
used for candidate speed claims. Subsequent plans use cooled single repetitions.

Read-only recovery records under `recovery/` preserve boot state, previous trace
logs, crash recorders and recovered interrupted slab256 results. The recovered
slab256 repetition was correct, but its 143.654 prefill tok/s lies between the two
prior baseline repetitions (145.573 and 139.386). It does not establish a model
throughput gain for slab256. These earlier measurements used the original fan policy. No clock or thermal-protection overrides were used.

## Reliability fixes

Committed on main as `7ea585b`; the original optimization candidates are `7ef122b`.

`scripts/tbstream-up.sh` now waits for asynchronous stream discovery, preserves
matching HopIDs, rejects conflicting existing streams and grants access only to
the resolved endpoints when required. Nine local SSH/configfs mock tests cover
those paths, including propagating remote failures.

Only the initial TP model-readiness acknowledgement gets a longer bounded
first-byte wait (five minutes, sleeping between polls). After the first byte,
normal message progress deadlines remain unchanged, as do active exchange
protocols. The prior failed node process lasted about 113 seconds while the driver
left after about 61 seconds. This motivates cold-I/O headroom; five minutes is not
a measured loading guarantee. Six new framing tests plus two exhaustive budget-boundary tests bring the macOS
CPU suite to 61 passing tests, with two manual transport benchmarks ignored. The
Linux ARM VM passed all 59 tests present before the two CPU-only boundary tests.
Final Clippy also passes; logs retain existing warnings.

## Fan headroom and changed benchmark conditions

At the user's request, the fan controller now follows the live SMC maximum at all
temperatures. The former service cached 1900 RPM at boot; both hosts accepted the
higher live target and reached 2362/2356 RPM after 30 seconds. The deployed policy
then tracked small live-limit changes, with initial observed fan speeds around 2360–2380 RPM and later readings above 2400 RPM as the live limit rose.
Both services are active and enabled. This is about 24% more measured RPM, not an
airflow or model-throughput measurement. Commit `a1e90ea` and the exact commands,
hashes, samples and failure tests are in [the fan record](fans/README.md).

The old decode plan was intentionally interrupted during cooling with no leases
or GPU jobs active, after three completed cases. Its partial results are retained.
The full prefill/decode matrices restart in `*-maxfan-plan.json`; later plans also
record `fan_policy.regime=dynamic-smc-maximum` plus before/after fan readings and
installed-script hashes. Analysis keeps the old and new fan regimes separate.

## Full-model comparisons before the fan change

The mirrored 2,048-token prefill / 128-token decode series used tiled attention,
slab64, and the same candidate binary for all eight runs. Every complete greedy
sequence and every recorded prefill score was identical.

| Setting | Prefill tok/s, forward order | Prefill tok/s, reverse order | Decode tok/s, forward / reverse |
| --- | ---: | ---: | ---: |
| Original GDN, original sum | 150.619 | 140.945 | 28.016 / 28.116 |
| Prepared GDN | 149.194 | 143.023 | 27.981 / 28.046 |
| Direct sum | 144.486* | 142.103* | 28.126 / 27.787 |
| Both | 145.900 | 143.130 | 28.159 / 26.851 |

*Direct-sum-only prefill windows fail the full-clock qualification: invite GPU 0
spent sampled intervals at 450/650 MHz, with 98.90%/98.74% full-clock time and
93°C/95°C peaks. These values are preserved but cannot establish a regression.

The baseline loses 6.42% throughput between its repetitions even though both
windows pass the sampled full-clock criterion. Prepared GDN changes throughput
by −0.95% and +1.47% against the nearest baseline. This does **not** establish an
end-to-end prefill gain. The combination and direct sum show no consistent decode
gain either. Prepared GDN remains opt-in on the strength of its component result;
no new GPU or reduction default is justified by this matrix. See
`prefill-plan-analysis.json` for every window, phase and matched comparison.

## Full-model comparisons with maximum fans

The new eight-run mirrored matrix repeats the same 2,048-token prefill and
128-token decode workload after the fan deployment, with all four GPUs cooled to
65°C before each case. These results are compared within this fan regime only.
Every greedy token and recorded prefill score matches exactly.

| Setting | Prefill tok/s, forward order | Prefill tok/s, reverse order | Decode tok/s, forward / reverse |
| --- | ---: | ---: | ---: |
| Original GDN, original sum | 150.749 | 147.100 | 28.109 / 28.000 |
| Prepared GDN | 152.203 | 148.606* | 28.045 / 27.837 |
| Direct sum | 148.418 | 146.471 | 28.168 / 28.123 |
| Both | 150.037 | 148.902* | 28.106 / 28.139 |

*The marked prefill windows fall below the 99% full-clock criterion because of
brief initial clock ramp-up samples. The prepared-only sample is 44.8 ms after
prefill starts; the combination has initial low-clock samples at 34 ms on incite
and 9 ms on invite. Later samples are at full clock. This is not the sustained
mid-run throttling observed before the fan change. All eight decode windows pass.

Prepared GDN is +0.965% and +1.024% relative to the nearest original-GDN baseline,
while baseline repetition drift is −2.420%. The small consistent observed gain,
with one unqualified prefill window, is insufficient to change the default.
The stronger evidence remains the exact, faster component measurement.

Direct sum reduces the measured decode CPU sum phase from 0.404 to 0.279 ms/token
and 0.389 to 0.282 ms/token: 30.9% and 27.5% less time, saving 0.107–0.125 ms/token.
Whole decode changes +0.207% and +0.440%, versus −0.389% baseline repetition drift;
no reliable model-throughput improvement is claimed. Prefill sum time changes by
less than 1%, with no prefill throughput gain. The final runtime therefore keeps
direct sum opt-in and restricts it to one-column forwards with exactly two local
ranks. Single-token MTP drafts also take this path; their benefit was not isolated.
Multi-column prefill and verification retain the original sum. The broader
candidate used for this matrix remains archived for reproduction.

See `prefill-maxfan-plan-analysis.json` for raw windows, CPU phases, qualification,
matched comparisons, and within-configuration repetition drift.

## Short output budgets

All 64 cases (16 each for plain, old MTP, new MTP and adaptive MTP) match exact
greedy IDs and prefill scores. The counting prompt exercises output limits 1–8;
the one-word prompt emits one visible token then EOS for limits 2–8.

For count limits 1–8, the old fixed-MTP implementation evaluated 36 target columns
and drafted 24 tokens; the new one evaluated 28 and drafted 16 (22.2% and 33.3% less
work). Speculative rounds fell 12 → 9, with three cheap serial boundary steps.
Observed summed decode time fell 733.23 → 628.78 ms (14.24%) in this single pass.
These small windows do not support sustained-clock or throughput claims.
For the EOS cases, target columns fell 21 → 18 and drafts 14 → 11, with observed
summed decode time 440.30 → 395.05 ms. No output or stop behavior changed.

The exhaustive CPU model additionally checks 847 combinations of draft depth,
remaining context and output budget, and 8,470 rollback/history scenarios. It is
not a substitute for real Qwen cache tests.

## Sustained decode with maximum fans

The mirrored old/new × plain/fixed-MTP comparison uses pp512/tg256, context4096,
three streams, tiled/slab64, and the baseline/candidate hashes listed above.
All eight generation windows meet the full-clock criterion. All greedy IDs and
prefill scores match exactly across binaries and decoding modes.

| Setting | Forward tok/s | Reverse tok/s |
| --- | ---: | ---: |
| Old plain | 28.731 | 28.789 |
| New plain | 28.853 | 28.578 |
| Old fixed MTP | 37.889 | 37.854 |
| New fixed MTP | 37.958 | 37.954 |

New versus old plain changes +0.424%/−0.735%, with no consistent improvement.
New versus old fixed MTP changes +0.180%/+0.264%: a tiny observed difference,
not a material throughput claim. Every fixed-MTP run performs the same 109 rounds,
218 proposals and 146 accepted drafts, so the short-output boundary optimization
does not save work on this particular continuation. Existing fixed MTP remains
about 32% faster than plain decoding. The default exchange-buffer change removes
allocations with tested wire compatibility; this matrix does not establish a
model-throughput benefit for that host bookkeeping reduction.
See `decode-maxfan-plan-analysis.json` for software-version comparisons kept
separate from decoding-mode comparisons.

## Adaptive MTP with maximum fans

Two mirrored 256-token cases compare plain, fixed MTP and adaptive MTP on the same
candidate binary. All twelve generation windows qualify at full clocks; every
complete greedy sequence and prefill score matches exactly.

| Prompt | Plain tok/s, forward / reverse | Fixed MTP tok/s, forward / reverse | Adaptive tok/s, forward / reverse |
| --- | ---: | ---: | ---: |
| Python CSV parser and tests | 29.238 / 29.448 | 47.617 / 43.765 | 45.928 / 44.175 |
| Chinese transaction-isolation explanation | 27.705 / 29.550 | 29.504 / 28.220 | 30.593 / 30.806 |

On Chinese, adaptive improves +3.691%/+9.164% against the paired fixed runs, and
both adaptive observations beat both plain observations. Fixed MTP processes 427
target columns and drafts 284 tokens; adaptive processes 351 target columns and
drafts 177, using 85 serial steps. That is 17.8% fewer target columns and 37.7%
fewer drafts for identical output. This supports keeping the controller opt-in
as a workload-specific optimization, not enabling it by default.

On code, adaptive changes −3.547%/+0.935% versus fixed MTP. It takes twelve serial
warmup/probe steps, with the same 270 total target columns and 172 versus 180
drafts. There is no consistent benefit. Fixed MTP itself drifts −8.09% between
repetitions despite qualified GPU clocks. The work, source, environment, CPU
performance governors, zero-us latency request and affinity are identical.
The second fixed run takes 473.12 ms longer; measured driver exchange accounts
for 395.42 ms of that difference (83.6%). Exchange includes peer work/waiting,
so this does not identify a network fault. See `code-fixed-drift.json`.

The controller remains off by default, resets observations per generation, and
uses completed-round costs. Its measured benefit here must not be generalized to
all Chinese text, all workloads, or sustained services. See
`adaptive-maxfan-plan-analysis.json` for phase/work counts and paired comparisons.

## Full-model GDN profiling

The separate instrumented pp2048/tg1 pair passes recorded token/top-ID and dispatch
checks. All four ranks show a smaller prepare-through-recurrence envelope:

| Rank | Original GDN envelope, ms | Prepared envelope, ms | Reduction |
| --- | ---: | ---: | ---: |
| incite 0 | 940.58 | 868.72 | 7.64% |
| incite 1 | 929.09 | 850.91 | 8.42% |
| invite 2 | 984.19 | 961.79 | 2.28% |
| invite 3 | 981.80 | 954.82 | 2.75% |

The median paired reduction is 5.19%. Recurrence alone improves 14.8–16.9%, but
preparation costs 77.65–121.85 ms per rank. Adjacent `qwen_gated_norm` spans grow
72–108 ms, and complete DeltaNet segment totals remain essentially unchanged
(−0.33% to +0.42%). Thus neither the 21.7% standalone result nor the smaller
full-model GDN envelope establishes a complete DeltaNet/application improvement.
Prepared GDN remains an opt-in experiment.

All sampled clocks during these prefill windows are full; invite peaks at
89–90°C. TOP/BOTTOM_OF_PIPE dispatch spans overlap and may include dependency
waits. Two microbatch queues can overlap too. Summed dispatch/segment times are
not an additive wall-time breakdown or an Amdahl bound. The original GDN envelope
is roughly 4.9% of summed GPU segment time, a diagnostic ratio only.

`gemm2` projection spans dominate, especially FFN 4352×5120 (1,024 dispatches,
about 9.2–9.8 seconds summed per rank) and 5120×4352 (about 3.4 seconds). These are
promising next profiling targets; no new projection optimization is claimed.
`plans/profile-maxfan-plan/gdn-profile-summary.json` and `gdn-dispatch-shapes.json`
retain per-rank details. Instrumented wall throughput is not used to claim a gain.

## Final-source long-context correctness

The final `8f9eefc…9706bf` build passes `qwen-check --tokens 4300 --steps 32
--context 8192`, with MTP, tiled/slab64, prepared GDN and narrowed direct sum.
This crosses the 4,096-token sliding-window boundary. Checkpoint restoration is
exact; every verification column matches with maximum score error 0; all 32
speculative-versus-serial continuation tokens match. Both GPU scopes finish,
original governors are restored and both leases are released. This is a
correctness check, not a throughput benchmark. Evidence is under
`plans/long-check-maxfan-plan`.

## Server harness recovery

The first final serving attempt correctly received HTTP 400 because the test
client replayed a token-limited output message with `status=incomplete` as input.
No model correctness result is inferred from that attempt. The client now creates
valid assistant input with the exact generated text and omits output lifecycle
metadata. The original response captures remain unchanged. All 14 local harness
tests pass, including a mock reproducing this rejection.

The conservative runner left both jobs and leases intact. Recovery checked the
unique server identity and a fresh serial `/health` barrier, stopped only the
exact recorded driver process, waited for the peer's session teardown, and then
stopped that exact peer. Both scopes finished with original CPU governors
restored before tickets incite116/invite104 were released. Original artifacts
and recovery commands/proofs are under `plans/serve-final/input-error-recovery`.
The corrected four-mode plan uses fresh names under `plans/serve-final-v2`.

## Final server and cache validation

All **148 requests** pass on the final binary: 37 each for cached plain, cached
fixed MTP, cached adaptive MTP and plain with prefix caching disabled. These are
correctness checks, not server-throughput measurements. The cases cover output
budgets 1–8, full-history continuations, repeated continuations, edited prefixes,
deterministic replay, JSON/SSE output and natural EOS.

All three cached modes match exact visible output, response status, token usage
and cached-token counts. Each has eight followup cache hits; seven (budgets 2–8)
reuse generated state beyond the original 35-token prompt, with 36–42 tokens
reused. The uncached oracle reports zero cached tokens on every request and
matches every output/status/usage field except that explicitly excluded cache
counter. The API does not expose internal KV values or generated token IDs;
model-level exactness is supplied separately by `qwen-check`.

`final-evidence-audit.json` passes: it verifies source/archive and both-host binary
hashes, correctness results, expected experimental flags, completed process
scopes, restored CPU governors, fan snapshots and successful release of the exact
lease IDs. All four serving cases finish and both queues report no running or
queued jobs in the saved `incite/final-queue-status.txt` and
`invite/final-queue-status.txt`.

Reproduction from the repository root, using new output/run names for another run:

```sh
python3 results/qwen/hardware-20261007/scripts/cooled-serve-plan.py \
  results/qwen/hardware-20261007/scripts/serve-final-v2-plan.json \
  --out results/qwen/hardware-20261007/plans/serve-final-v2 --max-plan-minutes 40
python3 results/qwen/hardware-20261007/scripts/audit-final-evidence.py \
  --out results/qwen/hardware-20261007/final-evidence-audit.json
```

The saved command above identifies this run; its output directory already exists
and the controller refuses to overwrite it. See `scripts/SERVE-FINAL-PLAN.md` for
safe previews and unique-name requirements.

## Decisions at the end of this test campaign (superseded by cleanup)

- Default: reuse TP exchange progress storage, with unchanged protocol and
  measured lower allocator/bookkeeping cost. No general model-throughput gain
  established in the old/new matrix.
- Default: avoid unused MTP work at the output boundary. The short-budget cases
  use fewer target columns/drafts and preserve exact outputs; final cache tests pass.
- Opt-in: prepared GDN. Faster isolated recurrence/component work, but no reliable
  complete DeltaNet or model-prefill gain; existing GPU defaults remain unchanged.
- Opt-in: direct sum only for one column and two local ranks. Repeated decode
  CPU-phase saving, with no established whole-model speedup.
- Opt-in: adaptive MTP. Measured benefit on the Chinese fixture; no general gain
  or code-generation advantage established.
- Reliability: bounded cold-peer readiness, idempotent stream discovery/access,
  and live maximum fans at every temperature. Fan RPM is verified; airflow and
  continuous thermal/throughput improvement have not been measured.

The final source is committed on main as `53c228d`; engine hashes match the saved final build exactly. The final binary remains `trashcan-final` in both experiment workspaces. Earlier
binaries keep their names for reproduction. No shared production checkout was
replaced. After validation, the fan interval guard was deployed with the same
maximum-fan policy; its independent post-test checks are in the fan record.


Final fan tachometers are 2404–2409 RPM on incite and 2417–2421 RPM on invite. Both services are active/enabled; all 24 CPU governors per host are schedutil, no experiment model process remains, and all three streams remain accessible. See the per-host `final-state.json` captures.
