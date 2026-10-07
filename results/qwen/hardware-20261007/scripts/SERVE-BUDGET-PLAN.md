# TP4 Responses short-budget and cache regression

This supplements `qwen-check`/`qwen-bench`: it exercises the actual resident
`Qwen::generate` server path, native stop handling, HTTP output, full-history
continuations, and cache reuse after an output limit. It is correctness coverage,
not a serving-throughput benchmark. No hardware runs were performed while creating
these files.

## Run the prepared cooled plan

The default plan uses `trashcan-ready` on both nodes and records the hash of
`../cold-readiness/source.tar.gz` (relative to the plan file). It runs plain,
fixed MTP, adaptive, then plain with prefix caching disabled, with six-minute lease estimates. It checks
all four GPUs are at or below **65 °C before acquiring either lease**, checks again
after both grants, and releases unused tickets if they warmed while queued. It
reuses `cooled-plan.py`'s lifecycle and `release_evidence` unchanged: finished
scope records and restored governors are mandatory before release. Any uncertain
GPU/job state stops the plan with both tickets retained. A safely completed
correctness failure releases both tickets and stops before the next mode. MTP
and adaptive compare against plain after their own safe teardown. The final plain
case checks the same responses against generation that recomputes every prefix.

After the decode performance plan, run from the repository root:

```sh
python3 results/qwen/hardware-20261007/scripts/cooled-serve-plan.py \
  --out results/qwen/hardware-20261007/plans/serve-budget-cold \
  --max-plan-minutes 30
```

For a local-only command preview, use `--dry-run` and a different unused `--out`
directory. This wrapper never edits the existing cooled/run/scoped helpers. It
supplies only server-specific case validation and command construction to its
private imported instance of `cooled-plan.py`. `serve-budget-plan.json` is the
reviewable four-case plan. The whole-plan time bound is checked between cases
and cooling polls; it never interrupts a GPU operation.

## Run one mode at a time

The caller reads both gpuq rule sets, obtains and waits for paired leases, and owns
release/yield decisions. Cool and unload between modes. The runner accepts granted
ticket IDs; it never acquires or releases them. It also checks both hosts are at or
below 72 °C before starting either GPU command. An initial cooling timeout issues
no GPU command. The server and peer stay inside their lease wrappers for the entire
case. The CPU-only client sends localhost HTTP to that leased server and has no
second lease wrapper.

From the repository root, with the caller's current ticket IDs:

```sh
python3 results/qwen/hardware-20261007/scripts/run-serve-budget.py \
  --incite "$serve_incite_ticket" --invite "$serve_invite_ticket" \
  --binary trashcan --name serve-budget-plain --mode plain --stop-probe \
  --env 'TRASHCAN_QWEN_ATTN=tiled TRASHCAN_QWEN_ATTN_SLAB=64'
```

After confirming both commands exited and governors were restored, release the
leases, cool, obtain fresh paired leases, and repeat with:

```sh
python3 results/qwen/hardware-20261007/scripts/run-serve-budget.py \
  --incite "$serve_incite_ticket" --invite "$serve_invite_ticket" \
  --binary trashcan --name serve-budget-mtp --mode mtp --stop-probe \
  --env 'TRASHCAN_QWEN_ATTN=tiled TRASHCAN_QWEN_ATTN_SLAB=64'
```

An optional third run uses `--name serve-budget-adaptive --mode adaptive` with the
same remaining arguments. Modes use the same binary; `plain` omits the MTP sidecar,
`mtp` uses fixed depth 2, and `adaptive` additionally enables the adaptive controller.
Explicit `--draft-tokens` can test another supported depth without changing the
client corpus. `--source-archive PATH` records the exact build-source archive hash;
use the archive matching the tested binary. Both remote scope records include the
actual binary SHA-256, kernel, boot ID, governors before/during/after, and latency
request. Fan controls and the installed fan-script hash are captured before/after;
the plan records `fan_policy.regime=dynamic-smc-maximum`. These are read-only
metadata; the harness does not change fan controls. Harness source snapshots/hashes
and exact SSH commands are saved locally.

The fourth case uses `--mode plain --no-prefix-cache` and a `--reference` pointing
to the completed cached plain capture. Every response must report zero cached
tokens; the comparison omits only `usage.input_tokens_details.cached_tokens`.

`--dry-run` writes a local manifest/source snapshot and makes no remote calls.
Unique run names are required; neither prior local nor remote result directories
are overwritten. The default server binds localhost:8017. The runner checks its
unique model ID before requests so another service is not mistaken for this one.

## Coverage and bounds

The server uses context 1024, batch 64, eight checkpoint slots, greedy sampling,
non-thinking chat, and default MTP depth 2. The client uses budgets 1 through 8,
with at most eight generated tokens per request and two-second idle pauses between
budget groups. Each budget performs:

1. An initial short output that must hit the requested output limit.
2. An immediate full-history continuation using the actual assistant output.
3. The identical continuation again, switching JSON/SSE framing.
4. For budgets 1/2/3, an edited original user prompt to exercise prefix invalidation.
5. The original short request again, checking deterministic replay after the branch.

This is 35 requests. `--stop-probe` adds a greedy one-word prompt and its SSE replay
(maximum eight tokens each); both must report natural completion. Total: 37 short
requests. Early EOS in a boundary fixture is a coverage failure rather than a pass.
Continuation input preserves each assistant text part exactly and omits output-only
metadata, including the `incomplete` lifecycle status from a token-limited response.
The original output remains unchanged in the capture.
The stop fixture likewise fails explicitly if it does not reach EOS within its bound.
`--budgets 1,2,3` provides a smaller initial case if needed. Client wall budget is
120 seconds between requests, per-request HTTP timeout 30 seconds, runner wall
budget 240 seconds; no active GPU job is killed on a timeout. Batch size bounds
individual prefill submissions; long-running/incomplete cases are left intact for
inspection, not automatically cancelled. This plan does not test tool-call early
completion; that belongs to the separate server API suite.

Within one mode, replay comparisons omit cached-token counts because a replay can
legitimately have a different cached prefix. We record every count and require at
least one full-history follow-up at budget 2 or above to reuse more tokens than its
initial input contained. Prompt-only cache hits therefore cannot claim coverage
of generated state. Counts and evidence are saved in `client-summary.json`.
Individual continuations can miss because of template re-encoding, so a hit is
not required on every budget. Across cached modes, comparison requires exact outputs, status, token usage **including
cached-token counts**, and normalized full-history input equality. IDs and timestamps
are excluded. The API exposes neither raw token IDs nor internal KV/history arrays;
this verifies API behavior and observable cache reuse, alongside existing direct
model token-ID checks. The uncached oracle requires exact output/status/usage parity
while explicitly excluding only cached-token counts and asserting those counts are
zero in its own capture.

## Compare collected captures

```sh
python3 results/qwen/hardware-20261007/scripts/serve-budget-client.py --compare \
  results/qwen/hardware-20261007/serve-runs/serve-budget-plain/incite/client \
  results/qwen/hardware-20261007/serve-runs/serve-budget-mtp/incite/client
```

For adaptive, replace the second path with `serve-budget-adaptive/incite/client`.
Compare only runs with identical budget sets, prompts, stop-probe settings, binary
and other environment settings. The comparator requires complete successful captures.
Raw JSON/SSE payloads, request bodies, status/headers and timing windows are retained
in `http.jsonl`; parsed generation records are in `responses.jsonl`.

## Idle proof and teardown

This is a source-specific barrier, not a general property of `/health`:

- `server.rs::run` invokes `handle` inline in one `incoming_requests` loop; no GPU
  worker is dispatched. `Qwen::generate` writes the terminal HTTP output, then calls
  `chain.checkpoint`, then returns to that loop.
- Local checkpoint copies call `GpuStage::checkpoint` → `Device::run` →
  `run_reusable`, which waits on its fence. A subsequent `/health` handled by that
  same serial loop therefore proves the **driver's local** checkpoint finished.
- TP peer checkpoint messages have no acknowledgment. Health does not prove peer
  idle. The runner leaves the peer running until its own session-end teardown log
  appears, then applies the existing `run-tp4.py` exact-PID/start-time/argv check.

The client confirms the unique server identity and a valid successful generation
terminal response, then requests that serial barrier. HTTP or terminal-response
failure leaves `safe_to_stop=false`; a later generic health response cannot erase
uncertain GPU state. Server request/cleanup errors also block automated shutdown.
After safe completion, only the exact recorded driver PID receives SIGTERM. It
closes the stream; the peer finishes its local work and exits its session before
its exact process is stopped. No broad pkill, module reset, reboot, or service
mutation is used. A timeout/error without these proofs retains processes and leases
for the caller to inspect. Check both finished `scope.json` files and restored
governors before release; `cooled-plan.py::release_evidence` can read the completed
run directory unchanged.

## Local checks

`python3 test-serve-budget.py` from this directory runs a serial local HTTP
mock and validates all 37 requests, JSON/SSE replay, cache-count mismatch detection,
wrong-server rejection, safe handling of early EOS coverage failure, refusal to
mark an HTTP/GPU error idle, and all four runner dry-runs with an SSH tripwire.
Fourteen tests passed, including valid replay of incomplete output, prompt-only hit rejection, one qualifying generated-state
hit, the uncached oracle and non-cache usage mismatch rejection, the 65 °C wrapper
dry-run, raw SIGTERM exit handling, and release-evidence checks. No SSH, GPU, model
loading or lease operations occur in these tests.
