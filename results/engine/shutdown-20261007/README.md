# Cooperative TP cancellation and GPU submission retirement

**Draft; not integrated or ready for full TP4 deployment.** The initial 9B suite
passed, but a later 27B TP4 run with that candidate stalled and reset both incite
GPUs at 11:14. The revised cleanup implementation was then tested with 27B-shaped
fake-peer sessions; full real distributed cancellation is still pending.
`pending-cancellation.patch.gz` preserves the current draft relative to main’s
prefill changes. Decompress it and apply with `git apply` only when continuing this validation.

The draft replaces the earlier signal-handler cleanup. A termination signal now
sets an atomic flag; normal control flow closes peer lanes, releases queued GPU
waits, waits for completion, and destroys resources before preserving the original
signal exit status. The signal handler performs no Vulkan calls, mutex operations,
or timed drain. SIGINT, SIGTERM and SIGHUP are supported; repeated signals preserve
the first signal's status. Handler installation occurs before TP startup/loading.

Transport polling, GPU polling, model loading, graph warming, peer listeners, and
the idle TP HTTP server observe cancellation. If submission fails after earlier
work launched, cleanup covers that partial schedule too. The latest version pulses each gated rank from a separate scoped worker while
the calling thread waits for completion, then joins those workers before destroying
the device. This revises the initial shared polling loop after the 27B failure. A failed or cancelled exchange closes its
peer connection so the remote process also observes the lost session.
The failed session is then unusable: a resident HTTP server rejects subsequent
requests with an instruction to reload the model, rather than computing with
only its remaining local ranks.

GPU input/parameter writes and graph-cache changes now wait for the previous
submission's timeline completion first, including both prefill queues. Previously
the gated submission wait occurred at the next submit, after those CPU mutations.
Completion targets are updated only after successful Vulkan submission, preventing
cleanup from waiting for a value that was never submitted.

This addresses GPU resource lifetime and termination behavior. The earlier
whole-system shutdown cause remains unresolved.

## Initial 9B validation on incite (f748a035)

The cancellation suite ran on incite while invite was offline or booting after repasting. Qwen3.8 27B's full TP4
configuration cannot fit in incite's 12 GiB alone, so GPU tests used the resident
Qwen3.5 9B Q4_0 model and its MTP file on incite's two D700s. They exercise the same
TP event/timeline scheduling, MTP, dual-queue prefill, and cleanup implementations.
The initial real 27B TP4 corpus run failed after 12 completed cases with a
transport stall and compute-ring timeouts; it cannot count as a successful validation.
The revised drain-worker binary passed three 27B-shaped fake-peer sessions on
invite, then a real TP4 repeat ended with incite’s whole-system power-off. A later
prefill baseline completed, but the candidate comparison was interrupted by a
user-confirmed circuit-breaker trip. Real cross-Mac interruption remains unverified. Same-host TCP was used only to test peer-loss behavior on incite.

`scripts/test-qwen-shutdown.py` passed:

- Plain and MTP inference: two repetitions each of 512-token prefill and 128-token
  generation, before and after the change. Generated token IDs matched exactly.
- Idle TP listener cancellation by SIGINT, SIGTERM, and SIGHUP.
- SIGTERM during loading; active event-gated decode by all three signals; repeated
  SIGINT followed by SIGTERM, preserving SIGINT status.
- Active timeline-gated decode, per-segment submission, dual-queue prefill, and
  MTP verification cancellation.
- A fake peer closed TCP after receiving the first byte of an all-reduce, leaving
  real GPU work gated. Both event-gated decode and dual-queue timeline prefill
  sessions cleaned up. That same node then served a real inference session.
- Idle resident TP HTTP server cancellation.
- A resident HTTP driver whose peer was stopped returned HTTP 500 for the lost
  session, then rejected a second request with the model-reload instruction.
- Fresh MTP inference after all interruptions, with unchanged token IDs.

All **16 signal cancellations** returned the expected signal status. Active decode
and MTP cancellation took about 0.31–0.42 s; dual-queue prefill was the slowest at
**1.22 s**. The two failed-peer sessions cleaned up in **0.10/0.71 s**. The same
kernel boot remained active, and the final journal contains no GPU reset, compute
timeout, PCIe completion timeout or panic. No verification process remains.

`cargo test --release`: **43 passed**, including seven isolated subprocess cases
covering blocked accept, handshake, idle/partial read, send, exchange, and first
signal preservation. `cargo clippy --release --all-targets` passed with the same
19 existing warnings and no new warnings. Python and changed-source whitespace
checks passed.

## Initial 9B throughput check (f748a035)

Mean of two repetitions per configuration; a short regression check, not a new
sustained throughput claim:

| Mode | Baseline prefill tok/s | Candidate prefill tok/s | Baseline generation tok/s | Candidate generation tok/s |
| --- | ---: | ---: | ---: | ---: |
| Plain | 256.05 | 255.84 | 46.27 | 46.25 |
| MTP | 249.42 | 249.93 | 66.08 | 66.09 |

Generation differences are below 0.1%; prefill differences are at most 0.21%.
Both D700s held **850/1370 MHz in every timed generation sample**. The candidate
plain prefill had one initial 450 MHz sample on GPU0; other comparison prefill
samples were at full clocks. Governor was performance and CPU latency request 0 us.
The final post-cancellation MTP run also matched outputs and held full clocks in
every timed sample. The whole suite peaked at 66/64 C. CPU governors were restored and lease 61
released promptly after the suite. System crash capture remained active.

## Initial tested source and reproduction

Baseline SHA-256:
`b52ecdeb0403a3678ea7863b5157a813cdc546c1584c0444d0093d85d0554ddb`.
Candidate SHA-256:
`f748a0353938fde8d463b861fd956d4bdd31e070875a5651fc2fde42eec8efd3`.
Baseline is the previously verified `reviewable` binary. Candidate was cross-built
in the kbuild VM and copied to `~/trashcan-exp/shutdown-20261007/candidate` on
incite. The tested source snapshot is `source.tar.gz`; build and test logs are
included. No build was run on either Mac Pro.

```sh
HOSTS=incite scripts/cross-build.sh trashcan-exp/shutdown-20261007/candidate
# Request/wait for an incite gpuq lease first; files/scripts staged in this workspace.
ssh incite 'cd ~/trashcan-exp/shutdown-20261007 && gpuq run ID -- \
  scripts/with-clocks.sh results/clocks.txt sudo -n python3 scripts/latency-run.py \
  python3 scripts/test-qwen-shutdown.py'
```

`incite/events.jsonl` records exact commands, signals, binary hashes and times.
`summary.json` joins each benchmark's exact prefill/generation windows to clock
samples. Per-case stderr/stdout and profile files, the complete suite clock log,
final kernel journal, and external telemetry records are preserved. An initial
launcher attempt failed on a governor permission check before GPU work; it was
corrected before the first full suite. The suite was then repeated after adding
the failed-session safeguard; this directory records that final source and run.

Vulkan reference requirements: command buffers being freed must no longer be
[pending](https://docs.vulkan.org/refpages/latest/refpages/source/vkFreeCommandBuffers.html),
and [timeline waits](https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitSemaphores.html)
provide host waits for submitted completion values. A
[device event state](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetEventStatus.html)
can change while its command buffer is still pending.

## Later revision and pending work

The revised drain-worker binary SHA-256 is
`412c608cb051f08674a01a58314975638ac6bfc1223f6e30acbd50105afab23a`.
`large-peer-drain/` records the three passing 27B-shaped fake-peer sessions on invite.
The initial `source.tar.gz` and the 9B suite above identify f748a035, not this revision.
The newer integrated source is archived in `results/qwen/prefill-slab-20261007/source.tar.gz`.
Failures and system evidence are in `results/kernel/repaste-20261007/` and
`results/kernel/system-shutdown-20261007/`. Both Macs passed Apple Diagnostics per
the user. GPU testing is stopped until the user confirms they are back on another circuit.

Before integrating: repeat the complete cancellation suite against the revised source,
validate real TP4 cancellation and peer loss over Thunderbolt on a fresh boot, collect
external traces, and confirm subsequent model sessions remain correct. The measurements
above do not establish the cause of any system power-off.
