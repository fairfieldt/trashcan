# Release readiness: next working session

Prepared 2026-10-25 after the 00:38 America/Los_Angeles request, for the next session after sleep
(Sunday daytime; “tomorrow” did not mean a scheduled Monday task). Planning and historical evidence belong on
`lab`; public setup instructions belong on `main`. No release or upstream submission is authorized by this plan.

## Start with these three actions

1. **Review the existing cleanup, then choose the preview scope.** Task #1361 owns the README/repo scrub in
   `~/dev/claude-work/trashcan-release-20261025/main`; do not start a competing cleanup. Recommend an experimental
   source preview with measured 1-, 2-, 4- and 6-Mac recipes, explicitly unqualified 3-/5-Mac configurations, and
   optional lights. Decide the project license with the maintainer (#1361 is already raising it). Acceptance of
   upstream patches need not block the preview. If “all counts 1–6 tested” is a release promise, qualify 3 and 5
   before tagging instead of implying the existing results cover them.
2. **Resume A1 / task #1291 once the cluster is available.** Recover or rebuild the baseline and mixer candidate,
   record their complete hashes, run the one-Mac gate (~15 min), then the six-Mac gate (~25 min). Keep the runtime
   change uncommitted until it passes. Confirm the 256-token allocation and end-to-end benefit; do not advertise
   the modeled 6 MiB margin as a measured result. Reuse these leases for observations that do not alter the test
   configuration; do not mix a kernel rollout or lights deployment into the comparison.
3. **Make one reproducible setup path and package the existing lights daemon.** While waiting for hardware,
   turn the surviving provisioner and example node inventory into a documented install/start/stop/verify path,
   with pinned artifacts. Import or version-pin `homelab/portlights`; its GPU activity detection already supplies
   blinking without inference-engine changes. Hardware validation follows packaging, on one node first.

After those: qualify the published multi-Mac recipes and recovery procedure, then refresh the independent upstream
submission series. Kernel submission preparation can proceed alongside packaging without touching cluster state.

## Ownership and dependencies

| Work | Existing owner / proposed role | Depends on | Completion evidence |
|---|---|---|---|
| Public README/repo cleanup | #1361 and its existing helpers | Maintainer license choice; downstream review | Clean clone builds/tests, no broken setup references |
| Mixer runtime candidate | A1, #1291 | Reachable Macs, recovered builds, gpuq leases | Identical dumps, allocation logs, clocks, paired end-to-end runs |
| Reproducible install and lifecycle | Release packaging role; new grouped To-do | #1361's final file layout, pinned kernel/model/binary manifest | Fresh-node install and safe repeat run, start/stop/restart and recovery transcript |
| Lights integration | Packaging role, reusing #1127's implementation | 0009 + 0010 running; debugfs; portlights package | Real inference triggers blinking, idle/stop returns SMC control, no inference regression |
| Upstream preparation | Kernel role; maintainer reviews/signs/sends | Current target-tree checks; missing tests below | Separate reviewable series with refreshed bases, build/checkpatch logs and honest test scope |
| Preview acceptance | Maintainer + release verifier | Scope, license, packaged setup, promised topology tests | Immutable source/artifact hashes and a signed-off release checklist; publication separate |

No new cleanup task or duplicate mixer test task is needed. Webb #1327 checked To-dos: only **A1** and **A14** were
open. A14 is the separate blog prototype review (#1354); it is not a release gate and the article stays on `lab`.
Proposed new To-dos are the four grouped workstreams below. Webb owns creating/deduplicating them and the single
Inbox handoff; this document does not send notifications.

## What the evidence actually supports

Historical results are evidence for the recorded source/environment, not a retest of the cleaned release candidate.
The supported hardware is Mac Pro 6,1, two 6 GB D700s per node, Debian 13, amdgpu SI, RADV. Other cards/OSes have no
qualification here. The measured barrier-dependent Flash-Next code uses Mesa 25.0.7; a Mesa or relevant shader change
requires the ISA audit and stress check in [decode kernels](../results/qwen4/decode-kernels-20261014/README.md).

| Macs | Reproducible recipe to publish | Evidence / boundary |
|---|---|---|
| 1 | Gemma 4 12B QAT Q4_0 over two GPUs; optional assistant. Qwen3.5-9B TP2 with optional MTP | [Gemma validation](../results/engine/validated-followup-20261003/README.md), [Qwen](qwen.md#measured-incite-2026-10-06). Full small-model inference, not full Flash-Next |
| 2 | Qwen3.8-27B TP4; document one-cable minimum separately from three stream lanes | [Merged TP4](../results/qwen/tp-merged-20261006/README.md), [wire checks](../results/qwen/tp4-wire-20261009/README.md). 27B exceeds one Mac's VRAM |
| 3 | Proposed Qwen 27B pipeline qualification using generic node chain | No dedicated 3-Mac end-to-end acceptance record located. The generic pipeline is not evidence of TP6 or of Flash-Next fitting |
| 4 | Qwen3.8-27B TP8, explicit ring versus full mesh cabling | [TP8 mesh](../results/qwen/tp8-mesh-20261009/README.md), [round 2](../results/qwen/tp8-round2-20261009/README.md). Headline rates require the measured topology/settings |
| 5 | Proposed Qwen 27B pipeline qualification using generic node chain | No dedicated 5-Mac acceptance record located. Flash-Next's five-Mac fit is not demonstrated |
| 6 | Flash-Next G-iq4, 12 GPU stages, head split 12, 32K configured context, batch 128, SSD n-gram table | [Chain correctness/server](../results/qwen4/chain-20261012/README.md), [latest decode pairs](../results/qwen4/decode-kernels-20261014/README.md), [long-context work](../results/qwen4/long-context-20261013/README.md). Do not describe configured cache capacity as a full 32K prompt acceptance test |

Flash-Next's full 177B model has **not** been demonstrated on 1–5 Macs. Four-Mac host-expert caching and the
lower-bit alternative are proposals in [VRAM](qwen4.md#vram), not implemented release recipes. A single Mac loading
selected stages/layers is not full-model loading. 204,800-position stage allocation/execution and retrieval checks
are not a six-Mac 200K end-to-end qualification; the 200K chain commands are marked unrun. Keep performance ceilings,
modeled fits and targets distinct from measurements.

The historical six-Mac G-iq4 recipe is about 116 GiB of GGUF, including 51 GiB of n-gram table, which stays on the
driver's SSD. Publish exact per-node shard/file requirements, checksums, RAM/disk requirements and source/conversion
revisions; do not assume other operators already have the lab's files or that arbitrary GGUFs work. Until reduced
per-node copies are tested, follow the measured same-model-file layout. Model download/redistribution permissions
remain separate from the engine's license.

## P0 — Complete cleanup review and preserve A1 (existing work)

- [ ] Review #1361's final diff and clean-clone validation, particularly references relocated from `results/` into
  `tests/reference-vulkan/`, removed cross-build helpers and crash services, and kernel docs. Do not cherry-pick the
  dirty runtime candidate into the scrub accidentally. Public examples use node roles and configurable inventory.
- [ ] Keep A1 as the sole mixer gate. [Snapshot manifest](../results/qwen4/mixer-copies-20261023/snapshot.json) and
  [candidate.patch](../results/qwen4/mixer-copies-20261023/candidate.patch) preserve the two dirty source files against
  `fc22c1fb`; [tools](../results/qwen4/mixer-copies-20261023/tools/mc-s1.sh) preserve the prepared runs. This is a source
  snapshot outside the active runtime, not an approved or GPU-validated implementation.
- [ ] Recover `trashcan-b7f4979` / `trashcan-mixc1` or rebuild both in the kbuild VM, recording the source, toolchain
  and full binary hashes. The previously reported `~/dev/claude-work/mixcopies-20261023/` directory is absent on the
  inspecting machine. No newer GPU results were found in the candidate directory. Six SSH probes timed out; that
  establishes unreachability, not that machines are powered off or that their services changed.
- [ ] Execute the existing one-Mac allocation/dump/timing plan, then `mc-chain.sh`: val14, val100, batch-256 val100,
  pp512 baseline/candidate and candidate batch128/256 at pp2048/8192. Compare every dump and token sequence, inspect
  resets, clocks and device-local usage. The modeled saving is 27.9 MiB/GPU, batch256 5863 of 5869 MiB: verify it.
  `mc-s1.sh` prints individual failures and can still reach “done”; inspect all statuses. `cmpdump.py` now exits 1 on
  differing, truncated or missing tensors; it previously only printed differences. A successful script exit alone
  is not acceptance. Keep batch128 if batch256 lacks a repeatable end-to-end benefit or safe allocation margin.

Lab run entry points (after reviewing the archived scripts' local paths and rebuilding as needed):

```sh
# Read gpuq rules; request/wait on every participating machine before GPU use.
# Build only with scripts/cross-build.sh in kbuild. Do not build on the Macs.
results/qwen4/mixer-copies-20261023/tools/deploy.sh
# Follow results/qwen4/mixer-copies-20261023/README.md for one-Mac lease-run and six-Mac chain6 wrapper.
# Remote GPU commands must be: ssh MACHINE 'gpuq run ID -- CMD'
# Compare all b1/b4/p16/p128/l2101 dumps with tools/cmpdump.py and inspect each exit status.
```

Record source/hash, commands, correctness, sustained clocks and paired outcomes under `results/`. Release leases
promptly; honor yield requests. No kernel installation, reboot or coordinated link reset is part of A1.

## P1 — Package setup, topology and lifecycle (new grouped To-do)

Sources to reuse: [node provisioner](../nodes/macpro/provision.sh), [node guide](../nodes/macpro/README.md),
[tblink](../nodes/macpro/root/usr/local/sbin/tblink), [unit](../nodes/macpro/root/etc/systemd/system/tblink.service),
[reset helper](../scripts/tb-reset.sh), [fan controller](../scripts/fancurve.sh), [kernel build](../kernel/build-7.3.sh),
[chain runner](../results/qwen4/chain-20261012/tools/q4run.py), [ring discovery](../results/qwen4/chain-20261012/tools/ring.py).
These links describe the archival lab versions; review the cleaned main versions before implementing.

- [ ] Define one inventory/config and manifest: roles/SSH identity, workspace/model paths, node/controller UUIDs,
  cable map and expected peers, addresses/ports, kernel package/config/patch hashes, Mesa version, engine/model
  hashes and quantization tool revisions. Generate the node UUID registry; do not ship the lab's registry as a
  functioning example. No tailnet, central scheduler or particular build host should be mandatory for outsiders.
- [ ] Provide a source-to-kernel-package path on a documented Linux build host, prerequisites, checksum verification,
  image/headers installation, boot parameters, fallback selection and read-only post-boot verification. The current
  builder pins 7.3-rc7 and config `7.2.9-p2p.config`; package suffix/revision must identify the exact patch set.
  Historical deployed `7.3.0-rc7-p2p-s2` lacks 0009/0010; building today's full set is a new kernel qualification.
- [ ] Package only required node services plus explicitly optional diagnostics: fan control, suspend prevention,
  amdgpu/permissions, debugfs/configfs prerequisites, NetworkManager Thunderbolt exclusion, tblink stream creation,
  udev render access and persistent logs. Explain crash capture/panic policy and fallback recovery choices without
  silently reinstating removed lab receivers. Installation should preview changed files, be repeatable, and offer
  uninstall/restoration; it must not reset links or reboot implicitly. Replace the archival provisioner's SSH tar
  model transfer with documented external downloads or appropriate bulk-copy tooling (lab: `lan-copy`).
- [ ] Promote a configurable launcher from the tested runner: start nodes before driver, bounded readiness and
  model-load deadlines, reject mismatched artifacts and occupied ports, stop all children on failure/SIGINT, and
  restore clocks/CPU latency settings. The archival runner uses fixed lab paths/names and changes DPM in its precheck
  outside `gpuq run`; do not ship that wrapper unchanged. Engine node commands exist, but a turnkey release launcher
  is not established. Measure idle behavior: the old 0-us latency wrapper holds CPUs busy between requests.
- [ ] Publish exact 1/2/4/6 examples with a preflight (kernel, two correct VBIOSes, local-VRAM P2P, memory budget,
  link widths/peer addresses, stream availability), smoke test, expected results and stop/restart instructions.
  Qualify 3-/5-Mac 27B pipelines if claiming all counts. Include TCP fallback tests and clear transport reporting
  in CLI logs; do not require experimental USB4STREAM options for a basic launch.
- [ ] Hardware acceptance: install/repeat on a fresh node; start/stop/restart each promised topology; stop a worker
  and verify bounded failure/cleanup; exercise fallback/reconnect only during an explicitly scheduled maintenance
  session. `tblink` addresses and recreates streams, but deliberately does not auto-reset a stuck controller.
  Reset all peers together only with no active jobs and separate authorization; after two failed rounds, the
  historical procedure can need a cable replug. Firmware can hang before chime with self-looping controller cables;
  broken controllers can need a power cycle. Document an operator's safe escalation, not unattended recovery claims.

## P1 — Integrate existing portlights (new grouped To-do)

The implementation is in **`~/dev/homelab/portlights/`**, last relevant commit `eaec80c`, from task **#1127**:
`src/main.rs` (node/lifecycle), `src/blinken.rs` (GPU fences), `src/panel.rs` (bulk sysfs API),
`deploy/portlights-node.service`, `deploy.sh`, `README.md`. One std-only Rust binary; the conductor/audio features
are separate from inference blinking. Existing deployment script hardcodes six lab hosts and the row file.

Dependencies: kernel [0009](../kernel/patches-7.3/0009-applesmc-mac-pro-port-lights.patch) then
[0010](../kernel/patches-7.3/0010-applesmc-port-lights-levels.patch), applesmc sysfs bulk levels and power controls,
root access to debugfs `amdgpu_fence_info`, and a working fan service. Historical hardware validation is **indite
only**, via module override, not the all-node production kernel. No current-state assertion is possible while SSH
is unreachable. [Hardware evidence](../results/kernel/port-lights-20261018/README.md).

- [ ] Choose package location (recommended optional component in trashcan, preserving history/attribution, or a
  pinned separately installable package); add configurable deployment, no default lab hosts, and source/package
  license review. For inference-only installations, make remote conductor input opt-in rather than opening UDP
  7465 on all interfaces by default. No audio/conductor dependency is needed for GPU-driven blinking.
- [ ] Preserve existing behavior: incoming conductor frames take priority for 0.5 s; any GPU's fence activity
  activates blinking, retained for 10 s; idle returns the panel to SMC control with power breathing; stop restores
  steady power. This detects all GPU work, not specifically inference. Only add an engine activity API if that
  distinction becomes an explicit requirement. Review startup before GPUs/debugfs exist: discovery currently
  happens once; panel reopen handles applesmc reload, but GPU re-enumeration/fence counter reset needs evaluation.
- [ ] One-node acceptance under a lease: actual small-model inference on each GPU, idle transition, service restart,
  absent-driver waiting, applesmc reload/reopen with fan control preserved, SIGTERM and failure cleanup; then test
  six-node inference. Compare lights on/off under the same kernel for throughput, clocks, CPU/SMC load and fan
  operation. Prior bulk-write measurements (8.05 ms all 20 changed, 2.15 ms five) are not an inference overhead test.
  Defer kernel installation/module reload and rollout until separately authorized.

## P1 — Refresh upstream patch preparation (new grouped To-do)

All draft series are on `lab`, removed from cleaned `main`. Bases/recipients below are **recorded snapshots**, not
fresh upstream checks; recheck target trees, existing fixes, maintainer lists and submission requirements before
sending. No patch has been sent according to the task records inspected. Keep subsystem series separate.

| Submission unit / archive | Recorded base and evidence | Work before submission |
|---|---|---|
| VFCT: amdgpu + radeon, 2-patch series; [draft](../kernel/upstream/vfct-exact-match/README.md) | `amd-staging-drm-next` 53750a937; unchanged-function replay; W=1 with/without ACPI; 6.12.111 boot checks for both drivers | Refresh base and Fixes links, rerun replay/build matrix; qualify on intended modern kernel. Radeon VBIOS source was inferred, not directly logged. Do not hot-load radeon to retest; historical attempt hung |
| PCI P2PDMA: one 8086:0e00 entry; [draft](../kernel/upstream/p2pdma-ivy-bridge-ep/README.md) | `pci/next` 1ee0e2a03ac; 5 runs/15 GiB, 0 bad words on 6.12.111 with translated IOMMU | Fresh build/checkpatch; show config-identical allowlist on/off behavior if needed (old stock comparison also changed CONFIG_PCI_P2PDMA). Do not add untested 0e01 |
| Thunderbolt stream: Tx E2E and HopID allocation unwind, independent fixes; [draft](../kernel/upstream/usb4stream-falcon-ridge/README.md) | `next-20261006` eea3fef32a9c; Tx fix hardware-tested on 7.2.9; HopID fix W=1 only | Rebase on thunderbolt `next`; consider separate submissions so tested Tx fix needn't wait for error-path qualification. Exercise allocation-failure/unwind with safe fault injection/test fixture; do not deliberately wedge the shared cluster |
| Applesmc LED/sysfs support, then bulk levels, 2-patch series; [draft](../kernel/upstream/applesmc-port-lights/README.md) | v7.3; file matched hwmon-next as of Oct 19; W=1 each stage, strict checkpatch except signoff; indite visual/race/reload evidence | Refresh hwmon-next and ABI KernelVersion/Contact; justify bulk interface using measured transaction savings. Sparse not run. Failed bulk-write retry is code-reviewed but not hardware fault-injected; add a mock/fault-injection test where feasible, retain honest caveat |

- [ ] Prepare an additional Thunderbolt review queue for local **0004 throttling parameter**, **0006 CLOSE behavior**
  and **0007 RX-flush close delay**; they are not all covered by the two-patch Falcon Ridge draft. For 0006 confirm
  the intended API semantics with the subsystem and test old-session CLOSE/new-session traffic, blocking/nonblocking
  and busy-poll paths; for 0007 retain close latency and shutdown correctness evidence. 0004 needs the measured
  latency rationale and an upstream API decision. Keep error-frame handling/disconnect proposals in
  [reliable-stream results](../results/kernel/usb4stream-reliable-20261013/README.md) separate from applied fixes.
- [ ] Exclude diagnostic 0008 (known faulty poll-mode experiment) and already-upstream SI GPIO/DC fix 0002
  (`cd883d0a3937`, as recorded locally). Make the deployable patch manifest and submission queue agree.
- [ ] For each refreshed series retain exact base, format-patch output, apply/build/W=1/checkpatch commands and logs,
  hardware configuration, test limitations and current get_maintainer output. Confirm `Assisted-by` formatting;
  only the human submitter certifies DCO/signoff, approves first-person cover-letter claims and sends the patches.
  Do not automatically add a signoff, publish a PR or send email.

## P2 — Preview acceptance (new grouped To-do)

- [ ] Maintainer selects license and preview scope; root license/third-party notices are present. No license is
  inferred from a GitHub-visible repository or from `engine/shaders/LICENSE.llama.cpp`.
- [ ] Install from a clean clone or source archive with documented dependencies, `cargo build --release --locked`
  and appropriate CPU/shader checks; missing glslc must fail clearly before a user tries GPU inference. Installer,
  kernel, engine, optional lights and model manifest are versioned and hash-verifiable. No personal paths,
  unreachable evidence links or undeclared lab services are needed for the public run path.
- [ ] Another operator can follow one-node instructions, then every topology promised in the scope. Record
  correctness, memory/device-local placement, clocks, smoke/server behavior, teardown and known constraints on
  the exact candidate. Historical benchmark numbers stay historical until retested; no new speed claim from A1
  without end-to-end evidence. Preserve readable evidence on `lab` and link stable revisions from release notes.
- [ ] Document known limits: hardware/OS, supported quantizations and contexts, 2 s GPU job limit, Mesa/barrier
  dependency, finite link recovery, no unattended cluster recovery guarantee. Portlights can remain optional;
  upstream acceptance, broad hardware support, full 200K chain qualification, 3-/5-Mac Flash-Next, new model formats,
  binary package repositories and music visualizer polish can follow the first preview.

Definition of done for the recommended first preview: a licensed, immutable source candidate; one complete
external setup path and pinned kernel/model artifacts; repeatable correctness and lifecycle checks for the stated
1/2/4/6 recipes; no hidden lab dependencies; honest unsupported-topology/context limits; optional lights install
and rollback instructions if included. A1 can either be validated and included or explicitly excluded. Publication
and upstream submission remain separate maintainer actions.

## Checks performed during preparation

See [preparation checks](../results/release-readiness-20261025/checks.md). No GPUs, CUDA/model loading, kernel
installation, reboot, service changes, link reset or cluster reconfiguration were performed.
