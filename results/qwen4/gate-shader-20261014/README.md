# The bounded gate shader, round 3: single-GPU checks on alight (2026-10-14)

**Result: the gate releases once the polling wave evicts the gate's line from the GPU's L2 between polls.** Before
each poll the wave reads 8 MiB of device-local memory at a 4 KiB stride (2048 lines). Every one of 2,600 release-always
runs with that sweep saw the host's write:
- both D700s of alight;
- 14 processes, each a fresh placement of the gate and the sweep (in 7 more, never released, every poll missed the
  L2);
- host-cached and write-combined gate memory;
- the top and the lowest DPM level.

Release latency, host write to the GPU timestamp after the gate, at a random phase (1,800 runs, top level):
- p50 11.5 µs, p90 17 µs, p99 18.4-19.0 µs, max 24 µs;
- the event gate (`vkCmdWaitEvents`) released in 5-13 µs in the six-Mac traces.

Costs:
- the wave polls every 12.3-13.5 µs;
- GPU 0 drew 51.5 W while gates waited, against 48.1 W idle at the high level the engine holds (+3.4 W);
- an already-open gate costs +4.5 µs per graph.

No amdgpu timeout, reset or fault on alight in any of the three leases. One stop came from this round's own check:
GFX6's shader clock is not a time base. `s_memtime` counts cycles of the current shader clock (841 MHz at the top
DPM level, 297 MHz at the lowest), so a tick bound stretches up to 2.8x when a GPU drops to its lowest level. Gate
bounds are now counted at 300 MHz and hold at every level.

Follows [results/qwen4/gate-shader-20261013](../gate-shader-20261013/README.md) (rounds 1-2: no poll variant saw the
host's write; polls ran at L2-hit speed). Tests 1-3 below are the three checks asked for. vgem was not tested: it was
to be tried only if all three failed, and test 2 works.

## Test 1: what ACO emitted

Machine code from `qwen4 gate-isa` (VK_KHR_pipeline_executable_properties, Mesa 25.0.7). It was dumped on alight's
GPU 0 and on RADV's null Tahiti device (`RADV_FORCE_FAMILY=tahiti`, no GPU), and the two dumps are identical
([alight/lease1/isa-gpu0.txt](alight/lease1/isa-gpu0.txt)). ACO has no GFX6 disassembler, so the dump is ACO's final
program, cache bits included.

| poll | shader | the poll instruction | glc | slc | sees the host's write without a sweep |
|---|---|---|---|---|---|
| volatile load (`coherent volatile` buffer: Coherent and Volatile decorations) | `qwen4_gate_clock` | `buffer_load_dword ... glc` | yes | no | no (rounds 1-2; test 3) |
| `atomicOr(v, 0)` | `qwen4_gate_clock` (flags 1) | `buffer_atomic_or ... glc` (returns the old value) | yes | no | no (round 1) |
| Volatile and Nontemporal together (Vulkan memory model) | `qwen4_gate_nt` | `buffer_load_dword ... glc slc` | yes | yes | no (round 2; test 3) |
| device-scope atomic load, acquire (new) | `qwen4_gate_atomic` | `buffer_load_dword ... glc` | yes | no | no: ran to its bound, host-cached and write-combined memory |
| volatile load + L2 sweep (new) | `qwen4_gate_sweep` | `buffer_load_dword ... glc`; sweep: 8 `buffer_load_dword ... offen glc` in flight | yes | no | **yes** |
| nontemporal load + L2 sweep (new) | `qwen4_gate_sweep_nt` | `buffer_load_dword ... glc slc`; sweep: 8 plain loads in flight | yes | yes | **yes** |

ACO emits every bit asked for, so there is nothing to fix in the compiler. With glc and slc together the poll still
ran at L2-hit speed (0.45-0.47 µs per poll) and never saw the write. On Tahiti, slc does not make a load miss a line
that is already in the L2.

Two details of the machine code:
- The sweep gate's second `s_memtime` comes after the poll's `s_waitcnt vmcnt(0)`, so the per-poll load times
  below are real.
- Under the Vulkan memory model ACO waits for every volatile load. So the nontemporal variant's sweep buffer is not
  volatile; its addresses depend on the poll count through two zero push constants, so the sweep cannot be hoisted.

## Test 3: host fences

| host write | gate | result |
|---|---|---|
| host-cached memory, `clflush` of the line + `mfence` | volatile loads; nontemporal loads | ran to the 50 ms bound (0.47-0.53 µs per poll) |
| write-combined memory, `sfence` | volatile loads; nontemporal loads | ran to the 50 ms bound (0.45-0.54 µs per poll) |
| either, with an L2 sweep (test 2) | volatile or nontemporal | released every run (below) |

The fence makes no difference either way. Alone it cannot help: the GPU never reads memory again while its L2 holds
the line. With the sweep, `mfence`, `sfence` and `clflush` give the same release latency. The host's write and fence
take 0.12-0.30 µs (p50).

## Test 2: evicting the gate's line from the L2

`qwen4_gate_sweep`: one 64-lane wave. Each round it polls the gate word (host memory) and times that load, then
reads `lines` dwords of device-local memory `stride` bytes apart. The sweep-size table comes from:
- lease 1: 100 runs per row, release 200 µs after the start;
- lease 2: 200 runs per row, release at 200 µs plus a random 0-1000 µs.

Rows of 100 runs are lease 1, rows of 200 lease 2; a failing row stopped after its first 1-3 failures. "Poll load" is
one poll's load. An L2 hit takes 0.27-0.31 µs; a miss that reads host memory over PCIe takes 0.8-1.0 µs.

| sweep: span / stride / lines | µs per poll | poll load µs | released |
|---|---|---|---|
| none | 0.7 | 0.27 | 0 of 1 |
| 256 KiB / 64 B / 4,096 | 10.6 | 0.27 | 0 of 3 |
| 512 KiB / 64 B / 8,192 | 22.8 | 0.30 | 0 of 3 |
| 768 KiB / 64 B / 12,288 | 35.9 | 0.85 | 100 of 100 |
| 1 MiB / 64 B / 16,384 | 47.4-47.7 | 0.84-0.87 | 100 of 100; 200 of 200 |
| 2 MiB / 64 B / 32,768 | 94 | 0.90 | 100 of 100 |
| 4 MiB / 64 B / 65,536 | 186 | 0.97 | 100 of 100 |
| 1 MiB / 256 B / 4,096 | 15.2 | 0.28 | 0 of 3 |
| 4 MiB / 256 B / 16,384 | 103 | 0.89 | 100 of 100 |
| 4 MiB / 1 KiB / 4,096 | 24.7 | 0.83 | 100 of 100 |
| 4 MiB / 2 KiB / 2,048 | 12.7 | 0.88 | 200 of 200 |
| 2 MiB / 4 KiB / 512 | 2.9 | 0.31 | 0 of 3 |
| 4 MiB / 4 KiB / 1,024 | 6.4 | 0.29 | 0 of 3 |
| 6 MiB / 4 KiB / 1,536 | 9.6 | 0.82 | 200 of 200 |
| **8 MiB / 4 KiB / 2,048** | **12.3-13.5** | **0.82-0.88** | **every run (2,600)** |
| 12 MiB / 4 KiB / 3,072 | 19.8 | 0.83 | 200 of 200 |
| 16 MiB / 4 KiB / 4,096 | 23.5 | 0.83 | 200 of 200 |
| 32 MiB / 4 KiB / 8,192 | 49.0 | 0.90 | 100 of 100 |
| 8 MiB / 8 KiB / 1,024 | 6.1 | 0.29 | 0 of 3 |
| 16 MiB / 8 KiB / 2,048 | 13.0 | 0.27 | 0 of 3 |
| 32 MiB / 16 KiB / 2,048 | 15.0 | 0.90 | 200 of 200 |

What the table shows:
- A contiguous sweep works from 768 KiB, the L2's size, up.
- Sparser sweeps need far fewer lines, and the stride matters as much as the count. At a 4 KiB stride, 2,048 lines
  work and 1,024 do not. At an 8 KiB stride, 2,048 lines failed in both processes tried. At 2 KiB and 16 KiB strides,
  2,048 lines worked.
- The L2's set mapping is undocumented, so sparse sweeps may depend on where the buffers land.

8 MiB at 4 KiB was chosen as the cheapest sweep with margin: 6 MiB works, 4 MiB does not. It evicted the line in all
21 processes that used it (14 with release-always runs, 7 never released; every poll load took 0.77 µs or more):
- 10 with host-cached memory on GPU 0 and 5 on GPU 1;
- 2 with write-combined memory, 2 with nontemporal polls;
- 2 at the low DPM level.

Each round costs:
- the poll's load, ~0.8 µs;
- the sweep, 11.8-12.7 µs: 128 KiB read from VRAM, ~10 GB/s while waiting, one wave on one CU.

Release latency at a random phase (`latency.txt`; to the GPU timestamp after the gate, which includes the gate
dispatch's end, ~1.5 µs):

| gate | runs | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| 8 MiB / 4 KiB, host-cached, GPU 0 | 1,200 | 6.3 | 11.6 | 16.7 | 18.4 | 24.0 |
| 8 MiB / 4 KiB, host-cached, GPU 1 | 600 | 6.3 | 11.4 | 17.0 | 19.0 | 19.9 |
| 8 MiB / 4 KiB, write-combined + `sfence`, GPU 0 | 400 | 7.0 | 11.9 | 17.3 | 19.1 | 39.9 |
| 8 MiB / 4 KiB, nontemporal polls, GPU 0 | 200 | 9.8 | 14.8 | 19.6 | 21.2 | 21.5 |
| 6 MiB / 4 KiB | 200 | 6.0 | 9.9 | 13.9 | 15.1 | 15.6 |
| 16 MiB / 4 KiB | 200 | 7.3 | 16.8 | 26.9 | 30.8 | 31.8 |
| 1 MiB / 64 B (contiguous) | 200 | 12.7 | 34.2 | 50.4 | 54.5 | 56.6 |
| 8 MiB / 4 KiB at the low DPM level (300/150 MHz) | 100 | 16.3 | 38.1 | 60.3 | 64.5 | 64.5 |

Nontemporal polls bring nothing: the same latency to the open poll, and ~3 µs more to end the gate (the memory
model's waits on the result stores).

The cost of a gate that is already open (200 runs each, GPU µs from the timestamp before the gate to the one after
the input barriers; host µs from the start event seen to the end event seen):

| | GPU 0 | GPU 1 | host |
|---|---|---|---|
| no gate | 2.07 | 2.07 | 3.31-3.40 |
| `qwen4_gate_clock` (one lane) | 6.37 | | 7.73 |
| `qwen4_gate_sweep` (the open first poll skips the sweep) | 6.67 | 6.52 | 7.89-7.92 |

That is +4.5 µs per graph (round 1 measured +2.4 µs for the clockless `qwen4_gate`). The first poll of every
submission read host memory (1.0-1.8 µs at the top level) and saw the current value. So the L2 keeps no host line from one submission
to the next, and an open gate needs no sweep.

## The shader clock and the bounds

Lease 2 stopped on its own check. GPU 1 had idled for ~2 s at amdgpu's `auto` level and dropped to 300/150 MHz.
The gate's 16-poll calibration run then counted 300.5 µs of shader clock against 862.5 µs of command-processor
timestamps. `s_memtime` counts the shader clock's cycles:
- 841.2-841.4 MHz at the top level;
- 296.9 MHz at the forced low level.

Rounds 1-2 measured 838.8 M ticks/s only because their GPUs were at the top level. A bound in ticks therefore
stretches up to 2.8x at the lowest level, and thermal throttling (invite's GPU0 at 450 MHz) stretches it too.

Since then the gate's tick bound is counted at 300 MHz (`--min-mhz`), so it holds at every DPM level, and at the top
level the gate ends at 0.35x its bound. Two backstops remain:
- a poll-count bound sized from a 16-poll run timed by the command processor;
- every run must end within its bound by the command processor's timestamps, with a shader clock of 250-900 MHz,
  or the step stops.

Never-released gates, each ended by its bound ("clock" = shader-clock bound, "polls" = poll-count bound):

| gate | level | bound | ended after | by |
|---|---|---|---|---|
| ticks counted at 838.8 MHz (leases 1-2) | | | | |
| `qwen4_gate_atomic`, GPU 0 | top | 50 ms | 49.77 ms | clock |
| `qwen4_gate_sweep` without sweep, GPU 0 | top | 50 ms | 49.77 ms | clock |
| 8 MiB / 4 KiB sweep, GPU 0 | top | 50 / 100 ms | 49.78 / 99.10 ms | clock / polls |
| every failed release run of tests 1-3 | top | 50 ms | 45.6-49.8 ms | clock or polls |
| ticks counted at 300 MHz (lease 3) | | | | |
| 8 MiB / 4 KiB sweep, GPU 0 | low (300/150) | 50 ms | 49.41 ms | polls |
| `qwen4_gate_sweep` without sweep, GPU 0 | low | 50 ms | 50.33 ms | clock |
| `qwen4_gate_clock`, GPU 0 | low | 50 ms | 50.32 ms | clock |
| 8 MiB / 4 KiB sweep, GPU 1 | top | 50 / 100 ms | 17.77 / 35.60 ms | clock |
| 8 MiB / 4 KiB sweep, GPU 0 | top | 100 ms | 35.60 ms | clock |
| nontemporal 8 MiB / 4 KiB; 1 MiB contiguous; `qwen4_gate_atomic` on GPU 1 | top | 50 ms | 17.76-17.80 ms | clock |
| 3 x 430 gates back to back (power phases) | top | 100 ms | every one by its bound | clock |

The longest submission of the round was 99.1 ms (a 100 ms bound at the top level, lease 2). Order was as asked: in every
lease, release-always steps first, then never-released ones.

## Power while gates wait

GPU 0 held at the high level, as the engine's clock holder holds the GPUs during a run. GPU 1 at `auto`, idle.
There were 15 s per phase of never-released gates back to back, each ended by its bound. Power is SMC PG0R, sampled
every second by `system-crash-capture` on alight and received over netconsole
([alight/lease3/smc-telemetry.log](alight/lease3/smc-telemetry.log), [power.txt](power.txt)).

| GPU 0 | PG0R |
|---|---|
| idle at the high level (before, after) | 47.8 W, 48.4 W |
| plain poll gate, no sweep (L2 hits) | 49.7 W |
| **8 MiB / 4 KiB sweep gate** | **51.5 W (+3.4 W)** |
| 1 MiB contiguous sweep gate | 54.6 W (+6.5 W) |
| idle at `auto` (for reference, from the telemetry before the leases) | 13.4 W |

In a six-Mac decode step, 11 of the 12 GPUs wait at any time, so sweep gates would add about 37 W. For comparison,
the six Macs draw ~1,360 W in decode, and holding the CPUs at `performance` cost +195 W.

## Incidents

- **Kernel log:** no amdgpu timeout, reset or fault, no CmpltTO or Oops on alight in any lease. The brief's pattern
  (`grep -i ... BUG ...`) matches six boot-time lines (for example "Kernel Debug File System" and
  "[Firmware Bug]"). They are listed as the baseline in each lease log, and any new match would have stopped the
  lease.
- **Lease 2 stopped on its own check** (above): one 16-poll run (0.86 ms) on GPU 1 at 300 MHz.
- DPM: both GPUs accepted `high` (then `auto`) before every lease. `amdgpu_pm_info` showed power level 3
  (850/1370 MHz, pcie gen 3), and both were at `auto` after each lease. No module loads or unloads; vgem was not
  loaded.

## Recommendation

**The gate shader, with the 8 MiB / 4 KiB sweep, is worth a six-Mac A/B once six-Mac work resumes.** It is the
first gate on these GPUs that bounds itself on the GPU and still sees the host:
- It releases ~11.5 µs (p50) after the host's write, against 5-13 µs for the event gate. The event gate took stage
  submission to GPU start from 2.24-2.44 ms to 0.25-0.30 ms per token, about 2 ms (+4% decode). The sweep gate
  should keep most of that.
- Its costs: about 0.05 ms per token of extra release latency, +4.5 µs per already-open gate, and ~37 W on six Macs.

What the A/B build needs:
- **The gate word alone in a page of host-cached memory, the 8 MiB device-local sweep at a 4 KiB stride.** The host
  writes the word and fences (`mfence`); `clflush` buys nothing.
- **A startup check that the gate's polls miss the L2.** A poll load of 0.27 µs means the line stayed in the L2; a
  miss takes ~0.8 µs. If the check fails, enlarge the sweep (16 MiB at an 8 KiB stride failed in both processes
  tried). The set mapping is not documented, so this guards against an unlucky placement.
- **The bound counted at the lowest shader clock (300 MHz).** For example, 0.5 s at 300 MHz is 0.18 s at the top
  level, against amdgpu's 2 s. Keep the poll-count backstop timed by the command processor.
- **A timed-out gate must not let the stage run on stale input.** The gate records how it ended; the graph's work
  must be skipped (predicated) or redone by the host.

vgem was not tested. It is the fallback if the A/B finds the sweep's power or its placement sensitivity too costly:
the kernel holds the job and nothing polls on the GPU, but vgem must be loaded on every Mac. Six-Mac work is on hold
until the power setup is confirmed, so no A/B was run.

## What ran

| lease (alight, one-Mac gpuq) | source | binary `trashcan-gate3` (md5) | script | log |
|---|---|---|---|---|
| #106: ISA dumps; tests 1-3 release-always; sweep sizes; open gates | b27bd0ac | a78b097d | [tools/gate3-lease1.sh](tools/gate3-lease1.sh) | [lease1.log](lease1.log) |
| #108: jittered releases, sweep scan, robustness over fresh allocations; stopped by its own check | 7805c813 | afba41ae | [tools/gate3-lease2.sh](tools/gate3-lease2.sh) | [lease2.log](lease2.log) |
| #109: low DPM level, never-released, open gates on GPU 1, power | 2fc17c69 | 6846e624 | [tools/gate3-lease3.sh](tools/gate3-lease3.sh) | [lease3.log](lease3.log) |

- Branch `gate-shader-2-20261014`, from `host-gate-shader-20261013`; not for merge.
- Built with `scripts/cross-build.sh` in kbuild, `cargo test` (75 tests) before each deploy.
- Every step ran under `scripts/with-clocks.sh` with `TRASHCAN_TP_PROFILE=1` (calibrated timestamps). Clock logs
  are in `alight/leaseN/clocks/` ([tools/gate3-clocks.py](tools/gate3-clocks.py)). The GPU under test was at 850/1370
  MHz in every step except two groups at 300/150: the four at the forced low level and the GPU 1 step that stopped
  lease 2.
- [tools/gate3-common.sh](tools/gate3-common.sh): the power-level and kernel-log checks, one step per process.

Code:
- `engine/shaders/qwen4_gate_atomic.comp`;
- `qwen4_gate_sweep.glsl` with `qwen4_gate_sweep.comp` and `qwen4_gate_sweep_nt.comp`;
- `engine/src/qwen4/bench.rs`: `gate_check` (steps `release`, `never`, `open`, `busy`; `--fence`, `--sweep-kib`,
  `--stride`, `--jitter-us`, `--min-mhz`) and `gate_isa`.

Tables:
- [summary.txt](summary.txt): every step ([tools/gate3-summary.py](tools/gate3-summary.py));
- [latency.txt](latency.txt): pooled latencies ([tools/gate3-latency.py](tools/gate3-latency.py));
- [power.txt](power.txt): [tools/gate3-power.py](tools/gate3-power.py).
- Raw JSON, one line per step, is in `alight/lease1`, `alight/lease2` and `alight/lease3`.

Reproduce, from the workspace `~/trashcan-exp/flashnext` on alight with the binary and scripts copied there:
`gpuq --on alight request ... ; ssh alight 'gpuq run ID -- bash ~/trashcan-exp/flashnext/gate3-lease3.sh'`.
`DRY=1 bash gate3-leaseN.sh` lists a lease's steps without touching the GPUs.
