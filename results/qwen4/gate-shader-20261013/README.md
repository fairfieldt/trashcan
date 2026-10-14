# A gate that bounds itself on the GPU: single-GPU checks (2026-10-13)

**Result: no variant released.** On a D700 (Tahiti, RADV, Mesa 25.0.7, kernel 7.2.9-p2p), a shader lane polling a
word of host memory never saw the host write that word while it polled. Four variants were tried:
- volatile (`glc`) loads on the engine's host-cached, coherent memory;
- `atomicOr(word, 0)` on the same memory;
- volatile loads on a host-visible, coherent, non-cached memory type (RADV's write-combined GTT);
- nontemporal loads (Vulkan memory model `Nontemporal`; in ACO, SLC on GFX6) on host-cached memory.

So there is no release latency to report. Every lane ran to its bound, 0.10 s in the release-always runs.

Each poll took 0.40-0.56 µs in every variant. That is the speed of an L2 hit, not of a PCIe round trip (~1-2 µs). The
line stays in the GPU's L2 and the CPU's store never reaches the lane.

The bounds themselves work. Every submission ended by its shader-clock bound or its poll bound:
- never-released runs ended at 49.8 ms (bound 50 ms) and 99.6 ms (bound 100 ms);
- `s_memtime` counts 838.7-838.9 M ticks per second on both GPUs, at the `auto` and `high` levels;
- no amdgpu timeout or reset on alight in either round.

This is the follow-up to the held-back gating of [results/qwen4/decode-host-20261013](../decode-host-20261013/README.md).
There, graphs submitted ahead of their input wait on the command processor for a host event (`vkCmdWaitEvents`),
which takes ~2 ms of `vkQueueSubmit` per token off the critical path. But the wait has no bound: amdgpu resets a GPU
whose job waits 2 s, and a crashed process leaves its gates armed. The idea tested here was a one-lane gate shader in
front of the graph instead. It polls the gate word and gives up after a fixed number of polls or shader-clock ticks,
so the GPU would end the wait itself whatever the host does.

## What ran

Code: branch `host-gate-shader-20261013`.
- `engine/shaders/qwen4_gate.comp`: volatile loads or `atomicOr`, poll bound.
- `qwen4_gate_clock.comp`: plus a shader-clock bound, `VK_KHR_shader_clock`.
- `qwen4_gate_nt.comp`: nontemporal loads under the Vulkan memory model.
- `qwen4 gate-check` (`engine/src/qwen4/bench.rs`) records a command buffer like a gated stage's: a timestamp, an
  event the command processor sets just before the gate, the gate dispatch, a timestamp after it, the barriers a stage
  needs before reading host input, a timestamp and an end event.
- The host waits for the first event, waits 200 µs, writes 1 to the gate word (then a full fence), and spins on the
  end event. Word 1 tells how the lane ended: 1 opened, 2 out of polls, 3 out of time.

Each round ran on alight under a one-Mac gpuq lease (`tools/`). Both GPUs held 850/1370 MHz (clock logs in
`alight/clocks/`).

**Round 1** (`round1.log`, `alight/host-gate1-*.json`; GPU 0 at `auto` and held `high`, GPU 1 at `auto`). It ran
never-released first, against the order asked for later; nothing came near 2 s.

| | volatile loads | `atomicOr` |
|---|---|---|
| never released, 100 / 1k / 10k / 100k polls | 0.40-0.49 µs per poll, 0.05-44 ms | 0.48-0.56 µs per poll, 0.05-51 ms |
| released 200 µs after the gate started (bound 0.25 s) | not seen: ran to the poll bound, 216-225 ms | not seen: 222-232 ms |

Also measured in round 1:
- Clock bound: 2^20 ticks ended at 1.25 ms; the 0.25 s bound at 249.1 ms.
- An open gate (word already 1): +2.4 µs of GPU time per graph (4.44 against 2.07 µs). From the gate's start to the
  host seeing the end it took 5.7 µs against 3.3 µs without the gate.

**Round 2** (`round2.log`, `alight/host-gate2-alight.json`; GPU 0). Done as asked:
- both GPUs' power level first: each accepted `high`;
- release-always first, then never released;
- bounds 50-100 ms;
- the kernel log checked after every step: 0 amdgpu timeouts or resets throughout.

| variant | release-always (200 runs, bound 100 ms) | never released, 50 ms | never released, 100 ms |
|---|---|---|---|
| (b) non-cached host memory (`HOST_VISIBLE \| HOST_COHERENT`), volatile loads | the first run did not see the write: 99.6 ms, 185,440 polls (0.54 µs each) | 49.8 ms, clock bound | 99.6 ms, clock bound |
| (a) nontemporal loads, host-cached memory | the first run did not see the write: 99.6 ms, 225,320 polls (0.44 µs each) | 49.8 ms, clock bound | 99.6 ms, clock bound |

## What is next for the gated design

- **The command processor sees host writes** (`vkCmdWaitEvents` released within 5-13 µs in the six-Mac traces), but
  it cannot bound its own wait on GFX6: `WAIT_REG_MEM` has no timeout, and SI has no soft recovery. A shader can bound
  its wait, but it cannot see host writes. So on these GPUs a self-bounding GPU-side gate is not available through
  Vulkan.
- **Untried variants** with little hope, given that every poll ran at L2-hit speed:
  - nontemporal loads on the non-cached memory together;
  - device-local host-visible (BAR) memory.
  
  It may also be worth checking ACO's disassembly for the SLC bit, to rule out the compiler dropping it.
- **A kernel-side gate**: the submission waits on a semaphore holding a fence the host can signal (a vgem fence;
  `CONFIG_DRM_VGEM=m` on 7.2.9-p2p, the module not loaded). The job then stays in the kernel's scheduler and never
  reaches the GPU before its input. It holds no ring and runs no timeout, and a dead process's queued jobs are dropped
  with its context. That needs the vgem module loaded on the Macs, a system change for the user to decide.
- **Without any gate**: make `vkQueueSubmit` itself cheaper (0.13-0.22 ms per stage on the critical path; profile the
  CS ioctl against RADV's per-submit work on GFX6). Or keep the event gate with crash-signal handlers (SIGSEGV,
  SIGABRT, SIGBUS set the gates) and accept the SIGKILL exposure. That is a policy decision; today's rule is no
  unbounded host-event waits.

The gated design itself (notices, graphs submitted ahead, the 1 s watchdog with its unit tests) is on branch
`host-gating-20261013`.
