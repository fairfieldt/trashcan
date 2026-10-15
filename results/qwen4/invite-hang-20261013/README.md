# invite hang, 2026-10-13 15:35

**What happened.** At 15:35:23 invite's first GPU (0000:02:00.0, behind root port 0000:00:02.0) stopped answering on
PCIe and the machine hung. It did not panic: `panic=10` would have rebooted it, pstore is empty, and the persistent
journal ends at 15:35:21. The user power-cycled it at about 15:40. Five other Macs were unaffected.

The last kernel lines, over netconsole (`netconsole-invite-1527-1536.txt`):

```
15:35:23 pcieport 0000:00:02.0: AER: Uncorrectable (Non-Fatal) error message received from 0000:00:02.0
15:35:23 pcieport 0000:00:02.0: PCIe Bus Error: severity=Uncorrectable (Non-Fatal), type=Transaction Layer
15:35:23 pcieport 0000:00:02.0:    [14] CmpltTO                (First)
15:35:23 snd_hda_intel 0000:02:00.1: Unable to change power state from D3hot to D0, device inaccessible
```

The once-a-second telemetry up to 15:35:22 (`telemetry-invite-last.txt`) was normal:
- both GPUs at 8 GT/s x16 and 56-61 °C;
- AER counters at zero;
- SMC power rising as compute started.

## Timeline on invite (`commands-1520-1540.txt`)

| time | event |
|---|---|
| 15:26:57 | decode-host agent's watchdog test (`host-stall`, `TRASHCAN_Q4_TEST_STALL=520:3000`) holds gated GPU work for 3 s |
| 15:27:21 | amdgpu job timeout on both GPUs (`ring comp_1.0.0 timeout`); PCI CONFIG reset of both, "VRAM is lost", "GPU reset(1) succeeded", "device wedged, but no recovery needed". The same reset happened on alight and enlight, and a soft one on indite. |
| 15:31:42-15:34:15 | load-time A/B, 8 six-Mac runs (main 5e48d7e and the new loader), all rc 0 on invite |
| 15:34:16, 15:34:40 | long-context agent's `trashcan-long-r21`: validate of the 14-token and the 100-token prompt, rc 0 |
| 15:35:02 | `trashcan-long-r21` validate of the 3061-token prompt starts (stages load ~20 s, then the prefill) |
| 15:35:23 | completion timeout on GPU0, hang |

amdgpu's job timeout on this kernel is 2 s (`modinfo amdgpu`: "GPU lockup timeout in ms (default: 2000"), not the
10 s that AGENTS.md said. The stall test's watchdog released its gates at 2 s and lost the race.

## Causes considered

- **The forced reset at 15:27 left GPU0 fragile.** Tahiti's reset is a PCI config reset with no soft recovery.
  The GPUs then worked for 8 runs, and alight and enlight survived the same reset.
- **The long-context build's kernels at a 3061-token prompt.** These were its first six-Mac long prompt. Earlier
  six-Mac attempts failed for other reasons (s2: leases reclaimed; s3: a script error). invite runs layers 32-39.
- **A random hardware fault.**

Not involved: GPU-to-GPU P2P. qwen4 uses none: stage buffers are host-visible system memory, and copies between a
Mac's two GPUs go through the CPU.

## After the power cycle

- Clean amdgpu init on both GPUs, links at 8 GT/s x16, no AER errors. DevSta flags are the same as on healthy incite.
- alight, enlight and indite (reset at 15:27 too) show zero AER errors since.

Health check: main's binary (5e48d7e) on invite's chain layers: stage-bench `--ranges 32:36,36:40`, decode and three
prefill chunks at 64, 3072, 8192 and 30080 positions, three rounds (`hc-invite.sh`, `hc-invite.log`). All 12 runs
rc 0 at 850/1370 MHz, at most 74 °C. No new amdgpu or PCIe error lines, and every AER counter is still zero.

The user then approved rebooting alight, enlight and indite, so that every GPU starts clean. All three came back with no
errors. `scripts/tb-reset.sh` on all six (two rounds) brought every ring link up on two lanes.

## Changes made

- No more tests that can trigger a kernel GPU reset on the shared Macs.
- The decode-host agent's GPU-side gating is held until it bounds its own wait.
- The 27B TP gated-exchange stall limit goes from 2 s to 0.5 s, in the transport branch.
- AGENTS.md corrected to the 2 s job timeout.
- `trashcan-long-r21` stays off the shared runs until it clears two checks: its per-kernel bounds review, done and clean
  (with a CPU model of every address on adversarial scores), and a single-Mac and then a six-Mac 3061-token run with
  netconsole watched.

## Second incident: invite GPU1 compute timeout, 16:05

At 16:05:46, 25 minutes after the power cycle and with no forced reset before it, invite's GPU1 (0000:06:00.0)
logged `ring comp_1.1.1 timeout, signaled seq=11368, emitted seq=11369`. A PCI CONFIG reset followed and succeeded.
This was main's binary (`trashcan-adf8cfd`) on a plain `bench --pp 512 --tg 128` in the decode-host agent's session,
on invite's node (layers 36-39). No other Mac logged anything.
- The coredump (`devcd-invite-gpu1-1605.dump`) shows the unfinished work was one ordinary submission: pipeline sync,
  three IBs (the middle one a 3064-dword layer command buffer), then the fence that never signaled. No host-gated
  wait was involved.
- Sessions that tried to start on invite after it were stopped, invite was rebooted, and Thunderbolt was reset on
  all six (two rounds).

**Diagnostic.** `tools/stress.sh` under `chain6.sh`, main 2293550, 32 runs in two passes:
- first with the chain reversed (invite holds layers 16-23, inflight 32-39), then in the usual order;
- each pass: 12 × `bench --pp 512 --tg 128`, 3 × `--pp 4096`, and the 3061-token validate;
- every Mac's kernel log checked after each run.

The first attempt (`stress1.log`) stopped at run 6 for a runner fault, not a GPU one. alight's node never launched,
and the run's time limit then killed `q4run.py` before its cleanup. That left idle nodes holding port 7071, which
spoiled the next session (the verify agent's). `q4run.py` now:
- refuses to start while any engine process or anything on port 7071 remains;
- shows and retries a failed node launch;
- stops its nodes when it is stopped.

Second attempt (`stress2.log`): **32 of 32 runs rc 0, no GPU or PCIe errors on any Mac in either order.** prefill
480-487 tok/s at 512 tokens and 760-764 at 4096; decode 19.2-19.6 tok/s.

So invite runs clean from a fresh boot, whichever layers it holds. The 16:05 timeout is unexplained: it came with
main's code and no forced reset first. Two GPU faults on one Mac in half an hour, and none on the other five all day,
make invite the first suspect if it happens again. Its GPU and PCIe logs stay watched.

## inflight: thunderbolt driver Oops during a link-reset fault test, 17:59

During the USB4STREAM agent's window 1, a fault step on inflight reset Thunderbolt by unloading and reloading the
modules. The window ran link-test only, with no GPU work. Over netconsole (`netconsole-inflight-oops-1759.txt`):
1. 14 s of `thunderbolt 0-3: failed request link state change, aborting`;
2. the module unloaded and was loaded again, then `thunderbolt 0000:16:00.0: error -ETIMEDOUT: failed to allocate
   host router`;
3. a WARNING `TX ring 0 is still active` at `nhi_shutdown` (nhi.c:1120) from modprobe;
4. `BUG: kernel NULL pointer dereference` in `tb_xdp_handle_request` (workqueue).

That is the thunderbolt driver's failed-probe path racing an XDomain request still queued. inflight stopped answering
(no ping, telemetry ends 17:59:11) and needs a manual power-off. The other five Macs were left with no stream groups
and no test processes.

Until the driver's error path is understood or fixed, Thunderbolt module reloads are not a fault to inject on the
shared Macs.

## invite GPU 1: compute ring timeout and GPU reset, 2026-10-14 13:22

The main suite for 9baedca ran on all six Macs on Linux 7.3.0-rc7-p2p, with the stream transport. During pp2048,
invite's GPU 1 (06:00.0) hung (`gpu1-timeout-20261014-1322/`: dmesg, devcoredump, the run's clock log):
- `ring comp_1.0.0 timeout, signaled seq=4123, emitted seq=4125` (process trashcan-9baedca, the node);
- the soft reset of gfx_v6_0 failed (-110);
- a PCI CONFIG reset then succeeded, with VRAM lost.

Nothing pointed to a cause beforehand:
- both GPUs were at 850/1370 MHz and 60-62 °C;
- no AER or PCIe errors appeared since boot;
- the suite's validation, pp512 and earlier runs had passed.
The `tbstream: RX CRC error` lines before it are not invite-specific: every Mac logged a few during stream traffic
today (enlight 13, invite 6, inflight 2, alight 2, indite 1) and yesterday.

The suite was stopped, and invite was rebooted under a lease, per the rule after any GPU reset. It came back healthy
on 7.3: both GPUs accept the high level, no errors. Thunderbolt was then reset on all six.

This is invite's third GPU incident in two days, and no other Mac has had one outside deliberate tests:
- GPU 0 dropped off the bus on 2026-10-13 at 15:35, 8 minutes after a forced reset;
- GPU 1 timed out on 2026-10-13 at 16:05 (7.2.9, main's code over TCP);
- GPU 1 timed out now (7.3, the stream build).
invite's GPU 0 also runs 10-15 °C hotter than the other eleven. invite was repasted on 2026-10-07.

## invite went down as a whole, 2026-10-14 18:25:10

During the main suite's pp2048 run for 4f36bf0 (7.3.0-rc7-p2p-s2), invite stopped all at once:
- netconsole logged no kernel message, and its last telemetry came at 18:25:09.8;
- both neighbours logged "host disconnected" for its cable in the same second (enlight thunderbolt 0-3, indite 1-3);
- it dropped off the tailnet and didn't come back on its own.
Shortly before, at about 18:24:50, a 50 MB binary was copied by rsync into ~/trashcan-exp/flashnext on all six,
invite fifth.
Separately, the USB4STREAM agent measured writes to the log directory stalling 0.3-1.27 s several times a minute on all
five node Macs (ext4 on the internal Apple SATA SSD; results/kernel/usb4stream-busypoll-20261014/logcheck). incite, on
an NVMe SSD, stayed under 2.2 ms. Whether either is connected to invite's failure is not known. invite was left down at
the end of the day, as the user asked.
