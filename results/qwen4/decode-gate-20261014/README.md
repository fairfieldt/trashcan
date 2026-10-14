# Decode graphs submitted ahead behind a sweep gate (2026-10-14)

**Result on one Mac (alight, Linux 7.3.0-rc7-p2p, on main 25a85140 with its light barriers).** With the gates on,
decode on the chain code (the model's first 6 layers over alight's two GPUs, 512 steps after a 100-token prompt) is
1.6-3.3% faster than with them off in all 8 alternated pairs: +2.3% and +2.9% in one process (both GPU orders), +2.1%
and +3.1% with a node and a driver on one GPU each (both ways round). A gate opens 12.1-14.0 us (p50) after the host
writes its ticket (p99 19.5-25.5 us), against 0.11-0.22 ms for the `vkQueueSubmit` it takes off the critical path.
Validation dumps, 2,048 greedy tokens and MTP decoding are bit-identical to main in every layout. No gate gave up in
38,858 releases of normal runs on the final binary and its predecessor except once (a 101 ms gap in a kill test, below).
Eight kill -9 tests with gated graphs pending, on each of two binaries, left nothing in the kernel log and both GPUs
running work.

**Incident.** At 13:33:15 alight's GPU 1 failed with a PCIe completion timeout while a bug in these tests' script had
put two engine processes on that GPU, overcommitting its VRAM ([below](#incident-alight-gpu-1-at-133315)). The scripts
now refuse that, and the engine now never lets a gate sweep memory outside VRAM.

**Six-Mac A/B: not run yet**, held until the 7.3 stream-driver patch is rolled out.

## What the branch does

A decode step on the chain pays `vkQueueSubmit` (0.11-0.22 ms on a D700) on every stage, in series, after the stage's
input arrives (results/qwen4/decode-host-20261013). This branch submits a decode step's or verification batch's graphs
(up to 8 tokens) before their input, so that the submission overlaps the previous stage's work:

- **Notices down the chain.** Before a decode step or verification batch the driver sends `{"next": {pos, n, verify}}`
  down the chain; each machine passes it on and submits its stages ahead (`Local::presubmit`). The driver submits its
  own later stages ahead right after its first stage's submit, and its head scoring before it waits for the final
  hidden states. Only when no prompt chunks are in flight (so that the forward is the next frame on every link).
- **The gate.** Each graph submitted ahead starts with `qwen4_gate`: one wave polls a word of host-cached memory (a page
  of its own; the host writes the submission's ticket there, then an mfence, once the input is written) and opens when
  it reads its ticket. On Tahiti the polled line stays in the GPU's L2 and the wave never sees the host's write
  ([gate-shader-20261013](../gate-shader-20261013/README.md)), so between polls the wave reads 8 MiB of the stage's
  first-layer expert weights, a line every 4 KiB, which evicts it ([gate-shader-20261014](../gate-shader-20261014/README.md)).
  A poll takes 13-15 us.
- **Bounds.** A gate gives up after 0.1 s at the top DPM level, counted in shader-clock ticks (GFX6's shader clock runs
  at the current level's clock: 0.28 s at the lowest level), or after 12,208 polls (0.1 s over the fastest a sweep can
  read VRAM, so the ticks end a wait unless the clock stops); amdgpu's job timeout is 2 s. After every wait the host
  checks the gate's ticks against the GPU timestamps around it (bound kept, clock 250-900 MHz); a failed check turns
  gates off for the process.
- **A gate that gives up** writes a predicate that skips every dispatch of its graph (conditional rendering, 0.01 us per
  dispatch; copies and timestamps still run), and the host runs the stage again, without a gate, once its input is
  there. Abandoned work (an error, the session ending) is cancelled through the gate word: the gate ends at its next
  poll.
- **The sweep reads VRAM only.** From memory amdgpu moved to system memory a sweep is a stream of PCIe reads (see the
  incident). The sweep reads a device-local buffer; gates are off for the process when amdgpu reports any of the
  process's VRAM evicted (`amd-evicted-vram` in the fdinfo of its render nodes, at startup and before every submission
  ahead), and a gate whose sweep takes more than 32 shader-clock ticks per line (from VRAM 5-7, from evicted memory
  about 180-700) gives up at once and turns gates off for the process.
- **Startup check per GPU** (`gate::probe`): each gate polls its own page, never released, under the 8 MiB sweep; a page
  whose polls still hit the L2 is replaced by a fresh one (up to 8 per sweep), and then the sweep grows to 16 and 32 MiB.
  Any failure (no page, a shader clock that disagrees with the timestamps, a slow sweep, an error) leaves that GPU
  without gates, with a log line, and never fails the session.
- **Light barriers** (main 25a85140). A gated graph is decode-sized, so its dispatches end with light barriers. The
  gate is not on `SCALAR_SAFE`: the full barrier the engine puts before it runs while the gate would wait anyway, and
  invalidates the L2 at the graph's start. After the gate come the predicate's barrier, the host-visibility barrier and
  the graph's own full barrier before its input copies. `qwen4_gate` (and the skip check's `qwen4_count`) pass
  `coherence_audit.py --rule coherent` of results/qwen4/decode-kernels-20261014 ([audit/](audit/)): the gate word, the
  ticket and the sweep are vector loads with glc, and nothing reads the predicate through the scalar cache
  (`qwen4_count` read it with `s_buffer_load` until this round).

`TRASHCAN_Q4_GATE=0` turns the gates off (the measurements' off arm). Test hooks: `TRASHCAN_Q4_LAYERS=N` (the first N
trunk layers, for the chain code on one Mac) and `TRASHCAN_Q4_STALL=POS:MS[:heads]` (the driver pauses after the notice
of the forward at POS, or after submitting its head scoring ahead). Each process prints its gates' statistics
(`gate: {...}`; released, open at the first poll, timeouts, abandoned, slow sweeps, release latency with
`TRASHCAN_TP_PROFILE=1`, waits, poll loads, sweep ticks).

## Tested source

Linux 7.3.0-rc7-p2p, Mesa 25.0.7 (RADV/ACO), alight only, one gpuq lease at a time; every engine process alone on its
GPU (except the incident's three tests). One Mac runs the chain code on the model's first 6 layers
(`TRASHCAN_Q4_LAYERS=6 --cuts 3 --head-split 2 --context 4096 --batch 128`): the driver's stage holds layers 0-2 and
half the head, the second stage layers 3-5, the final mixer and the other half; as one process on both GPUs, or as a
node and a driver on one GPU each over localhost.

| binary | source | sha256 | used in |
|---|---|---|---|
| `trashcan-dgbase2` | 9baedca9 + the `TRASHCAN_Q4_LAYERS` hook | dbb798d83cd6d0e1... | the A side of `correct/`, `correct-dg6/` |
| `trashcan-dg5` | gates, the 0.1 s bound, per-page startup check | 1a0fcf79e3edb022... | `correct/`, `safety-incident/` |
| `trashcan-dg6` | dg5 + the incident's engine fixes | 954f7ba828c27961... | `safety/`, `correct-dg6/` |
| `trashcan-dgbase3` | main 25a85140 + the hook | 0e73f231004c4fe3... | the A side of `correct-dg7/` |
| `trashcan-dg7` | this branch's code on 25a85140 | f89a70ad5ea70d03... | `safety-dg7/`, `correct-dg7/`, `measure-dg7/` |

Earlier runs: `skip/` (the skip check, binary `trashcan-dg1`), `correct-first/` (`trashcan-dg2`, against
`trashcan-dgbase`, 0f6233a4 + the hook), `diag/` (`trashcan-dg3`). Validation dumps are kept as SHA-256 manifests
(`dumps.sha256` in each run directory, `tools/dg-manifest.py`); identical manifests mean bit-identical dumps.

## Skip check (`skip/`, `qwen4 gate-skip-check`)

100 small dispatches behind a gate that has given up: conditional rendering skips them for 0.009 / 0.010 us each
(GPU 0 / 1; the barriers between them still run), against 0.41 / 0.47 us for a flag every dispatch reads. Alternating
open and given-up gates 400 times, the command processor saw every predicate the gate wrote (0 wrong counts on both
GPUs). A gate that finds its ticket at the first poll costs 5.9-6.1 us of GPU time with its barrier. The same check runs
on both GPUs after every kill test below (40 alternating runs each, 0 wrong counts).

## Startup check on 7.3

Under the 8 MiB sweep, the first gate page of the second GPU a process opens has its polls hit the L2 in most
processes (poll loads 204-764 ticks, some under 400), and the second page passes (all poll loads 600-10,000 ticks);
the first GPU's pages pass at once. The first correctness run (`correct-first/`, `trashcan-dg2`) checked a scratch page
instead of the gates' own: every gate of the second stage gave up (2,047 of 2,047 after 214 ms each with that build's
bound; the output stayed bit-identical through the redo), which is why the check now probes the very pages the gates
poll and replaces a page that fails. A sweep of 8 MiB takes 11,400-14,500 shader-clock ticks (5-7 per line, at
795-826 MHz), 12.4-15.2 us per poll. Every process found its pages with the 8 MiB sweep in two tries at most, and
reported none of its VRAM evicted.

## Correctness (`correct/`, `correct-dg6/`, `correct-dg7/`, `tools/dg-correct.sh`)

Each binary against main with the same hook, each run once: one process on both GPUs (the second stage submitted
ahead), both GPU orders; a node and a driver on one GPU each (the node's stage on the notice, the driver's head
scoring), both ways round; each the validation dumps of a 100-token prompt (93 tensors) and 2,048 greedy tokens; then
`mtp spec` (512 tokens, plain and with 3 drafts, which must agree) in the two-process chain, both ways round.

| binary | validation dumps and 2,048 greedy tokens | MTP ids | gates released / given up |
|---|---|---|---|
| `trashcan-dg5` | identical in all 4 layouts | identical (2 layouts) | 14,326 / 0 |
| `trashcan-dg6` | identical in all 4 layouts | identical (2 layouts) | 14,326 / 0 |
| `trashcan-dg7` | identical in all 4 layouts | identical (2 layouts) | 14,326 / 0 |

(With 6 layers the model's tokens are not meaningful and no draft is accepted, so MTP decoding verifies every token in
batches of 4: the verification path runs gated too.)

## Safety (`safety/` with dg6, `safety-dg7/` with dg7, `tools/dg-safety.sh`)

A node and a driver on one GPU each, both ways round; the kernel log (amdgpu faults and PCIe AER) checked after every
step, the free VRAM of a GPU checked before every process.

- **Gates that give up**: the driver pauses 400 ms after the notice of the forward at 120 (the node's gate gives up and
  its stage runs again without a gate), or after submitting its head scoring ahead at 140 (its own head gate gives up).
  Each gate gave up after 101 ms (7,107-7,757 polls), and the 64 greedy tokens are the same as without the pause, on
  both binaries and both layouts.
- **kill -9 with gated graphs pending**: the node 0.05 s into a 3 s pause after the notice (its stage waiting behind
  its gate), the driver 0.05 s into a 3 s pause after submitting its head scoring ahead, then the node and the driver at
  a random moment of a long decode (8.5-11.6 s after the start), on each layout: 8 kills per binary, 16 in all. After
  each, both GPUs ran the skip check (16 of 16 per binary with 0 wrong counts), and the kernel log stayed at its
  boot baseline (11 lines, AER's setup among them) throughout.

One gate gave up in a normal decode: in dg6's `kill-random-driver-0` the node waited more than 101 ms for the forward
at 170 (the others of that run: p50 5.6 ms, max 32 ms; 561 released). Nothing on the GPUs explains it; the journal shows
six DPM level writes during that run, by the two processes' clock holders (each process sets both GPUs of the Mac: a
process that starts sets `auto`, one that goes idle for a second sets `auto`, one that gets work sets `high`), which
only happens with two engine processes on one Mac. The first run of `safety/` ran with 12 of its 41 clock samples
below 850 MHz for the same reason. No gate gave up outside the stall tests in dg7's runs (2,402 released in its plain
and kill runs, 14,326 in correctness, 6,144 in the measurements).

## Measurements (`measure-dg7/`, `tools/dg-measure.sh`, `tools/dg-bench.py`)

`bench --pp 100 --tg 512` with `TRASHCAN_TP_PROFILE=1` (calibrated timestamps), gates on and off
(`TRASHCAN_Q4_GATE=0`) alternated, two rounds per layout; every clock sample of every run at 850/1370 MHz.

| layout | round 1 on / off (tok/s) | round 2 on / off | gain |
|---|---|---|---|
| one process, GPUs 0,1 | 124.05 / 122.05 | 124.73 / 121.24 | +1.64%, +2.88% |
| one process, GPUs 1,0 | 125.23 / 121.91 | 125.03 / 121.36 | +2.72%, +3.03% |
| node on GPU 1, driver on GPU 0 | 120.37 / 118.13 | 121.30 / 118.64 | +1.90%, +2.24% |
| node on GPU 0, driver on GPU 1 | 120.80 / 116.99 | 121.29 / 117.77 | +3.26%, +2.99% |

About 0.15-0.25 ms per token: one gated graph per token in one process (the second stage), two with a node (its stage
and the driver's head scoring). Release latency, from the host's write of the ticket to the GPU timestamp past the
gate: p50 12.1-14.0 us, p90 17.6-19.5 us, p99 19.5-25.5 us; a gate waited 2.2-2.8 ms (p50) for its input. No gate gave
up (6,144 released). Power while gates wait: +3.4 W per waiting GPU on round 3's measurement
([gate-shader-20261014](../gate-shader-20261014/README.md)); the six-Mac A/B measures it in the decode window.

## Incident: alight GPU 1 at 13:33:15

**What ran.** Lease #119 on alight: `dg-correct.sh` with `trashcan-dg5` (13:23-13:30, every comparison identical), then
`dg-safety.sh` from 13:32:27. The stall tests and the first kill test (`kill-node-0`: node on GPU 1 killed 0.05 s into
its gate's wait, driver on GPU 0) passed their checks. Then:

- **The script's bug.** `killtest()` ran its after-kill GPU checks in a loop over the global `g` that the outer loop
  used to pick the GPUs: after the first kill test `g` was 1, so `kill-driver-1`, `kill-random-node-1` and
  `kill-random-driver-1` started the node and the driver both with `--gpus 1` (the intended GPU 0 tests never ran).
  Each process loads 4,634-4,664 MiB; together they overcommitted GPU 1's 5,888 MiB by about 3.4 GB, and RADV's budget
  said so (`device-local heap 4634 of 5888 MiB used, budget 4634`: no room beyond the process's own memory).
- **13:32:49.96 `kill-driver-1`**: both on GPU 1. The driver's startup check swept evicted weights (1.76 ms per poll),
  so its poll bound came out at 58 polls; its head gates gave up after 5, 2 and 2 ms, then a gate's clock check failed
  (211 MHz) and gates went off before the stall. The driver was killed at 13:32:55.5 with no gated work pending; the GPU
  checks after it passed and the kernel log stayed at its baseline.
- **13:32:58.84 `kill-random-node-1`**: both on GPU 1. The node's startup check sweep read evicted weights (440 us per
  poll), its check ran out of time, and that error ended the node's session (13:33:01.5); the driver exited. The node,
  idle, was killed at 13:33:10.54; the GPU checks after it passed (13:33:13.5-13:33:13.85).
- **13:33:13.88 `kill-random-driver-1`**: both on GPU 1 again, both loading their weights at once (the driver
  allocated its 4,664 MiB in 1.40 s, the node its 4,634 MiB in 2.39 s), then their gate checks (the driver's: 1.75 ms
  per poll, evicted weights again). **13:33:15** (by the kernel log, 2692.6 s after boot):

```
pcieport 0000:00:03.0: AER: Uncorrectable (Non-Fatal) Error message received from 0000:00:03.0
pcieport 0000:00:03.0: PCIe Bus Error: severity=Uncorrectable (Non-Fatal)
pcieport 0000:00:03.0:   device [8086:0e08] error status/mask=00004000/00000000
pcieport 0000:00:03.0:   [14] CmpltTO           | Requester   | Transaction Layer (First)
amdgpu 0000:06:00.0: PCI error: detected callback!!
amdgpu 0000:06:00.0: pci_channel_io_normal: state(1)!!
snd_hda_intel 0000:06:00.1: AER: can't recover (no error_detected callback)
amdgpu 0000:06:00.0: PCI error: detected callback!!
amdgpu 0000:06:00.0: pci_channel_io_perm_failure: state(3)!!
pcieport 0000:00:03.0: AER: device recovery failed
pcieport 0000:00:03.0: AER: Correctable Error message received from 0000:00:03.0
pcieport 0000:00:03.0: PCIe Bus Error: severity=Correctable
pcieport 0000:00:03.0:   device [8086:0e08] error status/mask=00002000/00000000
pcieport 0000:00:03.0:   [13] NonFatalErr       |             |
pcieport 0000:00:03.0: PCIe Bus Error: severity=Uncorrectable (Non-Fatal)
pcieport 0000:00:03.0:   device [8086:0e08] error status/mask=00010000/00000000
pcieport 0000:00:03.0:   [16] UnxCmplt          | Receiver    | Transaction Layer
pcieport 0000:00:03.0: AER:   TLP Header: 0x00000000 0x00000000 0x00000000 0x00000000
```

  Root port 00:03.0 sits above GPU 1 (06:00.0, card1; GPU 0 is 02:00.0 under 00:02.0). A request the root port sent
  toward GPU 1 (CPU side: a register or BAR access) timed out; the unexpected completion that followed is most likely
  its late answer. GPU 1 was left in permanent failure (AER recovery failed). The driver went on (its head gates gave up
  twice, then a clock check failed at 248 MHz and gates went off, about 13:33:17.4) until it was killed at 13:33:25.59;
  the script stopped at its kernel-log check, and alight was rebooted at 13:37 (by the coordinator, under the lease).
- **No gated decode graph was pending at 13:33:15**: both processes were loading their weights or running their
  startup checks, whose gates are never released and end within 24 polls or 0.05 s (with the weights evicted, the
  driver's check swept system memory for about 0.08 s in that second or the next). The journal shows DPM level writes by
  the two processes' clock holders at 13:33:14 (two), 13:33:15 (one) and 13:33:16 (three); each writes both GPUs'
  levels. Sweeps of evicted weights had also run in the two tests before (startup checks at about 13:32:51 and
  13:33:01, and the driver's head gates at about 13:32:52) without an AER line.

**Cause: not determined.** Candidates: the overcommit itself (two processes loading 9.3 GB into 5.9 GB, with amdgpu
evicting to system memory), the gates' sweeps of evicted weights (each poll then 2,048 reads over PCIe), DPM level
changes under that load, or an unrelated fault like invite GPU 1's ring timeout at 13:22 (no AER there). Every part of
it ran under this branch's tests, with the scripts' bug.

**Changes.**
- Scripts (`tools/dg-common.sh`, `dg-safety.sh`, `dg-correct.sh`, `dg-measure.sh`): the loop variable is local; a
  test that would put the node and the driver on one GPU stops the script; a process starts only when its GPU has the
  VRAM it loads free (4,800 MiB, amdgpu's counters, after waiting up to 10 s for a killed process to free its own); the
  kernel-log check matches AER lines (`AER|PCIe Bus Error|pci_channel_io|UnxCmplt` besides amdgpu faults, `CmpltTO`,
  BUG and Oops); a gate check that times its sweep outside 2-20 ticks per line stops the script; no engine process may
  outlive a script.
- Engine: the startup check never fails a session (a failed or slow check leaves the GPU without gates, logged); the
  poll bound comes from the clock bound (no calibration a slow sweep can shrink); the sweep must be device-local, gates
  go off for the process when amdgpu reports any of its VRAM evicted (startup and before every submission ahead) or a
  sweep is slower than VRAM (the gate gives up at once, so at most one sweep of evicted memory runs); a gate that gives
  up is redone without a gate.
- Rerun with one process per GPU (lease #124, dg6; lease #125, dg7 on 25a85140): all stall and kill tests passed.

Evidence: [alight/incident/](alight/incident/) (`dmesg-alight.txt`, the kernel log after the event; `procs-journal.txt`,
processes and journal around it; `devcoredump.txt`, empty: none was written), [alight/safety-incident/](alight/safety-incident/)
(every log of the stopped run and its `lease.log`).

An earlier slip in this task: on stopping the first correctness run, `pkill -f dg-correct.sh` ended the script's ssh
session but not its engine process (`trashcan-dg2`), which ran about a minute past the lease's release and into the
start of the next lease on alight (#115) by a few seconds before it was killed; the kernel log showed nothing new. The
scripts now kill their engine processes by name before they end.

## Six-Mac A/B

Not run yet (held until the 7.3 stream-driver patch is out). Plan: `chain6.sh` against `trashcan-25a8514`: validation,
decode after 512 tokens and MTP chat in alternated pairs (two pairs if a difference is under 3%), the decode window's
PSTR power; a pair counts only if both arms used the same transport on every link (the nodes' "stream up" lines), and
gate timeouts must be 0.

## Files

- `alight/*/lease.log`: each lease's log; `*/clocks/`: the clocks of every step (`scripts/with-clocks.sh`).
- `tools/`: the lease scripts (`dg-common.sh`, `dg-skip.sh`, `dg-correct.sh`, `dg-safety.sh`, `dg-measure.sh`,
  `dg-diag.sh`), `dg-compare.py` (A against B), `dg-gates.py` (gate statistics), `dg-bench.py` (the measurements),
  `dg-manifest.py` (dump manifests).
- `audit/`: `coherence_audit.py` on RADV's Tahiti programs of `qwen4_gate` and `qwen4_count`, before and after the
  latter's fix.
