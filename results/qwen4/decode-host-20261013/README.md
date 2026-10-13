# Flash-Next decode: the host path between the GPU stages (2026-10-13)

**Result.** (filled in after session 3)

## The timeline of a decode step on main

`TRASHCAN_Q4_TRACE=PREFIX` (given to every process of the chain; `engine/src/qwen4/trace.rs`) records host
CLOCK_MONOTONIC timestamps at every step of a forward: each frame received (first bytes, whole frame) and sent, the
reader thread's handoff, every stage's submission, fence wakeup and output copy, the GPU's own start and end of each
stage (the stage's first and last timestamp queries, moved to host time with `VK_EXT_calibrated_timestamps` read at
each collect), and the head scoring. Each process writes `PREFIX-HOST.json` when its session (node) or bench (driver)
ends. Before and after the decode steps the driver measures every machine's clock offset along the chain: 64
ping-pongs per link, the fastest kept (round trips 47-111 µs), offsets chained link by link and interpolated linearly
in between. `tools/timeline.py` moves every event to the driver's clock, cuts the run into decode steps at the
driver's `fwd` marks and follows the critical path through the chain. Tracing changes nothing measurable (session 1:
18.12 tok/s traced against 18.07 untraced and 18.09/18.13 for main).

Main 4c1346d, six Macs, `qwen4 bench --pp 512 --tg 128` on real text, 126 decode steps (session 1,
[timeline-base.txt](timeline-base.txt)): 55.14 ms per token, of which 48.81 ms are the 12 stages' GPU work and
**6.33 ms** everything else:

| | ms per token | |
|---|---|---|
| GPU work of the 12 layer stages | 48.81 | 3.98-4.22 ms per stage (4 layers) |
| stage submission to GPU start | 2.44 | `vkQueueSubmit` of a stage's decode command buffer takes 0.11-0.19 ms; the GPU then starts within ~10 µs |
| head scoring at the end | 1.37 | the driver's own two slices 0.98 ms (h in to candidates), then 0.26-0.35 ms waiting for the others' candidates, merge 0.04 |
| residual hops | 0.58 | 5 hops, 0.09-0.13 ms each for 40 KiB over TCP on Thunderbolt (send returned to whole frame read) |
| embed + n-gram rows (driver) | 0.55 | 16 threads spawned for 16 cached rows every step |
| final hidden state back up | 0.50 | 5 hops of 10 KiB, 0.064-0.077 ms each, plus 0.017-0.026 ms relaying on each node |
| host copies around stages | 0.29 | input writes, timestamp reads, output copies (40 KiB, read twice into new vectors) |
| sending the residual | 0.27 | 0.03-0.05 ms per machine (JSON header, BufWriter copy, write) |
| GPU end to fence wakeup | 0.22 | 0.012-0.018 ms per stage |
| reader thread to session thread | 0.10 | 0.011-0.020 ms per node |

Off the critical path: every Mac's two head slices take 0.96-1.03 ms from h to candidates (submission 30-53 µs, then
0.86-0.92 ms to the fence: GPU work); every node spends ~0.1 ms parsing, merging and serializing the JSON candidate
frame, so the candidates fall behind the final hidden state by ~0.07 ms per hop and arrive at the driver 0.26-0.35 ms
after its own scoring.

## Frames on the decode critical path

Frames are `[u32 header length][u64 blob length][JSON header][blob]` (`write_frame`/`read_frame` in `net.rs`, over
any `Write`/`Read`; `Conn` holds the transport, a TCP socket today). Per decode step of `n` tokens (1 for plain
decoding, up to 4 for MTP verification), on each of the five links of the chain:

| frame | direction | header | blob | on the critical path |
|---|---|---|---|---|
| `{"next": {pos, n[, verify]}}` | down | ~40 B | none | no: it precedes the residual by 8-45 ms |
| `{"forward": {pos, n, reply, dump}}` | down | ~60 B | the residual, n x 40,960 B | yes, the 5 hops in a row |
| `{"h": len}` | up | ~15 B | final hidden state, n x 10,240 B; verification adds the residual rows (n x 40,960 B) | yes, the 5 hops in a row |
| `{"top": 64, "captures": []}` (`{"tops": [64, ...]}`) | up | ~30 B | candidates, n x 64 x 8 B | only when they arrive after the driver's own head scoring |

## The changes

**a. Stage graphs and head scoring submitted ahead, gated on host events** (`gpu.rs`, `pipeline.rs`, `gate.rs`).
Every decode-sized graph (up to 8 tokens: decode steps and verification batches, one command buffer each) now starts
with a wait for a host-set event (`start`; RADV: a command-processor poll of host memory) and ends by setting a done
event, which the host polls instead of sleeping on the fence (the fence is waited for lazily, before the command
buffer or fence is reused). Without a notice the host sets `start` before submitting, so nothing changes; with one
the submission happens early, off the critical path, and `submit` only writes the input and sets the event:

- the later stage of a machine is submitted while the earlier one runs (`Local::forward_gated`);
- the driver sends a `{"next": {pos, n}}` notice down the chain at the start of a decode step (or verification
  batch); every machine passes it on and submits all its stages (the machine that ends the model also its head
  scoring, queued behind its last stage) before the residual arrives, 8-45 ms later;
- the driver and every node submit their head scoring once their residual has gone down, before the final hidden
  state returns.

Bounds (amdgpu resets a GPU whose job, waiting included, runs past ~10 s): work is submitted ahead only for decode-sized
steps whose input is on its way; a watchdog thread (`gate.rs`) releases any gate armed for more than 2 s, after which
the stage refuses to continue (its session fails cleanly, no GPU reset); SIGINT, SIGTERM and SIGHUP set every gate
before the process ends (the 27B's `qwen::gpu::register_gate`); `Drop` and every error path (`abandon`) release
pending gates and wait for the device; notices are passed on only by machines with no prompt chunks in flight, so a
notice never overtakes the chunks before it (the server test caught that: a 22-token prompt's 16-token chunk arrived
after the notice for its 6-token last chunk); a forward that does not match the announced one must start a new
sequence (position 0, which resets every recurrent state) and lets the announced work run on stale input first.

**b. Binary candidate frames, fewer copies, a cheaper n-gram gather** (`net.rs`, `ngram.rs`). Candidates travel as
(score, id) pairs of little-endian words instead of JSON (each node spent ~0.1 ms on JSON per token, and the result
chain fell behind the final hidden state); frame blobs are read straight into `Vec<f32>` (no conversion copy) and sent
with one vectored write from the caller's memory (no `BufWriter` copy); a stage's output is read once. A decode step's
16 n-gram rows (up to 64 rows) are read on the calling thread with `preadv2(RWF_NOWAIT)` when they are in the page
cache; only rows that need the disk go to threads (spawning 16 threads per step cost ~0.5 ms).

**c. The head's candidate kernels** (`qwen4_head_topk.comp`, `qwen4_head_merge.comp`). qwen35's `topk` and `merge`
pick 64 candidates in 64 sequential rounds of workgroup reductions each. qwen4's copies sort each chunk of 1024
logits with a bitonic network in shared memory (55 steps) and merge the chunks' sorted lists pairwise (a bitonic merge
of two sorted lists keeps the best 64 in seven steps, 16 lists at a time). Same candidates bit for bit (scores, ids,
order: higher score first, ties to the lower id); qwen35's kernels are untouched. `qwen4 head-topk-check` checks the
pair against qwen35's kernels and a CPU sort on random logits with many exact ties (and -0/+0) at every head shape
of the six-Mac split, a third, the whole vocabulary and a partial chunk, 1, 2 and 4 rows. On both of incite's D700s
(one-Mac lease, [incite/host-topk-check-0.json](incite/host-topk-check-0.json), `-1.json`) every candidate of every
shape matched, and the selection got faster at every shape (ms per call, mean of 200 back to back, GPU 0; GPU 1 within
0.016 ms):

| head rows | chunks | `topk` + `merge`, 1 / 2 / 4 rows | `qwen4_head_topk` + `qwen4_head_merge` |
|---|---|---|---|
| 21,504 (a twelfth: every stage but the last) | 21 | 0.509 / 0.510 / 0.505 | 0.133 / 0.136 / 0.146 |
| 11,776 (the last stage) | 12 | 0.507 / 0.527 / 0.509 | 0.100 / 0.100 / 0.104 |
| 82,944 (a third) | 81 | 0.513 / 0.553 / 0.777 | 0.279 / 0.320 / 0.400 |
| 248,320 (the whole vocabulary) | 243 | 0.774 / 1.017 / 1.542 | 0.699 / 0.821 / 1.047 |
| 1,000 | 1 | 0.514 / 0.511 / 0.511 | 0.092 / 0.092 / 0.091 |

**d. Spinning on a decode step's expected frames, one copy between the GPUs.** After a notice, the node's reader
thread watches the socket and the session thread the channel instead of sleeping (each wakeup cost 10-35 µs), and the
replies a machine waits for (the final hidden state, the candidates) are watched for, each for at most 100 ms before
sleeping. The first GPU's output goes into the second GPU's input straight between the two host buffers.

**e. CPUs held at the `performance` governor while work arrives** (`clocks.rs`, approved by the user for the
engine's busy periods). Between bursts of host work schedutil lets the package fall to 1.2 GHz (Ivy Bridge-EP shares
one frequency per package), and the E5-2697 v2 nodes ran their host work 2-3x slower than indite's E5-1650 v2. The
GPU clock holder now also writes `performance` to every cpufreq policy when work arrives and restores each policy's
governor as found after 1 s without work (the same timeout as the GPU level) and on `Drop`; a root helper started with
the first clocks of a process (its own session) restores both the GPU levels and the governors when the process ends
however it ends (it polls for the process twice a second); a failed write is reported once and the CPUs are left
alone.

## Tested source

## Commands

## Correctness

## Speed

## Power

## MTP

## Robustness

## What is left

- **USB4STREAM for a decode step's frames** (not built: it would put stream traffic on the six-Mac ring while other
  agents depend on it; the 27B's TP8 ran streams on all six cables among incite, invite, enlight and indite,
  results/qwen/tp8-mesh-20261009, so the new exposure is mostly inflight and alight). The 27B's host-only exchanges
  (results/qwen/tp4-viability-20261006) put a 20 KiB exchange at 77-116 µs over TCP and 34 µs over one stream. Design:
  TCP keeps every control frame and prefill; a per-link stream (`scripts/tbstream-up.sh` for the five chain links,
  node options `--stream-up`/`--stream-down`, the driver's `--stream-down`) carries the frames of a step that a notice
  announced (the residual after the notice, then the final hidden state and the candidates back up), all far below a
  stream's 1 MiB receive ring (it has no Tx flow control), each read with a 2 s deadline; a greet handshake per session
  (`link.rs`), and TCP whenever a side has no stream.
