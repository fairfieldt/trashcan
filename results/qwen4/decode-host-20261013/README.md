# Flash-Next decode: the host path between the GPU stages (2026-10-13)

**Result.** The merge candidate (branch `worktree-agent-a6c9aa0624bfab1e3`) cuts a decode step's host work from
6.33 ms per token (main 4c1346d, traced) to 4.52 ms. It was measured as commit a480b086 on main eb4dbeb (session 7).
The branch's engine commit is now 30ad868d on main 9f54fb80, ported onto main e9f34e61's two-slot stages. It was
built and checked as b15bf062 on e9f34e61, the same engine (9f54fb80 added only results). Session 8: validation bit
for bit, one decode pair 20.72 -> 21.08 tok/s, MTP ids identical.
- Six Macs, two alternated pairs against main eb4dbeb: decode 19.22 -> 19.76 tok/s (+2.8%, -1.44 ms per token).
- MTP chat replies 25.70 -> 26.62 tok/s (one pass each).
- Those absolute numbers are low for both sides: invite's GPU0 was latched in amdgpu's thermal state at 450 MHz
  after overheating in an earlier run, and its stage paced every step. The comparison is fair. Unthrottled, in
  session 6 (main 2293550, the candidate before the rebase): 19.30 -> 19.79 tok/s decode, MTP 23.35 -> 23.56.
- Greedy tokens and logits are main's, bit for bit, on the reference prompt and p100. MTP ids are those of plain
  greedy decoding, and the server checks pass.

What changed:
- bitonic top-k for the head slices (0.51 -> 0.13 ms per slice);
- binary candidate frames;
- n-gram rows read inline;
- polled stage ends;
- one copy between a Mac's GPUs;
- spinning on expected frames.

Nothing waits on the GPU for the host.

Two measured changes are held back:
- Graphs submitted ahead of their input and gated on host events: +4.9% decode alone, and about 2 ms per token (+4%)
  on top of the candidate. They wait until the GPU can bound its own wait, because amdgpu resets a GPU after 2 s and
  a process that dies mid-decode would leave its gates armed.
- Holding the CPUs at the `performance` governor: +1.1% decode for +195 W (+14%) on six Macs.

## The timeline of a decode step on main

`TRASHCAN_Q4_TRACE=PREFIX`, given to every process of the chain (`engine/src/qwen4/trace.rs`), records host
CLOCK_MONOTONIC timestamps at every step of a forward:
- each frame received (first bytes, whole frame) and sent, and the reader thread's handoff;
- every stage's submission, fence wakeup and output copy;
- the GPU's own start and end of each stage: the stage's first and last timestamp queries, moved to host time with
  `VK_EXT_calibrated_timestamps` read at each collect;
- the head scoring.

Each process writes `PREFIX-HOST.json` when its session (node) or bench (driver) ends. Before and after the decode
steps the driver measures every machine's clock offset along the chain: 64 ping-pongs per link, the fastest kept
(round trips 47-111 µs), offsets chained link by link and interpolated linearly in between. `tools/timeline.py`
moves every event to the driver's clock, cuts the run into decode steps at the driver's `fwd` marks and follows the
critical path through the chain. Tracing changes nothing measurable: in session 1 the traced run made 18.12 tok/s
against 18.07 untraced and 18.09/18.13 for main.

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

Off the critical path, every Mac's two head slices take 0.96-1.03 ms from h to candidates: submission 30-53 µs, then
0.86-0.92 ms of GPU work to the fence. Every node spends ~0.1 ms parsing, merging and serializing the JSON candidate
frame. So the candidates fall behind the final hidden state by ~0.07 ms per hop and reach the driver 0.26-0.35 ms
after its own scoring.

The same accounting for the merge candidate g3 (session 7, main eb4dbeb, traced run,
[timeline-g3.txt](timeline-g3.txt)) and for f4, the candidate's changes plus the held-back gating and CPU hold
(session 3, main 4307169, [timeline-f4.txt](timeline-f4.txt)). GPU work differs with main's kernels: 48.81 ms on
4c1346d, 46.0 ms since. In session 7 one GPU (invite GPU0) ran at 450 MHz, so its stage took 5.94 ms against
3.54-3.70 ms for the others (see Speed). Host time per token:

| | main 4c1346d | g3 | f4 (held back) |
|---|---|---|---|
| step | 55.14 | 50.51 | 48.03 |
| GPU work of the 12 layer stages | 48.81 | 45.99 | 46.03 |
| **everything else** | **6.33** | **4.52** | **2.00** |
| stage submission to GPU start | 2.44 | 2.24 | 0.25 |
| head scoring: the driver's slices on the GPU | (in the next row) | 0.50 | 0.49 |
| head scoring, host side: submission, waiting for the others' candidates, merge | 1.37 | 0.42 | 0.04 |
| residual hops (wire) | 0.58 | 0.51 | 0.43 |
| final hidden state back up | 0.50 | 0.41 | 0.39 |
| sending the residual | 0.27 | 0.21 | 0.17 |
| host copies around stages | 0.29 | 0.14 | 0.12 |
| embed + n-gram rows (driver) | 0.55 | 0.04 | 0.06 |
| reader thread to session thread | 0.10 | 0.04 | 0.03 |
| GPU end to host wakeup | 0.22 | 0.02 | 0.02 |

In g3 the driver's head work is 0.6 ms from h in to candidates:
- 0.10 ms to submit its slices;
- 0.37 ms of logits, then 0.13 ms of top-k on the GPU (0.51 ms with qwen35's kernels, from the kernel check);
- 0.30 ms waiting for the others' candidates, which start only when h reaches each Mac.

With the gating (f4) every Mac's head scoring is submitted before h arrives, so the others' candidates arrive with
the driver's own. What remains in g3 is mostly `vkQueueSubmit` (0.13-0.22 ms for each of the 12 stages) and the
hops.

## Frames on the decode critical path

Frames are `[u32 header length][u64 blob length][JSON header][blob]`. `write_frame` and `read_frame` in `net.rs`
work over any `Write`/`Read`; `Conn` holds the transport, a TCP socket today. Per decode step of `n` tokens (1 for
plain decoding, up to 4 for MTP verification), on each of the five links of the chain:

| frame | direction | header | blob | on the critical path |
|---|---|---|---|---|
| `{"forward": {pos, n, reply, dump}}` | down | ~60 B | the residual, n x 40,960 B | yes, the 5 hops in a row |
| `{"h": len}` | up | ~15 B | final hidden state, n x 10,240 B; verification adds the residual rows (n x 40,960 B) | yes, the 5 hops in a row |
| `{"top": 64, "captures": []}` (`{"tops": [64, ...]}`) | up | ~30 B | candidates, n x 64 x 8 B | only when they arrive after the driver's own head scoring |

## The merge candidate

Nothing in it waits on the GPU for the host: every graph is submitted when its input is written, as on main.

**Head candidate kernels** (`qwen4_head_topk.comp`, `qwen4_head_merge.comp`). qwen35's `topk` and `merge` pick 64
candidates in 64 sequential rounds of workgroup reductions each. qwen4's own copies work differently:
- each chunk of 1024 logits is sorted with a bitonic network in shared memory (55 steps);
- the chunks' sorted lists are merged pairwise: a bitonic merge of two sorted lists keeps the best 64 in seven
  steps, 16 lists at a time.

The candidates are the same bit for bit (scores, ids, order: higher score first, ties to the lower id), and qwen35's
kernels are untouched. `qwen4 head-topk-check` checks the pair against qwen35's kernels and a CPU sort on random
logits with many exact ties (and -0/+0). It covers every head shape of the six-Mac split, a third, the whole
vocabulary and a partial chunk, at 1, 2 and 4 rows. On both of incite's D700s (one-Mac lease,
[incite/host-topk-check-0.json](incite/host-topk-check-0.json), `-1.json`) every candidate of every shape matched,
and the selection got faster at every shape (ms per call, mean of 200 back to back, GPU 0; GPU 1 within 0.016 ms):

| head rows | chunks | `topk` + `merge`, 1 / 2 / 4 rows | `qwen4_head_topk` + `qwen4_head_merge` |
|---|---|---|---|
| 21,504 (a twelfth: every stage but the last) | 21 | 0.509 / 0.510 / 0.505 | 0.133 / 0.136 / 0.146 |
| 11,776 (the last stage) | 12 | 0.507 / 0.527 / 0.509 | 0.100 / 0.100 / 0.104 |
| 82,944 (a third) | 81 | 0.513 / 0.553 / 0.777 | 0.279 / 0.320 / 0.400 |
| 248,320 (the whole vocabulary) | 243 | 0.774 / 1.017 / 1.542 | 0.699 / 0.821 / 1.047 |
| 1,000 | 1 | 0.514 / 0.511 / 0.511 | 0.092 / 0.092 / 0.091 |

**Binary candidate frames, fewer copies, n-gram rows read inline** (`net.rs`, `ngram.rs`):
- Candidates travel as (score, id) pairs of little-endian words instead of JSON. Each node spent ~0.1 ms on JSON per
  token, and the candidates fell behind the final hidden state.
- Frame blobs are read straight into `Vec<f32>`, with no conversion copy, and sent with one vectored write from the
  caller's memory, with no `BufWriter` copy. A stage's output is read once.
- A decode step's 16 n-gram rows (up to 64 rows) are read on the calling thread with `preadv2(RWF_NOWAIT)` when they
  are in the page cache. Only rows that need the disk go to threads; spawning 16 threads per step cost ~0.5 ms.

**Polled stage ends** (`gpu.rs`). A decode-sized graph (up to 8 tokens: decode steps and verification batches) and
the head scoring end with a barrier and `vkCmdSetEvent`. The host spins on that event for up to 50 ms instead of
sleeping on the fence, which woke it 12-18 µs late per stage. The fence is waited for before the command buffer or
fence is reused.

**One copy between a Mac's GPUs** (`Local::forward_direct`). In a decode step or verification batch the first GPU's
output goes straight from its host buffer into the second GPU's input. Main copied it into a new vector first.

**Spinning on expected frames** (`net.rs`). A sleeping thread wakes 10-35 µs after its data arrives, so threads that
expect a frame shortly watch for it, each for at most 100 ms before they sleep:
- a machine waiting for the final hidden state and the candidates from below;
- a node's reader thread (on the socket) and session thread (on the channel) after a decode step's forward or a
  rollback, when the next step is due within one step.

## Held back: graphs submitted ahead, gated on host events

Branch `host-gating-20261013`; not in the candidate.

**What it did.**
- Every decode-sized graph starts with a wait for a host-set event (`start`; on RADV a command-processor poll of host
  memory), so it can be submitted before its input exists.
- The driver sends a `{"next": {pos, n}}` notice down the chain at the start of a decode step. Each machine passes
  it on and submits its stages; the machine that ends the model also submits its head scoring.
- The machine's later stage is submitted while the earlier one runs, and every machine submits its head scoring
  before the final hidden state returns.
- The input's arrival then only writes the input and sets the event.

**Measured gain.**
- Session 2 (main 4c1346d): 18.18 -> 19.07 tok/s (+4.9%) alone.
- On top of everything else: d4 (the candidate's changes plus gating, no CPU hold) ran 48.51 ms per token in
  session 3, against the candidate's 50.52 ms (g2) in session 6. That is about 2 ms per token (+4%).
- The trace ([timeline-a.txt](timeline-a.txt), [timeline-f4.txt](timeline-f4.txt)) puts stage submission to GPU
  start at 0.25-0.30 ms per token, against 2.44 on main. Pre-recorded command buffers submitted when the input
  arrives would not help: main already caches recorded graphs, and the cost is `vkQueueSubmit` itself.

**Why it is held back.**
- amdgpu's job timeout on the Macs' 7.2.9-p2p kernel is 2 s (`lockup_timeout` default), not 10 s.
- The design's watchdog released gates after 2 s. Session 3's stall test (the driver pausing 3 s between a notice
  and its forward) lost that race: GPU resets on alight, invite and enlight, and a soft one on indite
  ([results/qwen4/invite-hang-20261013](../invite-hang-20261013/README.md)).
- The branch now releases gates at 1 s, with unit tests on simulated gates (`gate.rs`).
- But a process killed by SIGKILL, an abort or a segfault mid-decode still leaves its gates armed, and the kernel
  then resets the GPU.

**Next.** A gate that bounds itself on the GPU (branch `host-gate-shader-20261013`, `qwen4 gate-check`). One lane
polls a word of host memory and gives up after a fixed number of polls or shader-clock ticks (~0.25 s), so no wait
outlives its bound whatever the host does. It needs a single-GPU check first: do host writes reach a polling lane on
Tahiti, and how soon. Alternatively the wait could move into the kernel's scheduler, with a vgem fence as the
submission's wait semaphore; that needs the vgem module on the Macs.

## Held back: CPUs at the `performance` governor

The GPU clock holder (`clocks.rs`) also wrote `performance` to every cpufreq policy while work arrived. Between bursts
of host work schedutil lets an Ivy Bridge-EP package fall to 1.2 GHz. The E5-2697 v2 nodes ran their host work 2-3x
slower than indite's E5-1650 v2. It paid too little for its power. Session 3, main 4307169, six-Mac SMC PSTR over the
decode windows of the bench runs, from the netconsole telemetry (`tools/summary.py --power`,
`tools/components.py`):

| run | decode | six Macs' PSTR | CPU packages (PCPC) | GPUs (PG0R + PG1R) |
|---|---|---|---|---|
| main | 19.14 tok/s | 1352-1382 W | 347-364 W | 709-754 W |
| f4 without the hold (d4) | 20.61 tok/s | 1357-1364 W | 340-343 W | 730-735 W |
| f4 (with the hold) | 20.84 tok/s | 1554-1559 W | 474-476 W | 777-778 W |

- The hold bought +1.1% decode (0.53 ms per token) for +195 W (+14%). Everything else in f4, spinning included, drew
  what main does.
- Over the MTP runs main drew 1240 W on average, f4 1377 W, with 5-second peaks of 1546-1555 W.
- The engine's clocks are main's again.

**Idle power while the chain runs.** The chain's runs start every engine process through `scripts/latency-run.py`,
which holds a 0 µs CPU latency request, so idle cores busy-poll instead of sleeping. In session 3's server test the
CPU packages drew 321-353 W for the six Macs while the server waited between requests (PSTR 824 W with the GPUs back
on `auto`, 1240 W while they were still held high). With no engine running they drew 48 W (PSTR 486 W). That costs
nothing in a benchmark, but it matters for a server that idles for long.

## Tested source

| | source | binary (md5) |
|---|---|---|
| main (session 8) | e9f34e61 | `trashcan-e9f34e6` (1e671282) |
| **the branch, g4** (session 8) | **b15bf062 on e9f34e61** (now 30ad868d on 9f54fb80, the same engine): g3 ported onto the two-slot stages | `trashcan-host-g4` (dbdd8297) |
| main (session 7) | eb4dbeb | `trashcan-eb4dbeb` (c0857f9d) |
| merge candidate g3 (session 7, the A/B) | a480b086 on eb4dbeb | `trashcan-host-g3` (e338b8c1) |
| main (session 6) | 2293550 | `trashcan-2293550` (f3b8926a) |
| g2 (session 6): g3 before the rebase onto eb4dbeb and the trace flush | f3c068de on 2293550 | `trashcan-host-g2` (f5f392e4) |
| g1 (session 5): g2 with the CPU hold, on adf8cfd8 | 785fe7fa | `trashcan-host-g1` (af107ae7) |
| main (session 5) | adf8cfd8 | `trashcan-adf8cfd` (ad644d12) |
| f4 (session 3): the candidate's changes plus gating and the CPU hold, on 4307169 | f259478c (the gated tree, now branch `host-gating-20261013`) | `trashcan-host-f4` (b8354144) |
| d4 (session 3): f4 without the CPU hold | f259478c with main's `clocks.rs` | `trashcan-host-d4` (f8a1665c) |
| main (session 3) | 4307169 | `trashcan-4307169` (13faba5b) |
| a (session 2): gating and polled stage ends alone, on 4c1346d | the first gated commit, since rebased (eeb0aada on 4307169) | `trashcan-host-a` (bfebeaf6) |
| main (sessions 1 and 2) | 4c1346d | `trashcan-4c1346d` (98420e8e) |

Six Macs, incite driving `indite.tb0, inflight.tb1, alight.tb0, invite.tb1, enlight.tb0`, `--gpus 0,1 --head-split 12
--context 32768 --batch 128 --ngram disk`, model G-iq4. Every GPU held 850/1370 MHz in every decode window, except
invite's GPU0 in session 7 (450 MHz throughout, both variants; see Speed). Clock logs are under `HOST/clocks/`.

## Commands

Each session ran through `chain6.sh`, which leases all six Macs, and `results/qwen4/chain-20261012/tools/q4run.py`.
The scripts are `tools/session1.sh`, `session2.sh`, `session3.sh`, `session5.sh`, `session6.sh`, `session7.sh` and
`session8.sh`; each one prints its runs with `DRY=1`. The head kernels' check is `tools/topk-check.sh` (one-Mac lease). Analysis:

```sh
tools/fetch.sh                                   # copy every Mac's host-* outputs here
python3 tools/summary.py . 'host-s7-*' --power NETCONSOLE_LOG...   # decode/prefill, clocks and PSTR per run
python3 tools/timeline.py incite/host-trace7g3-incite.json.gz indite/host-trace7g3-indite.json.gz \
    inflight/host-trace7g3-inflight.json.gz alight/host-trace7g3-alight.json.gz invite/host-trace7g3-invite.json.gz \
    enlight/host-trace7g3-enlight.json.gz        # a decode step's critical path
python3 tools/mtp_summary.py . 'host-s7-mtp-*'
python3 tools/components.py NETCONSOLE_LOG 2026-10-13 HH:MM:SS HH:MM:SS   # PSTR, CPU package and GPU watts
```

## Correctness

- `qwen4 validate`, the 14-token reference prompt and p100 (`--greedy 10`): greedy tokens and logits bit for bit
  main's.
  - Session 8: g4 (the branch) against e9f34e61.
  - Session 7: g3 against eb4dbeb.
  - Session 6: g2 against 2293550.
  - Session 5: g1 against adf8cfd.
  - Session 3: f4 against 4307169.
  - Session 2: a against 4c1346d.
- Reference prompt greedy tokens: 9419 0 2500 628 353 1438 488 3242 30 248046. p100: 248046 198 248045 248045 248046
  198 248045 74455 198 248068.
- MTP (`qwen4 mtp spec`, three chat prompts): the ids are those of plain greedy decoding in every run. Tokens per
  round differ between runs and builds because each round's draft count follows the measured costs.
- Head candidate kernels: bit for bit on both of incite's GPUs (above).
- On this Mac, the driver and two CPU nodes with G-iq4 (`--cpu --cuts 16,32 --head-split 3`): validate gives the same
  1332 files (logits and every capture) as main 2293550's build, and `mtp spec` gives the plain greedy ids. This exercises
  the frames, the node loop and verification without GPUs.
- `cargo test`: the frame tests check headers, floats and candidates (scores' bits, -0, every id), a 16 MiB frame
  through partial writes, and that blobs must be whole words.

## Speed

Six Macs, `qwen4 bench --pp 512 --tg 128` on real text (`long.ids`). Decode tok/s, with ms per token in brackets.

**Session 7, the merge A/B** (main eb4dbeb against g3, two untraced alternated pairs):

| | main | g3 |
|---|---|---|
| pair 1 | 19.26 | 19.82 |
| pair 2 | 19.17 | 19.71 |
| mean | 19.22 (52.04 ms) | **19.76 (50.60 ms)**, +2.8%, -1.44 ms per token |
| prefill 512 | 384.8 | 384.6 |

- One traced g3 run after the pairs, not part of the A/B: 19.79.
- Throughout session 7, invite's GPU0 held 450 MHz instead of 850 (72-75 °C); every other GPU held 850/1370. It
  was latched in amdgpu's thermal state (power level 1) after overheating in a run before the session, and refused
  the `high` level until a reboot. Its stage took 5.94 ms against 3.54-3.70 ms for the other eleven, which costs both
  variants about 2.3 ms per token in decode. In prefill the slowest stage paces the pipeline: 385 tok/s at 512
  tokens against 486-488 in session 6. Session 6 has the unthrottled numbers.

**Session 8, the branch after the port onto main e9f34e61** (one pair, every GPU at 850/1370): main 20.72 tok/s
(48.26 ms), g4 21.08 (47.44 ms, +1.7%). Prefill 512: 520.5 against 521.0. MTP chat mean of g4: plain 20.87, MTP
27.76, ids identical. PSTR: main 1628 W, g4 1592 W.

**Earlier sessions:**

| session | main | candidate (all with the CPU hold dropped, except where noted) |
|---|---|---|
| 6 (2293550, one pair) | 19.30 (51.82 ms) | g2 19.79 (50.52 ms, +2.6%); traced 19.97 |
| 3 (4307169, two pairs) | 19.07, 19.20 (52.26 ms) | d4 (gating, no hold) 20.61, 20.62 (48.51 ms, +7.7%); f4 (gating and hold) 20.86, 20.83 (47.98 ms, +8.9%); traced f4 20.82 |
| 2 (4c1346d, three pairs) | 18.24, 18.18, 18.11 | a (gating and polled stage ends alone) 19.08, 19.07, 19.06 (+4.9%) |

The gating's share on top of the candidate is about 2.0 ms per token (+4%). That is d4's 48.51 ms in session 3
against g2's 50.52 ms in session 6. Those are different main commits, but decode kernels didn't change between them,
and main measured 19.14 against 19.30.

## MTP

`qwen4 mtp spec --draft-tokens 3 --tg 192 --runs 1` on the three chat prompts (water, csv, transformer). These are
means over the prompts (tok/s, plain greedy decoding and speculative). The ids are identical to plain greedy decoding
in every run.

| session | main plain / MTP | candidate plain / MTP |
|---|---|---|
| 7 (eb4dbeb) | 19.23 / 25.70 | g3 19.87 / **26.62** (+3.6%) |
| 6 (2293550) | 19.46 / 23.35 | g2 20.00 / 23.56 (+0.9%) |
| 3 (4307169) | 19.24 / 22.79 | f4 21.19 / 24.65 (+8.2%) |

Tokens per round vary between runs and builds, by up to 0.3, because each round's draft count follows the costs it
measured (`spec.rs`, `Policy`). That makes single MTP passes noisier than decode.

## Power

Six-Mac SMC PSTR over the decode windows of the bench runs (`tools/summary.py --power`):

| session | main | candidate |
|---|---|---|
| 8 | 1628 W | g4 1592 W |
| 7 | 1614 W (2 runs) | g3 1570 W (2 runs) |
| 6 | 1545 W | g2 1581 W |
| 3 | 1352-1382 W | d4 1357-1364 W |

The spinning threads (a node's reader and session threads after a decode step, any machine waiting for a reply, the
host waiting for a stage's done event) show no consistent cost: +36 W in session 6, -44 W in session 7, -36 W in
session 8, none in session 3. Absolute levels differ between sessions by up to 250 W. The CPU hold's cost is under Held back.

## Robustness

- Nothing in the candidate waits on the GPU for the host. Every wait is bounded:
  - spinning on a done event for at most 50 ms, then the fence;
  - spinning on a socket or channel for at most 100 ms, then sleeping.
- The server test (`tools/host-serve-idle.sh`: two chat turns, 15 s without requests, two more, then
  `scripts/test-server.py`) passed with g2 in session 6. The second turn of each pair reused the first's state, the
  time to first token was 0.29-0.34 s, and all six groups of `test-server.py` checks passed
  ([incite/host-serve-g2.json](incite/host-serve-g2.json)). It also passed with f4 in session 3.
- Session 5 stopped after its validations. During the first bench run, of main's `trashcan-adf8cfd`, invite's second
  GPU was reset: a layer command buffer of the prefill ran past 2 s
  ([results/qwen4/invite-hang-20261013](../invite-hang-20261013/README.md)). Sessions 6, 7 and 8 ran after invite's
  diagnostics, with no amdgpu errors on any Mac before or after.
- The traced runs need every node to write its trace before the runner stops it. At the end of a traced bench the
  driver asks each node to write its trace now (`Chain::write_traces`). Session 6's traced run predates that and lost
  four node traces.

## What is left

1. **Graphs submitted ahead, with a gate that bounds itself on the GPU** (about 2 ms per token, the largest piece of
   host work left: `vkQueueSubmit` of each stage's graph). Branches `host-gating-20261013` (the gated design, watchdog
   at 1 s) and `host-gate-shader-20261013` (the bounded gate and `qwen4 gate-check`). Plan:
   - first the single-GPU check: host writes seen by a polling lane, release latency, bounds near 0.25 s;
   - then a six-Mac A/B against the candidate.

   If it holds, the 27B's tensor-parallel gates can use the same gate.
2. **USB4STREAM for a decode step's frames**, not built here. A dedicated transport agent owns it; `write_frame` and
   `read_frame` already work over any `Write`/`Read`.
   - The 27B's host-only exchanges (results/qwen/tp4-viability-20261006) put a 20 KiB exchange at 77-116 µs over TCP
     and 34 µs over one stream.
   - Here the 10 hops of a decode step take 0.054-0.081 ms each, about 0.7 ms per token, so streams might save
     0.3-0.4 ms per token.
   - Design: TCP keeps every control frame and prefill. A per-link stream carries a decode step's residual down and
     its final hidden state and candidates back up. All are far below a stream's 1 MiB receive ring (it has no Tx
     flow control). Each read gets a 2 s deadline. A greet handshake per session (`link.rs`), and TCP whenever a side
     has no stream.
3. **The 0 µs PM QoS request while idle** (above): hold it only while a request runs.
