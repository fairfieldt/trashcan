# The six-Mac "France" answer and the TCP fallback beside it (2026-10-26)

The failing run of [qualify-20261026](../qualify-20261026/) (6-flashnext, greedy smoke, answer "France", 2.5 s) had a
fallback to TCP on invite's stream to enlight just before the request. Both had a cause of their own; both are fixed
on main:

| | Cause | Fix (main) |
|---|---|---|
| Wrong answer | A prefill race in the engine, not the link: a stage reads a chunk's output after queueing the next chunk, and both wrote one host buffer. A host delay longer than one stage time there sent the next chunk's residual on. | `be839c7e` qwen4: output per submission slot |
| Fallback | The link gave its stream up after 100 ms without an acknowledgement. The other side's link had not been turned for about a second (a node thread waiting for the disk while models load). | `79ed078f` link: 10 s, a fallback is fatal, credits over TCP when the ring is full |
| Fallback policy | A fallback was silent; `--tcp-fallback` failed first requests | `79ed078f`, `2dd1294c` deploy: no TCP fallback, the launcher fails on any link off its stream |

Hardware: incite, invite, indite, enlight, inflight, alight on 7.3.0-rc7-p2p (+ps9c9a5449) after the rollout, chain
incite → enlight → invite → indite → alight → inflight (the failing run's), every run under gpuq leases (chain6.sh).
Binaries cross-built in the kbuild VM from main plus the changes, in `~/trashcan-exp/fallback` (smokes) and
`~/trashcan-exp/flashnext/trashcan-fb4|fb5` (validation, A/B).

## The wrong answer: a prefill race

`smoke` sends a 24-token prompt. `Chain::chunk_size` cuts it into 16 + 8 tokens on 12 stages, so the two chunks are
pipelined (`Local::push`, `step`): on every stage the second chunk is submitted behind the first, then the host waits
for the first and reads its output. `GpuStage` had two slots for host inputs, fences and timestamps but one host
`output` buffer, which every forward writes at its end. If the first chunk had already finished (the second arrived
late, e.g. after the driver's cold n-gram disk reads), the second starts at once, and a host delay of one stage time
before the read gives the stage the second chunk's rows. `qwen4 validate` does not pipeline (dump mode drains), which
is why validation never saw it.

| Runs (warm cache unless noted) | Binary | Answers |
|---|---|---|
| [v2-cold](runs/v2-cold/): page cache dropped on all six before each run | link changes, no output fix | 6 Paris, **1 France** (run 4), 1 start failure (launcher) |
| [v3d-sleep300](runs/v3d-sleep300/): 300 ms sleep inserted between queueing and reading ([patch](tools/diag-step-sleep.patch)) | no output fix | **3 of 3 France** |
| [v4d-sleep300](runs/v4d-sleep300/): the same sleep | output per slot | 3 of 3 Paris |
| validate on the 14-token reference prompt, cold, 8 runs ([cold-val-v2.txt](validate/cold-val-v2.txt)) | no output fix | 8 of 8 reference tokens |

Run 4 of v2-cold answered "France" with no stream failure at all; its only link event was a 1 s pause at link start
on invite. So the fallback in the -s2 run did not cause the answer; both came from the same cold start, where host
threads wait for the disk.

## The fallback: a peer that did not turn its link for a second

In the -s2 run invite (accepting side) reported "nothing acknowledged for 100ms" at 13:50:05.732 and enlight
"the other side closed its stream" at 13:50:06.721, then epoch f3 up at 06.784-06.788 on both. enlight is the
connecting side; after a fallback it opens the next epoch `backoff` (1 s) after the `now` its `stream_down` was given.
The new epoch 63 ms after enlight's message means that `now` was about 1 s old: enlight's link thread had been inside
one turn (`pump`, holding the link) from before 05.73 until 06.72, so nothing acknowledged invite's frames, and
invite's FAIL over TCP was read only after the stream's CLOSE. Load times in that run were 3.6-9.4 s per GPU against
0.9 s warm: the weights came from the disk.

Reproduced on the new kernel with the turn diagnostics now in the link (a turn or a pause over 20 ms on the stream is
logged with the thread's page faults): after dropping the page cache, invite's `link-service` thread took one turn of
1 s with 2 major faults (v2-cold run 4, [stage2-invite.log](runs/v2-cold/20261026-144358-6-flashnext/stage2-invite.log)).
During that second invite's SSD (APPLE SM1024F, SATA, mq-deadline) had IO pressure near 100 % with almost no
throughput ([mon](runs/v2-cold/mon-144349-4/invite/)). During v4-cold, alight's disk answered nothing for about 8 s
(IO "some" ~100 %, reads ~0, 3-6 tasks in D state; the earlier log-write probe in
`usb4stream-busypoll-20261014` saw writes stall up to 1.27 s on the same SSDs).

On warm caches (the normal case) none of this shows: 20 smokes with the first diagnostic build (v1-20) had no slow turn
and no fallback.

The kernel stream driver is not involved: nonblocking reads and writes take `mutex_trylock` and return `EAGAIN`
(`tbstream_dev_lock`), no kernel message on either Mac, and -s2 and ps9c9a5449 carry the same 0005-0007 code.

## Link changes (main 79ed078f)

- Fatal fallback: a link with a stream at both ends sends data on the stream only. Once the stream has carried data, a
  failed stream ends the link ("stream lost (...); the link ends") and the Flash-Next server stops. An epoch that fails
  before any data is retried (the one-way epochs of earlier work); a stream not up within 20 s ends the link.
- `dead` and `idle_dead` 100 ms / 3 s → 10 s, so a paused peer (1 s here, up to 8 s with a stalled disk) does not end
  a healthy stream. Unit test `a_peer_that_stops_for_a_second_does_not_lose_its_stream`.
- Credit deadlock: with both sides' windows full and the frames returning credits lost, neither side could send even an
  acknowledgement until `dead`. The simulator's periodic faults hid and then exaggerated this (a fault period equal to
  the retransmission round of 63 records + 3 pushes + 1 ack hit the same record forever); faults are now pseudo-random,
  and acknowledgements and credits go over TCP (REPORT) when the stream has no room.
- Clean end: a closing link sends BYE on the stream ahead of its CLOSE; a CLOSE without one waits up to 200 ms for the
  peer's word over TCP. A stopped peer reads as "closed the link", not as a lost stream (the coordinator's teardown
  "stream down" on enlight's link to incite).
- Byte exactness of the old switchover: [tools/stress_switchover.rs](tools/stress_switchover.rs) (old code), 300
  randomized forced failures mid-transfer (dead stream for 0-300 ms, CLOSE from either side, either side giving up,
  1-in-3 loss), reader and writer threads on both sides, every byte checked across the epoch changes: no loss,
  repetition or reordering.

`cargo test`: 102 passed (20 consecutive runs of the link tests clean after the changes).

## Hardware verification (final binary v5, main 2dd1294c's engine)

| Batch | Runs | Wrong answers | Stream lost / not up / fallback | Slow turns | Other |
|---|---|---|---|---|---|
| [v5-warm](runs/v5-warm/) | 20 | 0 (19 Paris) | 0 | 0 | 1 start failure: gpuq heartbeat DNS failure on inflight (`Temporary failure in name resolution`), node never started |
| [v5-cold](runs/v5-cold/) (page cache dropped on all six each run) | 8 | 0 (8 Paris) | 0 | 0 | 40 of 40 links up at their first epoch |
| [v4-warm](runs/v4-warm/) | 20 | 0 (20 Paris) | 0 | 0 | |
| [v4-cold](runs/v4-cold/) | 8 | 0 (5 Paris) | 2 runs: stream not up within 5 s | 0 | v4 read the binary's pages at the first link and waited in an 8 s disk stall on alight (removed in v5, stream_wait 5 → 20 s); 1 start failure, pid file not yet written on a cold disk (launcher fixed) |

Correctness ([validate/summary.txt](validate/summary.txt)): `qwen4 validate --greedy 10` on the 14-token reference
prompt and the 100-token prompt, candidate (fb4, fb5) and base (A16's 034690eb): identical greedy tokens, stage dumps
690 of 690 and 595 of 595 tensors identical.

Decode A/B (one pair, pp512 + tg128, `--batch 128`, sustained 850/1370 MHz on every GPU): base 22.37 tok/s, candidate
22.48 tok/s; prefill 517.1 vs 517.3 tok/s. No regression.

## For the maintainer

- The SATA SSDs of the node Macs stall for seconds under load (8 s seen on alight after a cache drop, 1.27 s log writes
  earlier). The engine now tolerates it, but anything synchronous on those disks during serving will hang that long.
- The launcher's start check treated a missing pid file as an ended process (a cold node can take seconds to write it);
  fixed in 2dd1294c.
- [qualify-20261026](../qualify-20261026/README.md) puts the failed smoke down to the TCP fallback; it was the engine's
  prefill race.

## Reproduce

`tools/smokes.sh N LAUNCHER OUTDIR [launcher args]` (under chain6.sh; `COLD=1` drops the page cache first; monitors
from `tools/mon.sh`), `tools/cold-val.sh N` (cold validate), `tools/fb-val.sh` (validate and A/B, run from
`results/qwen4/chain-20261012`), `tools/verify-all.sh` (all three).
