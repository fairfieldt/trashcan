# USB4STREAM on Linux 7.3: busy polling, the CLOSE regression, patches 0006 and 0007 (2026-10-14)

**Result.** 7.3's busy polling does not release lone frames, so the link keeps its three extra frames after every
write. 7.3 also brought a regression that kept links off their streams: once a stream device has read a CLOSE, every
later read returns end of file and every write fails with ENXIO until it is closed, and on these Macs a CLOSE from the
end of one session often reaches the next. In the 9baedca main suite on 7.3, 9 of 25 chain links needed retries to
reach their stream and 2 never did. `kernel/patches-7.3/0006` restores 7.2's behaviour. On one cable (invite +
enlight), back-to-back sessions (normal exits, killed clients, server crashes): 9 of 20 reached their stream within
100 ms on 7.3, 192 of 200 with 0006, and 197 and 198 of 200 with the link's later changes, every remaining failure
an epoch with one dead direction (now given up after 50 ms and retried at once). `0007` makes a close take under a
millisecond instead of 500 ms. Rolled out as `7.3.0-rc7-p2p-s2` (0006 + 0007) with the merged link code, all six links
churned at once: 300 of 300 sessions reached their streams within 100 ms (at most 53 ms), with no failed or one-way
epoch. The lone-frame hold is a property of a cable direction that survives Thunderbolt
resets and reboots; the path tables are the same on every cable, a CLOSE held by it is what reaches the next
session, and no cable with an end on the diagnostic build `-p2p-s3` (0008 at its defaults) holds.

## Busy polling does not release lone frames

The question from window 2: does a busy-polled ring read a lone frame at once (the receiving controllers read a frame
only once up to three more have arrived behind it)? `tools/probe-window.sh` (stream-bp1, 13:23) ran the raw ping-pong
probe on tblink's streams with `busy_poll` 0, then 1. Most probes died with ENXIO (next section); the two links that
ran with `busy_poll` 1 still held frames (enlight -> incite 3 fillers per round trip, indite -> invite 2; `bp1/`).

`tools/probe-fresh.sh` (stream-bp2, 13:40) repeated it on fresh streams beside tblink's, each used once (`pb0`: HopID
14, `busy_poll` 0; `pb1`: HopID 13, `busy_poll` 1), on all six links at once: 25 round trips of single 112-byte frames,
no follow-up frames; a side that waits 20 ms writes a filler (`bp2/`). No round trip completed without fillers:

| link (client -> server) | busy_poll 0 | busy_poll 1 |
|---|---|---|
| incite -> inflight | 3 (60 ms) | 3 (60 ms) |
| inflight -> alight | 3 | 2 (40 ms) |
| alight -> indite | 3 | 2 |
| indite -> invite | 2 | 3 |
| invite -> enlight | 3 | 2 |
| enlight -> incite | 2 | 3 |

Fillers per round trip, both directions together (p50 = p90 in every run). With `busy_poll` the driver's `read()`
polls the descriptors itself (`tb_ring_poll`), so a frame it does not see has not been completed by the controller:
the hold is in the hardware, not in interrupt delivery. The link keeps `PUSH = 3`. One side effect: with `busy_poll`
1 a close took under 1 ms unless the other end had closed first (500 ms); without it, every close takes 500 ms.

## 7.3: a CLOSE that has been read fails every later read and write

`thunderbolt: stream: Add support for busy polling` (e120d14d03b3, in 7.3) rewrote the wait loops of `read_iter()`
and `write_iter()` and moved the `sdev->closed` check (set when a CLOSE packet is read) to the top, before the ring is
looked at. On 7.2.9 a CLOSE gave the reader end of file once, frames behind it could still be read, and only a
blocking read or write that would have waited returned 0 or ENXIO. On 7.3 every read returns 0 and every write fails
with ENXIO until the device is closed.

Every close sends a CLOSE, and on these Macs a CLOSE from the end of one session can reach the next session at the
other end: one that arrives while that end is still open, or closing, and is not read before it closes. One sent to an
end that is already closed is dropped (`tools/closecheck.py`: eight opens and closes with the other end closed left
nothing there, on either kernel; `cable-1/close-*.jsonl`). The link ignores a CLOSE while a stream epoch opens; on 7.3
the device then refuses to write (the epoch fails with `write: No such device or address`) or returns only end of
file (no answer on the stream), and the link tries again after 1, 2, 4 ... s.

In the main suite with trashcan-9baedca on 7.3 (13:13-13:22, `suite-9baedca-7.3/`: the driver's and every node's
log), 5 sessions x 5 chain links:

| session | links on the stream at once | after failed epochs | never |
|---|---|---|---|
| val14 | 5 | | |
| val100 | 4 | | invite -> enlight (3 failed) |
| pp512-1 | 3 | indite -> invite (3), invite -> enlight (1) | |
| pp512-2 | 1 | inflight -> alight (1), alight -> indite (2), indite -> invite (2), invite -> enlight (3) | |
| pp2048 | 3 | indite -> invite (1) | invite -> enlight (6) |

16 of 25 links reached their stream on the first epoch, 7 after 1-3 failed epochs (on TCP until then) and 2 never
(the whole session on TCP); 22 failed epochs, 19 with ENXIO at the accepting end and 3 with no answer. No stream went
down once up, and no data was damaged. On 7.2.9 (window 2 and the TP4 session) every epoch outside the fault steps
opened at once. pp512-1's prefill (346 tok/s against 509 in pp512-2, with the driver's own GPU times identical) is
not explained by these logs.

## Patch 0006

`kernel/patches-7.3/0006-thunderbolt-stream-io-after-close.patch` checks `sdev->closed` only where a blocking read
or write would otherwise wait, as 7.2 did: a non-blocking read with nothing to read returns EAGAIN, a CLOSE gives end
of file once, and writes are never refused because of a CLOSE. `removed` (the configfs group deleted) is still checked
first, as in 7.3. Dropping the check from `write_iter()` alone would not do: reads would then return only end of
file. Discarding frames that arrived before the open would not be smaller, nor reliable: the old frames are delivered
after the new open, and the last ones are held like any lone frame until more arrive, so the driver cannot tell them
from new ones. 0006 is the one to send upstream (Fixes: e120d14d03b3). It compiles without warnings at W=1;
checkpatch notes only the missing Signed-off-by.

Built as `7.3.0-rc7-p2p-s` (`KLOCAL=-p2p-s REV=2 kernel/cross-build-7.3.sh`), so it installs beside 7.3.0-rc7-p2p and
boots once with `kernel-boot.sh once`: `linux-image-7.3.0-rc7-p2p-s_7.3.0~rc7-2p2p_amd64.deb` sha256
`68fc52d360fcd060ed4b15034c549f90ff48ee5812893ee6aca95c844ea9f78e`, headers
`3e73b6c8166e587adaad31d346f9e8907194e2d820d6fcd69033214f7c4545fa`.

## One cable, invite + enlight (`cable-1/`, 14:08-14:20)

`tools/cable-session.sh` under leases on both Macs: on 7.3.0-rc7-p2p, the CLOSE probe each way and 20 sessions of
`link-test churn`; then 7.3.0-rc7-p2p-s installed and picked for one boot on both, a reboot, the coordinator's
Thunderbolt reset, and the probe, 200 sessions and the link-test suite. Churn runs one client process per session on
invite against a server on enlight that is restarted after each crash, in the pattern normal exit, client killed 50-250
ms after its stream came up, normal exit, server exits 0.2 s into the test. A session passes when its stream is up
within 100 ms of its link's start with no failed epoch.

| | 7.3.0-rc7-p2p (20 sessions) | 7.3.0-rc7-p2p-s, 0006 (200) |
|---|---|---|
| passed | 9 | 192 |
| on the stream at all | 11 | 198 |
| failed epochs, client / server | 6 / 6 | 1 / 1 |
| of them `write: No such device or address` | 6 | 0 |
| time to stream p50 / p90 / max | 7.8 / 1020 / 1085 ms | 12.9 / 34.5 / 4036 ms |

On 7.3.0-rc7-p2p, every failure is the CLOSE regression: ENXIO at the server after a server crash, and four sessions
after kills or crashes that never reached the stream (the client wrote 82 frames and read nothing, the server logged
nothing). With 0006 the suite after the churn passed (5 of 5 tests on the stream: 64 B round trip 25 us p50, 5 MiB
524 / 538 MB/s), and 8 sessions still failed:

| sessions | time to stream | what happened | recovery |
|---|---|---|---|
| 54, 66, 118 (normal exit, after a kill) | 114-116 ms | the server's open of the device waited 106-109 ms for the previous session's close (every close takes 500 ms) | 114-116 ms |
| 43, 195 (server exits) | 713, 1468 ms | no failed epoch, no slow open; probes were lone frames, every 50 ms after the first 20, each held by the receiver until three more arrived | the same |
| 41 (client killed) | 4036 ms | the first epoch had no answer for 3 s; the retry came up at once | 4.0 s |
| 110, 170 (normal exit, after a kill) | never | the client opened its device and wrote 81-82 frames, read none; the server logged nothing for the epoch | the next session, about 2.8 s later |

## Link changes after the first cable

- Probes are followed by PUSH frames (while the other ring has room), and go every 2 ms for the first 500 ms of an
  epoch, then every 50 ms until the probe timeout (they were 20 at 2 ms, then 44 at 50 ms). An epoch whose other end
  opens late comes up within milliseconds of that open (test `an_epoch_comes_up_soon_after_a_late_open`).
- An epoch that fails, comes up more than 100 ms after it began, or is still opening when its link closes is described
  in its message: when its device opened and its first record arrived, probes and frames sent, records and ends of file
  read, and whether the other side read ours. An open of the device that waits over 100 ms says so.

## Churns 2 and 3: one-way epochs (`cable-2/`, `cable-3/`)

`tools/churn-session.sh` on the same pair and kernel (7.3.0-rc7-p2p-s), 200 sessions each, no reboot:

| | churn 2 (99ae8c8e: probes, epoch logging) | churn 3 (33360c63: + connecting side waits) |
|---|---|---|
| passed | 197 | 198 |
| time to stream p50 / p99 / max | 9.8 / 90.7 / 4044 ms | 10.0 / 51.3 / 360 ms |
| slow opens (> 100 ms) | 0 | 1 (350 ms) |
| failures | 3 one-way epochs | 1 one-way epoch, 1 open waiting for a close |

The slow-probe tail is gone. The epoch messages show what the remaining failures are: in every one, one direction of
the epoch carried nothing while the other worked.

| churn, session | dead direction | what each side saw | recovery |
|---|---|---|---|
| 2, 33 (killed) | invite -> enlight | invite read 476 of enlight's records; enlight read 0 of invite's 285 frames (its device open at 8 ms) | 4.04 s (3 s probe timeout, then the next epoch at once) |
| 2, 110 (normal, after a kill) | invite -> enlight | 446 records one way, 0 the other (enlight's device open at 64 ms) | none in the session (2.7 s, over TCP) |
| 2, 183 (server exits) | invite -> enlight | 450 records one way, 0 the other | none until the server exited |
| 3, 24 (normal, after a server crash) | enlight -> invite | invite read enlight's first 4 records and nothing after; enlight read invite's frames and came up at 14 ms | given up at 114 ms; the retry's open waited for that epoch's 500 ms close, and the session (723 ms) ended first |

Churn 1's sessions 41, 110 and 170 fit the same picture. Every case came right after a session that ended while the
other end was busy (a kill, a crash, or a normal exit). A new epoch, with both devices reopened, has always worked.

Two raw-device experiments (`tools/deadpath.py`, `deadpath/`, CPU only) tested whether frames reaching an end that is
closed, or still opening, leave that direction dead. Neither did:

- bursts of 0, 4, 16, 64 and 300 data frames toward a closed end, which opened 0.1 s or 1 s later, each way, twice:
  in all 28 rounds both directions carried 19-20 of 20 test frames afterwards. The burst frames themselves were
  dropped (0 arrived), except in two rounds where the end opened while 300 frames were still being written (29 and 31
  arrived);
- a frame every millisecond toward an end that opened in the middle of the stream, 10 times each way: in all 20 rounds
  both directions worked, and the opening end read the 223-242 frames sent after it opened.

So the order of opens is not the cause. The link now holds its writes until it hears the other side anyway (nothing
is sent toward an end that may not be open), gives a one-way epoch up 50 ms after it first hears the other side (a
working direction shows within about 5 ms: probes go every 2 ms with pushes), and tries the first failed epoch again at
once, so that with a fast close a one-way epoch costs about 0.1 s instead of 3 s or the session.

## thunderbolt_net does not hold where the stream does (`hold/`)

`tools/holdcheck-session.sh` on invite + enlight (7.3.0-rc7-p2p-s2, 15:06): single 64-byte UDP packets there and back
over the cable's Thunderbolt IP link, one at a time with nothing else on it (`tools/udpping.py`, 2000 each way): invite
-> enlight p50 124 us, p99 147, max 425; enlight -> invite p50 95 us, p99 143, max 526; none timed out. A held packet
would never have come back. The stream on the same cable right after: 3 fillers per round trip with either Mac as
client (`pp` counts a round trip, so one direction held 3; the per-direction `pos` probe came later).
thunderbolt_net runs on the same controllers, with its Rx ring also under E2E (`e2e=Y`) and the same Falcon Ridge E2E
remap, so the hold comes with how `stream.c` sets up or uses its rings and paths (the path map below finds the paths'
settings the same). The differences:

1. Order: thunderbolt_net starts both rings, posts all 256 Rx buffers and only then enables the DMA paths ("we enable
   the DMA paths last to make sure we have primed the Rx ring before any incoming packets are allowed to arrive");
   `stream.c` enables the paths first, then starts the rings, then posts the buffers.
2. Rx completions: thunderbolt_net uses `start_poll` (NAPI: the interrupt masks the ring, `tb_ring_poll()` drains it,
   `tb_ring_poll_complete()` unmasks it); `stream.c` takes a callback per frame from the ring's work item.
3. Rx re-posting: thunderbolt_net posts consumed buffers in batches (17 at a time, the rest at the end of a poll);
   `stream.c` posts each one again as soon as it is read.
4. Rx `eof_mask`: `stream.c` adds the CLOSE PDF (3) to thunderbolt_net's BIT(2).
5. The same in both: Tx `RING_FLAG_FRAME` only (with 0005), Rx `FRAME | E2E` with the E2E hop remapped to HopID 1 on
   Falcon Ridge, ring size 256, frame mode, an interrupt per descriptor. HopIDs and interrupt throttling differ, but
   the stream held frames on HopIDs 13-15 and with throttling at 0.

`kernel/patches-7.3-diag/0008` (diagnostic, not for upstream) makes each of 1-4 a configfs switch of the stream group,
used from the device's next open: `paths_last`, `poll_mode`, `rx_repost`, `rx_eof_close`, and `rx_e2e` for the Rx
ring's E2E. With all at their defaults the driver behaves as before. Built with 0006 and 0007 as `7.3.0-rc7-p2p-s3`.

## The 0008 switches on invite + enlight (`knobs-pos/`, `knobs-pp/`)

After the one-time boot into `7.3.0-rc7-p2p-s3` (15:21) and the coordinator's Thunderbolt reset (15:22), the pair did
not hold at all, so the switches could not be judged. `tools/knobs-session.sh` (15:23 with `pos`, stopped after the
fourth setting to switch to `pp`; 15:27 with `pp`), 10 fresh opens per direction per setting and 20 churn sessions:

| setting | fillers, every open | churn |
|---|---|---|
| defaults | 0 (pos and pp) | 20/20 |
| paths_last=1 | 0 (pos and pp) | 20/20 |
| poll_mode=1 | 0 (pos and pp) | 20/20 |
| rx_repost=1 | pp: 0, one open 1; first two opens lost (see below) | 20/20 |
| rx_e2e=0 | 0 | 20/20 |
| rx_eof_close=0 | 0 | (no churn: a CLOSE no longer ends a frame) |
| paths_last=1 poll_mode=1 | 0 | 20/20 |
| paths_last=1 rx_repost=1 | 0 | 20/20 |
| paths_last=1 poll_mode=1 rx_repost=1 | 0 | 20/20 |

No churn session had a failed or one-way epoch. `poll_mode=1` is faulty: invite and enlight each logged 56 kernel
warnings `interrupt for RX ring 3 is already disabled` (`tb_ring_stop()` from `tbstream_dev_stop()`, nhi.c:143)
between 15:30:00 and 15:34:25, during the churns of the three `poll_mode=1` settings: the device was released while
its RX interrupt was still masked for a poll. The first two opens of the next setting lost their invite side. This is
a bug in the diagnostic switch, not in the driver, and no form of it may reach a patch meant for merging: 0008 now
lives in `kernel/patches-7.3-diag/`, which `kernel/build-7.3.sh` applies only with `DIAG=1` and then only under a
release name of its own (`KLOCAL`, e.g. `-p2p-s3`). The default build, the rollout and anything sent upstream take
`kernel/patches-7.3/` (0001-0007) alone. 0008 is removed once the switch test on a cable that holds is done.

## The hold on a stock cable and on a mixed cable (`control-stock/`, `mixed/`)

`tools/probe-session.sh` (no switches), 10 fresh opens per direction:

- incite -> inflight, both on stock 7.3.0-rc7-p2p (15:30): every lone frame needed 2 fillers (9 opens, the first
  lost); `pp` 2 per round trip. Every run with the other Mac as client was lost to ENXIO.
- enlight (-p2p-s3) <-> incite (stock) (15:39): 0 fillers both ways, `pos` and `pp`, in every open, none lost. Stock
  incite's receiving end does not hold here, so the receiving build does not decide the hold.

## The path map (`pathmap/`, 16:02-16:10)

`tools/pathmap-session.sh` on all six (chain6, CPU only): (a) every Mac's Thunderbolt path tables and adapter
registers from debugfs (`tools/pathdump.sh`) with the stream devices closed, with all twelve held open
(`tools/holdopen.py`) and closed again; (b) `pos` and `pp` on each ring cable, 6 fresh opens per direction each, every
run's exit codes and errors kept per end. `tools/pathmap.py` makes `pathmap/map.md`.

The paths are set up the same everywhere. On every controller with a cable thunderbolt_net has Tx ring 2 to the other
host's HopID 8 and Rx HopID 8 to ring 2; the stream has Tx ring 3 to HopID 15 and Rx HopID 15 to ring 3
(thunderbolt_net came up first after the 15:22 reset; the stream's rings are allocated at each open). Every hop of
both drivers has 14 initial credits, flow control in and out on its Tx hop (NHI adapter) and in only on its Rx hop
(lane adapter), no drop, priority 5, weight 1, no shared buffers. DMA paths add no NFC credits on these routers; each
lane adapter shows NFC 0 of 60 buffers and each NHI adapter NFC 0 of 15 (`ADP_CS_4`), and every link is bonded (width
2) at the same speed. Between the snapshots only the stream hops' enable and flow-control bits change (they are
cleared on teardown).

The hold:

| cable | pos, A -> B | pos, B -> A | pp, either client | lost opens |
|---|---|---|---|---|
| incite -> inflight | 2 (5 of 5) | lost (ENXIO) | 2 | 12 of 24 |
| inflight -> alight | 1 (5 of 5) | lost (ENXIO) | 1 | 12 of 24 |
| alight -> indite | 2 (5 of 5) | lost (ENXIO) | 2 | 12 of 24 |
| incite <-> enlight | 0 | 0 | 0 | 0 |
| indite <-> invite | 0 | 0 | 0 | 0 |
| invite <-> enlight | 0 | 0 | 0 | 0 |

`pp` needs as many fillers per round trip as the held direction alone, so on the three cables between stock Macs one
direction holds, the ring direction; the cables with an end on -p2p-s3 do not hold either way, also into their stock
ends. All six Macs had the same Thunderbolt reset at 15:22:02 and 15:22:38, and all twelve links came up at
15:22:42-43, so the age of the link does not split them, nor does which host has the larger UUID.

The lost opens are the stale CLOSE of the previous section, and only where a direction holds: a CLOSE is a lone frame,
so it waits and arrives at the receiving end's next open, where stock 7.3 fails that session's writes with ENXIO.
With the clients alternating, every `pos` run with the receiving Mac as client failed with ENXIO before its first
round (the other end's first answer pushed the CLOSE out ahead of it), five of six
`pp` runs with the sending Mac as client failed at both ends, and the first run on each cable timed out (its server
had read the CLOSE left by the held-open snapshot). On the three cables that do not hold, no open was lost, stock
ends included.

So the hold follows the cable, and no path setting differs between the cables. What the holding cables share is the
build at both ends: -p2p-s3 (0008 at its defaults) holds nowhere it touches, while the pair held 3 per round trip on
-p2p-s2 (the same patches without 0008) right after its boot (15:05). A fresh boot does not clear it: alight, rebooted
at 13:37 on stock 7.3, and invite, rebooted at 13:25, both received with a hold at 13:40 (`bp2/`), when all six
cables held. Read as source, 0008 at its defaults does what -p2p-s2 does; the compiled code is the next thing to
compare, and 0008's parts are bisected on the pair.

## After another reset (`pathmap2/`, 16:29-16:40)

The coordinator reset Thunderbolt on all six at 16:22 (all twelve ends back on two lanes), and the same window ran
again with the clients in the order A, B, B, A. Nothing changed: the path tables match the first map's on every end;
the three cables between stock Macs still hold in the ring direction by the same counts (`pp` 2, 1 and 2 per round
trip, and `pos` measured the reverse directions at 0 in all nine clean opens), and the three cables touching invite or
enlight are at 0 both ways in all 72 opens. With this order every `pos` run with the sending Mac as client got no
answer (the receiving server's device had read a stale CLOSE; the window ends such a server 3 s after its client),
so the order A, B, A, B of the first map is the one that measures the holding direction itself.

So a Thunderbolt reset does not clear the hold (it also survived the resets at 15:05 and 15:22), nor does a reboot on
stock 7.3 (previous section), and a cable with an end on -p2p-s3 has none in either direction.

## Rollout of 7.3.0-rc7-p2p-s2 (`rollout/`, 16:53-17:10)

After the link changes were merged (main 1687af54, deployed as `trashcan-1687af5`), `tools/rollout-session.sh` under
chain6 installed `7.3.0-rc7-p2p-s2` (0001-0007, `linux-image` sha256 `a91c727d`, headers `bbcea1ad`) on all six for
one boot, rebooted them together (16:54; all up in 30-40 s), waited for the coordinator's Thunderbolt reset, ran
`check.sh` on each Mac (the VBIOS images as before, `xgpu p2p-verify` passed, one dmesg warning each: apple-gmux's at
boot, as on stock 7.3) and then `link-test churn` on all six links at once, 50 sessions each:

| link | passed | time to stream p50 / max | failed epochs (client / server) | one-way epochs |
|---|---|---|---|---|
| incite -> inflight | 50/50 | 9 / 46 ms | 0 / 0 | 0 |
| inflight -> alight | 50/50 | 8 / 53 ms | 0 / 0 | 0 |
| alight -> indite | 50/50 | 10 / 27 ms | 0 / 0 | 0 |
| indite -> invite | 50/50 | 11 / 39 ms | 0 / 0 | 0 |
| invite -> enlight | 50/50 | 10 / 53 ms | 0 / 0 | 0 |
| enlight -> incite | 50/50 | 7 / 35 ms | 0 / 0 | 0 |

Every link passed, so the window made `7.3.0-rc7-p2p-s2` GRUB's default on all six (17:10).

## Patch 0007

`kernel/patches-7.3/0007-thunderbolt-stream-no-rx-flush-on-close.patch`: `tbstream_dev_stop()` flushed the RX ring
with a 500 ms timeout before stopping it. `tb_ring_flush()` waits for the ring's in-flight list to empty, but an RX ring
keeps its buffers posted to the controller (each is posted again as soon as it is read), so the wait always ran into
the timeout: every close took 500-530 ms (`cable-1/closetime-before.jsonl`), and an open of the same device, which
takes the device's lock, waited 480-510 ms for it. 0007 stops the RX ring right away, as the busy-polling path already
does. Built as `7.3.0-rc7-p2p-s2` (0006 and 0007): `linux-image-7.3.0-rc7-p2p-s2_7.3.0~rc7-3p2p_amd64.deb` sha256
`a91c727d0b5720bf4f71ed5b98e58a333c21ba18820ae626e7786b56fdf44bdb`, headers
`bbcea1ad2c036720d585002d70f1b5abc835662b53ed09fdde880e23613cb86a`.

Can stopping the RX ring at once hand stale frames to the next session? The frames the flush covered are those that
complete into the ring during its 500 ms: they are freed with the ring either way, unread. What reaches the next open
comes from elsewhere: frames the receiving controller still holds (the lone-frame hold: the last up to three frames to
arrive are not completed until more follow, and nothing follows once the other end has stopped sending), and frames the
other end writes while this end is closed, which wait in the other end's transmit ring and arrive when this end opens,
as long as the other end is still open. Neither depends on the flush. Stopping sooner can at most move to the next open
the few frames in flight between the ring stop and the path teardown. The link handles all of them: every record
carries its session and epoch, records of other sessions or epochs are dropped (`stale` in the link's counters), a
CLOSE while an epoch opens is ignored (and with 0006 no longer disables the device), and each open reads and drops what
the device holds before the epoch begins.

## Link changes

- `Link::report()`: how a link carried its data (`transport`: `stream` from its first epoch, `retried`, or `tcp`), its
  first stream-up time, epochs, failed epochs, fallbacks and bytes per path. The first `stream up` line gives the time
  since the link started.
- A link with a stream at both ends that has been on TCP for 5 s says so on stderr, once per stretch.
- qwen4: the driver asks every machine down the chain for its link's report and prints
  `links: incite->inflight stream (up in 3 ms), ...` on stderr; bench results and MTP rows carry `links`
  (`{"incite->inflight": "stream", ...}`) and `link_details`.
- `link-test run` prints `{"label", "up_ms"}` as soon as its stream is up; `--server-exits S` ends the server process
  S seconds into the test, as a crash would.
- `link-test churn`: sessions back to back, one client process each, in a pattern of normal exits, killed clients and
  server crashes (the server runs in a loop that restarts it); one JSON line per session and a summary; it fails unless
  every session reached its stream within `--within-ms` (100) of its link's start with no failed epoch.

## Window scripts

`cable-session.sh` started enlight's server with `pgrep ... || setsid nohup bash -c '...' >> log 2>&1 < /dev/null &`:
the `&` put the whole list in the background, and that subshell kept ssh's output open, so the ssh never returned.
At 14:10 the subshell (one of this session's own processes, on enlight) was ended by hand, which let the session go on
with the server it had started; before the second phase the server was started with the corrected command, so the
script's own start found it running. The fix puts only the loop in the background. The simulation could not see this:
its ssh answered remote commands from patterns. `tools/simtest.sh` now runs each remote snippet with bash in a
simulated Mac (`tools/sim/`: a home per Mac with stub builds and probes, stubs for `tblink`, `uname -r`, `sudo`,
`systemctl`, `setsid`, `dpkg-query`, `grub-editenv`) and, like ssh, returns only once nothing holds the snippet's
output; the old script now hangs in the simulation, which reports and stops it. Scripts are queued as `bash SCRIPT`.

## Files

- `tools/probe-window.sh`, `tools/probe-fresh.sh`, `tools/stream-pg.sh`: the busy-polling probes; `bp1/`, `bp2/`.
- `suite-9baedca-7.3/`: the 7.3 main suite's driver and node logs (copied from the Macs).
- `tools/cable-session.sh`: the first one-cable session; `cable-1/` its data. `tools/churn-session.sh`: churn on one
  cable without a reboot.
- `tools/closecheck.py`, `tools/closetime.py`: a CLOSE planted at a closed end; close and reopen times.
- `tools/churn_table.py`: a churn run by session kind, and every session that did not pass.
- `tools/deadpath-session.sh`, `tools/deadpath.py`: frames toward a closed or opening end; `deadpath/`.
- `tools/holdcheck-session.sh`, `tools/udpping.py`: thunderbolt_net and the stream on one cable; `hold/`.
- `tools/knobs-session.sh`, `tools/stream-knobs.sh`, `tools/knobs_table.py`: the 0008 switches; `knobs-pos/`,
  `knobs-pp/`. `tools/probe-session.sh`, `tools/probe_runs.py`: the probes on a cable as it stands;
  `control-stock/`, `mixed/`.
- `tools/pathmap-session.sh`, `tools/pathdump.sh`, `tools/holdopen.py`, `tools/pathmap.py`: the six-Mac path map;
  `pathmap/`, `pathmap2/` (each with `map.md`; the path dumps are gzipped).
- `tools/rollout-session.sh`: the six-Mac rollout of -p2p-s2; `rollout/` (with each Mac's `check.sh` output in
  `rollout/checks/`).
- `tools/simtest.sh`, `tools/sim/`: the window simulation.
