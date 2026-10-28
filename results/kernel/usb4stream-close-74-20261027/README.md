# USB4STREAM CLOSE handling and Tx E2E on a 7.4 base (2026-10-27)

**Result.** On mainline 8a36f53fd5ec (the 7.4 merge window, with thunderbolt.git `next` merged) the stale CLOSE that
made 0006 necessary on 7.3 does not happen, and 0006 is not needed there: 500 of 500 churn sessions of the deployed
engine's link reached their stream within 100 ms with no failed epoch (three runs, busy_poll 0 and 1), and 80 of 80
direct stale-CLOSE rounds started a clean session. The reason looks like the end of the lone-frame hold: on 7.4 a
single 112-byte frame with nothing behind it arrives at once (round trip of two lone frames 0.2-0.3 ms, 0.04-0.07 ms
with busy_poll), where on 7.3 it waited for three more frames (a 20 ms filler, `../usb4stream-busypoll-20261014/`).
A CLOSE now reaches an open end while it is open, and 7.4's release consumes it (`tbstream_dev_advance_both()`), so
nothing is left for the next session. The 7.3 control of the same probe was not run: the session stopped first (below).
0005 (no Tx E2E) is still needed on this base: with it 500 of 500 sessions; with only 0005 reverted in
`thunderbolt_stream.ko`, 0 of 7 sessions got data across the stream.

The session ended early: **incite, which was not under test, panicked** at 17:25:56 in the Thunderbolt hotplug worker
of its 7.3.0-rc7-p2p kernel when inflight rebooted back to 7.3 (`incite-panic-1725/`). Testing stopped there and the
hardware was left alone; the six-Mac links need the usual reset (below).

## Setup

- inflight and alight, one cable (inflight:1 - alight:1, two lanes), under gpuq leases #159 and #178.
- Kernel `7.3.0-mw8a36f53+` (`../linux-7.4mw-20261027/`): mainline 8a36f53fd5ec plus the six upstream packages in
  `kernel/upstream/` (applesmc port lights, P2PDMA Ivy Bridge-EP, VFCT exact match, USB4STREAM no Tx E2E and HopID
  release, thunderbolt_net ethtool coalescing); no 0006, no 0007. Booted once with `grub-reboot`; GRUB's default
  (`/etc/default/grub.d/91-trashcan-default.cfg`) stayed 7.3.0-rc7-p2p (patch set ps9c9a5449), which both Macs run
  again.
- After the first boot alight had no XDomain to inflight ("failed read XDomain properties"); `scripts/tb-reset.sh
  inflight alight` brought the link up on two lanes with a stream at both ends.
- Link code: the deployed engine binary `~/trashcan-exp/a16/trashcan` (2026-10-26), `link-test`.

## Tests

| Run | What | Result |
|---|---|---|
| `run1-74/` | closecheck probe each way; 200 churn sessions; link-test suite (busy_poll 0) | probe: a CLOSE to a closed end is dropped (0 ends of file, write ok, both ways); 200/200 within 100 ms (p50 10.5 ms, max 53 ms), 0 failed epochs; suite 5/5 |
| `stale1-74/` | `tools/closestale.py` via `tools/stale-session.sh`, 10 rounds per direction x nb/block x busy_poll 0/1 | 80/80 pass |
| `stale2-74/` | the same, 5 rounds each, with timing | 40/40 pass; lone-frame round trips in the table below |
| `noe2e/` | `thunderbolt_stream.ko` with only 0005 reverted (`revert-0005.diff`), 10-session churn | 0 of 7 sessions reached the stream (60 s each, "first record at never"); churn stopped by its timeout. `noe2e/invalid-no-server/` is a discarded first try whose server never started |
| `run2-74-after-reset/` | after `tb-reset.sh` reloaded the stock module: probe, 100 sessions, suite | 100/100 (max 55 ms), suite 5/5 |
| `run3-74-busypoll/` | busy_poll 1 on both ends: probe, 200 sessions, suite | 200/200 (max 57 ms), suite 5/5 |

A stale-CLOSE round (X under test, Y the other end): X opens its device and waits 0.6 s without reading; Y opens and
closes (a CLOSE to an open end); X closes; X opens a new session and waits up to 1 s for data; 0.2 s later Y opens,
writes one 112-byte frame and reads the reply. Pass: X reads the frame and its own write succeeds. With busy_poll 0 X
saw `POLLHUP` before it closed in every round, so the CLOSE had arrived; the new session saw no `POLLHUP`, no end of
file, read the frame and wrote. With busy_poll 1 `poll()` reports only `POLLERR` (busy-polled devices don't support
poll), and reads and writes behaved the same.

Lone-frame round trip (Y writes one frame, X answers with one, Y's read time; ms, 5 rounds):

| busy_poll | blocking reads | non-blocking reads (1 ms sleep between tries) |
|---|---|---|
| 0 | 0.20-0.31 | 1.11-1.16 |
| 1 | 0.04-0.07 | 1.10-1.21 |

Suite on 7.4 (`run1-74/suite.jsonl`), for comparison with 7.3 in `docs/kernel.md`: 64 B ping-pong p50 42.5 us, 5 MiB
bulk 613 / 580 MB/s. The link still sends its fillers (PUSH = 3); with the hold gone they may no longer be needed on 7.4
(not measured).

## incite's panic (`incite-panic-1725/`)

incite (7.3.0-rc7-p2p #1p2p+ps9c9a5449, up since 11:01:26, not leased and not touched by this session) logged, by
uptime:

- 21417 s (16:58:23): `2-3: host disconnected` - inflight rebooting into the test kernel;
- 21423 s: `new host found, vendor=0xa27 device=0xa, Apple, Inc. Macintosh` (inflight's firmware), then at 21459 s
  `failed read XDomain properties from 5ce665e8-...`: Linux on inflight was never seen on that link again;
- 22907 s (17:23:13): `host disconnected` (the Thunderbolt reset of inflight);
- 23070 s (17:25:56, inflight rebooting back): a BUG in `kobject_get()` called from `tb_handle_hotplug()`, then a
  general protection fault in `string()` while printing it, and `Kernel panic - not syncing: Fatal exception`.

So the hotplug worker took a reference on an XDomain object that was already gone, most likely the one whose
properties could not be read. It came back at 17:26:26 by itself (`kernel.panic` = 10, `panic_on_oops` = 1), with
its webchat service. The same record has 1256 "failed request link state change" lines on incite's 2-3 at uptime
1971-2634 s (11:34-11:45), long before this session. Not reproduced; whether 7.4 has the same bug was not checked.

## State after the session

Both Macs on 7.3.0-rc7-p2p, leases released. `links-after/`: invite-enlight and invite-indite up with streams;
incite-inflight, inflight-alight, alight-indite and enlight-incite down or on firmware. To restore all six:
`scripts/tb-reset.sh incite invite indite enlight inflight alight` (or `deploy/trashcan-cluster links-reload-all --yes`
on main), then `deploy/trashcan-cluster links`.

## Tools

`tools/session.sh` (probe, churn, suite), `tools/stale-session.sh` and `tools/closestale.py` (stale-CLOSE rounds),
`../usb4stream-busypoll-20261014/tools/closecheck.py` (installed on the Macs as
`~/trashcan-exp/flashnext/stream-closecheck.py`). Monitor: reachability every 10 s and `dmesg -w` from both Macs
(`~/dev/claude-work/kernel-upstream-20261027/mon/`, not copied here).
