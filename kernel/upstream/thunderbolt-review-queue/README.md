# Thunderbolt review queue: local 0004, 0006, 0007 (not for sending as is; all three resolved 2026-10-27)

The lab kernel carries three Thunderbolt patches that are not in the two stream submissions
(`../thunderbolt-stream-tx-e2e/`, `../thunderbolt-stream-hopid/`). Checked against mainline d135ab453a74 (2026-10-26,
which contains thunderbolt.git `next` a93a8e320000 and `fixes` 96a8479804e3) by reading the code and `patch
--dry-run`. No hardware tests were run for this review: the ones each item needs could wedge the shared links, so they
are listed for a planned session instead.

| Patch | Mainline today | Recommendation |
|---|---|---|
| 0007 no RX flush on close | **Superseded**: 20bea8ceeb6a ("thunderbolt: stream: Use polling with RX ring", 7.4) removed the RX `tb_ring_flush()` | Drop from the upstream queue; keep locally while on 7.3 |
| 0006 I/O after CLOSE | **Conflicts** with 8d55710a8c43 ("Improve CLOSE handling on write side", 7.4); doesn't apply | **Resolved 2026-10-27: not needed on 7.4** (the stale CLOSE no longer reaches the next session; `../thunderbolt-stream-close/`). Keep locally while on 7.3; ask about 7.3.y in the mail draft there |
| 0004 thunderbolt_net throttling parameter | Applies; 128 us still hard-coded | **Replaced 2026-10-27** by `../thunderbolt-net-coalesce/` (ethtool coalescing, ready for review); keep 0004 locally until that is merged |

## 0007: close waits 500 ms for the RX ring

Local fix: `tbstream_dev_stop()` no longer calls `tb_ring_flush(rx, 500)`, which always timed out because an RX ring
keeps its buffers posted (every close took 500 ms and a reopen waited for it; with 0007 a close takes under 1 ms,
`results/kernel/usb4stream-busypoll-20261014/`).

In mainline the RX ring is polled and `tbstream_dev_stop()` flushes only the TX ring (`stream.c` 707-745), so the
problem is gone from 7.4 on. 0007 would only matter for a 7.3.y stable backport. Stable takes mainline commits;
20bea8ceeb6a is a latency feature with its own dependencies, so a 7.3.y-only fix would have to name it and explain the
difference. Not worth proposing unless someone else hits the 500 ms close on 7.3. Keep 0007 in `patches-7.3/` until
the lab moves to a 7.4-based kernel, then drop it.

## 0006: a read CLOSE fails later I/O

Problem (7.3, since e120d14d03b3 "Add support for busy polling"): once a CLOSE has been read, every read returns end
of file and every write `-ENXIO` until the device is closed. On these Falcon Ridge hosts a CLOSE that reaches an end
while it is still open or closing, and isn't read before it closes, is there for the next session, so the next
session fails, and two hosts that retry can keep failing each other. Measured
(`results/kernel/usb4stream-busypoll-20261014/`): on 7.3, 9 of 25 chain links needed retries and 2 never got a stream;
on one cable 9 of 20 back-to-back sessions reached their stream within 100 ms on 7.3 and 192 of 200 with 0006; with
0006, 0007 and the link's later changes (`7.3.0-rc7-p2p-s2`), 300 of 300 sessions across all six links did. 0006 restores
7.2's behaviour: `closed` is checked only where a blocking read or write would otherwise wait.

Mainline changed CLOSE handling on purpose since: 8d55710a8c43 adds `close_received`, set by the RX callback when a
CLOSE lands in the ring (before anyone reads it), returns `-EPIPE` from `write()` once it is set and reports `EPOLLHUP`;
`read()` still returns end of file from the first CLOSE until the device is reopened (`closed`), and both flags are
cleared only on the first open. So upstream treats a CLOSE as the end of the session for both directions, and on
Falcon Ridge a stale CLOSE from the previous session would now fail writes even without being read. 0006 doesn't apply
and its approach contradicts the new semantics.

Open API questions for Mika Westerberg / linux-usb before writing a mainline patch:

1. Is a CLOSE meant to end the session for good (current mainline), or only to give the reader end of file at that
   point in the stream (7.2)? Should a writer get `-EPIPE` for a CLOSE it hasn't read?
2. How should a newly opened end tell a CLOSE left over from the previous session from one sent in this session?
   Options: discard RX frames that arrive before the first frame of the new session; drain the RX ring at open; a
   session/epoch number in the frame (a protocol change for both peers); a CLOSE handshake on open.
3. Is a CLOSE that arrives while an end is closing surviving into its next open expected controller behaviour,
   or specific to Falcon Ridge's E2E/path setup? (One sent to an already closed end is dropped:
   `tools/closecheck.py`, `cable-1/close-*.jsonl` in the results above.)

Tests needed for any mainline proposal (on a 7.4-based kernel, under leases, one cable at a time, with the reset
procedure ready): old-session CLOSE followed by new-session traffic each way; blocking and non-blocking (`O_NONBLOCK`,
`IOCB_NOWAIT`) reads and writes; busy-poll mode; `poll()` `EPOLLHUP`; and the 200-session cable test from the
evidence above, before and after.

## 0004: thunderbolt_net ring interrupt throttling

Local patch: `throttling` module parameter for `thunderbolt_net`'s ring interrupt throttle (`TBNET_THROTTLING`, 128000
ns). The nodes set 32000 (`nodes/macpro/root/etc/modprobe.d/thunderbolt-net.conf`).

Measured rationale (incite, 7.2.9, `results/kernel/tb-loopback-20261006/`): sockperf round trip p50 127.8 us at 128 us
and 40.2 us at 32 us, with the same throughput (9.23/9.21 Gb/s one way, 7.90+7.87 / 7.87+7.88 Gb/s both ways). On 6.12
the NHI sweep showed latency flat below 32 us and bidirectional throughput dropping sharply below 8 us.

Upstream API questions (netdev and Thunderbolt maintainers):

1. netdev doesn't take new module parameters for this; the usual interface is ethtool interrupt coalescing
   (`ethtool -C thunderbolt0 rx-usecs N tx-usecs N`, `.get_coalesce`/`.set_coalesce`). `tb_ring_throttling()` returns
   `-EBUSY` on a running ring, so a change would apply at the next ring start (login) or need the rings restarted:
   acceptable to netdev?
2. Or change the default to 32 us? That needs measurements beyond Falcon Ridge (Titan Ridge, USB4 hosts) and CPU
   cost data, which the lab can't provide.

Status: no patch drafted for either option. Until decided, 0004 stays a local tuning knob.
