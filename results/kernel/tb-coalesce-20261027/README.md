# thunderbolt_net interrupt throttling: module parameter (0004) against ethtool coalescing (2026-10-27)

**Result.** `kernel/upstream/thunderbolt-net-coalesce/` (ethtool `rx-usecs`/`tx-usecs` for thunderbolt_net, applied
when the rings start) replaces the local module parameter of `kernel/patches-7.3/0004`. On the inflight-alight cable,
7.4 merge-window kernel `7.3.0-mw8a36f53+`, both ends holding a 0 us CPU latency request: 64 B round trip p50
127.8 us at the default 128 us and 59.1 us at 32 us (`ethtool -C thunderbolt0 rx-usecs 32 tx-usecs 32` on both ends,
then a reconnect), with the same one-way throughput. A value set while connected applies from the next connection
(58.5 us kept after setting 128, 127.8 us after reconnecting), and values above the 16-bit register are refused.

| Tag | Kernel, setting | sockperf 64 B p50 / avg (us) | iperf3 inflight->alight / alight->inflight (Gb/s, 3 x 10 s) |
|---|---|---|---|
| `73-param32000-nolat/` | 7.3.0-rc7-p2p, module parameter 32000 ns, no latency request | 81.7 / 82.4 | 9.02 8.51 8.64 / 8.74 8.73 9.11 |
| `73-param32000/` | 7.3.0-rc7-p2p, module parameter 32000 ns | 81.9 / 80.7 | 7.19 8.06 7.35 / 8.76 7.85 7.19 |
| `74-default128/` | 7.4 base, default (128 us) | 127.8 / 127.7 | 9.98 10.06 8.57 / 9.48 9.82 9.74 |
| `74-ethtool32/` | 7.4 base, ethtool 32 us, reconnected | 59.1 / 64.4 | 9.15 8.25 7.86 / 9.55 9.36 9.70 |
| `74-set128-noreconnect/` | then ethtool 128 us, still connected | 58.5 / 60.4 | (in summary.json) |
| `74-set128-reconnect/` | after reconnecting | 127.8 / 127.8 | (in summary.json) |

`range-check.txt`: `ethtool -C thunderbolt0 rx-usecs 20000` fails with "interval must be at most 16776 us".

The 7.3 rows are a different kernel (7.3-rc7's thunderbolt_net and NHI against the 7.4 merge window's), so they are
context, not an A/B: the A/B is 74-default128 against 74-ethtool32 on the same boot and cable. The 2026-10-06 loopback
measurement (`../tb-loopback-20261006/`, one Mac, 7.2.9, pinned IRQs) gave 127.8 / 40.2 us.

Reconnect: `ip link set thunderbolt0 down; ip link set thunderbolt0 up` on both ends (ndo_stop/ndo_open, a new
ThunderboltIP login, the rings started again). Tool: `net-bench.sh TAG` (run from the Mac).

Not done: the nodes still set `options thunderbolt_net throttling=32000`
(`nodes/macpro/root/etc/modprobe.d/thunderbolt-net.conf`); on a kernel with this patch that option is ignored
("unknown parameter 'throttling' ignored") and the node would need `ethtool -C` (e.g. a udev rule) instead.
