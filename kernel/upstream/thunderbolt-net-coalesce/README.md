# net: thunderbolt: Make ring interrupt throttling configurable with ethtool

One patch to `drivers/net/thunderbolt/main.c`, replacing the local module parameter of
`kernel/patches-7.3/0004-thunderbolt-net-throttling-param.patch` with the interface netdev uses for this: ethtool
interrupt coalescing (`.get_coalesce`/`.set_coalesce`, `ETHTOOL_COALESCE_USECS`: `rx-usecs`, `tx-usecs`). The NHI only
takes a new interval while a ring is stopped (`tb_ring_throttling()` returns -EBUSY otherwise), so the driver keeps the
values and applies them in `tbnet_connected_work()` before `tb_ring_start()`; a change made while connected applies
from the next connection, and the driver says so through extack. Limit 16776 us (16-bit register in 256 ns units).
Default unchanged (128 us). Not sent.

**Status: ready for maintainer review; send after v7.4-rc1** (net-next is closed during the merge window).

## Base

`git://git.kernel.org/pub/scm/linux/kernel/git/netdev/net-next.git` `main` at 734bef548a96 ("Merge tag
'net-next-7.4'", 2026-10-22). Also applies to mainline 8a36f53fd5ec. After v7.4-rc1, fetch net-next, update
`BASE_COMMIT` in `check.conf` and rerun `../check-series.sh`.

## Checks (2026-10-27, kbuild VM)

`../check-series.sh thunderbolt-net-coalesce LOGDIR` (defconfig + `CONFIGFS_FS`, `USB4=m`, `USB4_NET=m`). Log in
`results/kernel/upstream-20261027/thunderbolt-net-coalesce/`: `drivers/net/thunderbolt/main.o` with W=1 and sparse, no
warnings before or after; checkpatch `--strict` only `MISSING_SIGN_OFF`; get_maintainer: Mika Westerberg, Yehezkel
Bernat (THUNDERBOLT NETWORK DRIVER), Andrew Lunn, David S. Miller, Eric Dumazet, Jakub Kicinski, Paolo Abeni
(NETWORKING DRIVERS), netdev, linux-kernel.

## Hardware tested

inflight and alight (Mac Pro 6,1, Falcon Ridge, software connection manager), one Thunderbolt 2 cable, kernel
`7.3.0-mw8a36f53+` (mainline 8a36f53fd5ec + this and the other packages), 2026-10-27:
`results/kernel/tb-coalesce-20261027/` - p50 round trip 127.8 us at the default, 59.1 us at 32 us, throughput
unchanged; deferred application and the range check as in the notes.

## For the nodes, if it is merged

The nodes set `options thunderbolt_net throttling=32000` (`nodes/macpro/root/etc/modprobe.d/thunderbolt-net.conf`).
With this patch they would set `ethtool -C IFACE rx-usecs 32 tx-usecs 32` when a ThunderboltIP interface appears
(before its login, so it applies to the first connection), and 0004 would be dropped from the local patch set.
