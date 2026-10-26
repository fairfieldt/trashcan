# Apple SSD stalls on the node Macs: cache flushes (2026-10-26)

Cold starts of the six-Mac Flash-Next chain saw the node SSDs answer nothing for 1-8 s
([link-fallback-20261026](../../deploy/link-fallback-20261026/README.md)). Cause: the drives' **cache flush**. With
the volatile write cache on, a flush during heavy reads usually takes ~30 ms, but every few seconds one takes
0.3-1.2 s, during which **no I/O completes at all**; consecutive slow flushes stop the disk for seconds. The fix is to
turn the drive's write cache off, after which the kernel sends no flushes: main `344c0d12` (`install.sh` component
`disk`, udev rule [60-trashcan-apple-ssd.rules](60-trashcan-apple-ssd.rules)).

## The disks

[survey/](survey/) (lsblk, sysfs queue settings, mounts, smartctl, kernel log, all six, idle, 15:53):

| Node | Disk | Firmware | Power-on h | Host writes | /home inode tables |
|---|---|---|---|---|---|
| incite | Intel SSDPEKNW010T8 NVMe (FUA, no issue) | | | | |
| invite | APPLE SSD SM1024F | UXM6JA1Q | 48,312 | 83 TiB | zeroed |
| indite | APPLE SSD SM1024F | UXM8JA1Q | 35,553 | 61 TiB | 3611/6623 not zeroed, `noinit_itable` (paused) |
| enlight | APPLE SSD SM1024F | UXM8JA1Q | 54,970 | 192 TiB | 3855/6619 not zeroed, `noinit_itable` (paused) |
| inflight | APPLE SSD SM1024F | UXM8JA1Q | 62,395 | 38 TiB | zeroed |
| alight | APPLE SSD SM1024F | UXM6JA1Q | 41,621 | 24 TiB | zeroed |

All five Apple drives: SATA behind AHCI (`ata1: 6.0 Gbps`, NCQ depth 32), `Write cache: enabled ... doesn't support
DPO or FUA` (so every fsync/journal commit is a separate non-queued FLUSH CACHE that libata runs with the queue
drained), mq-deadline, nr_requests 64, read_ahead 128 KiB, max_sectors 4096 KiB, wbt 2 ms, ALPM `max_performance`
(no link power management), no `discard` mount option (weekly `fstrim.timer` only), swap present but 0 used, no
reallocated or pending sectors. Write-zeroes absent (the `itable` story in nodes/macpro/README.md); ext4lazyinit is
not running anywhere (alight, the 8 s staller, is fully zeroed), so it is not the cause.

Background flush sources on every node: journald with `SyncIntervalSec=1s` (`90-trashcan-crash.conf`; each sync
also puts the journal files offline, fsync twice; journald had written 44 GB in 2 h of uptime on alight) and
`system-crash-capture.py` fsyncing telemetry every second. Idle: ~3 flushes/s at ~5 ms each, 2 MB/s written.

## Reproduction

[tools/repro.sh](tools/repro.sh) OUTDIR: drop the page cache; 4 parallel `dd` readers, 5 GB each from the Flash-Next
shards (20 GB, a node's cold load); an fsync writer (4 KiB append + fsync every 100 ms; `FSYNC=0` leaves it out);
a probe doing a 4 KiB O_DIRECT read at a random offset of another shard every 50 ms (what a major page fault waits
for); `WRITER=1` adds a 1 GiB buffered write + sync. [tools/sampler.py](tools/sampler.py) logs diskstats (reads,
writes, flushes, average flush time, in-flight) and IO PSI every 0.25 s (`disk.log` in each run). gpuq leases on
alight (#156) and inflight (#139) throughout; nothing else ran.

| Run (alight UXM6 / inflight UXM8) | Load time | Time with reads < 20 MB/s | Longest 4 KiB read | Longest fsync |
|---|---|---|---|---|
| write back (default), alight base-1, base-2 | 25.6, 38.9 s | 6.0, 13.0 s | 1229, 1313 ms | 1367, 2511 ms |
| write back, inflight base-1, base-2 | 23.2, 25.6 s | 4.0, 5.75 s | 1203, 1226 ms | 1371, 1339 ms |
| write back + WRITER, alight | 31.6 s | 10.0 s | 1275 ms | 2661 ms |
| **write through**, alight wt-1, wt-2, installed-1 | 17.9, 17.3, 17.3 s | 0 | 39, 43, 39 ms | 118, 94, 124 ms |
| **write through**, inflight wt-1, wt-2 | 17.4, 17.3 s | 0 | 48, 37 ms | 122, 98 ms |
| **write through** + WRITER, alight wt, installed / inflight | 19.3, 19.3 / 19.3 s | 0 | 54, 40 / 52 ms | 260, 277 / 277 ms |

The timeline in `runs/alight/base-2/disk.log` shows the mechanism: ~1.17 GB/s with ~20 flushes/s at 28 ms; then a
quarter-second sample with a flush outstanding and 0 reads, 11-15 commands in flight, IO PSI 100 %, for 0.5-1.25 s,
and the flush that ends it reports 425-1209 ms (several at ~610 ms). Stall time grows with the number of flushes
issued during the reads.

Background only (`FSYNC=0`, 8 runs per setting over both nodes, [alight-bg.txt](alight-bg.txt),
[inflight-bg.txt](inflight-bg.txt)): with ~1.5 flushes/s the stall is rarer: write back 1 of 8 runs had a 742 ms read
(plus one 1036 ms read in alight nofsync-2), write through 0 of 8 (max 62 ms). Load time is the same (16.6-16.7 s).
The real cold start adds the engine's logs, pid files and the launchers' writes, hence the stalls seen there.

Other settings, inflight, same reproduction with the fsync writer ([inflight-matrix.txt](inflight-matrix.txt)):

| Setting | Result |
|---|---|
| `device/queue_depth` 1 (no NCQ) | worse: fsyncs up to 10.2 s, 30 reads > 500 ms |
| `max_sectors_kb` 512 | no help: 3-5 s with no reads, reads up to 1.19 s |
| scheduler `none` | worse: 23 s with no reads, load 48.9 s |
| `cache_type` write through | fixes it (above) |

Firmware: UXM6JA1Q (alight, invite) and UXM8JA1Q (inflight, indite, enlight) behave the same. TRIM, ALPM, swap,
readahead and inode-table zeroing are ruled out above (none active during the stalls).

## Cost of write through ([alight-wbench.txt](alight-wbench.txt), idle, [tools/wbench.sh](tools/wbench.sh))

| | write back | write through |
|---|---|---|
| 4 GiB sequential write + fdatasync | 472, 621 MB/s | 362, 366 MB/s |
| 4 KiB append + fsync (x200) | p50 9 ms, max 1.19-1.21 s | p50 3.7 ms, max 4.3 ms |
| 3000 x 16 KiB files + sync | 1.60, 1.54 s | 0.52, 0.51 s |

Model copies onto a node (lan-copy, ~505 MB/s before, disk-bound) will be ~30 % slower. Durability improves (no
volatile cache to lose on a power cut).

## State left behind

- alight: `install.sh --skip tblink,fan,network,power,gpu,logs,panic,ssh --apply` from main's change (rule in
  `/etc/udev/rules.d/60-trashcan-apple-ssd.rules`, `/var/lib/trashcan-node/files`), write cache off now and at boot.
  The rule was checked with `udevadm trigger --action=add` after setting write back by hand; `--uninstall` was
  checked too (it restores write back) and the component re-applied.
- inflight: all settings restored (write back, queue_depth 32, max_sectors 4096, mq-deadline).
- invite, indite, enlight, incite: untouched.

## Not done / next

- Apply on the other four Apple-SSD nodes (`sudo nodes/macpro/install.sh --apply`, or `--skip` everything but `disk`)
  and repeat the six-Mac cold smokes (`link-fallback-20261026/tools/smokes.sh` with `COLD=1`) to confirm end to end.
- indite and enlight still have `noinit_itable`; finishing `itable` there is independent of this.
