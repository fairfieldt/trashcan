# thunderbolt: stream: Don't use end-to-end flow control on the Tx ring

One patch to `drivers/thunderbolt/stream.c`, the upstream form of `kernel/patches-7.3/0005-thunderbolt-stream-no-tx-e2e.patch`.
USB4STREAM's Tx ring uses E2E flow control, never gets credits on Falcon Ridge, and the host router then stops
answering config reads until the NHI is reset; thunderbolt_net hit the same failure and reverted Tx E2E in
1881f2efbf7f. The patch keeps E2E on the Rx ring only. Not sent.

This was 1/2 of the earlier `usb4stream-falcon-ridge` draft. It is now a submission of its own so that the
hardware-tested fix doesn't wait for the HopID fix (`../thunderbolt-stream-hopid/`), which has no hardware run.

**Status: ready for maintainer review.** A fix (`Fixes: 6db21d817b43`, v7.2); can go to Mika's fixes branch once it
is on v7.4-rc1.

## Base

`git://git.kernel.org/pub/scm/linux/kernel/git/westeri/thunderbolt.git` `next` at
a93a8e3200002f0c345fec4c340390e9a60aabc7 ("thunderbolt: stream: Make read return framing error to the userspace"),
fetched 2026-10-26. Also applies to mainline d135ab453a74 (which has merged that `next`; its `stream.c` differs only by
the kmalloc_obj conversions and FMODE_NOWAIT). It does not apply to the current `fixes` branch (96a8479804e3), which
predates the RX-polling change in the context; after v7.4-rc1 both branches will contain it.

## Changes since the 2026-10-06 draft

- Rebased on thunderbolt `next` (applies as before), sent alone instead of as 1/2.
- `Assisted-by` removed (see `../README.md`); notes below `---` (`notes/1.txt`) state where it was tested, that it
  was not run on this base, and the tool use.
- Cc list from get_maintainer now includes Alan Borzeszkowski (author of 6db21d817b43).

## Checks (2026-10-26, kbuild VM)

`../check-series.sh thunderbolt-stream-tx-e2e LOGDIR` (defconfig + `CONFIGFS_FS`, `USB4=m`, `USB4_CONFIGFS=m`,
`USB4_STREAM=m`). Logs in `results/kernel/upstream-20261026/thunderbolt-stream-tx-e2e/`: `drivers/thunderbolt/stream.o`
with W=1 and sparse, no warnings before or after; checkpatch `--strict` only `MISSING_SIGN_OFF`; get_maintainer:
Andreas Noever, Mika Westerberg, Yehezkel Bernat (THUNDERBOLT DRIVER), Alan Borzeszkowski (blamed fix), linux-usb,
linux-kernel.

## Hardware tested

- incite (Mac Pro 6,1, three Falcon Ridge 4C controllers cabled to each other, software connection manager), 7.2.9,
  2026-10-06: stock 0/10 streams, controller wedged (`results/kernel/tb-loopback-20261006/runs/g-bisect-stock`;
  dynamic debug of the E2E Tx ring in `runs/h-stream-once-1/dmesg.txt`); with the change streams carry data both ways,
  64 B round trips 25 us p50, 9.6 Gb/s one way per cable, 15.9 Gb/s over three cables, no config read timeouts
  (`runs/h-stream-once-rxe2e-2`, `runs/f-*`, `runs/rev2/`).
- Six Mac Pros, 7.3.0-rc7-p2p-s2 (with 0005) since 2026-10-14: the inter-Mac stream links of the lab cluster run on it
  (`results/kernel/usb4stream-busypoll-20261014/`, rollout section).

## Test limits

- Not run on this base: mainline/`next` `stream.c` has the RX polling (20bea8ceeb6a), busy-polling lock and CLOSE
  changes merged for 7.4, which the tested kernels lack. The changed lines and their meaning are the same; the
  combination with those changes is build-tested only. No new hardware test was made for this refresh (it could wedge
  the shared links).
- Only Falcon Ridge (Thunderbolt 2 cabling). Other host routers that do return Tx E2E credits lose nothing that
  thunderbolt_net doesn't already do without, but none was tested.

## Before sending

1. Read the patch and its notes (first person, for you).
2. Sign off: `perl -0pi -e 's/\n---\n/\nSigned-off-by: Tom Fairfield <tom\@fairfieldt.com>\n---\n/' series/0001-*.patch`
3. `./scripts/checkpatch.pl --strict PATH/series/0001-*.patch` in a kernel tree, then send (`--dry-run` first):

```sh
git send-email --to="Mika Westerberg <westeri@kernel.org>" --to=linux-usb@vger.kernel.org \
  --cc="Andreas Noever <andreas.noever@gmail.com>" --cc="Yehezkel Bernat <YehezkelShB@gmail.com>" \
  --cc="Alan Borzeszkowski <alan.borzeszkowski@linux.intel.com>" --cc=linux-kernel@vger.kernel.org series/0001-*.patch
```

Optionally also `--cc="Fan Ye <fy15309206903@gmail.com>"` (author of the thunderbolt_net revert 1881f2efbf7f the
patch follows).
