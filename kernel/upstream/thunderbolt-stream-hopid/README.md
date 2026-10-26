# thunderbolt: stream: Don't release HopIDs that were never allocated

One patch to `drivers/thunderbolt/stream.c` (2/2 of the earlier `usb4stream-falcon-ridge` draft, now a separate
submission). `tbstream_dev_attach_stream()` ignores HopID allocation failures, and `tbstream_dev_alloc_in/out_hopid()`
keep a released HopID when the new allocation fails, so detach/release later free IDs the stream doesn't own (another
user's, or unallocated: the `ida_free` WARN seen on a Mac Pro). The patch forgets HopIDs that couldn't be allocated
and clears a released one at once. No local equivalent: the lab kernels don't carry it. Not sent.

**Status: ready for maintainer review**, with the test limit below stated in the patch notes. A fix
(`Fixes: 6db21d817b43`, v7.2).

## Base

thunderbolt.git `next` at a93a8e3200002f0c345fec4c340390e9a60aabc7, fetched 2026-10-26. Also applies to mainline
d135ab453a74 and to thunderbolt `fixes` (96a8479804e3).

## Changes since the 2026-10-06 draft

- Standalone: no reference to the Tx E2E patch; the changelog says the WARN was seen but the paths were found by
  reading the code, not reproduced on purpose.
- New userspace test fixture (`harness/`), results in the notes below `---` (`notes/1.txt`) with an open question.
- `Assisted-by` removed (see `../README.md`).

## Checks (2026-10-26)

- `../check-series.sh thunderbolt-stream-hopid LOGDIR` (same config as the Tx patch). Logs in
  `results/kernel/upstream-20261026/thunderbolt-stream-hopid/`: `stream.o` W=1 and sparse clean before and after;
  checkpatch `--strict` only `MISSING_SIGN_OFF`; get_maintainer as for the Tx patch.
- `harness/run.sh STREAM_C OUTDIR` compiles `tbstream_dev_xdomain()`, `tbstream_dev_release()`,
  `tbstream_dev_alloc_in/out_hopid()`, `tbstream_dev_attach_stream()` and `tbstream_dev_detach_stream()` unchanged
  from the given `stream.c` against `shim.h`, a model of the XDomain HopID IDAs (lowest free ID in range, as
  `ida_alloc_range()`; freeing an unallocated ID counts as the `ida_free` WARN; HopIDs can be owned by another user;
  an allocation can be made to fail). Six scenarios: normal attach/detach; configured HopID held by another user;
  allocation failure on attach then release; peer-supplied HopIDs with one taken; changing a HopID to a taken one;
  changing a HopID with a failing allocation. Result (`hopid-replay.txt`): `stream.c` at the base fails 5 of 6 (three
  `ida_free` of unallocated IDs, two of another user's IDs); with the patch all 6 pass with nothing left allocated.

## Test limits

- No hardware run, by choice: reaching the failing paths on the cluster means provoking HopID conflicts or a wedged
  host router on shared links. The fixture is userspace and models `ida_alloc_range()`/`ida_free()`; it doesn't cover
  locking, configfs or the real XDomain lifecycle.
- The observed WARN (`results/kernel/tb-loopback-20261006/runs/g-bisect-stock/dmesg-tail.txt`,
  `runs/g-bisect-revert-b69af182/dmesg-tail.txt`) is consistent with these paths but was not tied to one of them.
- Behaviour change to review (also in the notes): after a failed allocation on attach the configured HopID reads back
  as 0 instead of being retried at the next attach. Keeping it would need a separate "allocated" flag.

## Before sending

1. Read the patch and its notes (first person, for you); decide on the open question.
2. Sign off: `perl -0pi -e 's/\n---\n/\nSigned-off-by: Tom Fairfield <tom\@fairfieldt.com>\n---\n/' series/0001-*.patch`
3. `./scripts/checkpatch.pl --strict PATH/series/0001-*.patch` in a kernel tree, then send (`--dry-run` first):

```sh
git send-email --to="Mika Westerberg <westeri@kernel.org>" --to=linux-usb@vger.kernel.org \
  --cc="Andreas Noever <andreas.noever@gmail.com>" --cc="Yehezkel Bernat <YehezkelShB@gmail.com>" \
  --cc="Alan Borzeszkowski <alan.borzeszkowski@linux.intel.com>" --cc=linux-kernel@vger.kernel.org series/0001-*.patch
```
