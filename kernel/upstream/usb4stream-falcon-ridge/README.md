# USB4STREAM fixes for Falcon Ridge hosts, for upstream

Two patches to `drivers/thunderbolt/stream.c`, drafted against linux-next `next-20261006` (eea3fef32a9c). Not sent.
Both carry `Assisted-by` and no `Signed-off-by`; the submitter adds their own. checkpatch `--strict` reports only
the missing sign-off.

| Patch | Problem | Local equivalent | Tested |
|---|---|---|---|
| 0001 Don't use end-to-end flow control on the Tx ring | USB4STREAM's Tx ring uses E2E, never gets credits on Falcon Ridge, and the host router then stops answering config reads | `kernel/patches-7.2/0005` | On incite with 7.2.9 (`results/kernel/tb-loopback-20261006/`) |
| 0002 Don't release HopIDs that were never allocated | Attach ignores HopID allocation failures, so release frees IDs the stream doesn't own (`ida_free` WARN) | none | Compiles on 7.2.9 with `W=1`; not run |

The series doesn't build on 7.2.9 as is: linux-next's `stream.c` has since gained receive polling and busy polling
(`RING_FLAG_NO_INTERRUPT`), which 0001's context includes.

## Sending

- Recipients (`MAINTAINERS`, THUNDERBOLT DRIVER): Mika Westerberg <westeri@kernel.org>, Andreas Noever, Yehezkel
  Bernat, linux-usb@vger.kernel.org. Worth Cc'ing Alan Borzeszkowski (co-author of `stream.c`) and Fan Ye (author of
  the `thunderbolt_net` revert 1881f2efbf7f that 0001 follows).
- Rebase onto Mika's `thunderbolt.git` `next` branch first; `stream.c` changes often.

## Evidence

- Wedge: `results/kernel/tb-loopback-20261006/runs/h-stream-once-1/dmesg.txt` (dynamic debug of one stream open on
  stock 7.2.9: Tx ring started with E2E, 0.5 s to stop, then config read timeouts) and `runs/g-bisect-stock`
  (0/10 streams, controller wedged).
- With the fix: `runs/h-stream-once-rxe2e-2`, `runs/f-*`, `runs/rev2/`.
- `ida_free` WARN: `runs/g-bisect-stock/dmesg-tail.txt` and `runs/g-bisect-revert-b69af182/dmesg-tail.txt`
  (rmdir after the controller had re-enumerated the XDomain). The cause in 0002 comes from reading the code; it was
  not reproduced on purpose.
