# USB4STREAM CLOSE handling: local 0006 (resolved for 7.4)

`kernel/patches-7.3/0006-thunderbolt-stream-io-after-close.patch` restores 7.2's behaviour after a CLOSE has been read
(7.3's e120d14d03b3 makes every later read end of file and every write -ENXIO). It conflicts with mainline's CLOSE
handling (8d55710a8c43) and is not for sending as is.

**Status (2026-10-27): not needed on 7.4.** On the 7.4 merge window the stale CLOSE that 0006 works around doesn't
reach the next session, because the lone-frame hold that carried it is gone: 500/500 churn sessions and 80/80
stale-CLOSE rounds clean without 0006 (`results/kernel/usb4stream-close-74-20261027/`). Keep 0006 in `patches-7.3/`
while the Macs run 7.3; drop it with 0007 when they move to a 7.4-based kernel.

`mail-draft.txt`: a report to Mika Westerberg with that evidence, asking whether a 7.3.y-only fix is wanted, and
mentioning the 7.3 hotplug panic seen on incite the same day. Not sent; the maintainer of this repo edits and sends it.
