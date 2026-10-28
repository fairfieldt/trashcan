# Test kernel 7.3.0-mw8a36f53+ (7.4 merge window), 2026-10-27

`build.sh`, run in the kbuild VM: mainline 8a36f53fd5ec ("v7.3-17277-g8a36f53fd5ec", 2026-10-27) with the upstream
packages applied by `git am` from `kernel/upstream/*/series/` (applesmc-port-lights 1-2, p2pdma-ivy-bridge-ep,
vfct-exact-match 1-2, thunderbolt-stream-tx-e2e, thunderbolt-stream-hopid, thunderbolt-net-coalesce), config as
`kernel/build-7.3.sh` (configs/7.2.9-p2p.config, olddefconfig, the same options asserted), packages
`linux-{image,headers}-7.3.0-mw8a36f53+_7.3.0+mw8a36f53fd5ec-1_amd64.deb` (staged in
`~/dev/claude-work/trashcan-kbuild/7.4mw/`, not committed). The `+` in the release comes from setlocalversion (git tree
not at a tag). Not a deployment kernel: no 0006/0007, no 0004 (replaced by the coalescing patch).

Booted once on inflight and alight (`kernel-boot.sh once`, DEBS=~/kernel/7.4mw/debs); both are back on 7.3.0-rc7-p2p.
Used for `../usb4stream-close-74-20261027/` and `../tb-coalesce-20261027/`.

Seen at boot on both Macs, not caused by our patches and not followed up:

- `WARNING: drivers/platform/x86/apple-gmux.c:884 at gmux_probe.cold+0x65/0x3d6 [apple_gmux]` during probe (gmux
  version 4.0.11 [indexed]). Not new: 7.3.0-rc7-p2p warns the same way at boot (inflight, invite), which is the `W`
  taint in incite's panic (`../usb4stream-close-74-20261027/incite-panic-1725/`). Worth a report to the apple-gmux
  maintainers.
- alight logged `thunderbolt 0-3: failed request link state change, aborting` about once a second on its cable to
  indite (7.3 at the other end), and inflight `failed to send properties changed notification` once.
- `thunderbolt_net: unknown parameter 'throttling' ignored` (the nodes' modprobe option for 0004).
- `TCP: thunderbolt0: Driver has suspect GRO implementation, TCP performance may be compromised.` after the interface
  was taken down and up.
- With the patched applesmc in the tree, the 20 port lights registered (`applesmc applesmc.768: 20 port lights`); each
  D700 loaded its own VBIOS from VFCT (113-C3861XA-028 at 02:00.0, 113-C3861XB-028 at 06:00.0).
