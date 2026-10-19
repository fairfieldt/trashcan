# The Mac Pros' kernel

All six Mac Pros run **Linux 7.3 from kernel.org, now 7.3-rc7, built as `7.3.0-rc7-p2p-s2`**, with six local patches
(0001, 0003-0007) and 7.2.9's config. The previous build, `7.3.0-rc7-p2p` (without 0006 and 0007), and `7.2.9-p2p`
stay installed as fallbacks, and Debian's own kernels are still in the GRUB menu. `7.3.0-rc7-p2p-s2` is
`REV=3 KLOCAL=-p2p-s2 kernel/cross-build-7.3.sh` (package `7.3.0~rc7-3p2p`); its rollout is in
`results/kernel/usb4stream-busypoll-20261014/` ("Rollout of 7.3.0-rc7-p2p-s2").

7.3 is needed for USB4STREAM busy polling (`busy_poll` in each stream's configfs group). On the D700s it measures the
same as 7.2.9: same VBIOS selection and clocks, P2P passes, the engine test suite passes with the same errors, and
Flash-Next's stage times, P2P bandwidth and the six-Mac main suite match 7.2.9's (`results/kernel/linux-7.3-20261014/`).

## Patches (`kernel/patches-7.3/`)

| Patch | What it does | Why we need it | Upstream |
|---|---|---|---|
| 0001 `amdgpu-vfct-exact-match-first` | Pick the VFCT VBIOS image whose PCI bus matches exactly, before mainline's soft filter | Otherwise both D700s get 02:00.0's VBIOS and the second card misbehaves | Prepared in `kernel/upstream/vfct-exact-match/`, not sent |
| 0003 `p2pdma-whitelist-ivy-bridge-ep` | Add the Xeon E5 v2 host bridge (8086:0e00) to the P2PDMA whitelist | Without it dma-buf imports between the D700s migrate to system memory | Prepared in `kernel/upstream/p2pdma-ivy-bridge-ep/`, not sent |
| 0004 `thunderbolt-net-throttling-param` | `thunderbolt_net` ring interrupt throttle as the `throttling` module parameter (default 128000 ns) | 128 us sets back-to-back latency; 32000 gives 40 us round trips at the same throughput | Local tuning knob |
| 0005 `thunderbolt-stream-no-tx-e2e` | USB4STREAM Tx ring without E2E flow control (Rx keeps it) | With Tx-side E2E the Falcon Ridge NHI wedges and stops answering config reads | `kernel/upstream/usb4stream-falcon-ridge/` 0001 rebased onto 7.3's `stream.c`, not sent; same failure upstream fixed in `thunderbolt_net` with revert 1881f2efbf7f |
| 0006 `thunderbolt-stream-io-after-close` | A CLOSE that has been read no longer fails every later read (end of file) and write (`ENXIO`) until the device is closed, as in 7.2 | On 7.3 a CLOSE left from the previous session kept links off their streams: 9 of 25 chain links needed retries, 2 never got there | Candidate (`Fixes: e120d14d03b3`), not sent |
| 0007 `thunderbolt-stream-no-rx-flush-on-close` | A close stops the RX ring without waiting for it to drain | The RX ring never drains (its buffers stay posted), so every close took 500 ms and a reopen waited for it | Candidate (`Fixes: 6db21d817b43`), not sent |
| 0009 `applesmc-mac-pro-port-lights` | The I/O panel's 20 port lights as LED class devices (`smc::usb`, `smc::power`, ...) plus `port_lights`, `port_lights_motion`, `port_lights_hold_ms`, `port_lights_show` (Hello, You Again) and `power_light` on the applesmc device, documented in `Documentation/ABI/testing/sysfs-devices-platform-applesmc` | Control of the port lights (`results/kernel/port-lights-20261018/`). Not in `-p2p-s2` yet; `kernel/applesmc-override.sh` installs 0009 and 0010 as a module override (on indite since 2026-10-19) | Prepared with 0010 in `kernel/upstream/applesmc-port-lights/`, not sent |
| 0010 `applesmc-port-lights-levels` | `port_lights_levels`: all 20 LEDs in one write, sending only the changed ones | `portlights` sets all 20 up to 40 times a second (8.1 ms for all 20, 2.2 ms for 5); kept apart from 0009 because it duplicates the LED class path and needs its own justification upstream | Prepared with 0009 in `kernel/upstream/applesmc-port-lights/`, not sent |

`kernel/patches-7.3-diag/` holds diagnostic patches (now 0008, experiment switches for the stream's lone-frame
hold). `kernel/build-7.3.sh` applies them only with `DIAG=1`, and only under a release name of its own (`KLOCAL`).
They are never rolled out or sent upstream.

The numbers follow `kernel/patches-7.2/`, the fallback's patches. Its 0002 (`si-dpm-gpio-dc-mobile-only`, without
which the cards stay in the DC power state at 300/150 MHz) is upstream in 7.3 as cd883d0a3937, with its companion
08c9100f6529 ("Fix updating clock limits on AC/DC"). The status of each patch on 7.3 is in
`results/kernel/linux-7.3-20261014/README.md`.

## Config

`kernel/configs/7.2.9-p2p.config` (7.2.9's build config) run through `make olddefconfig`; the result is
`kernel/configs/7.3.0-rc7-p2p.config`. `kernel/build-7.3.sh` reasserts the settings below and stops if an option we
rely on is missing (amdgpu with SI, `DRM_VGEM`, netconsole, the USB4 modules, P2PDMA).

| Option | Value | Why |
|---|---|---|
| `PCI_P2PDMA` | `y` | P2P DMA between the D700s' BARs |
| `USB4_STREAM` | `m` | USB4STREAM |
| `DEBUG_INFO_*`, `DEBUG_INFO_BTF` | off | Halves build time |
| `LOCALVERSION` | `-p2p` | Kernel release `7.3.0-rc7-p2p` |

## Building

On the Mac, in the Lima VM `kbuild` (Debian 13 arm64 with Debian's `gcc-x86-64-linux-gnu` 14.2.0-19, the same
compiler version as the Mac Pros' own; one-time setup in `kernel/cross-build-7.2.sh`):

```bash
REV=1 HOSTS="incite invite" kernel/cross-build-7.3.sh
```

It copies `kernel/build-7.3.sh`, `kernel/patches-7.3/` and `kernel/configs/` into the VM, builds
`linux-{image,headers}-7.3.0-rc7-p2p_7.3.0~rc7-${REV}p2p_amd64.deb`, keeps them and the config in
`~/dev/claude-work/trashcan-kbuild/`, and copies the packages to each of `HOSTS` in `~/kernel/7.3/debs/`. It takes
about 5 minutes on an idle Mac; when the Mac is busy, set `JOBS=12` or so, since 30 VM threads competing with other
work slow it tenfold. Bump `REV` for every rebuild so `dpkg -i` upgrades in place. For 7.3 final (or a later 7.3.N),
add the tarball's sha256 from kernel.org's signed `sha256sums.asc` to the table in `kernel/build-7.3.sh` and set
`VER=7.3`. The headers package's host tools are mixed (objtool arm64, fixdep x86-64), so out-of-tree modules don't
build on the Macs; build them in the kbuild VM (`results/kernel/port-lights-20261018/`).

## Installing and booting

For a new Mac Pro, `nodes/macpro/provision.sh` does all of this (`nodes/macpro/README.md`). On a node, with
`results/kernel/linux-7.3-20261014/kernel-boot.sh` copied there:

```bash
./kernel-boot.sh once 7.3.0-rc7-p2p      # dpkg -i the image and headers from ~/kernel/7.3/debs, then grub-reboot
sudo systemctl reboot
./kernel-boot.sh default 7.3.0-rc7-p2p   # once it checks out: pin GRUB_DEFAULT to it
```

Don't install `linux-libc-dev`; it would replace Debian's userspace kernel headers. `kernel-boot.sh` runs `dpkg`
with `--force-unsafe-io`, since the original Apple SSDs take seconds per cache flush.

GRUB's default is pinned by entry id in `/etc/default/grub` (`GRUB_DEFAULT`), so installing a kernel never changes
what boots, and a one-time boot that fails goes back to the default after a power cycle. `GRUB_DEFAULT` has pointed
at `7.3.0-rc7-p2p-s2` on all six Macs since 2026-10-14 17:10 (before that, `7.3.0-rc7-p2p`). The kernel command line is
`quiet radeon.si_support=0 amdgpu.si_support=1 panic=10`.

**Unplug Thunderbolt cables that loop the machine's own controllers before rebooting.** With three cables forming a
triangle between the three controllers, the Mac hangs in firmware before the startup chime, rebooting from 6.12 and
7.2 alike; unplugging them lets it chime and boot at once. One cable between two controllers reboots normally. Linux
shuts down cleanly either way. After a reboot, the Mac's Thunderbolt links to the other Macs need a reset of all of
them together (`scripts/tb-reset.sh`, `docs/hardware.md`).

## Fallback: 7.2.9-p2p

`7.2.9-p2p` (package `7.2.9-2p2p`: `kernel/build-7.2.sh`, `kernel/patches-7.2/`, `results/kernel/linux-7.2-20261006/`)
is still installed on every Mac. To boot it once, find its entry id and pass it to `grub-reboot`:

```bash
sudo grep -o "gnulinux-advanced-[^']*\|gnulinux-7.2.9-p2p-advanced-[^']*" /boot/grub/grub.cfg | sort -u
sudo grub-reboot "gnulinux-advanced-ROOT>gnulinux-7.2.9-p2p-advanced-ROOT"    # ROOT: the ids printed above
sudo systemctl reboot
```

On incite that is `gnulinux-advanced-59c789b8-55b5-41a4-ae2f-fba0c8922d2b>gnulinux-7.2.9-p2p-advanced-59c789b8-55b5-41a4-ae2f-fba0c8922d2b`.
To go back to it for good, `DEBS=~/kernel/7.2/debs ./kernel-boot.sh default 7.2.9-p2p`. 7.2.9 has the old applesmc
layout and no `busy_poll`; everything else in our tools works on both.

## Checking a new kernel

In `results/kernel/linux-7.3-20261014/`: `run.sh`, under the Mac's gpuq lease, records the system state and each
card's VBIOS (`113-C3861XA-028` on 02:00.0, `113-C3861XB-028` on 06:00.0), forces DPM high and back, and runs `xgpu
p2p-verify` x3, `xgpu p2p-wall`, a Flash-Next stage benchmark and `scripts/test-engine.sh`; `check.sh` is its short
form for a rollout; `tb.sh` checks the Thunderbolt links, streams and netconsole. Compare with the directories there,
then run the six-Mac main suite (`results/qwen4/main-suite-20261013/`) against the last kernel's numbers. For one
cable's stream throughput and latency, `results/kernel/tb-loopback-20261006/` has `tbstream-bench.c`, `bench.sh` and
`stream-run.sh`.

## Thunderbolt

- `thunderbolt_net`: nodes load it with `throttling=32000` (`/etc/modprobe.d/thunderbolt-net.conf`, from
  `nodes/macpro/root/`) for 40 us round trips; the kernel default of 128000 gives 128 us. NetworkManager
  leaves `thunderbolt*` interfaces alone (`/etc/NetworkManager/conf.d/90-thunderbolt-unmanaged.conf`).
- USB4STREAM: `tblink` (`nodes/macpro/root`) loads `thunderbolt_stream` and gives every working link to another
  node the stream `trashcan` under the link's XDomain stream service, with input and output HopID 15 at both ends
  (thunderbolt_net takes the lowest free ones from 8; the controllers allow up to 15), and creates it again whenever
  it is missing or sits under a stale service name; `tblink links` shows each link's `/dev/tbstreamN`, and the devices
  belong to `render` (`/etc/udev/rules.d/70-tbstream.rules`). By hand, for one cable from `0-3` to `1-3`:

  ```bash
  cfg=/sys/kernel/config/thunderbolt/stream
  sudo mkdir -p $cfg/0-3.1/NAME $cfg/1-3.1/NAME
  echo 15 | sudo tee $cfg/0-3.1/NAME/in_hopid $cfg/0-3.1/NAME/out_hopid $cfg/1-3.1/NAME/in_hopid $cfg/1-3.1/NAME/out_hopid
  ```

  `index` in each directory gives the `/dev/tbstreamX` number. Each group also has `ring_size`, `throttling` (the
  rings' interrupt throttle, 8192 ns by default, separate from `thunderbolt_net`'s) and, new in 7.3, `busy_poll`
  (0 by default): set before the stream is opened, it runs the stream's rings without interrupts, `read()` and
  `write()` poll them, and `poll()` returns `EPOLLERR`. The engine's links (`engine/src/link`) find a link's
  stream from the TCP connection's interface and move their data to it, with their own sequence numbers,
  acknowledgements, CRC32C, credits and retransmission, falling back to TCP
  (`results/kernel/usb4stream-reliable-20261013/`). Measured on a loopback cable: 25 us round trip for 64 B, 9.6-9.7
  Gb/s one way per cable, 15.9 Gb/s with one stream on each of three cables at once
  (`results/kernel/tb-loopback-20261006/`).
- What the driver does (7.2.9 with 0005), measured between two Macs (`results/kernel/usb4stream-reliable-20261013/`):
  a receiving controller reads a frame only once up to three more have arrived behind it on that stream (0, 1 or 3
  by receiving end, the same at every ring position and with the interrupt throttling at 0; a ping-pong of single
  frames waits for whatever comes next, with three more frames written right after each 46-142 us round trips from
  Python); the Tx ring has no flow control, so frames that find the other end's 256-frame ring
  full are lost without a trace; frames flagged with a CRC error or a buffer overrun are logged (`tbstream: RX CRC
  error`) and passed to the reader like good ones; `read()` returns consecutive frames back to back; frames written
  while the other end is closed pile up and arrive in a burst when it opens (past the ring's 256 frames they are
  lost), and every close sends a CLOSE, so the end that closes last leaves one waiting at the other, which reads it as
  end of file at its next open (main's TP lanes then fail their handshake); every close takes 500-530 ms (it waits for the receive ring to drain, which never happens); an fd kept open
  across an XDomain reconnect stays valid but carries nothing; a stream's configfs group is named after the service
  (`0-3.1`), whose index can change at a reconnect.
- On 7.3 (`results/kernel/usb4stream-busypoll-20261014/`): without 0006, once a stream device has read a CLOSE every
  later read returns end of file and every write fails with `ENXIO` until it is closed; with 0007 a close takes under
  a millisecond. `busy_poll` does not release held frames. The hold belongs to a direction of a cable (0-3 frames,
  surviving Thunderbolt resets), not to the paths' settings, which are the same on every cable; a CLOSE it holds
  reaches the other end's next session. With 0008 at its defaults (`7.3.0-rc7-p2p-s3`) no cable touching that kernel
  held; what 0008 changes is being bisected.
- Between two Macs with the engine's links (link-test, every byte checked, three extra frames after every write): a
  64 B ping-pong takes 32-47 us p50 on a stream against 40-95 us over TCP, a 40 KiB exchange 79-88 us against 169-219,
  a chain token (40 KiB on, 10 KiB back) 121-160 us against 283-351, and 5 MiB one way runs at 556-604 MB/s against
  292-470. End to end on six Macs, Flash-Next decodes 2.2% and MTP 2.0% faster on the streams than over TCP; Qwen3.8-27B
  TP4 on two Macs decodes 10.7% faster (MTP 4.0% slower in one pair). The old lanes (`results/qwen/tp4-viability-20261006/`) only exchanged: 13.5 us for 64 B and 21.8 us for 20 KiB over
  three streams, against 38 and 71-116 us over TCP.
- Fixes worth making in the driver (proposed, not applied; the engine works without them):

  | Fix | Expected benefit |
  |---|---|
  | Read a received frame as soon as it arrives (7.3's `busy_poll`, which polls the rings, does not: the frame is not completed; see the 0008 bisection) | No extra frames after each write; request/reply traffic at the stream's latency |
  | Flush only the Tx ring on release (the Rx ring's in-flight list never empties): done in 0007 | A close in milliseconds instead of at least 0.5 s: faster session turnover, `rmmod` and resets not held up |
  | Re-post frames flagged with a CRC error or a buffer overrun instead of returning their bytes, and count them (a configfs attribute) | Damaged data never reaches a reader; the loss shows as a gap and in a counter instead of in dmesg |
  | When the XDomain goes away under an open fd, stop its rings and fail later reads, writes and polls (`-ENXIO`/`EPOLLHUP`) until it is reopened | A dead stream is reported at once instead of found by timeouts |
  | Name configfs groups by the XDomain and the service key, or re-attach a group whose service index changed | Streams survive reconnects without being recreated |

  The second and third are compile-checked diffs in `results/kernel/usb4stream-reliable-20261013/proposed/`.
- Known upstream bug: removing a stream directory after a failed start warns in `ida_free` from
  `tbstream_dev_release` (fix drafted in `kernel/upstream/usb4stream-falcon-ridge/`, not applied locally).

## 7.3 and our tools

7.3 moved applesmc's fan files into hwmon (`docs/hardware.md`, "Fan control"); `fancurve.sh` and the telemetry
handle both layouts. The other sysfs and debugfs paths our tools use are unchanged (`results/kernel/linux-7.3-20261014/`,
"7.3 sysfs changes"), and amdgpu's job timeout is still 2 s.

## When something goes wrong

- Kernel messages stream to the Mac: `netconsole.service` (`/usr/local/sbin/netconsole-start`) sends to the
  address in `/etc/default/netconsole-target` (port 6666) from whichever NIC holds the default route; on the Mac,
  `scripts/netconsole-listen.py` appends them, prefixed with the sender's IP, to
  `~/dev/claude-work/trashcan-kbuild/netconsole-incite.log`. If the Mac's address changes, rerun
  `nodes/macpro/provision.sh HOST system`.
- `/etc/sysctl.d/90-trashcan-panic.conf`: an oops, hard lockup or soft lockup panics, and a panic reboots after
  10 s. Apple's firmware disables the iTCO hardware watchdog (`NO_REBOOT`), so this is the only automatic recovery.
- After an oops or with hung tasks, `systemctl reboot` can wait forever on stuck processes. Use
  `sudo /usr/local/sbin/emergency-reboot` (SysRq sync, remount read-only, reset) instead.
- A Thunderbolt controller that stops answering config reads: unload the Thunderbolt modules and write `1` to
  `/sys/bus/pci/devices/0000:{16,5d,a4}:00.0/reset` (`results/kernel/tb-loopback-20261006/tb-reset.sh`). If that
  fails, reboot. Never remove the PCI devices or reset the PLX downstream ports by hand: that crashed the PCI core
  (`pcie_update_aspm_capable` NULL dereference), and the following reboot stalled on a stop job and needed a power
  cycle (three Thunderbolt cables were also attached).
- A reboot that never reaches the startup chime with Thunderbolt cables attached is the firmware hang above, not a
  kernel problem.
- Do not reload the Thunderbolt modules on a node that is part of a run. On 2026-10-13 a `tblink reload` on inflight
  left its links down ("failed request link state change, aborting" every second), two resets of a neighbouring pair
  did not bring them back, and the reset of all six Macs that followed hung inflight in a NULL pointer dereference in
  `tb_xdp_handle_request` after "failed to allocate host router" (`results/kernel/usb4stream-reliable-20261013/`); it
  needed a power cycle.
- After hard power cuts on 2026-10-06 incite's left Ethernet port (`enp12s0`) stopped getting a link; the cable is
  now in `enp11s0` (DHCP).

## Older kernels

`6.12.111-p2p`: `kernel/build.sh` builds Debian's `linux-source-6.12` with `kernel/patches/0001-0003` on a Mac Pro
(about 28 minutes). It has no USB4STREAM; its `thunderbolt_net` uses a fixed 128 us NHI throttle
(`results/kernel/tb-loopback-20261006/` shows a 32 us out-of-tree build). It is still installed on incite.
