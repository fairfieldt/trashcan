# The Mac Pro kernel

The Mac Pros run **Linux 7.3 from kernel.org, 7.3-rc7, built as `7.3.0-rc7-p2p`** with local patches
(`kernel/patches-7.3/`) and the config in `kernel/configs/`. The engine depends on 0001-0007, and the measurements in
these docs were taken on a kernel with 0001-0007 only (built as `7.3.0-rc7-p2p-s2`; same patch code and config).
0009 and 0010 (the applesmc port lights) are optional and newer; the engine does not need them. They have been
tested as an applesmc module on 7.3.0-rc7 kernels without them (`lights/driver.sh`, [lights.md](lights.md)), which
needs no new kernel. `kernel/build-7.3.sh` builds one of two patch sets:

| `KSET` | Patches | Kernel release | Patch set id | Status |
|---|---|---|---|---|
| `engine` (default) | 0001, 0003-0007 | `7.3.0-rc7-p2p` | `9c9a5449` | the patches every published measurement ran on |
| `lights` | 0001, 0003-0007, 0009, 0010 | `7.3.0-rc7-p2p-lights` | `86bb27d4` | not qualified as a kernel |

The id is the first 8 hex digits of the sha256 of `sha256sum` over the set's patch files, in order. It ends the
package version (`7.3.0~rc7-1p2p+ps9c9a5449`), so `dpkg -l` and `deploy/trashcan-cluster preflight` tell which
patches a kernel carries; [deploy/manifest.toml](../deploy/manifest.toml) lists the ids and file hashes. Debian's own
kernels stay in the GRUB menu as a fallback, but without the patches the D700s run at 300/150 MHz and P2P DMA
between them is refused ([hardware.md](hardware.md#clock-bug)).

7.3 is needed for USB4STREAM busy polling (`busy_poll` in each stream's configfs group).

## Patches (`kernel/patches-7.3/`)

| Patch | What it does | Why we need it | Upstream |
|---|---|---|---|
| 0001 `amdgpu-vfct-exact-match-first` | Pick the VFCT VBIOS image whose PCI bus matches exactly, before mainline's soft filter | Otherwise both D700s get 02:00.0's VBIOS and the second card misbehaves | Not yet submitted |
| 0003 `p2pdma-whitelist-ivy-bridge-ep` | Add the Xeon E5 v2 host bridge (8086:0e00) to the P2PDMA whitelist | Without it dma-buf imports between the D700s migrate to system memory | Not yet submitted |
| 0004 `thunderbolt-net-throttling-param` | `thunderbolt_net` ring interrupt throttle as the `throttling` module parameter (default 128000 ns) | 128 us sets back-to-back latency; 32000 gives 40 us round trips at the same throughput | Local tuning knob |
| 0005 `thunderbolt-stream-no-tx-e2e` | USB4STREAM Tx ring without E2E flow control (Rx keeps it) | With Tx-side E2E the Falcon Ridge NHI wedges and stops answering config reads | Not yet submitted; the same failure was fixed upstream in `thunderbolt_net` with revert 1881f2efbf7f |
| 0006 `thunderbolt-stream-io-after-close` | A CLOSE that has been read no longer fails every later read (end of file) and write (`ENXIO`) until the device is closed, as in 7.2 | On 7.3 a CLOSE left from the previous session kept links off their streams: 9 of 25 chain links needed retries, 2 never got there | Candidate (`Fixes: e120d14d03b3`), not sent |
| 0007 `thunderbolt-stream-no-rx-flush-on-close` | A close stops the RX ring without waiting for it to drain | The RX ring never drains (its buffers stay posted), so every close took 500 ms and a reopen waited for it | Candidate (`Fixes: 6db21d817b43`), not sent |
| 0009 `applesmc-mac-pro-port-lights` | The I/O panel's 20 port lights as LED class devices (`smc::usb`, `smc::power`, ...) plus `port_lights`, `port_lights_motion`, `port_lights_hold_ms`, `port_lights_show` (Hello, You Again) and `power_light` on the applesmc device, documented in `Documentation/ABI/testing/sysfs-devices-platform-applesmc` | Optional: control of the port lights ([lights.md](lights.md)); not needed by the engine | Not yet submitted |
| 0010 `applesmc-port-lights-levels` | `port_lights_levels`: all 20 LEDs in one write, sending only the changed ones | Optional, on top of 0009: setting all 20 up to 40 times a second (8.1 ms for all 20, 2.2 ms for 5); kept apart from 0009 because it duplicates the LED class path and needs its own justification upstream | Not yet submitted |

The numbering skips 0002 and 0008. 0002 (`si-dpm-gpio-dc-mobile-only`, without which the cards stay in the DC power
state at 300/150 MHz) is upstream in 7.3 as cd883d0a3937, with its companion 08c9100f6529 ("Fix updating clock limits
on AC/DC").

## Config

`kernel/configs/7.3.0-rc7-p2p.config`. `kernel/build-7.3.sh` runs it through `make olddefconfig`, reasserts the
settings below and stops if an option we rely on is missing (amdgpu with SI, `DRM_VGEM`, netconsole, the USB4
modules, P2PDMA).

| Option | Value | Why |
|---|---|---|
| `PCI_P2PDMA` | `y` | P2P DMA between the D700s' BARs |
| `USB4_STREAM` | `m` | USB4STREAM |
| `DEBUG_INFO_*`, `DEBUG_INFO_BTF` | off | Halves build time |
| `LOCALVERSION` | `-p2p` (`-p2p-lights` with `KSET=lights`) | Kernel release `7.3.0-rc7-p2p` |

## Building

`kernel/build-7.3.sh` downloads the kernel.org tarball, checks its sha256, applies the patch set, configures and
builds Debian packages with `make bindeb-pkg`. Build natively on Debian 13 x86_64 (a Mac Pro takes about 40
minutes), or cross-compile on any other Debian 13 host with Debian's `gcc-x86-64-linux-gnu`: the script sets
`ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-` itself when the host is not x86_64. A Debian 13 arm64 VM on a Mac
(e.g. Lima) works; keep the tree on the VM's own disk, since macOS filesystems are case-insensitive. Build
dependencies:

```bash
sudo apt install build-essential bc bison flex cpio rsync kmod python3 libssl-dev libelf-dev debhelper curl
sudo apt install gcc-x86-64-linux-gnu      # cross-compiling only
```

Then:

```bash
REV=1 kernel/build-7.3.sh                  # KSET=lights for the port-light patches
```

It leaves in `WORK` (default `~/kernel/7.3`):

- `linux-image-7.3.0-rc7-p2p_7.3.0~rc7-1p2p+ps9c9a5449_amd64.deb` and the matching `linux-headers-` package,
- `SHA256SUMS` for the two packages,
- `kernel-build-7.3.0-rc7-p2p.txt`: release, package version, patch set and the sha256 of the source tarball, base
  config, resulting config, every patch and both packages, the compiler and the build time,
- `config-7.3.0-rc7-p2p`.

Keep the three files together; `kernel/install-kernel.sh` checks them. The packages themselves are not
bit-for-bit reproducible (they carry build times), so compare builds by `kernel-build-*.txt`: source, config and
patch hashes. Bump `REV` for every rebuild so `dpkg -i` upgrades in place. `JOBS` overrides `make -j` (default
`nproc`); `KLOCAL` changes the release suffix, so a test build such as `KLOCAL=-p2p-test` installs beside the
running kernel. For 7.3 final (or a later 7.3.N), add the tarball's sha256 from kernel.org's signed
`sha256sums.asc` to the table in `kernel/build-7.3.sh` and set `VER=7.3`.

A cross-built headers package has mixed host tools (e.g. objtool for the build host, fixdep for x86-64), so
out-of-tree modules don't build against it on the Mac Pro; build those on the cross host or build the kernel
natively.

## Installing and booting

Copy `WORK` (the two packages, `SHA256SUMS`, `kernel-build-*.txt`) and `kernel/install-kernel.sh` to the node, then:

```bash
sudo ./install-kernel.sh install DIR            # preview: checksums, versions, what it would run
sudo ./install-kernel.sh install DIR --apply    # install, and boot the new kernel once at the next reboot
sudo systemctl reboot                           # when nothing runs on the node
./install-kernel.sh verify DIR                  # read-only: release, package, config, command line, VBIOS, modules
sudo ./install-kernel.sh pin 7.3.0-rc7-p2p --apply   # make it GRUB's default, once verify passes
```

`nodes/macpro/provision.sh NODE kernel` does the copy and the `install --apply`. `install` refuses packages whose
checksums do not match. It installs with `dpkg --force-unsafe-io` (the original Apple SSDs take seconds per cache
flush; don't install `linux-libc-dev`, which would replace Debian's userspace kernel headers) and sets a one-time
boot (`grub-reboot`), leaving GRUB's default alone: if the new kernel does not come up, a power cycle boots the
previous default. `pin` writes the default by GRUB entry id to `/etc/default/grub.d/91-trashcan-default.cfg`, so
installing another kernel never changes what boots. `once RELEASE --apply` boots any installed kernel once (e.g.
Debian's as a fallback); `list` shows the installed kernels, the default and the next boot. None of them reboots.

The kernel command line needs `radeon.si_support=0 amdgpu.si_support=1` so amdgpu drives the D700s.
`nodes/macpro/install.sh` adds them (`/etc/default/grub.d/90-trashcan.cfg`) unless GRUB has them already.

**Unplug Thunderbolt cables that loop the machine's own controllers before rebooting.** With three cables forming a
triangle between the three controllers, the Mac hangs in firmware before the startup chime, whatever the kernel;
unplugging them lets it chime and boot at once. One cable between two controllers reboots normally. Linux shuts down
cleanly either way. After a reboot, the Mac's Thunderbolt links to other Macs can need recovery
([deploy.md](deploy.md#thunderbolt-link-recovery)).

## Checking a new kernel

- `kernel/install-kernel.sh verify DIR` passes. It checks the items below that need no GPU work.
- `uname -r` prints `7.3.0-rc7-p2p` and `/proc/cmdline` has `amdgpu.si_support=1`.
- Each card has its own VBIOS: `cat /sys/bus/pci/drivers/amdgpu/0000:*/vbios_version` gives `113-C3861XA-028`
  (02:00.0) and `113-C3861XB-028` (06:00.0).
- Under load the clocks reach 850 MHz core / 1370 MHz memory (`freq1_input`, `freq2_input` under
  `/sys/class/drm/card{0,1}/device/hwmon/hwmon*/`; `scripts/clockmon.sh` logs them). Forcing
  `power_dpm_force_performance_level` to `high` and back to `auto` should work on both cards.
- `bench/xgpu/xgpu p2p-verify` passes (see [bench/xgpu/README.md](../bench/xgpu/README.md)), and `xgpu p2p-wall`
  shows peer bandwidth.
- `scripts/test-engine.sh` passes.
- With Thunderbolt cables between Macs, `tblink links` lists every link with its `/dev/tbstreamN`.

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
  acknowledgements, CRC32C, credits and retransmission, falling back to TCP. Measured on a loopback cable: 25 us
  round trip for 64 B, 9.6-9.7 Gb/s one way per cable, 15.9 Gb/s with one stream on each of three cables at once.
- What the driver does (measured on 7.2.9 with 0005, between two Macs):
  a receiving controller reads a frame only once up to three more have arrived behind it on that stream (0, 1 or 3
  by receiving end, the same at every ring position and with the interrupt throttling at 0; a ping-pong of single
  frames waits for whatever comes next, with three more frames written right after each 46-142 us round trips from
  Python); the Tx ring has no flow control, so frames that find the other end's 256-frame ring
  full are lost without a trace; frames flagged with a CRC error or a buffer overrun are logged (`tbstream: RX CRC
  error`) and passed to the reader like good ones; `read()` returns consecutive frames back to back; frames written
  while the other end is closed pile up and arrive in a burst when it opens (past the ring's 256 frames they are
  lost), and every close sends a CLOSE, so the end that closes last leaves one waiting at the other, which reads it as
  end of file at its next open (the engine's TP lanes then fail their handshake); every close takes 500-530 ms (it
  waits for the receive ring to drain, which never happens); an fd kept open across an XDomain reconnect stays valid
  but carries nothing; a stream's configfs group is named after the service (`0-3.1`), whose index can change at a
  reconnect.
- On 7.3: without 0006, once a stream device has read a CLOSE every later read returns end of file and every write
  fails with `ENXIO` until it is closed; with 0007 a close takes under a millisecond. `busy_poll` does not release
  held frames. The hold belongs to a direction of a cable (0-3 frames, surviving Thunderbolt resets), not to the
  paths' settings, which are the same on every cable; a CLOSE it holds reaches the other end's next session.
- Between two Macs with the engine's links (link-test, every byte checked, three extra frames after every write): a
  64 B ping-pong takes 32-47 us p50 on a stream against 40-95 us over TCP, a 40 KiB exchange 79-88 us against 169-219,
  a chain token (40 KiB on, 10 KiB back) 121-160 us against 283-351, and 5 MiB one way runs at 556-604 MB/s against
  292-470. End to end on six Macs, Flash-Next decodes 2.2% and MTP 2.0% faster on the streams than over TCP; Qwen3.8-27B
  TP4 on two Macs decodes 10.7% faster (MTP 4.0% slower in one pair). The old lanes only exchanged: 13.5 us for 64 B
  and 21.8 us for 20 KiB over three streams, against 38 and 71-116 us over TCP.
- Fixes worth making in the driver (proposed, not applied; the engine works without them):

  | Fix | Expected benefit |
  |---|---|
  | Read a received frame as soon as it arrives (7.3's `busy_poll`, which polls the rings, does not: the frame is not completed) | No extra frames after each write; request/reply traffic at the stream's latency |
  | Flush only the Tx ring on release (the Rx ring's in-flight list never empties): done in 0007 | A close in milliseconds instead of at least 0.5 s: faster session turnover, `rmmod` and resets not held up |
  | Re-post frames flagged with a CRC error or a buffer overrun instead of returning their bytes, and count them (a configfs attribute) | Damaged data never reaches a reader; the loss shows as a gap and in a counter instead of in dmesg |
  | When the XDomain goes away under an open fd, stop its rings and fail later reads, writes and polls (`-ENXIO`/`EPOLLHUP`) until it is reopened | A dead stream is reported at once instead of found by timeouts |
  | Name configfs groups by the XDomain and the service key, or re-attach a group whose service index changed | Streams survive reconnects without being recreated |

- Known upstream bug: removing a stream directory after a failed start warns in `ida_free` from
  `tbstream_dev_release` (not fixed here).

## 7.3 and our tools

7.3 moved applesmc's fan files into hwmon ([hardware.md](hardware.md#fan-control)); `fancurve.sh` handles both
layouts. The other sysfs and debugfs paths our tools use are unchanged, and amdgpu's job timeout is still 2 s.

## When something goes wrong

- To keep the kernel messages of a crash, stream them to another machine with netconsole (built into the kernel).
- `/etc/sysctl.d/90-trashcan-panic.conf`: an oops, hard lockup or soft lockup panics, and a panic reboots after
  10 s. Apple's firmware disables the iTCO hardware watchdog (`NO_REBOOT`), so this is the only automatic recovery.
- After an oops or with hung tasks, `systemctl reboot` can wait forever on stuck processes. Use
  `sudo /usr/local/sbin/emergency-reboot` (SysRq sync, remount read-only, reset) instead.
- A Thunderbolt controller that stops answering config reads: unload the Thunderbolt modules and write `1` to
  `/sys/bus/pci/devices/0000:{16,5d,a4}:00.0/reset` (what `scripts/tb-reset.sh` does). If that fails, reboot. Never
  remove the PCI devices or reset the PLX downstream ports by hand: that crashed the PCI core
  (`pcie_update_aspm_capable` NULL dereference), and the following reboot stalled on a stop job and needed a power
  cycle.
- A reboot that never reaches the startup chime with Thunderbolt cables attached is the firmware hang above, not a
  kernel problem.
- Do not reload the Thunderbolt modules on a node that is part of a run. A `tblink reload` once left a node's links
  down ("failed request link state change, aborting" every second), resets of a neighbouring pair did not bring them
  back, and the reset of all six Macs that followed hung that node in a NULL pointer dereference in
  `tb_xdp_handle_request` after "failed to allocate host router"; it needed a power cycle.
