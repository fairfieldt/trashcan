# incite's kernel

incite runs **Linux 7.2.9 from kernel.org, built as `7.2.9-p2p`**, with five local patches and a few config
changes. Debian's `6.12.111-p2p` build (`kernel/build.sh`, `kernel/patches/`) stays installed as the fallback, and
the stock Debian kernel `6.12.111+deb13-amd64` is still in the GRUB menu.

7.2 is needed for USB4STREAM (`/dev/tbstreamX`, new in 7.2). On the D700s it is at least as good as 6.12: same
VBIOS selection and clocks, P2P passes, the engine test suite passes, and pp512 prefill is about 2% faster
(`results/kernel/linux-7.2-20261006/`).

## Patches (`kernel/patches-7.2/`)

| Patch | What it does | Why we need it | Upstream |
|---|---|---|---|
| 0001 `amdgpu-vfct-exact-match-first` | Pick the VFCT VBIOS image whose PCI bus matches exactly, before mainline's soft filter | Otherwise both D700s get 02:00.0's VBIOS and the second card misbehaves | Prepared in `kernel/upstream/vfct-exact-match/`, not sent |
| 0002 `si-dpm-gpio-dc-mobile-only` | Set `PPSMC_SYSTEMFLAG_GPIO_DC` only on mobile SI parts | Without it the cards stay in the DC (battery) power state at 300/150 MHz | Backport of Timur Kristóf's cd883d0a3937: in mainline 7.3-rc6 and tagged for stable, not in 7.2.9. Drop it when the 7.2.x in use has it (the build stops at this patch then) |
| 0003 `p2pdma-whitelist-ivy-bridge-ep` | Add the Xeon E5 v2 host bridge (8086:0e00) to the P2PDMA whitelist | Without it dma-buf imports between the D700s migrate to system memory | Prepared in `kernel/upstream/p2pdma-ivy-bridge-ep/`, not sent |
| 0004 `thunderbolt-net-throttling-param` | `thunderbolt_net` ring interrupt throttle as the `throttling` module parameter (default 128000 ns) | 128 us sets back-to-back latency; 32000 gives 40 us round trips at the same throughput | Local tuning knob |
| 0005 `thunderbolt-stream-no-tx-e2e` | USB4STREAM Tx ring without E2E flow control (Rx keeps it) | With Tx-side E2E the Falcon Ridge NHI wedges and stops answering config reads | Prepared in `kernel/upstream/usb4stream-falcon-ridge/` (with a HopID fix), not sent; same failure upstream fixed in `thunderbolt_net` with revert 1881f2efbf7f |

0001 and 0003 are the upstream-form versions of the 6.12 patches of the same names. 0002 is unchanged from 6.12 and
identical to the upstream commit. 7.2 already has the related upstream fix 86b6cf8387a4 ("Fix AC/DC switch
notification"); the companion 08c9100f6529 ("Fix updating clock limits on AC/DC", 7.3-rc6) isn't needed on the D700s.

## Config

`kernel/configs/6.12.111-p2p.config` (incite's 6.12 build: Debian's config with `PCI_P2PDMA`,
`DMABUF_MOVE_NOTIFY` and no debug info) run through `make olddefconfig`, plus:

| Option | Value | Why |
|---|---|---|
| `PCI_P2PDMA` | `y` | P2P DMA between the D700s' BARs |
| `USB4_STREAM` | `m` | USB4STREAM |
| `DEBUG_INFO_*`, `DEBUG_INFO_BTF` | off | Halves build time |
| `LOCALVERSION` | `-p2p` | Kernel release `7.2.9-p2p` |

6.12 also needed `DMABUF_MOVE_NOTIFY=y`; 7.2 has no such option and always behaves as if it were set.

## Building

On the Mac, in the Lima VM `kbuild` (Debian 13 arm64 with Debian's `gcc-x86-64-linux-gnu` 14.2.0-19, the same
compiler version as incite's native one):

```bash
REV=2 kernel/cross-build-7.2.sh
```

It copies `kernel/build-7.2.sh`, `kernel/patches-7.2/` and `kernel/configs/` into the VM, builds
`linux-{image,headers}-7.2.9-p2p_7.2.9-${REV}p2p_amd64.deb`, keeps them in `~/dev/claude-work/trashcan-kbuild/`
and copies them to `incite:~/kernel/7.2/debs/`. It takes about 5 minutes on an idle Mac (about 30 natively on
incite); when the Mac is busy, set `JOBS=12` or so, since 30 VM threads competing with other work slow it tenfold.
`kernel/build-7.2.sh` also runs natively on incite. Bump `REV` for every rebuild so `dpkg -i` upgrades in place.
The headers package contains x86-64 host tools, so out-of-tree modules build on incite.

## Installing and booting

For a new Mac Pro, `nodes/macpro/provision.sh` does all of this (`nodes/macpro/README.md`). By hand:

```bash
cd ~/kernel/7.2/debs && sudo dpkg -i linux-image-7.2.9-p2p_*.deb linux-headers-7.2.9-p2p_*.deb
```

Don't install `linux-libc-dev`; it would replace Debian's userspace kernel headers.

GRUB's default is pinned by entry id in `/etc/default/grub` (`GRUB_DEFAULT`), so installing a kernel never changes
what boots. Try a new kernel once with:

```bash
sudo grub-reboot "gnulinux-advanced-59c789b8-55b5-41a4-ae2f-fba0c8922d2b>gnulinux-7.2.9-p2p-advanced-59c789b8-55b5-41a4-ae2f-fba0c8922d2b"
sudo systemctl reboot
```

If that boot fails, a power cycle returns to the default. `GRUB_DEFAULT` has pointed at `7.2.9-p2p` since
2026-10-06. To boot the 6.12 fallback once, use the same command with `6.12.111-p2p` in place of `7.2.9-p2p`. The
kernel command line is `quiet radeon.si_support=0 amdgpu.si_support=1 panic=10`.

**Unplug Thunderbolt cables that loop the machine's own controllers before rebooting.** With three cables forming a
triangle between the three controllers, the Mac hangs in firmware before the startup chime, rebooting from 6.12 and
7.2 alike; unplugging them lets it chime and boot at once. One cable between two controllers reboots normally. Linux
shuts down cleanly either way.

## Checking a new kernel

1. Each card's VBIOS: `cat /sys/bus/pci/devices/0000:0{2,6}:00.0/vbios_version` gives `113-C3861XA-028` and
   `113-C3861XB-028`.
2. Clocks reach 850/1370 MHz under load (`scripts/clockmon.sh`).
3. `TAG=name results/kernel/linux-7.2-20261006/run.sh` (from `~/trashcan`, takes the shared lock): `xgpu p2p-verify`
   x3, `xgpu p2p-wall` and four pp512 benches with the pinned engine binary. Compare with the directories there.
4. `scripts/test-engine.sh`.
5. For Thunderbolt, copy `results/kernel/tb-loopback-20261006/` to incite, build `tbstream-bench.c`, and run
   `bench.sh` and `stream-run.sh` through `locked.sh` on a cross-controller cable.

## Thunderbolt on 7.2

- `thunderbolt_net`: nodes load it with `throttling=32000` (`/etc/modprobe.d/thunderbolt-net.conf`, from
  `nodes/macpro/root/`) for 40 us round trips; the kernel default of 128000 gives 128 us. NetworkManager
  leaves `thunderbolt*` interfaces alone (`/etc/NetworkManager/conf.d/90-thunderbolt-unmanaged.conf`).
- USB4STREAM: `sudo modprobe thunderbolt_stream` (nothing loads it automatically), then create the stream on one
  end and fill in the other end's HopIDs by hand; the far end only auto-configures if the peer's properties arrived
  before its directory was created. For one cable from `0-3` to `1-3`:

  ```bash
  cfg=/sys/kernel/config/thunderbolt/stream
  sudo mkdir -p $cfg/0-3.1/data $cfg/1-3.1/data
  echo -1 | sudo tee $cfg/0-3.1/data/in_hopid $cfg/0-3.1/data/out_hopid
  sudo cat $cfg/0-3.1/data/out_hopid | sudo tee $cfg/1-3.1/data/in_hopid
  sudo cat $cfg/0-3.1/data/in_hopid | sudo tee $cfg/1-3.1/data/out_hopid
  ```

  `index` in each directory gives the `/dev/tbstreamX` number. Measured: 25 us round trip for 64 B, 9.6-9.7 Gb/s one
  way per cable, 15.9 Gb/s with one stream on each of three cables at once (`results/kernel/tb-loopback-20261006/`).
- Known upstream bug: removing a stream directory after a failed start warns in `ida_free` from
  `tbstream_dev_release` (fix drafted in `kernel/upstream/usb4stream-falcon-ridge/`, not applied locally).

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
- After hard power cuts on 2026-10-06 the left Ethernet port (`enp12s0`) stopped getting a link; the cable is now
  in `enp11s0` (192.168.0.83).

## Fallback: 6.12.111-p2p

`kernel/build.sh` builds Debian's `linux-source-6.12` with `kernel/patches/0001-0003` on incite (about 28 minutes).
It has no USB4STREAM; its `thunderbolt_net` uses a fixed 128 us NHI throttle (`results/kernel/tb-loopback-20261006/`
shows a 32 us out-of-tree build).
