# Setting up a Mac Pro 6,1 node

`provision.sh` turns a Mac Pro 6,1 with a fresh Debian 13 install into a trashcan node: kernel `7.3.0-rc7-p2p`
(`docs/kernel.md`), the node services ([`install.sh`](#node-services-installsh)), the Rust toolchain, and the
trashcan workspace built and checked. It runs from your workstation over SSH, every step is idempotent, and
running it against a provisioned node changes nothing. Clusters of several nodes are described, checked, started
and stopped with `deploy/trashcan-cluster` ([docs/deploy.md](../../docs/deploy.md)).

## Before running it

1. Install Debian 13 on the Mac Pro with the SSH server, using the whole internal SSD. Give each node its own
   hostname.

   On an original Apple SSD (`APPLE SSD SM...`), the install crawls once the base system starts copying (see
   [Slow disk](#slow-disk-right-after-install-apple-ssds)). Right after partitioning, switch to a shell
   (Ctrl+Alt+F2, Enter) and run the following, adding `/target/home` if the installer made a separate `/home`.
   Switch back with Ctrl+Alt+F1 (F5 with the graphical installer).

   ```bash
   mount -o remount,noinit_itable /target
   ```

2. On the node, give your user passwordless sudo. If the installer set a root password, use `su -` first:

   ```bash
   echo "$USER ALL=(ALL) NOPASSWD:ALL" | sudo tee /etc/sudoers.d/90-$USER
   ```

3. From your workstation, authorize your SSH key:

   ```bash
   ssh-copy-id NODE
   ```

4. Build the kernel packages with `kernel/build-7.3.sh` (natively on a Debian 13 x86_64 machine or a Mac Pro, or
   cross-compiled; [docs/kernel.md](../../docs/kernel.md)) and copy its `WORK` directory (packages, `SHA256SUMS`,
   `kernel-build-7.3.0-rc7-p2p.txt`) to your workstation.
5. Unplug any Thunderbolt cable that loops the machine's own controllers; the firmware hangs on reboot otherwise.

## Running it

```bash
DEBS=~/kernel/7.3 nodes/macpro/provision.sh NODE      # DEBS: kernel/build-7.3.sh's WORK directory
nodes/macpro/provision.sh NODE reboot verify
ssh NODE sudo .cache/trashcan-kernel/install-kernel.sh pin 7.3.0-rc7-p2p --apply   # once verify passes
nodes/macpro/provision.sh NODE itable      # Apple SSDs only; the disk stays slow while this runs
```

`verify` runs on the GPUs: make sure nothing else is using them. It runs `scripts/test-engine.sh` if the node has
the Gemma 4 12B QAT GGUF at `~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf` (and uses the assistant at
`~/models/gemma-4-12b-assistant/` if present; `scripts/fetch-assistant.py` downloads it).

| Step | What it does |
|---|---|
| `check` | SSH and passwordless sudo work; prints model, CPU, memory, OS, GPUs and Thunderbolt controllers |
| `disk` | On a drive without write-zeroes, pauses ext4's inode-table zeroing (`noinit_itable`, now and in `/etc/fstab`) so the rest runs at normal disk speed |
| `packages` | Build tools, AMD firmware, Mesa Vulkan, `glslc`, kernel build dependencies, `iperf3`, `sockperf` |
| `rust` | rustup with the stable toolchain |
| `system` | Runs [`install.sh`](#node-services-installsh) with `--apply` (`INSTALL_ARGS` overrides): fan control, tblink, suspend masked, amdgpu command line, groups, journal, panic policy, sshd keepalive |
| `tblink` | Prints how to install the Thunderbolt node registry: `deploy/trashcan-cluster registry --push --apply` generates it from all nodes ([docs/deploy.md](../../docs/deploy.md)) |
| `kernel` | Copies the packages, `SHA256SUMS` and `kernel-build-*.txt` from `DEBS` and runs `kernel/install-kernel.sh install --apply`: checksums checked, packages installed, the new kernel booted once at the next reboot. GRUB's default stays the previous kernel until you `pin` it |
| `workspace` | Copies the repository to `~/trashcan` on the node (without `.git`, `target/` and git-ignored files), then `cargo build --release` and `make -C bench/xgpu` |
| `reboot` | Reboots and waits for SSH (up to 5 minutes) |
| `itable` | Finishes the zeroing that `disk` paused, all at once (`init_itable=0`), then removes `noinit_itable` from `/etc/fstab`. Not a default step: run it last, while nothing else uses the node |
| `verify` | Running kernel and command line; each GPU's VBIOS and VRAM; Vulkan devices; `xgpu p2p-verify`; `scripts/test-engine.sh` with the GPU clocks logged |

Run any subset by naming steps, for example `provision.sh NODE system` after editing a file under `root/`, or
`provision.sh NODE workspace` after changing the source.

## Node services (`install.sh`)

`install.sh` runs on the node as root and installs only what the engine and the links need. Without `--apply` it
prints every file it would create or change (with a diff) and every command it would run, and changes nothing.
Running it again after `--apply` reports `nothing to change`. It never reboots and never resets a Thunderbolt link.

```bash
sudo nodes/macpro/install.sh                    # preview
sudo nodes/macpro/install.sh --apply
sudo nodes/macpro/install.sh --uninstall        # preview of the removal; add --apply to remove
sudo nodes/macpro/install.sh --skip panic,ssh   # leave components out
sudo nodes/macpro/install.sh --lights --lights-args --build-module --apply   # with the port lights
deploy/trashcan-cluster -c cluster.toml install [--nodes A,B] [--apply]   # the same from your workstation
```

| Component | Installs | Takes effect |
|---|---|---|
| `tblink` | `/usr/local/sbin/tblink`, `tblink.service`, the `tbstream` udev rule (devices to `render`), `thunderbolt_net throttling=32000`; `--registry FILE` installs `/etc/tblink/nodes.tsv` | at once; a restart keeps every link; the throttle when `thunderbolt_net` next loads |
| `fan` | `fancurve.sh`, `fancurve.service` | at once |
| `network` | NetworkManager leaves `thunderbolt*` alone; with ifupdown, DHCP on every Ethernet port that has no stanza (the Debian installer configures only the port cabled during the install) | at once |
| `power` | masks the sleep, suspend and hibernate targets (the Mac Pro does not resume) | at once |
| `gpu` | `radeon.si_support=0 amdgpu.si_support=1` in `/etc/default/grub.d/90-trashcan.cfg` unless GRUB has them already; the user in `render` and `video`; `/etc/sudoers.d/trashcan-amdgpu-dpm`, which lets `render` write the GPUs' DPM level files only | next boot; next login |
| `logs` | persistent journal | at once |
| `panic` | reboot 10 s after a panic, oops or lockup (`/etc/sysctl.d/90-trashcan-panic.conf`); `emergency-reboot` | at once |
| `ssh` | sshd drops sessions whose client went away within about a minute | at once |
| `disk` | only on an original Apple SSD (`APPLE SSD SM...`): a udev rule that turns the drive's write cache off ([why](#cache-flushes-stall-the-apple-ssds)) | at once |
| `lights` | only with `--lights [--lights-args "ARGS"]`: runs `lights/install.sh` (`--dry-run` in a preview) with ARGS, e.g. `--build-module` ([docs/lights.md](../../docs/lights.md)); `--uninstall` runs `lights/uninstall.sh` if this installer installed them | at once |

State is kept in `/var/lib/trashcan-node`: the files installed, the originals they replaced, the targets masked, the
groups added. `--uninstall --apply` restores the originals, removes the rest, unmasks, takes the user out of the
groups it added and disables the two services (the fan returns to the SMC). Kernel crash capture (netconsole,
kdump) is not part of it: with `panic`, a crashed node reboots after 10 s and the persistent journal keeps what
was written before the crash.

## What to check in `verify`

- Each card reports its own VBIOS (`113-C3861XA-028` and `113-C3861XB-028` on a D700 pair).
- Max clocks reach 850/1370 on both cards (D700). Lower clocks mean a card is running with the wrong VBIOS or power
  source ([docs/hardware.md](../../docs/hardware.md)).
- `xgpu p2p-verify` passes and the export stays in VRAM.
- `test-engine` passes. It needs about 4.5 GiB per GPU, so it only fits on D700s (6 GB), not D500s (3 GB) or
  D300s (2 GB).

## Slow disk right after install (Apple SSDs)

The Mac Pro's original Apple SSDs (e.g. `APPLE SSD SM1024F`) don't support the write-zeroes command, so ext4's
background inode-table zeroing (`ext4lazyinit`) writes real zeroes after every mount until a new filesystem is fully
initialized. On one node this took writes from 1 ms to 400-700 ms and made the installer and the first `apt-get`
crawl, although the drive itself does 1.4 GB/s reads and 870 MB/s writes. On another, writes took 380 ms during the
zeroing and 18 ms once it was paused.

The fix has three parts:

- During the install, remount `/target` (and `/target/home`) with `noinit_itable` (step 1 above).
- `disk`, run by default, pauses the zeroing on every ext4 filesystem the installer didn't finish. It keeps the pause
  across reboots with `noinit_itable` in `/etc/fstab`.
- `itable` finishes it at the end, when the node is otherwise idle. It doesn't have to be done for the node to
  work: ext4 tracks which inodes were never used either way, and the zeroing mainly helps `e2fsck` repair a damaged
  filesystem.

Even with the zeroing paused, these drives are slow to flush their cache: a synced 4 KB write took 130 ms on one
node, and 1.7 s on another while `dpkg` was installing the kernel. `dpkg` syncs every file it installs, so there the
kernel image's 4,150 files went in at about 2 a second, and 25 minutes later it wasn't finished. The `packages` and
`kernel` steps therefore run `dpkg` with `--force-unsafe-io`. With it, both kernel packages (11,700 files) installed in
20 s. The only risk is losing files to a power cut during the install itself. To look at a node's disk:

```bash
cat /proc/pressure/io                                  # "some avg60" near 90 means the disk is saturated
sudo dumpe2fs /dev/sdXN | grep -c ITABLE_ZEROED       # groups done, against grep -c "^Group [0-9]"
cat /sys/block/sda/queue/write_zeroes_max_bytes        # 0: no write-zeroes, so the zeroing is slow
```

A node with an Intel NVMe drive, which supports write-zeroes, never showed this.

### Cache flushes stall the Apple SSDs

With its write cache on, an Apple SSD stalls all I/O while it flushes the cache under load. These drives take no FUA
and their flush is not queued, so every `fsync` and every ext4 journal commit sends one, and libata holds every other
command until it ends. While the disk reads at full speed (a cold model load), a flush usually takes 30 ms, but every
few seconds one takes 0.3-1.2 s with no read completing, and fsyncs then take 1.3-2.7 s. A cold model load whose
threads fault in pages waits that long, and a few flushes in a row stop the disk for several seconds.

The `disk` component of `install.sh` turns the write cache off (`cache_type` `write through`, now and by a udev rule at
boot). The kernel then sends no flushes. In a reproduction (page cache dropped, 20 GB read by 4 readers, a 4 KiB
fsync every 100 ms, a 4 KiB O_DIRECT read every 50 ms), on two nodes with different firmware:

| Write cache | Load time | Time with reads < 20 MB/s | Longest 4 KiB read | Longest fsync |
|---|---|---|---|---|
| on | 23-39 s | 4-13 s | 1.2-1.3 s | 1.3-2.7 s |
| off | 17.3-17.9 s | 0 | 37-48 ms | 94-124 ms |

The cost is sequential writes: 360 MB/s instead of 470-620 MB/s (a 4 GiB `dd` with `fdatasync`). Small synced writes
get faster: a 4 KiB append and `fsync` takes 3.7 ms instead of 9 ms (up to 1.2 s), and 3,000 small files and a
`sync` take 0.5 s instead of 1.6 s. A power cut can no longer lose acknowledged writes. A smaller queue depth, smaller
requests or another I/O scheduler did not help.

## Thunderbolt links between nodes (`tblink`)

`tblink.service` (`root/usr/local/sbin/tblink`) gives every cable between two nodes a fixed /30, whatever port or
interface name it lands on. `/etc/tblink/nodes.tsv` lists each node's index, hostname and the UUID of each controller
by Apple bus (bus 0 = bottom row, 1 = left column, 2 = right column). `deploy/trashcan-cluster registry` reads the
UUIDs from every node in your inventory and writes the table, with the indexes from the inventory (never change
them); `registry --push --apply` installs it on every node. A node that is not in the table gets no links.
For nodes a < b on buses da and db, the link is 10.99.(16a+b).(4(3da+db)+1) on a and +2 on b, named
`<peer>.tb<local bus>` in `/etc/hosts`:

```bash
ssh mac0 tblink status          # links, lanes, addresses, carrier, stream; flags a link still showing the Mac firmware
ping mac1.tb1                   # from mac0: the bus 1 cable
/usr/local/sbin/tblink links    # for programs: tab-separated with a header; state is up, no_carrier,
                                # no_interface, firmware (stuck on a rebooted peer), loop or unknown;
                                # stream is the link's /dev/tbstreamN or -
```

tblink also gives every working link a USB4STREAM, `trashcan`, under the link's XDomain stream service in configfs
(`/sys/kernel/config/thunderbolt/stream/0-3.1/trashcan`), with input and output HopID 15 at both ends, so neither
end needs the other's values; it loads `thunderbolt_stream` when needed, creates the stream again whenever it is
missing (after a module reload, a controller reset, a reconnect) or sits under a service name that is no longer the
stream service (the `.0`/`.1` index follows discovery order), and removes the streams of links that went away. The
engine moves each link's data to its stream by itself and falls back to TCP without it (`engine/src/link`).
`/etc/udev/rules.d/70-tbstream.rules` gives the devices to `render` (mode 0660). With `1` in `/etc/tblink/busy_poll`
(not there by default) tblink sets the streams' `busy_poll`, at its next pass on a stream nothing has open: the
rings then run without interrupts, which takes the interrupt and its work item out of every frame's way
([docs/kernel.md](../../docs/kernel.md#thunderbolt) has the numbers), and the engine's idle links sleep in short naps
instead of in `poll()`. Each end of a cable has its own setting. `tblink reload` removes the streams
before it unloads the modules (a process with a stream open keeps `thunderbolt_stream` loaded: stop it first), and
`scripts/tb-reset.sh` stops tblink on the Macs it resets and starts it again afterwards.

**After a node reboots, replug its neighbours' ends of the cables.** While a Mac reboots, its firmware answers on
the link (`Apple, Inc. Macintosh`), and the neighbour keeps that stale peer after Linux takes over: no carrier, and
`failed request link state change` on the rebooted node. Replugging just the neighbour's ends brought all three
links between two Macs back on two lanes within seconds. `tblink reload` (reloading the Thunderbolt modules, neighbour
first) recovered them once and only partly the second time. tblink reports a stuck link but doesn't reset
controllers: unbinding one controller's driver hit a thunderbolt bug (duplicate XDomain sysfs device) and coincided
with the peer Mac resetting. To reset several Macs' controllers together, use `scripts/tb-reset.sh HOST...`.
