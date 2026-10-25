# Setting up a Mac Pro 6,1 node

`provision.sh` turns a Mac Pro 6,1 with a fresh Debian 13 install into a trashcan node: kernel `7.3.0-rc7-p2p`
(`docs/kernel.md`), the D700 GPU setup, panic recovery, the fan curve, Thunderbolt links to other nodes, the Rust
toolchain, and the trashcan workspace built and checked. It runs from your workstation over SSH, every step is
idempotent, and running it against a provisioned node changes nothing.

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
   cross-compiled with `ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-`) and copy the `linux-image-7.3.0-rc7-p2p_*` and
   `linux-headers-7.3.0-rc7-p2p_*` packages to a directory on your workstation.
5. Unplug any Thunderbolt cable that loops the machine's own controllers; the firmware hangs on reboot otherwise.

## Running it

```bash
DEBS=~/kernel-debs nodes/macpro/provision.sh NODE
nodes/macpro/provision.sh NODE reboot verify
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
| `system` | Installs `root/` (panic sysctls, persistent journal, `emergency-reboot`, `tblink` with the links' streams and their udev rule, `thunderbolt_net throttling=32000`, NetworkManager ignoring `thunderbolt*`, sshd keepalive) and the fan curve from `scripts/`; masks suspend (the Mac Pro doesn't resume); sets the GRUB command line `quiet radeon.si_support=0 amdgpu.si_support=1 panic=10`; adds the login user to `render` and `video` |
| `tblink` | Records the node's Thunderbolt controller UUIDs in `tblink/nodes.tsv` and copies the table to every node in it |
| `kernel` | Installs the `7.3.0-rc7-p2p` image and headers from `DEBS` and pins `GRUB_DEFAULT` to that entry by id |
| `workspace` | Copies the repository to `~/trashcan` on the node (without `.git`, `target/` and git-ignored files), then `cargo build --release` and `make -C bench/xgpu` |
| `reboot` | Reboots and waits for SSH (up to 5 minutes) |
| `itable` | Finishes the zeroing that `disk` paused, all at once (`init_itable=0`), then removes `noinit_itable` from `/etc/fstab`. Not a default step: run it last, while nothing else uses the node |
| `verify` | Running kernel and command line; each GPU's VBIOS and VRAM; Vulkan devices; `xgpu p2p-verify`; `scripts/test-engine.sh` with the GPU clocks logged |

Run any subset by naming steps, for example `provision.sh NODE system` after editing a file under `root/`, or
`provision.sh NODE workspace` after changing the source.

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

## Thunderbolt links between nodes (`tblink`)

`tblink.service` (`root/usr/local/sbin/tblink`) gives every cable between two nodes a fixed /30, whatever port or
interface name it lands on. `tblink/nodes.tsv` lists each node's index, hostname and the UUID of each controller by
Apple bus (bus 0 = bottom row, 1 = left column, 2 = right column). It ships empty: the `tblink` step adds the node it
provisions (indexes from 1, in provisioning order, never changing) and copies the table to every node in it, reaching
the other nodes by the hostnames in the table. Keep your copy of the table; a node that is not in it gets no links.
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
`/etc/udev/rules.d/70-tbstream.rules` gives the devices to `render` (mode 0660). `tblink reload` removes the streams
before it unloads the modules (a process with a stream open keeps `thunderbolt_stream` loaded: stop it first), and
`scripts/tb-reset.sh` stops tblink on the Macs it resets and starts it again afterwards.

**After a node reboots, replug its neighbours' ends of the cables.** While a Mac reboots, its firmware answers on
the link (`Apple, Inc. Macintosh`), and the neighbour keeps that stale peer after Linux takes over: no carrier, and
`failed request link state change` on the rebooted node. Replugging just the neighbour's ends brought all three
links between two Macs back on two lanes within seconds. `tblink reload` (reloading the Thunderbolt modules, neighbour
first) recovered them once and only partly the second time. tblink reports a stuck link but doesn't reset
controllers: unbinding one controller's driver hit a thunderbolt bug (duplicate XDomain sysfs device) and coincided
with the peer Mac resetting. To reset several Macs' controllers together, use `scripts/tb-reset.sh HOST...`.
