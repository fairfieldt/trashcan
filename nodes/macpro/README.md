# Setting up a Mac Pro 6,1 node

`provision.sh` turns a Mac Pro 6,1 with a fresh Debian 13 install into a node like incite: kernel `7.2.9-p2p`
(`docs/kernel.md`), the D700 GPU setup, crash capture and recovery, the fan curve, the Rust toolchain, and the
trashcan workspace built and checked. It runs from this Mac over SSH, every step is idempotent, and running it
against incite changes nothing.

## Before running it (on the new machine)

1. Install Debian 13 with user `fairfieldt`, the SSH server and, optionally, GNOME (incite has it). Use the whole
   internal SSD. Name it like the others (incite, invite, indite).

   On an original Apple SSD (`APPLE SSD SM...`), the install crawls once the base system starts copying (see
   [Slow disk](#slow-disk-right-after-install-apple-ssds)). Right after partitioning, switch to a shell
   (Ctrl+Alt+F2, Enter) and run the following, adding `/target/home` if the installer made a separate `/home`.
   Switch back with Ctrl+Alt+F1 (F5 with the graphical installer).

   ```bash
   mount -o remount,noinit_itable /target
   ```

2. Give `fairfieldt` passwordless sudo. If the installer set a root password, use `su -` first:

   ```bash
   echo 'fairfieldt ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/90-fairfieldt
   ```

3. From this Mac, authorize the key. Until the node is on the tailnet, add a temporary `Host NEW_HOST` entry with
   its LAN address (`HostName 192.168.0.N`) to `~/.ssh/config`:

   ```bash
   ssh-copy-id fairfieldt@NEW_HOST
   ```

4. Have the kernel packages ready: `REV=2 kernel/cross-build-7.2.sh` leaves them in
   `~/dev/claude-work/trashcan-kbuild/` (they are already there unless the kernel changed).
5. Unplug any Thunderbolt cable that loops the machine's own controllers; the firmware hangs on reboot otherwise.

## Running it

```bash
nodes/macpro/provision.sh NEW_HOST
ssh NEW_HOST tailscale up            # once; open the URL it prints
# now delete the temporary Host entry: NEW_HOST resolves to NEW_HOST.tdf.sh from here on
nodes/macpro/provision.sh NEW_HOST models reboot verify   # verify runs on the GPUs: hold a gpuq lease
nodes/macpro/provision.sh NEW_HOST itable                 # Apple SSDs only; the disk stays slow while this runs
```

Then commit `tblink/nodes.tsv`, [add the node to gpuq](#adding-the-node-to-gpuq), add its tailnet address to `NODES`
in `scripts/netconsole-listen.py` (and rerun `scripts/install-netconsole-receiver.py`) so its GPU crash dumps are
copied, and fill in its row under [Nodes](#nodes).

Reach nodes by their tailnet names. `~/.ssh/config` turns a bare name like `indite` into `indite.tdf.sh`
(`CanonicalizeHostname yes`, `CanonicalDomains tdf.sh`), because LAN addresses come from DHCP and change: incite moved
from .83 to .82, and indite from .70 to .71 on its first reboot. Tailscale carries about 450-500 Mb/s from this Mac,
against 940 Mb/s on the LAN. The 8 GB default models took 70 s over the LAN to invite and 114 s to indite.

| Step | What it does |
|---|---|
| `check` | SSH and passwordless sudo work; prints model, CPU, memory, OS, GPUs and Thunderbolt controllers |
| `disk` | On a drive without write-zeroes, pauses ext4's inode-table zeroing (`noinit_itable`, now and in `/etc/fstab`) so the rest runs at normal disk speed |
| `packages` | Build tools, AMD firmware, Mesa Vulkan, `glslc`, kernel build dependencies, `iperf3`, `sockperf` |
| `tailscale` | Tailscale's apt repository and `tailscaled`, with `fairfieldt` as operator |
| `rust` | rustup with the stable toolchain |
| `system` | Installs `root/` (panic sysctls, netconsole, `emergency-reboot`, `tblink` with the links' streams and their udev rule, `thunderbolt_net throttling=32000`, NetworkManager ignoring `thunderbolt*`, sshd keepalive) and the fan curve from `scripts/`; points netconsole at this Mac's current address; masks suspend (the Mac Pro doesn't resume); sets the GRUB command line `quiet radeon.si_support=0 amdgpu.si_support=1 panic=10`; adds `fairfieldt` to `render`; creates `~/trashcan-exp/.lock` |
| `tblink` | Records the node's Thunderbolt controller UUIDs in `tblink/nodes.tsv` (commit it) and copies the table to every node |
| `kernel` | Installs the `7.2.9-p2p` image and headers and pins `GRUB_DEFAULT` to that entry by id |
| `workspace` | `HOST=NEW_HOST scripts/sync.sh push`, then `cargo build --release` and `make -C bench/xgpu` |
| `reboot` | Reboots and waits for SSH (up to 5 minutes) |
| `models` | Copies `MODELS` (default: the Gemma 4 12B QAT GGUF and its assistant) from `MODEL_SRC` (default incite), streamed through this Mac with `tar`; skips directories whose size already matches |
| `itable` | Finishes the zeroing that `disk` paused, all at once (`init_itable=0`), then removes `noinit_itable` from `/etc/fstab`. Not a default step: run it last, while nothing else uses the node |
| `verify` | Running kernel and command line; each GPU's VBIOS and VRAM; Vulkan devices; a netconsole line reaching this Mac; `xgpu p2p-verify`; `scripts/test-engine.sh` with the GPU clocks logged |

Run any subset by naming steps, for example `provision.sh NEW_HOST system` after editing a file under `root/`.

## What to check in `verify`

- Each card reports its own VBIOS (on incite `113-C3861XA-028` and `113-C3861XB-028`).
- Max clocks reach 850/1370 on both cards (D700). Lower clocks mean the power-source fix isn't active.
- `xgpu p2p-verify` passes and the export stays in VRAM.
- `test-engine` passes. It needs about 4.5 GiB per GPU, so it only fits on D700s (6 GB), not D500s (3 GB) or
  D300s (2 GB).

## Slow disk right after install (Apple SSDs)

The Mac Pro's original Apple SSDs (invite and indite: `APPLE SSD SM1024F`) don't support the write-zeroes command,
so ext4's background inode-table zeroing (`ext4lazyinit`) writes real zeroes after every mount until a new
filesystem is fully initialized. On invite this took writes from 1 ms to 400-700 ms and made the installer and the
first `apt-get` crawl, although the drive itself does 1.4 GB/s reads and 870 MB/s writes. On indite, writes took
380 ms during the zeroing and 18 ms once it was paused.

The fix has three parts:

- During the install, remount `/target` (and `/target/home`) with `noinit_itable` (step 1 above).
- `disk`, run by default, pauses the zeroing on every ext4 filesystem the installer didn't finish. It keeps the pause
  across reboots with `noinit_itable` in `/etc/fstab`.
- `itable` finishes it at the end, when the node is otherwise idle. It doesn't have to be done for the node to
  work: ext4 tracks which inodes were never used either way, and the zeroing mainly helps `e2fsck` repair a damaged
  filesystem.

Even with the zeroing paused, these drives are slow to flush their cache: a synced 4 KB write took 130 ms on invite,
and 1.7 s on indite while `dpkg` was installing the kernel. `dpkg` syncs every file it installs, so on indite the
kernel image's 4,150 files went in at about 2 a second, and 25 minutes later it wasn't finished. The `packages` and
`kernel` steps therefore run `dpkg` with `--force-unsafe-io`. With it, both kernel packages (11,700 files) installed in
20 s. The only risk is losing files to a power cut during the install itself. To look at a node's disk:

```bash
cat /proc/pressure/io                                  # "some avg60" near 90 means the disk is saturated
sudo dumpe2fs /dev/sdXN | grep -c ITABLE_ZEROED       # groups done, against grep -c "^Group [0-9]"
cat /sys/block/sda/queue/write_zeroes_max_bytes        # 0: no write-zeroes, so the zeroing is slow
```

incite's Intel NVMe drive supports write-zeroes and never showed this.

## Thunderbolt links between nodes (`tblink`)

`tblink.service` (`root/usr/local/sbin/tblink`) gives every cable between two nodes a fixed /30, whatever port or
interface name it lands on. `tblink/nodes.tsv` lists each node's index, hostname and the UUID of each controller by
Apple bus (bus 0 = bottom row, 1 = left column, 2 = right column); the `tblink` step adds a node and copies the table
to every node. For nodes a < b on buses da and db, the link is 10.99.(16a+b).(4(3da+db)+1) on a and +2 on b, named
`<peer>.tb<local bus>` in `/etc/hosts`:

```bash
ssh incite tblink status        # links, lanes, addresses, carrier, stream; flags a link still showing the Mac firmware
ping invite.tb1                 # from incite: the bus 1 cable
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
incite-invite links back on two lanes within seconds. `tblink reload` (reloading the Thunderbolt modules, neighbour
first) recovered them once and only partly the second time. tblink reports a stuck link but doesn't reset
controllers: unbinding one controller's driver hit a thunderbolt bug (duplicate XDomain sysfs device) and coincided
with the peer Mac resetting.

## Adding the node to gpuq

GPU work on a node goes through its gpuq queue, which lives in the homelab repo (`../homelab/gpuq`). For a node
NAME, copy `deploy/agent-indite.service` to `deploy/agent-NAME.service` and replace `indite` with NAME. Add NAME to:
`MACHINES` in `deploy.sh`, `deploy/hub.json`, `MACHINES` and the two help texts in `src/cli.rs`, and the tables in
`README.md` and `RULES.md`. Then deploy, check the node shows up, and commit:

```bash
cd ../homelab/gpuq
./deploy.sh --hub && ./deploy.sh --NAME && ./deploy.sh --client
gpuq machines
```

The agent needs `~/trashcan-exp/.lock`, which the `system` step creates.

## Nodes

| Node | Name | CPU | Memory | Disk | tblink index |
|---|---|---|---|---|---|
| incite | incite.tdf.sh | Xeon E5-2697 v2, 12 cores | 64 GB | Intel NVMe 1 TB | 1 |
| invite | invite.tdf.sh | Xeon E5-2697 v2, 12 cores | 64 GB | Apple SSD SM1024F | 2 |
| indite | indite.tdf.sh | Xeon E5-1650 v2, 6 cores, 3.5 GHz | 48 GB (3 of 4 slots) | Apple SSD SM1024F | 3 |
| enlight | enlight.tdf.sh | Xeon E5-2697 v2, 12 cores | 64 GB | Apple SSD SM1024F | 4 |
| inflight | inflight.tdf.sh | Xeon E5-2697 v2, 12 cores | 64 GB | Apple SSD SM1024F | 5 |
| alight | alight.tdf.sh | Xeon E5-2697 v2, 12 cores | 64 GB | Apple SSD SM1024F | 6 |

All have two D700s (Tahiti XT, 6 GB).

## Not covered

- Other models: set `MODELS` to more directory names under incite's `~/models`.
- The left Ethernet port of incite stopped linking after hard power cuts; on a new node, use whichever port links.
