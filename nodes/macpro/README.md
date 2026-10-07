# Setting up a Mac Pro 6,1 node

`provision.sh` turns a Mac Pro 6,1 with a fresh Debian 13 install into a node like incite: kernel `7.2.9-p2p`
(`docs/kernel.md`), the D700 GPU setup, crash capture and recovery, the fan curve, the Rust toolchain, and the
trashcan workspace built and checked. It runs from this Mac over SSH, every step is idempotent, and running it
against incite changes nothing.

## Before running it (on the new machine)

1. Install Debian 13 with user `fairfieldt`, the SSH server and, optionally, GNOME (incite has it). Use the whole
   internal SSD.
2. Give `fairfieldt` passwordless sudo. If the installer set a root password, use `su -` first:

   ```bash
   echo 'fairfieldt ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/90-fairfieldt
   ```

3. From this Mac, authorize the key and add a `Host` entry to `~/.ssh/config` if the name doesn't resolve:

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
MODEL_SRC=incite-lan nodes/macpro/provision.sh NEW_HOST models reboot verify
```

Use LAN addresses (`~/.ssh/config` has `invite` and `incite-lan`) for the workspace and model copies: from this Mac
the LAN carries 940 Mb/s, Tailscale about 450-500 Mb/s (WireGuard encryption in the Tailscale app). The 8 GB default
models take about 70 s over the LAN.

| Step | What it does |
|---|---|
| `check` | SSH and passwordless sudo work; prints model, CPU, memory, OS, GPUs and Thunderbolt controllers |
| `packages` | Build tools, AMD firmware, Mesa Vulkan, `glslc`, kernel build dependencies, `iperf3`, `sockperf` |
| `tailscale` | Tailscale's apt repository and `tailscaled`, with `fairfieldt` as operator |
| `rust` | rustup with the stable toolchain |
| `system` | Installs `root/` (panic sysctls, netconsole, `emergency-reboot`, NetworkManager ignoring `thunderbolt*`, sshd keepalive) and the fan curve from `scripts/`; points netconsole at this Mac's current address; masks suspend (the Mac Pro doesn't resume); sets the GRUB command line `quiet radeon.si_support=0 amdgpu.si_support=1 panic=10`; adds `fairfieldt` to `render`; creates `~/trashcan-exp/.lock` |
| `kernel` | Installs the `7.2.9-p2p` image and headers and pins `GRUB_DEFAULT` to that entry by id |
| `workspace` | `HOST=NEW_HOST scripts/sync.sh push`, then `cargo build --release` and `make -C bench/xgpu` |
| `reboot` | Reboots and waits for SSH (up to 5 minutes) |
| `models` | Copies `MODELS` (default: the Gemma 4 12B QAT GGUF and its assistant) from `MODEL_SRC` (default incite), streamed through this Mac with `tar`; skips directories whose size already matches |
| `verify` | Running kernel and command line; each GPU's VBIOS and VRAM; Vulkan devices; a netconsole line reaching this Mac; `xgpu p2p-verify`; `scripts/test-engine.sh` with the GPU clocks logged |

Run any subset by naming steps, for example `provision.sh NEW_HOST system` after editing a file under `root/`.

## What to check in `verify`

- Each card reports its own VBIOS (on incite `113-C3861XA-028` and `113-C3861XB-028`).
- Max clocks reach 850/1370 on both cards (D700). Lower clocks mean the power-source fix isn't active.
- `xgpu p2p-verify` passes and the export stays in VRAM.
- `test-engine` passes. It needs about 4.5 GiB per GPU, so it only fits on D700s (6 GB), not D500s (3 GB) or
  D300s (2 GB).

## Slow disk right after install (Apple SSDs)

The Mac Pro's original Apple SSDs (invite: `APPLE SSD SM1024F`) don't support the write-zeroes command, so ext4's
background inode-table zeroing (`ext4lazyinit`, shown as `init_itable`) writes real zeroes after every mount until a
new filesystem is fully initialized. On invite this took writes from 1 ms to 400-700 ms and made the installer and
the first `apt-get` crawl, although the drive itself does 1.4 GB/s reads and 870 MB/s writes. If a new node feels
slow:

```bash
cat /proc/pressure/io                                  # "some avg60" near 90 means the disk is saturated
sudo dumpe2fs /dev/sdXN | grep -c ITABLE_ZEROED       # groups done, against grep -c "^Group [0-9]"
sudo fstrim -av                                        # a reinstalled SSD hasn't been told its old data is free
sudo mount -o remount,noinit_itable /home              # pause the zeroing until the next mount
```

It resumes after a reboot and only has to run once. incite's Intel NVMe drive supports write-zeroes and never
showed this.

## Not covered

- `gpuq`: its deploy script lists machines by name (`../homelab/gpuq/deploy.sh`); add the node there.
- Other models: set `MODELS` to more directory names under incite's `~/models`.
- The left Ethernet port of incite stopped linking after hard power cuts; on a new node, use whichever port links.
