# Deploying trashcan on one to six Mac Pros

This page takes Mac Pros with Debian 13 to a running Responses server for one of the published recipes, and
covers checking, stopping, restarting and recovering it. Everything runs from your workstation over SSH with
`deploy/trashcan-cluster` (Python 3.11 or later; nothing else to install there).

| Macs | Recipe | Model | Cables |
|---|---|---|---|
| 1 | `1-gemma` | Gemma 4 12B QAT Q4_0 over both GPUs; `--mtp` adds the Gemma assistant | none |
| 1 | `1-qwen9b` | Qwen3.5-9B, tensor parallel over both GPUs (TP2); `--mtp` adds its MTP layer | none |
| 2 | `2-qwen27b` | Qwen3.8-27B, tensor parallel over four GPUs (TP4) | at least one between the two; one lane per cable, up to three |
| 4 | `4-qwen27b` | Qwen3.8-27B, tensor parallel over eight GPUs (TP8) | a ring: machines 0-1, 0-2, 1-3, 2-3 (or all six pairs) |
| 6 | `6-flashnext` | Qwen3.8-Flash-Next G-iq4, 12 GPU stages, 32K context | a chain along the Macs (the measured cabling is a ring, each Mac on bus 0 and bus 1) |

The recipes are defined in [deploy/recipes.toml](../deploy/recipes.toml) and their files, kernel and Mesa version
pinned in [deploy/manifest.toml](../deploy/manifest.toml). The README's rates were measured with these models on
this hardware; see [qwen.md](qwen.md) and [qwen4.md](qwen4.md) for the conditions. Three and five Macs have no
qualified recipe, and Flash-Next does not fit on fewer than six.

## 1. Each node

1. Install Debian 13 and prepare SSH and sudo: [nodes/macpro/README.md](../nodes/macpro/README.md#before-running-it).
2. Kernel: build `7.3.0-rc7-p2p` ([kernel.md](kernel.md#building)), then install it and boot it once
   ([kernel.md](kernel.md#installing-and-booting)):

   ```bash
   DEBS=~/kernel/7.3 nodes/macpro/provision.sh NODE check packages rust kernel
   ```

3. Node services ([nodes/macpro/README.md](../nodes/macpro/README.md#node-services-installsh)): fan control, tblink,
   suspend masked, amdgpu command line and permissions, persistent journal, panic policy, and with `--lights` the
   port lights ([lights.md](lights.md)). Preview, then apply:

   ```bash
   deploy/trashcan-cluster -c cluster.toml install --nodes NODE           # shows every change
   deploy/trashcan-cluster -c cluster.toml install --nodes NODE --apply
   deploy/trashcan-cluster -c cluster.toml install --nodes NODE --apply   # again: "nothing to change"
   ```

4. Reboot each node when convenient, then check the kernel and pin it:

   ```bash
   ssh NODE ./.cache/trashcan-kernel/install-kernel.sh verify .cache/trashcan-kernel
   ssh NODE sudo ./.cache/trashcan-kernel/install-kernel.sh pin 7.3.0-rc7-p2p --apply
   ```

5. The engine: build it once (`cargo build --release --locked`, on a node or on another Debian 13 x86_64 machine)
   and copy the same binary to every node, at the inventory's `workspace`/`binary` path
   (`nodes/macpro/provision.sh NODE workspace` builds on the node itself instead; every node must then end up with
   the same binary). Without `glslc` the build succeeds but GPU commands fail.
6. Models: put the files of your recipe on every node it uses, under the inventory's `models` directory, at the
   paths in the manifest. Gemma comes from Hugging Face; the Qwen files are quantized with the recipes in
   [qwen.md](qwen.md#quantization) and [qwen4.md](qwen4.md). Flash-Next needs all six shards (116 GiB) on all six
   Macs. Use any bulk copy you trust; `preflight` checks every file's size and sha256 against the manifest.

## 2. The inventory

Copy [deploy/cluster.example.toml](../deploy/cluster.example.toml) to `cluster.toml` and fill in your nodes (SSH
names, fixed indexes) and cables (`NODE:BUS` at each end: bus 0 is the bottom row of Thunderbolt ports, 1 the left
column, 2 the right column). Then:

```bash
deploy/trashcan-cluster -c cluster.toml check                     # inventory, recipes and manifest agree
deploy/trashcan-cluster -c cluster.toml registry                  # read each node's controller UUIDs
deploy/trashcan-cluster -c cluster.toml registry --push           # compare with the nodes' /etc/tblink/nodes.tsv
deploy/trashcan-cluster -c cluster.toml registry --push --apply   # install it where it differs
deploy/trashcan-cluster -c cluster.toml links                     # live links against your cable map
```

The registry is generated from the nodes themselves and written to the inventory's `state` directory
(`cluster-state/nodes.tsv`); tblink addresses each cable from it within seconds, without resetting any link.
`links` prints one line per cable: `OK` (up, two lanes, stream), `DEGRADED` (up on one lane or without a stream:
it works, slower), `DOWN`, `MISSING` (not seen) or `EXTRA` (not in your map). `TRASHCAN_CLUSTER=cluster.toml` saves
the `-c`.

## 3. Check, start, stop

```bash
deploy/trashcan-cluster preflight 6-flashnext      # read-only; hashes model files the first time (minutes)
deploy/trashcan-cluster smoke 6-flashnext          # start, one request, check the answer, stop
deploy/trashcan-cluster up 6-flashnext             # start and leave it running
deploy/trashcan-cluster status
deploy/trashcan-cluster down                       # stop it
deploy/trashcan-cluster run 6-flashnext            # start; Ctrl-C stops everything
```

`--nodes A,B,...` picks the nodes (the first is the driver; default: the first nodes of the inventory).
`--mtp` adds speculative decoding.

`preflight` checks every node of the recipe and fails on any of:

- a kernel that is not one built from `kernel/` (release or patch set id in the manifest), a command line without
  `amdgpu.si_support=1 radeon.si_support=0`, or no `CONFIG_PCI_P2PDMA`;
- not two amdgpu GPUs with the two different D700 VBIOS images, less than 6 GB of VRAM, or more than 512 MiB of
  VRAM already in use (another job on the GPU);
- the user not in `render`, `fancurve.service` not running;
- an engine process already running, or a port the recipe needs already in use;
- engine binaries that differ between nodes (or from `engine.binary_sha256` when set);
- a missing model file or one whose size or sha256 differs from the manifest;
- for Flash-Next, a Mesa version other than the audited one.

It warns about a kernel that is accepted but not yet qualified, a GPU left at the `high` DPM level, `tblink` not
running, no configfs, and cables that are down, missing or degraded (one lane, no stream). Model files are hashed
once per node and cached (`~/.cache/trashcan-cluster/sha256.json` by path, size, modification time and inode).

The launcher then:

1. starts the nodes before the driver (Flash-Next stages from the last; TP machines from the highest), each waiting
   until its port listens (60 s), resolving each address from the live links (`<peer>.tb<bus>` names from tblink);
2. starts the server on the first node and waits until `/health` answers (the recipe's `load_deadline_s`: 180 s
   for one Mac, 300 s for the 27B, 600 s for Flash-Next);
3. on any process ending early, a deadline passing, Ctrl-C, SIGTERM or SIGHUP, stops every process: the driver
   first, then all nodes at once, each with SIGTERM and after 30 s SIGKILL;
4. sets every GPU it used back to the `auto` DPM level (the engine raises its GPUs to `high` while it works and a
   killed process leaves them there, about 35 W more per idle D700) and copies the logs to `cluster-state/logs/`.

Each process's pid file and log are in the inventory's `run_dir` on its node. The server listens on 127.0.0.1 of
the first node; reach it with `ssh -L 8000:127.0.0.1:8000 mac0`, or add `--host 0.0.0.0` to the recipe's driver
line (the server has no authentication). `smoke` sends "What is the capital of France?" with greedy decoding and
checks that the answer contains "Paris", then reports the time, token usage and the driver's VRAM in use.

To restart, `down` then `up`. To change a recipe's options (context, batch, port), edit `deploy/recipes.toml`.
`cpu_latency_us = "0"` in the inventory holds a 0 µs CPU latency request for each engine process's lifetime, the
setting the published rates were measured with ([qwen.md](qwen.md)); it keeps idle cores busy-polling, about 55 W
more per Mac while waiting for requests, and is off by default. `--wrap 'NODE=PREFIX'` runs node commands under a
prefix (a job scheduler, `nice`, ...), `*` for every node.

When a process ends during a run (a crash, a killed process, a node going down), `run` notices within about 5 s
and stops the rest; with `up`, `status` shows it and `down` stops the rest. Its neighbours fail their next exchange
with it: a tensor-parallel exchange that moves no data for 500 ms ends the session (`lanes stalled for 500ms`). Each
link runs on its cable's USB4STREAM (`link ...: stream up`); a stream that fails ends the session instead of going on
over TCP (`link ...: stream lost (...); the link ends`), and so does one that is not up within 20 s. `smoke`, `run`,
`up` and `status` check the engines' logs and fail on any link off its stream (`LINK FAILURE`), including a
Thunderbolt link without a stream at one end (`link: no stream`). A link whose other machine stops shows as
`closed the link`, not as a failure.

## Thunderbolt link recovery

After a Mac reboots, its neighbours can keep a stale view of the cable (`tblink status` shows "peer is the Mac
firmware", the rebooted Mac logs "failed request link state change"), and after a cold start of every Mac some
links can stay down or come up on one lane. tblink reports a stuck link; it does not reset controllers by itself.
Recover with nothing running on any node (`status` shows no run), in this order, checking `links` after each step:

1. **Reload every node's Thunderbolt modules at the same moment:** `deploy/trashcan-cluster links-reload-all --yes`
   (each node runs `tblink reload` at the same second, then the tool waits 30 s and prints `links`). On 2026-10-26
   one round brought two one-lane links of a four-Mac mesh to two lanes, and a recabled six-Mac ring fully up.
   Reloading one Mac at a time took three rounds after that day's cold start, because each reload dropped a link
   that was working.
2. **Reset every node's controllers together:** `scripts/tb-reset.sh NODE...` with all nodes (unloads the modules,
   resets the three controllers of every Mac, loads them again on all within a second). Historically: five of six
   links on two lanes after one round, the last after a second round (four Macs, six cables).
3. **Replug** the cable of a link still missing after two rounds, at the neighbour's end first.
4. **Power cycle** the Mac whose controller stays broken ("failed to allocate host router", or no answer to
   config reads).

Never reload or reset Thunderbolt on a node while an engine runs on it: a process with a stream open keeps the
module loaded, and a reset during a run has hung a node. There is no unattended recovery: each step is an
operator's decision. Firmware hangs before the startup chime when cables loop a Mac's own controllers; unplug those
before rebooting.

## Crashes

`install.sh`'s `panic` component makes an oops, hard lockup or soft lockup panic and a panic reboot after 10 s;
Apple's firmware disables the hardware watchdog, so this is the only automatic recovery. The journal is
persistent, so `journalctl -b -1` shows what was written before the crash. Capturing the kernel's last messages
needs a second machine (netconsole) and is not installed. `--skip panic` leaves a crashed node as it is, for
inspection at the console. After a node reboots, check its links (above) before starting a run.

## Known limits

- Mac Pro 6,1 with two 6 GB D700s, Debian 13, amdgpu (SI) and RADV; nothing else is qualified.
- Flash-Next's decode relies on RADV's barrier behaviour in Mesa 25.0.7 ([qwen4.md](qwen4.md)); after a Mesa
  upgrade `preflight` refuses it until the ISA audit is rerun and the manifest updated.
- amdgpu resets a GPU when one job runs longer than 2 s; the engine keeps its submissions well under that.
- Flash-Next's 32K context is the configured cache capacity, not a qualified 32K-token prompt.
- 3- and 5-Mac configurations, other model files and quantizations, and the `lights` kernel patch set are not
  qualified.
- Every link of a recipe must be a Thunderbolt cable with a stream at both ends; the launcher has no Ethernet
  fallback (TP8 with one cable replaced by Ethernet failed its first request in two of three runs).
