# A16: reproducible node and cluster deployment, hardware acceptance (2026-10-26)

What landed on main: `1920bf46` (deploy/: inventory, manifest, recipes, `trashcan-cluster`, docs/deploy.md),
`e73949c4` (nodes/macpro/install.sh, provision.sh using it, registry generated), `22c78924` (kernel patch sets with
ids, build record, kernel/install-kernel.sh, docs/kernel.md). The runs below used the working tree of those commits
just before they were made; changed afterwards: the `--lights` hook of install.sh (previewed only,
`installer/7-preview-with-lights-hook.txt`) and the smoke request printing an HTTP error body.

Machines: the six Mac Pros, Debian 13, `7.3.0-rc7-p2p-s2`, Mesa 25.0.7-2+deb13u1, cabled as a ring
(incite:0-inflight:0, inflight:1-alight:1, alight:0-indite:0, indite:1-invite:1, invite:0-enlight:0,
enlight:1-incite:1; incite-inflight, invite-enlight and enlight-incite on one lane). Engine: main `91c3b82a`
cross-built in kbuild (`cargo build --release --locked`), sha256 `034690eb0c1cb2ab873d5b7f4842d99b72195fa6220ac1d684da93f17b2e2a78`,
in `~/trashcan-exp/a16/trashcan` on all six. Inventory: [tools/cluster.toml](tools/cluster.toml). Every GPU step ran
under gpuq leases (`tools/lease.sh` for subsets, chain6.sh for the six), node commands as `gpuq run ID --`
(`--wrap`). No kernel was installed, nothing rebooted, no link was reset.

## Results

| Check | How | Result |
|---|---|---|
| Registry generated from the nodes | `registry --push` | identical to the deployed `/etc/tblink/nodes.tsv` on all six ([generated](registry-generated.tsv), [deployed](registry-deployed.tsv)); nothing pushed |
| Model checksums | `trashcan-cluster hash` on every Mac ([hashes/](hashes/)) | every copy agrees across Macs (6 Gemma, 2 9B, 4 27B, 6 Flash-Next, 2 Flash-Next MTP); Gemma's equals Hugging Face's LFS sha256. Now in deploy/manifest.toml. 4.5-6.8 min per Mac for ~150 GB (unleased: disk/CPU only, queues empty) |
| Node installer, one node (inflight, lease #123) | [installer/](installer/) | preview: 8 changes, nothing written ([state identical](installer/1-preview-state.txt)); apply: 8 changes; repeat: `nothing to change`; uninstall preview then apply: originals restored, sudoers rule removed, state dir gone, file hashes identical to before; preview afterwards identical to the first. Links up on the same lanes and streams throughout |
| Kernel package path | kbuild VM, `KSET=engine REV=1` ([kernel-build/](kernel-build/)) | `linux-image-7.3.0-rc7-p2p_7.3.0~rc7-1p2p+ps9c9a5449`, 5 min 44 s; resulting config byte-identical to kernel/configs. `install-kernel.sh install` preview on inflight: checksums verified, would install and boot once; a corrupted copy is refused; `verify` reports the running `-s2` kernel as not this build (expected) and passes the hardware checks ([kernel/](kernel/install-preview-inflight.txt)). Not installed |
| 1-gemma, 1-qwen9b on incite | `smoke` ([launch/smoke-1mac.txt](launch/smoke-1mac.txt)) | ready 26 / 23 s, "Paris", stopped; 4.6 / 3.2 GiB VRAM per GPU |
| 2-qwen27b on incite+enlight, `--mtp` | [launch/lifecycle-2-qwen27b.txt](launch/lifecycle-2-qwen27b.txt), [updown](launch/updown-2-qwen27b.txt) | smoke PASS (ready 60 s cold, 23-28 s warm); node SIGKILLed: launcher stopped everything 9 s later, rc 1; SIGINT: everything stopped in 6 s, rc 130; up/status/down: a second `up` refused (VRAM in use, engine running, port busy); after every step no engine, no port, DPM `auto` on both. enlight's GPUs had been left at `high` by an earlier session; the first stop restored `auto` |
| 4-qwen27b (TP8) on incite, enlight, indite, invite, `--tcp-fallback` | [launch/lifecycle-4-qwen27b.txt](launch/lifecycle-4-qwen27b.txt), [repeat](launch/smoke-4-qwen27b-repeat.txt) | the ring cabling has no 4-cycle, so machine 0-2 (incite-indite) ran over TCP (tailnet). Start/stop, node kill (stopped in 12 s), SIGINT (6 s), up/down all clean. **The first request failed in 2 of 3 smokes** (HTTP 500; `lanes stalled for 500ms` on stream and TCP lanes alike, [logs/20261026-131214-4-qwen27b](logs/20261026-131214-4-qwen27b), [logs/20261026-131620-4-qwen27b](logs/20261026-131620-4-qwen27b)); the third passed. Not a qualification of the 4-Mac recipe: that needs the ring 0-1, 0-2, 1-3, 2-3 cabled |
| 6-flashnext, all six (chain6) | [launch/lifecycle-6-flashnext.txt](launch/lifecycle-6-flashnext.txt) | order from the live ring: incite, inflight, alight, indite, invite, enlight. smoke PASS (ready 37 s, 1.36 s request, driver 5830/5819 MiB); stage5 SIGKILLed: stopped in 8 s; SIGINT: 5 s; up/down clean; second up refused. Every Mac `auto` and port-free after each step |

The node logs of every run are in [logs/](logs/) (copied by `down`/stop). `20261026-130820-2-qwen27b` is the first
`up`, which hung because the detached start kept the SSH session open (fixed with `setsid -f` before the commit).

## Not done

- Installing and booting the `+ps9c9a5449` package (needs a maintainer-approved reboot of one Mac, then
  `install-kernel.sh verify DIR` and the recipes' smoke tests there; then the other Macs).
- `links-reload-all` (all six reloading Thunderbolt at once) is untested; it needs an authorized maintenance window.
- A fresh-node install (Debian installer through provision.sh) was not repeated; inflight is an already provisioned
  node, so its apply replaced 5 files and the other components were already in place.
- The 4-Mac recipe on its own cabling, and the first-request stall seen with one TCP lane.
- The lights hook only in preview (A15 installed the lights on the Macs).
