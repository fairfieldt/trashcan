# A15: port lights shipped in trashcan main (2026-10-26)

Tested source: trashcan main f2a2fe68 (`lights/`, `docs/lights.md`), imported from homelab portlights d876462.
`portlights` built in the kbuild VM (`cargo build --release -p portlights --target x86_64-unknown-linux-gnu`),
sha256 bc6229899a33... (deployed as /usr/local/bin/portlights on all six). Kernel on all six: 7.3.0-rc7-p2p-s2
(without 0009/0010 built in) with the applesmc override in /lib/modules/KREL/updates.

## Driver build

`lights/driver.sh build --kernel 7.3.0-rc7-p2p-s2 --headers <indite's /usr/src/linux-headers-...> --host-tools
~/kernel/7.3/linux-7.3-rc7` in the kbuild VM (arm64, cross): applesmc.c v7.3-rc7 from kernel.org, sha256
3f930296... (same file at v7.3), patches 0009+0010. The .ko is bit-identical (sha256 26cbf1c7...) to the one
`kernel/applesmc-override.sh` installed on 2026-10-19. Not tested: `--build-module` / a native build on a Mac
(lab rule: no builds on the Macs); header/kernel mismatch and unknown versions fail as intended (checked in the VM).

## One node (indite, gpuq #143)

- `onenode-1-install.log`: upgrade over the homelab install with `--conductor 7465`, then the same and a plain run:
  everything "unchanged", rc 0.
- `onenode-2-gpu.log`: Gemma 4 12B `bench --pp 512 --tg 256` (trashcan-4f36bf0): node went to blinkenlights, back to
  smc 10 s after the GPUs went quiet (port_lights auto, power_light breathe). Node CPU 2.11 s over 33.9 s (6.2% of one
  core) during the run; 0.117 s over 30 s idle (0.4%).
- `onenode-3-stop.log`: stop -> port_lights auto, power_light on (steady); start -> breathe; SIGKILL -> ExecStopPost
  gives the lights back (power on), Restart=always restarts in 5 s; SIGTERM of a standalone `blinken` -> auto/on.
- `onenode-4-smc-latency.log` (`tools/smc-latency.py`, 400 fan1_input reads 10 ms apart): node idle 0.44 ms median /
  0.77 p95; stopped 0.43/0.59; blinken busy 1.0 0.49/1.20 (max 2.14); busy 0.35 0.94/1.41 (max 2.94).
  `tools/smc-sim.rs` (the node's frame/threshold logic, 600 s): 50 writes/s, 475-754 LED changes/s, i.e. ~19-30% SMC
  occupancy at 0.4 ms per LED, an estimate.
- `onenode-5-driver.log`: `driver.sh uninstall` (stock applesmc; node waits, "no port lights"), `install` (20 lights,
  node reopens), repeat install "unchanged"; fancurve active with pwm1_enable 1 and target 2350 throughout.
  Found: ExecStopPost printed "Permission denied" without the driver, and after a raw rmmod/modprobe while idle the
  power light stayed steady. Both fixed before the commit (`[ -e port_lights ]`; idle check every 2 s):
  `onenode-6-raw-reload.log` shows on -> breathe within 12 s.
- `onenode-7-uninstall-install.log`: full `uninstall.sh` (service, binary, config, driver; stock applesmc, fancurve
  kept), repeat (nothing to do, rc 0), `install.sh --module ... --conductor 7465` (dry run, then real).
- `onenode-8-conductor.log`: UDP frames from pinkerton -> "conductor", then back to "smc".

## Six Macs (chain6, `tools/pair.sh`)

Same binary (trashcan-4f36bf0) and settings as portlights-20261026 demo pp2k: Flash-Next G-iq4, `bench --gpus 0,1
--head-split 12 --context 32768 --ngram disk --batch 128 --pp 2048 --tg 512`, chain from ring.py. portlights-node
running (on) or stopped (off) on all six, order on, off, off, on. `pair/summary.txt`:

| Run | pp tok/s | tg tok/s | node CPU s per Mac (35 s run) |
|---|---|---|---|
| on1 | 881.1 | 21.410 | 1.73-2.17 |
| off1 | 883.5 | 20.256 | - |
| off2 | 885.5 | 20.628 | - |
| on2 | 885.0 | 20.423 | 1.87-2.44 |

On mean 20.92, off mean 20.44 tok/s: no cost visible; the spread between repeats (~1 tok/s) is larger than any
difference. Clocks 850/1370 throughout (one 300/150 sample on on1 card0). Every Mac logged blinkenlights in both
on runs; fan target stayed 2350 (fancurve's floor), pwm1_enable 1, fancurve active in all runs (`pair/mon-*`).
After the pair: all six run the shipped version with `--listen 7465` (`pair/after.txt`).
