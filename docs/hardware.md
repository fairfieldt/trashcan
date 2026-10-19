# incite: Mac Pro 6,1 ("trashcan")

Probed 2026-10-02 over ssh (`ssh incite`, tailnet). Setup lives in `../homelab/hosts/incite/bootstrap.sh`;
models are listed in `../homelab/modelstore/manifest.yaml` and copied to `~/models` on incite.

## Host

| | |
|---|---|
| CPU | Xeon E5-2697 v2 (Ivy Bridge-EP), 12C/24T, 2.7 GHz, AVX and F16C (no AVX2/FMA) |
| RAM | 64 GB DDR3 (62 GiB usable), 78 GB swap |
| OS | Debian 13, kernel `7.3.0-rc7-p2p` (default since 2026-10-14, see [kernel.md](kernel.md)); `7.2.9-p2p` (the fallback) and stock `6.12.111+deb13-amd64` remain installed |
| GPU driver | amdgpu with `radeon.si_support=0 amdgpu.si_support=1`; the D700 fixes are built into the `-p2p` kernels |
| Vulkan | RADV (Mesa 25.0.7-2+deb13u1), Vulkan 1.3.305 |

## GPUs: 2x FirePro D700 (Tahiti XT, GCN 1.0 / Southern Islands)

| | GPU0 | GPU1 |
|---|---|---|
| PCI | 02:00.0 | 06:00.0 |
| Root port | 00:02.0 | 00:03.0 |
| Link | PCIe 3.0 x16 (8 GT/s, x16) | PCIe 3.0 x16 |
| BAR0 | 256 MB (no resizable BAR) | 256 MB |
| VRAM | 6 GiB | 6 GiB |

Both GPUs hang directly off CPU root ports, so there's no PCIe switch between them. Any GPU-to-GPU path
goes through the CPU root complex. With a 256 MB BAR, only 256 MB of each card's VRAM is CPU/peer visible.

Apple specifies 2048 shaders, a 384-bit memory bus, 264 GB/s and 3.5 TFLOP/s per D700
([Mac Pro technical specifications](https://support.apple.com/en-la/112025)). At the measured
850 MHz core / 1370 MHz memory clocks, the clock-derived physical peaks are **3.4816 TFLOP/s fp32**
(`2048 × 2 × 850 MHz`, counting FMA as two operations) and **263.04 GB/s**
(`1370 MHz × 4 transfers × 384/8 bytes`). Apple's figures are rounded nominal specifications.
GB/s here is decimal; GiB means `2^30` bytes.

Two cards provide 6.9632 TFLOP/s and 526.08 GB/s only when useful work runs concurrently on both.
Layer-split single-sequence decode visits the cards sequentially, so it cannot use the combined
bandwidth as its weight-streaming denominator. Tensor parallelism enables concurrency but adds
replicated work and synchronization. These figures are hardware upper bounds, not sustainable
kernel or application performance; they omit dequantization, reductions, instructions other than
FMA, and transport overhead.

What ggml-vulkan reports:

```
AMD Radeon R9 200 / HD 7900 Series (RADV TAHITI) (radv) | uma: 0 | fp16: 0 | bf16: 0 | fp4: 0 |
warp size: 64 | shared memory: 32768 | int dot: 0 | matrix cores: none
```

So it has no fp16 arithmetic, no integer dot product, and no cooperative matrix. Subgroup size is 64,
shared memory 32 KB, max workgroup 1024 invocations.

## Clock bug (relevant to every benchmark)

Stock amdgpu leaves both D700s at **300 MHz core / 150 MHz memory** even under load, which costs about 9x
memory bandwidth. Two fixes cure it: an exact-match VFCT VBIOS selection fix, plus a backport of Timur
Kristóf's 2026-09-23 SI `HARDWAREDC` power-source patch. Without them the second GPU gets the wrong VBIOS
and the cards stay in the battery power state. Both `-p2p` kernels carry them ([kernel.md](kernel.md)); the
stock Debian kernel gets them from a patched `amdgpu.ko` in `/lib/modules/6.12.111+deb13-amd64/updates/`
(built from `~/D700macpro` on 2026-10-02). The VFCT fix is prepared for upstream (amdgpu and radeon) in
`kernel/upstream/vfct-exact-match/`.

Idle reads 300/150 in `/sys/class/drm/card{0,1}/device/hwmon/hwmon*/freq{1,2}_input`. That's expected at
idle; under load it should reach 850/1370. **Check those clocks during every benchmark.**

## Kernel

See [kernel.md](kernel.md): which kernel incite runs, its patches and config, how to build it on the Mac, and how
to boot, check and recover it.

- GRUB's default is pinned by entry id in `/etc/default/grub` (original in `/etc/default/grub.bak-trashcan`), so
  installing a kernel never changes what boots.
- Under both `-p2p` kernels each card gets its own VBIOS (02:00.0 `113-C3861XA-028`, 06:00.0 `113-C3861XB-028`).
- The `apple-gmux` WARN at boot happens on every kernel.

## Network

Two Broadcom BCM57762 ports (`tg3`). Since hard power cuts on 2026-10-06 the left one (`enp12s0`) gets no link; the
cable is in `enp11s0` (DHCP; reach it as `incite.tdf.sh`). NetworkManager ignores `thunderbolt*`
interfaces (`/etc/NetworkManager/conf.d/90-thunderbolt-unmanaged.conf`); otherwise it runs DHCP on them and reports
failed connections.

Kernel messages stream to the Mac over netconsole, and an oops, lockup or panic reboots the machine after 10 s; see
[kernel.md](kernel.md#when-something-goes-wrong).

## Thunderbolt

Six Thunderbolt 2 ports on three Falcon Ridge 4C controllers (DSL5520). Each controller's NHI sits behind a PCIe
Gen2 x4 link, and all three share a PLX PEX 8723 switch with a Gen3 x8 uplink to root port 00:01.0. Facing the rear
panel (Apple KB HT5918):

| | Left | Right |
|---|---|---|
| Top | Bus 1 (`domain1`, NHI 5d:00.0) | Bus 2 (`domain2`, NHI a4:00.0) |
| Middle | Bus 1 | Bus 2 |
| Bottom | Bus 0 (`domain0`, NHI 16:00.0) | Bus 0, shared with HDMI |

A cable between two ports of one controller loops that controller and gets one lane; between controllers it bonds
two lanes (20 Gb/s per direction on the wire). Measured on 7.2: about 9.6 Gb/s per direction per cable, 25 us
round trips over USB4STREAM and 40 us over `thunderbolt_net` with `throttling=32000`
([results](../results/kernel/tb-loopback-20261006/README.md)).

With three cables looping the three controllers into a triangle, a reboot hangs in firmware before the startup chime;
unplug them first ([kernel.md](kernel.md#installing-and-booting)).

Between two Mac Pros (incite and invite, one cable per controller pair): 8.6-9.3 Gb/s per link each way, 25.9 Gb/s
over all three at once, 0.18 ms ping; a model copy with `tar | socat` runs at 505 MB/s, limited by the receiving
Apple SSD ([results](../results/kernel/tb-nodes-20261006/README.md)). After one Mac reboots, the other can keep a
stale peer from its firmware; reload `thunderbolt` on both, and replug a link that comes back with one lane. After
several Macs reboot, `scripts/tb-reset.sh HOST...` resets all their controllers together (unload and NHI reset on
every Mac, then the modules loaded on all at once), which brought the six cables of the four TP8 Macs back on two
lanes where resetting one Mac after another left links stale; a link still down after two rounds needs a replug
([results](../results/qwen/tp8-mesh-20261009/README.md#cables)). Create the streams again afterwards.

## Fan control

One SMC fan (`applesmc`) cools both D700s and the CPU; the GPUs' own `pwm1` drives nothing.
`fancurve.service` (`scripts/fancurve.sh`, installed in `/usr/local/sbin`) keeps the fan at
**2350 RPM or the live SMC maximum, whichever is higher, at every temperature** (since 2026-10-12;
`FAN_RPM` in the unit's environment changes the floor, `FAN_RPM=0` follows the SMC maximum alone). It reads
`fan1_max` every two seconds and reasserts manual mode and the exact target if the SMC changes either.

Linux 7.3 moved the fan files. Before 7.3 they are in `/sys/devices/platform/applesmc.768/`, with manual mode as
`fan1_manual` (1 manual, 0 automatic) and the target as `fan1_output`. From 7.3 they are hwmon attributes in
`/sys/devices/platform/applesmc.768/hwmon/hwmonN/`: `pwm1_enable` (1 manual, 2 automatic), `fan1_target`, and
`fan1_input`, `fan1_min` and `fan1_max` as before. `fancurve.sh` and the telemetry (`system-crash-capture.py`) use
whichever layout is there. Without the service, the SMC runs the fan at about 790 RPM on its own.

The SMC's maximum is not fixed. It reads 1900 RPM at boot; on 2026-10-09 it then rose in 50 RPM steps to
2359 (incite) and 2382 (indite) within 40 minutes and drifted back to about 1910 over the next six hours.
After the 2026-10-12 boots it stayed at 1900 for over an hour on both Macs, through 95 °C GPU load. The
input that drives it is unknown. A target above it works: on indite with `fan1_max` at 1900, a manual target
of 2350 reached 2349 RPM within 30 s. Following the maximum alone (2026-10-07 to 10-12) therefore left the
fan anywhere between 1900 and ~2380 RPM. At 1900 RPM, a warm 80K-context prefill settles at about 96/94 °C
on incite and 92/93 °C on indite. These are fan-speed measurements, not airflow. The 2026-10-07 probe
evidence is under `results/qwen/hardware-20261007/fans/`.

Two transient maximum-read failures may retain the last valid limit; the third failure returns control
to automatic SMC and exits for systemd to retry. At startup, an unavailable/invalid limit has no fallback.
Malformed limits, control/target read errors, or failed readback also return control to the SMC.
`systemctl stop fancurve` returns automatic SMC control. `INTERVAL` may be set to 1–10 seconds;
`SMC_PATH` is an optional sysfs path override used by local tests. The service is enabled by the existing
node system provisioner. Deploy only the fan script/unit and restart this service when changing fan policy;
the full system provisioner also restarts unrelated services.

## Port lights

The SMC lights the 20 LEDs behind the I/O panel for 15 s when the accelerometer reports motion. Patch 0009
(`docs/kernel.md`) makes them LED class devices (`/sys/class/leds/smc::*`) with `port_lights` (`auto`/`manual`),
`port_lights_motion`, `port_lights_hold_ms`, `port_lights_show` and `port_lights_levels` in
`/sys/devices/platform/applesmc.768/`. The SMC
keys, the LED map and how they were found are in `results/kernel/port-lights-20261018/`.

## Suspend is disabled

incite does not resume from suspend; it drops off the network and needs a power cycle. Since FreedomBox was removed
(2026-10-05), the GDM greeter's `gsd-power` suspended it after 900 s with no local activity. SSH sessions don't count
as activity, so this hit a P2P test at 22:09 that evening. Since 2026-10-05:

- `sleep`, `suspend`, `hibernate`, `hybrid-sleep` and `suspend-then-hibernate` targets are masked
  (`systemctl unmask ...` to undo).
- `/etc/gdm3/greeter.dconf-defaults` sets `sleep-inactive-ac-type='nothing'` (original in
  `greeter.dconf-defaults.bak-trashcan`, applied with `/usr/share/gdm/generate-config`). fairfieldt's own GNOME
  setting is `nothing` too.

The `trashcan devices` report confirms the shape: 32 CUs (2 SE x 2 SA x 8), 4 SIMD16 per CU, wave64, 10 waves per
SIMD, 256 VGPRs per SIMD, 32 KiB shared memory per workgroup, 5888 MiB VRAM heap, plus a 256 MiB CPU-visible BAR
heap.

The machine also logged a corrected ECC error on CPU0 channel 0 DIMM 0 during testing.

## Baseline so far

`llama-bench` on upstream 99b9548 (Vulkan, `-ngl 99`, default layer split), Gemma 4 12B QAT Q4_0:

| test | t/s |
|---|---|
| tg256 | 8.59 |

The GGUF's matrix payload is 6,956,974,080 bytes: this corresponds to **59.76 GB/s** of original-weight
payload at 8.59 tok/s, rather than 56 GB/s (which mixed binary file size with decimal bandwidth).
The layer-split weight-only bound is **37.81 tok/s** at 263.04 GB/s, before attention and all other work.
The complete file is 6,975,879,296 bytes, including metadata and non-matrix tensors.

The later warmed engine measurements in
[`performance-20261003`](../results/engine/performance-20261003/README.md) reach **22.14 tok/s** for
tg128 after pp512 and **160.87 tok/s** for pp512. Its implemented repacks stream a minimum of
4,040,294,400 matrix bytes per GPU per decode token; the equivalent effective rate is 89.47 GB/s
per GPU over the complete forward, not a measured DRAM counter. The FFN microbenchmarks reach
153.00–160.41 GB/s of original Q4 payload, or **170.00–178.23 GB/s of the actual repacked payload**.
Do not compare original Q4 byte rates directly with the physical bus limit: the repack expands
18-byte blocks to 20 bytes. Clocks in those runs reach 850/1370 under sustained load.

The subsequent [main-branch iterations](../results/engine/main-tuning-20261003/README.md) reach
**23.42 tok/s** for pooled tg256 after pp512 and **21.36 tok/s** after a 4096-token prefill.
Those are different workloads and experiment phases; their fresh paired comparisons and exact
build hashes are preserved in that report. Fixed pp512 remains about 160 tok/s.

[`engine-plan.md`](engine-plan.md#cost-model) and the reproducible
[`ceiling audit`](../results/engine/ceiling-audit-20261003/README.md) separate exact tensor costs,
ideal rooflines, recorded throughput, and unvalidated optimization targets. The audit checked all
667 tensor names, dimensions and types directly from incite's GGUF header without executing a model.
