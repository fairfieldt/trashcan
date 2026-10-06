# incite: Mac Pro 6,1 ("trashcan")

Probed 2026-10-02 over ssh (`ssh incite`, tailnet). Setup lives in `../homelab/hosts/incite/bootstrap.sh`;
models are listed in `../homelab/modelstore/manifest.yaml` and copied to `~/models` on incite.

## Host

| | |
|---|---|
| CPU | Xeon E5-2697 v2 (Ivy Bridge-EP), 12C/24T, 2.7 GHz, AVX only (no AVX2/FMA/F16C) |
| RAM | 64 GB DDR3 (62 GiB usable), 78 GB swap |
| OS | Debian 13, active kernel `6.12.111-p2p` (confirmed 2026-10-03); stock `6.12.111+deb13-amd64` remains installed |
| GPU driver | amdgpu with `radeon.si_support=0 amdgpu.si_support=1`; patched `amdgpu.ko` in `/lib/modules/$(uname -r)/updates/` |
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
memory bandwidth. `~/D700macpro` on incite has two fixes: an exact-match VFCT VBIOS selection fix, plus a
backport of Timur Kristóf's 2026-09-23 SI `HARDWAREDC` power-source patch. Without them the second GPU
gets the wrong VBIOS and the cards stay in the battery power state. The patched module was built and
installed at 19:14 on 2026-10-02. The VFCT fix is prepared for upstream (amdgpu and radeon) in
`kernel/upstream/vfct-exact-match/`.

Idle reads 300/150 in `/sys/class/drm/card{0,1}/device/hwmon/hwmon*/freq{1,2}_input`. That's expected at
idle; under load it should reach 850/1370. **Check those clocks during every benchmark.**

## Kernel

`kernel/build.sh` builds Debian's 6.12.111 source plus the D700 patches and PCI P2P, as `6.12.111-p2p` (about 28
minutes on incite). It was installed 2026-10-02.

- Since 2026-10-02 it's the **default** boot entry: `GRUB_DEFAULT="gnulinux-advanced-59c789b8-55b5-41a4-ae2f-fba0c8922d2b>gnulinux-6.12.111-p2p-advanced-59c789b8-55b5-41a4-ae2f-fba0c8922d2b"`.
- The original `/etc/default/grub` is in `/etc/default/grub.bak-trashcan`. The stock kernel (`6.12.111+deb13-amd64`) is still in the GRUB menu under Advanced options.
- To boot stock once: `sudo grub-reboot "gnulinux-advanced-59c789b8-55b5-41a4-ae2f-fba0c8922d2b>gnulinux-6.12.111+deb13-amd64-advanced-59c789b8-55b5-41a4-ae2f-fba0c8922d2b"`.
- A Debian kernel upgrade won't replace the P2P kernel as default, because the default is pinned by entry id. If the
  `6.12.111+deb13` packages are removed, re-run `update-grub` and check the default still exists.
- Under it, each card gets its own VBIOS (02:00.0 `113-C3861XA-028`, 06:00.0 `113-C3861XB-028`), so the D700
  patches are in effect.
- The `apple-gmux` WARN at boot also happens on the stock kernel.

## Fan curve

One SMC fan (`applesmc`, 790-1900 rpm) cools both D700s and the CPU; the GPUs' own `pwm1` drives nothing. Since
2026-10-05 `fancurve.service` (`scripts/fancurve.sh`, installed in `/usr/local/sbin`) sets it from the hottest of the
two GPU edge sensors and the CPU package: 790 rpm at 40 °C or below, linear to 1900 rpm at 50 °C. Idle is about 58 °C,
so in practice the fan stays at 1900 rpm. An unreadable sensor means full speed. `systemctl stop fancurve` returns
the fan to SMC control. To change the thresholds, set `Environment=T_LOW=... T_HIGH=...` in a drop-in.

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
