# Where gpud's gap to amdgpu went (2026-11-02)

Source: trashcan main 10225e20, then 95fc385f (the fix) for `driver/`; engine binary from main 16a91c5e throughout;
Debian's Mesa RADV 25.0.7. Machine: incite, text-only boot, gpuq leases #329 and #330. Workload: Qwen3.5-9B Q4_0
tensor parallel over both D700s (`qwen-bench --tp`); per-kernel GPU times from the engine's own profiler
(`TRASHCAN_TP_PROFILE=FILE TRASHCAN_TP_PROFILE_DISPATCH=1`, summarized with tools/tp_profile.py, copy here).

## 1. Not the clocks

`clocks-gpud-tp2.txt` (`drminfo clockmon`, gpud's AMDGPU_INFO_SENSOR): both GPUs at 850/1370 MHz through the measured
phases, as amdgpu (`../gpud-tp2-20261102/clocks-native-tp2.txt`); the low-clock stretches are model load and upload.

## 2. Not the host side

Per decode token through gpud (`prof-gpud.jsonl.xz`): step 24.1 ms, of it GPU segments 23.1 ms. amdgpu's whole step
is 19.8 ms. The GPU itself ran the same work slower; launches and waits are not the gap.

## 3. Found: VRAM page-table fragments

Per kernel, gpud vs amdgpu (`prof-gpud-d`, `prof-native-d`): the 14 MB weight matvecs at native speed, the LM head
(124160x4096 Q6_K, 417 MB per GPU) 5938 vs 3282 us, small kernels 24-86% slower. `serve-02-A-unaligned.log`
(fragment size per mapping, GPUD_DEBUG): the engine's 256 MiB weight arenas sat at VRAM offsets like 64 MiB + 256 KiB
with 2 MiB-aligned virtual addresses, so they were mapped in 256 KiB fragments. RADV aligns large buffers' virtual
addresses to the 2 MiB fragment size and leaves physical alignment to the kernel, whose VRAM manager uses 2 MiB
blocks; gpud honoured only the requested alignment. With buffers of 2 MiB and more 2 MiB-aligned (main 95fc385f,
`serve-02-B-aligned.log`: every mapping in 2 MiB fragments):

| | headvec | GPU per token | TP2 tg128 | TP2 pp512 |
|---|---|---|---|---|
| amdgpu | 3282 us | 18.5 ms | 50.5 / 51.0 | 252.7 / 260.5 |
| gpud before | 5938 us | 23.0 ms | 40.5 / 40.3 | 218.0 / 221.2 |
| gpud, 2 MiB-aligned | 3321 us | 20.0 ms | 48.2 / 46.6 | 225.3 / 228.7 |

(`prof-A`/`prof-B`, `bench-gpud-tp2-b.txt`, `bench-gpud-tp2-frag.txt`.)

## 4. What remains: small kernels, not IOMMU translation, not shader placement

Small latency-bound kernels stay slower on gpud (attention merge 53.9 vs 29.3 us, gdn 32.4 vs 23.4, matf32 62.8 vs
46.2; `norm-bench` fused norm 31.7 vs 19.0 us, timed between GPU timestamps around one dispatch).

- IOMMU translation: amdgpu booted with its DMA translated (`intel_iommu=on`, no passthrough; entry `native-iommu`)
  is unchanged on these kernels (`nb-native-iommu.txt`: fused norm 18.96 us; `prof-native-iommu.jsonl.xz`: merge
  29.3, gdn 23.4) and loses about 2% overall (`bench-native-iommu.txt`: tg 49.6 / 49.9, pp 255.6 / 261.3).
- Shader code in host memory: `RADV_PERFTEST=dmashaders` (shaders DMA'd into invisible VRAM) changes nothing
  (`bench-C.txt.xz`, `prof-C.jsonl.xz`).
- Left: RADV's CPU-written buffers that amdgpu keeps in CPU-visible VRAM and gpud in host memory (descriptor pools,
  the upload buffer of push constants; `nb-debug.txt.xz`: VRAM-domain buffers with CPU access land in GTT). gpud
  gives clients no CPU-visible VRAM. Not tested: amdgpu cannot start with `amdgpu.vis_vramlimit=1` (gmc_v6_0 sw_init
  -12), so the native counter-experiment needs another way.
