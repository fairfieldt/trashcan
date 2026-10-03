# incite: Mac Pro 6,1 ("trashcan")

Probed 2026-10-02 over ssh (`ssh incite`, tailnet). Setup lives in `../homelab/hosts/incite/bootstrap.sh`;
models are listed in `../homelab/modelstore/manifest.yaml` and copied to `~/models` on incite.

## Host

| | |
|---|---|
| CPU | Xeon E5-2697 v2 (Ivy Bridge-EP), 12C/24T, 2.7 GHz, AVX only (no AVX2/FMA/F16C) |
| RAM | 64 GB DDR3 (62 GiB usable), 78 GB swap |
| OS | Debian 13, kernel 6.12.111+deb13-amd64 |
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

Per-card nominal: 2048 shaders, 850 MHz core, 1370 MHz GDDR5 on a 384-bit bus, so about 264 GB/s
and 3.5 TFLOPS fp32.

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
installed at 19:14 on 2026-10-02.

Idle reads 300/150 in `/sys/class/drm/card{0,1}/device/hwmon/hwmon*/freq{1,2}_input`. That's expected at
idle; under load it should reach 850/1370. **Check those clocks during every benchmark.**

## Baseline so far

`llama-bench` on upstream 99b9548 (Vulkan, `-ngl 99`, default layer split), Gemma 4 12B QAT Q4_0:

| test | t/s |
|---|---|
| tg256 | 8.59 |

That's about 56 GB/s of effective weight streaming against a 264 GB/s per-card peak. The layer-split
ceiling from bandwidth alone is about 38 tok/s, so the single-GPU kernels have a lot of headroom
before TP even matters.
