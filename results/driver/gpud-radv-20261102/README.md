# gpud milestones 4-5 on incite: SI rings in userspace, Mesa RADV on gpud (2026-11-02)

Source: trashcan main 39d54ccd (rings) and 41b0f026 (serve, client, RADV), `driver/`, built in the kbuild VM. Machine:
incite, 02:00.0 on vfio-pci as in `../vfio-20261102`, 06:00.0 on amdgpu (the oracle), gpuq lease #328. The engine
binary is trashcan main 16a91c5e's, unchanged; RADV is Debian's Mesa, unchanged, with `libgpud_drm.so` preloaded.

## Bring-up (`gpud init`)

`init-1..3.txt`: Linux 7.3-rc7's si_common, gmc_v6_0, si_ih, gfx_v6_0 and si_dma run in gpud; the two DMA rings pass,
the GFX ring stalls (the ME waits on the CE: the CP keeps state a previous driver's halt left). `init-4-*.txt`,
`init-5.txt`: with a GFX soft reset before CP init, all five rings (gfx, two compute, two DMA) pass ring and IB
tests, init 156 ms. `init-4-vram1.txt` (rings in VRAM, a diagnostic) ended in a segfault in the teardown. (That soft
reset also re-enabled coarse clock gating, which made later teardowns hang the GPU: `../gpud-teardown-20261102`.)

## RADV through gpud serve

- `vulkaninfo-gpud*.txt`: RADV TAHITI on 0000:02:00.0 through gpud. `drminfo-*.txt`: AMDGPU_INFO answers from gpud
  and from amdgpu on 06:00.0; they match except min clocks, heap sizes and UVD firmware.
- `vkcompute-gpud.txt`: `c = a * b + i` on 4M floats exact in host-visible and device-local memory; 20 copies of
  256 MiB in VRAM at 15.7 GB/s.
- `bench-native-*.txt` (amdgpu, 06:00.0) and `bench-gpud-*.txt` (gpud, 02:00.0): the engine's norm, top-k, attention
  and prefill-attention micro-benchmarks all pass on both, with the same error figures.

## Speed: clocks, not the driver

gpud runs the GPU at its boot clocks: there is no DPM (si_dpm/SMC) in gpud yet. Times, native vs gpud: norm 65 vs
168 us, fused norm 19 vs 55 us, prefill attention 74 vs 184 ms (scalar) and 4.9 vs 19.7 ms (tiled). The VRAM copy on
amdgpu (`vkcompute-amdgpu-06-*.txt`, copied from the terminal):

| | copy bandwidth |
|---|---|
| amdgpu, DPM auto | 205.1 GB/s |
| amdgpu, `power_dpm_force_performance_level` low | 18.4 GB/s |
| gpud (boot clocks) | 15.7 GB/s |

The host-visible case is faster on gpud (8 ms vs 74-78 ms); not investigated (gpud's host buffers are ordinary
cached pages of a memfd; how amdgpu maps RADV's host-visible types there wasn't checked).
