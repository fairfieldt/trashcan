# SI power management on gpud (2026-11-02)

Source: trashcan main a3081523 (`driver/`): Linux 7.3-rc7's atom.c, amdgpu_atombios.c, si_dpm.c, si_smc.c and
legacy_dpm.c in gpud. Engine from main 16a91c5e, Debian's Mesa RADV with `libgpud_drm.so`. Machine: incite, 02:00.0
through gpud (VBIOS 113-C3861XA-028 from ACPI VFCT), 06:00.0 on amdgpu (DPM auto) for the native numbers. gpuq
lease #329.

## Power states

`init-dry2.txt`: a one-off build that skipped only si_dpm's hw_init, with `GPUD_DPM=1 GPUD_DEBUG=1` (amdgpu.dpm=1
prints every parsed state). Boot 450 MHz sclk / 150 MHz mclk at 881 mV; performance 300/150 @ 825 mV, 450/1370 @ 881,
650/1370 @ 893, 850/1370 @ 1062; ULV 300/150 @ 800. The top level is amdgpu's reported maximum (850 / 1370 MHz). Before
DPM, gpud ran at the boot level: memory at 150 MHz, which is the 15.7 GB/s it measured.

## With DPM

| | before (boot clocks) | gpud with DPM | amdgpu |
|---|---|---|---|
| vkcompute VRAM copy (20 x 256 MiB) | 15.7 GB/s | 204.0-204.6 GB/s (185-190 on a first run) | 205.1 GB/s |
| Qwen3.5-9B prefill, 23 tokens | 4.58 tok/s | 26.49 tok/s | 31.64 tok/s |
| Qwen3.5-9B decode, 82 tokens | 2.12 tok/s | 20.65 tok/s | 24.68 tok/s |

Chat: `qwen-chat ... --gpus 0 --context 2048 --batch 128 --max-tokens 120` (`chat-gpud-dpm.txt`, `chat-native.txt`),
same answer. `qwen-validate` with DPM on: the same greedy tokens and all 371 activation tensors bit-identical to amdgpu
(clocks do not change the arithmetic). The vkcompute numbers are from the terminal (no files).

Engine micro-benchmarks (`bench-gpud-dpm-*.txt` vs `bench-native-*.txt`): all pass; long kernels at native speed
(scalar prefill attention 73.49 vs 73.64 ms, attention 0.51 vs 0.50 ms), short ones pay gpud's per-submission cost
(norm 76 vs 65 us, fused norm 31.7 vs 19.0 us, tiled prefill attention 5.36 vs 4.94 ms).

## Start paths

- After amdgpu's teardown on a fresh boot (`init-dpm-fresh.txt`): DPM up, 204.0 GB/s.
- After gpud's own SIGTERM teardown (`init-dpm1.txt` runs si_dpm's hw_fini): DPM up again, 204.6 GB/s.
- After a SIGKILL: si_dpm refuses to start over a running SMC (`init-dpm-afterkill.txt`, hw_init -22). gpud now runs
  si_dpm's sw_init and hw_fini first when it finds the SMC running (`init-dpm-restart.txt`: "stopped the SMC a
  previous owner left running"), then DPM comes up, 204.2 GB/s.

Thermal interrupts from the SMC's controller arrive as under amdgpu ("IH: thermal high to low"). Clocks are inferred
from bandwidth and benchmark parity; gpud does not yet answer AMDGPU_INFO_SENSOR, so they were not read directly.
