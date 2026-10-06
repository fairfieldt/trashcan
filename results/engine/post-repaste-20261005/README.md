# Post-repaste hardware and performance check — 2026-10-05

Both D700s were repasted and incite reassembled. This rerun uses the pinned `prefill-tiled-20261003`
candidate binary (SHA256 `ca0c38ba…43e2`, identical), so any change is due to the hardware.

## Hardware

Debian kernel `6.12.111-p2p`. Both GPUs show: PCIe 8 GT/s x16, a 256 MB BAR0 and 6144 MiB of VRAM.
They load their own VBIOS: 02:00.0 `113-C3861XA-028`, 06:00.0 `113-C3861XB-028`. Both reach
850/1370 MHz under load. RADV shows both devices. EDAC corrected and uncorrected counts are 0.
The only boot WARN is the known `apple-gmux` one. The kernel logged no amdgpu fault, reset or
thermal message during the runs. Under manual `fan1_output=1900`, the SMC fan spun from 831 to
about 1900 rpm in about 15 s (`fan-max-set.txt`).

## Short benches (`run.sh`, pp/tg128/batch512)

| Run | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| pp512 reps 1–4 | 166.80 / 173.60 / 170.34 / 173.78 | 25.80 / 28.68 / 26.39 / 26.55 |
| pp4096 reps 1–4 | 168.77 / 168.72 / 168.56 / 168.11 | 25.17 / 24.87 / 24.58 / 24.54 |
| 10-03 candidate, pp512 | 174.44 / 174.35 | 28.25 / 28.27 |
| 10-03 candidate, pp4096 | 168.99 / 168.80 | 26.39 / 26.25 |

Prefill matches the 10-03 run. Decode is lower and noisier, but the profiled pp4096 decode token
spends the same time in GPU kernels as before: 34.09/34.05 ms now vs 34.12/34.09 ms. All of the
difference is in host-side gaps (`*.tsv`). Its cause is unresolved. CPU `thermal_throttle`
counters stayed at 0.

## 80K long run (`run-80k.sh`), stopped at user request

Stopped after 40,960 tokens (80 chunks). The fan was at manual max from position 9,216 onward;
the 10-03 run used automatic SMC fan control. All 80 top-1 IDs and scores match the 10-03 run
exactly. At 40,960 tokens the elapsed time is 319.9 s, vs 314.5 s on 10-03 (+1.7%).

| First ~315 s of load | Peak GPU temp | Samples with card1 at 450 MHz |
|---|---|---:|
| 10-03 (auto fan) | 98/96 °C | 142 of 1135 |
| Post-repaste (max fan) | 98/96 °C | 159 of 1155 |

The CPU package peaked at 93 °C (coretemp crit 99 °C). Under sustained load, thermals match
the pre-repaste run, even at max fan. The repaste shows no measurable gain, and no damage.
