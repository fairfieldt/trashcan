# Kernel A/B: 6.12.111-p2p vs 7.2.9-p2p on incite (2026-10-06)

`run.sh` runs the same checks on whichever kernel is booted: system state and VBIOS per card, `xgpu p2p-verify` x3,
`xgpu p2p-wall`, then four `bench --pp 512 --tg 128 --batch 512` runs of the pinned engine binary
`results/engine/prefill-tiled-20261003/bin/candidate` (SHA256 `ca0c38ba…43e2`) with GPU clocks logged. Default
CPU settings (schedutil, turbo on), fan at 1900 rpm.

`7.2.9-p2p` is the first 7.2.9 build (package `7.2.9-1p2p`), which also carried the thunderbolt-ibverbs Thunderbolt
patches. `7.2.9-p2p-rev2` is the kernel now in use (`7.2.9-2p2p`, patches 0001-0005, see `docs/kernel.md`). Each
7.2 directory has `test-engine.log` from `scripts/test-engine.sh` with binary `ca0c38ba…`.

| | 6.12.111-p2p | 7.2.9-p2p (first build) | 7.2.9-p2p-rev2 (in use) |
|---|---|---|---|
| VBIOS 02:00.0 / 06:00.0 | 113-C3861XA-028 / 113-C3861XB-028 | same | same |
| Clocks under load | 850/1370 on both cards | same | same |
| `p2p-verify` | 3/3 PASS, export stays in VRAM | 3/3 PASS | 3/3 PASS |
| P2P write, submit-to-fence, 1 / 8 / 32 MiB | 8.17 / 8.21 / 8.34 GB/s | 7.84 / 7.70 / 8.08 GB/s | 8.11 / 7.91 / 8.25 GB/s |
| pp512 prefill, 4 runs (tok/s) | 172.34, 174.35, 170.24, 174.31 | 175.77, 175.05, 178.74, 177.53 | 175.58, 178.66, 178.60, 178.61 |
| tg128 decode, same runs (tok/s) | 26.90, 26.43, 28.71, 26.29 | 27.30, 27.36, 29.47, 27.79 | 27.09, 26.85, 29.55, 26.92 |
| `scripts/test-engine.sh` | not rerun here | 30/30 unit tests; 6/6 validations PASS, worst relative error 9.95e-4 | same |

Prefill is about 2-3% faster on 7.2. Decode varies more between runs than between kernels. P2P writes differ by
a few percent between runs (one run each; 7.9 GB/s on 6.12 on 2026-10-05), with no consistent direction.

New on 7.2: amdgpu drives SI's VCE 1 video encoder and asks for `amdgpu/vce_1_0_0.bin`, which Debian's firmware
doesn't ship. Only hardware video encoding is affected. The only boot WARN on either kernel is `apple-gmux`.
