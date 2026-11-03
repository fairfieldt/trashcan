# Both D700s on gpud, Qwen3.5-9B tensor parallel across them (2026-11-02)

Source: trashcan main 10225e20 (`driver/`), gpud and libgpud_drm.so built in the kbuild VM; engine binary from main
16a91c5e for both the native and the gpud runs; Debian's Mesa RADV 25.0.7. Machine: incite, booted text-only
(`systemctl set-default multi-user.target`, gdm off: it ran gnome-shell on 06:00.0) with the one-time `gpud-iommu`
entry; gpuq lease #329. Model: `Qwen3.5-9B-trashcan-Q4_0.gguf`.

## Native baseline (both GPUs on amdgpu)

`qwen-bench MODEL --tp --pp 512 --tg 128 --reps 2` under `scripts/with-clocks.sh` (`bench-native-tp2.txt`,
`clocks-native-tp2.txt`): pp512 252.7 / 260.5 tok/s, tg128 50.5 / 51.0 tok/s; both cards at 850/1370 MHz under load,
47 / 46 C max. Chat (`chat-native-tp2.txt`): prefill 23 tokens 49.98 tok/s, decode 82 tokens 50.39 tok/s.

## gpud

Both GPUs moved to vfio-pci (`to-vfio.sh 02`, `to-vfio.sh 06`; groups 51 and 52), one `gpud serve` each
(`serve-all.sh 02 06`; `serve-02.log` with VBIOS 113-C3861XA-028, `serve-06.log` with 113-C3861XB-028, DPM on both).
RADV through `libgpud_drm.so` sees both (`trashcan devices`: GPU0 02:00.0, GPU1 06:00.0, as native); vkcompute on
06:00.0: exact, 205.8 GB/s.

| | pp512 | tg128 | chat prefill (23) | chat decode (82) |
|---|---|---|---|---|
| amdgpu | 252.7 / 260.5 | 50.5 / 51.0 | 49.98 | 50.39 |
| gpud | 218.0 / 221.2 | 40.5 / 40.3 | 36.90 | 41.43 |

tok/s; `bench-gpud-tp2.txt`, `chat-gpud-tp2.txt`. The chat text is the same as native TP2. Clocks were not sampled on
the gpud run (gpud has no hwmon; `with-clocks.sh` reads amdgpu's).

The engine's Qwen TP moves each GPU's partial through host memory, the CPU adds them and writes the total back, and
the CPU synchronizes (VkEvents for decode, host-signalled timelines for prefill): two independent gpud servers and
a client that keeps per-device state are enough. The gap to amdgpu (85% prefill, 80% decode; 84% for one GPU) is
not explained yet.

While both GPUs are on vfio-pci, fancurve.service cannot read their temperatures through amdgpu and holds the fan
at 2350 rpm; the Mac SMC's own GPU diode sensors (TG0D / TG1D) read about 40 C at idle.
