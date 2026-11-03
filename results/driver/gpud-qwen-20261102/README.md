# Qwen3.5-9B on one D700 through gpud (2026-11-02)

Source: trashcan main 1d21c098 (`driver/`), gpud built in the kbuild VM; engine binary from main 16a91c5e, unchanged;
Debian's Mesa RADV with `libgpud_drm.so` preloaded. Machine: incite, 02:00.0 on vfio-pci under `gpud serve` (boot
clocks: no DPM in gpud), 06:00.0 on amdgpu with `power_dpm_force_performance_level` auto. Model:
`Qwen3.5-9B-trashcan-Q4_0.gguf` (docs/qwen.md). gpuq lease #329.

## Correctness

`trashcan qwen-validate MODEL OUT --gpus 0 --context 2048 --batch 128 --greedy 10` (`qv-native.txt`, `qv-gpud.txt`):
one stage, all 32 layers and the head on one GPU, 5523.0 MiB allocated. Both give greedy tokens
`[9419, 0, 2500, 628, 353, 1438, 488, 3242, 30, 248046]`, and all 371 activation dumps of the reference prompt
(every layer's intermediates and the logits) are bit-identical between amdgpu and gpud.

`trashcan qwen-chat MODEL "Explain in three sentences what a GPU driver does." --gpus 0 --context 2048 --batch 128
--max-tokens 120` (seed 1, temperature 0.7; `chat-native.txt`, `chat-gpud.txt`): the same 82-token answer from both.

## Speed

| | prefill (23 tokens) | decode (82 tokens) |
|---|---|---|
| amdgpu, DPM auto | 31.64 tok/s | 24.68 tok/s |
| gpud, boot clocks | 4.58 tok/s | 2.12 tok/s |

At boot clocks the VRAM copy test runs at 15.7 GB/s (205 GB/s under amdgpu at DPM auto, `../gpud-radv-20261102`); a
decode token reads about 5.59 GB of weights, so the boot-clock ceiling is about 2.8 tok/s and gpud reaches 75% of it.
The gap to amdgpu is the clocks: SI DPM is not ported.

## The bug this needed

The first run through gpud stopped after 380 compute jobs: fence stuck for minutes, GRBM_STATUS 0xa0003028 (CP busy,
no shader waves), no VM faults, ring rptr == wptr, the CP alternating between SURFACE_SYNC and EVENT_WRITE_EOP
headers. gpud_submit wrote the ring without checking for room; rings are sized for amdgpu_sched_hw_submission (2)
largest jobs, a limit amdgpu's DRM scheduler keeps, and the engine queues work ahead, so the 8 KiB compute ring
wrapped over unread packets. gpud now waits for the job two back on the ring before writing (main 1d21c098). The
`serve.log` here is the server of the runs after the fix.
