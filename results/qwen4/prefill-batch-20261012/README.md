# Flash-Next prefill cost by chunk size (2026-10-12)

Why a ~20-token prompt takes about 1 s through the six-Mac chain (results/qwen4/chain-20261012, Server).

incite, binary of c21ec4e (the engine holds the GPUs at the high DPM level while working; no clock log was taken, and
the 1-token layer times match the high-clock decode times). G-iq4, the first Mac's stages: layers 0-3 on GPU0, 4-7 on
GPU1, each with a twelfth of the head, context 32768. For each batch B:

```sh
trashcan qwen4 stage-bench G-iq4-00001-of-00006.gguf --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch B \
    --head-rows 0:21504,21504:43008 --decode 4 --start 64 --prefill 3
```

`prefill` is the mean of 3 chunks of B tokens at position 0 (after a warm-up chunk). GPU ms for the 8 layers, phases
summed over layers ([summary.txt](summary.txt), `sb-batchB.json`):

| batch | wall | per token | mixers (16) | DeltaNet (6) | router | experts | QSA (2) projections + index + attention |
|---|---|---|---|---|---|---|---|
| 1 | 11.0 | 11.0 | 2.7 | 2.9 | 0.8 | 2.3 | 1.1 |
| 4 | 21.4 | 5.4 | 4.9 | 4.8 | 0.9 | 7.5 | 1.7 |
| 8 | 40.8 | 5.1 | 8.6 | 9.4 | 2.6 | 14.4 | 3.3 |
| 15 | 124.5 | 8.3 | 51.7 | 28.4 | 3.6 | 26.7 | 9.4 |
| 16 | 157.5 | 9.9 | 51.8 | 28.6 | 4.0 | 58.4 | 9.7 |
| 24 | 179.6 | 7.5 | 52.1 | 30.4 | 4.8 | 76.4 | 10.1 |
| 32 | 200.8 | 6.3 | 52.7 | 32.0 | 6.0 | 93.2 | 10.5 |
| 64 | 278.8 | 4.4 | 56.4 | 39.4 | 10.1 | 149.8 | 12.8 |
| 128 | 453.1 | 3.5 | 61.8 | 69.4 | 17.0 | 259.0 | 21.2 |

(The n-gram module of layer 1 adds 0.4-4.5 ms, included in the wall time.)

- **Mixers.** Up to 8 tokens the fused mixer kernels (`qwen4_mix_down`/`qwen4_mix_up`) take 0.17-0.54 ms per mixer;
  from 9 tokens the prefill path (`qwen4_hc_norm`, Q8_0 GEMMs, `qwen4_act`, `qwen4_hc_mix`) takes ~3.2 ms per mixer
  whatever the batch (51.7 ms at 15 tokens, 61.8 at 128) for about 7 MB of weights. A fixed cost that large points at
  too little parallelism (the 10240 -> 320 bottleneck gives a 64 x 64 tiled GEMM 5 row tiles), not at bandwidth.
- **Experts.** The per-(token, slot) kernels cost ~1.8 ms per token for 8 layers up to 15 tokens; at 16 the grouped
  kernels take over (`GROUPED_MIN`) and cost 2.2 times as much (58.4 against 26.7 ms at 15). Linearly extrapolated, the
  per-slot path (~228 ms) would still be below the grouped one (259 ms) at 128 tokens.
- DeltaNet grows from 9.4 ms at 8 tokens to 28.4 at 15 (the prefill path's projections and the sequential
  recurrence).

A 21-token chat turn is one chunk of 21 through 48 layers: ~6 x 170 ms, which is the ~1 s to first token measured.
