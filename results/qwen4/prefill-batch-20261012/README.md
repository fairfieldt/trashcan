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

## The change: fused mixers to 64 tokens, per-slot experts at every size

Same stage-bench with two temporary switches (`thresholds/`, `thr-ab.sh`, binary of 0802e42 plus the switches),
8 layers, GPU ms per chunk:

| chunk | baseline | fused mixers to any size | per-slot experts | both | mixers: GEMM / fused | experts: grouped / per-slot |
|---|---|---|---|---|---|---|
| 16 | 157.4 | 120.2 | 127.3 | 89.5 | 51.8 / 14.8 | 58.6 / 28.4 |
| 32 | 200.7 | 175.4 | 163.5 | 138.1 | 52.6 / 27.4 | 92.9 / 55.9 |
| 48 | 241.7 | 225.5 | 203.6 | 186.8 | 55.5 / 39.8 | 121.9 / 83.5 |
| 64 | 278.4 | 274.8 | 240.6 | 235.8 | 56.4 / 52.2 | 149.5 / 111.0 |
| 96 | 380.1 | 398.3 | 342.4 | 360.2 | 59.3 / 77.4 | 205.0 / 166.1 |
| 128 | 456.4 | 496.8 | 416.2 | 458.8 | 61.9 / 102.4 | 259.2 / 221.2 |
| 256 (context 8192, twice) | 805.3, 798.7 | | 778.5, 780.7 | | | 463.4, 463.2 / 441.6, 441.7 |

The per-slot expert kernels beat the grouped ones at every chunk size and need 28 MiB less VRAM per GPU (no per-slot
output buffer), so the grouped path (`qwen4_moe_route`, `qwen4_moe_{gu,down}_grouped`, `qwen4_moe_sum`) is removed.
The fused mixers win up to 48 tokens and tie at 64; `FUSED_MAX` = 64.

### Correctness (six Macs, `ab1.sh`)

| check | result |
|---|---|
| 14-token reference prompt (one fused chunk now), against the CPU reference | the same 10 greedy tokens, logits rel 2.84e-4 ([compare-val6-new-vs-cpu.txt](compare-val6-new-vs-cpu.txt)) |
| 100-token prompt (`p100.ids`, the first 100 tokens of a document; one chunk: GEMM mixers, per-slot experts), new against old binary | the same 10 greedy tokens, logits rel 1.3e-5, residual after layer 47 rel 1.5e-5 ([compare-val100-new-vs-old.txt](compare-val100-new-vs-old.txt)) |
| the same, against the CPU reference (`qwen4 ref` on the M3 Ultra, 48 s) | the same 10 greedy tokens and top-10; logits rel 3.5e-2, cos 0.9994, for the old binary alike ([compare-val100-new-vs-cpu.txt](compare-val100-new-vs-cpu.txt)) |

Against the CPU reference the residual stays within rel 2.5e-4 through layer 23 (the f16 K/V attention layers at
~6e-4) and jumps at layers 24-27 and 40-43, the pattern of near-tied expert choices flipping (see docs/qwen4.md, Reference
validation); old and new binaries agree there to 1.5e-5.

### End to end (six Macs, real text)

`qwen4 bench --tokens-file long.ids` (a 3061-token document prompt), `--head-split 12 --context 32768 --batch 128
--ngram disk`, alternating; baseline = 0802e42 plus `bench --tokens-file` (`ab2.sh`). Driver outputs in
results/qwen4/chain-20261012/incite/ (`real*`, `turns-*`):

| run | prefill tok/s | decode tok/s |
|---|---|---|
| baseline, pp512 | 123.8, 122.6 | 13.43, 13.46 |
| new, pp512 | 131.4, 132.5, 131.9 | 13.44, 13.54, 13.53 |
| baseline, pp2048 | 197.1 | 12.41 |
| new, pp2048 | 207.9 | 12.33 |

Time to first token through `trashcan serve` (`tools/q4-turns.py`): a 22-token first turn 1.09 -> 0.76 s, a second
turn of 21 new tokens (78 reused) 1.02 -> 0.66 s, with the same replies.
