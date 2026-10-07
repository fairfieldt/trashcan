# qwen4exp GPU stages on incite (2026-10-06)

incite: Mac Pro 6,1, kernel 7.2.9-p2p, both D700s at 850/1370 MHz in every run (`*.clocks.txt`). Binaries from
`scripts/cross-build.sh`, runs through `~/trashcan-exp/locked-run.sh qwen-flash`. GGUFs are described in
[../quant-20261006/README.md](../quant-20261006/README.md): G-q4_0 (all matrices Q4_0), G-q8d (Q8_0 dense, Q4_0
experts), G-iq4 (Q8_0 dense, IQ4_XS/IQ4_NL experts: the six-Mac recipe). Source: the commit that adds each file.

## Validation

GPU layers 0-5 on incite, layers 6-47 and the head on a CPU node on the M3 Ultra, compared with the CPU reference
on the same GGUF (14-token chat prompt, 10 greedy tokens):

```sh
# Mac
trashcan qwen4 ref MODEL ref --greedy 10
trashcan qwen4 node MODEL --listen 0.0.0.0:7071 --cpu
# incite
./bin/trashcan qwen4 validate MODEL results/val --gpus 0,1 --cuts 3,6 --nodes 192.168.0.212:7071 \
    --context 64 --batch 4 --greedy 10
# Mac
python3 tools/compare_activations.py val ref --all
```

| file | GGUF | result |
|---|---|---|
| `validate-gpu-layers-0-6.txt` | G-q4_0, unfused mixers | same tokens, logits rel 4.3e-5 |
| `validate-gpu-layers-0-6-fused.txt` | G-q4_0, fused mixers | same tokens |
| `validate-gq8d-layers-0-6.txt` | G-q8d | same tokens, logits rel 2.5e-5; the rebased build reproduced it exactly |
| `validate-giq4-layers-0-6.txt` | G-iq4 | same tokens, logits rel 1.7e-5, MoE outputs rel 5e-7 to 1.1e-6 |
| `validate-giq4-headsplit3.txt` | G-iq4, `--head-split 3` (head rows on GPU0, GPU1 and the Mac node) | same tokens, logits rel 1.7e-5 (3.4e-7 from the unsplit run); two CPU processes with `--head-split 2` match the reference exactly |
| `validate-long-sparse-score.txt` | G-q4_0, 3061-token prompt in 256-token chunks (`stage-bench --tokens-file --out`, context 4096) | QSA output rel 6.6e-4; `qwen4_idx_score` bit-identical to the previous kernels |

`Qcur-3` shows rel 0.94 with cos 1.000000: the GPU captures the query after the 1/16 attention scale.

## Timing

`qwen4 stage-bench` runs stages without a chain and reports GPU milliseconds per layer and phase (decode steps
after `--start` tokens, prefill chunks of `--batch` from position 0, and the last chunk of the `--start` prefill).

| file | command (after `qwen4 stage-bench MODEL`) | notes |
|---|---|---|
| `stagebench-gq4-0-6.json` | `--ranges 0:3,3:6 --prefill 2`, G-q4_0, unfused mixers | 9.05 ms per decode step (six layers) |
| `stagebench-gq4-0-6-fused.json` | same, fused mixers | 6.65 ms |
| `stagebench-grouped-cols{2,8}.json` | same, 2 or 8 slots per grouped expert pass | slower than 4 (54, 81 ms) |
| `stagebench-gq8d-0-6.json` | same, G-q8d | 8.29 ms |
| `stagebench-giq4.json` | same, G-iq4 | 8.62 ms; experts 0.29 ms / 57.7 ms per chunk |
| `stagebench-gq8d-2-4-32k-before.json` | `--ranges 2:4 --gpus 0 --context 32768 --start 32512 --decode 16 --prefill 1`, scores in `qwen4_idx_select` | QSA layer 13.25 ms per decode step |
| `stagebench-gq8d-2-4-32k.json` | same with `qwen4_idx_score` | 2.16 ms (selection 0.62, attention 0.56) |
| `stagebench-giq4-0-4-32k.json` | `--ranges 0:4 --gpus 0 --context 32768 --start 32512 --decode 16 --prefill 1`, G-iq4 | the first GPU of six Macs: 5843 of 5888 MiB, runs |
| `stagebench-giq4-0-4-32k-b128-head12.json` | `--ranges 0:4 --gpus 0 --context 32768 --batch 128 --head-rows 0:21504 --start 32640 --decode 16 --prefill 1` | the first GPU of six Macs with its twelfth of the head: 5830 MiB, decode 6.27 ms + 1.0 ms head slice at 32K |
| `stagebench-giq4-46-48-head.json` | `--ranges 46:48 --head --gpus 1 --prefill 2`, G-iq4 | head 4.6 ms per token |
| `stagebench-giq4-44-48-head.json` | `--ranges 44:48 --head --gpus 1 --prefill 2`, G-iq4 | 6480 MiB on a 5888 MiB heap: RADV put the excess in system memory and the head took 84 ms. Stages now refuse to load past the budget |

`head-slice-bench.log` (`head-slice-bench.sh`): wall time to score one hidden state with 20,736, 82,944, 124,160
and 248,320 head rows on one D700 (`--head-rows`): 1.12, 2.03, 2.62 and 4.60 ms; 1,024 rows take 0.79 ms, so about
0.75 ms of every slice is fixed (the single-workgroup `topk`/`merge` passes and the submission round trip).

## VRAM

`python3 tools/qwen4_budget.py --check` reproduces every `GPUn: ... MiB allocated` line above within 0.2 MiB.
