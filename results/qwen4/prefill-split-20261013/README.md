# Flash-Next prefill: the split out projections run in the chain (2026-10-13)

**Result.** Q8_0 out projections of 16-64-token prefill chunks now use the K-split GEMM in chain runs. Six Macs: prefill
**465 -> 487 tok/s at 512 tokens, 589 -> 622 at 1024, 678 -> 710 at 2048** (+4.7-5.6%). Greedy tokens are still the
CPU reference's.

**Bug.** `out_proj` took the split path only when `splits * d <= w`, as if each token's partial sums had its own
`w`-float row of `hgate`. The GEMMs and `qwen4_sum_parts` actually lay them out `[part][token][row]` for the batch's
tokens, which needs `splits * n * d <= w * max_batch`. With the chain's `--batch 128`, chunks of 16-64 tokens chose
8 splits and then failed the per-token test (8 * 2560 > 10240), so they fell back to the unsplit GEMM. `stage-bench
--batch n` sets `max_batch = n`, chose 4 splits and passed. That is why the prefill-kernels round measured split
projections the chain never ran (found by the load-time agent, results/qwen4/load-time-20261013). Now the split count
is the first of 8 and 4 whose K split and partial sums fit.

Six Macs, `tools/session.sh` under `chain6.sh` (after main's suite on adf8cfd), `--head-split 12 --context 32768
--batch 128 --ngram disk`, one run each, `long.ids`. Binaries: `trashcan-adf8cfd` (main), `trashcan-split8` (this change)
and `trashcan-split4` (the same with `OUT_SPLITS = 4`):

| prompt | main | split, 8 parts when they fit | split, 4 parts |
|---|---|---|---|
| 512 | 465.2 tok/s | 487.1 | 244.9 (one-off*) |
| 1024 | 588.8 | 621.8 | 620.2 |
| 2048 | 677.9 | 709.8 | 710.8 |

\* The driver's own stages took the same time as with 8 parts (38.2 / 35.0 ms for the last chunk). The extra second
was elsewhere on the first run of a fresh binary on Macs rebooted minutes before. 4 parts matched 8 at 1024 and 2048.

Correctness (split8 against the CPU reference):
- 14-token prompt: logits rel 2.77e-4 (main 2.78e-4), top-10 10/10, greedy 10/10.
- p100 (chunks of 16 and 4): rel 3.48e-2 (main the same), greedy 10/10.
