# Flash-Next prefill chunk size by prompt length (2026-10-12)

**Result.** With a Mac's two GPUs working on consecutive chunks (90cb371), a prompt takes about (chunks + 11) chunk
times through the 12 stages, so short prompts want small chunks. `Chain::prefill` (and the server's prompts) picks the
chunk size from the prompt length. First rule (88def0d): about sqrt(16 n / (stages - 1)), a power of two from 16 to
`--batch`; six Macs, real text: 512 tokens 159 -> 261 tok/s, 2048 tokens 322 -> 364, 3000 tokens 363 -> 379. After the
expert, mixer and decode kernels (ef4ce87) the optimum moved; the rule is now measured bands in tokens per stage after
the first (< 12: 16, < 70: 32, < 450: 64, else 128): 256 tokens 236 -> 260 tok/s, 1024 tokens 452 -> 512 against
ef4ce87's rule (see [Re-fit](#re-fit-after-the-kernel-work-ef4ce87)).

## Sweep: fixed chunk sizes on main (90cb371)

`tools/chunks.sh` (`sweep/`, `sweep.log`): `qwen4 bench --tokens-file long.ids --head-split 12 --context 32768
--ngram disk` with `--batch` 128, 64 or 32 (the chunk size), alternating.

| prompt | 128 | 64 | 32 |
|---|---|---|---|
| 512 | 156.8, 158.6 tok/s | 216.1, 218.2 | 260.3, 260.3 |
| 2048 | 323.1 | 364.3 | 361.7 |

Model: a prompt of n tokens in chunks of c takes about (n / c + 11) t(c), with t(c) the stage time of a chunk (the
slower GPU of a Mac, 4 layers): 45, 69, 118 and 208 ms at 16, 32, 64 and 128 tokens (stage-bench,
results/qwen4/prefill-batch-20261012), close to t(c) = 23 ms + 1.45 ms x c. It predicts the six measurements within
~10% (2048 tokens 5-10% optimistic) and has its minimum at c = sqrt(a n / (b (S - 1))) = sqrt(16 n / 11): 16 for up
to ~300 tokens, 32 to ~1.4K, 64 to ~5.5K, 128 beyond (4K is a tie between 64 and 128).

## The rule against main

`engine/src/qwen4/net.rs` `Chain::chunk_size`; `Chain::extend` (prefill and the server's prompt suffixes) and
`qwen4 validate` use it. `tools/ab-chunks.sh` under `chain6.sh`: `trashcan-chk` against `trashcan-main` (90cb371,
fixed 128-token chunks), `--batch 128`, alternating (`incite/ac-*`, node logs and clocks in the host directories):

| prompt (chunk chosen) | main | adaptive |
|---|---|---|
| 512 (32) | 159.2, 158.7 tok/s | 261.4, 260.7 tok/s (+64%) |
| 2048 (64) | 322.2, 322.1 | 364.4, 364.3 (+13%) |
| 3000 (64) | 362.9 | 378.8 (+4%) |
| 8192 (128, the same as main) | 447.4 | did not start (the enlight node was not listening yet; no difference expected) |

Decode after the prefill: 13.3-14.7 tok/s either way. Every GPU held 850/1370 MHz while working.

## Correctness

- CPU chain on this Mac (driver and two CPU nodes, `--batch 64`: the 100-token prompt in chunks of 32, 32, 32 and 4)
  against the single-shot CPU reference of the same prompt: every tensor rel 0, the same 10 greedy tokens
  ([compare-cpuchain-val100-vs-cpu.txt](compare-cpuchain-val100-vs-cpu.txt)).
- Six Macs with the rule (88def0d, `tools/val-chunks.sh`): the 100-token prompt in 16-token chunks against the CPU
  reference: the same 10 greedy tokens and top-10, logits rel 3.48e-2 as with one chunk
  ([compare-gpu-val100-chunks16-vs-cpu.txt](compare-gpu-val100-chunks16-vs-cpu.txt)); against the one-chunk GPU run:
  logits rel 1.3e-5, residual after layer 47 rel 1.5e-5
  ([compare-gpu-val100-chunks16-vs-onechunk.txt](compare-gpu-val100-chunks16-vs-onechunk.txt)); the 14-token reference
  prompt: logits rel 2.84e-4, the same tokens ([compare-gpu-val14-vs-cpu.txt](compare-gpu-val14-vs-cpu.txt)). (An
  earlier six-Mac validation ran before `qwen4 validate` used the rule and repeated the one-chunk result.)
- Multi-chunk GPU runs at other chunk sizes were bit-identical to main in results/qwen4/prefill-overlap-20261012 (64
  and 128); the fused mixers and per-slot experts at 16-64 tokens were validated in results/qwen4/prefill-batch-20261012.

## Re-fit after the kernel work (ef4ce87)

Stage times of b5b79c7 (inflight, layers 0-7, `tools/refit.sh`, `refit/refit-b*.json`, the slower GPU of the Mac per
chunk): 40.8, 51.1, 68.7 and 117.1 ms at 16, 32, 64 and 128 tokens (before: 45, 69, 118, 208), t(c) = 28.6 + 0.681 c:
a chunk now costs about as much as 42 tokens. The sqrt rule with 42 (`trashcan-chk42`, `tools/ab-chunks42.sh`,
`refit/a42-*`) against ef4ce87's rule with 16, six Macs, real text, alternating:

| prompt (chunk: 16-rule -> 42-rule) | ef4ce87 | 42-rule |
|---|---|---|
| 1024 (32 -> 64) | 454.1, 454.2 tok/s | 506.8, 513.0 tok/s (+12%) |
| 4096 (64 -> 128) | 649.3, 649.9 | 633.7, 627.7 (-3%) |

On the chain 128-token chunks did worse than the stage times predict, so 4096 tokens prefer 64. At 8192 tokens
(`tools/ab-cap64.sh`, `refit/cap-*`, `--batch` 64 against 128): 64-token chunks 676.9, 676.0 tok/s, 128-token chunks
705.8, 707.4 (+4.4%). No single sqrt constant puts both boundaries (32/64 below 1024, 64/128 between 4096 and 8192)
where they were measured, so the rule became bands in tokens per stage after the first. Against ef4ce87's rule where the
bands choose differently (`trashcan-bands`, `tools/ab-bands.sh`, `bands/bd-*`):

| prompt (chunk: ef4ce87 -> bands) | ef4ce87 | bands |
|---|---|---|
| 256 (16 -> 32) | 237.4, 236.1 tok/s | 257.9, 261.4 tok/s (+10%) |
| 1024 (32 -> 64) | 453.3, 452.0 | 512.4, 512.1 (+13%) |
| 6144 (128 -> 64 with the 560 bound tested) | 677.6 | 667.8 (-1.4%) |

so the 64/128 bound moved from 560 to 450 tokens per stage (4950 tokens on six Macs): 6144 and 8192 tokens keep 128 as
ef4ce87 does, 4096 keeps 64. Decode unchanged (16-18 tok/s after these prompts). The 100-token prompt (16-token
chunks) with the bands: the CPU reference's 10 greedy tokens and top-10, logits rel 3.48e-2 as before
([compare-gpu-val100-bands-vs-cpu.txt](compare-gpu-val100-bands-vs-cpu.txt);
[compare-gpu-val100-chk42-vs-cpu.txt](compare-gpu-val100-chk42-vs-cpu.txt) for the 42-rule).
