# Flash-Next prefill chunk size by prompt length (2026-10-12)

**Result.** With a Mac's two GPUs working on consecutive chunks (90cb371), a prompt takes about (chunks + 11) chunk
times through the 12 stages, so short prompts want small chunks. `Chain::prefill` (and the server's prompts) now picks
the chunk size from the prompt length: about sqrt(16 n / (stages - 1)), the nearest power of two from 16 to `--batch`.
Six Macs, real text: 512 tokens 159 -> 261 tok/s, 2048 tokens 322 -> 364, 3000 tokens 363 -> 379; 8192 tokens keep
128-token chunks (447 tok/s on main).

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
- The first six-Mac validation ran before `qwen4 validate` used the rule (it chunked by `--batch`), so it repeated the
  one-chunk result; the GPU check with the rule is queued.
- Multi-chunk GPU runs at other chunk sizes were bit-identical to main in results/qwen4/prefill-overlap-20261012 (64
  and 128); the fused mixers and per-slot experts at 16-64 tokens were validated in results/qwen4/prefill-batch-20261012.
