# Qwen rollback, MTP speculative decoding and prefix reuse, 2026-10-06

Tested source: commit `2dd5f05` (engine unchanged since; later commits add results and docs). The final
runs (`final/`, `reuse-edit/`) used the incite binary with sha256
`2998e3f6e1061ca5e1b77cc32dc799c2a9be80d57286856dcff8687fb808fd5d`, cross-built with
`scripts/cross-build.sh trashcan-exp/qwen-dense/bin/trashcan`. Earlier runs (`sweep-*`, `reuse-9b*`,
`gpu-9b-check-k3.json`, `hybrid-27b-check.json`) used intermediate builds of the same series; their
numbers are reproduced by the final runs where they overlap.

incite, kernel 7.2.9-p2p, both D700s sustained 850/1370 MHz during compute (`clocks/`, sampled by
`~/trashcan-exp/locked-run.sh`; the samples also cover loading and idle time). Timed commands hold
`~/trashcan-exp/.lock` and the scoped 0-us CPU latency request (`scripts/latency-run.py`). Models:
`Qwen3.5-9B-trashcan-Q4_0.gguf` with `mtp-Qwen3.5-9B-trashcan-Q4_0.gguf`, and
`Qwen3.8-27B-trashcan-Q4_0.gguf` with `mtp-Qwen3.8-27B-trashcan-Q4_0.gguf` (docs/qwen.md has the recipes).

## Correctness

| run | file | result |
|---|---|---|
| `qwen-check`, 9B GPUs, 600-token prompt, 256 steps, k=2, 64K draft head | `final/check-256.json` | pass: restore bitwise exact; 170 verification batches with rollbacks match serial decode (top score error 2.3e-4); speculative output identical over 256 tokens |
| same, k=3, 64 steps (full head) | `gpu-9b-check-k3.json` | pass |
| `qwen-check`, 27B hybrid chain (CPU 0-30, GPU 30-48, 48-64 + head + MTP), k=2 | `hybrid-27b-check.json` | pass: score error 3.5e-4, identical, 70% acceptance |
| `qwen-check`, 9B CPU path, one process and a two-process TCP chain | `cpu-9b-check.json` | pass, verification bitwise exact |
| GPU vs CPU MTP layer: three chained drafts after the reference prompt | `val-9b-*-mtp/mtp_drafts.txt` | same top tokens (the target's greedy continuation), top-10 overlap 10/10, score difference at most 0.005 |
| acceptance vs llama.cpp `--spec-type draft-mtp --spec-draft-n-max 3`, same 48-token prompt, 200 tokens | `final/check-llama-k3.json`, `llama-mtp-ref-200.json` | 126/219 (57.5%) vs 126/214 (58.9%) drafts accepted, same opening text |
| depth sweeps: speculative greedy output vs plain decoding | `sweep-*/`, `final/greedy-*/` | byte-identical for every depth and draft head |
| `scripts/test-server.py` against `trashcan serve`, 9B | `test-server.txt` | pass on the CPU path and on the GPUs with MTP |

## Rates

Greedy, the three Gemma MTP prompts, 128 tokens, thinking off, after a warm-up, in both orders
(`final/greedy-a`, `final/greedy-b`; tok/s):

| | transformer | photosynthesis | CSV function | pooled |
|---|---|---|---|---|
| plain decode | 22.93 / 23.09 | 23.08 / 23.17 | 23.12 / 23.11 | 23.04 / 23.12 |
| MTP, two drafts, 64K draft head | 34.07 / 34.21 | 30.29 / 30.41 | 41.21 / 41.23 | 34.63 / 34.73 |

+50% pooled. Per round: 59.5 ms verification, 7.9-8.0 ms drafting, 0.4 ms rollback, 2.37 tokens
(first draft accepted 77%, both 60%). Depth and draft-head sweeps (`sweep-9b-1`, `sweep-9b-v64k`,
`sweep-9b-v32k`) are tabulated in docs/qwen.md: two drafts beat one and three for every head, and the 64K
head beats both the full head and the 32K head.

Temperature 0.7, top-k 20, top-p 0.8 (`final/t07`; outputs differ by design):

| k | transformer | photosynthesis | CSV function | pooled | acceptance |
|---|---|---|---|---|---|
| 0 | 23.06 | 23.24 | 23.21 | 23.17 | |
| 1 | 33.13 | 32.05 | 35.14 | 33.39 | 84% |
| 2 | 34.90 | 27.08 | 38.57 | 32.78 | 74% first, 51% both |
| 3 | 32.90 | 27.11 | 33.06 | 30.76 | |

Building blocks (`final/spec-bench.json`, 600-token prompt, ms per call): decode 45.2; verification of
2/3/4 tokens 52.4/61.1/75.9; rollback 1.1-1.3; draft steps (64K head) 3.96 each; checkpoint save 24.0 and
restore 6.0. Prefill pp512 (`final/bench-*.json`): 126.65 tok/s plain, 121.55 with the MTP layer
(-4%); tg64 23.07 plain, 35.28 with MTP.

## Prefix reuse

`reuse-9b` and `reuse-9b-b` (`scripts/qwen-reuse.sh`, both orders): a 2.7K-token document turn and three
short questions; with reuse the follow-ups prefill 84/56/92 tokens instead of ~2.8K, text identical, decode
rates equal within 0.2 tok/s once the first-run warm-up is excluded. `reuse-edit`: a 5866-token prompt
edited near its end restores the 4096-position interval checkpoint (1770 tokens prefilled), text identical.
`think-27b-cpu`: Qwen3.8-27B on the Mac's CPU path with thinking on reuses the whole first turn (100 of 118
tokens).

## Reproduction

```sh
scripts/cross-build.sh trashcan-exp/qwen-dense/bin/trashcan
rsync -a scripts/qwen-*.sh results/qwen/spec-20261006/run-*.sh incite:trashcan-exp/qwen-dense/scripts/
ssh incite '~/trashcan-exp/locked-run.sh qwen-dense scripts/run-final.sh'
ssh incite '~/trashcan-exp/locked-run.sh qwen-dense scripts/run-checks.sh'
ssh incite '~/trashcan-exp/locked-run.sh qwen-dense scripts/qwen-hybrid.sh 30,48 qwen-check --draft-tokens 2 --tokens 48 --steps 24 --context 512'
```

The first process after the GPUs idled for a few minutes decoded about 10% slower in one run, so
measurements start with a warm-up and repeat in both orders.
