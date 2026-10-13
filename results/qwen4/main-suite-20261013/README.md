# Flash-Next main suite

Main's numbers, measured once per main commit. Agents compare against them instead of re-measuring main in every
session: run-to-run noise is about 1%, and main's decode rate moved <1% between sessions on 2026-10-13. The suite runs
each item once in about 6 minutes: `tools/main-suite.sh` under `chain6.sh` (`BIN=trashcan-HASH`, `DRY=1` lists the runs).

Six Macs, G-iq4, `--head-split 12 --context 32768 --batch 128 --ngram disk`. Every GPU at 850/1370 MHz while computing.

## 4307169 (small-batch experts), 2026-10-13 15:06

| item | result |
|---|---|
| validate, 14-token reference prompt | greedy tokens the CPU reference's (10/10); logits rel 2.78e-4, top-10 overlap 10/10 |
| validate, p100 (chunks of 16, then 4) | greedy tokens the CPU reference's (10/10); logits rel 3.48e-2, top-10 10/10 (same as on 2026-10-12) |
| decode, 128 tokens after a 512-token prompt | 19.06, 19.04 tok/s |
| prefill 512 tokens | 458.0, 456.4 tok/s |
| prefill 2048 (decode after it) | 671.7 tok/s (17.03 tok/s) |
| prefill 8192 | 797.6 tok/s (16.89 tok/s) |
| prefill 30000 | 842.6 tok/s (16.30 tok/s) |
| MTP, chat prompts (192 tokens, 3 drafts) | water 21.98, csv 25.89, transformer 21.47 tok/s (plain 19.27-19.44); output identical to plain |
| `trashcan serve` test with MTP | 7 checks pass; server up after 27 s |

Raw outputs: `ms-4307169/`.
