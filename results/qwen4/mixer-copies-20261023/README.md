# Flash-Next mixers without the separate down copy (2026-10-23; validated and merged 2026-10-26)

Q8_0 mixers with write rows read only the fused matrix (bottleneck rows then write rows): the fused kernels up to 8
tokens and the split-K GEMM above. Their separate down matrix and f32 write weights were uploaded but never read. The
candidate does not upload them. The budget model (`tools/qwen4_budget.py`) puts the saving at 27.9 MiB per GPU of six
Macs (layers 0-3 with the head's twelfth at 32K and 128-token chunks: 5812 -> 5785 MiB), which would fit 256-token
chunks at 32K (5863 of the 5869 MiB budget).

No kernel changes. Main's binary `trashcan-b7f4979` (sha256 7dc00f89...), candidate `trashcan-mixc1` (4ecd139a...,
main plus the uncommitted change to engine/src/qwen4/gpu.rs).

## Result (2026-10-26): pass, merged to main as 371a1dbc

Rerun in `run-20261026/` with rebuilt binaries (the 2026-10-23 binaries were gone). Base: main 91c3b82a
(`trashcan-91c3b82`, sha256 396909e9104b9adaf3ece93095eb15a92f9fdb8aedeba1a9189818b3c672e42e). Candidate:
91c3b82a plus `run-20261026/candidate-on-91c3b82.patch` (`trashcan-mixc2`,
cdce621f569deeaed51837bdff6c7e01d28310f3e6901622ad5f75c9990ac407); gpu.rs applied unchanged, qwen4_budget.py
needed a 3-way merge of a comment main had reworded. Both built with `run-20261026/tools/cross-build.sh` (lab's
script taking `SRC=`, since main no longer has it) in kbuild: Debian 13.7, rustc/cargo 1.99.0, x86_64-linux-gnu-gcc
14.2.0, glibc 2.41, shaderc 2025.2; no warnings. All six Macs on 7.3.0-rc7-p2p-s2, portlights-node.service active on
all of them for both arms, ring incite -> inflight -> alight -> indite -> invite -> enlight, all hops on the stream.

- **Allocation.** One D700 (incite GPU0, layers 0-3 + head rows 0..21504, 32K): 5802.3 -> 5774.5 MiB at 128-token
  chunks, 5868.3 -> 5840.5 at 256. Six-Mac chain: 27.8 MiB less on every one of the 12 GPUs
  (`chain/alloc-summary.txt`; model 27.9).
- **Batch 256.** Main also allocates 256-token chunks at 32K (5868.3 MiB on incite GPU0 against a reported budget of
  5868, val100 tokens and dump identical), contrary to the expected failure; the candidate leaves 24.5 MiB there.
  The candidate at 256 against 128 (one pair each): pp2048 886.7 vs 882.0 tok/s, pp8192 1130.5 vs 1124.6 (+0.5%,
  within noise). No end-to-end benefit: keep `--batch 128`.
- **Correctness.** `tools/cmpdump.py` exits 0 on all five stage dumps (b1, b4, p16, p128, l2101: 49/49/49/57/49
  tensors identical) and on the chain validation dumps (val14 690/690, val100 595/595, and val100 at batch 256,
  candidate and base, against base at 128). Greedy tokens identical in all six validations.
- **Speed.** One D700, layers 4-7, alternated pair: decode layers 3.290/3.291 ms (base) vs 3.298/3.298 ms, 128-token
  chunk 90.91/90.77 vs 90.91/90.86 ms. Six Macs, pp512 tg128: base 521.3 tok/s prefill, 22.22 tok/s decode;
  candidate 522.1, 22.63 (one pair: unchanged within noise).
- Every command exited 0 (`mc2-s1.log`, `mc2-chain.log`); 0 amdgpu reset/timeout lines in dmesg on all six Macs
  after the runs; sclk 850 MHz sustained (300 MHz only in edge samples). Stage dumps (`*.f32`, 98 MB) are not
  committed; they are in ~/dev/claude-work/trashcan-release-20261026/a1/ on the driving Mac.

## Plan (2026-10-23)

```
tools/deploy.sh
# one Mac: allocation at 128/256-token chunks, dumps across all mixer paths, one timing pair
gpuq --on incite request mc-s1 --est 15 ; ../decode-kernels-20261014/tools/lease-run.sh incite ID mc-s1.sh "mc-s1"
python3 tools/cmpdump.py DUMP_CAND DUMP_BASE 3    # each of b1 b4 p16 p128 l2101
# six Macs: validation, one decode pair, 256-token chunks
DRY=1 tools/mc-chain.sh
~/dev/claude-work/flashnext-20261012/chain6.sh mc1 25 OWNER -- tools/mc-chain.sh
```

## Readiness snapshot (2026-10-25)

Still unvalidated: no GPU runs were found in this directory. All six SSH probes timed out; no deployment
or current kernel state was confirmed. The previously recorded local binary directory was absent on the
inspecting machine. Recover or rebuild the two binaries and record complete hashes before running.

`candidate.patch` preserves the two uncommitted source changes against `fc22c1fb`; `snapshot.json` records
source and patch hashes. Applying it to that base was checked without changing the runtime. This archive
does not merge or approve the candidate. If main's cleanup changes relevant code, rebase and revalidate it.

`tools/cmpdump.py` now returns nonzero for nonidentical, missing or truncated tensors and an empty reference
index. Reproduce the synthetic checks with `python3 results/release-readiness-20261025/verify-cmpdump.py`.
`mc-s1.sh` reports per-command errors without necessarily exiting unsuccessfully: inspect all statuses,
including the expected baseline batch256 allocation failure. See `docs/release-readiness.md` for A1's gates.
