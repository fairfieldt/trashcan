# Flash-Next mixers without the separate down copy (2026-10-23, in progress)

Q8_0 mixers with write rows read only the fused matrix (bottleneck rows then write rows): the fused kernels up to 8
tokens and the split-K GEMM above. Their separate down matrix and f32 write weights were uploaded but never read. The
candidate does not upload them. The budget model (`tools/qwen4_budget.py`) puts the saving at 27.9 MiB per GPU of six
Macs (layers 0-3 with the head's twelfth at 32K and 128-token chunks: 5812 -> 5785 MiB), which would fit 256-token
chunks at 32K (5863 of the 5869 MiB budget).

No kernel changes. Main's binary `trashcan-b7f4979` (sha256 7dc00f89...), candidate `trashcan-mixc1` (4ecd139a...,
main plus the uncommitted change to engine/src/qwen4/gpu.rs).

## Plan

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
