# Flash-Next chain setup: every machine loads its stages at once (2026-10-13)

**Result.** A chain run spent ~100 s before its first token because every node loaded its stages before forwarding the
setup request downstream, and the driver loaded its own only after the whole chain had answered: six Macs one after
another. Now each node forwards the setup first, loads its stages, then waits for the machines after it; the driver
sends the setup, loads its stages, then waits. Six Macs, whole `qwen4 validate` runs (14-token reference prompt):
**114 and 118 s -> 30 and 31 s**; outputs bit-identical.

Binaries: `trashcan-4c1346d` (main) and `trashcan-setpar` (main + this change), `tools/check.sh` under `chain6.sh`
(`check.log`), alternating, `--head-split 12 --context 32768 --batch 128 --ngram disk`, page cache warm:

| run | main | setup at once |
|---|---|---|
| validate 1 (load + 14-token prompt + 10 tokens + dump) | 114 s | 30 s |
| validate 2 | 118 s | 31 s |
| bench pp512 tg64 (load + run) | | 34 s: prefill 370.5 tok/s, decode 18.34 tok/s |

Correctness: `sp-val14-setpar-1` against `sp-val14-4c1346d-1`: all 678 dumped tensors rel 0, `logits.f32`
byte-identical, the same 10 greedy tokens (the CPU reference's). Before the GPU run the protocol was checked on a CPU
chain on the M3 Ultra (driver and two CPU nodes, the 14-token prompt): rel 0 against the CPU reference.

This also shortens `trashcan serve`'s start on the chain (106 s before; not re-measured) and every six-Mac A/B session
(6-8 chain loads each).
