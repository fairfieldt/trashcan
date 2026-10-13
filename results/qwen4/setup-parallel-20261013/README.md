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

## Runner and connect retry (later the same day)

`tools/q4run.py` now starts the driver as soon as every node's port answers, instead of after a fixed 3 s. It logs the
command inside the driver's ssh and stops all nodes at once (TERM, then KILL, with one shared deadline). `Conn::connect`
retries a refused connection every 100 ms for up to 60 s, so a machine that starts before the next one listens waits
instead of failing. One MTP run that morning failed with `Connection refused` to `indite.tb0:7071`.

Six Macs, `tools/runner-check.sh` under `chain6.sh`. `trashcan-retry` is main 6f6635a plus the retry:

| run | nodes up | driver | nodes stopped | whole run |
|---|---|---|---|---|
| validate, main 6f6635a | 2.1 s | 20 s | 0.6 s | 23 s |
| validate, 6f6635a + retry | 0.7 s | 22 s | 0.5 s | 23 s |

The same validate runs took 30 and 31 s with the previous runner. `logits.f32` is byte-identical between the two
binaries, and the greedy tokens are the CPU reference's. `loadprof-indite.txt` is the per-GPU load profile (warm 20 s
per Mac, cold 43 s) that the load-time work started from.
