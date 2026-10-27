# Flash-Next default context: 114,688 positions with a Q8 K/V cache (2026-10-27)

**Result.** The six-Mac recipe (`6-flashnext`) goes from `--context 32768` (f16 K/V) to `--context 114688 --kv q8`, the
most a Q8 cache leaves room for on the fullest GPU. Against the old recipe, same binary: the same greedy tokens on the
14-token and p100 prompts, the same prefill, decode within 0.5%, MTP about 2% slower (single runs), output identical
to plain decoding. The fullest GPU (incite GPU 0: layers 0-3 and head rows 0..21504) allocates 5850 of 5888 MiB
(5774.5 before). The launcher's smoke of the edited recipe with MTP passes. A 100,000-token prompt did not finish: all
six Macs lost power about 2.5 minutes into its prefill.

## Tested source

- Binary `trashcan-v5` (sha256 7fb133ad...), the same on all six: main 2dd1294c's engine, which is origin/main 39e8e767's
  (no engine commit since). Built for results/deploy/link-fallback-20261026.
- Recipe and docs: main after 39e8e767 (`deploy/recipes.toml`, `docs/deploy.md`, `docs/qwen4.md`).
- Six Macs on the ring incite-inflight-alight-indite-invite-enlight, recabled that morning (incite-inflight on bus 2 at
  both ends; `tools/cluster.toml`); every link on its stream in every run. Fans at 2350 rpm throughout (the old
  fancurve policy; the new one was installed after these runs).

## Commands

`tools/ab.sh` under `~/dev/claude-work/flashnext-20261012/chain6.sh` (leases on all six), runs through
`results/qwen4/chain-20261012/tools/q4run.py`; A and B differ only in `--context 32768` against
`--context 114688 --kv q8` (both `--head-split 12 --batch 128 --ngram disk`). `tools/mtpsum.py` summarizes the MTP runs.
Log: `ab.log`. Activation dumps of the validation runs were left out (`runs/*-val*/` keep `greedy.txt`).

## Results

| run | A: 32K, f16 | B: 114,688, Q8 |
|---|---|---|
| greedy, 14-token prompt | 9419 0 2500 628 353 1438 488 3242 30 248046 | identical (and as main suite ms-e9f34e6) |
| greedy, p100 | 248046 198 248045 248045 248046 198 248045 74455 198 248068 | identical |
| decode after 512 (ABBA) | 22.42, 22.36 tok/s | 22.20, 22.35 tok/s (−0.5%) |
| prefill 512 | 514.9, 516.9 tok/s | 514.8, 515.8 tok/s |
| prefill 2048, decode after | 882.4, 21.54 tok/s | 884.6, 21.60 tok/s |
| MTP water / csv / transformer | 29.70 / 33.11 / 30.57 tok/s | 29.49 / 32.05 / 29.68 tok/s (−2.3% mean) |
| MTP plain decode, same runs | 21.71 / 21.47 / 22.00 | 21.86 / 21.96 / 21.53 |
| MTP output identical to plain | yes | yes |
| fullest GPU, allocated | 5774.5 MiB | 5850.0 MiB (others 5816.2, last 5793.0) |

The QSA layer's decode step on incite GPU 0 goes from 0.864 to 0.908 ms (Q8 attention), about 0.5 ms per token over
the 12 QSA layers. The MTP difference is from one run per prompt; acceptance per draft moved slightly
(csv 0.953 -> 0.946, transformer 0.901 -> 0.908), tokens per round the same.

**Clocks.** Every GPU at 850/1370 MHz while working (single 450 MHz samples at DPM transitions), at most 63 C
(`runs/clocks-*`, `[clocks]` lines in the logs).

## 100,000-token prompt

`b-pp100k` (`bench --pp 100000 --tg 32 --tokens-file long-100k.ids`, B) loaded (the allocations above) and was in its
prefill when all six Macs lost power together, about 151 s after the start: uptime reset on every Mac, the tailnet
showed them offline at the same moment. The earlier 100K run (results/qwen4/long-context-20261013) finished with
invite's GPU 0 stuck at 450 MHz the whole time; this was the first long load with all 12 GPUs at 850 MHz. The launcher
smoke that followed in `ab.sh` did not run. The cause of the power loss is not established.

## Launcher smoke

After the Macs were back (a Thunderbolt reload on all six, twice, brought every link to two lanes and its stream):
`tools/smoke.sh` under chain6.sh runs `deploy/trashcan-cluster smoke 6-flashnext --mtp --nodes
incite,inflight,alight,indite,invite,enlight` with the edited recipe: ready after 56 s, the greedy answer contains
"Paris" (1.83 s), every process stopped and the DPM levels restored (`runs/smoke-mtp.log`). The new fan control
(results/hardware/fan-curve-20261027) was running; every Mac's fan went to full speed during the run. Two attempts
before the Thunderbolt reload failed at the start ("no Thunderbolt link up between ..."), as expected with the links
down after the power loss.
