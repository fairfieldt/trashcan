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

## adf8cfd (load time: weights repacked into reused staging buffers), 2026-10-13 15:59

Measured right after invite's power cycle and the reboots of alight, enlight and indite (`results/qwen4/invite-hang-20261013`).

| item | result |
|---|---|
| validate, 14-token reference prompt | greedy tokens the CPU reference's (10/10); logits rel 2.78e-4, top-10 10/10 |
| validate, p100 | greedy tokens the CPU reference's (10/10); logits rel 3.48e-2, top-10 10/10 |
| decode, 128 tokens after a 512-token prompt | 19.58, 19.22 tok/s |
| prefill 512 tokens | 465.9, 465.4 tok/s |
| prefill 2048 (decode after it) | 677.1 tok/s (17.07 tok/s) |
| prefill 8192 | 795.8 tok/s (16.98 tok/s) |
| prefill 30000 | 850.0 tok/s (16.49 tok/s) |
| MTP, chat prompts | water 22.32, csv 26.10, transformer 21.24 tok/s (plain 19.30-19.41); output identical to plain |
| `trashcan serve` test with MTP | 7 checks pass; server up after 10 s |

The whole suite took 3.6 min (validate runs 6 s once the page cache was warm). Raw outputs: `ms-adf8cfd/`.

## e9f34e6 (verify batches, prefill split fix, long-context round), 2026-10-13 17:05

Measured after invite's reboot cleared its GPU 0 thermal DPM latch. Every GPU held 850/1370 MHz, including invite's
GPU 0 through the 30K prefill.

| item | result |
|---|---|
| validate, 14-token reference prompt | greedy tokens the CPU reference's (10/10); logits rel 2.79e-4, top-10 10/10 |
| validate, p100 | greedy tokens the CPU reference's (10/10); logits rel 3.48e-2, top-10 10/10 |
| decode, 128 tokens after a 512-token prompt | 20.67, 20.51 tok/s |
| prefill 512 tokens | 520.5, 519.5 tok/s |
| prefill 2048 (decode after it) | 887.7 tok/s (20.10 tok/s) |
| prefill 8192 | 1123.0 tok/s (19.14 tok/s) |
| prefill 30000 | 1246.9 tok/s (19.37 tok/s) |
| MTP, chat prompts | water 26.24, csv 29.94, transformer 25.71 tok/s (plain 20.19-20.48); output identical to plain |
| `trashcan serve` test with MTP | 7 checks pass |

Against adf8cfd: prefill 466 -> 520 at 512, 677 -> 888 at 2K, 796 -> 1123 at 8K, 850 -> 1247 at 30K; decode 19.4 -> 20.6;
MTP on chat 22.3/26.1/21.2 -> 26.2/29.9/25.7.

## 15e5457 morning check, 2026-10-14: lost to a power loss at 11:04:53

The first six-Mac run of the day, on the new ring order (incite, inflight, alight, indite, invite, enlight, read from
the live links by `results/qwen4/chain-20261012/tools/ring.py`), passed val14, val100 and pp512-1 (rc 0). Then all
six Mac Pros lost power at the same moment, during the start of pp512-2:
- every Mac's journal ends at 11:04:52-53 with no shutdown sequence;
- netconsole shows no kernel error from any Mac before it;
- the workstation restarted too: it booted at 11:08:48, with no clean-shutdown record in `last`.

Wall power from the SMC (`power-loss-20261014-pstr.csv`, PSTR per Mac, one sample per second):

| time | per Mac | six Macs |
|---|---|---|
| 11:03:30 (idle) | 71-82 W | ~460 W |
| 11:04:06 | 157-243 W | ~1.16 kW |
| 11:04:22-34 | 227-259 W | ~1.45-1.5 kW |

The power went about 30 s after the six reached ~1.5 kW. The 2026-10-09 strip trip went the same way: about 45 s at
~1.65 kW on one strip ([results/qwen/tp8-mesh-20261009](../../qwen/tp8-mesh-20261009)). The cause is not confirmed.

## 15e5457 (MTP rounds merged; engine of main d5ad870a), 2026-10-14 11:17, rerun after the restart

New ring order (incite, inflight, alight, indite, invite, enlight). Every GPU held 850/1370 MHz through the 30K prefill.
Peak temperatures were 54-61 °C, except invite's GPU 0 at 74 °C.

| item | result |
|---|---|
| validate, 14-token reference prompt | greedy tokens the CPU reference's (10/10); logits rel 2.79e-4, top-10 10/10 |
| validate, p100 | greedy tokens the CPU reference's (10/10); logits rel 3.48e-2, top-10 10/10 |
| decode, 128 tokens after a 512-token prompt | 21.09, 21.20 tok/s |
| prefill 512 tokens | 522.9, 523.3 tok/s |
| prefill 2048 (decode after it) | 889.8 tok/s (20.56 tok/s) |
| prefill 8192 | 1118.0 tok/s (19.68 tok/s) |
| prefill 30000 | 1249.3 tok/s (19.80 tok/s) |
| MTP, chat prompts | water 28.92, csv 31.64, transformer 29.27 tok/s (plain 20.37-20.72); output identical to plain |
| `trashcan serve` test with MTP | 7 checks pass |

Against e9f34e6: decode 20.6 -> 21.1 (host path), MTP on chat 26.2/29.9/25.7 -> 28.9/31.6/29.3 (host path and MTP rounds),
prefill unchanged.

## 4f36bf0 (all of 2026-10-14's merges, links fixed), 18:20, on 7.3.0-rc7-p2p-s2: partial

Today's main: the USB4STREAM transport, decode kernels round 2, the per-GPU clock fix, and the link fixes. All six Macs
run 7.3.0-rc7-p2p-s2 (0006 + 0007). Each run's `links:` line showed every link on its stream, up in 4-6 ms.

| item | result |
|---|---|
| validate, 14-token reference prompt | greedy tokens the CPU reference's (10/10); logits rel 2.79e-4, top-10 10/10 |
| validate, p100 | greedy tokens the CPU reference's (10/10); logits rel 3.48e-2, top-10 10/10 |
| decode, 128 tokens after a 512-token prompt | **22.52, 22.54 tok/s** (15e5457 on 7.2.9 this morning: 21.09, 21.20) |
| prefill 512 tokens | 522.3, 521.4 tok/s |

invite went down as a whole at 18:25:10, during pp2048: no kernel message, both neighbours logged its cable
disconnecting, and it dropped off the tailnet. The suite was stopped. pp2048, pp8192, pp30000, MTP and the server test
did not run.
