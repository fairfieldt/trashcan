# Qwen3.8-27B TP4: the f16 all-reduce payload on two Macs (2026-10-09)

**Result.** On TP4 (incite + invite, one Thunderbolt cable), the f16 payload that ddc8df8 made the default below
prefill size is faster than exact f32 sums in every alternated pair: decode 28.48 against 28.17 tok/s (+1.1%, 0.42 ms
less exchange per token) and MTP 39.17 against 38.38 (+2.1%, verification 59.3 against 60.6 ms per round). Prefill
is the same (its all-reduces are f32 either way), every run generated the same text, and the gates pass: the CPU
reference's greedy tokens, and `qwen-check --mtp` and `--verify 3` with column score error 0. The default therefore
stays for two-machine groups as for four; the engine is unchanged. Back-to-back MTP runs heat invite's first D700 past
90 °C, where it throttles, so the MTP comparison was repeated with the cards cooled to 55 °C before each run (Heat,
below).

## Setup

- **Source.** a25cdd0 (`main`), whose engine is f3ac672's: f16 below prefill size by default in every multi-machine
  group, full-mesh all-reduces only for four fully cabled machines. One binary, cross-built with
  `scripts/cross-build.sh` into `~/trashcan-exp/tp4wire/trashcan` on incite and invite; sha256 `d13b69ff…2ab91`,
  byte-identical to the `trashcan-final` of results/qwen/tp8-mesh-20261009. f32 runs set `TRASHCAN_TP_WIRE=f32`; the
  driver reads it and passes the format to the node at setup, and each driver log names the payload of its session
  (`tp: 4 ranks over 2 machine(s), GPUs per machine [2, 2], f16 all-reduce payload`).
- **Machines.** incite (driver, machine 0) and invite (node, machine 1), kernel 7.2.9-p2p, over the one incite-invite
  Thunderbolt 2 cable (cable 1 of results/qwen/tp8-mesh-20261009, two lanes): its `tp8` USB4STREAM, `/dev/tbstream0`
  on both Macs (`scripts/tbstream-up.sh incite invite tp8` confirmed it). The other five cables of the TP8 cluster
  stayed connected and idle, as did enlight and indite.
- **Method.** gpuq leases on both Macs (incite #132, invite #120), the performance governor on both for the session
  (schedutil restored afterwards), every process under `scripts/with-clocks.sh` and `scripts/latency-run.py` (0 µs),
  the node included (`tools/tp4.sh`, which runs `../tp8-opt-20261008/tools/tprun.py`). One warm-up run (`w-warm`, not
  counted), then the phases of `tools/wire.sh`: `qwen-bench --pp 512 --tg 128 --reps 2`, and the same with `--mtp`
  (k = 2, 64K draft head), three runs a side in the order f16, f32, f32, f16, f16, f32; then the gates. tok/s are per
  repetition; exchange and GPU wait are incite's milliseconds per generated token. Tables from `tools/runs.py`,
  `tools/windows.py` and `../tp8-opt-20261008/tools/pairs.py` / `mtppairs.py`. Every card ran at 850/1370 MHz while
  computing, except invite's first card in the first MTP set (Heat).

## Decode (`w-dec-*`)

| candidate | n | f32 tok/s | f16 tok/s | change | GPU wait ms/token | exchange ms/token | pp512 tok/s |
|---|---|---|---|---|---|---|---|
| f16 payload (default) against f32 | 6+6 | 28.17 (27.99-28.27) | 28.48 (28.43-28.51) | +1.1% | 29.05 → 29.06 | 4.86 → 4.44 | 147 → 147 |

| run | payload | pp512 tok/s | tok/s | exchange ms/token | GPU wait ms/token |
|---|---|---|---|---|---|
| w-dec-f16-1 | f16 | 147.4, 149.1 | 28.48, 28.43 | 4.44, 4.49 | 29.05, 29.05 |
| w-dec-f32-1 | f32 | 148.4, 148.5 | 27.99, 28.04 | 5.01, 4.96 | 29.04, 29.05 |
| w-dec-f32-2 | f32 | 148.0, 148.8 | 28.24, 28.24 | 4.81, 4.85 | 29.05, 29.05 |
| w-dec-f16-2 | f16 | 147.5, 146.9 | 28.50, 28.51 | 4.43, 4.44 | 29.06, 29.05 |
| w-dec-f16-3 | f16 | 147.0, 143.3 | 28.48, 28.50 | 4.43, 4.44 | 29.06, 29.06 |
| w-dec-f32-3 | f32 | 148.2, 140.4 | 28.23, 28.27 | 4.78, 4.77 | 29.05, 29.06 |

f16 is ahead in each pair (+1.6%, +0.9%, +0.8%) and every f16 repetition is faster than every f32 one. The GPUs take
the same time; the exchanges, which include waiting for invite's partial sums, take 0.42 ms less per token, about
3 µs per all-reduce for 10 KiB fewer bytes each way. The f32 rate matches the one-cable TP4 of 2026-10-07 (28.28,
28.27, results/qwen/tp8-20261007); f16 is above the three-cable record (28.44, 28.24).

## MTP, cooled (`w-mtpc-*`)

Each run waited until every card of both Macs read at most 55 °C (`tools/cool.sh`: 234 s before the first, after the
gates; 55-69 s after each MTP run).

| candidate | n | f32 tok/s | f16 tok/s | change | verification ms/round | tokens/round | exchange ms/token | GPU wait ms/token |
|---|---|---|---|---|---|---|---|---|
| f16 payload (default) against f32 | 6+6 | 38.38 (38.28-38.47) | 39.17 (38.93-39.44) | +2.1% | 60.6 → 59.3 | 2.47 → 2.47 | 3.62 → 3.01 | 20.11 → 20.12 |

| run | payload | pp512 tok/s | tok/s | exchange ms/token | verification ms/round | hottest card (invite's first) |
|---|---|---|---|---|---|---|
| w-mtpc-f16-1 | f16 | 144.7, 146.6 | 39.03, 39.12 | 2.76, 2.79 | 59.4, 59.4 | 74, 78 °C |
| w-mtpc-f32-1 | f32 | 145.7, 146.4 | 38.28, 38.47 | 3.33, 3.28 | 60.7, 60.5 | 73, 79 °C |
| w-mtpc-f32-2 | f32 | 145.8, 146.9 | 38.35, 38.46 | 3.77, 3.75 | 60.6, 60.5 | 73, 78 °C |
| w-mtpc-f16-2 | f16 | 145.6, 146.6 | 39.36, 39.44 | 2.99, 3.04 | 58.9, 58.9 | 74, 79 °C |
| w-mtpc-f16-3 | f16 | 146.0, 146.5 | 38.93, 39.13 | 3.27, 3.25 | 59.6, 59.4 | 73, 78 °C |
| w-mtpc-f32-3 | f32 | 146.1, 146.6 | 38.33, 38.38 | 3.77, 3.80 | 60.6, 60.6 | 73, 78 °C |

f16 is ahead in each pair (+1.8%, +2.6%, +1.8%) and every f16 repetition is faster than every f32 one; drafts were
accepted alike (2.47 tokens per round). No card sampled below 850 MHz while busy in any generation window. The f32
rate matches the one-cable TP4 of 2026-10-07 (38.33, 38.85).

## Heat: the first MTP set (`w-mtp-*`, not used)

Run right after the decode set, one run about every 40 s, MTP heated invite's first D700 (02:00.0) into throttling.
Per repetition, that card's busy clock samples below 850 MHz in the generation window and its hottest reading
(`tools/windows.py`):

| run | tok/s | exchange ms/token | invite's first card |
|---|---|---|---|
| w-mtp-f16-1 | 39.11, 34.47 | 3.21, 6.37 | 0/12, 87 °C; 8/14, 91 °C |
| w-mtp-f32-1 | 38.36, 33.89 | 3.69, 6.77 | 0/13, 87 °C; 8/14, 92 °C |
| w-mtp-f32-2 | 37.34, 32.98 | 3.88, 7.08 | 1/13, 88 °C; 7/15, 93 °C |
| w-mtp-f16-2 | 36.99, 32.13 | 4.38, 8.27 | 3/13, 88 °C; 10/15, 93 °C |
| w-mtp-f16-3 | 36.12, 31.76 | 5.05, 8.57 | 3/14, 89 °C; 12/15, 93 °C |
| w-mtp-f32-3 | 35.79, 31.47 | 4.86, 8.44 | 4/13, 89 °C; 14/16, 94 °C |

The second repetition of every run was 12% slower than the first, and both sides fell by 8.5% over the set, while
incite's GPU wait stayed at 20.1 ms per token: the exchanges stretched as incite waited for invite's slowed card. The
set's means (f16 35.10, f32 34.97 tok/s) say nothing about the payload. The decode set did not throttle (invite's
first card reached 90 °C in its last run with no slow sample), and the cooled MTP set peaked at 79 °C. The fans ran at
the SMC maximum throughout (`fancurve.service`, about 2270 rpm). The 2026-10-07 improvements round saw invite's second
GPU throttle near 90 °C in verification (results/qwen/improvements-20261007).

## Correctness gates (`tools/wire.sh gates`)

| gate | f16 (default) | f32 |
|---|---|---|
| `qwen-validate` (14-token prompt) | against our CPU reference (`~/dev/claude-work/trashcan-20261006/ref/qwen38-27b-q4-ref`): worst 4.11e-4 over 946 tensors, logits top-10 10/10, greedy 10/10 ([val-f16-vs-ref.txt](val-f16-vs-ref.txt)); against llama.cpp's CPU dump (`qwen38-27b-q4-cpu`): logits 3.28e-2, greedy 10/10 ([val-f16-vs-llamacpp.txt](val-f16-vs-llamacpp.txt)). Both are the TP4 record's figures (results/qwen/tp-prefill-20261006): the prompt's all-reduces are prefill-sized, so f32; the ten greedy steps use f16 | not run |
| `qwen-check --mtp` | pass: restore exact, speculative output identical, 83% of drafts accepted, column score error 0 | the same |
| `qwen-check --verify 3` | pass: no column mismatches, column score error 0 | the same |

With two machines a verification column's elements are added in the same single pair as its decode step's, so f32's
column score error is 0 here too (TP8's ring: 2.25e-4). The four checks generated the same text.

## Files

- `tools/tp4.sh`: one TP4 run over incite and invite (node first, then the driver; leases, clocks, latency request).
- `tools/wire.sh`: the phases (`warm`, `decode`, `mtp`, `gates`, `mtpcool`); `tools/cool.sh`: waits until every
  card reads at most a given temperature.
- `tools/runs.py`, `tools/windows.py`: the per-run and per-window tables above.
- `incite/`: driver `.json`/`.err` of every run, `commands.log` (the warm-up's line added by hand: `results/` did not
  exist yet when it ran) and `clocks/`; `invite/`: node logs and `clocks/`.
- `val-f16-vs-ref.txt`, `val-f16-vs-llamacpp.txt`: `tools/compare_activations.py` on the validation dump, which is
  in `~/trashcan-exp/tp4wire/results/w-val-f16` on incite and `~/dev/claude-work/trashcan-20261009/tp4wire-w-val-f16`
  on the Mac.
