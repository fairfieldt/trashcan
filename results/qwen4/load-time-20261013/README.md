# Flash-Next model load: weights repacked straight into reused staging buffers (2026-10-13)

**Result.** A GPU stage copied every weight four times on one core before the GPU saw it: the expert rows into one
`Vec` per layer, the repack into another, then into a fresh host-visible staging buffer per tensor, followed by a
blocking copy. Now the matrices are repacked straight from the GGUF mapping into three reused 64 MiB staging buffers,
by every core, and each buffer's copies run on the GPU while the next one is filled; one wait at the end. One Mac
(indite, layers 0-7 on its two GPUs, 11.8 GB): **`Local::new` 20.4 -> 1.96 s warm, 41.1 -> 11.9 s from a cold page
cache** (the SSD reads at 1.2 GB/s at best: 9.8 s); the whole stage-bench process 18.4-22.1 -> 2.49-2.56 s warm,
42.0-43.9 -> 13.0 s cold. The GPU holds the same bytes as on main (174 buffers read back and hashed, 11.8 GB). Six
Macs: **a whole `qwen4 validate` run 22.9-23.2 -> 5.3 s, `trashcan serve` up after 19-20 -> 3 s**, outputs
bit-identical. `qwen4 stage-bench --batch 16,32,64,128` times several chunk sizes on one load.

Tested source: the two engine commits of this branch, on each base as main moved: 6f6635a (`v1`, sessions 0-1,
without the in-flight guard of `Buffer::drop`); 7da6f93, the prefill kernels, whose `repack_q8h` pads the Q8_0 scales
to 64 bytes (session 2 without the guard, session 3 as c8ac8b1); 4307169, the small-batch experts (session 4); and
5e48d7e, connect retries and a `q4run.py` that starts the driver once the nodes listen (the six Macs, as
`trashcan-load`). The load path is the same in all; the rebases after 7da6f93 were conflict-free. Baselines
`trashcan-6f6635a`, `trashcan-7da6f93`, `trashcan-4307169` and `trashcan-5e48d7e`. The timing builds add the counters
of [tools/loadprof.rs](tools/loadprof.rs) (main: [tools/prof-main.patch](tools/prof-main.patch) with
[tools/loadprof-main.rs](tools/loadprof-main.rs); this branch: [tools/instrument.py](tools/instrument.py)); they are
not part of the branch.

## Where a load went (main)

`tools/load-sb.sh` on indite (6-core E5-1650 v2): `qwen4 stage-bench G-iq4 --ranges 0:4,4:8 --gpus 0,1 --context
32768 --batch 128 --head-rows 0:21504,21504:43008 --decode 1 --start 16 --prefill 0`, wall time of the process; `cold`
drops the page cache first (`sync; echo 1 > /proc/sys/vm/drop_caches`). Per GPU stage (5.9 GB; GPU0 / GPU1, warm,
[indite/load-prof0-w1.err](indite/load-prof0-w1.err) and `-w2`, [indite/load-prof4-w1.err](indite/load-prof4-w1.err)):

| phase | main 6f6635a | this branch (`prof4`) |
|---|---|---|
| stage total | 10.61 / 9.28 s (10.67 / 10.08) | 0.94 / 0.87 s |
| expert rows concatenated into a `Vec` per layer (the "unaccounted" 2.5 s) | 2.44 / 2.07 | - |
| repack into a new `Vec` (one core) | 2.89 / 2.58 | - |
| a host-visible staging buffer per tensor: allocate, write, free | 1.03 + 3.37 + 0.09 / 0.91 + 2.89 + 0.08 | 3 x 64 MiB once: 0.02 / 0.01 |
| copy, submit and wait, per tensor | 0.63 / 0.63 | - |
| repack straight into staging, every core | - | 0.67 / 0.66 |
| waiting for a staging buffer's copies | - | 0.13 / 0.14 (one wait at the end: 0.00 / 0.01) |
| `Device::new` (pipelines from the driver's cache) | 0.04 / 0.02 | 0.08 / 0.01 |
| the rest (f32 conversions, device buffers, activations) | 0.12 / 0.10 | 0.04 / 0.04 |

The concatenation was the 2.5 s the earlier profile could not place. The staging writes ran at 1.8 GB/s into memory
that was new for every tensor; the parallel repack into reused staging buffers runs at 8.9 GB/s, and then the copies
set the pace: a stage waits 0.13 s for them (5.9 GB in 0.80 s, about 7.4 GB/s with the CPU filling the next buffer
meanwhile).

Cold, per stage: main 20.2 / 20.3 s, of which the concatenation read the expert rows fault by fault (12.9 / 12.4 s,
0.5 GB/s); this branch 5.78 / 5.61 s, the repack's twelve threads faulting in parallel (1.05 GB/s,
[indite/load-prof4-c1.err](indite/load-prof4-c1.err)). `dd` with O_DIRECT reads this SSD at 1.2 GB/s with 64 MiB
requests, 1.1 GB/s with 4 MiB, 0.87 with 128 KiB ([indite/session1.log](indite/session1.log)).

## The change

- `quant::Planes` / `Kind::fill`: the GPU layouts (Q4_0 with f32 scales, Q4_0/IQ4_NL and Q8_0 with f16 scales,
  IQ4_XS, Q6_K head) as two planes, a head per block (scale or header) and a body, so any run of whole blocks can be
  repacked into its two places; `repack_q6`/`repack_q8h` stay for the other callers, and a unit test checks the planes
  against the whole-matrix repacks byte for byte, also written in pieces.
- `vulkan::Uploader`: three 64 MiB mapped staging buffers cycled per device. `planes` takes a matrix as runs of GGUF
  blocks (the experts' 1026 runs of gate and up rows, or the mixer's bottleneck and write rows) and records the
  copies of as many blocks as fit; when a staging buffer is full its repacks run on every core in ~1 MiB pieces
  (`std::thread::scope`, only plain slices cross threads), then its copies are submitted with a fence and the next
  buffer is filled while they run. `bytes` stages small f32 data. Vulkan stays on the calling thread, and a buffer
  dropped while copies are in flight (an error mid-load) waits for them.
- `GpuStage::new` uploads through one `Uploader` and waits once, at the end; the experts no longer pass through a
  concatenated `Vec`. Device buffers, their order and sizes are unchanged.
- `Local::new` reports each stage's load time (`GPU0: 5802.3 MiB allocated in 0.93 s; ...`).

Tried and dropped: a thread reading the stages' tensors ahead with `madvise(MADV_WILLNEED)` in 4 MiB requests (`v2`):
no gain warm (2.60 against 2.52 s); cold the read-ahead itself took 10.9 s, about what the repack's parallel faults
take, and the queued reads held up the second device's creation by 10.5 s (cold runs 14.1 and 22.9 s against 12.9
and 14.2; [indite/load-prof2-c1.err](indite/load-prof2-c1.err)).

## Single Mac, main against this branch

indite, alternating ([indite/session1.log](indite/session1.log), [indite/session2.log](indite/session2.log),
[indite/session3.log](indite/session3.log), [indite/session4.log](indite/session4.log)):

| run | main | this branch |
|---|---|---|
| warm, process | 6f6635a: 19.2, 18.4, 21.5, 20.5 s; 7da6f93: 22.1, 19.9 s; 4307169: 20.0 s | v1: 2.52, 2.52 s; on 7da6f93: 2.56, 2.50 s, c8ac8b1: 2.53, 2.54 s; on 4307169: 2.49 s |
| warm, `Local::new` (timing builds) | 20.05, 20.85 s | 1.98, 1.94 s |
| cold, process | 6f6635a: 43.9, 43.4 s; 7da6f93: 42.7, 42.0 s | v1: 12.9, 14.2, 13.1 s; on 7da6f93: 13.0, 13.0, 13.0 s, c8ac8b1: 13.0 s |
| cold, `Local::new` (timing builds) | 41.1 s | 11.8, 11.9 s |

The rest of a process (embedding the prompt, n-gram rows, three graphs recorded, a few chunks, the exit) takes 0.6 s
warm and 1.1 s cold.

## Correctness

- Weights: the hash builds ([tools/hash.py](tools/hash.py)) read back every buffer a stage uploads, after loading, and
  print an FNV-1a hash of its bytes in upload order. 174 buffers (11.8 GB) identical: `v1` against main 6f6635a, the
  branch on 7da6f93 (session 2) and c8ac8b1 (session 3) against main 7da6f93 with its 64-byte Q8_0 padding
  ([indite/load-hashfinal-w1.err](indite/load-hashfinal-w1.err)), and the branch on 4307169 against 4307169
  (session 4).
- stage-bench dumps of the 100-token prompt (`--tokens-file p100.ids`, layers 0-7 on both GPUs): all 114 tensors
  byte-identical (`cmp`), against 6f6635a, 7da6f93 and 4307169. The same device-local memory is allocated (5802.3 and
  5768.5 MiB).
- Six Macs: `qwen4 validate` on the 14-token reference prompt bit-identical to main 5e48d7e (all 679 dumped tensors,
  `logits.f32`, the greedy tokens; [Six Macs](#six-macs)).
- `cargo test`: 72 passed (in the kbuild VM).

## Six Macs

[tools/ab6.sh](tools/ab6.sh) under `chain6.sh`, one session of 2 min 20 s ([ab6.log](ab6.log)): incite drives, nodes
indite, inflight, alight, invite and enlight, `--head-split 12 --context 32768 --batch 128 --ngram disk`, the
`q4run.py` of 5e48d7e (the driver starts once every node listens); main 5e48d7e against this branch on 5e48d7e,
alternating:

| run | main 5e48d7e | this branch |
|---|---|---|
| `qwen4 validate` (14-token prompt, 10 greedy tokens, dump), whole run | 22.9, 23.2 s | 5.3, 5.3 s |
| its driver process | 21, 22 s | 4, 4 s |
| `trashcan serve` (`tools/q4serve-test.sh`, `--context 16384`): server up after | 20, 19 s | 3, 3 s |
| the server run as a whole (up, the two-turn chat check, stop) | 29.7, 28.4 s | 12.3, 11.9 s |
| each GPU stage's load (`allocated in`, driver and node logs) | | 0.87-0.91 s on every Mac |

Correctness: both validate runs of the branch against main's first: all 679 dumped tensors byte-identical (`cmp`), the
same 10 greedy tokens, `logits.f32` byte-identical; main's second run identical to its first. The chat check passes
with both (the second turn reuses the first's state; [incite/load-serve-new-1.json](incite/load-serve-new-1.json)).
The GPUs held 850/1370 MHz through the branch's runs; during main's loads they fell to 300/150
([incite/clocks](incite/clocks), `nodes/*/clocks`). invite's node logs are missing: it went down about a minute after
the session (a PCIe completion timeout on GPU0 during the next job), before they were copied.

## stage-bench: several sizes on one load

`--batch 16,32,64,128` loads once, the largest size sizing the buffers, and prints one JSON line per size in the format
of a single `--batch` (same keys; a single size prints what it printed before). indite, layers 0-7, `--prefill 3
--decode 8 --start 64`, one load against a process per size ([indite/load-multi.json](indite/load-multi.json),
`load-single-*.json`; `tools/multi-check.py`):

| batch | prefill wall ms | prefill GPU ms, layers 0-3 / 4-7 | chunk at 64 GPU ms | decode wall ms |
|---|---|---|---|---|
| 16 | 58.5 / 52.5 | 29.5, 27.4 / 26.4, 24.5 | 28.7, 27.0 / 25.6, 23.8 | 9.48 / 9.43 |
| 32 | 88.6 / 84.4 | 44.2, 41.4 / 42.2, 39.5 | 43.1, 40.7 / 41.3, 38.9 | 8.88 / 9.53 |
| 64 | 133.7 / 127.9 | 64.1, 61.7 / 61.2, 58.8 | 63.5, 61.8 / 61.2, 58.9 | 8.90 / 9.53 |
| 128 | 226.7 / 225.9 | 107.1, 103.1 / 107.2, 103.2 | 64.0, 61.9 / 64.0, 61.9 | 8.92 / 9.44 |

One load with four sizes took 6.0 s; four processes 12.8 s with this load, about 88 s with main's.

The smaller sizes are 5-12% slower on one load, and that is the chain's configuration, not noise: since the prefill
kernels (583d2dc) the Q8_0 out projections of a prefill batch take their K splits from the stage's batch capacity
(`splits` = 8 when `8 n d <= hc d max_batch`, else 4), but the split path also needs `splits * d <= hc * d`, so with 8
it falls back to the unsplit GEMM. A stage loaded for `--batch 128` (a chain, or this sweep) therefore runs chunks of
16-64 tokens with unsplit out projections, while a stage loaded for `--batch n` alone always gets 4 splits. At 128
tokens both use 4 splits and agree to 0.1 ms. The dispatch profile of a 16-token chunk on layers 0-3
(`TRASHCAN_Q4_PROFILE=1`, [indite/session3.log](indite/session3.log), `tools/outproj.py`) shows it: loaded for 128,
the four out projections run `qwen4_q8_gemm_s16r1p` unsplit in 5.44 ms; loaded for 16, split 4 ways in 2.31 ms plus
0.03 ms of `qwen4_sum_parts`. So per-size stage-bench runs overstate the chain's prefill speed for small chunks, and
the chain is missing the split path there, 2-3 ms per stage of 4 layers and chunk of 16-64 tokens (see `out_proj` in
`engine/src/qwen4/gpu.rs`).

## What is left

- A warm load is now bound by host memory bandwidth: the repack reads every byte and writes it (each write line read
  for ownership first), and the copies read it again, about 47 GB of traffic for 11.8 GB in 1.6 s (29 GB/s on this
  quad-channel DDR3-1866 Mac). Interleaving the two GPUs' staging buffers (each stage waits 0.13 s for its copies)
  would not change that total; non-temporal stores in the repack could save about a quarter of it, and copying from
  the page cache itself (an imported host pointer, repacked on the GPU) most of it.
- A cold load reads at 1.05 GB/s, within 15% of what `dd` gets from this SSD.
- On the chain a validate run now takes 5.3 s: 0.7 s until the nodes listen, 4 s of driver (1.8 s of it loading),
  0.5 s to stop the nodes. Several measurements on one load (nodes keeping their stages between sessions with the same
  setup, or one driver process running a list of runs) would save the loading and the process starts, about 3 s of
  each run.
