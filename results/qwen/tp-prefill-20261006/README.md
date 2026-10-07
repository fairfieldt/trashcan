# Tensor-parallel prefill: hide the all-reduces behind GPU work (2026-10-06)

Before this work a tensor-parallel prefill chunk ran the GPUs, then the host all-reduce, in strict alternation.
Now each chunk runs as two micro-batches on two compute queues: the host all-reduces one micro-batch while the
GPUs run the other, and the second queue fills the first one's bubbles. Every prefill candidate is bitwise the same
as before.

| Model, layout (incite + invite, USB4STREAM × 3 unless noted) | pp512 tok/s before | after | pp2048 before | after |
|---|---|---|---|---|
| **Qwen3.8-27B, TP4 across both Macs** | 89.8, 91.6 | **143.3, 143.0** | 87.3 | **136.3** |
| Qwen3.8-27B, TP4, TCP × 3 | 81.9 (`../tp-20261006`) | 140.0 (before the warm-up below) | | |
| **Qwen3.5-9B, TP2 on incite** | 209.9, 208.2 | **251.0, 250.7** | 203.0 | **238.3** |

Decode is unchanged (single-token forwards take the old path): 27B tg 16.2-17.0 tok/s (before 16.6), 9B tg 32.1
(before 32.2-32.3), with the same GPU wait per token (27B 45 ms, 9B 29.7 ms). The spread in 27B tg is the cross-Mac
exchange (9.5-13.3 ms per token across all runs, versions alike).

Hardware bound vs measured: with every all-reduce removed (`TRASHCAN_TP_FREE`, wrong results), the same two-queue
graphs run at 145.2 tok/s for the 27B, measured before the graphs were recorded at load: the GPU floor of this code.
The full run of that version reached 139.3, 96% of it. Before this work the GPU floor was 140.8 and the full run
reached 64% of it.

## Tested source and commands

Branch `worktree-agent-ad218b8d68d6edc6c`, built with `HOSTS="incite invite" scripts/cross-build.sh
trashcan-exp/prefill/trashcan` (kbuild VM). Each run name's prefix gives the commit:

| prefix | commit | change |
|---|---|---|
| `base-`, `free-*-pp512`, `freevram-` | 1ccb822 | prefill phases in `qwen-bench` (`pp_tp_ms`); `TRASHCAN_TP_VRAM_IO` timing switch |
| `mb-`, `free-*-m2`, `check-9b-*` | 1f88548 | two interleaved micro-batches on one queue; optional host threads |
| `prof-` | 93c5521 | per-kernel timestamps (`TRASHCAN_TP_PROFILE`) |
| `gemm-bench-` | 47773ad + the `gemm8` shaders below (not kept) | `qwen-gemm-bench` |
| `dq-`, `final-`, `free-dq-`, `check-27b-*`, `validate-*` | 662fa3b | two compute queues; 6 host threads across Macs |
| `warm-` | b4ee502 | full-chunk graphs recorded at load; no allocations per TP submission |

```sh
# incite (driver), inside a gpuq lease, performance governor on both Macs, scripts as in ../tp-20261006
./run.sh NAME ./trashcan qwen-bench ~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf \
    --tp-stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 --pp 512 --tg 32
./run.sh NAME ./trashcan qwen-bench ~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf --tp --pp 512 --tg 32
# invite (the other Mac's ranks), inside its own lease
./node.sh NAME ~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf [--mtp MTP.gguf]
```

`run.sh` is `scripts/with-clocks.sh results/tp-prefill/clocks/NAME.txt sudo -n env $ENV python3
scripts/latency-run.py CMD > NAME.json 2> NAME.err`; `node.sh` runs `qwen-tp-node MODEL --stream ...` under
`with-clocks.sh`. `ENV` carried the switches (`TRASHCAN_TP_MICRO`, `TRASHCAN_TP_THREADS`, `TRASHCAN_TP_QUEUES`,
`TRASHCAN_TP_FREE`, `TRASHCAN_TP_VRAM_IO`, `TRASHCAN_TP_PROFILE`); `incite/commands.log` and `invite/commands.log`
list every command. The driver passes its micro-batch and thread counts to the node with each forward; FREE,
VRAM_IO and the queue count are per process, so the node was restarted with the same setting.

## Where the time went before (`base-`, `free-`, `freevram-`)

27B TP4, one 512-token chunk, 5.59-5.70 s:

| phase | seconds | per all-reduce |
|---|---|---|
| waiting for the local GPUs | 3.65 | 28.5 ms per segment |
| adding the two local partials (one core) | 0.51 | 4.0 ms |
| exchange with the other Mac (10 MiB each way, includes waiting for it) | 0.85-0.95 | 6.6-7.4 ms |
| adding the other Mac's sum, writing both GPUs' totals | 0.53 | 4.2 ms |

The GPUs and the host took turns. Without all-reduces the GPUs need 3.64 s (140.8 tok/s), and 3.33 s (153.6
tok/s) with partials and totals kept in VRAM, so the host-memory traffic cost the GPUs about 0.3 s. The 9B TP2:
2.44 s = 2.09 s GPU + 0.33 s host (64 all-reduces of 8 MiB, no exchange); floor 242.5 tok/s, 254.1 without
host-memory traffic.

Per-kernel GPU time of one 27B rank for 512 tokens (`prof-27b-tp4-m1`, `-m2` with two micro-batches; GEMM rows x K):

| kernel | calls | ms (one batch) | ms (two micro-batches) | TFLOP/s |
|---|---|---|---|---|
| gemm 4352x5120 (FFN gate, up) | 128 | 1334 | 1350 | 2.19 |
| gemm 5120x4352 (FFN down, writes host memory) | 64 | 737 | 767 | 1.98 |
| gemm 5120x1536 (mixer out, writes host memory) | 64 | 418 | 432 | 1.23 |
| gemm 2560x5120 (DeltaNet qkv) | 48 | 343 | 351 | 1.88 |
| qwen_gdn | 48 | 190 | 190 | |
| gemm 1536x5120 (DeltaNet z) | 48 | 176 | 164 | 2.20 |
| qwen_add_norm adding the total (reads host memory) | 128 | 147 | 164 | |
| gemm 3072x5120 (attention q) | 16 | 122 | 128 | 2.10 |
| attention | 16 | 63 | 83 | |
| everything else | | 61 | 57 | |
| **busy** | | **3591** | **3686** | |

GEMMs are 88% of the GPU time and run at 55-63% of the D700's 3.48 TFLOP/s; the mixer out-projections are held to
1.2 TFLOP/s by their 10 MiB of shader stores into host memory (about 1.5 GB/s, `docs/findings.md` 2026-10-02).

## Steps and what each measured

**Two micro-batches (1f88548).** A prefill batch of at least 128 tokens runs as two micro-batches with their own
rows of the batch buffers and their own `pos, n` parameters, and their segments alternate on the queue: segment `j`
is segment `j / 2` of micro-batch `j % 2`. The all-reduce of one micro-batch releases its next segment while the
GPUs run the other's. Per-token arithmetic is unchanged, so the candidates are bit for bit those of the whole batch
(`pp_top` in every bench JSON).

| run (27B TP4 unless noted) | pp512 tok/s |
|---|---|
| one batch | 89.8, 91.6 |
| one batch, sums on 6 threads | 107.9 |
| **2 micro-batches** | **133.7, 133.2** |
| 2 micro-batches, 6 threads | 133.2 |
| 3 / 4 micro-batches, 6 threads | 125.9 / 119.5 |
| GPU floor, 2 micro-batches (FREE) | 136.8 |
| 9B TP2 (incite): 2 micro-batches; floor | 233.8, 234.1; 233.9 |

27B pp2048 128.8. The GPUs were 96% busy (profile); the second micro-batch cost 0.1 s of GPU time (narrower GEMMs,
twice the segments). More micro-batches cost more than they hide. Host threads helped only without overlap.

**Two queues (662fa3b).** Each GPU's second micro-batch runs on the second queue of the compute family; its segment
`i` waits for the first micro-batch's segment `i` (whose recurrent state and attention cache it continues), so the
first micro-batch's next segment runs beside it and fills its bubbles: `qwen_gdn` and `qwen_conv` occupy few CUs, the
out-projections wait on host-memory stores, `add_norm` on host-memory reads. Each micro-batch has its own timelines;
the second works in its own half of the work buffers (+60 MiB per GPU for the 27B, +78 MiB for the 9B). With the
GPUs now mostly busy, the host all-reduce of one micro-batch is on the path to its next segment, so the sums run on 6
threads when the group spans two Macs.

| run (same binary, node restarted with the same queue count) | pp512 tok/s |
|---|---|
| 27B TP4, one queue, 6 threads | 132.5, 133.0 |
| 27B TP4, two queues, 1 thread | 134.1, 136.5 |
| **27B TP4, two queues, 6 threads** | **140.5, 140.4** (pp2048 135.6, pp4096 126.5) |
| 27B TP4, two queues, GPU floor (FREE) | 145.2 |
| 27B TP4 over TCP × 3 | 140.0 |
| 9B TP2 (invite): one queue / two queues | 217.8, 217.3 / 233.2, 233.2 |
| 9B TP2 (incite): two queues, 1 / 6 threads | 247.0 / 244.9, 244.6 |

**Graphs at load (b4ee502).** The first prompt chunk used to record the two micro-batch graphs (about 80 ms for the
27B: `pp_seconds` minus the loop phases is 0.11 s for the first chunk and 0.03 s for later ones). They are now
recorded when the group loads. TP submissions and all-reduces also use fixed arrays instead of vectors, which keeps
decode's submit time at 0.39-0.41 ms per 9B token (0.44-0.46 with the vectors). Final: 27B 143.3, 143.0 (pp2048
136.3); 9B 251.0, 250.7 (pp2048 238.3).

## Correctness

All with the two-queue code (662fa3b; the 9B check and the 27B check repeated on b4ee502):

| check | result |
|---|---|
| `qwen-validate` 27B TP4 vs our CPU reference | worst 4.11e-4 over 946 tensors, logits 4.11e-4, top-10 10/10, greedy 10/10 (`tp4-27b-vs-qwen38-27b-q4-ref.txt`) |
| same vs llama.cpp CPU | logits 3.28e-2, greedy 10/10 (`tp4-27b-vs-qwen38-27b-q4-cpu.txt`) |
| same vs the TP4 dump of `../tp-20261006` | 0 difference over 258 tensors (`tp4-27b-vs-previous-tp4.txt`) |
| `qwen-validate` 9B TP2 vs our CPU reference / llama.cpp / previous TP2 | 3.74e-4 (logits 2.35e-4) / logits 2.14e-2 / 0 difference; greedy 10/10 |
| `qwen-check` 27B TP4, 600 tokens, verify 3, 128 steps | pass: restore exact, 85 rounds, 0 column mismatches, score error 0, text as before |
| `qwen-check --mtp` 27B TP4, 256 steps | pass: speculative identical, 148 of 216 drafts accepted (85 + 63), as before |
| `qwen-check` 9B TP2 with and without `--mtp` (invite) | pass; micro-batched and whole-batch prefills give identical reports apart from timings (144 of 222 drafts accepted) |
| prefill candidates, every bench | `pp_top` bit for bit equal across all versions and settings for each prompt length |
| long prompts | 27B pp2048 and pp4096, 9B pp2048 and pp4096 (invite) |

The 600-token checks prefill 512 tokens as micro-batches and 88 as one batch.

## What did not help

- **More micro-batches** (3: 125.9, 4: 119.5 against 133 for 2 on one queue): the extra segments cost more GPU time
  than they hide.
- **An 8 × 8 register tile per lane** for the Q4_0 GEMM (one wave per 64 × 64 tile, k-major LDS so each lane reads
  its rows and tokens as vec4): bit for bit identical but half the speed, 0.9-1.1 against 1.9-2.1 TFLOP/s at the
  27B's per-rank shapes (`invite/gemm-bench-n256.json`; RADV: 100 VGPRs, no spills, one wave per SIMD, uncoalesced
  tile loads). The existing `gemm` already shares most LDS reads by broadcast; LDS bandwidth is not its limit.
- **Host threads without overlap**: they cut the 27B's sum and total from 1.04 s to 0.31 s per chunk (107.9 tok/s),
  but once the all-reduces overlap GPU work on one queue they no longer matter (133.2 against 133.7). With two
  queues they do again (+4.5%), and on one Mac they still do not (9B: 244.9 with 6 threads, 247.0 with 1).

Not tried: SDMA copies (RADV exposes no transfer queue on GCN 1.0, `docs/findings.md`), f16 partials and bigger
stream rounds (the exchange and the host work are hidden behind GPU work now; f16 would also change the numerics).

## What is left

The 27B runs within a few percent of the two-queue GPU floor (145.2 tok/s), and that floor is the GPUs: 3.4 s per
512 tokens per rank, 88% of it Q4_0 GEMMs at 55-63% of peak; then `qwen_gdn` (0.19 s, a sequential walk over the
tokens) and the host-memory traffic, both now partly hidden by the second queue. Long prompts slow down in the scalar
prefill attention (pp4096 126.5).

## Files

- `incite/*.json`, `*.err`: driver runs (bench JSON with `pp_tp_ms`, `pp_top`, `tg_tp_ms`; `pp_profile` in `prof-`);
  `incite/clocks/`: clock logs; `incite/tp4-27b/greedy.txt`.
- `invite/`: node logs and clocks, the 9B checks, profiles and A/B runs on invite, the GEMM bench, `tp2-9b/greedy.txt`.
- `*-vs-*.txt`: `tools/compare_activations.py` against `~/dev/claude-work/trashcan-20261006/ref/` and the previous
  TP dumps in `~/dev/claude-work/trashcan-20261006/tp/` (the new dumps are in
  `~/dev/claude-work/trashcan-20261006/prefill/validate/`).

Clocks: 91 clock summaries, both cards at 850/1370 MHz in every run while computing (the low samples are model
loading and idle time) except the GEMM bench, which used one card; maximum 81 C.
