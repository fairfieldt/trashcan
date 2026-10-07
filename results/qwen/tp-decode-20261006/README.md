# Tensor-parallel Qwen decode: 27B TP4 at 28.5 tok/s, 39.6 with MTP (2026-10-06)

Decode work on the tensor-parallel path of `../tp-20261006/` (`engine/src/qwen/tp.rs`, `gpu.rs`). Same machines,
models and USB4STREAM lanes; prefill is unchanged.

| Model, layout | tg128 before | tg128 after | with MTP before | with MTP after |
|---|---|---|---|---|
| Qwen3.8-27B, TP4 over incite + invite | 18.96, 18.46 (documented: 16.57) | **28.48, 28.63** | 28.62, 28.52 (documented: 26.0) | **39.51, 39.63** |
| Qwen3.5-9B, TP2 on incite | 31.91, 32.01 | **45.95, 45.96** | 50.07, 50.02 | **66.15, 66.13** |

tok/s after a 512-token prompt (`qwen-bench --pp 512 --tg 128`), two runs each. "Before" is the base binary (main at
933a991) measured in the same conditions as "after": performance governor and a 0 µs CPU latency request on both Macs,
the node on invite included. The documented runs started the node without the latency request (the command listed
there); in these conditions the base binary's cross-Mac exchanges cost 36-44 µs instead of the documented 94 µs.
The 128 generated tokens are identical to the base binary's for both models, with and without MTP.

Tested source: commit 692f50b (binary md5 585c4dd7…, cross-built with `scripts/cross-build.sh`); the base binary is
933a991 (md5 09fbc282…). GPU clocks 850/1370 MHz while computing in every run (`incite/clocks`, `invite/clocks`;
300/150 samples are model loading and idle).

## What changed

1. **Event-gated forwards.** Decode, verification and draft steps (≤ 8 columns) are one command buffer and one
   submission. Each segment after the first waits on a VkEvent (RADV: a command-processor `WAIT_REG_MEM` on host
   memory) that the host sets after writing the all-reduced data, and sets a done event once its partial is in host
   memory; the host polls that event's memory. Before, every segment was its own submission released by a timeline
   semaphore through the kernel. Per segment, release-to-start fell from 52-62 µs to 3 µs and completion notice from
   7-10 µs to 1-2 µs (`profiles/9b-semaphores.txt`, `9b-gated.txt`). Prefill keeps per-segment submissions
   (amdgpu kills a job after ~10 s).
2. **Fewer buffer objects per submission.** A submission lists every memory object its command buffers use and the
   kernel validates each: about 600 per 9B rank and 1,160 per 27B rank (weights, activations, two events per
   segment). The gated forward is recorded into one command
   buffer, TP ranks suballocate device memory from 256 MiB chunks, and all segments share one (release, done) event
   pair (the command processor resets release after each wait, the host resets done). Segment 0 now starts 108 µs
   after submission instead of 1,178 µs (9B).
3. **Residual on the host.** The host already holds each all-reduced total, so it keeps the residual, adds the total and
   writes the next segment's normed rows (eight fixed partial sums, identical on both Macs). The projections read
   them from host memory, which the D700's L2 caches after the first read, so the GPUs run no residual norm (it took
   14-17 µs per segment as one workgroup).
4. **Kernels at TP shapes.** TP ranks use `matvecq4` everywhere: back-to-back `matvecq4narrow` dispatches (512-thread
   workgroups) do not overlap, `matvecq4` ones do. The small f32 alpha/beta and K/V projections are issued before
   the large ones, so they run beside them. `qwen_matf32` issues 16 loads per lane for up to four columns (it was
   one load round trip per 64 elements and gated the whole projection group). Split-K attention takes twice the
   partitions on TP ranks (6-8 query heads per GPU). `qwen_add_norm_wide` is a straight-line `qwen_add_norm` for
   drafts and prefill (same sums).
5. **Head selection.** `qwen_topk` keeps the best 64 of each 1024 entries with a bitonic sort in shared memory, in up
   to three passes, replacing `topk` (64 argmax rounds, 256 µs) and the one-workgroup `merge` (245 µs); same
   candidates.
6. **Balanced MTP draft head.** Draft heads rank the first 65,536 ids, which all live on rank 0's vocabulary slice
   (62,080 rows, ~2 ms) and 3,456 on rank 1's. Each rank now holds a copy of an equal share (16,384 rows, 105 MiB on
   the 27B), built at load time from `--draft-vocab`. Draft time per round: 7.4 → 3.8 ms (27B).

Measurement switches: `TRASHCAN_TP_EVENTS=0` (per-segment submissions), `TRASHCAN_TP_ARENA=0`,
`TRASHCAN_TP_HOSTNORM=0`, `TRASHCAN_TP_PARTS=N` (partition multiplier, default 2), `TRASHCAN_TP_DRAFT_HEAD=0`,
`TRASHCAN_TP_PROFILE=FILE` (+ `TRASHCAN_TP_PROFILE_DISPATCH=1`): GPU timestamps per segment (and per dispatch)
aligned to host time with calibrated timestamps, one JSON line per rank per step; `tools/tp_profile.py` and
`tools/tp_timeline.py` summarize them.

## Steps, Qwen3.5-9B TP2 (tg tok/s, ms of GPU wait per token)

| Change | tg128 | wait |
|---|---|---|
| base | 31.91, 32.01 | 30.4 |
| event-gated forwards | 35.13, 35.17 | 28.0 |
| + `matvecq4` at TP shapes, straight-line norm, small projections first | 39.50, 39.52 | 24.9 |
| + `qwen_matf32` loads, one command buffer, device arena | 42.56, 42.55 | 23.2 |
| + one event pair | 42.74, 42.74 | 23.1 |
| + twice the attention partitions | 43.12, 43.06 | 22.9 |
| + residual on the host | 44.69, 44.65 | 21.9 |
| + head selection (final) | 45.95, 45.96 | 21.3 |

Single-change comparisons: arena off 41.21 (42.55 on); old projection order 38.44 (39.50); old `qwen_add_norm`
37.23 (39.50); partitions ×1/×2/×4 42.68/43.12/42.85; balanced draft head off 63.46 with MTP (66.15).

## Qwen3.8-27B TP4

| | tg128 | GPU wait | exchange | host add |
|---|---|---|---|---|
| base | 18.96, 18.46 | 45.9, 46.3 | 4.7, 5.7 | 0.7 |
| events, kernels, one command buffer, arena (fb89690 without the single event pair) | 26.56, 26.65 | 32.5 | 4.2, 4.1 | 0.7 |
| final | 28.48, 28.63 | 29.2 | 4.1, 4.4 | 1.5, 1.1 |

ms per token (`tg_tp_ms`). On the final configuration: host residual off 27.22, partitions ×1 26.80 (GPU wait +0.7
ms), old head selection 28.23 (`incite/v6-27b-*`).

With MTP (2 drafts, 72.1% accepted, 2.44 tokens per round in every run): verify 78 → 58 ms per round, draft
7.4 → 3.8 ms; 9B: verify 40.4 → 31.3, draft 7.35 → 4.8.

### Where a 27B token goes now (`profiles/27b-final-*.txt`)

| | incite (ranks 0, 1) | invite (ranks 2, 3) |
|---|---|---|
| step | 35.0 ms | 35.0 ms |
| GPU segments | 28.8 ms | 30.1 ms |
| exchange and add between segments | 5.5 ms | 4.1 ms |
| segment start after release, sum | 0.52 ms | 0.68 ms |
| per segment: DeltaNet / FFN / attention / head | 165 / 235 / 224 / 2,092 µs | 170 / 248 / 229 / 2,233 µs |

invite's GPUs run the streaming kernels 5-7% slower than incite's at the same 850/1370 MHz (FFN matvecs 146.6 vs
138.3 µs, head 2.01 vs 1.87 ms; latency-bound kernels are equal: `profiles/27b-v4-*-dispatches.txt`), so incite waits
for invite at most exchanges. A cross-Mac exchange itself costs 15-25 µs.

Bounds: a 27B TP4 rank streams ~4.2 GB per token (Q4_0 at 20 bytes per block plus a 397 MB Q6 head slice): 16 ms at
the D700's 263 GB/s, 21 ms at the 199 GB/s `matvecq4` reaches on a whole 17408×5120 matrix (`mvsweep.jsonl`). Measured:
28.8-30.1 ms of GPU time (140-146 GB/s including the small kernels) and 35 ms per token.

## Correctness

| | 9B TP2 | 27B TP4 |
|---|---|---|
| `qwen-validate` vs our CPU reference | worst 3.74e-4 over 474 tensors, logits 2.35e-4, top-10 10/10 | worst 4.11e-4 over 946 tensors, logits 4.11e-4, top-10 10/10 |
| vs llama.cpp CPU | logits 2.14e-2 | logits 3.28e-2 |
| greedy tokens | 10/10 identical | 10/10 identical |
| `qwen-check --verify 3 --tokens 600 --steps 128` | pass: restore exact, 85 rounds, 0 column mismatches, score error 0 | same |
| `qwen-check --mtp --steps 256` | pass: 170 rounds, speculative output identical, 64.9% accepted | pass: 170 rounds, identical, 68.5% |

Unchanged from `../tp-20261006/` (the prompt forward of `qwen-validate` takes the per-segment path; its greedy decode
and both checks take the new one). `val-*-vs-*.txt`, `incite/check-*-final.json`. A SIGTERM delivered mid-decode
ends the process with status 143 and no GPU timeout (the signal handler sets the release events until the
submission drains); the next run was normal.

## Tried and not kept

- `qwen_add_norm` with per-element bounds checks: 59 µs instead of 38 (each guarded load became a branch with a
  wait). The specialization-constant version without guards is the one kept (12-17 µs).
- Attention partial kernel batching four keys per memory round trip (same results): 96 µs vs 95. Four times the
  partitions: the partial kernel drops to 40 µs but the merge rises to 75 µs; a merge loading partitions in batches
  of 32: 71 µs.
- `matvecq4` with eight rows per wave (bit-identical): slower at every shape, e.g. 146.5 vs 164.6 GB/s at 4352×5120.
- Two or three projections in one dispatch (`qwen_q4multi`, bit-identical): 42.99 vs 43.06 tok/s on the 9B, 27.1-27.3
  vs 28.6-28.9 on the 27B.
- The host residual update and norm in one loop: 44.1 tok/s vs 44.4-44.7 with separate loops.
- Building the balanced draft head on the first draft step: a 1.9 s stall (9B) in the first round.

## Files

- `incite/*.json`: `qwen-bench`/`qwen-check` outputs (`base-*`: base binary; `final-*`: tested source; the others are the
  steps and comparisons above, in run order: `gated`, `v2`-`v7`, `parts*`, `merge-*`, `multi*`, `hostnorm*`, `dh-*`);
  `*.err`: stderr with clock summaries; `invite/node-*.log`: node logs.
- `profiles/`: summaries of `TRASHCAN_TP_PROFILE` runs and the final 27B profiles of both Macs (gzipped JSON lines).
- `mvsweep.jsonl`, `mvsweep.sh`: `matvec-bench --rows --cols` at TP shard shapes.
- `scripts/`: `bench.sh` (driver runs with clocks and latency request), `node.sh` (node on invite), `tp4.sh`,
  `tp4check.sh`, `tp4run.sh`, `final27b.sh`, `perf.sh` (governor).

```sh
# invite, inside a gpuq lease
./node.sh node-final-27b ./trashcan ~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf --mtp MTP --stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2
# incite, inside a gpuq lease
./bench.sh final-27b-a ./trashcan qwen-bench MODEL --tp-stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 --pp 512 --tg 128 [--mtp MTP]
./bench.sh final-9b-a ./trashcan qwen-bench ~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf --tp --pp 512 --tg 128 [--mtp MTP]
```
