# Qwen3.8-27B TP8 optimization round (2026-10-08): runbook

The plan, cost model and design are in [docs/tp8-optimization.md](../../../docs/tp8-optimization.md). This round was
implemented and checked without access to the Macs: nothing here has run on a D700 yet. This file says how to test it
on incite, invite, enlight and indite, what to expect, how to decide, and what to look at once it runs.

Branch `claude/wizardly-planck-823ef4` (AGENTS.md asks for work on `main`; this session was given that branch, so merge
it to `main` with the measurements once they are recorded). Source of the off-hardware checks below: 302a18f.

## What changed

Every candidate is behind a switch whose default is the current behaviour, except H1, which is bitwise identical and
on by default. Kernel switches are read by each process, so every node needs the same environment (`tools/tp8.sh`
passes ENV to all four). `TRASHCAN_TP_WIRE` is read by the driver only and handed to the nodes at session setup.

| id | switch | effect | bits |
|---|---|---|---|
| W1 | `TRASHCAN_TP_WIRE=f16` (or `f16:N`, all-reduces of at least N values) | all-reduce slices travel as f16 with one power-of-two shift per 256 values; multi-column batches sliced per column | sums change (two f16 roundings per all-reduce), identical on every machine |
| H1 | `TRASHCAN_TP_HOSTPASS=0` turns it off | the host's remote add, residual, norm and both GPUs' input rows in one fused pass | identical |
| A1 | `TRASHCAN_QWEN_DECODE_ATTN=1` | `qwen_attn_decode_gN` + `qwen_attn_merge` (output gate fused) for decode, verification and drafts | summation order changes |
| K1 | `TRASHCAN_QWEN_TOPK=wave` | `qwen_topk_wave`: head selection with subgroup barriers in 45 of 55 steps | identical |
| F1 | `TRASHCAN_QWEN_FUSED_FFN=1` | `qwen_ffn_gu1..4`: FFN gate, up and SwiGLU in one dispatch (1-4 columns) | identical |
| G1 | `TRASHCAN_QWEN_FUSED_GDN=1` | `qwen_gdn_conv` + `qwen_gated_norm_conv`: DeltaNet convolution inside the recurrence (up to 8 columns; dumps keep the old kernels) | identical |
| M1 | `TRASHCAN_QWEN_SHORTK=8\|16` | `qwen_matvec_rows8/16`: 8 or 16 rows per wave for one-column projections with rows of at most 2048 values (the out-projections) | identical |

Also new: `trashcan qwen-kernel-check` (every kernel above against the kernels it replaces and a CPU reference, timed),
a relaxed Vulkan device for it on non-64-lane hardware (`--any-device`), and a fix in `Buffer::drop`, which now
evicts descriptor sets cached by the buffer's handle (a later buffer could inherit a stale set, e.g. after the draft
head is replaced). The measurement switches of the earlier rounds (`TRASHCAN_TP_PARTS` etc.) are unchanged.

## Already checked off the Macs

- `cargo test --release`: 68 tests ([cargo-test.txt](cargo-test.txt)). New ones: four machines over localhost TCP keep
  bitwise-identical totals with the f16 payload for decode, verification (whose columns equal one-column steps bit
  for bit) and reduce-scatter/all-gather, within 2e-3 relative of the f32 sums, with massive values up to 4e5; two
  machines over three lanes; F16C and software rounding agree bit for bit; the fused host pass equals the old passes.
- `qwen-kernel-check --any-device` on lavapipe (llvmpipe, Mesa 25.2.8, 8-lane subgroups): 28 cases, all pass
  ([lavapipe-kernel-check.jsonl](lavapipe-kernel-check.jsonl)). Decode attention within 5e-7 of an f64 reference
  (G = 2, 3, 4, 6; 1-8 columns; up to 8000 keys and 128 partitions, empty partitions); top-k exactly the top 64 with
  tie order; fused FFN, fused DeltaNet (state, output, windows, both snapshot kinds) and the many-rows matvec within
  5e-7 relative. lavapipe runs one subgroup of a workgroup after another, so a missing workgroup barrier in the
  wave-local top-k would have shown as wrong candidates.
- All 76 shaders that existed at 261d453 compile to byte-identical SPIR-V; the new kernels are separate files.
- F16C codec on a 2.8 GHz Xeon: encoding 5120 values (with the in-place rounding) about 1.0 µs, decoding 0.4 µs.
  Expect up to twice that on the Ivy Bridge Macs.

Not checked: anything on RADV/GFX6 (register allocation, the bitwise claims, speed), and the Thunderbolt path.

## 1. Build and deploy (kbuild VM, about a minute)

```sh
# the candidate, to the new workspace on all four Macs
HOSTS="incite invite enlight indite" scripts/cross-build.sh trashcan-exp/tp8opt/trashcan-opt
# the record binary, 261d453, for the equivalence runs
git worktree add ../trashcan-261d453 261d453 && (cd ../trashcan-261d453 && \
  HOSTS="incite invite enlight indite" scripts/cross-build.sh trashcan-exp/tp8opt/trashcan-base)
# scripts the runs use (with-clocks.sh, latency-run.py, clockmon.sh, clocksummary.sh)
for h in incite invite enlight indite; do rsync -a scripts $h:trashcan-exp/tp8opt/; done
```

Streams as in results/qwen/tp8-20261007 (`scripts/tbstream-up.sh A B tp8` per cable, five cables). Leases:
`gpuq --on MACHINE request tp8opt --est MIN` and `gpuq --on MACHINE wait ID` on each Mac, then
`export TICKETS=incite=ID,invite=ID,enlight=ID,indite=ID`. Benchmarks need the performance governor on all four.
The tools below run from the machine that drives the leases, in this directory.

## 2. Kernel checks on the D700s (about 5 minutes per Mac, both GPUs)

```sh
tools/kcheck.sh trashcan-opt 200          # writes results/kcheck-HOST.jsonl on each Mac; prints exit codes
```

Expected on every Mac and both GPUs (one JSON line per case; the command exits 1 if any case fails):

| check | must hold | expected timing (old_us vs new_us) |
|---|---|---|
| attention | `max_err_vs_old` and `max_err_vs_reference` below 2e-4 x scale (typically 1e-6) | TP8 decode case (g 3, 520 keys, 64 partitions): old about 95-100 µs (partial 55.7 + merge 38.5 + gate in the profile), new 12-30 µs; 8000 keys / 128 partitions: old several hundred µs, new under 60 µs |
| topk | `bitwise_vs_old` true and `exact` true for all 4 cases | 31040 rows: old about 214 µs (three passes), new 70-120 µs |
| ffn | `bitwise_vs_old` true | 2176 x 5120: new at most old minus 3 µs (SwiGLU dispatch and a barrier gone); a slower fused kernel decides F1 against |
| gdn | `bitwise_vs_old` true for state, window, output and snapshots | TP8 decode case: old about 36 µs, new 30 µs or less |
| matvec | `bitwise_vs_old` true | 5120 x 768: old about 26 µs; rows16 should be faster for M1 to stay |

A `bitwise_vs_old` of false with the reference error still small means ACO compiled an expression differently: the
kernel is correct but not identical. For ffn, gdn and matvec, accept it only if the end-to-end gates below pass, and
record that the candidate changes sums. For topk it would be a real fault (a barrier): do not enable K1. A Vulkan error
or `VK_ERROR_*` at pipeline creation points at specialization-constant arrays in `qwen_attn_decode` or `qwen_gdn_conv`.

## 3. Equivalence with the record (about 3 minutes)

```sh
tools/ab.sh base         # 261d453 and the new binary, defaults, alternating
```

Expect the same generated tokens and 35.0-35.5 tok/s decode, 200-207 pp512 for both (`tools/summarize.py` on the
pulled `results/`). The new binary's default differs only by H1 (bitwise identical).

## 4. Correctness gates (about 20 minutes)

```sh
tools/ab.sh gates        # qwen-validate, qwen-check, qwen-check --mtp, qwen-check --verify 3 per configuration
```

Then on incite (`tools/compare_activations.py A B`, B the reference):

| comparison | expected |
|---|---|
| `results/val-default` vs the record's dump (`results/tp8-rsag-val` in the tp8 workspace) | every tensor identical (rel 0) |
| `val-a1`, `val-k1`, `val-f1`, `val-m1` vs `val-default` | identical, with the same greedy tokens. The dumped forward is the 14-token prompt, which A1 and M1 (decode widths only) do not touch; their greedy steps do. G1 is not used by dumps; its gate is `check-g1` and the kernel check |
| `val-w1`, `val-all` vs the CPU reference dump (the one behind `tp8-split-vs-ref.txt`) | worst rel at most about 1.5e-3 (record: 4.0e-4), logits top-10 10/10, greedy 10/10 |
| every `check-*` | `pass` true, `restore_exact` true, no column mismatches |
| `check-mtp-*` | speculative output identical, acceptance close to 83% |
| `check-v3-w1` | `max_column_score_error` near 0 (record 2.25e-4): with W1 a verification column is added in the same pairs as a decode step; a nonzero value at a partition-bucket boundary is possible with A1 |

Any failure: stop and keep that candidate off; the others are independent.

## 5. Performance A/B (about 90 minutes)

```sh
tools/ab.sh decode       # each candidate against the default, alternating, then all together
tools/ab.sh mtp
tools/ab.sh prefill
python3 tools/summarize.py RESULTS_DIR dec- mtp- pp-   # RESULTS_DIR: incite:trashcan-exp/tp8opt/results, pulled
```

Expected effect against the default of the same phase (estimates from the cost model; the measurement decides):

| candidate | decode tg128 (35.0) | host phases per token | MTP (46.0) | prefill pp512 (204) |
|---|---|---|---|---|
| W1 | 36.5-37.5 | exchange 7.5 -> 5.8-6.5 ms | 49-51; verify 50.6 -> 46-47 ms/round | 207-212 (GPU-bound near 2.42 s per chunk); exchange 2.07 -> 1.1-1.3 s |
| H1 (default on; `h1off` is the A/B) | +0.1-0.3 | add 1.3 -> 1.1-1.2 ms | +0.3 | none |
| A1 | 36.3-37.0 | wait 19.2 -> 18.0-18.3 ms | 47.5-48.5 | none |
| K1 | +0.1-0.2 | wait -0.1 ms | +0.3 | none |
| F1 | +0.2-0.4 | wait -0.2-0.3 ms | +0.3 | none |
| G1 | +0.2-0.4 | wait -0.2-0.3 ms | +0.3 | none |
| M1 (16 rows) | 0 to +0.8 | wait 0 to -0.7 ms | 0 to +0.5 | none |
| all | 39-41 | step 28.4 -> 24.5-25.5 ms | 50-54 | as W1 |

Keep a candidate when it is faster in every alternated pair (AGENTS.md); the small ones (K1, F1, G1) are near the
run-to-run spread of about 0.2 tok/s, so judge them from the `all` runs with and without them if their own pairs are
inconclusive, and from the GPU `wait` per token, which varies far less than tok/s.

## 6. Sweeps, long context, profile (about 15 minutes)

```sh
tools/ab.sh sweep        # TRASHCAN_TP_PARTS 1/2/4 with everything on; W1 as f16:15360; M1 with 8 rows
tools/ab.sh long         # decode after 7680 tokens, A1 against the default
tools/ab.sh profile      # dispatch stamps on all four Macs, default and everything on
python3 tools/tp_profile.py results/profile-prof-all.jsonl --n 1 --dispatch   # (repository tools/)
```

- Partitions: the new attention makes the merge cheap, so the best multiplier may change (today 2). Keep the fastest.
- `f16:15360` keeps decode in f32 and compresses verification and prefill. Use it if W1's decode pairs show no gain
  (the codec cost cancelled the shorter transfers) while MTP and prefill gain.
- Long context: at 7.7K keys the old split-K attention costs an estimated 4-5 ms per token (64 keys per partition);
  expect A1 to raise this decode by 15-20%.
- Profile: per-segment GPU times should drop by about 70-80 µs on attention segments (A1), 5-8 µs on DeltaNet
  segments (G1) and 4 µs on FFN segments (F1); `exchange+add` per segment by 10-15 µs (W1, H1).

## 7. Decide and clean up

Per AGENTS.md: make each accepted candidate unconditional and delete its switch; delete every rejected kernel, its
pipeline registration and its switch; keep `qwen-kernel-check` cases only for kernels still in use (and the old
kernels they compare against, while those still serve prefill or other widths). Record the measurements, clocks and
commands here, update docs/qwen.md (TP8 numbers, switches) and docs/findings.md, and merge to `main`. Then run
`scripts/test-server.py` against `serve --tp-peer ... --mtp` with the final binary.

## 8. Things to address once live

1. **Bitwise claims on GFX6.** K1, F1, G1 and M1 are bitwise by construction (same expressions in the same order), but
   ACO decides contraction per shader. Section 2 settles it; if one differs, decide as described there.
2. **Registers.** `qwen_attn_decode_g6` (TP4 and the 9B) keeps about 170-200 values per lane; check
   `RADV_DEBUG=shaderstats` for spills. If it spills, lower `KB` from 8 to 4 in the shader.
3. **W1 quality.** Two f16 roundings per element per all-reduce, 128 per token. Compare greedy outputs on the mixed
   corpus (`qwen-bench --corpus tests/fixtures/qwen-bench-corpus.jsonl`) with and without W1, and MTP acceptance. Text
   may diverge after a near tie; a systematic change in acceptance or in the validate error would argue for
   `f16:N` with a large N or for dropping W1.
4. **W1 on prefill tails.** Reduce-scatter/all-gather in f16 needs segments of whole 256-value blocks (micro-batches of
   an even number of rows); odd tails fall back to the per-column f16 rounds, single threaded. Harmless for 512/2048
   prompts; check `pp_tp_ms.exchange` on odd-length prompts if they matter.
5. **Partition buckets.** A verification column and a decode step at a bucket boundary (`partitions()`) split keys
   differently, so their scores differ in the last bits; only `check-v3` with W1 can show it.
6. **H1 gain is small.** Measured locally at 0.6 µs per all-reduce, not the 4 µs the plan assumed: the five passes were
   already vectorized. Keep it only if `h1off` is measurably slower; otherwise delete it.
7. **Host codec placement.** The codec runs inside `exchange` on the calling thread. If W1's decode gain is small,
   a profile of `exchange+add` per segment against the bytes saved will show whether to encode the local sum while the
   second rank finishes, or to overlap encoding of the second slice with the first slice's send.
8. **Follow-ups that need the kernel, the driver or hardware** (docs/tp8-optimization.md): USB4STREAM receive busy
   polling from linux-next (X1; measure `tbstream-bench pingpong` first), RADV's L2 writeback and invalidate at every
   GFX6 compute barrier (X2; a patched `libvulkan_radeon.so` through `VK_ICD_FILENAMES`), the sixth cable (X3), FFN
   columns shifted from the 5-7% slower GPUs (B1), and pre-submitted decode graphs (S1).

## Files

- `tools/tprun.py`, `tools/tp8.sh`: one TP8 run over the four Macs (from results/qwen/tp8-20261007, with
  `TP8_ROOT`/`TP8_MODEL`/`TP8_MTP` and `--profile`).
- `tools/ab.sh`: the runs of sections 3-6, by phase. `tools/kcheck.sh`: section 2. `tools/summarize.py`: tables of
  rates, host phases, MTP costs, token equality and clocks per configuration.
- `lavapipe-kernel-check.jsonl`, `lavapipe-device.txt`, `cargo-test.txt`: the off-hardware checks above.
