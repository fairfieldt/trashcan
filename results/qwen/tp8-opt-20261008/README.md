# Qwen3.8-27B TP8 optimization round (2026-10-08): runbook

The plan, cost model and design are in [docs/tp8-optimization.md](../../../docs/tp8-optimization.md). This round was
implemented and checked without access to the Macs; the runbook below says how to test it on incite, invite, enlight
and indite, what to expect and how to decide. It ran on 2026-10-09: the results come first.

Branch `claude/wizardly-planck-823ef4` (AGENTS.md asks for work on `main`; this session was given that branch, so merge
it to `main` with the measurements once they are recorded). Source of the off-hardware checks below: 302a18f.

## Results on the Macs (2026-10-09)

**Outcome.** On the D700s every new kernel gives the bits it claims but runs slower than the kernel it replaces, and
none is faster end to end; the fused host pass (H1) is slower too. All of them are removed. W1 makes decode 3.6% and
MTP 2.1-2.4% faster but made prefill 5% slower, so it now covers only all-reduces below prefill size, where it is the
default. With that, TP8 decodes at 36.2 tok/s (record 35.0) and 47.0 with MTP (record 46.0); prefill is unchanged.

Tested: d76f73b (`trashcan-opt`, this round as merged) against 261d453 (`trashcan-base`, the TP8 record), then
`trashcan-eval`: the record with W1 below prefill size, the commit that records these results. Cross-built with
`scripts/cross-build.sh`; kernel 7.2.9-p2p on incite, invite, enlight and indite over the four cables of
results/qwen/tp8-20261007 (a sixth cable was installed for results/qwen/tp8-mesh-20261009; these runs do not use it).
gpuq leases, the performance governor and the 0 µs latency request on all four; every card ran at 850/1370 MHz while
computing (`clocks/`). Each A/B alternates single runs of `qwen-bench --pp 512 --tg 128 --reps 2` (with `--mtp` for
MTP), two runs a side (`tools/h1.sh`, `dec.sh`, `rest.sh`, `eval.sh`; tables from `tools/pairs.py` and
`mtppairs.py`). tok/s are per repetition; GPU wait and exchange are incite's milliseconds per generated token. The
generated tokens were the same in every run of every configuration.

### Kernel checks (`tools/kcheck.sh trashcan-opt 200`, all eight GPUs)

All 28 cases pass on every GPU. ACO kept the bitwise claims: top-k, FFN, DeltaNet and the many-rows matvec are
identical to the kernels they replace, and decode attention is within 5e-7 of the split-K kernels. Medians over the
eight GPUs (each within a few percent):

| kernel | case | against the old kernel | old µs | new µs |
|---|---|---|---|---|
| A1 attention | g 3, kv_heads 1, n 1, parts 64, pos 520 | within 2.1e-07 | 101 | 116 |
| A1 attention | g 3, kv_heads 1, n 3, parts 64, pos 520 | within 1.5e-07 | 103 | 107 |
| A1 attention | g 6, kv_heads 1, n 1, parts 64, pos 600 | within 3.3e-07 | 107 | 149 |
| A1 attention | g 6, kv_heads 2, n 2, parts 16, pos 37 | within 3.3e-07 | 45 | 78 |
| A1 attention | g 6, kv_heads 4, n 1, parts 64, pos 1500 | within 3.6e-07 | 321 | 397 |
| A1 attention | g 3, kv_heads 1, n 1, parts 128, pos 8000 | within 1.9e-07 | 395 | 323 |
| A1 attention | g 2, kv_heads 1, n 4, parts 32, pos 5 | within 2.4e-07 | 39 | 56 |
| A1 attention | g 4, kv_heads 1, n 8, parts 32, pos 300 | within 4.8e-07 | 95 | 100 |
| K1 top-k | cols 1, rows 31040 | identical | 217 | 228 |
| K1 top-k | cols 1, rows 8192 | identical | 146 | 153 |
| K1 top-k | cols 3, rows 31040 | identical | 231 | 243 |
| K1 top-k | cols 2, rows 1000 | identical | 75 | 79 |
| F1 FFN | k 5120, n 1, rows 2176 | identical | 91 | 100 |
| F1 FFN | k 5120, n 3, rows 2176 | identical | 141 | 157 |
| F1 FFN | k 5120, n 2, rows 4352 | identical | 193 | 196 |
| F1 FFN | k 5120, n 4, rows 1088 | identical | 114 | 153 |
| G1 DeltaNet | k_heads 2, n 1, pos 10, snapshots False, v_heads 6 | identical | 41 | 53 |
| G1 DeltaNet | k_heads 2, n 3, pos 10, snapshots True, v_heads 6 | identical | 56 | 71 |
| G1 DeltaNet | k_heads 4, n 2, pos 0, snapshots True, v_heads 12 | identical | 48 | 60 |
| G1 DeltaNet | k_heads 16, n 1, pos 5, snapshots False, v_heads 48 | identical | 77 | 106 |
| G1 DeltaNet | k_heads 2, n 8, pos 3, snapshots True, v_heads 6 | identical | 97 | 120 |
| M1 matvec | k 768, kernel qwen_matvec_rows8, rows 5120 | identical | 28 | 30 |
| M1 matvec | k 768, kernel qwen_matvec_rows16, rows 5120 | identical | 28 | 47 |
| M1 matvec | k 1536, kernel qwen_matvec_rows8, rows 5120 | identical | 36 | 43 |
| M1 matvec | k 1536, kernel qwen_matvec_rows16, rows 5120 | identical | 36 | 55 |
| M1 matvec | k 5120, kernel qwen_matvec_rows8, rows 2176 | identical | 51 | 66 |
| M1 matvec | k 5120, kernel qwen_matvec_rows16, rows 2176 | identical | 51 | 88 |

Only the decode attention at 8000 keys beats the old kernels. The fused and reshaped kernels run fewer, longer waves than
the dispatches they replace, which likely hides less memory latency on GCN; lavapipe (8-lane subgroups, run one after
another) checked their arithmetic but could not show their speed.

### Decode

| candidate | n | default tok/s | candidate tok/s | change | GPU wait ms/token | exchange ms/token | pp512 tok/s |
|---|---|---|---|---|---|---|---|
| merged defaults (H1 on) vs the record | 4+4 | 34.60 (34.42-34.77) | 33.66 (33.35-33.92) | -2.7% | 19.40 → 19.80 | 7.72 → 7.49 | 201 → 203 |
| H1 off | 4+4 | 33.73 (33.28-33.93) | 34.55 (34.51-34.62) | +2.4% | 19.80 → 19.78 | 7.72 → 7.60 | 204 → 202 |
| W1, every all-reduce | 4+4 | 33.65 (33.32-33.97) | 34.85 (34.75-34.91) | +3.6% | 19.79 → 19.80 | 7.85 → 6.73 | 204 → 193 |
| A1 | 4+4 | 33.50 (33.03-33.88) | 33.74 (33.67-33.80) | +0.7% | 19.78 → 19.91 | 7.98 → 7.63 | 203 → 203 |
| K1 | 4+4 | 33.96 (33.81-34.10) | 33.81 (33.79-33.83) | -0.4% | 19.79 → 19.80 | 7.57 → 7.68 | 202 → 204 |
| F1 | 4+4 | 33.95 (33.87-34.12) | 33.01 (32.88-33.10) | -2.8% | 19.78 → 20.38 | 7.54 → 7.55 | 204 → 204 |
| G1 | 4+4 | 33.41 (32.51-33.91) | 32.83 (32.59-32.99) | -1.7% | 19.78 → 20.42 | 8.02 → 7.91 | 193 → 203 |
| M1 (16 rows) | 4+4 | 33.48 (33.17-33.81) | 32.23 (32.13-32.32) | -3.7% | 19.80 → 21.34 | 7.96 → 7.61 | 206 → 205 |

The merged binary with its defaults was 2.7% slower than the record: 0.55 ms per token more host time with H1's fused
pass (`total` 1.42-1.52 against 0.91-0.93 ms with `TRASHCAN_TP_HOSTPASS=0`) and 0.4 ms more GPU time, which H1 does not
explain and which came with the kernel commits: with their switches off they still upload G1's per-layer convolution
buffer, enlarge the attention scratch and register the new pipelines. With H1 and the kernels removed, the binary
matches the record:

| candidate | n | default tok/s | candidate tok/s | change | GPU wait ms/token | exchange ms/token | pp512 tok/s |
|---|---|---|---|---|---|---|---|
| back-out binary vs the record | 4+4 | 34.99 (34.61-35.24) | 34.94 (34.91-35.01) | -0.1% | 19.41 → 19.42 | 7.63 → 7.58 | 203 → 201 |
| W1 below prefill size | 4+4 | 34.94 (34.91-35.01) | 36.20 (35.96-36.46) | +3.6% | 19.42 → 19.42 | 7.58 → 6.58 | 201 → 203 |

A1's pairs disagree (the first favoured it, the second the default) while incite's GPU time rose 0.13 ms; at a
520-token context it is not faster.

### MTP (k = 2, 64K draft head)

| candidate | n | default tok/s | candidate tok/s | change | verification ms/round | tokens/round |
|---|---|---|---|---|---|---|
| W1, every all-reduce | 4+4 | 44.76 (44.62-44.86) | 45.82 (44.47-46.30) | +2.4% | 52.1 → 50.8 | 2.47 → 2.47 |
| A1 | 4+4 | 44.54 (44.40-44.70) | 44.62 (44.58-44.65) | +0.2% | 52.4 → 52.3 | 2.47 → 2.47 |
| back-out binary vs the record | 4+4 | 42.31 (31.04-46.17) | 46.01 (45.63-46.24) | +8.7% | 55.9 → 50.6 | 2.47 → 2.47 |
| W1 below prefill size | 4+4 | 46.01 (45.63-46.24) | 46.97 (46.76-47.17) | +2.1% | 50.6 → 48.5 | 2.47 → 2.42 |

The record's first repetition (31.0 tok/s) hit a 16 ms/token exchange stall; its other three average 46.1. With W1
below prefill size the generated text is the same and one fewer draft was accepted over the 128 tokens (2.42 instead of
2.47 tokens per round); W1 on every all-reduce kept 2.47.

### Prefill

W1 on every all-reduce took the 512-token prefill from 204 to 193 tok/s: its exchanges took 2.22-2.30 s per prompt
against 1.92-2.06 in f32, although they move half the bytes. Its f16 reduce-scatter/all-gather spawns threads for each
of about twenty encode and decode passes per all-reduce and allocates its buffers per call, and the bytes saved do not
pay for that. Prefill-sized all-reduces therefore stay f32 (`trashcan-eval`: 201 against 203 tok/s).

### Long context

Not measured. Forty-five seconds into the first 7680-token prefill (`long-default-1`) the Macs' power strip tripped,
at about 1.65 kW drawn by the four (results/qwen/tp8-mesh-20261009, Power). A1, the one candidate expected to gain
there (the kernel check's 8000-key case: 323 against 395 µs per layer), is removed with the others; 90d044f and
302a18f hold it for a long-context A/B once the Macs have the power for it.

### Correctness gates (`trashcan-eval`, `tools/eval.sh gates`)

| gate | f32 | W1 below prefill size |
|---|---|---|
| `qwen-validate` (14-token prompt) | every tensor identical to the record's dump (`tp8-rsag-val`); against the CPU reference worst 4.02e-4, logits top-10 10/10, greedy 10/10 | the same dump (the prompt's all-reduces are prefill-sized, so f32) and the same 10 greedy tokens ([val-w1-vs-ref.txt](val-w1-vs-ref.txt)) |
| `qwen-check --mtp` | pass: restore exact, speculative output identical, 83% of drafts accepted | pass: the same, 83% |
| `qwen-check --verify 3` | pass: no column mismatches, column score error 2.25e-4 (the record's) | pass: column score error 0, since each verification column is added and rounded like its decode step |

The runbook's plain `qwen-check` stops at once (it needs `--verify` of at least 2); it is gone from the tools.

### Decisions

- **H1** removed; the record's host passes are back.
- **W1** kept for all-reduces below prefill size (`PARALLEL_MIN`, 65536 values: decode, verification and draft steps)
  and on by default; `TRASHCAN_TP_WIRE=f32` (driver only) keeps exact f32 sums for bitwise comparisons and validation.
  `f16:N` and the f16 reduce-scatter/all-gather are gone.
- **A1, K1, F1, G1, M1** removed with their shaders, pipelines, switches and `qwen-kernel-check`.
- `Buffer::drop` keeps evicting the descriptor sets cached by its handle (a fix that came with 90d044f).

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
  `TP8_ROOT`/`TP8_MODEL`/`TP8_MTP` and `--profile`). Since 2026-10-09 it refuses to start while an engine process of
  an earlier run is alive, and stops every node after the run (a node left running served the next run's session
  with the old run's settings once).
- `tools/ab.sh`: the runs of sections 3-6, by phase. `tools/kcheck.sh`: section 2 (`qwen-kernel-check` was removed with
  the kernels; 90d044f has it). `tools/summarize.py`: tables of rates, host phases, MTP costs, token equality and clocks
  per configuration.
- The runs of 2026-10-09: `tools/h1.sh` (record, defaults, H1 off), `tools/dec.sh` (each candidate against the
  default), `tools/rest.sh` (MTP, long context, gates of the merged binary), `tools/eval.sh` (the back-out binary);
  `tools/pairs.py` and `tools/mtppairs.py` make the tables. Their outputs per Mac in `incite/`, `invite/`, `enlight/`,
  `indite/` (driver `.json`/`.err`, node logs, `clocks/`, `kcheck-*.jsonl`; `incite/commands.log` lists every run);
  `val-w1-vs-ref.txt` compares the W1 dump with the CPU reference.
- `lavapipe-kernel-check.jsonl`, `lavapipe-device.txt`, `cargo-test.txt`: the off-hardware checks above.
