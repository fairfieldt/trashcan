# Performance investigation — 2026-10-03 (unverified leads, nothing integrated)

Nine investigators examined the engine at `82e559f` (binary `5b62aa40…`) in parallel on incite:
goal/time-budget audit, two structural lenses (schedule/graph, representation/algorithm), five kernel
families (Q4 decode, decode attention, small-batch verification, prefill, head and small kernels) and
platform. **Every number below is a prototype or micro-benchmark result from a private source copy.
Nothing here has been integrated, re-measured independently, or run through `scripts/test-engine.sh`.**
The repository source and `incite:~/trashcan` are unchanged.

The session was cut short. Only the representation investigator delivered a final report before this
file was written; the other lenses are summarized from their raw result files. Sections say which.

## Late reports: budget and attn lenses (arrived after the rest of this file was written)

Full texts are in `agent-reports/`. They supersede the estimates further down where they differ.

- **Full-model result (budget lens, one paired run, same binary, env toggles):** norm prototype
  (`norm4.comp`, 960 threads) plus wide attention merge (`attention_merge_wide.comp`) took tg256 from
  **23.32 to 28.88 tok/s (+23.9%)**, decode timeline 42.19 → 33.86 ms, pp512 unchanged (161.3 → 161.9).
  One repetition only. **Not validated against the reference or greedy text**; the micro bit-identity
  check used inputs that are multiples of 1/16, which does not establish equality on real activations.
  Files: `budget/results/app-base-c.json`, `app-cand-c.json`, `trace-cand-c.tsv`.
- **Itemized token (budget lens, in-app trace, ms):** FFN matvecs 14.29, fused norm 6.77, sliding
  attention matvecs 4.68, attention merge 4.30, attention partial 3.54, gaps 2.96, head 2.90, global
  attention matvecs 1.35, norm+RoPE 0.63, P2P push 0.50, GEGLU 0.24.
- **The attention merge, not the partial, is the main attention cost** (both lenses): 78.6 µs per
  sliding layer and 143.4 µs per global layer at 32 partitions, from serial load-and-wait chains.
  The wide merge brings it to about 18 µs in the app. Partition thresholds were tuned with the old
  merge and need a re-sweep. The "attn-wide" rows below are this variant.
- **Goals (budget lens):** near-term 28–29 tok/s tg256 once the two prototypes are validated; 31–32
  with attention projections and partial; 35 is a stretch needing every Q4 matvec at ≥200 GB/s; 40–50
  is not reachable for ordinary decode. With the fixes, MTP falls to about 1.01× ordinary unless
  four-column matvecs get about 1.4–1.8× faster; M5 is better restated as an absolute rate.
- **Stale-binary hazard in `exp.sh build`:** `rsync -a` preserves mtimes, so a file edited before the
  previous remote build finished may not be recompiled. Some lenses may have measured stale binaries;
  touch sources or check the binary hash before trusting a prototype number.
- **attn lens:** only its first sweep ran; `merge_flat`, `norm_flat` and later shaders are unmeasured
  and uncompiled. Its two-pass kernel: 768 keys sliding 164.0 → 90.1 µs, global 226.7 → 143.4 µs;
  worse at 1 key. Not additive with the merge fix.

## Where the evidence is

- Local: `~/dev/claude-work/trashcan-20261003/<lens>/` (`tree/` private sources, `results/`, `changes.patch` where saved)
- incite: `~/trashcan-exp/<lens>/` (same trees, builds, `results/`, clock logs under `results/clocks/`)
- Lenses: `budget sched repr q4 q4-app attn verify verify-b verify-c prefill head head-b platform`
- Helper scripts: `~/dev/claude-work/trashcan-20261003/bin/exp.sh` (private copy, build, locked run),
  `locked-run.sh` (one global lock plus clock sampling), `cancel-queued.sh`

Baseline reproduced through the helper before the investigation, both GPUs at 850/1370 MHz:
FFN gate 105.1 µs (175.4 GB/s repacked); decode attention at 768 keys / 32 partitions 164.2 µs
(sliding) and 226.6 µs (global).

## How the goals were calculated, and what that implies

- 65.10 tok/s = 4,040,294,400 repacked matrix bytes per GPU per token ÷ 263.04 GB/s. It counts weight
  streaming only. 51.57 tok/s adds 96 × 42 µs of hops. M3's 40–50 tok/s sits just under these.
- Measured token (tg256 after pp512, `main-tuning-20261003/fixed-dim/decode-fixed-2.tsv`): 42.4 ms =
  attention segments ~18.7 ms, FFN segments ~18.1 ms, head ~3.0 ms, gaps ~2.5 ms.
- The decode FFN mat-vec is **instruction-bound, not bandwidth-bound** (repr lens, micro + ISA): gate
  105.25 µs as shipped, 82.95 µs with cache-resident weights, 102.26 µs without activation loads; down
  100.16 / 83.24 / 97.11 µs. Weight streaming is about 21% of the kernel, activation re-reads about 3%.
  The inner loop is 129 vector instructions per iteration, 72 of them essential.
- Consequence: the weight-only bound is not the binding limit for FFN, and 40–50 tok/s for ordinary
  decode is not reachable by kernel tuning. A defensible near-term target is about 30 tok/s (estimate
  below); 40+ has to come from MTP with much cheaper multi-column verification.

## Itemized decode overhead (budget lens, raw micro results, two runs, both GPUs)

`budget/results/overhead-a-{1,2}.jsonl`, `overhead-b-{1,2}.jsonl`

| Item | µs |
|---|---:|
| Empty dispatch + barrier (1 workgroup) | 2.43 |
| Empty dispatch, no barrier | 0.03 |
| Empty dispatch + barrier, 480 / 8192 workgroups | 6.03 / 79.4 |
| **Fused norm (peer sum + post-norm + residual + next norm), as shipped** | **66.4** |
| Plain norm | 31.9 |
| norm_rope × 3, sliding / global | 11.0 / 18.5 |
| GEGLU | 4.5 |
| P2P copy of 15,360 bytes to the peer | 4.0 |

The fused norm runs 96 times per token: about 6.4 ms, 15% of the token, to normalize 3,840 floats in
one 256-thread workgroup. It closes the previously unexplained remainder of each segment (about 92 µs
per attention segment, 67 µs per FFN segment). Dispatch count itself is not a problem.

## Leads, in order of expected decode gain

1. **Fused norm restructure.** Budget lens `norm4_fused` 13.2 µs, reported bit-identical to the
   shipped kernel (`norm_vs_norm4 elements_not_bit_identical: 0`). Head lens variants 18.6–24.8 µs
   (`head/results/small3.jsonl`; `norm3840v` 18.6 µs bit-identical, `norm1024` 24.8 µs differs at
   1.6e-7). Estimate: about 50 µs × 96 ≈ 4.8–5.1 ms per token, roughly +13%. Micro only.
2. **Decode attention reformulated key-parallel / two-pass.** Three lenses reached it independently.
   - attn lens, 1,024 keys (`attn/results/plan1-b.jsonl`): sliding 191.0 → 101.3 µs (`tp_e1p16`);
     global 250.0 → 198.5 µs (`tp_h4e1p16`); CPU reference check passes.
   - repr lens (`repr/results/round3/attn.jsonl`): sliding 768 keys 164.2 → 114.3 µs, 1,024 keys
     188.7 → 126.1 µs; global 768 keys 226.7 → 161.9 µs, 4,096 keys 526.4 → 380.1 µs. It **loses**
     below 256 keys (1 key 43.6 → 67.3 µs) and at 8,192 keys (746.6 → 1,303.7 µs), so selection must
     be by visible length.
   - budget lens (`budget/results/attn-wide-b.jsonl`): 768 keys 102.0 µs sliding, 100.6 µs global;
     variant not characterized by me.
   - Estimate from the repr numbers at ~770 keys: 2.5 ms per token, about +6%; more at long context,
     nothing for short chats. Arithmetic order changes, so reference tolerance and greedy parity must
     be checked in the full model; the 15-token `validate` prompt never reaches this path.
3. **LM head with 6-bit packing** (`head/results/head3.jsonl`): 2,898 → 2,316 µs with bit-identical
   logits (`headvec6`, 503 MB instead of 629 MB); `headvecq6` 2,253 µs. About 0.6 ms per token, +1.4%.
4. **Q4 row-interleaved layout (not prototyped).** Storing a wave's four rows adjacently so one
   16-byte load fetches four packed words, and precomputing the bias per block: estimated 129 → 95
   instructions, about 105 → 80 µs per FFN projection, 3.6 ms per token. Low-to-medium confidence.
5. Small: f16 scale stream (gate −4.0%, down +1.5%, bit-identical; about +1% if applied to gate/up
   only); fused gate + up + GEGLU dispatch (202.6 → 197.8 µs, +0.5%).

Combined estimate for 1–3 if they carry into the full model: about 9 ms off 42.4 ms, roughly
30 tok/s. This is arithmetic from micro-benchmarks, not a measurement.

## MTP and verification

- Forward cost by column count, full model, three chats (repr lens, `repr/results/mtp-20261003T095813/`):
  1 col 38.2–40.0 ms; 2 cols 63.2–63.6; 3 cols 71.3–72.4; 4 cols 79.6–81.1; 5 cols 107.5–108.9;
  8 cols 132.7–133.8. The second column costs about +25 ms, columns 3–4 about +8.2 ms each, the fifth
  about +28 ms.
- Four-column FFN gate is instruction-bound: 293.0 µs shipped, 258.5 µs with cached weights,
  201.7 µs without activation loads.
- Prototype small-batch kernels (verify lens, `verify/results/sweep2.jsonl`, FFN gate, GPU0):
  2 cols 223.8 → 134.2 µs; 3 cols 257.7 → 172.4 µs; 4 cols 293.8 → 216.6 µs; 8 cols 546.0 → 412.4 µs.
- Simulated on the probe data (repr lens, `repr/analysis/mtp_scenarios.txt`), MTP speedup over ordinary:
  current costs 1.164× fixed-3, 1.239× confidence-gated depth, 1.565× oracle; with the second column
  at +8.2 ms 1.40× / 1.515× / 1.96×. M5 (1.5×) is out of reach by policy alone and needs the
  multi-column base cost removed. The ratio shrinks as ordinary decode gets faster unless verification
  improves with it.
- Confidence-gated depth (stop when drafter top-1 probability < 0.9): 1.164× → 1.239× pooled,
  simulated, thresholds tuned on the same three prompts.

## Prefill

Prefill lens, full-model run with a tiled attention prototype (`prefill/results/app512-tiled.json`):
pp512 175.7 tok/s against about 160 recorded. The lens's same-session baseline
(`app512-scalar.json`) and its correctness checks were not reviewed before this write-up.

## Negative results

- Weight format: no slack for a different quant. Nibble entropy 3.56–3.78 bits of 4; about 100 distinct
  scales per 120-block row (`repr/results/wstats.json`). Fewer bits only shrink the ~21% streaming
  share; int8 doubles traffic.
- Sharing activations through LDS: 250.8 µs (256 threads) and 122.5 µs (1,024 threads) against 105.25.
- Explicit byte extraction: identical ISA.
- Head (analysis only): excluding unused vocabulary rows saves at most 0.07 ms; truncated-weight
  screening prunes nothing; CPU drafting would cost at least 21 ms per draft against 4 ms.
- Fusing dispatches only to reduce their count: 2.4 µs each.

## Problems during the run

- Three GPU resets from experimental shader variants (ring `comp` timeouts at about 10:06, 10:27 and
  10:28 PDT). The driver recovered and later runs reached 850/1370 MHz. The responsible variants were
  not identified; the repr lens's lock windows exclude it. Any variant from the `q4`, `sched`,
  `verify-b` trees needs care before reuse.
- The single GPU lock queued about 30 full-model runs. 27 were cancelled before starting
  (`~/dev/claude-work/trashcan-20261003/queue-snapshot-1040.txt` lists them); their scripts remain in
  each lens's `scripts/` directory.
- Platform, schedule, Q4 and prefill lenses have result files that are not summarized here.

## Next steps

1. Re-measure leads 1–3 independently, one at a time, with a same-session baseline.
2. Integrate the norm restructure first (largest, bit-identical variant exists), then run
   `scripts/test-engine.sh`, the boundary checks and the fixed pp512/tg256 benchmark.
3. Integrate length-selected key-parallel attention; validate with `check-batches` and greedy chat
   parity above 256 tokens of context.
4. Integrate the 6-bit head.
5. Port the verify lens's small-batch kernels and re-run the three MTP chats; then try gated depth.
6. Read the unreviewed result files (platform, sched, q4, prefill) and identify the hanging variants.

## Late report: prefill lens (full text in `agent-reports/prefill.txt`)

- **Tiled batched attention, measured in the full model (2 runs each):** pp512 161.03 / 161.04 →
  176.17 / 175.75 tok/s (+9.3%); timeline 3112.6 → 2848.0 ms. Micro: 5.8–10.1× on the attention kernel.
  pp4096 is estimated at 165–172 tok/s (from 117), not run. `validate` and `check-batches` with the
  tiled kernel never ran, so correctness and greedy identity are unproven. It adds a 128 MiB score
  buffer per GPU at 8K context.
- **pp512 chunk itemized:** FFN GEMMs 63%, attention projection GEMMs 19.5%, `attention.comp` 10.2%,
  P2P copy 5.5% (172 ms, about 4.3 GB/s, twice the documented estimate), everything else 2.5%.
- The GEMM runs at 64% of peak, matching its instruction mix; LDS stride is a first-order effect
  (144 → 128 bytes is 2.7× slower). Remaining GEMM headroom is estimated at about 10%.
- Short prompts pay for a full 64-token tile (26 tokens cost the same as 64).

## Lenses without a final report

`sched`, `verify`, `q4`, `head` and `platform` were stopped before reporting. Their raw results and
private trees are in the evidence directories above; `agent-reports/*-incomplete.txt` are their
transcripts. The verify lens has a self-requeueing helper (`verify/tools/runq.sh`); check that nothing
of it is still running on incite before the next session.
