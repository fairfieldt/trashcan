# Recovered unfinished investigation lenses

Read on 2026-10-03 before continuing on main at `82e559f`. The only pre-existing
unstaged addition was `../investigation-20261003/`; tracked sources were clean.
These are recovered prototype results, **not newly accepted engine changes**.

All five transcripts belong to Claude session
`2a9622d1-2f28-4050-a00a-8bbc22541662`, started 09:19:58 PDT and interrupted at
10:46:40 PDT. The original JSONL files also exist under
`~/.claude/projects/-Users-fairfieldt-dev-trashcan/<session>/subagents/`.
The saved copies read here are in `../investigation-20261003/agent-reports/`.
The private sources and raw results are at
`/Users/fairfieldt/dev/claude-work/trashcan-20261003/<lens>/` and
`incite:~/trashcan-exp/<lens>/`.

| Lens | Transcript | Original agent ID | Main raw results read |
|---|---|---|---|
| Schedule | `sched-incomplete.txt` | `aecedd64b74a77df6` | `sched/results/{micro-1,overhead-1,hops-1,attn-old,attn-par}.jsonl`, baseline/item traces |
| Verification | `verify-incomplete.txt` | `a95a88f7e22a8e247` | `verify/results/{sweep1,sweep2}.jsonl`, `probe1.json`, `verify-b/results/sweep3.jsonl`, `verify/analysis/{itemize,final_model}.txt` |
| Q4 | `q4-incomplete.txt` | `aad95047644efa5b4` | `q4/results/lab{1,2,3,4,5}*.jsonl`, ISA counts, `q4-app/results/val-*` and `app-val-a.out` |
| Head | `head-incomplete.txt` | `a6307a871012c40c6` | `head/results/{head2,head3,small1,small2,small3,topk2}.jsonl`, binary hashes and build logs |
| Platform | `platform-incomplete.txt` | `a43aa44597e62ec3a` | `platform/results/batch{1,2,3}/`, including micro/full JSONL, clock ramp, CPU frequency and ioctl data |

## Schedule

- `micro-1.jsonl`: load-hoisted norm 66.7 -> 24.2 us, both GPUs/two
  repetitions; all three output vectors matched on the supplied synthetic inputs.
  A merge with independent 64-element slices and eight-partition load batches
  reduced 32-partition merge from 78.6 -> 15.7 us (dim256) and
  143.4 -> 16.5 us (dim512), with all elements matching.
- Stacked QKV: 73.45 -> 62.36 us sliding, 75.69 -> 68.47 us global.
  Fused gate/up/GEGLU: about 202.45 -> 197.53 us. These are micro results,
  with no completed full-model acceptance for these changes.
- A dependent tiny dispatch costs about 2.95 us with a full barrier, versus
  1.25 us with an execution-only barrier. The 0.03 us no-barrier result
  measures pipelined independent work, not a safe dependent graph.
- The **multi-job event relay caused the 10:26:47 and 10:27:55 GPU hangs**.
  `sched-pre.sh` and `sched.rs` document that amdgpu blocks the 33rd in-flight
  submission while the CPU relay has not started. Do not rerun `TC_EVENTS`.
  The single-job event micro had ~3.9 us median gaps versus ~25-40 us for
  semaphores, but asymmetric long tails; `TC_EVENTS1` was never accepted in
  the full application. Keep the production semaphore schedule.

## Verification

- Four-column FFN gate: shipped ~294.3 us, load-first `exp_f4` ~221.8 us,
  `exp_f4c` ~216.4 us. Two-column specialization ~134.2 versus 224.2 us;
  three-column ~172.5 versus 258.0 us. Two GPUs/two repetitions.
- `verify-b/sweep3.jsonl` has 148 passing records. LDS reduction reaches
  ~204.4 us (`exp_f4l`) or 198.3 us with a changed dot-product chain
  (`exp_f4cl`); head four-column ~3246.5 versus 6725.6 us. The measured
  LDS variants omit barriers. The barrier-conformant `*lb` variants and
  wider global shape follow-up were queued, not measured. Do not port the
  unbarriered source as an accepted result.
- `sweep2` has 26 failed records, including shipped kernels on later shapes.
  Transcript entry 207 fixes freed weight buffers whose recycled Vulkan
  handles hit stale cached descriptor sets. Treat the failed global/head
  rows and the unvalidated `probe1` candidate full-model rates as contaminated.
  A reference/greedy validation of final selection B did not finish.
- Baseline `probe1`: forward 1/2/3/4 columns ~40.1/65.2/73.3/81.6 ms;
  FFN segment sum 18.1/33.5/38.7/44.0 ms. The eight-prompt policy model
  predicts 1.128x for fixed depth 3 and 1.199x for a tuned margin rule on
  the old costs. Thresholds were tuned on that data; they are simulations,
  not independent live validation. The candidate scenario's 1.5x claims
  cannot be carried over to a faster ordinary decoder.
- The self-requeue helper is now `verify/tools/runq.sh.disabled`. No verify
  job or requeue loop was running on incite at the start of this follow-up.

## Q4

- **Row interleaving was prototyped**, contrary to the early README summary.
  The corrected harness's `lab5-1-103925.jsonl` reports gate ~104.8 ->
  93.1 us with interleaved four-row/f16 scales (`x_il4h`), and ~90.0 us
  with one-wave LDS-transposed reduction (`x_t4h`). Down ~100.2 ->
  93.5/90.5 us. The 89.3/89.9 us chained variants change arithmetic order.
  `lab5-2` uses randomized activations and explicitly shows chained variants
  are not bit-identical. Some other interleaved kernels also reformulate
  signed dequantization as unsigned dots minus a bias; validate real
  activations independently rather than assuming algebra implies identity.
- `q4-app/app-val-a.out` records passing b1/b4/b15 reference checks and
  identical selected dumps for the `il4h` format. Full-model performance
  and chat-parity jobs never completed; no format was accepted on main.
- The **10:05:52 GPU1 reset is attributed by the Q4 transcript to the lab
  descriptor-cache lifetime bug** (entry 246). Early lab3 results can bind
  freed/recycled weight buffers. Lab4/lab5 keep weight buffers alive and
  include CPU checks and output hashes; prioritize those later results.

## Head and small kernels

- `head3.jsonl`: int8-expanded head ~2901.5 us; packed signed six-bit
  groups plus f32 scale ~2318.3 us, 629,145,600 -> 503,316,480 bytes/GPU.
  A tighter Q6 block layout is ~2255.1 us and 416,808,964 bytes/GPU.
  Both preserve all sampled logits bit for bit in this micro run. The
  alternative wide grouping changes values. No complete app validation
  or benchmark of either format ran; multi-column head support must be
  accounted for before changing the stored representation.
- `small3`: fused norm ~66.68 us, scalar load-hoisted `norm3840` ~22.29,
  vector `norm3840v` ~18.63. **The vector variant changes reduction order**
  despite matching the chosen synthetic input. The 960-thread budget norm
  also changes the reduction. Follow-up instead validates the scalar
  load-hoisted reduction on random and real inputs.
- Top-k/merge at k=1 ~15.1/11.0 us; k=64 ~260.1/251.2 us. The application
  already selects k=1 for greedy decoding. Fixed-dimension norm/RoPE and
  direct shader P2P pushes remain unaccepted leads.

## Platform

- `batch1`: execution-only and omitted barriers produce wrong values in
  a dependent chain. Cache-aware barriers passed this micro check.
- `batch3/full.jsonl`: dispatch mode `2:0` (cache-aware dispatch barriers,
  original segment-edge barriers) takes tg256 sweep rates ~23.67/23.87 ->
  24.49/24.63 tok/s, matching generated IDs. The attention segment sum
  falls ~18.02 -> 16.60 ms. Mode `2:5` gives ~24.77/24.53. Earlier edge
  mode 3 produced different IDs and is rejected. Full reference/P2P
  acceptance of the later safe-looking modes was queued but never ran.
  Micro correctness is insufficient evidence to remove graph barriers.
- Removing timestamps, switching graphics/compute queues, forcing local
  BO lists, adding a CPU spinner, or introducing asymmetric work gave no
  reliable application improvement. Pinning to IRQ cores or enabling
  RADV's submit thread made the micro schedule worse.
- Cold-start clocks ramp to 850/1370 in ~260 ms. Submission gap distributions
  have ~2.6 and ~4.1 ms/token regimes; CPU/IRQ placement changes variability
  rather than establishing a new throughput target.

## Disposition

First independently validate the scalar load-hoisted norm and wide merge.
Then prioritize corrected Q4 interleaving and barrier-conformant small-batch
verification. Packed head and tiled prefill remain candidates. Preserve
per-GPU clocks, exact source/binary hashes, same-session baselines, reference
checks and greedy parity for every accepted change. No recovered micro number
is a hardware ceiling or an accepted full-model rate.

The [validated follow-up](README.md) completes independent acceptance and
application measurements for scalar fused norm and wide attention merge.
