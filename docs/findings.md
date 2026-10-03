# Findings log

Newest first. Numbers are from incite unless noted; raw outputs are under `results/`.

## 2026-10-03: measured MTP verification improvement

`results/engine/mtp-exploration-20261003/` records load-first Q4/Q6 multi-column
kernels and two/three-column specializations, preserving every output bit in
196 candidate matrix checks and fresh real-activation checks. Two runs per
prompt give pooled MTP **31.73 -> 39.96 tok/s (+25.9%)**, verification **75.77 ->
57.87 ms**, and MTP/ordinary **1.035x -> 1.306x**. Ordinary rates remain within
variation. The original 1.5x typical-chat target stays open.

The revised four-column kernel also beats the old eight-column Q6 head at
batches five through eight by 15.8–41.0%; production selects it there. Real
depth-four chat rates remain within variation. Depth one/two help explanation
prompts, while depth three wins CSV; keep the default policy. The existing
scalar 128-partition global attention path is 2.7x slower than selected shared
64 at 80K, so keep that selector too. Final acceptance includes six reference
paths, ring wrap/rollback, eight-column greedy64 and seeded stochastic parity,
with sustained 850/1370 MHz. Bounded-memory tiled prefill is the next structural
lead for the measured 80K workload.

## 2026-10-03: 80,000-token context measurement

`results/engine/context-80000-20261003/` records one fresh 4K/80K pair using the
validated `5925623` kernels plus optional CLI progress logging. Inputs are repeated
synthetic token IDs, batch512, followed by 128 forced greedy forwards; model loading
is outside the timers.

- 4K prefill/decode: 117.57 / 26.73 tok/s. At 80K: **28.75 / 12.35 tok/s**, with
  prefill taking **46m 22.5s** and decode 10.361s. The last full prefill chunk runs
  at 15.86 tok/s; global-attention bulk prefill cost grows with the causal prefix.
- Last-step GPU0 work grows 34.04 -> 67.00 ms, almost entirely from the eight global
  attention segments (4.01 -> 36.98 ms); the remaining GPU work stays near 30.03 ms.
  The last 64 decode steps give 13.81 tok/s, while 12.35 is the full 128-step average.
- Peak late physical VRAM is about 5.62 GiB per card, so this ordinary workload fits.
  Card1's visible aperture reaches 256 MiB and GTT usage grows during graph recording
  for the final batch128 chunk and decode. Its contribution to transient timing needs
  a separate controlled test; this is not a clean demonstration of a context ceiling.
- All 157 prefill chunks and 128 decode steps report finite top scores and correct
  positions. Both GPUs sustain 850/1370 MHz and no new reset occurs. This is benchmark
  sanity, with no independent 80K reference logits or retrieval-quality evaluation.

## 2026-10-03: validated fused norm and attention merge

Evidence, source/binary manifests and reproduction are in
`results/engine/validated-followup-20261003/`. Its recovery report identifies and summarizes
all five unfinished Claude lenses: schedule, verification, Q4, head and platform. Unsafe or
unaccepted experiments remain separate from the accepted implementation.

- Scalar load-hoisted fused norm keeps the original arithmetic order and all three outputs.
  Its one-column micro time falls from 65.80 to 19.17 us; 64 randomized records match the
  generic norm bit for bit and pass an f64 CPU check.
- Parallel partition-header loading and one output element per lane cut 768-key/32-partition
  attention time from 164.2 to 102.1 us sliding and 226.5 to 100.7 us global. The unchanged
  partial shader isolates the merge gain; 200 final cases compare every output bit with the
  original serial merge, within a 400-case baseline/final sweep.
- Two runs per phase give pooled tg256 **23.323 -> 26.062 -> 28.723 tok/s (+23.16%)**.
  Prefill remains near 161 tok/s. Mean last-token per-GPU work falls from 39.86 to 31.62 ms;
  semaphore gaps vary, so this is a kernel improvement rather than a demonstrated transport gain.
- One paired pp4096/tg128 run raises decode **21.149 -> 26.389 tok/s (+24.78%)**, with
  prefill unchanged near 117 tok/s. That percentage has single-pair confidence.
- Final binary `2682f29e` passes six 132-tensor reference paths, 30 CPU tests, tokenizer/chat
  fixtures, 10,000 bidirectional P2P payloads, the 250-token partition boundary and the
  1700-token ring-wrap/verification/rollback check. Fresh baseline/norm/final b1 dumps have
  identical bytes for all 133 activation/logit files and 32 greedy tokens. Both D700s sustain
  850/1370 MHz during compute; complete clock logs include loading and idle periods.
- Three ordinary chats rise from 25.50–25.62 to 30.28–30.88 tok/s; MTP rises from
  29.34/24.70/37.15 to 31.75/26.44/39.09 tok/s, with all output bytes and proposal counts
  identical. These single pairs give a pooled MTP/ordinary ratio of 1.156x -> 1.031x.
  Verification remains about 76 ms/round, so a faster ordinary decoder makes M5 harder.
  A separate 1788-token prompt improves 21.73 -> 27.59 tok/s with identical output.

Fine partition thresholds, Q4 interleaving, conformant verification kernels and head packing
remain leads requiring separate application acceptance. Hardware bounds are unchanged;
neither M3 nor M5 is complete from these decode measurements.

## 2026-10-03: main-branch attention, prefix reuse and ceiling audit

The performance branch is merged; this project now works directly on `main` as recorded in
`AGENTS.md`. Evidence and reproduction are in `results/engine/main-tuning-20261003/`.

- Exact GGUF accounting gives 4,040,294,400 matrix bytes per GPU per decode token and
  11.92230912 GFLOP per GPU. At the measured 1370 MHz memory clock, the weight-only bound is
  65.10 tok/s. Adding the historical 96 × 42 us transport assumption gives 51.57 tok/s, an optimistic
  sensitivity calculation rather than a measured practical ceiling. Four-column verification is
  47.68923648 GFLOP per GPU. `results/engine/ceiling-audit-20261003/` retains the full audit.
- Unsigned Q4 unpacking and shape-specific wave geometry improve warmed FFN gate/down payload
  bandwidth to about 158/165 GB/s per GPU; narrow attention projections improve more. Adaptive
  attention uses fewer partitions for short prefixes and more for long global attention.
- Three ordinary 128-token chats improve by 3.1–4.5%, with identical text. The paired MTP results
  are 29.17/24.50/37.09 tok/s; their combined gain over ordinary decoding is about 1.17×, below M5.
  These are single paired chat runs, and skipping the final unused CLI forward contributes to the gain.
- Long-context decode improves from 17.90 to 19.08 tok/s (+6.6%); prefill remains about 117.4 tok/s.
  The repeated pp512/tg256 workload is unchanged within run variation: 161.23/21.970 baseline versus
  161.00/21.869 after tuning. Its -0.46% decode difference is not a demonstrated improvement.
- A following shared-KV iteration groups two sliding or eight global query heads, retaining scalar
  loads for the shortest prefixes. Against a fresh rerun of the preceding core, paired tg256 improves
  from 22.028 to 22.687 tok/s (+2.99%), with pp512 unchanged near 160. Ordinary short chats are
  essentially unchanged in this phase, while all twelve ordinary/MTP outputs remain identical.
  Decode after a 4096-token prefill improves from 19.154 to 20.370 tok/s (+6.35%); prefill stays near 117.4.
- Compiling the shared kernels for their selected dimensions (256 sliding / 512 global), with
  one K/V plane per group, reduces their measured time by 6.1–14.4%. A fresh paired full-model
  phase raises pooled tg256 from 22.779 to 23.415 tok/s (+2.79%) and long-context decode from
  20.552 to 21.359 tok/s (+3.93%). Prefill remains unchanged. All twelve chat outputs match;
  ordinary chat improves 1.3–4.0% and MTP 1.8–2.4% in these single pairs.
- Resident graph growth exposed a separate allocation cliff: each target graph pair consumes about
  9.1 MiB of visible command storage per GPU in the batch128 diagnostic. At the 256 MiB aperture limit, gaps between GPU segments
  grew to 98–142 ms per token despite unchanged shader times and clocks. A controlled clear restored
  greedy requests from 4.915/5.979 to 1.666/1.701 s with identical outputs. Later growth reproduced
  the delay even for stochastic requests. The target now evicts its least-recent graph before recording
  a fifth pair; clearing after the driver has grown its backing pools is insufficient.
- Resident serving reuses only exact token prefixes with surviving sliding history, including tracking
  speculative writes that were rejected. Cached-token usage is explicit and `--no-prefix-cache`
  permits cold comparisons. A repeated 1403-token prompt with 128 output tokens falls from 19.31 to
  6.25 s; a full-history follow-up falls from 14.37 to 1.08 s, with matching output in those comparisons.
- The batch512 default passes 72 resident requests across batch32/128/512 and speculative512, with
  all 108 cross-configuration comparisons exact. Against batch32, a 512-token input falls from 7.865
  to 3.194 s, and the repeated 1763-token input with 256 output tokens falls from 46.462 to 11.989 s.
  The latter includes retained-prefix reuse. Both default ordinary/MTP protocol suites pass; observed
  visible-VRAM peaks remain below 256 MiB, with only 7.176 MiB headroom on the fuller card with MTP.
  These request snapshots validate this workload, not arbitrary other clients or 32K execution.
- Six reference paths pass all 132 tensors, logits and ten greedy tokens. Tokenizer/chat fixtures,
  10,000 bidirectional P2P payloads, 26 CPU tests, partition-boundary and 2300-token cache-wrap checks
  pass. Ordinary and MTP HTTP suites cover prefix limits, JSON/SSE, tools and sampler changes.
  All sampled points in the seven approximate benchmark decode windows show both GPUs at 850/1370 MHz.
  The later shared-KV/cache-bound build separately passes all six paths, fixtures, P2P and 30 CPU tests.

Several alternatives lost: universal Q4 wave shapes, small-batch unsigned/row-major kernels, transposed
GEMM tiles and packed attention loads. Retain the measured selectors; neither a hardware maximum nor
M3/M5 completion has been demonstrated.

## 2026-10-03: measured kernel and server improvements

Raw comparisons, source/binary hashes, clocks and validation logs are in
`results/engine/performance-20261003/`; its README describes the measurement limits and reproduction.

- Signed Q4 nibble extraction raises warmed original-weight bandwidth by 5.4% for FFN gate, 8.1% for
  FFN down, 8.2% for attention Q and 12.1% for attention output. The gate reaches about 153 GB/s per GPU.
- Wave reductions and a merge that advances sorted list heads cut one-column top-64 selection from
  about 2.20 to 0.51 ms per GPU, a 4.3× speedup, while preserving token-ID tie order.
- tg128 improves from 21.33 to 22.14 tok/s (+3.8%). A separate tg256 run confirms 21.26 to 21.88
  (+2.9%). pp512 stays near 161 tok/s. All six ordinary/MTP chat outputs equal the baseline text.
- The HTTP server caches one-candidate target graphs for greedy requests. End-to-end latency falls
  7.4–9.0% for ordinary replies and 8.7–9.2% with MTP on the three 128-token prompts. The Responses
  integration suite passes in both modes, including alternating greedy/stochastic requests.
- The final Q4 selector uses two four-column tiles at batches 5–8, improving the kernel by another
  20–67%. Batch 8 reaches 0.546 ms versus the original 0.945 ms. A seven-draft transformer chat rises
  from 14.36 to 20.91 tok/s (+45.6%), with identical output, 91/259 accepted proposals and 37 rounds;
  verification falls from 211.42 to 136.13 ms/round. Three drafts remain the default and are faster
  on this prompt (29.05 tok/s).
- Six reference paths pass all 132 tensors, logits and ten greedy tokens; worst relative activation
  error is 0.000982. The 1700-token / batch-512 cache-wrap and rollback check passes. The final selector
  separately passes the batch-8 reference and every verification column, with zero logit error in the
  80-token check. CPU tests, fixtures, P2P stress and local Clippy pass. Sustained clocks are 850/1370,
  with maximum recorded temperatures 86/84 C across the microbenchmark/full-validation log.

The warmed microbenchmark needed a reusable command-buffer execution helper: the one-shot `Device::run`
frees its command buffer. An early harness reused freed buffers and triggered a driver reset; the
recorded comparisons were rerun after correcting that lifetime. Original cold microbenchmarks are
excluded from the reported bandwidth comparison. M3's bandwidth/decode targets and M5's typical-chat
gain target remain open.

## 2026-10-02 (implementation): independent Rust/Vulkan engine

The target forward pass, tokenizer, chat, sampling, prefill and native BF16 MTP assistant are implemented.
Build and validation recipes, the pinned assistant manifest and milestone status are in `implementation.md`.
The executable runs without llama.cpp. The reference checks exercise single-token, four-token, GEMM,
CPU-relay and MTP paths against the saved activations and continuation.

### Transport and memory findings

- **Device addresses accidentally serialize the two GPUs on this RADV version.** Mesa 25.0.7's
  `radv_device.c` enables `use_global_bo_list` for buffer device addresses and descriptor indexing /
  update-after-bind. Every imported peer allocation then participates in each submission's implicit
  synchronization. Fixed descriptors with all these features disabled restore overlap.
- **Disjoint receive slots must be distinct DMA-BUF allocations.** Ranges in one allocation share implicit
  fences; separate exported buffers are necessary for the pre-recorded 97-segment timeline schedule.
- **The 256 MiB BAR limits large-batch P2P.** Ninety-six receive slots at batch 512 need 720 MiB. Slots
  are capped at batch 128 / 180 MiB. Larger batches now reuse two 512-token buffers (15 MiB), with GPU0/GPU1
  submissions interleaved per segment. Implicit fences then refer to preceding consumers, allowing reuse
  without CPU fence waits. This raised pp512 from about 123 to 159 tok/s in the transport trial. CPU relay
  remains a debug path; its preallocated staging was faster than per-hop allocation and recording.
- At context 8192 / batch 512, measured target allocations are 4552 MiB/GPU; the assistant adds 415 MiB.
  The 32K combined budget is about 5351 MiB/GPU, calculated rather than validated at that context.
- The user authorized temporary CPU PM QoS latency 0. `scripts/latency-run.py` holds the request only for
  its command. It reduced median decode semaphore gaps from about 123–133 us to about 41–42 us before
  the last dispatch-fusion changes. Display-manager configuration was not changed.

### Kernel and assistant findings

- Decode Q4_0 uses f32 scales followed by packed quants (20 bytes/block); the Q6_K head expands to signed
  bytes with f32 signed scales (320 bytes/256 weights). Original embeddings remain Q6_K on CPU.
- A 64×64×32 fp32 GEMM reached about 2.2 TFLOP/s per GPU in the FFN microbenchmark. Four-token
  verification still achieves much less compute throughput, limiting speculative gain.
- The final streamed-P2P benchmark reaches pp512 161.1 tok/s, satisfying M4. tg128 is 21.2 tok/s,
  below M3's 40–50 target. Three paired greedy chats have identical outputs and MTP gains of 1.22×,
  1.01× and 1.45× (combined 1.20×), below M5's 1.5× target. Maximum recorded temperatures are 84/82 C.
- Two-token FFN work is 41–49% faster with the four-row kernel; the final selector uses it for 2–4 tokens.
  Its reference checks pass. One-draft photosynthesis still runs at 22.89 tok/s versus 23.51 with three
  drafts, so the default stays at three. Acceptance and verification amortization both matter.
- Independent Q/K/V and gate/up operations share a trailing barrier. Post-normalization, residual/scaling
  and the following RMS normalization share a dispatch, with the same reduction order as separate norms.
- The pinned assistant contains no centroid tensors, despite centroid settings in its config. Its full
  BF16 vocabulary head is used. Its logit softcap is null; the target's is 30.
- Every draft step keeps the target query position and visible cache fixed, and reads target layers 46
  and 47. Stochastic verification uses P/Q acceptance and the residual distribution on rejection, not
  greedy comparison. A 50,000-trial unit test checks the target sampling law; seeded live runs repeat.
- Raw repetitive prompts can become unstable, as with the earlier llama.cpp investigation. The long
  verification fixture uses a rendered chat. Serial/chunked rounding can alter a near-tied greedy decision;
  the checker reports their common prefix separately and checks verification/rollback against the same
  chunked-prefill cache.

Final benchmark outputs and source hashes are retained under `results/engine/delivered-benchmark/`;
correctness logs are `delivered-validation-suite.log` and `delivered-ring-check.json`.

## 2026-10-02 (later): cross-GPU transport, P2P kernel, llama.cpp TP bugs, decision to build our own engine

### Cross-GPU sync and copy (`bench/xgpu`, `results/xgpu/`)

| one-way hop | stock kernel | P2P kernel |
|---|---|---|
| submit + fence wait round trip, one GPU (p50) | 41-65 us | 64-73 us |
| CPU relay (wait on fence, then submit to the other GPU) | 67.5 us | 73 us |
| CPU proxy (#25051 fallback: pre-submitted, host-signaled) | 72.6 us | 65 us |
| **exported OPAQUE_FD timeline semaphores, pre-submitted** | **44.4 us** | **41.9 us** |

- RADV on Mesa 25.0.7 can export and import timeline semaphores across the two GPUs, so #25051's native path
  works.
- 02:00.0, the card that drives the gdm login screen, has a slower fence round trip than 06:00.0.

**GPU-side spin-waiting across GPUs is impossible on GCN 1.0.**

- A shader ping-pong between the GPUs never completes, in any of these modes: `coherent volatile`, Vulkan
  memory-model atomics, RMW atomics, `nontemporal`. Nor with any memory: cached host import, write-combined GTT
  dma-buf, or P2P VRAM.
- A CPU thread watching host memory sees GPU0's store only when its dispatch ends (about 760 ms later), so the
  writer's L2 holds the store until the end-of-submission writeback.
- The same shader with both roles on one GPU runs at 0.6-0.9 us per hop.
- So cross-GPU sync has to go through the kernel: semaphores, about 42 us per hop.

**PCIe P2P** (custom kernel `6.12.111-p2p`: `CONFIG_PCI_P2PDMA`, `CONFIG_DMABUF_MOVE_NOTIFY`, Ivy Bridge-EP host
bridge whitelisted):

- A dma-buf exported from GPU1's VRAM and imported on GPU0 stays in VRAM (GPU1's own copies of it run at
  100 GB/s; on the stock kernel it migrates to GTT at 5-8 GB/s).
- GPU0 writing into GPU1's VRAM: 15.6-15.9 GB/s at 256 KiB-1 MiB, 9.3 GB/s at 8 MiB. Reading across: 2-3.7 GB/s.
- GPU writes into shared host memory (`external_memory_host`, a cached/snooped type): only 1.4 GB/s at 8 MiB;
  host -> VRAM about 6.5 GB/s.
- Takeaway: allreduces should push partials into the peer's VRAM.

### llama.cpp tensor split on Gemma 4 (now reference only)

1. **Fixed locally:** the `handle_reshape` assert at n_tokens == 1 (see below).
2. **Tied output head replicated** on both GPUs (see below).
3. **#25051's Vulkan allreduce corrupts output.** The text is coherent but has injected digits and repeated
   fragments: "two-2 sentences", "useds2".
   - It's deterministic, and the same with native and proxy sync.
   - With the meta backend's generic allreduce instead (`GGML_VK_COMM_DISABLE`, debug branch), the output is
     correct and matches layer split. tg is 6.4 vs 9.0 tok/s.
   - Root cause: in its small-tensor path, an inactive shard has no `GGML_TENSOR_FLAG_COMPUTE`. On Gemma 4's 1-KV-
     head global layers, one GPU has no attention heads, which makes its shard inactive. For such a shard the path
     zeroes what it sends but not its own local tensor, so it adds the peer's partial onto stale memory. CUDA's
     allreduce `cudaMemsetAsync`s the local tensor in the same case.
   - Not fixed: we stopped patching llama.cpp here.

- Raw (no chat template) prompts make this instruction-tuned model degenerate ("0111..."), on CPU too. Use the chat
  template when comparing outputs.
- Layer split with flash attention on GCN 1.0 gives the same text as without it.

**Decision:** build a purpose-built engine in Rust. See `engine-plan.md`.

## 2026-10-02: first baselines (stock Debian kernel 6.12.111, upstream 99b9548)

`scripts/bench-gemma.sh master-stock`, Gemma 4 12B QAT Q4_0, `-ngl 99`, both D700s:

| split | fa | pp512 t/s | tg128 t/s |
|---|---|---|---|
| layer | 0 | 36.1 | 12.27 |
| layer | 1 | 36.9 | 9.44 |
| tensor | 1 | 118.2 | crash (fixed below) |

- **Clocks are fine now.** Both cards hold 850 MHz core / 1370 MHz memory under load (max 77 C), so the
  patched amdgpu works. Idle drops to 300/150.
- **tg is bandwidth-starved.** Layer-split tg 12.27 t/s means about 85 GB/s of weight streaming, 32% of one
  card's 264 GB/s. The Q4_0 mat-vec shader reads 18-byte blocks with 16-bit loads, three load instructions per
  8 weights. That likely hits GCN 1.0's per-lane address rate before DRAM bandwidth. A repacked layout that
  allows 128-bit loads is a candidate. Check with `test-backend-ops perf -o MUL_MAT` first.
- **FA costs 23% of tg on GCN 1.0** (12.27 → 9.44), and `-sm tensor` forces FA on.
- **Tensor-split prefill is already 3.2x layer split**, even with the blocking host-staged allreduce.

## Bugs found

1. **`-sm tensor` asserts on Gemma 4 token generation**
   (`ggml-backend-meta.cpp:547 GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`).
   - Gemma 4's global layers have 1 KV head. `Kcur` `[256, n_tokens]` is reshaped to `[256, 1, n_tokens]` and
     then RMS-normed.
   - With `n_tokens == 1` the source has `ggml_n_dims == 1`, so `handle_reshape`'s "split is on the last dim"
     shortcut fires first. It keeps axis 0 for the 3D result.
   - The complete-rows branch below it, which moves the split to the singleton head axis, was meant for
     exactly this case. Prefill reaches it; decode doesn't.
   - Fix: check complete rows first. Local commit d99061874 on `tp-vulkan`.
2. **The tied output head is replicated on both GPUs.**
   - Gemma ties `output` to `token_embd`. The GPU copy is named `token_embd.weight`, misses the
     `output\.weight` rule in `llama_meta_device_get_split_state`, and defaults to MIRRORED.
   - So each GPU stores the full 262144x3840 head and computes all 262K logits every token.
   - Fix: give it the same vocab split (axis 1) as `output.weight`. The input embedding stays on the CPU
     (`llama-model.cpp:1606`), so only the output copy is affected. Local commit 226f2232d, not yet measured.

## Platform facts that shape the design

- No dedicated transfer (SDMA) queue family on RADV/Tahiti: 1 graphics queue plus a compute family with 2
  queues. ggml-vulkan uses the second compute queue as its "transfer" queue, so #25051's ring path (allreduces
  of 2 MB or more) is eligible.
- PCIe P2P between the cards needs a custom kernel. `CONFIG_PCI_P2PDMA` and `CONFIG_DMABUF_MOVE_NOTIFY` must both
  be on: without MOVE_NOTIFY, `amdgpu_dma_buf_pin` always pins exports into GTT. The host bridge `8086:0e00` also
  has to be whitelisted. See `kernel/`.
