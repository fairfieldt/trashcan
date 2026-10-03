# Engine plan: Gemma 4 12B QAT Q4_0 on 2x D700

This is a purpose-built inference engine in Rust (`engine/`, binary `trashcan`) for one model on one machine: incite,
a Mac Pro 6,1 with two FirePro D700s (GCN 1.0, RADV), running Gemma 4 12B QAT Q4_0 with tensor parallelism across
both GPUs. llama.cpp stays in the repo as the correctness reference (activations, tokenizer) and as a source of GLSL
shaders to start from (MIT).

- **This doc:** what we're building, why, and in what order.
- **`engine-spec.md`:** the exact math, constants, formats, tokenizer, validation assets, GPU facts and working
  conventions. An implementer needs both.
- **`findings.md`:** the measurements behind the decisions.

## Stack

| need | choice |
|---|---|
| Vulkan | `ash` 0.38 (raw bindings, `loaded` feature) |
| shaders | GLSL compiled with `glslc` from `build.rs` (skip with a warning when `glslc` is missing, e.g. on the Mac) |
| model file | our own GGUF reader (`engine/src/gguf.rs`) on `memmap2` |
| support | `anyhow`, `half`, `bytemuck`, `clap`; `minijinja` for the chat template when we get there |

We avoid wgpu, vulkano, rust-gpu and candle: they hide the multi-GPU controls we need, are experimental, or are the
wrong abstraction.

## Design in one paragraph

Both GPUs hold half of every big matrix (spec 3.6). The residual stream is replicated, and each layer has two
allreduces: after the attention output projection and after the FFN down projection. Allreduces use a P2P *push*
of an f32 partial into the peer's VRAM, plus a cross-device timeline semaphore. Larger prefill batches
reuse two wide P2P buffers with interleaved submissions to stay within the peer-visible BAR budget. GPU-side spin-waiting is impossible on
GCN 1.0, and P2P reads are slow. A token's work is fixed, so each GPU's command buffers are recorded once and
replayed; per token the CPU only writes a small params buffer (position, KV length) and submits. Weights are repacked
at load into a layout chosen for GCN 1.0 (wide aligned loads), shared by the decode mat-vec and the prefill GEMM.

## Cost model

The [reproducible ceiling audit](../results/engine/ceiling-audit-20261003/README.md) checks the actual
GGUF tensor directory and `model.rs`/`quant.rs` layouts. Numbers below are **per GPU**; the global
K projection is replicated. GB is decimal. Dense-equivalent FLOPs count one multiply-add as two
operations and exclude dequantization, norms, softmax and reductions.

| | decode (per token) | ordinary prefill (512-token chunk, empty prefix) |
|---|---|---|
| matrix payload | 4,040,294,400 bytes (3.763 GiB): Q4 layers 3,411,148,800 + Q6 head 629,145,600 | weights reused across token columns; one LM-head column per chunk |
| dense matrix work | 11.92230912 GFLOP | 10.91764224 GFLOP/token, including the amortized final head |
| attention dot products | context-dependent; 0.470 GFLOP at 1024 visible tokens | 0.117669888 GFLOP/token (QK and PV only) |
| physical peak at sustained clocks | 263.04 GB/s; weights alone take at least 15.36 ms | 3.4816 TFLOP/s; counted work alone takes at least 3.170 ms/token |
| ideal bound, omitted overheads | weight-only 65.10 tok/s | counted-compute-only 315.50 tok/s |
| transport sensitivity | 96 × 42 us = 4.032 ms; adding this assumption to weights gives 51.57 tok/s | 754,974,720 bytes sent in each direction per chunk; about 81 ms at the isolated 9.3 GB/s copy rate |
| historical engine at ceiling-audit snapshot, 2026-10-03 | tg128 22.14; separate tg256 21.88; ordinary 128-token chat 23.82–24.25 tok/s | pp512 160.87; repeat 161.05 tok/s |
| preceding measured build `5b62aa40`, 2026-10-03 | pooled tg256 **23.42 tok/s**; fresh pp4096/tg128 **21.36 tok/s** | pooled pp512 **159.94 tok/s** |
| norm + merge follow-up `2682f29e`, 2026-10-03 | pooled tg256 **28.72 tok/s** vs fresh 23.32; pp4096/tg128 **26.39 tok/s** vs fresh 21.15 (one pair) | pooled pp512 **160.76 tok/s**, unchanged vs fresh 160.83 |
| experiment goals | beat the measured baseline first; 40–50 remains an unvalidated stretch goal | first test 175–190 tok/s; 200 requires almost the whole graph to match the best FFN GEMM |
| recorded llama.cpp reference | 12.3 (layer split), about 9 (tensor split) | 36 (layer), 118 (tensor) |

**51.57 and 315.50 tok/s are not practical performance predictions.** The old “52 tok/s ceiling” combined
ideal bandwidth with a fixed historical hop measurement, while omitting attention and all kernel
inefficiency. In the ceiling audit's historical final-token profile, GPU segment work takes 41.74–41.77 ms and accumulated
semaphore gaps take 2.65–2.75 ms. These are one-token timestamp samples, not run-wide averages; even
eliminating every measured gap saves only about 6% of that sample. Kernel work dominated that sample.
The [main-branch tuning report](../results/engine/main-tuning-20261003/README.md) separates the preceding
fixed-dimension build from its fresh shared-KV baseline and the earlier shared-KV/core phases. Those
fixed rates pool two runs by total tokens/time; long-context rates use one pair. These explicit-batch
measurements do not by themselves validate the batch512 default or long-lived server behavior.

For prefill, the best isolated FFN GEMM reaches about 2.2 TFLOP/s. Applying that rate to all counted
pp512 work gives **199.36 tok/s before transport and other overheads**. That is an optimistic sensitivity
calculation, because other shapes and attention do not necessarily reach the same rate. GCN 1.0 has
no fp16 arithmetic, integer dot product or matrix units; dequantization and fp32 GEMM efficiency matter.

Memory traffic grows with context. At 8K, unique live K/V payload is 288 MiB per GPU (160 MiB sliding,
128 MiB global); at 32K it is 672 MiB. The scalar attention baseline requests K/V separately for
each query head: logical f16 payload requests are 1.3125 GiB and 4.3125 GiB, respectively, before
cache reuse. The selected split-attention decode kernels now share each K/V load across two query
heads in sliding layers and eight in global layers. At these full-context lengths, their logical
K/V payload requests reduce to one read of the unique payload: 288 MiB at 8K and 672 MiB at 32K,
excluding scratch and other traffic. The eight-partition short path and large-prefill attention
remain scalar. Logical payload counts are not hardware DRAM measurements: cache lines, cache reuse,
and execution details matter. The 32K figure is calculated, not validated by a 32K execution.
Long-context throughput requires its own measurement; a short-context decode target cannot be
carried over unchanged.

The [norm/merge follow-up](../results/engine/validated-followup-20261003/README.md) removes
about 8.24 ms of last-token GPU work through a scalar load-hoisted fused norm and a wide
attention-partition merge, preserving the original reduction order. Its +23.16% repeated decode
gain is a measured workload result; it does not alter the physical rooflines or complete M3/M5.
The recovered investigation shows that instruction issue and serialized load dependencies matter
alongside matrix traffic. Remaining Q4 and verification candidates need independent acceptance.

Optimize in this order, retaining changes only with correctness and sustained-clock measurements:

1. Q4 FFN projections: 2,654,208,000 bytes per GPU per decode token, **65.7%** of matrix payload.
   Compare per-shape kernel timings and count both original and repacked bytes. A reported 180 GB/s
   of original Q4 is 200 GB/s of repacked payload, or 76.0% of the physical bus peak.
2. Attention projections and the Q6 head: the head alone is **15.6%** of matrix payload, and the
   narrow attention projections have substantially lower measured bandwidth than FFN. Favor launch,
   occupancy and data-reuse improvements supported by the profile.
3. Small-batch verification and drafter overhead: improve accepted tokens per second on the same
   prompts, with output/rollback checks; larger draft counts alone have already been slower.
4. Prefill GEMM/attention and repeated-history work: optimize the complete prompt path and validate
   sliding-window wrap. Prefix caching changes cache lifetime and needs separate correctness coverage.

## Memory per GPU (5888 MiB non-visible heap plus 256 MiB visible heap)

| item | size |
|---|---|
| repacked Q4_0 layer weights, including replicated global `attn_k` | about 3253 MiB |
| expanded Q6_K LM head | 600 MiB |
| sliding KV: (1024 + batch) positions × 4 heads × 256 × (K,V) × f16 × 40 layers | 240 MiB at batch 512 |
| global KV: 1 head × 512 × (K,V) × f16 × 8 layers, replicated | 16 KiB/token: 128 MiB at 8K, 512 MiB at 32K |
| 96 dedicated P2P receive slots, capped at 128 tokens | 180 MiB |
| two reusable wide P2P slots | 15 MiB at batch 512 |
| relay receive, activations, norms, scratch and allocation padding | about 140 MiB at batch 512, including 8.03 MiB attention scratch |
| optional matched BF16 assistant | 415 MiB |

The earlier fixed-32-partition build measured 4552 MiB of target allocations at 8K context / batch 512,
or about 4967 MiB with the assistant. Increasing attention scratch to 64 partitions changes its payload
from 4.02 to 8.03 MiB per GPU. The current target measures **4556.1 MiB per GPU** at 8K / batch 512
in the core `final-{1,2}.json`, shared-KV phase, and final `fixed-dim/decode-fixed-{1,2}.json` under
`results/engine/main-tuning-20261003/`; adding the assistant gives a calculated
combined total of about 4971 MiB (4.85 GiB). At 32K the calculated combined total
is about 5355 MiB (5.23 GiB). Against 6144 MiB across both VRAM heaps, that leaves an arithmetic
789 MiB before uncounted driver/display/transient allocations. This aggregate does not establish
visible-heap headroom: the measured 8K default-MTP workload leaves only 7.176 MiB visible on the
fuller card.
Scratch is shared with the assistant and reserves the maximum eight verification columns regardless of
configured batch capacity. These allocation totals are not a residency guarantee; 32K execution has not
been validated. Imported aliases are counted only on the owning GPU. Mapped host staging and
driver-managed pipelines and cached command buffers are outside these totals.

The original budget omitted repack expansion and treated receive buffers as small. Full 512-token P2P
slots require 720 MiB per GPU, exceeding the 256 MiB peer-visible BAR. The implemented schedule caps the
P2P slots at 128 tokens and adds two full-width slots for larger batches (195 MiB total at batch 512).
Large batches interleave GPU0/GPU1 submissions for each segment before submitting the next segment.
This keeps implicit fences on reused slots ordered behind their consumers, without CPU fence waits.
Submitting all GPU0 work first with reused slots would introduce dependencies on future peer work.
CPU relay with persistent staging remains available through `--cpu-reduce`.

## Decode schedule (per GPU, per token)

```
seg 0      : params -> embed row (CPU dequant, upload both GPUs) -> layer 0 attention ... W_o partial -> push to peer, signal
seg 2L+1   : wait peer -> sum partials + post-attn norm + residual -> FFN norm -> gate/up -> W_down partial -> push, signal
seg 2L+2   : wait peer -> sum + post-FFN norm + residual + layer scale -> layer L+1 attention ... -> push, signal
seg 96     : wait peer -> ... final norm -> LM head (local vocab half) -> softcap -> local top-k -> host-visible result
```

- That's 97 submissions per GPU per token. Semaphore values go in the submit info, so the command buffers are static.
- Every P2P allreduce gets its own receive allocation; capacity is capped at 128 tokens (180 MiB total).
- Larger prefill batches reuse two wide slots and interleave submissions, with GPU semaphore orchestration.
- The CPU merges the two local top-k lists, samples, and starts the next token.

## Prefill

Prefill processes the prompt in chunks of up to 512 tokens through the same layer schedule, with batched kernels:

- **GEMM:** Q4_0 x f32 with dequantization into shared memory, fp32 FMA.
  - Implemented tile: 64x64 outputs per 256-thread workgroup, 4x4 accumulators per thread, a K step of 32
    (one Q4_0 block), with padded vec4 shared tiles (18 KiB). The FFN microbenchmark reaches about 2.2 TFLOP/s
    per GPU. A tested 128x64 tile was slower on this compiler/hardware.
- **Attention:** causal, with the 1024 window on sliding layers, over the chunk plus the cached prefix (tiled,
  flash-attention style, fp32).
- **Allreduce:** the same push-and-signal, with `N x 3840` f32 partials. Overlapping the transfer with the next
  layer's work can come later. The isolated-copy estimate is about 2.5% of the recorded 3.18-second
  pp512 run, but copy throughput and semaphore gaps must be measured in the complete graph.
- **KV writes:** the whole chunk. The sliding-layer ring must hold window + chunk (1536) so a chunk's queries still see
  their full 1024-key window.
- **Small batches:** a multi-column mat-vec (N = 2..8, weights read once, every column accumulated) is better than a
  GEMM. It's the same kernel MTP verification needs. Pick the crossover by measurement.
- **Initial bring-up:** prompts used the decode path one token at a time. The implemented CLI now uses
  chunked prefill; validation retains the serial path as an independent comparison.

## MTP drafter (speculative decoding, last)

Google ships a drafter matched to our QAT checkpoint: `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`
(Apache-2.0, not gated, 845 MB bf16 safetensors, revision 18934064). llama.cpp's implementation is
`llama.cpp/src/models/gemma4-assistant.cpp`.

Architecture:

- **Input:** concat(target `token_embd`[next token] x sqrt(3840), target `result_norm` hidden state) = 7680, then
  `nextn_proj_pre` -> 1024.
- **4 layers** (3 sliding, 1 global), hidden 1024, 16 Q heads (256 / 512 dims), FFN 8192, the same sandwich norms and
  layer scale. **Q-only attention:** K/V come from the target's KV cache (`num_kv_shared_layers: 4`); the drafter has
  no cache of its own.
- **Output:** final norm, logits over its own tied 262144 x 1024 embedding, and `nextn_proj_post` -> 3840, the hidden
  state for the next draft step.
- The config advertises a 2048-centroid index (`num_centroids`, `centroid_intermediate_top_k: 32`), but the pinned
  safetensors header contains no centroid tensors. The implementation uses the complete 262K-vocab BF16 head
  (537 MB); centroid acceleration remains a separate optimization requiring actual index weights.

Plan: draft k tokens (start with 3-4), then verify all k+1 positions in one target pass with the multi-column
kernels, accept the matching prefix for greedy sampling, or use P/Q rejection sampling for stochastic
sampling. Roll back logical positions and overwrite rejected KV slots. The sliding ring must reserve the
whole verification batch in addition to the 1024-token window.

- **Splitting the drafter:** it reads the target's KV heads, which are split across the GPUs, so it should be split the
  same way (Q heads `8g..8g+7` against local KV heads). That's 8 extra hops per draft token, about 0.35 ms.
- **Gain target:** 1.5-2x on tg, depending on acceptance; still unvalidated on typical chat. A four-token
  verification costs **47.68923648 GFLOP per GPU in matrix work alone**, because all four positions need
  the full LM head (ordinary prefill only computes the chunk's final head). The old 44 GFLOP estimate
  omitted these heads. The original 18 ms estimate assumed much higher small-batch compute throughput: the implemented
  transformer-chat measurement takes about 84 ms per verification and 4 ms per draft proposal. Small-batch
  kernel throughput is still the main optimization needed for the gain target.
  The validated 2026-10-03 load-first multi-column kernels lower weighted verification
  to 57.87 ms and give pooled 39.96 tok/s MTP versus 30.60 ordinary (1.306x) on
  three greedy prompts, two runs each. M5's typical-chat 1.5x goal remains open.
- **Before starting:**
  - Which target layers' caches the drafter reads. Find the mapping in llama.cpp's MTP context sharing
    (`src/llama-context.cpp`, `ctx_other`, `common/speculative.cpp` `draft-mtp`).
  - The centroid head format (the pinned asset has no centroid tensors).
  - How to fetch and pin the checkpoint (the homelab modelstore manifest is the usual place).
  - A loader: the `safetensors` crate, or convert to GGUF with llama.cpp's converter.

## Milestones

Each one ends with something measurable. Correctness is checked against `results/reference-vulkan` (spec section 6).
The implemented paths and recorded acceptance status are in `implementation.md`: M1, M2 and M4 pass.
M3 and M5 have working implementations, but their performance thresholds remain open.

- **M0, done.** Workspace; `trashcan devices` (all requirements met on both GPUs); `trashcan info` / `tensor` /
  `vocab`; reference activations and the tokenizer fixture captured.
- **M1: correct single-token forward, tensor-parallel, CPU-orchestrated.**
  - Load, shard (spec 3.6) and upload weights in llama.cpp's own layout.
  - Port the simplest llama.cpp kernels (mat-vec Q4_0 and Q6_K, RMS norm, RoPE, attention, GEGLU).
  - Allreduce through the CPU at first (read both partials, add, upload). It's slow but takes sync out of the picture
    while kernels are debugged; keep it as a debug mode later.
  - Process the 15 reference tokens one at a time and dump activations in the reference format.
  - **Done when** `compare_activations.py` against `reference-vulkan` is within the CPU-vs-Vulkan band (rel <= about
    4e-2 at the last layers, cos >= 0.999, same top-1) and 10 greedy tokens match.
- **M2: usable and fast-shaped.**
  - Tokenizer (passes `tests/fixtures/tokenizer-fixture.json`), chat template, sampling, stop tokens.
  - The P2P visibility test (spec 7.1), then the P2P push + semaphore allreduce.
  - Pre-recorded command buffers with the params buffer; vocab-split LM head with local top-k.
  - A `trashcan chat` CLI.
  - **Done when** greedy output matches, and tg is measured against llama.cpp's 12.3 tok/s with a per-segment
    timestamp breakdown.
- **M3: decode performance.**
  - Repacked Q4_0 layout and a GCN-tuned mat-vec at >= 180 GB/s of original Q4 payload per GPU
    (= 200 GB/s repacked payload; measure with a micro-benchmark first).
  - Fuse the partial sum + residual + norm.
  - Look at the hop latency (CPU C-states / PM QoS, IRQ affinity).
  - **Stretch acceptance target, not a predicted ceiling:** the microbenchmark sustains >=180 GB/s of
    original Q4 per GPU, activation checks still pass, and tg128 reaches 40-50 tok/s with clocks recorded.
    Current measured improvements remain useful even while this original stretch target is open.
- **M4: prefill.** GEMM, batched attention, chunked prefill with batched allreduce, multi-column mat-vec for small N.
  **Done when** sequential and chunked execution agree across sliding-cache wrap, verification columns agree,
  and pp512 sustains >=150 tok/s (llama.cpp tensor split: 118).
- **M5: MTP speculative decoding** (above). **Done when** acceptance and rollback preserve the target continuation,
  and a recorded typical-chat benchmark shows >=1.5x M3's tg.
- **Later:** long-context tuning. Global-layer K/V at 32K contains 0.5 GiB of unique payload per GPU.
  The scalar baseline's eight local query heads request 4 GiB of logical f16 payload before cache
  reuse; the selected shared-KV global kernel requests 0.5 GiB by loading once for all eight heads.
  This is payload accounting, not measured DRAM traffic or validated 32K throughput.
  The OpenAI-style HTTP endpoint is implemented; see `server.md`.

## Open questions and risks

1. **Receiver-side cache visibility for P2P pushes** (spec 7.1). The implementation passes 10,000 changing
   bidirectional payloads; retain the stress test when changing transport or barriers.
2. **Hop latency** (42 us): mostly kernel scheduling and interrupts. Try PM QoS (`/dev/cpu_dma_latency`) and pinning.
   The user authorized a temporary latency request of 0 during benchmarks on 2026-10-02. Use
   `scripts/latency-run.py`, which drops elevated privileges and closes the request when the command exits.
3. **Shader compiler quality on GCN 1.0:** inspect the ISA with `RADV_DEBUG=shaders` when tuning.
4. **Thermals:** 77 C after a few minutes of llama-bench. Watch sustained runs with `scripts/clockmon.sh`.
5. **Display manager:** gdm on 02:00.0 adds noise. Stopping it during benchmarks is a user decision.
6. **KV cache precision:** f16 to match llama.cpp's default. f32 is possible for the small global cache if precision
   shows up in comparisons.
