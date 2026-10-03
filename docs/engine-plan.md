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
allreduces: after the attention output projection and after the FFN down projection. Each allreduce is a P2P *push*
of an f32 partial into the peer's VRAM, plus a cross-device timeline semaphore. GPU-side spin-waiting is impossible on
GCN 1.0, and P2P reads are slow. A token's work is fixed, so each GPU's command buffers are recorded once and
replayed; per token the CPU only writes a small params buffer (position, KV length) and submits. Weights are repacked
at load into a layout chosen for GCN 1.0 (wide aligned loads), shared by the decode mat-vec and the prefill GEMM.

## Cost model

| | decode (per token) | prefill (per token, batched) |
|---|---|---|
| work per GPU | about 3.5 GB (3.25 GiB) of weights read | about 10.9 GFLOP (21.8 GFLOP total) |
| GPU peak | 264 GB/s -> 13.2 ms | 3.48 TFLOPS -> 3.1 ms |
| allreduce | 96 x about 42 us = about 4 ms (latency-bound, 15 KB each) | 96 x 7.5 MiB per 512-token chunk at about 9-15 GB/s = 50-80 ms per chunk, about 3% |
| ceiling | about 60 tok/s | about 320 tok/s |
| target | 40-50 tok/s | 150-200 tok/s |
| llama.cpp today | 12.3 (layer split), about 9 (tensor split) | 36 (layer), 118 (tensor) |

Decode is bandwidth-bound and sync-latency-bound. Prefill is compute-bound: GCN 1.0 has no fp16 math, no integer dot
and no matrix units, so the GEMM is fp32 FMA, and its efficiency is the whole game.

## Memory per GPU (5888 MiB VRAM)

| item | size |
|---|---|
| Q4_0 layer weights (half of 5847 MiB, plus the replicated 512-row global `attn_k`) | about 2930 MiB |
| LM head (half of the Q6_K `token_embd`) | 394 MiB |
| sliding KV cache: a ring of 1024 window + 512 chunk = 1536 positions x 4 heads x 256 x (K, V) x f16 x 40 layers | 240 MiB |
| global KV cache: 1 head x 512 x (K, V) x f16 x 8 layers, replicated | 16 KiB per token: 128 MiB at 8K, 512 MiB at 32K |
| activations, receive slots, scratch | well under 100 MiB |

That's about 3.7 GiB at 8K context and 4.1 GiB at 32K, which leaves room for the MTP drafter (below).

## Decode schedule (per GPU, per token)

```
seg 0      : params -> embed row (CPU dequant, upload both GPUs) -> layer 0 attention ... W_o partial -> push to peer, signal
seg 2L+1   : wait peer -> sum partials + post-attn norm + residual -> FFN norm -> gate/up -> W_down partial -> push, signal
seg 2L+2   : wait peer -> sum + post-FFN norm + residual + layer scale -> layer L+1 attention ... -> push, signal
seg 96     : wait peer -> ... final norm -> LM head (local vocab half) -> softcap -> local top-k -> host-visible result
```

- That's 97 submissions per GPU per token. Semaphore values go in the submit info, so the command buffers are static.
- Every allreduce gets its own receive slot (96 x 15 KB), so there are no reuse hazards within a token.
- The CPU merges the two local top-k lists, samples, and starts the next token.

## Prefill

Prefill processes the prompt in chunks of up to 512 tokens through the same layer schedule, with batched kernels:

- **GEMM:** Q4_0 x f32 with dequantization into shared memory, fp32 FMA.
  - Starting tile: 128x64 outputs per 256-thread workgroup, 8x4 accumulators per thread, a K step of 32 (one Q4_0
    block): 24 KiB of shared memory, so two workgroups fit in a CU's 64 KiB.
  - Start from llama.cpp's `mul_mm.comp` GCN warptile, then tune.
- **Attention:** causal, with the 1024 window on sliding layers, over the chunk plus the cached prefix (tiled,
  flash-attention style, fp32).
- **Allreduce:** the same push-and-signal, with `N x 3840` f32 partials. Overlapping the transfer with the next
  layer's work can come later; it's about 3%.
- **KV writes:** the whole chunk. The sliding-layer ring must hold window + chunk (1536) so a chunk's queries still see
  their full 1024-key window.
- **Small batches:** a multi-column mat-vec (N = 2..8, weights read once, every column accumulated) is better than a
  GEMM. It's the same kernel MTP verification needs. Pick the crossover by measurement.
- **Until M4:** prompts go through the decode path one token at a time. That's correct, just slow.

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
- The checkpoint also carries a 2048-centroid index (`num_centroids`, `centroid_intermediate_top_k: 32`) for computing
  logits over only part of the vocab. Without it, the drafter's 262K-vocab head (537 MB in bf16) would cost more per
  step than its layers.

Plan: draft k tokens (start with 3-4), then verify all k+1 positions in one target pass with the multi-column
kernels, accept the longest matching prefix, and roll back positions (simply overwrite the KV slots).

- **Splitting the drafter:** it reads the target's KV heads, which are split across the GPUs, so it should be split the
  same way (Q heads `8g..8g+7` against local KV heads). That's 8 extra hops per draft token, about 0.35 ms.
- **Gain:** on GCN 1.0 a 4-token verification is partly compute-bound: about 44 GFLOP per GPU, roughly 18 ms, against
  13-18 ms for one token. Realistic gain is 1.5-2x on tg, depending on acceptance.
- **Before starting:**
  - Which target layers' caches the drafter reads. Find the mapping in llama.cpp's MTP context sharing
    (`src/llama-context.cpp`, `ctx_other`, `common/speculative.cpp` `draft-mtp`).
  - The centroid head format.
  - How to fetch and pin the checkpoint (the homelab modelstore manifest is the usual place).
  - A loader: the `safetensors` crate, or convert to GGUF with llama.cpp's converter.

## Milestones

Each one ends with something measurable. Correctness is checked against `results/reference-vulkan` (spec section 6).

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
  - Repacked Q4_0 layout and a GCN-tuned mat-vec at >= 180 GB/s per GPU (measure with a micro-benchmark first).
  - Fuse the partial sum + residual + norm.
  - Look at the hop latency (CPU C-states / PM QoS, IRQ affinity).
  - **Target** 40-50 tok/s.
- **M4: prefill.** GEMM, batched attention, chunked prefill with batched allreduce, multi-column mat-vec for small N.
  **Target** >= 150 tok/s at pp512 (llama.cpp tensor split: 118).
- **M5: MTP speculative decoding** (above). **Target** >= 1.5x M3's tg on typical chat.
- **Later, optional:** an OpenAI-style HTTP endpoint, long-context tuning (global-layer attention at 32K reads about
  0.5 GiB per token per GPU).

## Open questions and risks

1. **Receiver-side cache visibility for P2P pushes** (spec 7.1). Test before trusting the allreduce.
2. **Hop latency** (42 us): mostly kernel scheduling and interrupts. Try PM QoS (`/dev/cpu_dma_latency`) and pinning.
   This is a system setting, so ask the user first.
3. **Shader compiler quality on GCN 1.0:** inspect the ISA with `RADV_DEBUG=shaders` when tuning.
4. **Thermals:** 77 C after a few minutes of llama-bench. Watch sustained runs with `scripts/clockmon.sh`.
5. **Display manager:** gdm on 02:00.0 adds noise. Stopping it during benchmarks is a user decision.
6. **KV cache precision:** f16 to match llama.cpp's default. f32 is possible for the small global cache if precision
   shows up in comparisons.
