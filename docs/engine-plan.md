# Engine plan: Gemma 4 12B QAT Q4_0 on 2x D700

This is a purpose-built inference engine in Rust (`engine/`, binary `trashcan`) for one model on one machine.
llama.cpp stays in the repo as the correctness reference (CPU backend activations, `llama-tokenize`) and as a
source of GLSL shaders to start from (MIT).

Stack: `ash` (raw Vulkan), GLSL compiled with `glslc` from `build.rs`, `memmap2` for the GGUF, `anyhow`, `half`,
`bytemuck`, `clap`. We avoid wgpu, vulkano and rust-gpu: they hide the multi-GPU controls we need, or are
experimental.

## The model, from the GGUF (`results/gemma4-12b-info.txt`)

| | sliding-window layers (40) | global layers (8: 5, 11, ..., 47) |
|---|---|---|
| attention | 16 Q heads x 256, 8 KV heads, window 1024 | 16 Q heads x **512**, **1 KV head, V = K** (no `attn_v`) |
| q / k / v / o | 3840 -> 4096 / 2048 / 2048, 4096 -> 3840 | 3840 -> 8192 / 512 / (shared), 8192 -> 3840 |
| RoPE | base 10k, 256 dims | base 1M, 512 dims, `rope_freqs` factors (proportional) |
| norms | Q norm and K norm (weighted, per head); V norm (unweighted RMS) | same |

Every layer:

1. Sandwich norms around attention and around the FFN.
2. FFN `down(gelu_tanh(gate(x)) * up(x))` with 3840 -> 15360 -> 3840.
3. A scalar `layer_output_scale` multiply at the end.

Around the stack:

- Input: embeddings scaled by sqrt(3840).
- Output: final RMS norm, then the tied `token_embd` (**Q6_K, 787.5 MiB**, 262144 x 3840) as the LM head, then
  a final logit softcap of 30.

There's no per-layer embedding and no cross-layer KV sharing. Weights total 5847 MiB of Q4_0, 787.5 MiB of Q6_K
and 3 MiB of F32.

## Tensor-parallel layout (per GPU)

| tensor | split | per GPU |
|---|---|---|
| sliding `attn_q` | by head (column) | 8 heads = 2048 rows |
| sliding `attn_k`, `attn_v` | by KV head | 4 heads = 1024 rows each |
| global `attn_q` | by head | 8 heads = 4096 rows |
| global `attn_k` (= V source) | **replicated** | 512 rows (1 MiB, both GPUs) |
| `attn_output` | by input column (row-parallel) | partial sum over its heads -> allreduce |
| `ffn_gate`, `ffn_up` | by row | 7680 rows each |
| `ffn_down` | by input column | partial sum -> allreduce |
| `token_embd` as LM head | by vocab | 131072 rows -> local top-k, merged on the CPU |
| norms, scales | replicated | |

Unlike llama.cpp's meta backend, which puts all of a global layer's attention on one GPU, both GPUs get 8 heads
on every layer.

Memory per GPU is about 2.9 GiB of Q4_0 plus 0.39 GiB of Q6_K, roughly 3.3 GiB of weights, out of 5.75 GiB of
VRAM. The KV cache is kept per GPU:

- Sliding layers: 4 heads x 256 x (K, V) x 1024 window x f16 = 4 MiB per layer, 160 MiB total.
- Global layers: the 1-head K/V is replicated, 2 KiB per token per layer, so 512 MiB at 32K context.

## Per-token cost model (decode)

- Weights: about 3.3 GB read per GPU per token. At 264 GB/s peak that's 12.5 ms; at a realistic 180 GB/s, 18 ms.
- Sync: 96 allreduces x about 42 us per cross-GPU semaphore hop (measured) = about 4 ms.
- So the ceiling is about 60 tok/s and the realistic target 40-50 tok/s. llama.cpp today: 12.3 tok/s with layer
  split, about 9 with tensor split.

## Decode schedule

The residual stream is replicated on both GPUs. Each token is 97 segments per GPU, separated by the 96 allreduces:

```
seg 0      : embed row (CPU dequant, upload) -> layer 0 attention ... o-proj partial -> push to peer
seg 2L+1   : wait peer -> sum partials + post-attn norm + residual -> FFN norm -> gate/up -> down partial -> push
seg 2L+2   : wait peer -> sum + post-FFN norm + residual + layer scale -> layer L+1 attention ... -> push
seg 96     : ... final norm -> LM head (local vocab half) -> softcap -> local top-k -> host-visible result
```

- **Push, don't pull.** Each GPU writes its partial straight into the peer's VRAM: a dma-buf imported over PCIe P2P
  at about 15 GB/s, versus 2-4 GB/s for P2P reads and 1.4 GB/s for GPU writes into shared host memory. Each
  allreduce gets its own receive slot, so there are no reuse hazards within a token.
- **Sync** uses timeline semaphores exported/imported as OPAQUE_FD: about 42 us per hop. GPU-side spin-waiting
  doesn't work on GCN 1.0, because a shader's stores stay in L2 until the end of the submission (see findings).
- **Command buffers are recorded once and replayed every token.** Per-token values (position, KV length) live in a
  small host-visible params buffer. Semaphore values go in the submit info, not the command buffer.
- **Data visibility needs a test first.** The writes in the peer's VRAM must be visible to its next submission. The
  signal-side L2 writeback should cover it, but we'll verify with a dedicated correctness test, because the
  receiving GPU's L2 could hold stale lines of its receive slot.

## Kernels (decode)

These start as ports of llama.cpp's GLSL for correctness, then get rewritten for GCN 1.0:

1. RMS norm (weighted / unweighted), fused with the partial sum and residual add where it sits after an allreduce
2. Q4_0 mat-vec (the hot path; later a repacked layout for 128-bit loads instead of 16-bit)
3. Q6_K mat-vec for the LM head, plus softcap and local top-k
4. Per-head Q/K norm + RoPE (two variants: 256 dims, and 512 dims with freq factors), KV cache write (f16)
5. Decode attention: sliding window 1024 with GQA 2:1 on the sliding layers, full context with MQA on the global
   layers
6. GELU-tanh gate x up

## Milestones

- **M0, done:** workspace, `trashcan devices` (both D700s meet every requirement), `trashcan info` (GGUF inventory).
- **M1:** load and shard weights onto both GPUs; one-token forward matching llama.cpp's CPU activations layer by
  layer. CPU-driven, unoptimized.
- **M2:** tokenizer (from the GGUF, checked against `llama-tokenize`), chat template, sampling; static schedule
  with P2P allreduce. A usable chat CLI, benchmarked against llama.cpp.
- **M3:** kernel work: Q4_0 repack, mat-vec bandwidth, fusion.
- **M4:** batched prefill (GEMM, tiled attention). Until then the prompt goes through the decode path one token at a
  time.
