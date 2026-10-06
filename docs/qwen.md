# Qwen models: Qwen3.8-27B (and Qwen3.5-9B), toward Qwen3.8-Flash-Next

The engine runs `qwen35` GGUFs: Qwen3.8-27B and the same-architecture Qwen3.5-9B, which fits on one Mac
and is the single-machine test model. The forward pass is a pipeline of **stages**, each owning a
contiguous range of layers on one device. Stages chain across the two local GPUs and across machines;
only the hidden state crosses a boundary. Qwen3.8-Flash-Next (`qwen4exp`) is the next target; see the end.

| | Qwen3.5-9B | Qwen3.8-27B |
|---|---|---|
| layers | 32 (24 Gated DeltaNet, 8 gated attention) | 64 (48 + 16) |
| hidden / FFN | 4096 / 12288 | 5120 / 17408 |
| attention | 16 Q, 4 KV heads of 256, RoPE on 64 dims | 24 Q, 4 KV heads of 256, RoPE on 64 dims |
| DeltaNet | 16 key heads, 32 value heads of 128 | 16 key heads, 48 value heads of 128 |
| vocabulary | 248,320, untied head | 248,320, untied head |
| our Q4_0 GGUF | 5.6 GiB | 15.9 GiB |
| uploaded bytes per token (layers + Q6 head) | 5.59 GB | about 17.3 GB |
| single-stream decode bandwidth ceiling at 263 GB/s | 47 tok/s | 15 tok/s |

The ceilings are physical bounds, not predictions: a pipeline reads each weight once per token, one GPU at
a time, so adding stages adds capacity, not single-stream decode speed. The 27B needs two Macs: its
uploaded weights alone exceed one Mac's 12 GB of VRAM.

## Layer math (`engine/src/qwen/cpu.rs` is the executable reference)

Every layer: `x += mixer(rms(x) * attn_norm)`, then `x += W_down(silu(W_gate h) * W_up h)` with
`h = rms(x) * post_attention_norm`. Layers `il` with `(il + 1) % 4 == 0` use attention, the others DeltaNet.

- **Gated DeltaNet:** `W_qkv h` gives `[q (16x128) | k (16x128) | v (heads x 128)]`, a causal depthwise
  convolution of width 4 over the previous three inputs and SiLU. q and k are L2-normalized
  (`x / sqrt(sum x^2 + eps)`). Value head `h` reads key head `h % 16` (ggml tiles heads; the GGUF
  converter reorders the value heads to match). Per value dim `i`:
  `S_i *= exp(g)`, `d = (v_i - S_i . k) * beta`, `S_i += d k`, `o_i = S_i . q / sqrt(128)`, with
  `g = softplus(W_alpha h + dt_bias) * A` (`ssm_a` already holds `-exp(A_log)`) and
  `beta = sigmoid(W_beta h)`. The output is RMS-normed per head (shared 128-weight `ssm_norm`), multiplied by
  `silu(W_gate h)` and projected by `ssm_out`. The `output_gate_type: "swish"` in the 27B config is this
  SiLU; transformers' `qwen3_5` ignores the key.
- **Gated attention:** `attn_q` emits `[q | gate]` per head (2 x 256). q and k get a per-head RMS norm with
  learned weights, then a NeoX rotation of the first 64 dims with base 1e7. Text positions repeat in all
  interleaved M-RoPE sections, which then reduces to that ordinary rotation. Softmax scale 1/16, GQA
  `kv = head / 6`, output multiplied by `sigmoid(gate)` before `attn_output`.
- **Head:** final RMS norm and the untied `output.weight`; no softcap, no embedding scale.

`trashcan qwen-ref MODEL OUT TOKENS...` runs this reference and writes the dump format of
`tools/dump-activations`. Against llama.cpp CPU on the 14-token ChatML prompt (`QWEN_REFERENCE` in
`main.rs`), Qwen3.5-9B BF16 agrees to rel 3.2e-3 at the logits and Q4_0 to rel 2.1e-2 (llama.cpp quantizes
activations to Q8 on that path); both give the identical 10 greedy tokens.

## Quantization

Mixed community quants use Q4_1/Q5_K/Q8_0 matrices the kernels do not read. Make a pure Q4_0 with a Q6_K
head (the existing head kernel), a Q8_0 embedding (dequantized row by row on the CPU) and f32 DeltaNet
gates:

```sh
llama-quantize --pure --imatrix imatrix_unsloth.gguf --output-tensor-type q6_k --token-embedding-type q8_0 \
  --tensor-type ssm_alpha=f32 --tensor-type ssm_beta=f32 Qwen3.8-27B-BF16.gguf Qwen3.8-27B-trashcan-Q4_0.gguf q4_0
```

The BF16 source is `ggml-org/Qwen3.8-27B-GGUF`, the importance matrix `unsloth/Qwen3.8-27B-GGUF`. The 9B
uses the same recipe without an importance matrix, from `unsloth/Qwen3.5-9B-GGUF`.

## GPU stages (`engine/src/qwen/gpu.rs`)

One command buffer per batch size, replayed with a new position. Q4_0 projections reuse the Gemma kernels
(`gemm` above 8 columns, `multivec4*` for 2-8, `matvecq4*` for decode); the head uses `headvec`, `topk` and
`merge` over 243 vocabulary chunks. New kernels:

| kernel | work |
|---|---|
| `qwen_add_norm` | residual add and RMS norm, one workgroup per token |
| `qwen_matf32` | f32 alpha/beta projection, one wave per row and eight tokens |
| `qwen_conv` | width-4 causal convolution + SiLU, one thread per channel walking the batch |
| `qwen_gdn` | DeltaNet recurrence: each wave keeps four state rows in registers across the batch |
| `qwen_gated_norm` | per-head RMS norm times `silu(z)` |
| `qwen_qk` | per-head RMS norm, partial NeoX rotation, scale, f16 cache store |
| `qwen_gate` | `sigmoid(gate)` on the attention output |
| `qwen_swiglu` | SwiGLU |

Attention uses the existing scalar kernels: split-K decode (`attention_partial*` + merge) for up to 8
columns and the unsplit `attention` for prefill. The tiled prefill kernels assume a power-of-two query
group (Qwen has 6 query heads per KV head), so long prompts are not yet fast.

The DeltaNet state is 128 x 128 f32 per value head (3 MiB per layer for the 27B), updated in place.
Position 0 restarts it from zero. Unlike the KV cache it cannot be rolled back, so the Qwen path re-prefills
each chat turn and has no prefix reuse or speculative rollback yet.

## Machines

`plan` (`pipeline.rs`) balances uploaded bytes (Q4_0 repacked to 20 bytes per block, plus the 1.25-byte Q6
head on the last stage) over every device in the chain. `--cuts a,b,...` overrides it with the first layer
of each later stage.

The driver embeds tokens, runs its own stages, then sends the hidden rows to the next machine
(`net.rs`). Each machine runs its stages and forwards; the last one returns the top 64 candidates back
along the chain. Prompt chunks are sent without waiting for a reply, so they pipeline across machines.
Machines only connect to their neighbours, which matches a Thunderbolt daisy chain. Payloads are small: one
27B decode token is 20 KiB per hop, a 512-token chunk 10 MiB.

```sh
# every machine after the first (the GGUF must be at the same path or given explicitly)
trashcan qwen-node ~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf --listen 0.0.0.0:7070
# the driver
trashcan qwen-chat ~/models/qwen3.8-27b/Qwen3.8-27B-trashcan-Q4_0.gguf --nodes 10.0.0.2:7070 "Hello"
trashcan qwen-bench MODEL --nodes 10.0.0.2:7070 --pp 512 --tg 128
trashcan qwen-validate MODEL OUT --nodes 10.0.0.2:7070   # then tools/compare_activations.py OUT REFERENCE
```

`--cpu` runs a machine's layers on the CPU reference path, so the protocol can be exercised without GPUs.
On a Mac, two or three local processes chained over `127.0.0.1` reproduce the single-process greedy
tokens and activations exactly, including streamed 4- and 5-token prompt chunks.

Within a machine, the two GPUs hand off through host memory (one fence wait per stage per forward). P2P
pushes with cross-device semaphores, as in the Gemma tensor-parallel path, are the obvious next step.
