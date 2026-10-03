# Engine implementation spec

This is the companion to `engine-plan.md`, the reference an implementer works from. Every number here was read from
the GGUF (`trashcan info` / `tensor` / `vocab`), from llama.cpp's source at 99b95488c (the `llama.cpp/` checkout), or
measured on incite. llama.cpp names are given so results can be compared against its activations.

Model file on incite: `~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf` (6.50 GiB, GGUF v3, 667 tensors,
from HF `google/gemma-4-12B-it-qat-q4_0-gguf` rev 29d0977).

## 1. Model constants

| constant | value | GGUF key |
|---|---|---|
| maximum context | 262144 | `gemma4.context_length` |
| layers | 48 | `gemma4.block_count` |
| hidden size `D` | 3840 | `gemma4.embedding_length` |
| FFN size `F` | 15360 | `gemma4.feed_forward_length` |
| Q heads | 16 (every layer) | `gemma4.attention.head_count` |
| KV heads | 8 on sliding layers, 1 on global layers | `gemma4.attention.head_count_kv` (per-layer array) |
| head dim | 256 on sliding layers, 512 on global layers | `key_length_swa` / `key_length` (same for values) |
| layer kinds | layer `il` is global iff `il % 6 == 5` (5, 11, ..., 47), else sliding | `gemma4.attention.sliding_window_pattern` (read it, don't hardcode) |
| sliding window | 1024 | `gemma4.attention.sliding_window` |
| RoPE base | 10000 (sliding), 1000000 (global) | `rope.freq_base_swa`, `rope.freq_base` |
| RoPE dims | 256 (sliding: all), 512 (global: all, with freq factors) | `rope.dimension_count[_swa]` |
| RMS norm eps | 1e-6 | `attention.layer_norm_rms_epsilon` |
| final logit softcap | 30.0 | `final_logit_softcapping` |
| vocab | 262144 | `tokenizer.ggml.tokens` length |
| attention scale | **1.0** (no 1/sqrt(d); the Q/K norms set the scale) | hardcoded in llama.cpp `gemma4.cpp` |
| shared KV layers | 0 | `attention.shared_kv_layers` |
| per-layer embeddings | none | `embedding_length_per_layer_input = 0` |

## 2. Tensors

All weights are stored ggml style: `dims[0]` is the contiguous (input/K) dimension, and a weight `[K, N]` maps K
inputs to N outputs, one row of K per output. Types: Q4_0 for every 2D layer weight, Q6_K for `token_embd`, F32 for
norms and scales.

| tensor | sliding layer | global layer |
|---|---|---|
| `blk.L.attn_norm.weight` | F32 [3840] | same |
| `blk.L.attn_q.weight` | Q4_0 [3840, 4096] | Q4_0 [3840, 8192] |
| `blk.L.attn_k.weight` | Q4_0 [3840, 2048] | Q4_0 [3840, 512] |
| `blk.L.attn_v.weight` | Q4_0 [3840, 2048] | **absent** (V comes from the K projection) |
| `blk.L.attn_q_norm.weight` | F32 [256] | F32 [512] |
| `blk.L.attn_k_norm.weight` | F32 [256] | F32 [512] |
| `blk.L.attn_output.weight` | Q4_0 [4096, 3840] | Q4_0 [8192, 3840] |
| `blk.L.post_attention_norm.weight` | F32 [3840] | same |
| `blk.L.ffn_norm.weight` | F32 [3840] | same |
| `blk.L.ffn_gate.weight`, `ffn_up.weight` | Q4_0 [3840, 15360] each | same |
| `blk.L.ffn_down.weight` | Q4_0 [15360, 3840] | same |
| `blk.L.post_ffw_norm.weight` | F32 [3840] | same |
| `blk.L.layer_output_scale.weight` | F32 [1] | same |

Global tensors: `token_embd.weight` Q6_K [3840, 262144] (787.5 MiB, also the LM head), `output_norm.weight` F32
[3840], `rope_freqs.weight` F32 [256] (global-layer RoPE frequency factors).

Value facts, so nobody "fixes" them by mistake:

- **Norm weights are direct multipliers** (`y = norm(x) * w`, no `1 + w`). They range widely, e.g. `attn_norm` from
  -143 to 193.
- **Q-norm and K-norm weights are constant within each tensor** (checked: all 48 layers), with different values
  per layer, e.g. layer 0: q 1.0234, k 0.1221; layer 5: q 1.0313, k 0.0605. Read them as vectors anyway.
- **`layer_output_scale` varies a lot per layer:** 0.0544, 0.1836, ..., 0.0037 at layer 11, ..., 0.0496 at layer 47.
  Print them all with `trashcan tensor MODEL 'blk.*.layer_output_scale.weight'`.
- **`rope_freqs`:** entries 0..63 are 1.0 and 64..255 are 1e30. This is "proportional" RoPE with
  `partial_rotary_factor` 0.25 (see 3.4).

## 3. Forward pass (one token at position `pos`)

Notation: `rms(x) = x / sqrt(mean(x^2) + 1e-6)` over the stated vector; all math in f32. Names in brackets are the
llama.cpp graph tensors in the reference dumps (section 6).

### 3.1 Input

```
e   = dequant_q6k(token_embd row `token`)            # 3840 floats
x   = e * sqrt(3840)                                  # [inp_scaled]  (sqrtf(3840) = 61.967733, in f32)
```

### 3.2 Layer L (input `x`, the residual stream)

```
h      = rms(x) * attn_norm                                          [attn_norm-L]
q      = W_q h                       (4096 or 8192)                  [Qcur-L]
k      = W_k h                       (2048 or 512)                   [Kcur-L]; on global layers dumped as [Vcur-L]
v      = W_v h on sliding layers; on global layers v = k (raw, before any norm)   [Vcur-L]
q      = per head: rms(q_h) * q_norm                                 [Qcur_normed-L]
k      = per head: rms(k_h) * k_norm                                 [Kcur_normed-L]
v      = per head: rms(v_h)          (no weight)                     [Vcur_normed-L]
q, k   = rope(q, pos), rope(k, pos)  (3.4)                            [Qcur_pos-L], [Kcur_pos-L]
store k, v into this layer's KV cache at pos (f16, as llama.cpp does)
o      = attention(q, K cache, V cache) (3.5)  -> concat heads (4096 or 8192)   [kqv_out-L]
a      = rms(W_o o) * post_attention_norm                            [attn_post_norm-L]
x      = x + a                                                       [attn_out-L]
f      = rms(x) * ffn_norm                                           [ffn_norm-L]
g      = gelu_tanh(W_gate f) * (W_up f)                              [ffn_gate-L], [ffn_up-L], [ffn_geglu-L]
d      = W_down g                                                    [ffn_out-L]
x      = (x + rms(d) * post_ffw_norm) * layer_output_scale            [ffn_post_norm-L], then [l_out-L]
```

`gelu_tanh(z) = 0.5 z (1 + tanh(0.7978845608 (z + 0.044715 z^3)))`. llama.cpp's CPU backend evaluates this through an
f16 lookup table. Its Vulkan backend and our engine use the formula.

### 3.3 Output

```
y      = rms(x) * output_norm                                        [result_norm]  (also the MTP drafter's input h)
logits = W_embd y                    (262144; W_embd = token_embd, the tied head)
logits = 30 * tanh(logits / 30)                                      [result_output]
```

### 3.4 RoPE (NeoX layout)

For a head vector of `n` dims (256 sliding, 512 global), pair `j` in `0 .. n/2` is `(x[j], x[j + n/2])`:

```
theta_j = pos * base^(-2j/n) / ff[j]          base 10000 (sliding) or 1e6 (global)
ff[j]   = 1 on sliding layers; rope_freqs[j] on global layers (1.0 for j < 64, 1e30 for j >= 64)
x'[j]       = x[j] cos(theta_j) - x[j + n/2] sin(theta_j)
x'[j + n/2] = x[j] sin(theta_j) + x[j + n/2] cos(theta_j)
```

So global layers rotate only dims 0..63 and 256..319; for the other pairs theta is about 0, so they pass through.
There's no YaRN and no frequency scaling (`freq_scale = 1`, `ext_factor = 0`, `attn_factor = 1`). Compute the angle
in f32 as llama.cpp does (`theta = pos * powf(base, -2j/n) / ff`).

### 3.5 Attention

- **Scores:** `s_t = q_h . k_t` with scale 1.0, then softmax over visible `t`, then output `sum_t p_t v_t`, all f32.
- **GQA mapping (sliding):** q head `h` uses KV head `h / 2` (8 KV heads for 16 Q heads).
- **MQA (global):** all 16 Q heads use the single KV head.
- **Visibility:** causal, and on sliding layers also windowed. Key position `p0` is visible to query position `p1` iff
  `p0 <= p1` and, on sliding layers, `p1 - p0 < 1024` (llama.cpp `is_masked_swa`, STANDARD: masked when
  `p1 - p0 >= n_swa`). So a sliding layer sees at most 1024 keys, including the current one.
- **No attention logit softcap** (that was Gemma 2).

### 3.6 Tensor-parallel split (2 GPUs)

The residual stream `x` is replicated. Per layer, GPU `g` (0 or 1) owns:

| weight | GPU g holds | note |
|---|---|---|
| `attn_q` | rows `[g*Nq/2, (g+1)*Nq/2)` = Q heads `8g .. 8g+7` | Nq = 4096 or 8192 |
| `attn_k`, `attn_v` (sliding) | rows `[g*1024, (g+1)*1024)` = KV heads `4g .. 4g+3` | matches GQA `h/2` |
| `attn_k` (global) | **all 512 rows (replicated)** | each GPU computes the single K/V head |
| `attn_output` | input columns `[g*Nq/2, (g+1)*Nq/2)` | output is a partial sum over its heads |
| `ffn_gate`, `ffn_up` | rows `[g*7680, (g+1)*7680)` | |
| `ffn_down` | input columns `[g*7680, (g+1)*7680)` | partial sum |
| `token_embd` as LM head | rows `[g*131072, (g+1)*131072)` | logits for its half of the vocab |
| norms, scales, `rope_freqs` | replicated | |

All column cuts (2048, 4096, 7680) fall on Q4_0 block boundaries (multiples of 32). Row cuts are trivial: Q4_0 and Q6_K
rows are independent. The allreduces are: after `W_o` (sum the two partials before `post_attention_norm`) and after
`W_down` (before `post_ffw_norm`). That's 96 per token.

## 4. Quantization formats

**Q4_0**: 32 weights per 18-byte block, blocks laid out consecutively along each row.

```
struct { f16 d; u8 qs[16]; }
w[j]      = ((qs[j] & 0x0F) - 8) * d      for j in 0..16
w[j + 16] = ((qs[j] >> 4)   - 8) * d
```

Blocks are only 2-byte aligned (18-byte stride), which is why llama.cpp's shaders use 16-bit loads. A row of K weights is
`K/32 * 18` bytes; a 3840-wide row is 120 blocks = 2160 bytes.

**Q6_K** (`token_embd` only): 256 weights per 210-byte superblock.

```
struct { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d; }
for half in 0..2 (128 weights each; ql += 64, qh += 32, scales += 8 per half):
  for l in 0..32:
    is = l / 16
    q1 = ((ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32   -> y[l]      = d * scales[is+0] * q1
    q2 = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32   -> y[l+32]   = d * scales[is+2] * q2
    q3 = ((ql[l]    >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32   -> y[l+64]   = d * scales[is+4] * q3
    q4 = ((ql[l+32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32   -> y[l+96]   = d * scales[is+6] * q4
```

A 3840-wide row is 15 superblocks = 3150 bytes, so row `t` of `token_embd` starts at byte `t * 3150`. (Reference:
`dequantize_row_q6_K` / `dequantize_row_q4_0` in `llama.cpp/ggml/src/ggml-quants.c`.)

## 5. Tokenizer, chat format, sampling

### 5.1 Tokenizer (`tokenizer.ggml.model = "gemma4"`)

SentencePiece-style BPE. Everything comes from the GGUF: `tokenizer.ggml.tokens`, `merges` (rank = index),
`token_type`. Ported from `llama.cpp/src/llama-vocab.cpp` (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`).

1. **Special tokens.** When parsing specials, split the text on exact matches of control (type 3), user-defined
   (type 4) and unknown (type 2) token strings, trying longer strings first. A match becomes that id.
   - `<unusedN>` tokens are type 1 and are *not* matched; they get BPE'd as text.
   - `</s>` (212) is a normal token (llama.cpp demotes it at load).
   - llama.cpp parses specials across the whole rendered prompt, user text included. Match that for validation.
2. **Normalize.** Replace every space `' '` with `'▁'` (U+2581). No space prefix (`add_space_prefix = false`).
3. **Split** into maximal runs of non-newline characters and runs of `'\n'` (regex `[^\n]+|[\n]+`).
4. **Newline runs:** a run of only `'\n'` that exists as a whole token becomes that token (e.g. `"\n\n\n"` -> 109).
5. **BPE:** start from UTF-8 characters (not bytes). Repeatedly merge the adjacent pair whose concatenation has the
   lowest merge rank (`merges` entries are `"left right"`). Ties go to the leftmost pair.
6. **Map** each final piece to its id. A piece not in the vocab falls back, byte by byte, to `<0xXX>` tokens
   (uppercase hex, type 6). A byte with no such token is dropped.
7. **BOS:** prepend `<bos>` (2). `add_bos_token = true`.

Decoding: concatenate pieces, turn `'▁'` back into `' '`, and assemble `<0xXX>` byte tokens into UTF-8.

**Fixture:** `tests/fixtures/tokenizer-fixture.json` has 30 strings (Unicode, CJK, emoji/ZWJ, CRLF, tabs, newline runs,
code, digits, specials, empty) with llama.cpp's ids, pieces and detokenized text. Every case must match.

### 5.2 Special tokens

| id | token | role |
|---|---|---|
| 0 | `<pad>` | |
| 1 | `<eos>` | end of generation |
| 2 | `<bos>` | start |
| 98 | `<\|think\|>` | enables thinking (in the system turn) |
| 100 / 101 | `<\|channel>` / `<channel\|>` | thought channel open / close |
| 105 / 106 | `<\|turn>` / `<turn\|>` | turn open / **end of turn** |
| 46-52 | `<\|tool>`, `<tool\|>`, `<\|tool_call>`, `<tool_call\|>`, `<\|tool_response>`, `<tool_response\|>`, `<\|"\|>` | tool use |
| 255999, 256000, 258880-258884 | image/audio/video markers | multimodal (unused) |

**Stop tokens:** `<eos>` 1 and `<turn|>` 106 (HF `generation_config` `eos_token_id: [1, 106]`); llama.cpp also stops on
`<|tool_response>` 50. The full list is in `results/gemma4-12b-special-tokens.txt`.

### 5.3 Chat format

The GGUF embeds Google's Jinja template (`tokenizer.chat_template`, 18.7 KB). Rendering it with `minijinja` is the exact
path. These are llama.cpp's renderings (`results/gemma4-chat-template-examples.txt`); BOS isn't part of the string:

```
thinking on (default):  "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello there<turn|>\n<|turn>model\n"
with a system prompt:   "<|turn>system\n<|think|>\nYou are terse.<turn|>\n<|turn>user\nHello there<turn|>\n<|turn>model\n"
thinking off:           "<|turn>user\nHello there<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
multi-turn:             "...<|turn>user\nHi<turn|>\n<|turn>model\nHello! How can I help?<turn|>\n<|turn>user\nTell me a fact.<turn|>\n<|turn>model\n"
```

With thinking on, the model writes `<|channel>thought\n...<channel|>` before the answer. A CLI should hide or dim
that part.

### 5.4 Sampling

GGUF defaults: temperature 1.0, top-k 64, top-p 0.95. Greedy (argmax) is used for validation. With the vocab-split LM
head, each GPU returns its local top-k (k >= 64) as (value, id) pairs over its 131072 ids. The CPU merges them and samples.
Full logits never cross PCIe during ordinary generation. Both the target and assistant suppress IDs
258883 and 258882 before top-k, matching `common/sampling.cpp` and the GGUF suppression metadata.
The assistant's logits are not softcapped (`final_logit_softcapping: null` in its pinned config).

## 6. Validation assets

**Reference activations:** `results/reference-cpu/` and `results/reference-vulkan/`.

- Written by `tools/dump-activations/dump.cpp`, built against the pristine upstream llama.cpp build in `~/llama.cpp`
  on incite.
- The prompt is the 15 token ids of the thinking-off chat prompt for "Hello there":
  `2 105 2364 107 9259 993 106 107 105 4368 107 100 45518 107 101`.
- For each tensor, `<name>.f32` holds **the last token's slice** as raw little-endian f32. `index.tsv` lists name,
  full shape (ne0..ne3) and slice length.
- `logits.f32` holds the last prompt token's 262144 softcapped logits. `greedy.txt` holds 32 greedy continuation ids.
  The first 10 are `9259 236888 2088 740 564 1601 611 3124 236881 106` ("Hello! How can I help you today?" then
  `<turn|>`).
- Tensors are dumped for all layers for `attn_out-L` and `l_out-L`; for layers 0 (sliding) and 5 (global) every
  intermediate in section 3.2 is dumped too. Also `inp_scaled`, `result_norm`, `result_output`.
- **Ignore** names containing `(reshaped)`, `(view)`, `norm-L` (several intermediates share that name and overwrite
  each other), `kq-L`, `kq_soft_max-L` and `kqv-L`: their slices aren't per-token rows. Use `kqv_out-L` for the
  attention output.

**What "correct" means:** compare against `reference-vulkan`; it uses f32 activations like our engine, while the CPU
backend quantizes activations to 8 bits inside Q4_0 dot products. The CPU and Vulkan references differ from each other
by:

- relative L2 error 5e-3 at layer 0, rising to 3-4e-2 at layers 44-47
- cosine >= 0.999 everywhere
- logits relative error 3.4e-2, the same top-1, top-10 overlap 9/10
- identical greedy tokens through `<turn|>`

An engine run is correct if it matches `reference-vulkan` at least that well. Compare with
`python3 tools/compare_activations.py MY_DUMP_DIR results/reference-vulkan [--all]`; no numpy needed. The engine
should write dumps in the same format: same names, last-token slice, `index.tsv`.

To regenerate (e.g. another prompt): `tools/dump-activations/build.sh` on incite, then
`./dump MODEL OUT_DIR cpu|vulkan N_GREEDY TOKEN_IDS...`.

## 7. GPU facts (both D700s, RADV, Mesa 25.0.7)

From `trashcan devices` (identical cards):

- **Shape:** 32 CUs = 2 shader engines x 2 arrays x 8 CUs. 4 SIMD16 per CU; a wave64 instruction takes 4 cycles.
  10 wave slots per SIMD, 256 VGPRs per SIMD (occupancy = min(10, 256 / VGPRs per wave)), 512 SGPRs.
- **Workgroups:** at most 1024 invocations, 32 KiB shared memory (Tahiti has 64 KiB of LDS per CU, so two such
  workgroups fit). Subgroup size 64, with arithmetic, ballot, shuffle and clustered ops.
- **Peaks:** 850 MHz x 2048 lanes x 2 = 3.48 TFLOPS fp32 per GPU, 264 GB/s GDDR5 (384-bit).
- **Precision:** no fp16 arithmetic and no integer dot product. 16-bit storage (`storageBuffer16BitAccess`), int16 and
  int64 are supported. Unpack f16 with `unpackHalf2x16`.
- **Memory types:**

  | type | heap | flags | use |
  |---|---|---|---|
  | 0, 1 | 0 (5888 MiB) | DEVICE_LOCAL | VRAM |
  | 2 | 1 (32 GiB) | HOST_VISIBLE, HOST_COHERENT | write-combined GTT |
  | 3, 4 | 2 (256 MiB) | DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT | the PCIe BAR window into VRAM |
  | 5, 6 | 1 | HOST_VISIBLE, HOST_COHERENT, HOST_CACHED | cached system memory; host-pointer imports land here |

  Storage buffer range is 4 GiB.
- **Queues:** family 0 is 1 x graphics+compute+transfer; family 1 is 2 x compute+transfer (async compute engines; use
  this one); family 2 is sparse. There's no transfer-only (SDMA) family.
- **Device order:** sort by PCI bus. GPU0 = 02:00.0 (card0, drives the gdm login screen, slightly noisier), GPU1 =
  06:00.0.

### 7.1 Cross-GPU mechanics

`bench/xgpu/xgpu.cpp` is working reference code for all of this. Results are in `results/xgpu/`.

**P2P buffer:**

1. On the owner GPU, allocate DEVICE_LOCAL memory with `VkExportMemoryAllocateInfo(DMA_BUF_BIT_EXT)` plus a
   dedicated allocation, and a buffer created with `VkExternalMemoryBufferCreateInfo(DMA_BUF)`.
2. Get the fd with `vkGetMemoryFdKHR`.
3. On the peer, create the buffer the same way, take the memory type from `vkGetMemoryFdPropertiesKHR` (it reports
   type 2), and import with `VkImportMemoryFdInfoKHR` plus a dedicated allocation.

On the P2P kernel the buffer stays in the owner's VRAM: the owner's own copies run at 100 GB/s. **Writes** from the
peer into it run at 15.6-15.9 GB/s (256 KiB-1 MiB) and 9.3 GB/s (8 MiB). **Reads** across run at only 2-3.7 GB/s.
So data always moves by the producer *pushing* into the consumer's VRAM.

**Cross-GPU sync:** timeline semaphores created with `VkExportSemaphoreCreateInfo(OPAQUE_FD)`, exported with
`vkGetSemaphoreFdKHR`, imported into a timeline semaphore on the other device with `vkImportSemaphoreFdKHR`. RADV
reports them exportable and importable.

Measured one-way hop with all work pre-submitted: 42 us (P2P kernel). Others:

- CPU proxy thread (poll one GPU's timeline, `vkSignalSemaphore` on the other): 65-73 us
- blocking fence relay: 67-73 us
- a single submit + fence round trip: 41-73 us
- our tests always submitted the signaling batch first. Waiting before the signal is submitted works in practice
  (PR #25051's native path relies on it), but RADV may route it through a submit thread. Prefer submitting the
  signaling side first.

**What doesn't work:** shader spin-waiting on another GPU's writes. A GCN 1.0 shader's stores stay in its L2 until
the end-of-submission writeback, whatever the memory type (cached host, WC GTT, P2P VRAM) or access mode (volatile,
memory-model atomics, RMW atomics, nontemporal). Every cross-GPU handoff must be a submission boundary with a
semaphore.

**Must verify before relying on it:**

- After GPU A pushes data into B's VRAM and signals, does B's *next submission* (after its semaphore wait) read fresh
  data and not stale lines in B's L2?
- Observed: a shader's stores become visible outside the GPU when its submission completes (end-of-submission L2
  writeback). Whether RADV invalidates the *waiting* GPU's L2 at the start of a submission is unconfirmed.
- Write a test that changes the payload every round and checks every element, many thousands of rounds, before the
  allreduce depends on it. If stale reads appear, fixes to try:
  - a pipeline barrier with `VK_ACCESS_MEMORY_READ_BIT` / `HOST`-style source scope at the start of the consumer
    submission
  - reading the received slot with `vkCmdCopyBuffer` first
  - a fresh slot per use

**Host -> VRAM uploads** (weights): about 6.5 GB/s via `vkCmdCopyBuffer` from a mapped staging buffer, so loading
3.5 GB per GPU takes about 0.5 s plus disk.

## 8. Shaders to start from (llama.cpp, MIT; keep the license header)

Under `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/`:

| need | file(s) | note |
|---|---|---|
| Q4_0 mat-vec | `mul_mat_vec.comp`, `mul_mat_vec_base.glsl`, `dequant_funcs.glsl`, `types.glsl` | about 32% of peak bandwidth on GCN 1.0 through llama.cpp; replace in M3 |
| Q6_K mat-vec (LM head) | `mul_mat_vec_q6_k.comp` | |
| GEMM for prefill | `mul_mm.comp`, `mul_mm_funcs.glsl` | GCN warptile config in `ggml-vulkan.cpp` (`AMD_GCN`) |
| RMS norm | `rms_norm.comp` | |
| RoPE NeoX | `rope_neox.comp`, `rope_head.glsl`, `rope_funcs.glsl` | supports freq factors |
| attention | `flash_attn.comp`, `flash_attn_base.glsl`, `flash_attn_split_k_reduce.comp` | decode and prefill; GCN-specific row/subgroup choices in `ggml-vulkan.cpp` |
| GELU-tanh GLU | `geglu.comp`, `glu_head.glsl`, `glu_main.glsl` | |

These shaders take their parameters through push constants laid out in `ggml-vulkan.cpp`. A port has to define its
own push-constant structs; read the `vk_*_push_constants` structs next to each pipeline's dispatch code.

## 9. Working conventions

- **Code** lives in this repo; `scripts/sync.sh push` copies `Cargo.*`, `engine/`, `scripts/`, `bench/`, `tools/`, `tests/`, `docs/` to
  `incite:~/trashcan/`. Build there with `cargo build --release` (Rust is in `~/.cargo/bin`, so
  `export PATH=$HOME/.cargo/bin:$PATH`).
- **On the Mac,** `cargo check` works because `ash` loads libvulkan at runtime. A `build.rs` that runs `glslc` must
  skip with a warning when `glslc` is missing (it is on the Mac), so checking still works there.
- **Results** go under `incite:~/trashcan/results/`; `scripts/sync.sh pull` brings them into `results/`.
- **Never write to `/tmp`**, `/var/folders`, `$TMPDIR` or `/dev/shm`, on the Mac or on incite. Use the repo or
  `~/trashcan`.
- **Benchmarks:** check clocks with `scripts/clockmon.sh` (850/1370 MHz under load). llama.cpp baselines (upstream
  99b9548): layer split tg128 12.27 tok/s (9.44 with FA), pp512 36 tok/s; tensor split (with #25051 and fixes) tg
  about 9 tok/s, pp512 118 tok/s.
- **incite** runs the custom kernel `6.12.111-p2p` by default (`kernel/`); without it P2P falls back to system memory.
  incite isn't managed by gpuq.
- **llama.cpp checkouts:**
  - local `llama.cpp/` (upstream plus our branches `tp-vulkan` / `tp-debug`)
  - incite `~/llama.cpp` (pristine upstream build, used for references)
  - incite `~/trashcan/llama.cpp` (the `tp-debug` branch)
  - our llama.cpp patches are also saved in `reference/llama.cpp-patches/`.
  - `llama.cpp/AGENTS.md` forbids AI-written PRs, issues and comments upstream. Never post there.

## 10. Implemented runtime conventions

The Rust runtime is described in `docs/implementation.md`, with runnable validation commands. GGUF and
safetensors are read directly; llama.cpp is not linked or needed at runtime.

- Q4 upload uses a structure of arrays: one f32 scale per block followed by the original 16 packed bytes per
  block. This is 20 bytes per 32 weights, shared by decode, small-batch verification and prefill.
- The Q6_K output head is expanded at upload to f32 signed scales per 16 weights followed by signed 8-bit
  quants. CPU embedding lookup still reads the original Q6_K blocks.
- **Disable buffer device addresses and descriptor indexing/update-after-bind.** Mesa 25.0.7 RADV's
  `radv_device.c` enables `use_global_bo_list` for these features. Including every imported DMA-BUF in each
  submission defeats the intended independent allreduce schedule. The runtime uses fixed descriptors instead.
- **Allocate each of the 96 receive slots separately.** Disjoint byte ranges in one DMA-BUF still share
  buffer-level implicit fences. Each slot uses the dedicated export/import recipe above. Cap their capacity
  at 128 tokens: 96 × 128 × 3840 × 4 = 180 MiB. Full 512-token capacity is 720 MiB, beyond the 256 MiB BAR.
  Batches above 128 reuse two additional wide slots (15 MiB at batch 512). Interleave GPU0/GPU1 submissions
  per segment before the next segment, so implicit fences on reused slots refer only to preceding work.
  This stays within the BAR without CPU fence waits. `--cpu-reduce` retains the persistent host relay path.
- For P2P, submit all 97 timeline-dependent segments in one `vkQueueSubmit` API call per GPU. This preserves the
  per-allreduce semaphore dependencies and avoids 194 CPU submission calls per token.
- Decode and verification attention use 32 cache partitions followed by a stable max/sum reduction. Large
  prefill batches use one online-softmax wave per query/head. All score scales remain exactly 1.0.
- Assistant revision `18934064dd4c5c6cc3621f6381e7d377fc8cb7bd` reads target layers 46 (sliding) and 47
  (global). Draft steps keep the same query position and see only committed target KV. Verification rewinds
  the logical position to the accepted prefix; the sliding ring includes the full verification capacity.
  Greedy verification accepts the matching prefix. Stochastic verification uses P/Q acceptance and a
  normalized max(P-Q,0) correction, using the target's temperature/top-k/top-p sampling law.
- The assistant config mentions centroids, but no centroid tensors exist in its safetensors header. The
  runtime computes its full BF16 head. The pinned download script verifies sizes and SHA-256 checksums.
