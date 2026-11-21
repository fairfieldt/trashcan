# Inkling 2 Mini (`inkling2`) on two or more Mac Pros

Inkling 2 Mini (Thinking Machines, released 2026-11-20, Apache 2.0) is a 26.2B-parameter multimodal MoE: 24 layers
of weights, of which the middle 12 run three times per token (48 layer applications), 8 of 128 experts per layer, a
ViT image encoder, and a 3-layer block drafter for speculative decoding. This document is the plan and the
reference for running it on Mac Pros: the layer math, the weights and the chosen quant, the machine layout, the
vision path, the drafter and the order of work.

Text and image prompts run tensor-parallel on two Mac Pros (TP4) with Unsloth's UD-Q4_K_XL and BF16 vision
projector through `inkling chat` and the Responses server (`serve`, recipe `2-inkling`). See [Validation](#validation)
for the llama.cpp comparison and [Performance](#performance) for measurements on incite and enlight.
The architecture details come from `unsloth/Inkling-2-Mini-GGUF`, the original `thinkingmachines/Inkling-2-Mini`
safetensors header, and llama.cpp at `29b52d398`.

| | |
|---|---|
| layers | 24 of weights; 0-5 prelude, 6-17 loop core (3 passes), 18-23 coda: 48 applications, 48 KV / convolution slots |
| hidden | 2304; block attention residuals over blocks of 6 applications instead of a summed residual |
| attention | 24 Q / 8 KV heads of 128; 1024-token sliding window on 5 of 6 layers (global: 5, 11, 17, 23); no RoPE |
| positions | learned relative-position bias over distances < 1024 (16 basis rows, per-head mixing from the input), log-scaled Q on global layers |
| MoE | 128 experts of width 1152, top 8, sigmoid router with a selection bias, routing weights normalized and scaled by 8; no shared expert |
| vocabulary | 200,058 tokens (GPT-4o pre-tokenizer), head and embedding padded to 201,024 rows; final softcap 30 |
| vision | ViT, 23 layers of 1024, patch 20, 2x2 merge, MLP projector to 2304; up to 4096 image tokens |
| drafter | DSpark: 3 dense layers, block size 8, Markov head of rank 256, reads target hidden states at applications 5, 23, 41, 48 |

## Layer math

llama.cpp's `src/models/inkling2.cpp` and `inkling2-loop.cpp` are the executable reference. `rms(x)` below is an RMS
norm without weights, eps 1e-6.

**Input.** `x = rms(token_embd[t]) * token_embd_norm`. Image embeddings from the vision projector are used as they
are, without this norm.

**Block attention residuals** (`mix`). The layer keeps a list of block states: at every application `ip` with
`ip % 6 == 0` the current residual is appended to the list and a new residual starts from the sublayer's output
alone; otherwise the output is added to the residual. A sublayer's input is
`mix(blocks, cur, w) = sum_i softmax_i((w . rms(s_i)) / sqrt(2304)) * rms(s_i)` over the stored blocks and the
current residual `cur`, with a learned query `w` per sublayer (`attn_res_query`, `ffn_res_query`, `output_res_query`).
Global layers have three attention queries (`attn_res_query` [2304, 3]) and so separate Q, K and V inputs. Layer 0
has no attention query and uses `x` directly.

**Attention** (application `il`, physical layer `p`):

- Inputs: SWA layers norm the mixed input with `attn_norm`; global layers norm the three inputs with
  `attn_q_inp_norm`, `attn_k_inp_norm`, `attn_v_inp_norm`.
- `q = W_q xq`, `k = conv_k(W_k xk)`, `v = conv_v(W_v xv)`, `r = W_rel xq` (16 per head), `g = W_gate xq`,
  `sink = W_sink xq` (one per head).
- Clipping: `q` is multiplied by `attn_q_clip` per channel and head, then every head's row is scaled to an L2 norm of
  at most 22.627 (`factor = clamp(22.627 / (|x| + 1e-6), 0, 1)`); `k` the same without a channel scale; `r` the same
  with `attn_rel_clip` and a threshold of 8. `v` is RMS-normed per head (no weight).
- Global layers multiply `q` and `r` by `tau = 1 + 0.1 ln(max(1, (pos + 1) / 8192))`.
- Scores: `s(h, j) = q_h . k_j / sqrt(128) + r_h . B[:, d] - sink_h`, where `d = pos - pos_j` and `B` is
  `attn_rel_bias` [16, 1024] with a zero column appended (any `d >= 1024` reads the zero column). The softmax
  includes one extra logit of 0 (the sink): `p_j = exp(s_j) / (1 + sum exp(s))`. SWA layers see the last 1024
  positions.
- Output: `o = conv_post(W_o (attention * sigmoid(g)))`.

**Short convolutions** (`convolve`, kernel 4). `conv(y)_t = (y_t + sum_{l=0..3} w_l y_{t-l} + dyn) / sqrt(5)`,
depthwise per channel, with the last three inputs kept per slot. On global layers K and V also get input-dependent
taps: `W_{k,v}_conv_proj xk` gives 4 taps for each of the 8 KV heads, `dyn = sum_l tap_{h,l} y_{t-l}`, and the scale
is `1/3` instead of `1/sqrt(5)`. The same convolution (static taps only) is applied to the attention output after
`W_o` (`attn_post_conv`) and to the MoE output (`ffn_post_conv`). Per slot the state is
`3 x (1024 + 1024 + 2304 + 2304)` values.

**MoE.** `h = rms(mix(blocks, ffn_inp, ffn_res_query)) * ffn_norm`. Router logits `l = W_router h` (f32);
selection takes the top 8 of `sigmoid(l) + exp_probs_b`; weights are `sigmoid(l)` of the selected experts normalized
to sum 1 (eps 1e-20) and multiplied by 8. Each expert: `[gate | up] = (W_gate_up h) * scale_gu[e]`,
`a = 4 tanh(gate / 4) sigmoid(gate) * 25 tanh(up / 25)` (llama.cpp's `LLM_FFN_SITU`), `y = (W_down a) * scale_d[e]`.
The per-expert scales exist only on the NVFP4 layers (0-19), and the gate/up scale has to be applied before the
activation. The MoE output goes through `ffn_post_conv` and is added to `ffn_inp`.

**Loop.** The prelude's output and block list are saved. Pass 1 runs layers 6-17 like any other layer (slots 6-17).
Passes 2 and 3 (slots 24-35 and 36-47) restart from the prelude's output and blocks. Before each of them, the
previous pass leaves a bank: its block appended at the core's midpoint (application 12) and its final output. For
core layer `p`, four messages `m_c = mix(bank, out, loop_msg_query_p[:, c])` (c = q, k, v, ffn) gate that layer's
inputs after the residual mix and before the norm:
`u' = u - tanh(W_f z) * u + tanh(W_g z) * m_c` with `z = W_down [rms(u); rms(m_c)]` (4608 -> 64 -> 2304). With the
messages, a SWA layer's Q, K and V inputs differ, so its three inputs are normed separately. The passes are strictly
sequential, because each pass needs the previous pass's output.

**Output.** `rms(mix(blocks, cur, output_res_query)) * output_norm`, then the Q8_0 head, `30 tanh(logits / 30)`, and
the padded rows dropped.

### GGUF layouts

Unsloth's root files use `general.architecture = inkling2` with loop keys and their own tensor names; the files in
`Shard_Rewrite/` hold the same tensors byte for byte under upstream's `inkling2_loop` names. The engine reads both:

| upstream (`inkling2_loop`) | Unsloth root (`inkling2`) |
|---|---|
| `blk.P.loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` | `blk.P.loop_{attn,k,v,mlp}_{down,gain,forget}` |
| `blk.P.loop_msg_query` [2304, 4] = queries for q, k, v, ffn | `loop_res_query` [2304, 24] and `loop_kv_res_query` [2304, 24]: for core layer `6 + i`, q = `res[2i]`, k = `kv[2i]`, v = `kv[2i+1]`, ffn = `res[2i+1]` |
| `blk.P.ffn_gate_up_exps.scale`, `ffn_down_exps.scale` | same |

Upstream llama.cpp at `29b52d398` drops the per-expert scales of passes 2 and 3: its generic scale pass runs on the
loop slots after they were copied from the physical layers. The reference build re-shares the slots after that pass
(a 7-line patch in `llama_model_base::load_tensors`), as Unsloth's notes describe.

## Weights and quantization

The release checkpoint stores the experts of layers 0-19 in NVFP4 (E2M1 values, an E4M3 scale per 16 and an f32
scale per expert), the experts of layers 20-23 in MXFP8, and everything else in BF16. Every Unsloth quant from
IQ4_XS up keeps the NVFP4 experts byte for byte; they differ only in how they quantize the rest. KLD is Unsloth's,
against UD-Q8_K_XL (BF16 plus the native NVFP4 experts, i.e. the release weights):

| file | formats | size | mean KLD | same top token |
|---|---|---|---|---|
| **UD-Q4_K_XL** | NVFP4 experts 0-19; Q8_0 everything else | 17.63 GB | 0.0107 | 98.0% |
| UD-Q4_K_M | adds Q5_K | 16.61 GB | 0.0108 | 97.9% |
| UD-IQ4_XS | NVFP4, Q8_0, Q6_K, Q5_K, Q5_0, Q4_K, IQ4_NL, IQ4_XS | 15.73 GB | 0.0109 | 97.7% |
| UD-Q3_K_XL | 11 formats | 13.63 GB | 0.0174 | 96.3% |
| UD-Q2_K_XL | 13 formats, IQ1/IQ2 experts | 9.14 GB | 0.0595 | 91.6% |

**Choice: UD-Q4_K_XL as published.** It needs two weight formats, NVFP4 (new) and Q8_0 (the existing
`qwen4_q8_*` kernels), no requantization, and has the best quality of the files. Our own runs
([KL divergence](#kl-divergence)) found that most of the published KLD comes from llama.cpp's CUDA default, which
multiplies the NVFP4 experts with 4-bit activations. With 8-bit activations, UD-Q4_K_XL is 0.0027 from UD-Q8_K_XL,
near the 0.0024 between two computations of the same weights. That leaves little for a per-tensor attribution to
resolve. A lossless
build (BF16 dense matrices and head, layers 20-23 repacked to MXFP8, about 17.9 GB) would read about 35% more bytes
per token; it is kept only if the measured quality difference justifies that.

NVFP4 on the D700s: GCN 1.0 has no FP4 or FP8 arithmetic, so the kernels decode E2M1 through a 16-entry table (as
IQ4_NL) and the E4M3 scale by bit manipulation or a 128-entry table, both amortized over 16 weights. ggml stores
NVFP4 in blocks of 64 (four scales, 32 bytes of values); tensor parallelism splits `ffn_down_exps`' 1152 inputs into
quarters of 288, which is not a multiple of 64, so the loader repacks to 16-value groups (lossless, same 4.5 bits per
weight).

Bytes per token (decode, UD-Q4_K_XL): experts 44 applications x 8 x 4.48 MB (NVFP4) + 4 x 8 x 8.46 MB (Q8_0) =
1.85 GB; dense projections 48 x 29.7 MB = 1.43 GB; loop controllers (f32) 24 x 9.4 MB = 0.23 GB; head 0.49 GB:
about 4.0 GB, against 17.3 GB for Qwen3.8-27B. The loop is why the dense part is so large: the core's weights are
read three times.

## Machines

One Mac holds 2 x 5869 MiB. The model's resident weights are 17.1 GB at UD-Q4_K_XL (15.1 GB with 4-bit experts in
layers 20-23), so **two Macs are the minimum**. One Mac would need 2-bit experts (KLD 0.06 at best, and a dozen new
formats) or experts in host RAM behind a VRAM cache.

**Two Macs, TP4** (the layout of Qwen3.8-27B, `qwen/tp.rs`): every GPU holds a quarter of every layer.

| per GPU | |
|---|---|
| attention | 6 Q heads and their 2 KV heads, with their rows of `W_q`, `W_gate`, `W_rel`, `W_sink`, their K/V convolution channels and their columns of `W_o` |
| MoE | a quarter of every expert's 1152 (288 rows of gate and of up, 288 columns of down) |
| replicated | router (f32), loop controllers, norms and residual queries |
| head | a quarter of the 201,024 rows |
| weights | about 4.4 GB |
| K/V | 1 KB per position per slot (f16): SWA slots about 51 MB; the 8 global slots 268 MB at 32K, 1.07 GB at 128K |

128K context fits with f16 K/V; Q8 K/V doubles that. Each application ends two segments with partial sums
(attention, MoE), so a decode token has 96 all-reduces plus the head's candidates (the 27B has 128). The host keeps
the residual as in Qwen TP: it also holds the block states, computes the residual mixes, the post-attention and
post-MoE convolutions (they act on the all-reduced 2304-vector) and the loop messages, and writes each segment's
normed inputs. Estimate: about 7 ms of weight streaming per token at the 27B's effective bandwidth, 3.5-4 ms of
exchanges and a few ms of per-segment overhead, so 13-15 ms per token (target 50+ tok/s, plausible ceiling ~70).

**Alternative, not planned:** a pipeline with the loop core on one Mac and the prelude, coda and head on the other
keeps the three passes off Thunderbolt (two hops per token) and suits prefill, but each Mac then streams with two
GPUs' bandwidth (an estimated 17-18 ms per token) and it needs a pipeline of TP groups that the engine doesn't have.
It is the fallback if exchanges dominate TP4.

**Four Macs, TP8:** 3 Q heads and 1 KV head per GPU, 144 expert rows. Mainly more context (256K with f16 K/V) or
room for the lossless build; the decode gain will be smaller than the 27B's (+43%), because each GPU's share of a
layer becomes very small.

## Vision

The projector (`mmproj-BF16.gguf`, 318M parameters, 0.64 GB) is a ViT run once per image:

- Preprocessing: RGB normalized as `(x - 127.5) / 127.5`, padded right and bottom to a multiple of 40 pixels with a
  pre-normalization value of -1, no resizing unless the image exceeds 4096 merged tokens (then shrunk with its aspect
  ratio kept).
- Patch embedding 20 x 20 x 3 -> 1024, plus a learned 32 x 32 position table resized bilinearly to the patch grid; a
  CLS token (with its own position embedding) in front.
- 23 pre-norm blocks: LayerNorm (eps 1e-5), 16-head attention with biases and 2D RoPE (base 10000; patch positions
  1-indexed, CLS at 0), layer scales `ls1`/`ls2`, MLP 1024 -> 4096 -> 1024 with erf GELU; `pre_ln` before the first
  block.
- CLS dropped, 2 x 2 patches merged (4096), projector `mm.1` 4096 -> 4096, erf GELU, `mm.2` 4096 -> 2304, no biases.

Implementation (`inkling/vision.rs`, shaders `vit_*.comp`): the tower runs on the driver's first GPU, one submission
per block (a 4096-token image keeps every GPU job well under amdgpu's 2 s limit). The linear layers are quantized
from BF16 to Q8_0 at load and use the Q8_0 GEMM; LayerNorm, the 2D RoPE, the bias/GELU/layer-scale steps and the
attention (16 queries per workgroup, K/V chunks of 32 staged in LDS, online softmax) have their own kernels. The
patches and the resized position table are built on the host, as is the 2 x 2 merge. Against llama.cpp's mtmd (CPU,
BF16 projector) on a 480 x 360 PNG (108 image tokens), the projector output agrees to 2.1% relative error (cosine
0.99978), and it takes 0.31 s on one D700. `trashcan inkling vision-check MMPROJ IMAGE --reference DIR` repeats that
comparison; `inkling chat --mmproj MMPROJ --image IMAGE` and `serve --mmproj MMPROJ` (`input_image` parts as base64 data
URLs) take images. The image rows enter the language model in place of the placeholder, skip `token_embd_norm`, and
reach the other machines as one binary message with the forward.

## Speculative decoding

The release ships a DSpark block drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`, 0.40 GB; BF16 0.65 GB;
`general.architecture = dflash`), whose llama.cpp support is an open pull request (#32218). Its math, from that
pull request:

- **Context.** For every token the target accepts, the residual entering applications 5, 23 and 41 (in execution
  order: the prelude's last layer, the second pass's sixth, the third pass's last) and the final hidden state (after
  the output mix and `output_norm`) are kept as four 2304-vectors. Draft layer `l` mixes them with
  `softmax(dspark_fuse[l])` (4 weights per layer), RMS-norms the mix with `dflare_norm`, and writes
  `rms(W_k_ctx c) * k_ctx_norm` (rotated at the token's position, RoPE base 1e6 on all 128 dimensions) and `W_v_ctx c`
  into its K/V cache. A fourth row of `dspark_fuse` mixes the same features into `f_rnn` (normed with
  `dspark_norm`), the recurrent head's context.
- **Block.** A draft step runs a block of 8 rows: the last accepted token's embedding (the target's `token_embd`,
  RMS-normed with `token_embd_norm`) and 7 copies of `mask_embd`. Three pre-norm layers: attention with 24 Q / 8 KV
  heads of 128, per-head Q and K norms, RoPE, non-causal within the block, over the context entries and the block
  (sliding window 1025 on layers 0 and 2); a SiLU-gated MLP of 9216. `output_norm`, then the target's head gives the
  base logits of every row.
- **Markov head.** Rows 1..7 refine their logits in order: `joint = joint_proj [state; markov_w1[prev]; row; f_rnn]
  + b` (5120 -> 768) splits into a gate `sigmoid`, a candidate `tanh` and a latent `tanh`; `state = cand + gate *
  (state - cand)` (rank 256, from zero); the logits are `30 tanh((base + head(hidden_out latent + b)) / 30)`, and
  `prev` is their first maximum (the anchor token for row 1).

Both terms inside the softcap use the same linear vocabulary head. The engine adds the normalized block row to
the Markov correction before projecting it, which lets each recurrent step score its token in one pass over the
head. Target verification restores K/V positions and convolution histories to the accepted prefix.

### Implementation (`inkling/draft.rs`, `inkling/spec.rs`)

Every rank holds the whole drafter and the head rows of its quarter of the vocabulary. The target's forwards keep
the four feature rows of each token (`Group::drive`); once tokens are committed the host mixes them
(`DraftHost::contexts`) and each rank writes the context K/V of its own copy in one submission. A round
(`spec::step`) after the pending token `t`:

1. **Draft.** Every rank runs the block and the first Markov step after `t` in one submission, then one submission
   per further step. For each head, the eight queries share a tile of keys and values. Long key ranges split over
   workgroups and a second kernel merges their softmax results. Ranks exchange their first maxima (value, then
   lowest id), 8 bytes per round of the exchange.
2. **Verify.** One gated forward of `[t, d1..dk]` (at most 8 rows) scores every row. Before it each slot saves its
   K/V convolution history and the batch's pre-convolution K/V rows (`inkling_hist_restore` reads them back); the
   host holds its attention- and MoE-output histories at the start of the round and saves the pre-convolution rows.
3. **Accept.** The target samples row 0; while its sample equals the next draft, that draft is kept and the next row
   is sampled. The first mismatch is the new pending token. The acceptance procedure preserves the target's
   sampling distribution given the same logits. Batched and single-row kernels can round logits differently.
4. **Roll back.** Position, GPU and host convolution histories return to just after the accepted rows (the K/V
   cache needs nothing: later rows overwrite rejected positions before they are read), and the drafter's context
   gets the accepted tokens.

`--draft dspark-Inkling-2-Mini-Q8_0.gguf` enables drafting for `chat`, `bench`, `validate` and `serve`. Put the file
at the same path on both Macs. `--draft-tokens` defaults to 3 and accepts 1 through 7. Set `--batch` to at least
one more than `--draft-tokens`. `bench` reports `tg_tokens_per_round` and the mean milliseconds per round in
block, steps, verify and rollback (`draft_round_ms`). `validate --draft ...` uses drafting for the free greedy run.

## Chat format

`tokenizer.ggml.pre = inkling2` is GPT-4o's pre-tokenizer regex; no BOS. Messages are
`<|message_ROLE|><|content_text|>...<|end_message|>` with roles `user`, `model`, `system`, `tool` (a tool message
or a named assistant appends the name to the role token); reasoning is `<|content_thinking|>`, tool calls are
`<|message_model|>NAME<|content_invoke_tool_json|>{"name":...,"args":{...}}<|end_message|>`, tools are declared in
`<|message_system|>tool_declare<|content_xml|>` + JSON. Before the first non-system message the template emits
`<|message_system|><|content_text|>Thinking effort level: E<|end_message|>` with E from 0 to 0.99 (none 0, minimal
0.1, low 0.2, medium 0.7, high 0.9 (default), xhigh/max 0.99). An assistant turn ends with
`<|content_model_end_sampling|>` (200006, the stop token). Sampling defaults: temperature 1, no top-k/top-p.

## Validation

The reference is llama.cpp `29b52d398` with the loop-scale patch, built with CUDA and run on an RTX PRO 6000, on the
`Shard_Rewrite` files: UD-Q8_K_XL (the release weights) for quality and KLD baselines, UD-Q4_K_XL for activations to
compare against the engine with the same weights. There is no separate CPU reference in the engine: each GPU stage
is checked against `tools/dump-activations` output from the reference, layer by layer on llama.cpp's own inputs (as
`qwen4 layer-check` does), then end to end (logits and greedy tokens). The per-layer checks use dumps from the CPU
backend. They are a consistency check, not an exact bound: ggml's CPU backend rounds activations to 8 bits in every
quantized product and stores K/V in f16 (see [KL divergence](#kl-divergence)). Vision: the projector's output for fixed
images, then the image prompt's logits.

On `80ce8c6b`, the two-Mac run compared all 48 layer applications on 25-token chat and 109-token tool prompts.
The worst relative residual errors against the same-quant llama.cpp CPU dumps were 2.983% and 2.864%.
Plain and three-token DSpark runs both matched 16/16 greedy tokens on each fixture and 24/24 after a
2,761-token document. These tokens also matched the archived Voyage UD-Q8_K_XL CUDA runs. The 108-row vision
fixture agreed with the llama.cpp mtmd projector to 2.10% relative error (0.999781 cosine similarity). The full
image chat produced identical reasoning and answer with and without drafting, over 65 generated tokens.
The Responses server also matched with and without drafting on streamed 1- and 4-token limits, two seeded
samples in JSON and SSE, an image request, and a native tool-result continuation.

### KL divergence

`trashcan inkling logits MODEL IDS OUT` runs chunks of a token stream through the group and writes the full softcapped
logits of every position (each rank runs the head over the kept output rows and the machines gather their vocabulary
slices; nothing changes on the prefill and decode paths). `tools/dump-logits` writes the same files from llama.cpp,
and `scripts/inkling-kld.py` compares two sets over the whole vocabulary in float64. All runs below use the same token
ids: the quality corpus as llama-perplexity tokenizes it, in 21 chunks of 2048 tokens, each from an empty context in
batches of 512, with every position scored (42,987 positions). The engine ran on `69c26fab` on invite and indite.

A llama.cpp reference without activation rounding comes from `tools/dump-logits/dequant`: it writes UD-Q4_K_XL with
the Q8_0 tensors converted to F32 and the NVFP4 tensors to F16, both exactly, so ggml multiplies by the same weights.
That run uses the CPU backend with an f32 K/V cache. It still rounds the NVFP4 experts' activations to f16. It was run
on every third chunk (14,329 positions). It is called "f32 reference" below.

| P | Q | positions | mean KLD | median | p99 | max | same top token |
|---|---|---:|---:|---:|---:|---:|---:|
| f32 reference | engine | 14,329 | 0.00126 ± 0.00004 | 0.00013 | 0.017 | 0.26 | 98.8% |
| f32 reference | llama.cpp CPU | 14,329 | 0.00238 ± 0.00007 | 0.00031 | 0.029 | 0.35 | 98.5% |
| f32 reference | llama.cpp CUDA, 8-bit activations | 14,329 | 0.00244 ± 0.00008 | 0.00030 | 0.028 | 0.42 | 98.5% |
| llama.cpp CPU | engine | 42,987 | 0.00238 ± 0.00004 | 0.00020 | 0.032 | 0.41 | 98.6% |
| llama.cpp CPU | llama.cpp CUDA, 8-bit activations | 42,987 | 0.00243 ± 0.00005 | 0.00020 | 0.031 | 1.38 | 98.5% |
| llama.cpp CUDA, 8-bit activations | engine | 42,987 | 0.00240 ± 0.00004 | 0.00019 | 0.032 | 0.52 | 98.5% |
| llama.cpp CPU | llama.cpp CUDA, default | 42,987 | 0.00667 ± 0.00011 | 0.00057 | 0.076 | 1.38 | 97.4% |
| UD-Q8_K_XL, CUDA, 8-bit activations | UD-Q4_K_XL, same | 14,329 | 0.00269 ± 0.00008 | 0.00036 | 0.033 | 0.43 | 98.4% |

Every row except the last uses UD-Q4_K_XL. The ± values are standard errors of the mean. The two GGUF layouts hold the
same tensors: `tools/inkling-gguf-compare.py` found all 744 `Shard_Rewrite` tensors byte for byte in the root file,
with the loop message queries rearranged as in [GGUF layouts](#gguf-layouts).

What this shows:

- The engine is closer to the f32 reference than either llama.cpp backend is. Its KLD from the f32 reference is about
  half of the CPU backend's (0.00126 against 0.00238) on the same positions, and it agrees on the top token more often.
  The engine-versus-CPU KLD (0.00238) is no larger than CPU versus CUDA with 8-bit activations (0.00243). It equals
  the CPU backend's own distance from the f32 reference. The per-layer errors against the CPU dumps and the higher
  free-run top logits on the fixtures are therefore not evidence of an engine error.
- The divergences are not uniform logit shifts. Between the CPU backend and the engine, the probability-weighted mean
  logit difference averages 0.14 in magnitude per position and the softmax removes it. A spread of 0.04 remains around
  that mean, and it explains the KLD (correlation 0.81, against 0.05 for the shift). KLD does not grow with the
  position in the chunk: the means of the eight 256-position bins range from 0.0020 to 0.0027.
- This model amplifies small numerical differences. Every pair of computations, including the f32 reference against
  the engine, has its largest KLD at the same positions: the Spearman rank correlations of per-position KLD between
  pairs are 0.87–0.92. Those positions are uncertain ones: where KLD is in the top 1%, the mean top-token probability
  is 0.44, against 0.76 overall. The top 1% of positions carries about 26% of each pair's total.
- llama.cpp's CUDA default is much noisier than the CPU backend. On Blackwell it multiplies the NVFP4 experts with
  4-bit activations. With `GGML_CUDA_MMQ_PREC=q8` the CPU-versus-CUDA KLD drops from 0.0067 to 0.0024, and that
  setting alone accounts for 0.0067 between two CUDA runs. The archived llama-perplexity baseline (UD-Q8_K_XL, CUDA
  default) gives UD-Q4_K_XL a KLD of 0.0080 on CUDA. Against that baseline, over the same 21,483 second-half positions,
  the engine scores 0.0067, the CPU backend 0.0067 and CUDA with 8-bit activations 0.0068. So most of the
  0.008–0.011 published for UD-Q4_K_XL reflects the arithmetic of the CUDA default. With 8-bit activations on both sides,
  UD-Q4_K_XL is 0.0027 from UD-Q8_K_XL, close to the 0.0024 that separates two computations of the same weights.

What it does not show: the f32 reference still rounds the expert activations to f16, so the engine's distance from
exact arithmetic is not measured. The table also cannot say which of the engine's roundings dominates its 0.00126.
It shares the f16 K/V cache with the CPU backend, which is one reason their errors are correlated. The raw outputs and
commands are archived in `results/inkling/kld-69c26fab` on Pinkerton.

**The f16 wire.** Decode and verification all-reduces (fewer than 65,536 values) can carry each machine's partial
sums rounded to f16 instead of exact f32 sums. Prefill-sized all-reduces stay f32, so this was measured with
`--batch 1`, every token a single-row forward, on the seven f32-reference chunks (14,329 positions, `bee58366` on
invite and indite):

| P | Q | mean KLD | median | p99 | same top token |
|---|---|---:|---:|---:|---:|
| f32 reference | engine, batch 1, f32 wire | 0.00128 ± 0.00004 | 0.00013 | 0.017 | 98.72% |
| f32 reference | engine, batch 1, f16 wire | 0.00130 ± 0.00004 | 0.00014 | 0.017 | 98.80% |
| engine, batch 1, f32 wire | engine, batch 1, f16 wire | 0.00087 ± 0.00003 | 0.00008 | 0.012 | 99.13% |
| engine, batch 512, f32 wire | engine, batch 1, f32 wire | 0.00055 ± 0.00002 | 0.00002 | 0.009 | 99.41% |

The f16 wire moves the output about as much as running the single-row kernels instead of the batch kernels does,
and no further from the reference: paired by position, its distance from the f32 reference is 0.000016 ± 0.000026
above the f32 wire's. It is therefore the default outside `validate` and `logits` (see [Running](#running)); the raw
outputs are in `results/inkling/kld-wire-bee58366` on Pinkerton.

## Performance

On incite and enlight (four D700s over Thunderbolt), `inkling bench MODEL --pp 1024 --tg 64 --reps 3` at batch
256 measured 424.5–428.6 tok/s prefill and 35.7–35.9 tok/s decode on `80ce8c6b` with drafting disabled. The
`37d0740f` baseline measured 115.7 and 29.1 tok/s for the same workload. The measured gains come from overlapping
host work and GPU segments in two micro-batches, reusing weights across expert pairs, grouping attention by KV
head, and reducing the cost of projections, controllers and copies. Both Macs ran with the performance governor,
a command-scoped 0 µs CPU latency request, Linux `7.3.0-rc7-p2p` and RADV 25.0.7. `scripts/with-clocks.sh`
recorded active clocks of 850 MHz core and 1370 MHz memory on all four D700s.

On the same build and pair, `inkling bench MODEL --prompt 'Write a Python function that returns the n-th Fibonacci
number, with a docstring.' --tg 192` ran at 37.86–38.02 tok/s across four plain runs split before and after three
drafted runs. `--draft DRAFT --draft-tokens 3` measured 43.11–43.16 tok/s (+13–14%) and averaged 3.00 emitted tokens
per round, including the pending target token. Mean time per round was 9.6 ms for the drafter block, 2.1 ms for its
Markov steps, 56.6 ms for verification and 1.2 ms for rollback and context updates.
At a 32,768-token context capacity the benchmark reported 4,846 MiB per GPU for the target and 5,486 MiB with
DSpark loaded, before loading the vision projector.

For a 2,770-token document rendered as a low-effort chat prompt, `inkling bench MODEL --prompt "$document" --tg 128
--reps 2` on `f2b51e5b` measured:

| Draft tokens | Prefill tok/s | Decode tok/s | Prefill plus decode |
|---|---:|---:|---:|
| 0 | 392.2–393.0 | 35.54–35.57 | 10.65–10.66 s |
| 3 | 374.9–375.6 | 42.25–42.33 | 10.40–10.42 s |
| 5 | 374.8–375.7 | 36.12–36.22 | 10.91–10.93 s |

Drafting costs additional prefill time to build its context. Three drafts saved about 0.24 seconds overall on
this document and output length. Five drafts took longer overall.

### Prefill batch size

This section describes `f3ca631d`; the defaults and numbers after the later prefill work are in
[the next one](#prefill-inputs-in-vram-four-micro-batches-and-16-bit-controllers).
Inkling chat, bench and the Responses server default to 512 prompt tokens per batch. A full batch runs as
two 256-row micro-batches. Each GPU reserves scratch space and sliding-window cache headroom for the largest
micro-batch it executes. Prompts below 64 tokens run whole, as do teacher-forced comparisons. At 32K context capacity
the default target allocations total 4,846 MiB per rank, the same as batch 256 on `26adafade0`.
For long prompts, `--batch 1024` uses 4,902 MiB per rank before the drafter and projector.

Measured on incite and enlight with UD-Q4_K_XL, f32 exchanges, the performance governor, a 0 µs CPU latency
request and active clocks of 850/1370 MHz (core/memory) on all four D700s (`scripts/with-clocks.sh`). Each case used
`inkling bench MODEL --context 32768 --batch B --tg 32 --reps 3` with `--pp N` for synthetic tokens or
`--prompt "$document"` for the low-effort document. The table gives warm medians (last two repetitions).
Prefill time includes the final scored row. The table uses greedy decode with drafting disabled.

| Prompt | Published batch 256 (s, tok/s) | Batch 512 (s, tok/s) | Batch 1024 (s, tok/s) | Decode tok/s, published / 512 |
|---|---:|---:|---:|---:|
| 64 tokens | 0.292, 219 | 0.253, 253 | — | 39.1 / 38.9 |
| 1,024 tokens | 2.392, 428 | 2.178, 470 | 2.107, 486 | 36.5 / 36.4 |
| 2,770-token document | 7.050, 393 | 6.380, 434 | 6.043, 458 | 36.6 / 36.3 |

Starting the overlap at 64 tokens also reduced 96-token prefill from 0.395 to 0.299 s (243 to 321 tok/s)
at fixed batch 512. Both configurations decoded at 38.8 tok/s.

Changing only the batch on the published binary also gave 469 and 434 tok/s at batch 512 on the two
longer prompts. At that batch size, 16 expert pairs per wave dropped the warm 1K result to 426–427 tok/s.
Scaling each Q8 weight before accumulation gave 466–467 tok/s. A Q8 workgroup covering 64 rows and
16 columns gave 461–462 tok/s. The raw runs and candidate sources are archived under
`results/inkling/prefill-f11b03b5b7` on Pinkerton.

With three DSpark drafts, warm document prefill rose from 376 to 413 tok/s (7.37 to 6.70 s),
with decode at 48.2 tok/s on both builds. Both allocated 5,486 MiB per rank including the drafter.
The default matched all 48 reported llama.cpp layer errors for both chat and tool fixtures
(worst 0.02983 and 0.02864), all 16 greedy tokens for each, and all 24 after the 2,761-token
prompt with and without drafting. The 108-token projector fixture retained cosine similarity 0.999781.

With DSpark and the BF16 projector loaded, a 2,770-token Responses request with one output token
and zero cached input tokens took 7.39 s on the published server, 6.71 s at batch 512 and 6.38 s
at batch 1024. During the document, image and tool checks, peak sampled driver VRAM was
6,086 MiB on the projector GPU and 5,573 MiB on the other GPU at all three settings.

### Prefill inputs in VRAM, four micro-batches and 16-bit controllers

This section describes `4837aacb`; the batch sizes and numbers of the section above are those of `f3ca631d`.
`inkling chat`, `bench` and `validate` now default to 1,024 prompt tokens per batch (the Responses server keeps its
`--batch` default of 512, and the two-Mac recipe passes `--batch 1024`). A batch runs as up to four micro-batches
of at most 256 tokens: two up to 512 tokens, three up to 768, four up to 1,024, and above that four of a quarter
of the batch each. The GPUs hold scratch for the largest micro-batch, so batch 1024 has the kernel scratch that
batch 512 had.

At 1,024 synthetic tokens and batch 512 the starting point (`51aa4c75`) spent 562 ms of GPU time per 256 tokens on
the driver's first rank and 569 ms of wall time, so the GPUs were busy 98.7% of the time, while the host thread
waited for them during 41% of it (0.94 s of 2.28 s; 0.90 s in exchanges, 0.36 s in its passes over the rows,
0.07 s adding the local ranks). The GPU time was 264 ms in the expert kernels, 118 ms in the Q8_0 projections,
38 ms copying inputs from host memory, 22 ms copying outputs to it, 48 ms in the loop controllers, 32 ms in
attention, 17 ms in the input norm, 10 ms in the router and 11 ms in the small kernels. The 2,770-token document
took 622 ms per 256 tokens: its experts cost 13% more (real text fills the expert tiles less evenly, and its last
batch is short) and its attention 57 ms instead of 32. A 64-token prompt took 1,050 ms per 256 tokens, 95% of the
wall time, with every kernel two to three times less efficient on 32-token micro-batches.

What changed, with the tok/s at 1,024 tokens on inflight and alight after each step:

| Step | tok/s | GPU ms per 256 tokens |
|---|---:|---:|
| `51aa4c75`, batch 512 | 449.9 | 562 |
| Prefill inputs written by the host straight into BAR-mapped VRAM | 485.2 | 513 |
| MoE outputs copied to host memory under the next submission's kernels | 488.5 | |
| Four micro-batches (batch 1024), every output's copy deferred | 516.3 | 492 |
| Controllers and router read from 16-bit copies; second controller kernel rewritten | 545.5 | 465 |

- **Inputs in VRAM.** Prefill batches take their input rows in a buffer of the 256 MiB VRAM heap that the PCI BAR
  maps. Twelve host threads write it at 8 GB/s per GPU (two GPUs at once; one thread reaches 4.4 GB/s, and reading
  it back from the host runs at 0.04 to 0.37 GB/s, so outputs still leave through a copy). The controllers and
  the norm read it in place: the copy from host memory is gone and the norm fell from 17 to 4 ms. The host's
  passes rose from 0.36 to 0.41 s per 1,024 tokens. Every host thread ends its rows with a store fence.
- **Deferred output copies.** A segment leaves its partial output in VRAM, and the next submission, which is
  another micro-batch's segment and already queued, copies it to host memory right after its first long dispatch.
  The command processor then copies while the shader cores work. The copy ends with a mark behind the output that
  the host polls. With two micro-batches only the MoE outputs can wait: deferring the attention outputs as well
  gave 485.3 tok/s against 485.4 without any deferral, because the host is late with the carrying submission.
- **Four micro-batches.** Once the input copy was gone, the host's work on a segment (about 2.5 ms of exchange
  for 2.36 MB each way at the cable's rate and 1 ms for its pass) no longer fitted behind the other micro-batch's
  attention segment (about 3.2 ms by the kernel times), and the GPUs idled. With three other micro-batches in flight it does. At batch 1024
  without deferred copies the result was 495.3 tok/s. Three micro-batches of 342 tokens gave 511.6.
- **16-bit controllers and router.** The loop controllers' matrices and the router are f32 tensors whose lower
  16 bits are zero. The prefill kernels read 16-bit copies and rebuild the same values: the first controller
  kernel went from 29.1 to 14.2 ms per 256 tokens and the router from 10.2 to 6.0, both having run at the
  bandwidth of their weights. The second controller kernel held 128 weights per lane in 140 registers, which
  leaves one wave per SIMD; walking them four ranks at a time for 16 tokens needs 44 registers and took it from
  18.4 to 9.3 ms. The results are bit-identical. The decode kernels read the same 16-bit matrices
  (24.4 to 23.0 and 18.3 to 15.9 us per controlled segment on incite), so a shard holds no f32 copy of them.
- **DSpark context.** After a prefill batch the drafter's context took 114 ms per 1,024 tokens: 13 ms for the
  rows' final hidden states and 6.4 ms per 64-token chunk, of which recording and running the GPU work took
  2.4 ms and one host thread mixing features the rest. Those loops now run on the host pool: document prefill
  with three drafts went from 463.9 to 478.5 tok/s, against 490.1 without the drafter.

Final numbers on inflight and alight, `4837aacb` at its default batch against `c2924857` (the base it is rebased on)
at its default of 512, alternated in one session, UD-Q4_K_XL, f32 exchanges, 32K context, performance governor,
0 µs CPU latency request, 850/1370 MHz on all four D700s, warm medians of three repetitions with 32 decoded tokens:

| Prompt | `c2924857` (s, tok/s) | `4837aacb` (s, tok/s) | Decode tok/s, before / after |
|---|---:|---:|---:|
| 64 tokens | 0.266, 240.8 | 0.252, 254.0 | 41.1 / 41.1 |
| 1,024 tokens | 2.273, 450.6 | 1.875, 546.2 | 38.7 / 38.6 |
| 2,770-token document | 6.759, 409.8 | 5.641, 491.0 | 38.6 / 37.7; 38.0 / 38.0 over 64 tokens |
| Document, three drafts | 7.075, 391.5 | 5.778, 479.4 | 54.9–56.3 / 56.0–56.4 |

Before the rebase, 4,096 tokens went from 427.0 to 509.9 tok/s. At `--batch 512` the new build measured 508.2 tok/s
at 1,024 tokens and 465.4 on the document; at `--batch 2048` (four 512-token micro-batches) 522.3 on the document.
The decode GPU time after the document is 19.98 ms per token on both builds.

This pair is slower than incite and enlight on the same binary (450 against 470 tok/s at 1,024 tokens on
`51aa4c75`), and inflight sets its pace: its D700s take 562 ms per 256 tokens where alight's take 534. The
difference is in the kernels that read the most weight bytes (the Q8_0 q/gate/r projection 88.7 against 78.2 ms,
the Q8_0 experts 28.8 against 23.6). Inflight's cards carry another revision of Hynix GDDR5 (`MC_SEQ_MISC0`
0x500056a9 against 0x500066aa) and run looser memory timings at the same 1370 MHz (`MC_ARB_DRAM_TIMING`
0x1a17090c against 0x14110709 under load). A tensor-parallel group waits for its slowest GPU at every segment.

Memory at 32K context: 4,854 MiB per rank at batch 1024, against 4,846 before the prefill work: 57 MiB of input
buffers in the BAR heap (28 at batch 512), 7 MiB more of per-micro-batch outputs and 14 MiB less of controller
scratch, while the controllers' 16-bit matrices take 56 MiB less than the f32 ones they replace (the router keeps
both forms on shards that run prefill batches). Before the decode kernels read the 16-bit matrices the shards
held both (4,962 MiB), and with the drafter loaded the BF16 projector no longer fitted on its GPU at batch 1024:
the device arena grows in 256 MiB chunks, the kernel moved buffers to host memory, and the 108-token image encode
took 0.66 s instead of 0.30 s. With one copy it fits again (0.30 s on incite and enlight, the same answer).

Validation of `4837aacb`: all 48 teacher-forced layer errors of the chat and tool fixtures equal the published
ones (worst 0.02983 and 0.02864), 16/16 greedy tokens each and 24/24 after the 2,761-token prompt, with and
without three drafts; the printed top logits of the plain runs are unchanged to the last digit.

Tried and dropped, in GPU ms per 256 tokens at 1,024 tokens:

- A Q8_0 GEMM whose waves each walk whole rows without shared memory (64 or 48 registers, four or five waves per
  SIMD): 115 and 108 for the q/gate/r projection against 89.5.
- The gate/up expert kernel loading a block (16 bytes) or two per row at a time instead of a 64-byte line (48 and
  64 registers instead of 84): 451 and 250 against 155. The waves of a compute unit evict each other's lines, so
  a line is fetched again for every load that touches it.
- The same kernel loading its scales for 16 blocks at once (128 registers, two waves per SIMD): 168 against 155.
- The first controller kernel with 16 tokens per workgroup on the 16-bit weights: 21.2 against 14.2.

Where the time is now (465 ms per 256 tokens at 1,024, GPUs 99% busy): experts 264 ms, Q8_0 projections 119,
attention 31, controllers 23, router 6, small kernels 15. With its scale loads removed the q/gate/r projection
takes 75.7 ms instead of 89.6, and with constant inputs 81.3 (both give wrong results; they bound what those
loads cost). The host thread is 80% busy (1.04 s of exchanges, 0.38 s of passes and 0.06 s of sums in 1.875 s),
so at this exchange rate the host caps two Macs near 690 tok/s. The exchange runs at the cable's rate (9.6 Gb/s
per direction on one cable, [hardware.md](hardware.md)); a second cable between the pair should about halve it
(not tried).

Scripts, logs, clocks, VRAM traces and the dropped candidates' sources are in `results/inkling/prefill-4837aacb`
on Pinkerton.

### Small verification batches

`ad764296` uses decode matrix kernels for 5–8 rows, schedules one host task per row in gated batches, and selects
each rank and row's candidates on the existing host pool. The paired runs below use `df18d039` as the baseline,
the same prompts and D700s, the f32 wire format, and `--reps 2` for each build and workload.
Both Macs sustained 850 MHz core and 1370 MHz memory during the paired runs.

On the 33-token code prompt with `--tg 192`, plain decode went from 37.87–38.01 to 38.42–38.48 tok/s. With
`--draft DRAFT --draft-tokens 3`, it went from 43.16–43.23 to 46.07–46.22 tok/s at the same 3.00 emitted tokens
per round. Verification went from 56.49–56.59 to 51.98–52.17 ms per round.

The same 2,770-token document with `--tg 128` gave these paired results:

| Draft tokens | Baseline decode tok/s | Small-batch decode tok/s | Baseline prefill plus decode | Small-batch prefill plus decode |
|---|---:|---:|---:|---:|
| 3 | 42.28–42.38 | 44.78–44.86 | 10.39–10.42 s | 10.22–10.25 s |
| 5 | 36.13–36.16 | 42.30–42.34 | 10.91–10.94 s | 10.39–10.41 s |

Document prefill was about 375 tok/s in both builds with drafting enabled. With five drafts, mean verification
fell from 85.36–85.44 to 69.66–69.77 ms per round at the same 3.88 emitted tokens per round.

The five-draft chat and seven-draft tool fixtures matched all 48 teacher-forced layer error measurements from the
baseline and all 16 reference tokens per fixture. Five drafts also matched all 24 reference tokens after the
2,761-token prompt. The five-draft Responses server matched the published plain server on streamed token limits,
two seeded samples in JSON and SSE, vision, and the native tool-result continuation.

The native optimizer also tried reusing the prefill expert tiles in small verification batches. In its four-row
probe after a 64-token prompt, that increased verification from 57.55 to 67.19 ms. The gate/up portion alone
rose from 8.25 to 16.05 ms. Small batches use the per-token expert kernels.

### DSpark decode on the released small-batch verification path

The following runs compare `84b41601b944` with a fresh build of `26adafade0` on incite and enlight.
Each `inkling bench` command used the same prompt on both builds, two repetitions, batch 256, context capacity
32,768 and an f32 wire. The code prompt generated 192 tokens and the document generated 128. All four D700s
sustained 850/1370 MHz core/memory during decoding.

| Prompt | Draft tokens | Baseline tok/s | Updated tok/s | Emitted tokens/round, both builds |
|---|---:|---:|---:|---:|
| Code, 33 tokens | 0 | 38.74–38.77 | 38.69–38.74 | 1.00 |
| Code, 33 tokens | 3 | 45.94–46.11 | 48.91–48.96 | 3.00 |
| Document, 2,770 tokens | 3 | 44.71–44.89 | 50.77–50.94 | 3.28 |
| Document, 2,770 tokens | 5 | 42.20–42.33 | 46.38–46.42 | 3.88 |

A second baseline run after the updated build gave 38.56–38.67 tok/s plain and 46.06–46.08 tok/s drafted
on code, and 44.56–44.77 tok/s on the document with three drafts.

On the document with three drafts, prefill plus decode fell from 10.22–10.24 s to 9.89–9.91 s. Prefill alone
was 375.3–376.0 versus 374.9–375.5 tok/s. Device usage with DSpark rose by 4.65 MiB per rank.
Mean milliseconds per round across the two repetitions were as follows. Rollback includes drafter context updates.

| Prompt and drafts | Drafter block, before → after | Further Markov steps | Verification | Rollback |
|---|---:|---:|---:|---:|
| Code, 3 | 9.56 → 6.59 | 2.07 → 2.05 | 52.26 → 51.21 | 1.15 → 1.31 |
| Document, 3 | 16.21 → 8.59 | 2.04 → 2.01 | 53.88 → 52.63 | 1.00 → 1.20 |
| Document, 5 | 16.23 → 9.22 | 4.05 → 3.99 | 69.84 → 68.60 | 1.48 → 1.58 |

The head change alone saved about 2 ms per round on both prompts. On the document, sharing tiled K/V across the
eight queries and splitting long key ranges saved another 4.9 ms. With three drafts, the final block layer
needs only the first four rows after attention. Limiting its output projection and MLP to those rows saved
0.74 ms on code. Deferring host convolution history updates until acceptance lowered four-row verification
from 52.14–52.20 to 51.26–51.31 ms in an independent code run.

The code prompt's 126 full four-row rounds covered 6,048 layer visits. The 32 routed slots per visit selected
19.64 distinct experts on average. Pairing slots for the same expert could share 28.1% of gate/up weight walks.
A kernel that computed those pairs together increased verification to 52.90–53.01 ms and lowered decode from
47.72–47.73 to 47.10–47.20 tok/s on the same drafter. That candidate was discarded.

The first tile layout used 33.5 KiB of workgroup shared memory and exited during pipeline creation on D700.
Reusing one tile for keys and values reduces this to 17.5 KiB. The resulting kernel matched an f64 CPU
reference within 7.4e-7 maximum absolute error, including
a 1,025-token wrapped window, the one-to-two partition boundary, and global attention at position 32,765.
All 48 teacher-forced layer errors matched the published measurements for both chat and tools. Free greedy
generation matched llama.cpp on 16/16 tokens for each and 24/24 after the 2,761-token prompt. The final
five-draft Responses server matched the published plain output on seeds 7 and 17 in JSON and SSE, output
limits, the image fixture and the tool-result continuation. The standalone vision reference had cosine 0.999781.

Build commands, runner scripts, candidate sources, clocks and raw outputs are archived in
`results/inkling/decode-8e9561`. The release build and all 112 engine tests ran CPU-only on Voyage with
`cargo build --release --offline -p trashcan` and `cargo test --release --offline -p trashcan`.

## Order of work

1. Reference: build, dumps for the fixture prompts (text and images), KLD baselines and the attribution runs above.
2. Loader for both layouts, tokenizer (GPT-4o regex in `tokenizer.rs`), chat rendering and parsing
   (`responses.rs`: thinking, tool calls, effort, images).
3. Kernels, each checked on one D700 against the dumps: NVFP4 matvec and MoE (gate/up with the scale before the
   activation, down) and prefill GEMMs; attention with the relative bias, sinks, clipping, sliding window and log
   scale (decode and prefill, from the split-K kernels); the convolutions with rollback state; the router and the
   activation; the loop controllers.
4. `engine/src/inkling/`: TP4 over two Macs on the Qwen TP machinery (segments, host residual, exchanges, rollback),
   generalized where it is Qwen-specific; `inkling chat`, `bench`, `validate`, then `serve`.
5. Vision encoder and image prompts.
6. Decode performance, the DSpark drafter, prefix reuse, long context (Q8 K/V), the two-Mac deploy recipe; TP8 on
   four Macs if it pays.

## Running

```sh
# the other Mac (machine 1)
trashcan inkling node ~/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_XL.gguf --listen mac1.tb1:7073
# the driver: chat, bench, or a check against a llama.cpp dump (tools/dump-activations, DUMP_FULL=l_out)
trashcan inkling chat MODEL --tp-connect mac1.tb1:7073 "Hello"
trashcan inkling bench MODEL --tp-connect mac1.tb1:7073 --pp 512 --tg 128
trashcan inkling validate MODEL REFERENCE_DIR --ids PROMPT.ids --tp-connect mac1.tb1:7073
```

`trashcan serve MODEL --tp-connect mac1.tb1:7073` serves the Responses API ([server.md](server.md)) with Inkling's
chat format rendered natively (`inkling/chat.rs`): reasoning effort from `reasoning.effort` (`none` when thinking is
off; the model still reasons briefly), tool calls and tool results by name, and prefix reuse when a request extends
the tokens already processed. `scripts/test-server.py` passes with a larger output budget for its sampled request
(32 tokens are spent on reasoning).

`TRASHCAN_INKLING_PROFILE=1` adds GPU time per kernel to `bench`. The all-reduces of decode and verification
batches carry f16 partial sums, as the Qwen groups' do; `validate` and `logits` use exact f32 sums, and
`TRASHCAN_TP_WIRE=f32` or `f16` on the driver chooses for any command. The startup line names the wire in use.

