# Inkling 2 Mini (`inkling2`) on two or more Mac Pros

Inkling 2 Mini (Thinking Machines, released 2026-11-20, Apache 2.0) is a 26.2B-parameter multimodal MoE: 24 layers
of weights, of which the middle 12 run three times per token (48 layer applications), 8 of 128 experts per layer, a
ViT image encoder, and a 3-layer block drafter for speculative decoding. This document is the plan and the
reference for running it on Mac Pros: the layer math, the weights and the chosen quant, the machine layout, the
vision path, the drafter and the order of work.

Status: text and image prompts run tensor-parallel on two Mac Pros (TP4, `engine/src/inkling/`, shaders
`engine/shaders/inkling_*.comp` and `vit_*.comp`) from Unsloth's UD-Q4_K_XL and BF16 vision projector as published,
through `inkling chat`, `inkling bench` and the Responses server (`serve`; deploy recipe `2-inkling`). Checked
against llama.cpp `29b52d398` (CPU, same GGUF): teacher-forced per-layer residual errors of 0.2-3% over all 48
applications on a 25- and a 109-token prompt, the same top token, and the reference's greedy tokens (16 of 16 on both
prompts, 24 of 24 after a 2,761-token document); the vision projector's output agrees with mtmd's to 2.1%. Decode 33
tok/s and prefill about 120 tok/s on two Macs over one cable (being optimized). The DSpark drafter is not implemented.
Every number below was read from the GGUFs of `unsloth/Inkling-2-Mini-GGUF`, from the original safetensors header of
`thinkingmachines/Inkling-2-Mini`, or from llama.cpp at `29b52d398`.

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
`qwen4_q8_*` kernels), no requantization, and has the best quality of the files. Since Q4_K_M (Q5_K gate/up in
layers 20-23) loses almost nothing against it, most of the 0.0107 likely comes from the BF16 dense matrices at Q8_0.
To check: our own KLD runs, attributing the divergence to the dense matrices, the head and layers 20-23. A lossless
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

The release ships a DSpark block drafter (`dspark-Inkling-2-Mini-Q8_0.gguf`, 0.40 GB; BF16 0.65 GB): 3 layers of
2304 (24 Q / 8 KV heads of 128, RoPE base 1e6, sliding window 1024 on layers 0 and 2, dense MLP 9216), drafting a
block of up to 7 tokens in one forward from the target's hidden states at applications 5, 23, 41 and 48, with a
rank-256 Markov RNN head chaining the block positions. It borrows the target's embedding and head. In TP the host
already holds those hidden states, so the taps cost nothing. Verifying a block reads each weight once for all its
tokens, which matters more here than usual because the loop triples the bytes per token. llama.cpp's support for
the Inkling drafter is an open pull request (#32218), so this comes after the plain model.

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
`qwen4 layer-check` does), then end to end (logits and greedy tokens). The CUDA backend computes some products in
reduced precision, so a few prompts are also dumped with the same build on the CPU backend to bound how close the
engine should get. Vision: the projector's output for fixed images, then the image prompt's logits.

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
trashcan inkling node ~/models/inkling-2-mini/Inkling-2-Mini-UD-Q4_K_XL.gguf --listen 0.0.0.0:7073
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

`TRASHCAN_INKLING_PROFILE=1` adds GPU time per kernel to `bench`. The all-reduces carry exact f32 sums;
`TRASHCAN_TP_WIRE=f16` on the driver selects the compact payload of the Qwen groups.

