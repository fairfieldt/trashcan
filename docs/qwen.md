# Qwen models: Qwen3.8-27B (and Qwen3.5-9B), toward Qwen3.8-Flash-Next

The engine runs `qwen35` GGUFs: Qwen3.8-27B and the same-architecture Qwen3.5-9B, which fits on one Mac
and is the single-machine test model. The forward pass is a pipeline of **stages**, each owning a
contiguous range of layers on one device. Stages chain across the two local GPUs and across machines;
only the hidden state crosses a boundary. Qwen3.8-Flash-Next (`qwen4exp`) has its own document, [qwen4.md](qwen4.md).

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

One command buffer per batch shape, replayed with a new position. Q4_0 projections reuse the Gemma kernels
(`gemm` above 8 columns, `multivec4` for 5-8, `matvecq4*` for decode); verification widths 2-4 use
`qwen_q4c2/3/4`. The head uses `headvec` (or the multi-column kernels for verification), `topk` and
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
| `qwen_q4c2/3/4` | 2-4 columns of a Q4_0 projection in the decode kernels' formulation: each row's nibbles are converted once and shared, and every column keeps `matvecq4`'s bias, FMA chain and reduction, so it equals a one-column decode bit for bit |

Attention uses the existing scalar kernels: split-K decode (`attention_partial*` + merge) for up to 8
columns and the unsplit `attention` for prefill. The tiled prefill kernels assume a power-of-two query
group (Qwen has 6 query heads per KV head), so long prompts are not yet fast.

## Recurrent state: rollback and checkpoints

Each DeltaNet layer keeps a 128 x 128 f32 state per value head (3 MiB per layer for the 27B) and a
three-row convolution window (120 KiB), updated in place; position 0 restarts both from zero. Attention KV
is addressed by position, so rewinding it only moves the position. The recurrent state is made rewindable
on every stage (GPU or CPU) and every machine of a chain:

- **Verification snapshots.** A `verify` forward of at most `--verify` tokens (eight at most) has
  `qwen_gdn` and `qwen_conv` also store the state after every token but the last, and ranks every column at
  the head. `rollback` to an accepted prefix copies one snapshot back: about 1 ms for the 9B on two GPUs, so
  rejected drafts are never recomputed. Snapshots take `verify - 1` state copies per DeltaNet layer
  (6.2 MiB per layer for the 27B with two drafts).
- **Checkpoint slots.** `save`/`restore` copy every DeltaNet state and convolution window (and the MTP
  layer's pending hidden row) to host memory and back: 33.5 MiB per slot for the 9B's first GPU (16
  DeltaNet layers), about 42 MiB per GPU for the 27B at 18 layers per GPU.

The chain protocol carries both (`net.rs`): `forward` has a `verify` flag and the batch's token ids,
`{"rollback": column}` keeps tokens `0..=column` of the latest verification batch on every machine, and
`{"save"|"restore": slot}` copy checkpoints. None of them waits for a reply.

`trashcan qwen-check MODEL [--mtp FILE]` checks all of this against ordinary greedy decoding after a long
prompt: the continuation after restoring a checkpoint must be bitwise identical; verification batches of
2..=verify tokens of that continuation, each followed by a rollback to a varying prefix, must rank the
same top token in every column; speculative decoding must be token-identical. With the `qwen_q4c` kernels
every column's top score also equals serial decoding's bit for bit in these checks.

## Machines

`plan` (`pipeline.rs`) balances uploaded bytes (Q4_0 repacked to 20 bytes per block, plus the 1.25-byte Q6
head and, when drafting, the MTP layer on the last stage) over every device in the chain: the 27B on four
GPUs with drafting becomes 0-18, 18-36, 36-54 and 54-64 + head + MTP, about 4.0 GiB of weights each.
`--cuts a,b,...` overrides it with the first layer of each later stage.

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

### Tensor parallelism (`--tp`, `tp.rs`)

Instead of a range of layers per GPU, every GPU can hold one rank's share of every layer (`Split` in `gpu.rs`):
its key heads with the value heads that read them, its query heads with their KV heads, a slice of the FFN and of
the vocabulary. The hidden state is replicated. Each rank runs the unchanged kernels with the per-rank head counts;
value head h reads key head h % k_heads, so taking each group of k_heads value heads' share in order keeps that
mapping locally. Each mixer and FFN ends a submission with its partial output projection in host memory; the
host adds the ranks' partials, exchanges the sum with the other machine over 1-3 lanes (`link.rs`, TCP or
USB4STREAM) and writes the total that the next submission adds to the residual: two all-reduces per layer,
2 × layers + 1 submissions per forward, all submitted ahead and released by a timeline semaphore. Each machine
merges its ranks' candidates and the machines exchange theirs. Rollback and checkpoints apply to every rank.

```sh
trashcan qwen-bench MODEL --tp --pp 512 --tg 128                       # this machine's GPUs
trashcan qwen-tp-node MODEL --mtp MTP.gguf --stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2   # the other machine
trashcan qwen-chat MODEL --mtp MTP.gguf --tp-stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 "Hello"
trashcan qwen-tp-node MODEL --listen 0.0.0.0:7072                       # or TCP, one connection per link
trashcan qwen-chat MODEL --tp-connect invite.tb0:7072,invite.tb1:7072,invite.tb2:7072 "Hello"
```

Qwen3.8-27B on incite and invite (four D700s, [results/qwen/tp-20261006](../results/qwen/tp-20261006/)): TP4
matches the CPU reference to 4.1e-4 with the same greedy tokens, and runs at 143 tok/s prefill and 16.6 tok/s
decode over USB4STREAM (140 / 15.6 over TCP), against 38 / 9.6 for the four-stage pipeline. The 9B gets 251 / 32
with TP2 on one Mac against 127 / 23. A decode token spends most of its time on the GPUs (about 102 GB/s per GPU
against 123 for a pipeline stage: narrower matrices and a cache flush per submission); the 128 cross-Mac
exchanges add 12 ms. With MTP drafting (the block split like a trunk layer) the 27B decodes at 26 tok/s on the two
Macs, 72% of drafts accepted; `qwen-check` passes on TP with and without drafting. `trashcan serve` takes the same
`--tp` options and passes `scripts/test-server.py` that way. Measurement switches:
`TRASHCAN_TP_SUBMIT=1` submits each segment when it is due instead of ahead; `TRASHCAN_TP_FREE=1` runs the
segments back to back without all-reduces (wrong results, the GPU floor).

Prefill hides its all-reduces behind GPU work ([results/qwen/tp-prefill-20261006](../results/qwen/tp-prefill-20261006/)).
A batch of at least 128 tokens runs as two micro-batches, each with its own rows of the batch buffers and its own
position parameters. Each GPU runs the second micro-batch on the second queue of the compute family, its segment i
waiting for the first micro-batch's segment i (whose recurrent state and attention cache it continues). The host
all-reduces one micro-batch while the GPUs run the other, and each queue fills the other's bubbles (`qwen_gdn`,
host-memory traffic). The prefill's candidates are bit for bit those of one batch. The 27B prefills 512 tokens in
3.6 s instead of 5.6, within a few percent of the GPU floor, where Q4_0 GEMMs at 55-63% of the D700's peak now take
most of the time. Switches: `TRASHCAN_TP_MICRO` (default 2, 1 disables), `TRASHCAN_TP_QUEUES=1` (one queue),
`TRASHCAN_TP_THREADS` (host threads for the sums; 6 across Macs, 1 on one Mac), `TRASHCAN_TP_PROFILE=1` (GPU time per
kernel in `qwen-bench`'s `pp_profile`), `TRASHCAN_TP_VRAM_IO=1` (timing only: partials and totals stay in VRAM).
`trashcan qwen-gemm-bench --shapes ROWSxK,...` times Q4_0 GEMM kernels and checks variants against `gemm` bit for bit.

The decision to use TP across Macs came from `trashcan allreduce-bench`
([results/qwen/tp4-viability-20261006](../results/qwen/tp4-viability-20261006/)): a cross-Mac all-reduce costs
about 104 µs over USB4STREAM and 138 µs over TCP. A prefill all-reduce (10 MiB per 512-token chunk) costs about
21.6 ms there; prefill now runs it beside the GPU work (above).

## Speculative decoding with the MTP layer

Qwen3.5 and Qwen3.8 ship one MTP (nextn) layer: `eh_proj` over `[RMS(embed(token)) * enorm |
RMS(hidden) * hnorm]`, a full gated-attention block with its own KV cache, a SwiGLU FFN, `RMS *
shared_head_norm`, then the shared LM head. As in llama.cpp's `graph_mtp`, position p pairs the token at p
with the target's normed final hidden row p - 1 (zeros at position 0), and a chained draft feeds the layer
its own normed output. `cpu::CpuMtp` is the reference; the GPU version runs on the last stage, next to the
head:

- every target batch also extends the MTP cache over its tokens (one more layer, 1-2 ms per verification
  batch for the 9B);
- a round drafts k tokens on the last machine (the drafted token's Q8_0 embedding row is dequantized on that
  machine's CPU), verifies the pending token and the drafts in one batch through every stage, accepts the
  matching prefix and rolls every stage back to the last accepted token. Above temperature 0 the drafts are
  sampled from the MTP's candidates and accepted by rejection sampling (`assistant::verify_prefix`), so
  the output follows the target distribution;
- `--draft-vocab N` (default 65536) makes the draft head rank only token ids below N, a prefix of the
  repacked head: 93% of generated tokens fall below 32768 and 98.5% below 65536 in our samples, and a 9B
  draft step drops from 8.4 ms (full head) to 4.0 ms.

The MTP weights come as a separate GGUF holding `blk.N.nextn.*` and the block's tensors (our trunk GGUFs
have none). The 27B's is ggml-org's `mtp-Qwen3.8-27B-BF16.gguf`; the 9B's is converted from the `mtp.*`
tensors of `Qwen/Qwen3.5-9B`, which `tools/fetch-mtp.py` downloads by HTTP range (4.2 GiB with the
embedding and head the converter keeps). Quantize like the trunk; the importance matrix has no entries for
the MTP layer, so its matrices are plain Q4_0:

```sh
tools/fetch-mtp.py Qwen/Qwen3.5-9B hf-mtp
python convert_hf_to_gguf.py --mtp --outtype bf16 --outfile mtp-Qwen3.5-9B-BF16.gguf hf-mtp
llama-quantize --pure --output-tensor-type q6_k --token-embedding-type q8_0 \
  mtp-Qwen3.5-9B-BF16.gguf mtp-Qwen3.5-9B-trashcan-Q4_0.gguf q4_0
```

```sh
trashcan qwen-chat MODEL --mtp MTP.gguf "Hello"      # two drafts per round (--draft-tokens)
trashcan qwen-check MODEL --mtp MTP.gguf             # correctness gate against greedy decoding
trashcan qwen-spec-bench MODEL --mtp MTP.gguf        # ms per decode, verification, rollback, draft, checkpoint
# in a chain, the last machine loads its own copy; the driver's --mtp turns drafting on
trashcan qwen-node MODEL --listen 0.0.0.0:7070 --mtp MTP.gguf
```

The cost model is `tok/s = (1 + a1 + a1 a2 + ...) / (T_verify(k + 1) + k T_draft)` with `a_i` the
acceptance rate of draft i given the previous ones. On the D700s the multi-column Q4_0 kernels make
verification grow quickly with width (see Measured), so two drafts win: three cost more than the third
draft's acceptance returns. `results/qwen/spec-20261006/` holds the sweeps.

## Prefix reuse

The driver keeps the token history of its live state and up to `--checkpoints` (default 8, least
recently used replaced) checkpoint slots. A new prompt continues from the live state, or restores the
latest checkpoint at or before its common prefix with the history, then prefills only the rest; at least
the final prompt token is recomputed for its candidates. Writing a position invalidates the checkpoints
beyond it. `qwen-chat` and the server checkpoint the start of the newest assistant turn (just after
`<|im_start|>assistant\n`), every 4096 prompt positions (so an edit deep in a long history keeps most of
it) and the end of each reply; `--checkpoints 0` disables reuse.

How much is reusable depends on the chat template. Qwen3.8's renders earlier assistant turns as they were
generated (`preserve_thinking` defaults to true; reasoning goes in `reasoning_content`), so a follow-up
turn reuses the whole processed history. Qwen3.5's drops the reasoning block of turns before the newest
user message, so a follow-up turn restores the turn-start checkpoint and re-prefills the previous reply.
Within a tool-calling loop both templates keep every turn after the newest user message, so a tool result
continues the cached sequence.

## Serving

`trashcan serve` detects a `qwen35` GGUF and serves it through the same Responses API (`docs/server.md`):
`--nodes`, `--cpu`, `--gpus`, `--cuts`, `--mtp`, `--draft-tokens` (default 2), `--draft-vocab` and
`--checkpoints` work as for the Qwen commands, and `--model-id` defaults to the GGUF's name.

```sh
trashcan serve ~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf --port 8000 \
  --mtp ~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
```

Reasoning is the text inside `<think>`; the prompt opens it when thinking is on. Tool calls use Qwen's
native `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n</function>\n</tool_call>`:
values of string-typed parameters stay text, others are parsed as JSON. Tool results become `tool`
messages. `reasoning.effort` selects Qwen3.8's template `reasoning_effort` (low; medium; xhigh for high and
above). `scripts/test-server.py` passes against Qwen3.5-9B on the GPUs with MTP drafting.

## Measured (incite, 2026-10-06)

Kernel 7.2.9-p2p, schedutil with turbo, sustained 850/1370 MHz on both cards, `scripts/latency-run.py`.
Reports and clocks: [results/qwen/bringup-20261006](../results/qwen/bringup-20261006/).

| run | result |
|---|---|
| Qwen3.5-9B, GPU0 layers 0-21, GPU1 21-32 + head | matches llama.cpp CPU (logits rel 2.1e-2, same 10 greedy tokens) |
| Qwen3.5-9B pp512 / tg128, two runs | 127.2 / 22.8 tok/s (ceiling 47 tok/s decode) |
| Qwen3.5-9B pp2048 / tg32 | 120.8 / 21.9 tok/s |
| Qwen3.8-27B, CPU layers 0-30 -> GPU node 30-48, 48-64 + head over TCP | matches llama.cpp CPU (logits rel 3.3e-2, same 10 greedy tokens) |

The 27B run is a correctness check: incite alone lacks the VRAM, so its first 30 layers ran on the CPU
reference path. With a second Mac, the same command uses that Mac's GPUs instead
(`--nodes MAC2:7070`, no `--cpu`). Per-GPU allocation was 4.2 GiB (18 layers) and 5.1 GiB (16 layers +
head), so two Macs fit the 27B at 16 layers per GPU with room for context.

Decode at 48% of the bandwidth ceiling is the starting point, not a tuned result. The obvious costs:
the host relay between stages, one fence wait per stage, unfused norms and the scalar attention kernels.

### Rollback, MTP and prefix reuse (incite, 2026-10-06)

Same machine and clocks (850/1370 MHz sustained). Reports, outputs and clock logs:
[results/qwen/spec-20261006](../results/qwen/spec-20261006/).

| check | result |
|---|---|
| `qwen-check` 9B GPUs, 600-token prompt, 256 steps, MTP k=2 | pass: restore bitwise exact, 170 verification batches with rollbacks match serial decode with bitwise-equal top scores (2.3e-4 apart before `qwen_q4c`), speculative output identical over 256 tokens |
| `qwen-check` 27B hybrid (CPU 0-30, GPU 30-48, 48-64 + head + MTP), 48 tokens, 24 steps, k=2 | pass: verification score error 3.5e-4, speculative output identical, 70% acceptance |
| same checks on the CPU path, one process and a two-process TCP chain | pass, verification bitwise exact |
| GPU MTP layer vs the CPU reference (`qwen-validate --mtp`, 9B, three chained drafts after the reference prompt) | same top token at every step, top-10 overlap 10/10, largest score difference 0.005 (rel 4.7e-4) over all 64 candidates |
| our MTP vs llama.cpp `--spec-type draft-mtp` (9B, k=3, full draft head, same 48-token prompt, 200 tokens) | 126/219 (57.5%) vs 126/214 (58.9%) drafts accepted, same opening text |
| `scripts/test-server.py`, 9B GPUs, MTP k=2 | pass (JSON, SSE, seeded sampling, tools, tool continuations) |

Costs from `qwen-spec-bench` (9B, 600-token prompt, ms per call):

| decode | verify 2 / 3 / 4 / 5 tokens | rollback | draft step: full head / 64K / 32K | checkpoint save / restore |
|---|---|---|---|---|
| 45.4 | 52.5 / 60.5 / 75.5 / 125.5 | 0.9 | 8.4 / 4.0 / 3.25 | 23.8 / 5.8 |

Five columns run the four-column kernel twice, hence the jump. The MTP cache update inside each
verification batch costs 1.5-2 ms (verification without the MTP layer: 50.9 / 59.2 / 73.6) and slows
prefill by 4% (pp512 126.46 vs 121.59 tok/s).

Greedy decode rates (tok/s), three prompts from the Gemma MTP round, 128 tokens, thinking off, before the
`qwen_q4c` verification kernels. Every speculative output is byte-identical to plain decoding:

| draft head | k | transformer | photosynthesis | CSV function |
|---|---|---|---|---|
| (plain decode) | 0 | 23.06 | 23.19 | 23.21 |
| full | 1 / 2 / 3 | 31.08 / 31.49 / 27.91 | 27.16 / 27.74 / 23.03 | 32.57 / 36.44 / 33.09 |
| first 65536 ids | 1 / 2 / 3 | 33.31 / **34.19** / 30.46 | 28.94 / **30.31** / 26.03 | 34.58 / **41.26** / 38.01 |
| first 32768 ids | 1 / 2 / 3 | 31.29 / 31.86 / 27.86 | 29.48 / 30.95 / 27.12 | 34.49 / 40.36 / 38.33 |

With `qwen_q4c` verification, two drafts from a 64K draft head (the defaults) give 30.7-41.7 tok/s:
pooled 35.0 against 23.2 tok/s plain in two runs after a warm-up, in both orders (+51%). Per round: about
59 ms of verification, 7.9 ms of drafting and 0.4 ms of rollback for 2.37 tokens (77% of first drafts and
60% of both accepted).

At temperature 0.7 (top-k 20, top-p 0.8), pooled tok/s is 23.21 plain, 33.47 with one draft (84% accepted),
33.16 with two and 30.86 with three. The 27B cannot be timed on one Mac (the hybrid chain spends seconds
per token in its CPU stage); with two Macs the same cost model applies, with the 27B's verification
crossing four GPUs.

Prefix reuse, 9B, MTP k=2: a 2.7K-token document turn followed by three short questions. With reuse each
follow-up prefills 84, 56 and 92 tokens (2675, 2755 and 2807 reused; about 1.4 s instead of 25 s), and the
text is identical to the run without reuse. A 5866-token prompt edited near its end restores the 4096
interval checkpoint and prefills 1770 tokens. Qwen3.8-27B with thinking on (CPU path, two short turns)
reuses the whole first turn, reasoning included: 100 of 118 tokens.

## Building

`scripts/cross-build.sh [DEST]` builds in the `kbuild` lima VM (aarch64 Debian 13 with an x86_64 cross
toolchain and the same glibc as incite) in about 10 s and copies the binary to incite, keeping compile
load off the benchmark machine. GPU runs on incite go through gpuq leases (`ssh incite 'gpuq run ID -- CMD'`, see AGENTS.md),
which holds `~/trashcan-exp/.lock` and records clocks.

## Next: Qwen3.8-Flash-Next (`qwen4exp`)

This was the plan; the implementation (`engine/src/qwen4/`), measurements and recipes are in
[qwen4.md](qwen4.md).

Shared with what exists: tokenizer, Gated DeltaNet (with a sigmoid output gate instead of SiLU and 48
value heads), the gated attention projections, partial RoPE, the Q6 head and the machine chain. New:

- **Gated residual (4 streams, rank 320)** replaces every norm. Before each sublayer, a mixer RMS-norms
  each stream, computes an element-wise read gate through a 10240 -> 320 -> 10240 bottleneck and a
  per-stream write gate; after it, each stream adds `gate_s * output`. The stage boundary payload becomes
  the 4 x 2560 residual (40 KiB per token). There are 97 mixers per token, about 10% of active weights,
  each two dependent GEMVs, so fusing is worth it.
- **N-gram embedding at `blk.1`**: two hashes of the previous one and two tokens select 16 rows of 160
  from a 320M-row table (51B parameters; about 29 GB at Q4_0 or 54 GB at Q8_0, 32-element blocks
  because 160 is not a multiple of 256). It belongs on the driver's CPU next to the token embedding:
  hash, gather, upload 10 KiB. A key/value gate and a dilated width-4 convolution (9 x 10240 f32 of
  history) then run on the GPU.
- **MoE**: softmax router over 512 experts, top 10 renormalized, plus a sigmoid-gated shared expert,
  all 640 wide. Routed experts total about 121B parameters (68 GB at Q4_0), touched at about 1.33 GB per
  token.
- **Qwen Sparse Attention**: 24 Q / 2 KV heads with a 4-head, 128-wide indexer over 4-token pooled keys.
  Up to 2051 visible tokens every block is selected, so attention is exactly dense there (still caching
  raw and pooled indexer keys); beyond that, top 512 blocks plus the tail.
- **MTP** layer: a full QSA layer with its own experts; it needs rollback snapshots of every recurrent
  state, as speculative decoding does here generally.

Placement keeps every layer's experts on the stage that owns the layer, so routing never crosses a
machine. 48 layers divide into 8 per Mac on six Macs and 12 on four. VRAM is the binding constraint:
about 72 GB on six Macs against roughly 68 GB of Q4_0 experts plus 2.5 GB of resident weights, KV and
state. Six Macs therefore need experts near 4 bits per weight with little slack, and four Macs need about
2.5-3 bits, or experts left in host RAM with a GPU cache of hot experts. The M3 Ultra (512 GB) holds
the whole model in llama.cpp, so the first step is the same as for Qwen3.8-27B: a CPU reference checked
against llama.cpp, then GPU stages checked layer by layer against it.

Measured since: the dense matrices need Q8_0 (all of them at Q4_0 cost KLD 0.20 against 0.04 for Q4_0 experts),
IQ4_XS/IQ4_NL experts fit six Macs once the head is spread over the GPUs, and 2-bit experts cost KLD 0.21, so four
Macs are better served by experts in host RAM with a VRAM cache.
