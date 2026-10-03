# Rust/Vulkan implementation

The engine runs the target GGUF directly on incite's two D700s. It implements the full text forward pass,
tensor-parallel decode, chunked prefill, independent tokenizer, embedded Jinja chat rendering, sampling,
and the matched four-layer BF16 assistant. It has no runtime llama.cpp dependency.

M1, M2 and M4 pass their acceptance checks. Decode bandwidth/speed (M3) and the required typical-chat
speculative gain (M5) remain performance work; their original targets are retained in `engine-plan.md`.

## Recorded acceptance

The 2026-10-03 tuning run raises tg128 from 21.33 to 22.14 tok/s and reduces one-column top-64
selection from about 2.20 ms to 0.51 ms per GPU. Prefill remains about 161 tok/s. Current measurements
and validation are retained in `results/engine/performance-20261003/`; the acceptance table below
preserves the original implementation measurements. M3 and M5 remain open.
The final small-batch selector also improves eight-column Q4 verification to about 0.546 ms versus
the original 0.945 ms in the FFN microbenchmark. Its seven-draft chat comparison and focused validation
are recorded separately from the main benchmark table; the default remains three drafts.

Incite, two D700s, kernel `6.12.111-p2p`, Mesa 25.0.7. Timed commands used the user-authorized temporary
CPU latency request of 0. Raw outputs, clocks, source/binary hashes and JSON timings are under
`results/engine/delivered-benchmark/`; `results/engine/summary.json` collects the measurements.

| measurement | result | acceptance |
|---|---|---|
| Q4 FFN mat-vec, 100 iterations, each GPU | 145.1 / 144.4 GB/s of original weights; 161.3 / 160.5 GB/s repacked | M3 requires ≥180 GB/s |
| pp512, batch 512 | **161.1 tok/s** | M4 requires ≥150: pass |
| tg128 after pp512 | **21.2 tok/s** | M3 requires 40–50: open |
| transformer chat, 128 tokens | 23.42 ordinary / 28.61 MTP tok/s; **1.22×** | identical output |
| photosynthesis chat, 128 tokens | 23.26 ordinary / 23.51 MTP tok/s; **1.01×** | identical output |
| Python CSV chat, 128 tokens | 23.50 ordinary / 34.09 MTP tok/s; **1.45×** | identical output |
| combined three-chat decode time | **1.20×** MTP gain | M5 requires ≥1.5×: open |

The paired chat runs use temperature 0 and three draft tokens. The combined gain divides total ordinary
decode time by total speculative decode time. Rates are rounded in the CLI logs. Recorded clocks include
850/1370 MHz during load, with maximum temperatures 84/82 C. Decode's median semaphore gaps are
27.9/27.3 us on the recorded final schedule. The earlier direct PM QoS comparison is retained in
`results/engine/latency-comparison.json` (123/133 us without the request versus 41/42 us with it).

The five reference paths pass all 132 tensors, logits and ten greedy tokens; worst relative activation
error is 0.000982, below the 0.04 tolerance target. The streamed-P2P long check passes at 1700 tokens /
batch 512: top-64 overlap 63, relative logit error 0.001761, identical 64-token serial/chunked greedy
continuation, all verification columns, and speculative acceptance/rollback. Interactive history and
`/reset` pass. The comparator rejects a nonfinite activation, and stochastic MTP is seeded/reproducible.
CPU tests and local Clippy pass; incite's toolchain does not have the Clippy component installed.

The last selector adjustment uses the existing four-row kernel for two-token verification as well as
three/four tokens: its two-token FFN microbenchmark is 41–49% faster than the one-row variant. The paired
three-chat table above predates that adjustment; its dedicated reference/continuation results are stored
separately. Ordinary pp512/tg128 use unchanged kernels. The original table's hashes are preserved.
The two-token reference passes all 132 tensors, logits and ten greedy tokens. A one-draft photosynthesis
run also produces the same 128-token continuation, at 22.89 tok/s; reducing draft length did not improve
that workload, so the default remains three proposals.

## Build and run

From this workspace, `scripts/sync.sh push` deploys source, tools, fixtures and docs to `incite:~/trashcan`.
On incite:

```sh
cd ~/trashcan
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release
model=$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
assistant=$HOME/models/gemma-4-12b-assistant/model.safetensors
target/release/trashcan chat "$model" 'Hello there' --temperature 0
target/release/trashcan chat "$model" --batch 512 --draft "$assistant"
```

Interactive chat accepts `/reset` and `/quit`. It renders and replays the conversation each turn.
Thinking is off by default; enable `--thinking`. Thought-channel text is hidden from the displayed answer.
Sampling defaults are temperature 1, top-k 64, top-p 0.95, seed 1. Temperature 0 selects greedy output
and only ranks one candidate on the GPU, including the assistant.
Generation stops at EOS/end-of-turn/end-of-function. A context overflow reports an error.

`trashcan serve "$model"` runs an optional OpenAI Responses HTTP server with SSE, reasoning and
tools. See [server setup](server.md) for client examples, defaults and supported features.

`--batch` is the maximum prefill/verification capacity (1–512). Verification supports at most eight
columns. `--draft-tokens` is 1–7, default 3, and requires `--batch` larger than that value. The main model's
sliding ring has 1024 + batch slots, so batched writes and rejected drafts do not erase needed history.

## Validate without llama.cpp

```sh
scripts/test-engine.sh "$model" "$assistant"
target/release/trashcan check-batches "$model" --tokens 1700 --batch 512 --draft "$assistant"
```

Set `TRASHCAN_LONG_CHECK=1` to include the long-cache check in the script.

The first script checks CPU unit tests, all 30 tokenizer fixtures, four reference chat templates, 10,000 fresh P2P payloads in both
directions, all 132 required activation tensors, softcapped logits and ten greedy tokens. It exercises
single-token, small-batch, GEMM and CPU-relay paths, and the assistant when present. The comparator exits
nonzero for missing tensors, nonfinite values, tolerance failures or continuation mismatches.

`check-batches` crosses the sliding-window wrap boundary, compares sequential and batched top-64
logits, checks all verification columns and a 64-token continuation from the same chunked-prefill cache,
and optionally exercises speculative acceptance/rollback. It separately reports the common greedy prefix
from serial versus chunked prefill; exact long-prompt greedy equality is not assumed. Logit comparisons
use a relative norm because accumulation order and f16 cache rounding differ between matrix kernels. It reports the absolute maximum as well.

For a focused check:

```sh
target/release/trashcan validate "$model" results/engine/check --batch 15
python3 tools/compare_activations.py results/engine/check results/reference-vulkan --check
```

## Measure

```sh
target/release/trashcan matvec-bench "$model" --iterations 100
target/release/trashcan matvec-bench "$model" --tensor blk.0.ffn_down.weight --iterations 1000
target/release/trashcan topk-bench --batch 8
target/release/trashcan bench "$model" --pp 512 --tg 128 --batch 512 --profile results/engine/bench.tsv
target/release/trashcan bench "$model" --pp 512 --tg 128 --batch 512 --draft "$assistant"
```

Bench emits JSON timings and assistant acceptance counts. The synthetic prefill repeats token 9259,
and decode intentionally continues through stop tokens. This measures kernels rather than typical-chat
acceptance. Profile TSVs contain GPU durations and semaphore gaps for all 97 segments; the prefill
profile uses a `.prefill.tsv` suffix. The JSON also separates CPU submission, fence wait and result reads.
Use `scripts/clockmon.sh` alongside sustained runs. The user authorized a temporary CPU latency request
of 0 for benchmarks. To measure it, prefix the engine command with
`sudo -n python3 scripts/latency-run.py`; the wrapper holds the request, runs the engine as the invoking
user, and closes the request when it exits. Display-manager settings remain as configured.

`scripts/bench-engine.sh` runs the microbenchmark, pp512/tg128, and three fixed 128-token greedy chats
(technical explanation, educational explanation and Python CSV code) with and without MTP, checks that
each chat output is identical, and records clocks, hardware and source hashes.
It uses the authorized temporary latency request by default; set `TRASHCAN_CPU_LATENCY=0` to measure
without it. The third argument selects the result directory.

`matvec-bench` warms each GPU for 500 ms before measuring, so an idle D700 has time to ramp its clocks.
`--warmup-ms` changes that duration. `--tensor` selects a Q4 projection with the engine's row/column
sharding; the default is layer 0's FFN gate. Every column uses a different nonuniform input, and three
output rows are checked against a CPU dot product. `topk-bench` checks GPU selection against a CPU sort
at ranks 1, 8 and 64, including tied scores, an incomplete final chunk, softcap and suppressed token IDs.

## Weight formats and GPU schedule

Q4_0 is repacked on upload into all f32 block scales followed by all 16-byte packed blocks. Quant values
retain GGUF nibble order. The cost is 20 bytes per 32 weights versus GGUF's 18. Decode, verification and
prefill share this representation. The output head expands Q6_K into signed 8-bit quants plus f32 signed
scales per 16 weights (320 versus 210 bytes per 256); embedding lookup remains original Q6_K on CPU.

Matrices, activations and peer receive buffers use cached fixed descriptor sets. `Buffer.address` is a
CPU-side handle plus byte offset used to select descriptors, not a Vulkan device address. Buffer device
addresses and descriptor indexing/update-after-bind remain disabled: Mesa 25.0.7 RADV enables a global
BO list for these features, bringing unrelated imported peer buffers into every submission.

Each allreduce receive slot is a separate dedicated DMA-BUF allocation. Disjoint ranges of one buffer
share implicit driver fences and serialize work. Producer copies its partial to the peer's slot, signals
an exported OPAQUE_FD timeline semaphore, and the next consumer segment waits on the imported semaphore.
For batches up to 128, all 97 P2P submissions are batched into one API call per GPU. The 96 external slots are capped at 128
tokens (180 MiB); full batch-512 slots would require 720 MiB, beyond the 256 MiB peer-visible BAR. For
batches above 128, two additional full-width receive buffers alternate between segments. The CPU submits
GPU0 and GPU1 for each segment before the next segment, so implicit fences on reused buffers never point
at future peer work. Timeline semaphores orchestrate the GPUs; the CPU waits only at the end of the forward
pass. This uses 195 MiB of P2P buffers at batch 512. `--cpu-reduce` instead uses persistent mapped host
staging: after each pair of fences, the CPU copies each partial to the peer's staging, then the next
pre-recorded segment uploads and sums it.

At context 8192 / batch 512 the target allocates 4552 MiB per GPU and the assistant adds 415 MiB.
At context 32768 the calculated combined total is about 5351 MiB. Imports alias the peer allocations
and are not counted twice. Host staging is separate; 32K execution has not been validated.

Attention uses scale 1.0 and f16 KV. Decode/small verification batches partition keys into 32 chunks and
combine online-softmax maxima, denominators and weighted sums. Prefill uses a wave per query/head. Global
RoPE rotates pairs 0–63 (dimensions 0–63 and 256–319), using the checkpoint's frequency factors.
Independent Q/K/V and gate/up operations share a trailing barrier. The target fuses post-normalization,
the residual update / layer scaling and the next RMS normalization while preserving their reduction order.

## Assistant asset and verification

`scripts/fetch-assistant.py` downloads and verifies the pinned public Apache-2.0 asset into
`~/models/gemma-4-12b-assistant`, without temporary directories:

- Repository: `google/gemma-4-12B-it-qat-q4_0-unquantized-assistant`
- Revision: `18934064dd4c5c6cc3621f6381e7d377fc8cb7bd`
- `model.safetensors`: 845719296 bytes; SHA-256 `67f1420cf24aa5065089aaed175223f7c245ccfda16111b6c56765afd7280db6`
- `config.json`: 2386 bytes; SHA-256 `7638c1d42f9fa73fe44b1a10604766b928b8985263f15775cd1a286a5a12799c`

The assistant reads target sliding layer 46 and global layer 47. Greedy verification accepts the matching
prefix. Stochastic verification samples proposals from the assistant's temperature/top-k/top-p distribution
Q, accepts a proposal d with probability min(1, P(d)/Q(d)), and samples a rejected token from normalized
max(P-Q, 0). If every proposal is accepted, it samples one bonus target token. P uses the same sampling
settings and target logits at that position. This preserves the target sampling law; a fixed seed is
reproducible within each mode, but speculative and ordinary sampling consume different random draws.
Only top-k candidates need to cross to the CPU because both distributions have truncated support.
Logical KV position rewinds to the committed prefix. Every draft uses the same target query position and
committed cache; the assistant has no private KV cache.

The config mentions centroids, but the pinned safetensors header has no centroid weights. The implementation
uses the full BF16 head, so the planned centroid optimization and speculative speed target need measurement.

### Assistant tensor and math reference

Shapes below are the original safetensors row-major shapes; `i` is 0–3 and `d` is 256 for layers 0–2,
512 for layer 3. Norm parameters are used directly, with no Gemma 3-style +1 shift; llama.cpp's
`conversion/gemma.py:Gemma4Model.norm_shift` returns 0.

| tensor | shape |
|---|---|
| `pre_projection.weight` | [1024, 7680] |
| `post_projection.weight` | [3840, 1024] |
| `model.embed_tokens.weight` (tied output head) | [262144, 1024] |
| `model.norm.weight` | [1024] |
| `model.layers.i.self_attn.q_proj.weight` | [16d, 1024] |
| `model.layers.i.self_attn.o_proj.weight` | [1024, 16d] |
| `model.layers.i.self_attn.q_norm.weight` | [d] |
| `model.layers.i.{input_layernorm,post_attention_layernorm,pre_feedforward_layernorm,post_feedforward_layernorm}.weight` | [1024] |
| `model.layers.i.mlp.{gate_proj,up_proj}.weight` | [8192, 1024] |
| `model.layers.i.mlp.down_proj.weight` | [1024, 8192] |
| `model.layers.i.layer_scalar` | [1] |

Let `RMS_w(x) = w ⊙ x / sqrt(mean(x²) + 1e-6)` and `GELU` use the PyTorch tanh approximation,
as in the target spec. For proposal token `t` and previous hidden vector `h`, the input is
`x = pre_projection · concat(sqrt(3840) · target_embedding[t], h)`.
Initially `h` is the target's final normalized hidden vector; later proposals use the previous
assistant post-projection. Each assistant layer computes:

```text
q = RoPE_i(RMS_q_norm(q_proj · RMS_input_layernorm(x)))
a = o_proj · attention(q, target_KV[layer 46 or 47], scale=1)
y = x + RMS_post_attention_layernorm(a)
f = down_proj · (GELU(gate_proj · RMS_pre_feedforward_layernorm(y))
                 ⊙ (up_proj · RMS_pre_feedforward_layernorm(y)))
x = layer_scalar · (y + RMS_post_feedforward_layernorm(f))
```

Finally `z = RMS_model.norm(x)`, logits are `model.embed_tokens.weight · z` with no softcap,
and the next hidden vector is `post_projection · z`. Query head count is 16; shared target KV head
counts are 8 / 1, with sliding window 1024 and the target's partial global RoPE. The vocabulary head
and query/gate/up rows are split in half across GPUs; output/down columns are split and allreduced.
Pre/post projections, norm parameters and the residual stream are replicated. The graph follows
`llama.cpp/src/models/gemma4-assistant.cpp`; fixed query position follows `common/speculative.cpp`.
