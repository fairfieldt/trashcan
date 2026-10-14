# trashcan

A purpose-built inference engine for Gemma 4 12B QAT Q4_0 on the 2013 Mac Pro (incite: 2x FirePro D700),
written in Rust on raw Vulkan, with tensor parallelism across the two GPUs. llama.cpp is kept as the reference.

- `docs/hardware.md`: the machine, the D700 clock bug, baseline numbers
- `docs/upstream-survey.md`: upstream TP/meta-backend state, Vulkan PRs and forks
- `docs/qwen.md`: Qwen3.8-27B / Qwen3.5-9B support, stages across GPUs and machines, MTP speculative
  decoding, prefix reuse, Flash-Next plan
- `docs/tp8-optimization.md`: Qwen3.8-27B TP8 (four Macs) cost model, bottlenecks and the optimization plan
- `docs/qwen4.md`: Qwen3.8-Flash-Next (`qwen4exp`): CPU reference, n-gram table placement, quantization and
  VRAM for six and four Macs, GPU stages, projected chain speed
- `docs/engine-plan.md`: the Rust engine design (TP layout, schedule, prefill, MTP, milestones)
- `docs/engine-spec.md`: implementation reference: exact math, constants, formats, tokenizer, validation assets, GPU facts, conventions
- `docs/findings.md`: measurements and bugs, newest first
- `engine/`: the Rust/Vulkan engine: decode, chunked prefill, tokenizer, chat and optional MTP
- `docs/implementation.md`: build/run commands, validation recipes and measured implementation status
- `bench/xgpu/`: cross-GPU sync and copy microbenchmark
- `kernel/`: incite's kernel, 7.2.9-p2p (D700 fixes, PCI P2P, USB4STREAM); see `docs/kernel.md`
- `nodes/macpro/`: `provision.sh` sets up a Mac Pro 6,1 with fresh Debian 13 as a node like incite
- `scripts/`: node helpers (`sync.sh` pushes the workspace and pulls `results/`; `HOST=` picks the node)
- `tools/`: llama.cpp activation dumper and `compare_activations.py`
- `tests/fixtures/`: tokenizer fixture from llama.cpp
- `reference/llama.cpp-patches/`: our local llama.cpp patches from the investigation
- `results/`: benchmark outputs, model inventory, reference activations (`reference-vulkan/` is the correctness target)
- `llama.cpp/`: upstream clone used as reference (own git repo, not tracked here; PR branches fetched as local refs)

Build on incite with `export PATH=$HOME/.cargo/bin:$PATH; cargo build --release` (requires `glslc`).
Run `scripts/test-engine.sh` for the reference checks and P2P stress test. A basic chat command is:

```sh
target/release/trashcan chat ~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf "Hello there" --temperature 0
```

Add `--draft ~/models/gemma-4-12b-assistant/model.safetensors` for the matched assistant. The functional paths
are implemented. M1, M2 and M4 pass; decode and MTP performance targets in M3/M5 remain open.
Work directly on `main`, as recorded in [AGENTS.md](AGENTS.md). The
[main-branch performance report](results/engine/main-tuning-20261003/README.md) records the preceding
validated improvements, rejected experiments, and corrected hardware bounds.
The [norm and attention-merge follow-up](results/engine/validated-followup-20261003/README.md)
raises repeated tg256 from 23.32 to 28.72 tok/s, with prefill unchanged, and records the recovered
unfinished investigation lenses.
The [MTP verification follow-up](results/engine/mtp-exploration-20261003/README.md)
raises repeated pooled MTP from 31.73 to 39.96 tok/s, preserving outputs and acceptance.
The [80K context measurement](results/engine/context-80000-20261003/README.md)
records 28.75 tok/s prefill and 12.35 tok/s decode, with long prefill still a major target.
The [bounded tiled prefill round](results/engine/prefill-tiled-20261003/README.md)
raises repeated 4K prefill by43.6% and reaches73,216 tokens3.55× faster than the
saved baseline. The user stopped that 80K run before completion and decode.

Qwen3.5-architecture models (Qwen3.8-27B, Qwen3.5-9B) run as a pipeline of layer stages across GPUs
and machines; see [docs/qwen.md](docs/qwen.md):

```sh
target/release/trashcan qwen-chat ~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf "Hello there"
```

Add `--mtp ~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf` for speculative decoding with the model's
own MTP layer; follow-up turns reuse the cached prefix. `--tp` splits every layer across the GPUs instead, and
`--tp-connect` across two Macs (an address per cable; `trashcan qwen-tp-node` on the other one); Qwen3.8-27B runs that way
on incite and invite at 150 tok/s prefill and 28.4 tok/s decode, 39.5 with MTP
([results/qwen/tp-merged-20261006](results/qwen/tp-merged-20261006/README.md), docs/qwen.md).

To serve the model through the OpenAI Responses API:

```sh
target/release/trashcan serve ~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf --port 8000
```

The server supports text, SSE streaming, reasoning, function/custom tools and full-history tool
continuations. It serves qwen35 GGUFs the same way. See [server setup and client examples](docs/server.md).
