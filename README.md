# trashcan

A purpose-built inference engine for Gemma 4 12B QAT Q4_0 on the 2013 Mac Pro (incite: 2x FirePro D700),
written in Rust on raw Vulkan, with tensor parallelism across the two GPUs. llama.cpp is kept as the reference.

- `docs/hardware.md`: the machine, the D700 clock bug, baseline numbers
- `docs/upstream-survey.md`: upstream TP/meta-backend state, Vulkan PRs and forks
- `docs/engine-plan.md`: the Rust engine design (TP layout, schedule, prefill, MTP, milestones)
- `docs/engine-spec.md`: implementation reference: exact math, constants, formats, tokenizer, validation assets, GPU facts, conventions
- `docs/findings.md`: measurements and bugs, newest first
- `engine/`: the Rust/Vulkan engine: decode, chunked prefill, tokenizer, chat and optional MTP
- `docs/implementation.md`: build/run commands, validation recipes and measured implementation status
- `bench/xgpu/`: cross-GPU sync and copy microbenchmark
- `kernel/`: custom kernel build (D700 fixes + PCI P2P)
- `scripts/`: incite helpers (`sync.sh` pushes the workspace and pulls `results/`)
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
[main-branch performance report](results/engine/main-tuning-20261003/README.md) records the latest
validated improvements, rejected experiments, and corrected hardware bounds.

To serve the model through the OpenAI Responses API:

```sh
target/release/trashcan serve ~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf --port 8000
```

The server supports text, SSE streaming, reasoning, function/custom tools and full-history tool
continuations. See [server setup and client examples](docs/server.md).
