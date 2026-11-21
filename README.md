# trashcan

An LLM inference engine for the 2013 Mac Pro (Mac Pro 6,1, "the trash can") and its two AMD FirePro D700 GPUs.
It is written in Rust on raw Vulkan (RADV) and splits each model across both GPUs, and across several Macs
connected by Thunderbolt 2.

Supported models:

| Model | Machines | Decode | With speculative decoding |
|---|---|---|---|
| Gemma 4 12B QAT Q4_0 | 1 Mac Pro (tensor parallel over both GPUs) | 28.7 tok/s | 40.0 tok/s (Gemma assistant) |
| Qwen3.5-9B | 1 Mac Pro (tensor parallel) | 46 tok/s | |
| Qwen3.8-27B | 2 Mac Pros (tensor parallel) | 28.4 tok/s | 39.5 tok/s (MTP) |
| Qwen3.8-27B | 4 Mac Pros (tensor parallel, fully cabled) | 40.7 tok/s | 52.0 tok/s (MTP) |
| Qwen3.8-Flash-Next (177B MoE) | 6 Mac Pros (pipeline) | 22.6 tok/s | 31.3 tok/s (MTP) |

All rates are single-stream decode with greedy sampling on Mac Pros with two D700s each. See the model docs for
prefill rates and test conditions.

Features: interactive chat, an OpenAI Responses API server (streaming, reasoning, tools), chunked prefill,
speculative decoding, prefix reuse across turns, and validation against reference activations.

## Requirements

- Mac Pro 6,1 with two FirePro D700s (6 GB each). Other GPUs are not supported.
- Debian 13 (x86_64) with Mesa's RADV Vulkan driver, booted with `radeon.si_support=0 amdgpu.si_support=1`.
- The patched Linux 7.3 kernel in [`kernel/`](kernel/): it adds PCI peer-to-peer DMA between the two GPUs, fixes
  VBIOS selection for the second D700, and fixes the Thunderbolt stream driver used between Macs.
  See [docs/kernel.md](docs/kernel.md).
- Rust (stable), `glslc` and the Vulkan headers to build.

To set up a fresh Debian install over SSH (kernel, packages, GPU and Thunderbolt configuration, build), use
[`nodes/macpro/provision.sh`](nodes/macpro/README.md). To check, start and stop the published one- to six-Mac
recipes, use [`deploy/trashcan-cluster`](docs/deploy.md).

## Build

```sh
sudo apt install build-essential pkg-config mesa-vulkan-drivers vulkan-tools libvulkan-dev glslc
curl -fsSL https://sh.rustup.rs | sh
cargo build --release
```

The binary is `target/release/trashcan`. Without `glslc` the build succeeds but GPU commands fail.

## Get a model

Gemma 4 12B:

```sh
hf download google/gemma-4-12B-it-qat-q4_0-gguf --local-dir ~/models/gemma-4-12b-qat-gguf
python3 scripts/fetch-assistant.py   # optional: the draft model for speculative decoding
```

Qwen models need a Q4_0 GGUF quantized with the recipe in [docs/qwen.md](docs/qwen.md#quantization).
Qwen3.8-Flash-Next needs the recipe in [docs/qwen4.md](docs/qwen4.md).

## Run

Chat (omit the prompt for an interactive session; `/reset` and `/quit` work there):

```sh
model=~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
target/release/trashcan chat "$model" "Hello there" --temperature 0
target/release/trashcan chat "$model" --draft ~/models/gemma-4-12b-assistant/model.safetensors
```

Serve the OpenAI Responses API on port 8000:

```sh
target/release/trashcan serve "$model" --port 8000
```

See [docs/server.md](docs/server.md) for client setup and supported features.

Qwen3.5-9B on one Mac, with speculative decoding from the model's MTP layer:

```sh
target/release/trashcan qwen-chat ~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf "Hello there" \
  --mtp ~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
```

Larger models run across several Macs connected by Thunderbolt cables. Start `trashcan qwen-tp-node` (tensor
parallel) or `trashcan qwen-node` (pipeline) on the other Macs, then run `qwen-chat` on the first one. See
[docs/qwen.md](docs/qwen.md) for cabling and commands, and [docs/qwen4.md](docs/qwen4.md) for Qwen3.8-Flash-Next.

`trashcan --help` lists every command.

## Test

```sh
cargo test
scripts/test-engine.sh [MODEL [ASSISTANT]]
```

`test-engine.sh` checks the tokenizer and chat template against fixtures, stress-tests GPU peer-to-peer copies,
and compares activations against [`tests/reference-vulkan/`](tests/reference-vulkan/).
[docs/implementation.md](docs/implementation.md) describes the validation and benchmark commands.

## Documentation

- [docs/hardware.md](docs/hardware.md): the Mac Pro 6,1 and D700s, the clock bug, Thunderbolt, fans
- [docs/deploy.md](docs/deploy.md): setting up nodes and running the 1-, 2-, 4- and 6-Mac recipes
- [docs/kernel.md](docs/kernel.md): building and installing the patched kernel
- [docs/implementation.md](docs/implementation.md): build, run, validation and benchmarks for Gemma 4
- [docs/engine-spec.md](docs/engine-spec.md): Gemma 4 math, constants, formats, tokenizer and GPU details
- [docs/qwen.md](docs/qwen.md): Qwen3.8-27B and Qwen3.5-9B: stages, tensor parallelism, MTP, prefix reuse
- [docs/qwen4.md](docs/qwen4.md): Qwen3.8-Flash-Next on a chain of Mac Pros
- [docs/inkling.md](docs/inkling.md): Inkling 2 Mini (planned): looped MoE layers, NVFP4, vision, two-Mac tensor parallelism
- [docs/server.md](docs/server.md): the Responses API server
- [docs/lights.md](docs/lights.md): the port lights as blinkenlights and a music visualizer (optional)

## Repository layout

- `deploy/`: cluster inventory example, recipes, pinned artifacts and `trashcan-cluster`
- `engine/`: the Rust/Vulkan engine and its GLSL compute shaders
- `kernel/`: kernel patches, config and build script
- `lights/`: `portlights`, the port lights service and visualizer, with its install scripts
- `nodes/macpro/`: setup script and system configuration for a Mac Pro node
- `scripts/`: tests, benchmarks, clock recording and fan control
- `tools/`: activation comparison, fixture generation and profiling tools
- `tests/`: tokenizer and chat fixtures, reference activations
- `bench/xgpu/`: cross-GPU sync and copy microbenchmark

## License

MIT; see [LICENSE](LICENSE). The GLSL shaders adapted from llama.cpp keep its MIT notice in
[engine/shaders/LICENSE.llama.cpp](engine/shaders/LICENSE.llama.cpp). The kernel patches in `kernel/` modify
GPL-2.0 Linux sources and are under GPL-2.0.
