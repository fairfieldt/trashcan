# trashcan

An LLM inference engine for the 2013 Mac Pro (Mac Pro 6,1, "the trash can") and its two AMD FirePro D700 GPUs.
It is written in Rust on raw Vulkan (RADV). It splits each model across both GPUs of a Mac, and larger models
across several Macs connected by Thunderbolt 2.

| Model | Macs | Decode | With speculative decoding |
|---|---|---|---|
| Gemma 4 12B QAT Q4_0 | 1 | 28.7 tok/s | 40.0 tok/s (Gemma assistant) |
| Qwen3.5-9B | 1 | 46 tok/s | |
| Qwen3.8-27B | 2 | 28.4 tok/s | 39.5 tok/s (MTP) |
| Qwen3.8-27B | 4, fully cabled | 40.7 tok/s | 52.0 tok/s (MTP) |
| Qwen3.8-Flash-Next (177B MoE) | 6, pipeline | 22.6 tok/s | 31.3 tok/s (MTP) |
| Inkling 2 Mini (26B looped MoE, vision) | 2 | 41.8 tok/s | 53.9 tok/s (DSpark) |
| Inkling 2 Mini | 4, fully cabled | 45.1 tok/s | 61.2 tok/s (DSpark) |

Every model runs tensor parallel across all of its GPUs except Flash-Next, which runs as a pipeline along a chain
of Macs. All rates are single-stream decode with greedy sampling on Mac Pros with two D700s each. The model docs
give prefill rates, context lengths and test conditions.

Features: interactive chat, an OpenAI Responses API server (streaming, reasoning, tools, image input for Inkling),
chunked prefill, speculative decoding, prefix reuse across turns, and validation against reference activations.

## Requirements

- Mac Pro 6,1 with two FirePro D700s (6 GB each). Other GPUs are not supported.
- Debian 13 (x86_64) with Mesa's RADV Vulkan driver, booted with `radeon.si_support=0 amdgpu.si_support=1`.
- The patched Linux 7.3 kernel in [`kernel/`](kernel/). It adds PCI peer-to-peer DMA between the two GPUs, fixes
  VBIOS selection for the second D700, and fixes the Thunderbolt stream driver used between Macs
  ([docs/kernel.md](docs/kernel.md)).
- Rust (stable), `glslc` and the Vulkan headers to build.

[`nodes/macpro/provision.sh`](nodes/macpro/README.md) sets up a fresh Debian install over SSH: kernel, packages,
GPU and Thunderbolt configuration, and the build.

## Build

```sh
sudo apt install build-essential pkg-config mesa-vulkan-drivers vulkan-tools libvulkan-dev glslc
curl -fsSL https://sh.rustup.rs | sh
cargo build --release
```

The binary is `target/release/trashcan` (called `trashcan` below); `trashcan --help` lists every command.
Without `glslc` the build succeeds but GPU commands fail.

## Get a model

Gemma 4 12B:

```sh
hf download google/gemma-4-12B-it-qat-q4_0-gguf --local-dir ~/models/gemma-4-12b-qat-gguf
python3 scripts/fetch-assistant.py   # optional: the draft model for speculative decoding
```

Qwen models need a Q4_0 GGUF quantized with the recipe in [docs/qwen.md](docs/qwen.md#quantization);
Qwen3.8-Flash-Next needs the recipe in [docs/qwen4.md](docs/qwen4.md).

Inkling 2 Mini runs Unsloth's GGUF as published, with its vision projector for images and its DSpark drafter for
speculative decoding:

```sh
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf mmproj-BF16.gguf \
  dspark-Inkling-2-Mini-Q8_0.gguf --local-dir ~/models/inkling-2-mini
```

## Run

Chat with Gemma (omit the prompt for an interactive session, where `/reset` and `/quit` work):

```sh
model=~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
trashcan chat "$model" "Hello there" --temperature 0
trashcan chat "$model" --draft ~/models/gemma-4-12b-assistant/model.safetensors
```

Serve the OpenAI Responses API on port 8000 ([docs/server.md](docs/server.md) covers clients and features):

```sh
trashcan serve "$model" --port 8000
```

Qwen3.5-9B, with speculative decoding from the model's MTP layer:

```sh
trashcan qwen-chat ~/models/qwen3.5-9b/Qwen3.5-9B-trashcan-Q4_0.gguf "Hello there" \
  --mtp ~/models/qwen3.5-9b/mtp-Qwen3.5-9B-trashcan-Q4_0.gguf
```

Models that span several Macs start a node on every Mac but the first, then chat or serve from the first:

- Qwen3.8-27B: `trashcan qwen-tp-node` on the others, `trashcan qwen-chat --tp` on the first
  ([docs/qwen.md](docs/qwen.md), which also covers cabling).
- Qwen3.8-Flash-Next: `trashcan qwen-node` along the chain ([docs/qwen4.md](docs/qwen4.md)).
- Inkling 2 Mini: `trashcan inkling node` on the others, `trashcan inkling chat` or `trashcan serve` on the first
  ([docs/inkling.md](docs/inkling.md#running)). For example, on two Macs:

  ```sh
  m=~/models/inkling-2-mini
  # on the second Mac
  trashcan inkling node $m/Inkling-2-Mini-UD-Q4_K_XL.gguf --listen mac1.tb1:7073
  # on the first
  trashcan inkling chat $m/Inkling-2-Mini-UD-Q4_K_XL.gguf --tp-connect mac1.tb1:7073 \
    --draft $m/dspark-Inkling-2-Mini-Q8_0.gguf "Hello"
  ```

## Deploy

[`deploy/trashcan-cluster`](docs/deploy.md) checks, starts and stops the published recipes, each a Responses
server on one, two, four or six Macs: `1-gemma`, `1-qwen9b`, `2-qwen27b`, `2-inkling`, `4-qwen27b` and
`6-flashnext`. Other machine counts have not been qualified.

## Test

```sh
cargo test
scripts/test-engine.sh [MODEL [ASSISTANT]]
```

`test-engine.sh` checks the tokenizer and chat template against fixtures, stress-tests GPU peer-to-peer copies,
and compares activations against [`tests/reference-vulkan/`](tests/reference-vulkan/).
[docs/implementation.md](docs/implementation.md) describes the validation and benchmark commands, and each model
doc describes its own checks against llama.cpp.

## Documentation

- [docs/hardware.md](docs/hardware.md): the Mac Pro 6,1 and D700s, the clock bug, Thunderbolt, fans
- [docs/kernel.md](docs/kernel.md): building and installing the patched kernel; the Thunderbolt links
- [docs/deploy.md](docs/deploy.md): setting up nodes and running the recipes
- [docs/implementation.md](docs/implementation.md): build, run, validation and benchmarks for Gemma 4
- [docs/engine-spec.md](docs/engine-spec.md): Gemma 4 math, constants, formats, tokenizer and GPU details
- [docs/qwen.md](docs/qwen.md): Qwen3.8-27B and Qwen3.5-9B: stages, tensor parallelism, MTP, prefix reuse
- [docs/qwen4.md](docs/qwen4.md): Qwen3.8-Flash-Next on a chain of Mac Pros
- [docs/inkling.md](docs/inkling.md): Inkling 2 Mini: looped MoE layers, NVFP4, vision, the DSpark drafter, two
  and four Macs
- [docs/server.md](docs/server.md): the Responses API server
- [docs/lights.md](docs/lights.md): the port lights as blinkenlights and a music visualizer (optional)

## Repository layout

- `engine/`: the Rust/Vulkan engine and its GLSL compute shaders
- `kernel/`: kernel patches, config and build script
- `nodes/macpro/`: setup script and system configuration for a Mac Pro node
- `deploy/`: cluster inventory example, recipes, pinned artifacts and `trashcan-cluster`
- `scripts/`: tests, benchmarks, clock recording and fan control
- `tools/`: activation and logit dumps, comparison, fixture generation and profiling tools
- `tests/`: tokenizer and chat fixtures, reference activations
- `bench/xgpu/`: cross-GPU sync and copy microbenchmark
- `lights/`: `portlights`, the port lights service and visualizer, with its install scripts
- `driver/`: userspace drivers for the Mac Pro: `gpud` for the D700s, `tbd` for Thunderbolt 2 (the engine links
  `tbd-client`), `ethd` for Ethernet ([driver/README.md](driver/README.md)). Not needed to run the engine on Linux.
- `dross/`: a snapshot of dross, a microkernel meant to run the engine and those drivers as an appliance, as of
  2026-11-03; it is developed in its own repository since. `vendor/dross/` pins its userspace SDK for the engine's
  optional `dross` feature.

## License

MIT; see [LICENSE](LICENSE). The GLSL shaders adapted from llama.cpp keep its MIT notice in
[engine/shaders/LICENSE.llama.cpp](engine/shaders/LICENSE.llama.cpp). The kernel patches in `kernel/` modify
GPL-2.0 Linux sources and are under GPL-2.0. Code imported into `driver/` and `dross/` keeps its own license
(see [driver/README.md](driver/README.md#licenses) and the `PROVENANCE.md` files).
