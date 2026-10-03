# trashcan

A purpose-built inference engine for Gemma 4 12B QAT Q4_0 on the 2013 Mac Pro (incite: 2x FirePro D700),
written in Rust on raw Vulkan, with tensor parallelism across the two GPUs. llama.cpp is kept as the reference.

- `docs/hardware.md`: the machine, the D700 clock bug, baseline numbers
- `docs/upstream-survey.md`: upstream TP/meta-backend state, Vulkan PRs and forks
- `docs/engine-plan.md`: the Rust engine design (TP layout, schedule, prefill, MTP, milestones)
- `docs/engine-spec.md`: implementation reference: exact math, constants, formats, tokenizer, validation assets, GPU facts, conventions
- `docs/findings.md`: measurements and bugs, newest first
- `engine/`: the engine (`cargo build --release` on incite; `trashcan devices`, `trashcan info MODEL`)
- `bench/xgpu/`: cross-GPU sync and copy microbenchmark
- `kernel/`: custom kernel build (D700 fixes + PCI P2P)
- `scripts/`: incite helpers (`sync.sh` pushes the workspace and pulls `results/`)
- `tools/`: llama.cpp activation dumper and `compare_activations.py`
- `tests/fixtures/`: tokenizer fixture from llama.cpp
- `reference/llama.cpp-patches/`: our local llama.cpp patches from the investigation
- `results/`: benchmark outputs, model inventory, reference activations (`reference-vulkan/` is the correctness target)
- `llama.cpp/`: upstream clone used as reference (own git repo, not tracked here; PR branches fetched as local refs)
