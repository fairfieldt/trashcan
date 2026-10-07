#!/bin/bash
# Run on incite from ~/trashcan. Does not need a llama.cpp build.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
model=${1:-$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf}
assistant=${2:-$HOME/models/gemma-4-12b-assistant/model.safetensors}
cargo test
cargo build --release
bin=target/release/trashcan
"$bin" tokenize "$model" --fixture tests/fixtures/tokenizer-fixture.json
"$bin" render "$model" --fixture tests/fixtures/chat-fixture.json
"$bin" p2p-test --rounds 10000
for batch in 1 2 4 15; do
    out=results/engine/validation-b$batch
    "$bin" validate "$model" "$out" --batch "$batch" --greedy 10
    python3 tools/compare_activations.py "$out" results/reference-vulkan --check
done
out=results/engine/validation-cpu-relay
"$bin" validate "$model" "$out" --cpu-reduce --greedy 10
python3 tools/compare_activations.py "$out" results/reference-vulkan --check
if [ -f "$assistant" ]; then
    out=results/engine/validation-mtp
    "$bin" validate "$model" "$out" --batch 8 --draft "$assistant" --greedy 10
    python3 tools/compare_activations.py "$out" results/reference-vulkan --check
fi

if [ "${TRASHCAN_LONG_CHECK:-0}" = 1 ]; then
    args=()
    if [ -f "$assistant" ]; then args=(--draft "$assistant"); fi
    "$bin" check-batches "$model" --tokens 1700 --batch 512 "${args[@]}"
fi
