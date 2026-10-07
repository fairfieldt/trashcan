#!/bin/bash
# scripts/test-engine.sh's checks with the cross-built binary, run from this workspace instead of ~/trashcan
# (`cargo test` ran on the Mac; nothing builds on the Macs). Needs tests/fixtures, results/reference-vulkan and
# tools/compare_activations.py next to it. usage: gemma-test.sh [MODEL [ASSISTANT]]
set -euo pipefail
cd ~/trashcan-exp/merged
model=${1:-$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf}
assistant=${2:-$HOME/models/gemma-4-12b-assistant/model.safetensors}
bin=./trashcan
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
echo "gemma checks passed"
