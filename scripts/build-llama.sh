#!/bin/bash
# Builds ~/trashcan/llama.cpp (whatever is checked out) with Vulkan into build-<name>, Release plus -g so
# backtraces have symbols. GGML_NATIVE picks AVX/F16C for the Ivy Bridge Xeon.
# usage (on incite): build-llama.sh NAME [extra cmake args...]
set -euo pipefail
name=$1; shift
cd "$HOME/trashcan/llama.cpp"
cmake -S . -B "build-$name" -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS=-g -DCMAKE_CXX_FLAGS=-g -DLLAMA_CURL=OFF "$@" > /dev/null
cmake --build "build-$name" -j"$(nproc)" --target llama-bench llama-cli llama-server llama-perplexity llama-completion test-backend-ops 2>&1 | tail -3
git log -1 --format="built build-$name at %h %s"
