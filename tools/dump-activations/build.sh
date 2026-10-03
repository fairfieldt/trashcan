#!/bin/bash
# Builds the dump tool on incite against the pristine upstream llama.cpp build in ~/llama.cpp (99b9548).
set -euo pipefail
cd "$(dirname "$0")"
L=$HOME/llama.cpp
g++ -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" dump.cpp -L"$L/build/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$L/build/bin" -o dump
