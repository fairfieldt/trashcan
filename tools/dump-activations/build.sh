#!/bin/bash
# Builds the dump tool on incite against the pristine upstream llama.cpp build in ~/llama.cpp (99b9548).
# LLAMA and BUILD select another checkout and build directory (e.g. a Metal build on a Mac).
set -euo pipefail
cd "$(dirname "$0")"
L=${LLAMA:-$HOME/llama.cpp}
B=${BUILD:-$L/build/bin}
${CXX:-g++} -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" dump.cpp -L"$B" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$B" -o dump
# The MTP reference needs a build with draft-mtp (llama.cpp b8e9ce8) and its internal llama-ext.h.
if [ -f "$L/src/llama-ext.h" ]; then
    ${CXX:-g++} -O2 -std=c++17 -I"$L/include" -I"$L/src" -I"$L/ggml/include" mtp.cpp -L"$B" -lllama -lggml \
        -lggml-base -Wl,-rpath,"$B" -o mtp
fi
