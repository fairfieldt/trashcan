#!/bin/bash
# Builds the dump tool against an unpatched upstream llama.cpp checkout and build at commit 99b9548.
# LLAMA is the checkout (default ~/llama.cpp) and BUILD its build directory holding the libraries (default
# $LLAMA/build/bin), e.g. a Vulkan build on a Mac Pro or a Metal build on a Mac.
# usage: LLAMA=PATH [BUILD=PATH] tools/dump-activations/build.sh
set -euo pipefail
cd "$(dirname "$0")"
L=${LLAMA:-$HOME/llama.cpp}
B=${BUILD:-$L/build/bin}
${CXX:-g++} -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" dump.cpp -L"$B" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$B" -o dump
