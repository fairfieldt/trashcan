#!/bin/bash
# Builds the logits tool and the dequantizer against a llama.cpp checkout and build (the Inkling reference: master
# 29b52d398 with the loop-scale patch). LLAMA is the checkout (default ~/llama.cpp), BUILD its library directory
# (default $LLAMA/build/bin). The binaries land in OUT_DIR (default: this directory).
# usage: LLAMA=PATH [BUILD=PATH] tools/dump-logits/build.sh [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")"
L=${LLAMA:-$HOME/llama.cpp}
B=${BUILD:-$L/build/bin}
O=${1:-.}
${CXX:-g++} -O2 -std=c++17 -I"$L/include" -I"$L/ggml/include" logits.cpp -L"$B" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$B" -o "$O/logits"
${CXX:-g++} -O2 -std=c++17 -I"$L/ggml/include" dequant.cpp -L"$B" -lggml-base -Wl,-rpath,"$B" -o "$O/dequant"
