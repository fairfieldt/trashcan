#!/bin/bash
# llama-bench for Gemma 4 12B QAT Q4_0 on both D700s, layer split vs tensor split, with clock logging.
# usage: bench-gemma.sh LABEL [extra llama-bench args...]
#   BIN=dir with llama-bench (default ~/llama.cpp/build/bin); results go to ~/trashcan/results/<date>-LABEL/
set -u
label=$1; shift
BIN=${BIN:-$HOME/llama.cpp/build/bin}
M=${M:-$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf}
out=$HOME/trashcan/results/$(date +%Y%m%d-%H%M)-$label
mkdir -p "$out"
"$HOME/trashcan/scripts/clockmon.sh" "$out/clocks.txt" &
cm=$!
trap 'kill $cm' EXIT

{
    echo "label: $label"
    echo "bin: $BIN ($(cd "$BIN/../.." && git log -1 --format='%h %s' 2>/dev/null))"
    echo "kernel: $(uname -r)"
    echo "env: $(env | grep -E '^(GGML_|RADV_)' | tr '\n' ' ')"
    echo "args: $*"
} > "$out/info.txt"

"$BIN/llama-bench" -m "$M" -ngl 99 -sm layer  -fa 0,1 -p 512 -n 128 -r 3 "$@" 2>&1 | tee "$out/layer.txt"
"$BIN/llama-bench" -m "$M" -ngl 99 -sm tensor -fa 1   -p 512 -n 128 -r 3 "$@" 2>&1 | tee "$out/tensor.txt"

kill $cm
trap - EXIT
"$HOME/trashcan/scripts/clocksummary.sh" "$out/clocks.txt" | tee "$out/clocks-summary.txt"
echo "results in $out"
