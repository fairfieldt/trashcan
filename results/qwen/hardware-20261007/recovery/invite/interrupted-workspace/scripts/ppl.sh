#!/bin/bash
# Correctness check: perplexity of Gemma 4 12B on a fixed text, layer split vs tensor split (both with FA, since
# tensor split forces it). Tensor parallel only changes summation order and the allreduce precision, so the two
# PPLs should agree closely; a large gap means a TP or kernel bug.
# usage (on incite): ppl.sh LABEL [extra llama-perplexity args...]
set -u
label=$1; shift
BIN=${BIN:-$HOME/trashcan/llama.cpp/build-tp/bin}
M=${M:-$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf}
text=$HOME/trashcan/ppl-text.txt
if [ ! -s "$text" ]; then
    cat "$HOME"/trashcan/llama.cpp/docs/build.md "$HOME"/trashcan/llama.cpp/tools/server/README.md > "$text"
fi
out=$HOME/trashcan/results/$(date +%Y%m%d-%H%M)-ppl-$label
mkdir -p "$out"
for sm in layer tensor; do
    "$BIN/llama-perplexity" -m "$M" -f "$text" -ngl 99 -sm $sm -fa 1 -c 512 --chunks 12 "$@" > "$out/$sm.txt" 2>&1
    echo "$sm: $(grep -E 'Final estimate' "$out/$sm.txt" || tail -3 "$out/$sm.txt")"
done
echo "results in $out"
