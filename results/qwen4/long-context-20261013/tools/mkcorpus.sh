#!/bin/bash
# Concatenate the repository's docs and results READMEs into one text (a long real prompt) and tokenize it as one
# user turn. usage: mkcorpus.sh REPO OUTDIR
set -euo pipefail
repo=$1 out=$2
cd "$repo"
{
    cat README.md
    for f in docs/qwen4.md docs/qwen.md docs/engine-spec.md docs/engine-plan.md docs/implementation.md docs/kernel.md \
        docs/findings.md docs/hardware.md docs/server.md docs/tp8-optimization.md docs/upstream-survey.md; do
        printf '\n\n# File: %s\n\n' "$f"
        cat "$f"
    done
    for f in $(find results -name README.md | sort); do
        printf '\n\n# File: %s\n\n' "$f"
        cat "$f"
    done
} > "$out/corpus.txt"
wc -c "$out/corpus.txt"
./target/release/trashcan qwen4 tokenize-chat /Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf \
    "$out/corpus.txt" > "$out/corpus.ids"
wc -w "$out/corpus.ids"
