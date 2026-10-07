#!/bin/bash
# Reproducible engine measurements on incite; PM QoS is scoped to each command.
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:-$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf}
assistant=${2:-$HOME/models/gemma-4-12b-assistant/model.safetensors}
out=${3:-results/engine/benchmark}
mkdir -p "$out"
bin=target/release/trashcan
run() {
    if [ "${TRASHCAN_CPU_LATENCY:-1}" = 1 ]; then
        sudo -n python3 scripts/latency-run.py "$@"
    else
        "$@"
    fi
}
uname -a > "$out/kernel.txt"
"$bin" devices > "$out/devices.txt"
sha256sum Cargo.toml Cargo.lock engine/Cargo.toml engine/build.rs engine/src/*.rs engine/shaders/*.comp engine/shaders/*.glsl > "$out/source.sha256"
sha256sum "$bin" > "$out/binary.sha256"
scripts/clockmon.sh "$out/clocks.txt" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
run "$bin" matvec-bench "$model" --iterations 100 > "$out/matvec.json" 2> "$out/matvec.log"
run "$bin" bench "$model" --pp 512 --tg 128 --batch 512 --profile "$out/profile.tsv" > "$out/bench.json" 2> "$out/bench.log"
bench_chat() {
    local name=$1 prompt=$2
    printf '%s\n' "$prompt" > "$out/$name-prompt.txt"
    run "$bin" chat "$model" "$prompt" --batch 128 --temperature 0 --max-tokens 128 > "$out/$name-target.txt" 2> "$out/$name-target.log"
    if [ -f "$assistant" ]; then
        run "$bin" chat "$model" "$prompt" --batch 128 --temperature 0 --max-tokens 128 --draft "$assistant" > "$out/$name-mtp.txt" 2> "$out/$name-mtp.log"
        cmp "$out/$name-target.txt" "$out/$name-mtp.txt"
        echo "PASS: ordinary and MTP greedy output is identical ($name)"
    fi
}
bench_chat chat 'Explain how a transformer works, step by step, in English.'
bench_chat chat-photosynthesis 'Explain photosynthesis to a ten-year-old in three short paragraphs.'
bench_chat chat-csv 'Write a Python function that reads a CSV file and returns rows where the age column is at least 18. Include a short example.'
