#!/bin/bash
# Recorded incite workflow, from ~/trashcan. Saved binaries pin the measured phases.
# Do not overlap with another GPU job; the caller must hold ~/trashcan-exp/.lock.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
R=results/engine/mtp-exploration-20261003
M=$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
D=$HOME/models/gemma-4-12b-assistant/model.safetensors
run() { sudo -n python3 scripts/latency-run.py "$@"; }

# At committed 013e343, copy target/release/trashcan to $R/bin/baseline and save
# baseline-{binary,source}.sha256. Git-content checks confirm all 40 engine files.
python3 "$R/run-phase.py" baseline

# Copy candidate engine/ with rsync -ac, then force a fresh shader/Rust build:
touch engine/build.rs engine/src/*.rs engine/shaders/*
cargo build --release > "$R/candidate-build.log" 2>&1
cp target/release/trashcan "$R/bin/candidate"
sha256sum target/release/trashcan > "$R/candidate-binary.sha256"
find engine -type f \( -name '*.rs' -o -name '*.comp' -o -name '*.glsl' -o -name Cargo.toml \) \
    -print0 | sort -z | xargs -0 sha256sum > "$R/candidate-source.sha256"
python3 "$R/run-micro.py" > "$R/micro-phase.log" 2>&1
python3 "$R/run-phase.py" candidate > "$R/candidate-phase.log" 2>&1
TRASHCAN_LONG_CHECK=1 bash scripts/test-engine.sh "$M" "$D" > "$R/validation-suite.log" 2>&1
sha256sum target/release/trashcan > "$R/load-first-acceptance-binary.sha256"
run target/release/trashcan check-batches "$M" --tokens 250 --batch 128 --draft "$D" \
    > "$R/boundary-250.out" 2> "$R/boundary-250.log"
# Fix benchmark defaults/GEMM comparison, rebuild, pin $R/bin/final (intermediate
# 35ffd4b8), and save load-first-{source,binary}.sha256 and the two source snapshots.
python3 "$R/run-exploration.py" > "$R/exploration-phase.log" 2>&1
# Change head batches 5..8 to multivec4, touch/build, pin $R/bin/head-tiles,
# save head-binary.sha256, then validate that selector independently:
python3 "$R/run-head-followup.py" > "$R/head-phase.log" 2>&1
TRASHCAN_LONG_CHECK=1 bash scripts/test-engine.sh "$M" "$D" > "$R/final-validation-suite.log" 2>&1
sha256sum target/release/trashcan > "$R/acceptance-binary.sha256"
cp "$R/acceptance-binary.sha256" "$R/final-binary.sha256"
find engine -type f \( -name '*.rs' -o -name '*.comp' -o -name '*.glsl' -o -name Cargo.toml \) \
    -print0 | sort -z | xargs -0 sha256sum > "$R/final-source.sha256"
# Each actual GPU phase ran clockmon.sh in the background with an EXIT kill trap.
# The full kernel journal since the round started is saved in reset-audit.txt.
python3 "$R/summarize.py" > "$R/summary-log.txt"
