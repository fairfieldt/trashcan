#!/bin/bash
# Recorded workflow from ~/trashcan on incite. run-long/run-acceptance take
# their own lock; use a separate flock scope for short GPU commands.
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
R=results/engine/prefill-tiled-20261003
# At 91c1672, pin baseline binary/source before copying candidate engine/.
# Sync candidate with rsync -ac, explicitly invalidate shader/Rust mtimes:
touch engine/build.rs engine/src/*.rs engine/shaders/*
cargo build --release > "$R/build.log" 2>&1
cp target/release/trashcan "$R/bin/candidate"
sha256sum target/release/trashcan > "$R/candidate-binary.sha256"
find engine -type f \( -name '*.rs' -o -name '*.comp' -o -name '*.glsl' -o -name Cargo.toml \) \
 -print0 | sort -z | xargs -0 sha256sum > "$R/candidate-source.sha256"
(
 exec 9>"$HOME/trashcan-exp/.lock"
 flock 9
 scripts/clockmon.sh "$R/short-clocks.txt" & mon=$!
 trap 'kill "$mon" 2>/dev/null || true' EXIT
 python3 "$R/run-short.py" > "$R/short-phase.log" 2>&1
)
# The actual long run was stopped at user request after73,216 tokens.
bash "$R/run-long.sh"
# Prepared but cancelled before execution at user request:
# bash "$R/run-acceptance.sh" > "$R/acceptance-phase.log" 2>&1
python3 "$R/summarize.py" > "$R/summary-log.txt"
