#!/bin/bash
set -euo pipefail
cd "$HOME/trashcan"
export PATH="$HOME/.cargo/bin:$PATH"
exec 9>"$HOME/trashcan-exp/.lock"
flock 9
R=results/engine/prefill-tiled-20261003
M=$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
D=$HOME/models/gemma-4-12b-assistant/model.safetensors
grep -q '^PASS short suite$' "$R/short-phase.log"
scripts/clockmon.sh "$R/acceptance-clocks.txt" & mon=$!
trap 'kill "$mon" 2>/dev/null || true' EXIT
TRASHCAN_LONG_CHECK=1 sudo -n --preserve-env=PATH,TRASHCAN_LONG_CHECK python3 scripts/latency-run.py \
 bash scripts/test-engine.sh "$M" "$D" > "$R/validation-suite.log" 2>&1
sha256sum target/release/trashcan > "$R/acceptance-binary.sha256"
# The static 15-token reference never crosses the tiled cutover. check-batches
# above (with assistant) and the short suite (without) exercise actual tiles.
sudo -n python3 scripts/latency-run.py target/release/trashcan check-batches "$M" --tokens 250 --batch 128 --draft "$D" > "$R/boundary.out" 2> "$R/boundary.log"
sudo -n python3 scripts/latency-run.py target/release/trashcan check-batches "$M" --tokens 4096 --batch 512 > "$R/check4096.out" 2> "$R/check4096.log"
sudo -n journalctl -k --since '2026-10-03 12:20:00' > "$R/reset-audit.txt"
echo 'PASS acceptance suite'
