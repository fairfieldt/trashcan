#!/bin/bash
set -euo pipefail
cd "$HOME/trashcan"
exec 9>"$HOME/trashcan-exp/.lock"
flock 9
R=results/engine/prefill-tiled-20261003
M=$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
grep -q '^PASS short suite$' "$R/short-phase.log"
scripts/clockmon.sh "$R/long-clocks.txt" & clock_pid=$!
trap 'kill "$clock_pid" 2>/dev/null || true' EXIT
sudo -n journalctl -k --since '2026-10-03 12:20:00' > "$R/resets-before.txt"
date -Is > "$R/long-start.txt"
sudo -n python3 scripts/latency-run.py "$R/bin/candidate" bench "$M" \
 --pp 80000 --tg 128 --batch 512 --progress --profile "$R/ctx80000.tsv" \
 > "$R/ctx80000.out" 2> "$R/ctx80000.log" & bench_pid=$!
python3 "$R/memorymon.py" "$bench_pid" "$R/long-memory.csv" & memory_pid=$!
if wait "$bench_pid"; then result=0; else result=$?; fi
wait "$memory_pid"
date -Is > "$R/long-end.txt"
sudo -n journalctl -k --since '2026-10-03 12:20:00' > "$R/resets-after.txt"
exit "$result"
