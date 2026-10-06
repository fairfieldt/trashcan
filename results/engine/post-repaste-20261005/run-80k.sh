#!/bin/bash
# 80K-context thermal/perf run with the pinned prefill-tiled-20261003 candidate binary,
# matching that report's run-long.sh command.
set -euo pipefail
cd "$HOME/trashcan"
exec 9>"$HOME/trashcan-exp/.lock"; flock 9
R=results/engine/post-repaste-20261005
B=results/engine/prefill-tiled-20261003/bin/candidate
M=$HOME/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
scripts/clockmon.sh "$R/ctx80000-clocks.txt" & mon=$!
bash "$R/cpumon.sh" "$R/ctx80000-cpu.txt" & cpu=$!
( while :; do echo "$(date +%s) $(cat /sys/class/hwmon/hwmon3/device/fan1_input)"; sleep 1; done > "$R/ctx80000-fan.txt" ) & fan=$!
trap "kill $mon $cpu $fan 2>/dev/null || true" EXIT
since=$(date "+%Y-%m-%d %H:%M:%S"); date -Is > "$R/ctx80000-start.txt"
sudo -n python3 scripts/latency-run.py $B bench $M --pp 80000 --tg 128 --batch 512 --progress --profile $R/ctx80000.tsv > $R/ctx80000.out 2> $R/ctx80000.log
date -Is > "$R/ctx80000-end.txt"
sudo -n journalctl -k --since "$since" > "$R/ctx80000-kernel.txt"
cat $R/ctx80000.out
