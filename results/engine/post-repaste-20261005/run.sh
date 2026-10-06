#!/bin/bash
# Post-repaste sanity check: pinned candidate binary from prefill-tiled-20261003 (commit b96946e).
set -euo pipefail
cd ~/trashcan
R=results/engine/post-repaste-20261005
B=results/engine/prefill-tiled-20261003/bin/candidate
M=~/models/gemma-4-12b-qat-gguf/gemma-4-12b-it-qat-q4_0.gguf
exec 9>"$HOME/trashcan-exp/.lock"; flock 9
scripts/clockmon.sh "$R/clocks${TAG:-}.txt" & mon=$!
( while :; do echo "$(date +%s) $(cat /sys/class/hwmon/hwmon3/device/fan1_input)"; sleep 1; done > "$R/fan${TAG:-}.txt" ) & fan=$!
trap "kill $mon $fan 2>/dev/null || true" EXIT
for rep in ${REPS:-1 2}; do for pp in ${PPS:-512 4096}; do
  l=pp$pp-$rep
  sudo -n python3 scripts/latency-run.py $B bench $M --pp $pp --tg 128 --batch 512 --progress --profile $R/$l.tsv > $R/$l.out 2> $R/$l.log
  echo PASS $l; cat $R/$l.out
done; done
