#!/bin/bash
set -euo pipefail
cd ~/trashcan-exp/prefill-lds-20261007
name=$1; shift
mkdir -p results/clocks
printf '%s %s %s\n' "$(date -Is)" "$name" "$*" >> results/commands.log
scripts/with-clocks.sh "results/clocks/$name.txt" sudo -n env ${ENV:-} python3 scripts/latency-run.py "$@" > "results/$name.json" 2> "results/$name.err"
