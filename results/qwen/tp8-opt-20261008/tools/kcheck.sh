#!/bin/bash
# qwen-kernel-check on every Mac's two D700s under a gpuq lease, with clocks: the new kernels against the ones they
# replace (bit for bit where they should match), their CPU references, and timings.
# usage: TICKETS=incite=ID,invite=ID,enlight=ID,indite=ID kcheck.sh [BINARY] [ITERATIONS]
set -euo pipefail
ROOT=${TP8_ROOT:-/home/fairfieldt/trashcan-exp/tp8opt}
binary=${1:-trashcan-opt}
iterations=${2:-200}
for t in ${TICKETS//,/ }; do
  host=${t%%=*} id=${t#*=}
  ssh "$host" "gpuq run $id -- bash -c 'cd $ROOT; mkdir -p results/clocks; scripts/with-clocks.sh results/clocks/kcheck-$host.txt ./$binary qwen-kernel-check --iterations $iterations > results/kcheck-$host.jsonl 2> results/kcheck-$host.err; echo $host: \$?; tail -2 results/kcheck-$host.err'" &
done
wait
