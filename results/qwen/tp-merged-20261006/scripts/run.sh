#!/bin/bash
# One run on this Mac inside a gpuq lease: clock log, 0 us CPU latency request, switches from ENV through sudo.
# usage: [ENV="VAR=1 ..."] run.sh NAME CMD [ARGS...]
#   stdout -> results/NAME.json, stderr -> results/NAME.err, clocks -> results/clocks/NAME.txt
set -uo pipefail
cd ~/trashcan-exp/merged
name=$1; shift
mkdir -p results/clocks
echo "$(date -Is) $(hostname) $name: ${ENV:-} $*" >> results/commands.log
scripts/with-clocks.sh results/clocks/$name.txt sudo -n env ${ENV:-} python3 scripts/latency-run.py "$@" > results/$name.json 2> results/$name.err
rc=$?
grep -E "\[clocks\]|rror|panick" results/$name.err | tail -5
echo "$name rc=$rc"
exit $rc
