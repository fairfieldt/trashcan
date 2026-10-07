#!/bin/bash
# Run one benchmark under the lease with a clock log and a 0 us CPU latency request.
# usage: [ENV="VAR=1 ..."] run.sh NAME CMD [ARGS...]
#   stdout -> results/tp-prefill/NAME.json, stderr -> NAME.err, clocks -> clocks/NAME.txt
set -uo pipefail
cd ~/trashcan-exp/prefill
name=$1; shift
out=results/tp-prefill
mkdir -p $out/clocks
echo "$(date -Is) $name: ${ENV:-} $*" >> $out/commands.log
scripts/with-clocks.sh $out/clocks/$name.txt sudo -n env ${ENV:-} python3 scripts/latency-run.py "$@" > $out/$name.json 2> $out/$name.err
rc=$?
grep -E "clocks|error|Error|panick" $out/$name.err | tail -5
head -c 3000 $out/$name.json; echo
exit $rc
