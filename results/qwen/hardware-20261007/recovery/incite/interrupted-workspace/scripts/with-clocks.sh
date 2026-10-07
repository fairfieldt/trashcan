#!/bin/bash
# Run a command while sampling both D700s' clocks, then print the clock summary, so benchmark records carry
# sustained clocks (AGENTS.md). Run it under a gpuq lease, from a workspace holding scripts/:
#   ssh incite 'cd ~/trashcan-exp/NAME && gpuq run ID -- scripts/with-clocks.sh results/clocks/RUN.txt CMD ARGS'
# usage: scripts/with-clocks.sh CLOCKLOG CMD [ARGS...]
set -uo pipefail
log=$1; shift
here=$(dirname "$0")
mkdir -p "$(dirname "$log")"
"$here/clockmon.sh" "$log" &
mon=$!
"$@"
rc=$?
kill $mon 2>/dev/null; wait $mon 2>/dev/null
echo "[clocks] $("$here/clocksummary.sh" "$log" | tr '\n' ' ') log=$log" >&2
exit $rc
