#!/bin/bash
# Run one benchmark command on a node inside a gpuq lease (`gpuq run ID -- ~/trashcan-exp/xmac/xrun.sh CMD...`),
# logging the D700 clocks for its lifetime and holding a 0 us CPU latency request (scripts/latency-run.py).
# gpuq already holds ~/trashcan-exp/.lock.
# usage: xrun.sh CMD [ARG ...]
set -uo pipefail
dir=$HOME/trashcan-exp/xmac
mkdir -p "$dir/results/clocks"
cd "$dir"
clocks=$dir/results/clocks/$(date +%Y%m%dT%H%M%S)-$$.txt
"$HOME/trashcan/scripts/clockmon.sh" "$clocks" &
mon=$!
trap 'kill $mon 2>/dev/null' EXIT INT TERM
sudo -n python3 "$HOME/trashcan/scripts/latency-run.py" "$@"
rc=$?
kill $mon 2>/dev/null; wait $mon 2>/dev/null
echo "[clocks] $("$HOME/trashcan/scripts/clocksummary.sh" "$clocks" | tr '\n' ' ') log=$clocks" >&2
exit $rc
