#!/bin/bash
# Under chain6.sh (all six leases): does a link's stderr write hold the link up? A write-latency probe
# (stream-wlat.py) runs on every Mac in the directory of the nodes' logs, while RUNS bench-check runs
# (benchcheck-session.sh) go with OLD, the build that writes link messages while holding the link (41da714), and RUNS
# with NEW, the build that writes them from a thread of their own (4f36bf0). Each run's links come from the driver's
# `links:` line (time to stream per link); the probe lists every write over 20 ms with its time of day.
# usage: chain6.sh stream-logcheck 20 OWNER -- bash results/kernel/usb4stream-busypoll-20261014/tools/logcheck-session.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/logcheck-$(date +%Y%m%d-%H%M%S)}
OLD=${OLD:-trashcan-41da714} NEW=${NEW:-trashcan-4f36bf0} RUNS=${RUNS:-5}
W=trashcan-exp/flashnext
MACS="incite inflight alight indite invite enlight"
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$@"; }
child=
wl=()
stop_probes() {
    local h p
    for h in $MACS; do on "$h" "touch ~/$W/results/.wlat-$h.stop" > /dev/null 2>&1; done
    for p in "${wl[@]}"; do wait "$p"; done
    wl=()
}
cleanup() {
    local h
    for h in $MACS; do on "$h" "touch ~/$W/results/.wlat-$h.stop; pkill -f '[s]tream-wlat.py'; true" >> "$OUT/cleanup.txt" 2>&1; done
}
if [ "${DRY:-0}" != 1 ]; then
    [ -n "${TICKETS:-}" ] || { echo "run under chain6.sh (TICKETS)" >&2; exit 1; }
    trap 'cleanup' EXIT
fi
trap '[ -n "$child" ] && kill -TERM $child 2> /dev/null && wait $child; exit 130' INT TERM
part() { # NAME CMD...: one part in the background (a TERM reaches it), its rc
    "${@:2}" &
    child=$!
    wait $child
    local rc=$?
    child=
    log "part $1 rc $rc"
    return $rc
}
log "output $OUT; old $OLD, new $NEW, $RUNS runs each"
for h in $MACS; do
    on "$h" "rm -f ~/$W/results/.wlat-$h.stop; python3 ~/$W/stream-wlat.py ~/$W/results/.wlat-$h 1200" > "$OUT/wlat-$h.json" 2>&1 &
    wl+=($!)
done
part old env OUT="$OUT/old" BIN="$OLD" TAG="lc-${OLD#trashcan-}" RUNS="$RUNS" bash "$here/benchcheck-session.sh"
part new env OUT="$OUT/new" BIN="$NEW" TAG="lc-${NEW#trashcan-}" RUNS="$RUNS" bash "$here/benchcheck-session.sh"
stop_probes
for h in $MACS; do
    log "  writes on $h: $(cut -c1-300 "$OUT/wlat-$h.json")"
done
log "done"
echo "stream-logcheck done: $OUT"
