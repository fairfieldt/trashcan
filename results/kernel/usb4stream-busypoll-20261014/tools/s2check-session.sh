#!/bin/bash
# One six-Mac window under chain6.sh (all six leases) on 7.3.0-rc7-p2p-s2, in turn:
#  1. the lone-frame hold map (pathmap-session.sh: path tables, pos and pp on all six cables, OPENS per direction);
#  2. churn with busy and stopped servers (churnall-session.sh) with the old client build (CLIENT_OLD, before the
#     evidence rule) against the new server build: what the old one-way rule does with a busy accepting side;
#  3. the six-Mac bench check (benchcheck-session.sh): chain runs with the stages loaded, every link on its stream at
#     first try.
# Each part keeps its own output under OUT. DRY=1 passes through.
# usage: chain6.sh stream-s2check 25 OWNER -- bash results/kernel/usb4stream-busypoll-20261014/tools/s2check-session.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/s2check-$(date +%Y%m%d-%H%M%S)}
CLIENT_OLD=${CLIENT_OLD:-trashcan-exp/flashnext/trashcan-1687af5}
mkdir -p "$OUT"
child=
trap '[ -n "$child" ] && kill -TERM $child 2> /dev/null && wait $child; exit 130' INT TERM
part() { # NAME CMD...: one part in the background (a TERM reaches it), its rc
    "${@:2}" &
    child=$!
    wait $child
    local rc=$?
    child=
    echo "[$(date +%T)] part $1 rc $rc" | tee -a "$OUT/steps.log"
    return $rc
}
echo "[$(date +%T)] output $OUT" | tee -a "$OUT/steps.log"
part pathmap env OUT="$OUT/pathmap" OPENS="${OPENS:-4}" bash "$here/pathmap-session.sh"
r1=$?
part churn-old env OUT="$OUT/churn-old" CLIENT_BIN="$CLIENT_OLD" \
    MODES="load-old:50:1000:--load-ms_1000_--load-mb_256 freeze-old:30:1000:--freeze-ms_300" bash "$here/churnall-session.sh"
r2=$?
part bench env OUT="$OUT/bench" RUNS="${RUNS:-5}" bash "$here/benchcheck-session.sh"
r3=$?
echo "stream-s2check done: pathmap rc $r1, churn rc $r2, bench rc $r3: $OUT"
