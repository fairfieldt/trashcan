#!/bin/bash
# The raw ping-pong probe (tbcheck.py pp, CPU only, no follow-up frames) on all six ring links over fresh streams beside
# tblink's: `pb0` (HopID 14, busy_poll 0) and `pb1` (HopID 13, busy_poll 1), each used once, so no CLOSE of an earlier
# session waits in them (on 7.3 a device that has read a CLOSE refuses every write). Both are removed at the end. Under
# chain6.sh on 7.3; no module reloads or resets. DRY=1 prints the steps.
# usage: chain6.sh stream-bp2 8 OWNER -- results/kernel/usb4stream-busypoll-20261014/tools/probe-fresh.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
tools=$repo/results/kernel/usb4stream-reliable-20261013/tools
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/bp2-$(date +%Y%m%d-%H%M%S)}
F=trashcan-exp/flashnext
MACS="incite indite inflight alight invite enlight"
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

python3 "$tools/ring6.py" > "$OUT/ring.txt" || { log "the six links do not make a ring: $(cat "$OUT/ring.txt")"; exit 1; }
R=($(awk '{print $1}' "$OUT/ring.txt"))
next_of() { awk -v h="$1" '$1 == h {print $2}' "$OUT/ring.txt"; }
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under chain6.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    trap 'for j in $(jobs -p); do pkill -P $j 2>/dev/null; kill $j 2>/dev/null; done
        sleep 1
        for h in $MACS; do ssh "$h" "pkill -f \"[s]tream-tbcheck.py\"; sleep 0.6; ~/'$F'/stream-pg.sh down pb0; ~/'$F'/stream-pg.sh down pb1" >> "$OUT/cleanup.txt" 2>&1; done' EXIT
    trap 'exit 130' INT TERM
fi

log "output $OUT"
log "ring: $(awk '{print $1 "->" $3}' "$OUT/ring.txt" | tr '\n' ' ')"
for h in $MACS; do
    on "$h" "uname -r | grep -q '^7\.3' && test -x ~/$F/stream-pg.sh && test -f ~/$F/stream-tbcheck.py" || { log "$h: not on 7.3 or files missing"; exit 1; }
done
: > "$OUT/devs.txt"
for name in pb0 pb1; do
    case $name in pb0) hop=14 busy=0 ;; pb1) hop=13 busy=1 ;; esac
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        for pair in "$h $nx" "$nx $h"; do
            set -- $pair
            line=$(on "$1" "~/$F/stream-pg.sh up $2 $name $hop $busy" 2>> "$OUT/setup.err")
            echo "$name $1 $2 $line" >> "$OUT/devs.txt"
        done
    done
done
dry || { log "  probe streams: $(awk '{print $1, $2 "->" $3, $4, $5}' "$OUT/devs.txt" | tr '\n' ';')"; grep -q . "$OUT/setup.err" && { log "  setup failed: $(cat "$OUT/setup.err" | tr '\n' ';')"; exit 1; }; }
dev_of() { awk -v n="$1" -v h="$2" -v p="$3" '$1 == n && $2 == h && $3 == p {print $4}' "$OUT/devs.txt"; }

probe() { # name rounds: single 112-byte frames there and back on every link at once, no follow-up frames
    local name=$1 n=$2 h nx pids=""
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        on "$nx" "timeout -k 5 60 python3 ~/$F/stream-tbcheck.py pp $(dev_of "$name" "$nx" "$h") server $n 112 0 51" >> "$OUT/probe-$name.jsonl" 2>> "$OUT/probe-$name.err" &
        pids="$pids $!"
    done
    sleep 0.5
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        on "$h" "timeout -k 5 60 python3 ~/$F/stream-tbcheck.py pp $(dev_of "$name" "$h" "$nx") client $n 112 0 51" >> "$OUT/probe-$name.jsonl" 2>> "$OUT/probe-$name.err" &
        pids="$pids $!"
    done
    wait $pids
    dry || python3 "$tools/probe_table.py" "$OUT/probe-$name.jsonl" | tee -a "$OUT/steps.log"
}
log "1. busy_poll 0 (pb0), no follow-up frames"
probe pb0 25
log "2. busy_poll 1 (pb1), no follow-up frames"
probe pb1 25
log "done"
echo "stream-bp2 done: $OUT"
