#!/bin/bash
# link-test churn on all six ring links at once, under chain6.sh (all six leases), CPU only, no reboots or resets: on
# each cable, sessions from the Mac before it in the ring to the Mac after it, whose link-test server runs in a loop
# that restarts it after each crash. One round per mode in MODES, each NAME:SESSIONS:WITHIN_MS:SERVER_OPTIONS (the
# server's load: `--load-ms`, `--freeze-ms`); a link passes a round when every session reached its stream within
# WITHIN_MS of its link's start with no failed epoch. CLIENT_BIN: the clients' build (default BIN, the servers').
# DRY=1 prints the steps.
# usage: chain6.sh stream-churnall 15 OWNER -- bash results/kernel/usb4stream-busypoll-20261014/tools/churnall-session.sh
set -uo pipefail
BIN=${BIN:-trashcan-exp/flashnext/trashcan-41da714}
CLIENT_BIN=${CLIENT_BIN:-$BIN}
BN=$(basename "$BIN") CN=$(basename "$CLIENT_BIN")
MODES=${MODES:-"base:50:100: load:50:100:--load-ms_1000_--load-mb_256 freeze:30:1000:--freeze-ms_300"}
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/churnall-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
RING="incite inflight alight indite invite enlight"
PORT0=7090
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }
next() { local r=($RING) i; for i in 0 1 2 3 4 5; do [ "${r[$i]}" = "$1" ] && echo "${r[$(((i + 1) % 6))]}"; done; }
port() { local r=($RING) i; for i in 0 1 2 3 4 5; do [ "${r[$i]}" = "$1" ] && echo $((PORT0 + i)); done; }

cleanup() {
    local h
    for h in $RING; do
        ssh -o ConnectTimeout=10 "$h" "touch ~/$W/stream-churnall.stop; pkill -f '[l]ink-test serve --listen 0.0.0.0:709'; pkill -f '[${BN:0:1}]${BN:1} link-test'; pkill -f '[${CN:0:1}]${CN:1} link-test'; true" >> "$OUT/cleanup.txt" 2>&1
    done
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under chain6.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

log "output $OUT; server build ~/$BIN, client build ~/$CLIENT_BIN; modes $MODES"
ok=1
for h in $RING; do
    on "$h" "uname -r; echo \$(md5sum < ~/$BIN | cut -c1-12) \$(md5sum < ~/$CLIENT_BIN | cut -c1-12); /usr/local/sbin/tblink links" > "$OUT/state-$h.txt" 2>&1
    dry && continue
    n=$(next "$h")
    b=$(awk -F'\t' -v p="$n" 'NR > 3 && $2 == p {print ($6 == "up" && $5 == 2 && $10 ~ /^\/dev\/tbstream/) ? "up" : $6 "/" $5 "/" $10}' "$OUT/state-$h.txt")
    [ "$b" = up ] || { ok=0; log "  $h -> $n: ${b:-no link}"; }
done
if ! dry; then
    log "  kernels and build: $(for h in $RING; do printf '%s %s %s; ' "$h" "$(sed -n 1p "$OUT/state-$h.txt")" "$(sed -n 2p "$OUT/state-$h.txt")"; done)"
    [ $ok = 1 ] || { log "STOP: not every ring cable is up on two lanes with a stream at both ends"; exit 1; }
fi

failed=0
for mode in $MODES; do
    IFS=: read -r name sessions within sopts <<< "$mode"
    sopts=${sopts//_/ }
    log "$name: $sessions sessions per link, within $within ms; server options: ${sopts:-none}"
    for h in $RING; do
        n=$(next "$h") p=$(port "$h")
        on "$n" "cat ~/$W/stream-churnall-$p.log 2> /dev/null | wc -l" > "$OUT/serve-start-$name-$p.txt" 2> /dev/null
        # Only the loop goes to the background, its output in the log, so ssh returns at once.
        on "$n" "rm -f ~/$W/stream-churnall.stop; pgrep -f '^[^ ]*/$BN link-test serve --listen 0.0.0.0:$p' > /dev/null || { setsid nohup bash -c 'while [ ! -f ~/$W/stream-churnall.stop ]; do ~/$BIN link-test serve --listen 0.0.0.0:$p $sopts; sleep 0.05; done' >> ~/$W/stream-churnall-$p.log 2>&1 < /dev/null & }"
    done
    pids=()
    for h in $RING; do
        n=$(next "$h") p=$(port "$h")
        bus=$(awk -F'\t' -v q="$n" 'NR > 3 && $2 == q {print $3}' "$OUT/state-$h.txt" 2> /dev/null)
        on "$h" "timeout -k 10 900 ~/$CLIENT_BIN link-test churn --connect $n.tb${bus:-0}:$p --sessions $sessions --within-ms $within --label $name-$h-$n" > "$OUT/churn-$name-$h-$n.jsonl" 2> "$OUT/churn-$name-$h-$n.err" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    for h in $RING; do
        n=$(next "$h") p=$(port "$h")
        m=$(tr -dc 0-9 < "$OUT/serve-start-$name-$p.txt" 2> /dev/null)
        on "$n" "touch ~/$W/stream-churnall.stop; pkill -f '[l]ink-test serve --listen 0.0.0.0:$p'; sleep 0.5; tail -n +$((${m:-0} + 1)) ~/$W/stream-churnall-$p.log" > "$OUT/serve-$name-$h-$n.log" 2>&1
    done
    for h in $RING; do
        n=$(next "$h")
        s=$(tail -1 "$OUT/churn-$name-$h-$n.jsonl" 2> /dev/null)
        case "$s" in *'"pass":true'* | *'"pass": true'*) ;; *) failed=$((failed + 1)) ;; esac
        dry || log "  $h -> $n: $(python3 "$(dirname "$0")/churn_table.py" --line "$OUT/churn-$name-$h-$n.jsonl" 2>&1 | tail -1); server: $(grep -c ' not opened (' "$OUT/serve-$name-$h-$n.log") failed epochs"
    done
done
dry && failed=0
log "done: $failed link rounds did not pass"
echo "stream-churnall done: $failed link rounds did not pass"
