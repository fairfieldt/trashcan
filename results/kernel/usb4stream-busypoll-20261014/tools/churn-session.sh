#!/bin/bash
# link-test churn on one cable, A -> B, under leases on both Macs (pair2.sh), CPU only; no module reloads, Thunderbolt
# resets or reboots: the kernels and the A-B link; B's link-test server (restarted after each crash); N churn sessions
# from A (normal exits, killed clients and server crashes mixed); the server's log of them; the link-test suite.
# DRY=1 prints the steps.
# usage: pair2.sh stream-churn 20 OWNER A B -- results/kernel/usb4stream-busypoll-20261014/tools/churn-session.sh A B N
set -uo pipefail
A=$1 B=$2 N=${3:-200}
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/churn-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
BIN=$W/trashcan-stream-churn
PORT=7075
STOP=$W/stream-cc.stop
DEV_A=/dev/tbstreamA DEV_B=/dev/tbstreamB ADDR=$B.tbN    # from the live links (state)
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

cleanup() {
    for h in $A $B; do
        ssh -o ConnectTimeout=10 "$h" "touch ~/$STOP; pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; pkill -f '[t]rashcan-stream-churn link-test'; true" >> "$OUT/cleanup.txt" 2>&1
    done
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under pair2.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

log "output $OUT; $A connects, $B accepts; $N sessions"
for h in $A $B; do
    on "$h" "uname -r; md5sum < ~/$BIN | cut -c1-12; /usr/local/sbin/tblink links" > "$OUT/links-$h.txt" 2>&1
done
if ! dry; then
    log "  $A: $(head -2 "$OUT/links-$A.txt" | tr '\n' ' '); $B: $(head -2 "$OUT/links-$B.txt" | tr '\n' ' ')"
    DEV_A=$(awk -F'\t' -v p="$B" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$A.txt")
    DEV_B=$(awk -F'\t' -v p="$A" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$B.txt")
    ADDR=$B.tb$(awk -F'\t' -v p="$B" 'NR > 3 && $2 == p {print $3}' "$OUT/links-$A.txt")
    log "  $A $DEV_A <-> $B $DEV_B, $A connects to $ADDR"
    case "$DEV_A$DEV_B" in /dev/tbstream*/dev/tbstream*) ;; *) log "STOP: the $A-$B link is not up on two lanes with a stream at both ends"; exit 1 ;; esac
fi

# B's server, restarted after each crash: only the loop goes to the background, with its output in the log.
n=$(on "$B" "cat ~/$W/stream-cc-serve.log 2>/dev/null | wc -l")
on "$B" "rm -f ~/$STOP; pgrep -f '^[^ ]*/trashcan-stream-churn link-test serve --listen 0.0.0.0:$PORT' > /dev/null || { setsid nohup bash -c 'while [ ! -f ~/$STOP ]; do ~/$BIN link-test serve --listen 0.0.0.0:$PORT; sleep 0.05; done' >> ~/$W/stream-cc-serve.log 2>&1 < /dev/null & }"
log "1. $N churn sessions"
on "$A" "timeout -k 10 1800 ~/$BIN link-test churn --connect $ADDR:$PORT --sessions $N --label churn" > "$OUT/churn.jsonl" 2> "$OUT/churn.err"
rc=$?
dry || log "  churn rc $rc: $(tail -1 "$OUT/churn.jsonl")"
log "2. the link-test suite"
for m in "--mode pingpong --size 64 --count 5000" "--mode pingpong --size 20480 --count 2000" \
    "--mode exchange --size 40960 --count 2000" "--mode bulk --size 5242880 --count 20" "--mode chain --count 1000"; do
    on "$A" "timeout -k 10 300 ~/$BIN link-test run --connect $ADDR:$PORT --label suite $m" >> "$OUT/suite.jsonl" 2>> "$OUT/suite.err"
done
dry || log "  suite: $(grep -c '"result"' "$OUT/suite.jsonl") of 5 tests passed, $(grep -c ' not opened (' "$OUT/suite.err") failed epochs, $(grep -c 'stream down' "$OUT/suite.err") fallbacks"
on "$B" "touch ~/$STOP; pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; sleep 0.5; tail -n +$((${n:-0} + 1)) ~/$W/stream-cc-serve.log" > "$OUT/serve.log" 2>&1
dry || log "  server: $(grep -c ' not opened (' "$OUT/serve.log") failed epochs, $(grep -c 'exiting at' "$OUT/serve.log") exits, $(grep -c 'opening the stream device took' "$OUT/serve.log") slow opens"
log "done"
echo "stream-churn done: $OUT"
