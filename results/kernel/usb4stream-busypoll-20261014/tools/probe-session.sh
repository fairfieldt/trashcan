#!/bin/bash
# One cable, A <-> B, on any kernel (no switches set), under leases on both Macs (pair2.sh), CPU only; no reloads,
# resets or reboots: the lone-frame hold as it stands, the control for knobs-session.sh. The raw probes over OPENS fresh
# opens per direction (PROBES, default "pp pos": tbcheck.py pp, single frames there and back with nothing after them,
# fillers per round trip; pos: the fillers a lone frame needed before the other side answered, its answers pushed by
# three fillers), then CHURN (default 0) link-test churn sessions.
# DRY=1 prints the steps.
# usage: pair2.sh stream-probe 10 OWNER A B -- bash results/kernel/usb4stream-busypoll-20261014/tools/probe-session.sh A B
set -uo pipefail
A=$1 B=$2
OPENS=${OPENS:-10} CHURN=${CHURN:-0}
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/probe-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
BIN=$W/trashcan-stream-churn
PORT=7075
STOP=$W/stream-cc.stop
DEFAULTS="rx_e2e=1 paths_last=0 poll_mode=0 rx_repost=0 rx_eof_close=1"
# Settings in the coordinator's order: defaults, setup order, poll mode, re-posting, E2E, eof mask, combinations.
SETTINGS=as-it-is
DEV_A=/dev/tbstreamA DEV_B=/dev/tbstreamB ADDR=$B.tbN
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

cleanup() {
    for h in $A $B; do
        ssh -o ConnectTimeout=10 "$h" "touch ~/$STOP; pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; pkill -f '[t]rashcan-stream-churn link-test'; pkill -f '[s]tream-tbcheck.py'; true" >> "$OUT/cleanup.txt" 2>&1
    done
    sleep 1
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under pair2.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

log "output $OUT; $A connects, $B accepts; $OPENS opens per direction and $CHURN churn sessions per setting"
for h in $A $B; do
    on "$h" "uname -r; md5sum < ~/$BIN | cut -c1-12; /usr/local/sbin/tblink links" > "$OUT/links-$h.txt" 2>&1
done
if ! dry; then
    DEV_A=$(awk -F'\t' -v p="$B" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$A.txt")
    DEV_B=$(awk -F'\t' -v p="$A" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$B.txt")
    ADDR=$B.tb$(awk -F'\t' -v p="$B" 'NR > 3 && $2 == p {print $3}' "$OUT/links-$A.txt")
    log "  $A $(head -1 "$OUT/links-$A.txt") $DEV_A <-> $B $(head -1 "$OUT/links-$B.txt") $DEV_B; build $(sed -n 2p "$OUT/links-$A.txt")"
    case "$DEV_A$DEV_B" in /dev/tbstream*/dev/tbstream*) ;; *) log "STOP: the $A-$B link is not up on two lanes with a stream at both ends"; exit 1 ;; esac
fi

run=200
pos() { # client client-dev server server-dev tag
    run=$((run + 1))
    on "$3" "timeout -k 5 30 python3 ~/$W/stream-tbcheck.py pos $4 server 8 $run" >> "$OUT/pos-$5.jsonl" 2>> "$OUT/pos-$5.err" &
    local ps=$!
    sleep 0.5
    on "$1" "timeout -k 5 30 python3 ~/$W/stream-tbcheck.py pos $2 client 8 $run" >> "$OUT/pos-$5.jsonl" 2>> "$OUT/pos-$5.err"
    wait $ps
}
pp() { # client client-dev server server-dev tag: lone frames both ways (the echo is not followed by anything)
    run=$((run + 1))
    on "$3" "timeout -k 5 30 python3 ~/$W/stream-tbcheck.py pp $4 server 10 112 0 $run" >> "$OUT/pp-$5.jsonl" 2>> "$OUT/pp-$5.err" &
    local ps=$!
    sleep 0.5
    on "$1" "timeout -k 5 30 python3 ~/$W/stream-tbcheck.py pp $2 client 10 112 0 $run" >> "$OUT/pp-$5.jsonl" 2>> "$OUT/pp-$5.err"
    wait $ps
}

n=0
IFS='|' read -r -a settings <<< "$SETTINGS"
for setting in "${settings[@]}"; do
    n=$((n + 1))
    tag=s$n
    knobs="$DEFAULTS"
    [ "$setting" = defaults ] || knobs="$DEFAULTS $setting"
    log "$tag. $setting"
    for probe in ${PROBES:-pp pos}; do
        for i in $(seq "$OPENS"); do
            $probe "$A" "$DEV_A" "$B" "$DEV_B" "$tag"
            $probe "$B" "$DEV_B" "$A" "$DEV_A" "$tag"
        done
        dry || log "  $probe: $(python3 - "$OUT/$probe-$tag.jsonl" "$probe" <<'EOF'
import json, sys
by = {}
for l in open(sys.argv[1]):
    if not l.startswith('{'):
        continue
    v = json.loads(l)
    if v.get('side') != 'client':
        continue
    if sys.argv[2] == 'pos':
        by.setdefault(v['host'], []).append(sorted(set(r[2] for r in v.get('raw', []))) or ['none'])
    else:
        by.setdefault(v['host'], []).append(v.get('fillers', {}).get('p50', 'none'))
for h, opens in by.items():
    what = 'fillers into the other end' if sys.argv[2] == 'pos' else 'fillers per round trip'
    print(f"client {h}: {what}, per open {opens};", end=' ')
EOF
)"
    done
    if [ "$CHURN" -gt 0 ] && [ "${setting#*rx_eof_close=0}" = "$setting" ]; then
        m=$(on "$B" "cat ~/$W/stream-cc-serve.log 2>/dev/null | wc -l")
        on "$B" "rm -f ~/$STOP; pgrep -f '^[^ ]*/trashcan-stream-churn link-test serve --listen 0.0.0.0:$PORT' > /dev/null || { setsid nohup bash -c 'while [ ! -f ~/$STOP ]; do ~/$BIN link-test serve --listen 0.0.0.0:$PORT; sleep 0.05; done' >> ~/$W/stream-cc-serve.log 2>&1 < /dev/null & }"
        on "$A" "timeout -k 10 900 ~/$BIN link-test churn --connect $ADDR:$PORT --sessions $CHURN --label $tag" > "$OUT/churn-$tag.jsonl" 2> "$OUT/churn-$tag.err"
        on "$B" "touch ~/$STOP; pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; sleep 0.5; tail -n +$((${m:-0} + 1)) ~/$W/stream-cc-serve.log" > "$OUT/serve-$tag.log" 2>&1
        dry || log "  churn: $(tail -1 "$OUT/churn-$tag.jsonl" | cut -c1-200); server: $(grep -c ' not opened (' "$OUT/serve-$tag.log") failed epochs"
    fi
done
log "done"
echo "stream-probe done: $OUT"
