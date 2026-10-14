#!/bin/bash
# One cable, A <-> B, under leases on both Macs (pair2.sh), CPU only; no reloads, resets or reboots: does
# thunderbolt_net hold lone frames as the stream does on the same controllers? Single UDP packets there and back over
# the cable's Thunderbolt IP link (udpping.py: 2000 x 64 B, B's server, then A's), then the stream's raw lone-frame
# probe (tbcheck.py pp, 50 round trips, no follow-up frames: fillers show the hold), A's client, then B's.
# DRY=1 prints the steps.
# usage: pair2.sh stream-hold 10 OWNER A B -- bash results/kernel/usb4stream-busypoll-20261014/tools/holdcheck-session.sh A B
set -uo pipefail
A=$1 B=$2
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/hold-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
PORT=7090
DEV_A=/dev/tbstreamA DEV_B=/dev/tbstreamB IP_A=10.0.0.A IP_B=10.0.0.B
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

cleanup() {
    for h in $A $B; do ssh -o ConnectTimeout=10 "$h" "pkill -f '[s]tream-udpping.py'; pkill -f '[s]tream-tbcheck.py'; true" >> "$OUT/cleanup.txt" 2>&1; done
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under pair2.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

log "output $OUT"
for h in $A $B; do
    on "$h" "uname -r; cat /sys/module/thunderbolt_net/parameters/e2e /sys/module/thunderbolt_net/parameters/throttling 2>/dev/null | tr '\n' ' '; echo; /usr/local/sbin/tblink links" > "$OUT/links-$h.txt" 2>&1
done
if ! dry; then
    row() { awk -F'\t' -v p="$2" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2' "$OUT/links-$1.txt"; }
    DEV_A=$(row "$A" "$B" | cut -f10) IP_A=$(row "$A" "$B" | cut -f8)
    DEV_B=$(row "$B" "$A" | cut -f10) IP_B=$(row "$B" "$A" | cut -f8)
    log "  $A $(head -1 "$OUT/links-$A.txt") net e2e/throttling $(sed -n 2p "$OUT/links-$A.txt") $IP_A $DEV_A"
    log "  $B $(head -1 "$OUT/links-$B.txt") net e2e/throttling $(sed -n 2p "$OUT/links-$B.txt") $IP_B $DEV_B"
    case "$DEV_A$DEV_B" in /dev/tbstream*/dev/tbstream*) ;; *) log "STOP: the $A-$B link is not up on two lanes with a stream at both ends"; exit 1 ;; esac
fi

udp() { # client server server-ip
    on "$2" "timeout 60 python3 ~/$W/stream-udpping.py server $3 $PORT 25" >> "$OUT/udp.jsonl" 2>> "$OUT/udp.err" &
    local ps=$!
    sleep 1
    on "$1" "timeout 60 python3 ~/$W/stream-udpping.py client $3 $PORT 2000 64" >> "$OUT/udp.jsonl" 2>> "$OUT/udp.err"
    wait $ps
}
log "1. UDP over the Thunderbolt IP link, single 64-byte packets"
udp "$A" "$B" "$IP_B"
udp "$B" "$A" "$IP_A"
dry || log "  $(python3 -c '
import json, sys
for l in open(sys.argv[1]):
    v = json.loads(l)
    if v["role"] == "client":
        r = v["rtt_us"]
        print("%s -> %s: %d round trips, p50 %s us, p90 %s, p99 %s, max %s, %d timed out;" % (v["host"], v["addr"], v["n"], r["p50"], r["p90"], r["p99"], r["max"], v["lost"]), end=" ")
' "$OUT/udp.jsonl")"

pp() { # client client-dev server server-dev run
    on "$3" "timeout -k 5 60 python3 ~/$W/stream-tbcheck.py pp $4 server 50 112 0 $5" >> "$OUT/pp.jsonl" 2>> "$OUT/pp.err" &
    local ps=$!
    sleep 0.5
    on "$1" "timeout -k 5 60 python3 ~/$W/stream-tbcheck.py pp $2 client 50 112 0 $5" >> "$OUT/pp.jsonl" 2>> "$OUT/pp.err"
    wait $ps
}
log "2. the stream, single 112-byte frames, no follow-up frames"
pp "$A" "$DEV_A" "$B" "$DEV_B" 61
pp "$B" "$DEV_B" "$A" "$DEV_A" 62
dry || log "  $(python3 -c '
import json, sys
for l in open(sys.argv[1]):
    v = json.loads(l)
    if v.get("side") == "client":
        print("%s client: %d round trips, p50 %.0f us, fillers p50 %s;" % (v["host"], v["rounds"], v["us"].get("p50", 0), v["fillers"].get("p50")), end=" ")
' "$OUT/pp.jsonl")"
log "done"
echo "stream-hold done: $OUT"
