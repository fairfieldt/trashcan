#!/bin/bash
# USB4STREAM on the inflight-alight cable, both ends on the running kernel (no install or reboot): the stale-CLOSE probe
# each way (closecheck.py), N churn sessions of the deployed engine's link (pass: every stream up within 100 ms, no
# failed epoch) and the link-test suite. Steps and pass rules as results/kernel/usb4stream-busypoll-20261014/tools/
# cable-session.sh. CPU only; run from the Mac under gpuq leases on both.
# usage: session.sh OUT [SESSIONS]      (A=inflight connects, B=alight accepts; BUSY=1 sets busy_poll on both devices)
set -uo pipefail
OUT=$1 N=${2:-200}
A=inflight B=alight
BIN=trashcan-exp/a16/trashcan
CC=trashcan-exp/flashnext/stream-closecheck.py
PORT=7075 STOP=trashcan-exp/kernel74/serve.stop SLOG=trashcan-exp/kernel74/serve.log
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }
cleanup() { for h in $A $B; do on $h "touch ~/$STOP; pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; pkill -f '[l]ink-test churn'; true" >/dev/null 2>&1; done; }
trap cleanup EXIT
for h in $A $B; do on $h "uname -r; /usr/local/sbin/tblink links" > "$OUT/links-$h.txt" 2>&1; done
DEV_A=$(awk -F'\t' -v p=$B 'NR > 1 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$A.txt")
DEV_B=$(awk -F'\t' -v p=$A 'NR > 1 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$B.txt")
ADDR=$B.tb$(awk -F'\t' -v p=$B 'NR > 1 && $2 == p {print $3}' "$OUT/links-$A.txt")
log "kernels $(head -1 "$OUT/links-$A.txt") / $(head -1 "$OUT/links-$B.txt"); $A $DEV_A <-> $B $DEV_B, connect $ADDR"
case "$DEV_A$DEV_B" in /dev/tbstream*/dev/tbstream*) ;; *) log "STOP: link not up on two lanes with a stream"; exit 1 ;; esac
bp=${BUSY:-0}
for h in $A $B; do on $h "for f in /sys/kernel/config/thunderbolt/stream/*/trashcan/busy_poll; do echo $bp | sudo -n tee \$f >/dev/null; echo \$f=\$(cat \$f); done" >> "$OUT/busy_poll.txt" 2>&1; done
log "busy_poll $bp: $(tr '\n' ' ' < "$OUT/busy_poll.txt")"

log "probe: a CLOSE planted at each end, then read there"
on $A "timeout -k 5 30 python3 ~/$CC plant $DEV_A 8" >> "$OUT/close.jsonl" 2>> "$OUT/close.err"
on $B "timeout -k 5 30 python3 ~/$CC check $DEV_B 1.0" >> "$OUT/close.jsonl" 2>> "$OUT/close.err"
on $B "timeout -k 5 30 python3 ~/$CC plant $DEV_B 8" >> "$OUT/close.jsonl" 2>> "$OUT/close.err"
on $A "timeout -k 5 30 python3 ~/$CC check $DEV_A 1.0" >> "$OUT/close.jsonl" 2>> "$OUT/close.err"
log "  $(python3 -c '
import json, sys
for l in open(sys.argv[1]):
    v = json.loads(l)
    if v["step"] == "check":
        print("%s: %d ends of file, write %s, %d after it%s;" % (v["host"], v["eofs"], v["write"], v["eofs_after"], " (every read)" if v["sticky"] else ""), end=" ")
' "$OUT/close.jsonl")"

log "churn: $N sessions"
on $B "rm -f ~/$STOP; : > ~/$SLOG; setsid nohup bash -c 'while [ ! -f ~/$STOP ]; do ~/$BIN link-test serve --listen 0.0.0.0:$PORT; sleep 0.05; done' >> ~/$SLOG 2>&1 < /dev/null &"
sleep 1
on $A "timeout -k 10 2400 ~/$BIN link-test churn --connect $ADDR:$PORT --sessions $N --label c74" > "$OUT/churn.jsonl" 2> "$OUT/churn.err"
rc=$?
log "  churn rc $rc: $(tail -1 "$OUT/churn.jsonl")"
log "suite"
for m in "--mode pingpong --size 64 --count 5000" "--mode pingpong --size 20480 --count 2000" \
    "--mode exchange --size 40960 --count 2000" "--mode bulk --size 5242880 --count 20" "--mode chain --count 1000"; do
    on $A "timeout -k 10 300 ~/$BIN link-test run --connect $ADDR:$PORT --label s74 $m" >> "$OUT/suite.jsonl" 2>> "$OUT/suite.err"
done
on $B "cat ~/$SLOG" > "$OUT/serve.log" 2>&1
log "  suite: $(grep -c '"result"' "$OUT/suite.jsonl") of 5 passed; server: $(grep -c ' not opened (' "$OUT/serve.log") failed epochs, $(grep -c 'exiting at' "$OUT/serve.log") exits; client: $(grep -c ' not opened (' "$OUT/churn.err" "$OUT/suite.err" | awk -F: '{s+=$2} END {print s}') failed epochs, $(grep -c 'stream down' "$OUT/suite.err") fallbacks"
for h in $A $B; do on $h "sudo -n dmesg -T | tail -40" > "$OUT/dmesg-$h.txt" 2>&1; done
