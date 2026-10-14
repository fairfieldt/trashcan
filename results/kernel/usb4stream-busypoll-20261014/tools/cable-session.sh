#!/bin/bash
# The USB4STREAM CLOSE fix (kernel/patches-7.3/0006) on one cable, A <-> B, under leases on both Macs (pair2.sh), CPU
# only; no module reloads or Thunderbolt resets:
#   before (7.3.0-rc7-p2p, the running kernel): the stale-CLOSE probe each way (closecheck.py) and 20 link-test churn
#     sessions (A's client processes, B's server restarted after each crash);
#   (BEFORE=0 skips that) install NEW (7.3.0-rc7-p2p-s) from ~/DEBS_DIR (kernel/7.3/debs-s) on both, for the next boot (kernel-boot.sh
#     once); then wait for $OUT/go-reboot (the coordinator is told first), reboot both, wait for them on the new
#     kernel, then wait for $OUT/links-up (the coordinator resets Thunderbolt);
#   after: the probe each way, 200 churn sessions (pass: every stream up within 100 ms, no failed epoch) and the
#     link-test suite.
# AFTER=0 ends once both Macs are up on NEW (no links-up wait, no after phase). GATES=auto skips the two waits
# (simtest only). DRY=1 prints the steps.
# usage: pair2.sh stream-cc 45 OWNER A B -- results/kernel/usb4stream-busypoll-20261014/tools/cable-session.sh A B
set -uo pipefail
A=$1 B=$2
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/cc-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
BIN=$W/trashcan-stream-churn
CC=$W/stream-closecheck.py
KB=trashcan-exp/kernel73/kernel-boot.sh
NEW=${NEW:-7.3.0-rc7-p2p-s}
DEBS_DIR=${DEBS_DIR:-kernel/7.3/debs-s}
PORT=7075
STOP=$W/stream-cc.stop
DEV_A=/dev/tbstreamA DEV_B=/dev/tbstreamB ADDR=$B.tbN    # from the live links (state)
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

cleanup() {
    for h in $A $B; do
        ssh -o ConnectTimeout=10 "$h" "touch ~/$STOP; pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; pkill -f '[t]rashcan-stream-churn link-test'; pkill -f '[s]tream-closecheck.py'; true" >> "$OUT/cleanup.txt" 2>&1
    done
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under pair2.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

state() { # tag: kernels, the A-B link, devices and B's address as A resolves it (links-TAG-HOST.txt)
    local h
    for h in $A $B; do
        on "$h" "uname -r; dpkg-query -W -f '\${Version}\n' linux-image-\$(uname -r); /usr/local/sbin/tblink links" > "$OUT/links-$1-$h.txt" 2>&1
    done
    dry && return 0
    log "  kernels: $A $(head -2 "$OUT/links-$1-$A.txt" | tr '\n' ' '), $B $(head -2 "$OUT/links-$1-$B.txt" | tr '\n' ' ')"
    DEV_A=$(awk -F'\t' -v p="$B" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$1-$A.txt")
    DEV_B=$(awk -F'\t' -v p="$A" 'NR > 3 && $2 == p && $6 == "up" && $5 == 2 {print $10}' "$OUT/links-$1-$B.txt")
    ADDR=$B.tb$(awk -F'\t' -v p="$B" 'NR > 3 && $2 == p {print $3}' "$OUT/links-$1-$A.txt")
    log "  $A $DEV_A <-> $B $DEV_B, $A connects to $ADDR"
    case "$DEV_A$DEV_B" in /dev/tbstream*/dev/tbstream*) ;; *) log "STOP: the $A-$B link is not up on two lanes with a stream at both ends"; return 1 ;; esac
}

probe() { # tag: a CLOSE planted at each end in turn, then read there (closecheck.py)
    on "$A" "timeout -k 5 30 python3 ~/$CC plant $DEV_A 8" >> "$OUT/close-$1.jsonl" 2>> "$OUT/close-$1.err"
    on "$B" "timeout -k 5 30 python3 ~/$CC check $DEV_B 1.0" >> "$OUT/close-$1.jsonl" 2>> "$OUT/close-$1.err"
    on "$B" "timeout -k 5 30 python3 ~/$CC plant $DEV_B 8" >> "$OUT/close-$1.jsonl" 2>> "$OUT/close-$1.err"
    on "$A" "timeout -k 5 30 python3 ~/$CC check $DEV_A 1.0" >> "$OUT/close-$1.jsonl" 2>> "$OUT/close-$1.err"
    dry || log "  $(python3 -c '
import json, sys
for l in open(sys.argv[1]):
    v = json.loads(l)
    if v["step"] == "check":
        print("%s: %d ends of file, write %s, %d after it%s;" % (v["host"], v["eofs"], v["write"], v["eofs_after"], " (every read)" if v["sticky"] else ""), end=" ")
' "$OUT/close-$1.jsonl")"
}

server() { # start|stop: B's link-test server, restarted after each crash
    if [ "$1" = start ]; then
        # Only the loop goes to the background, its output in the log (backgrounding the whole `pgrep || ...` list
        # kept ssh waiting on the list's output: stream-cc, 14:10).
        on "$B" "rm -f ~/$STOP; pgrep -f '^[^ ]*/trashcan-stream-churn link-test serve --listen 0.0.0.0:$PORT' > /dev/null || { setsid nohup bash -c 'while [ ! -f ~/$STOP ]; do ~/$BIN link-test serve --listen 0.0.0.0:$PORT; sleep 0.05; done' >> ~/$W/stream-cc-serve.log 2>&1 < /dev/null & }"
    else
        on "$B" "touch ~/$STOP; pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; sleep 0.5; true"
    fi
}

churn() { # tag sessions
    local n rc
    n=$(on "$B" "cat ~/$W/stream-cc-serve.log 2>/dev/null | wc -l")
    server start
    on "$A" "timeout -k 10 1800 ~/$BIN link-test churn --connect $ADDR:$PORT --sessions $2 --label cc-$1" > "$OUT/churn-$1.jsonl" 2> "$OUT/churn-$1.err"
    rc=$?
    on "$B" "tail -n +$((${n:-0} + 1)) ~/$W/stream-cc-serve.log" > "$OUT/serve-$1.log" 2>&1
    dry || log "  churn rc $rc: $(tail -1 "$OUT/churn-$1.jsonl"); server: $(grep -c ' not opened (' "$OUT/serve-$1.log") failed epochs, $(grep -c 'exiting at' "$OUT/serve-$1.log") exits"
}

suite() { # tag
    local m
    for m in "--mode pingpong --size 64 --count 5000" "--mode pingpong --size 20480 --count 2000" \
        "--mode exchange --size 40960 --count 2000" "--mode bulk --size 5242880 --count 20" "--mode chain --count 1000"; do
        on "$A" "timeout -k 10 300 ~/$BIN link-test run --connect $ADDR:$PORT --label lt-$1 $m" >> "$OUT/suite-$1.jsonl" 2>> "$OUT/suite-$1.err"
    done
    dry || log "  suite: $(grep -c '"result"' "$OUT/suite-$1.jsonl") of 5 tests passed, $(grep -c ' not opened (' "$OUT/suite-$1.err") failed epochs, $(grep -c 'stream down' "$OUT/suite-$1.err") fallbacks"
}

gate() { # name minutes: wait for $OUT/NAME (GATES=auto: at once)
    log "waiting for $OUT/$1 (up to $2 min)"
    if [ "${GATES:-wait}" = auto ] || dry; then touch "$OUT/$1"; fi
    local t=0
    while [ ! -f "$OUT/$1" ]; do
        [ $t -ge $(($2 * 60)) ] && { log "STOP: no $1 after $2 min"; return 1; }
        sleep 5
        t=$((t + 5))
    done
    log "  $1"
}

log "output $OUT; $A connects, $B accepts"
log "1. before: the running kernel"
state before || exit 1
if [ "${BEFORE:-1}" = 1 ]; then
    probe before
    churn before 20
    server stop
fi

log "2. install $NEW on both, for the next boot only"
for h in $A $B; do
    on "$h" "DEBS=~/$DEBS_DIR nice -n 10 ~/$KB once $NEW && sudo -n grub-editenv list" > "$OUT/install-$h.txt" 2>&1 || { log "STOP: $h: install failed: $(tail -3 "$OUT/install-$h.txt" | tr '\n' ' ')"; exit 1; }
    dry || log "  $h: $(tail -2 "$OUT/install-$h.txt" | tr '\n' ' ')"
done
touch "$OUT/ready-to-reboot"
gate go-reboot 45 || exit 1

log "3. reboot $A and $B"
for h in $A $B; do on "$h" "sudo -n systemctl reboot" > /dev/null 2>&1; done
first=${REBOOT_WAIT:-20}
dry || sleep "$first"
for h in $A $B; do
    t=0
    until dry || [ "$(on "$h" 'uname -r' 2> /dev/null)" = "$NEW" ]; do
        [ $t -ge 600 ] && { log "STOP: $h not back on $NEW after 10 min"; exit 1; }
        sleep 5
        t=$((t + 5))
    done
    dry || log "  $h up on $NEW after $((t + first)) s"
done
if [ "${AFTER:-1}" = 0 ]; then
    log "done (both on $NEW; the links need a Thunderbolt reset)"
    echo "stream-cc done: $OUT"
    exit 0
fi
touch "$OUT/waiting-for-links"
gate links-up 60 || exit 1

log "4. after: $NEW"
state after || exit 1
probe after
churn after 200
suite after
server stop
log "done"
echo "stream-cc done: $OUT"
