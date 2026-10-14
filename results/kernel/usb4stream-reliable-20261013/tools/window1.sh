#!/bin/bash
# Window 1 on all six Mac Pros (about 30 min, run under chain6.sh): USB4STREAM driver checks; `trashcan link-test`
# over TCP and then over the streams on all six ring links at once; a 10-minute soak of all links; one pass of each
# fault (peer killed, stream given up, peer stopped, damaged frames, ring overrun, two processes on a cable, delayed
# start). The streams come from one pass of the new tblink (stream-ops.sh up) on the Macs concerned, as tblink.service
# will make them. Leaves every Mac as found.
# DRY=1 prints the steps and checks the files on every Mac, touching nothing else.
# usage: chain6.sh stream-w1 35 OWNER -- results/kernel/usb4stream-reliable-20261013/tools/window1.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/w1-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
BIN=$W/trashcan-stream
OPS=$W/stream-ops.sh
PORT=7075
MACS="incite indite inflight alight invite enlight"
mkdir -p "$OUT"

# Cable i of the ring joins Mac i and Mac i+1; a Mac's client goes to the next Mac over this address.
next_addr() {
    case $1 in incite) echo indite.tb0 ;; indite) echo inflight.tb1 ;; inflight) echo alight.tb0 ;;
        alight) echo invite.tb1 ;; invite) echo enlight.tb0 ;; enlight) echo incite.tb1 ;; esac
}
next_mac() { next_addr "$1" | cut -d. -f1; }
prev_mac() {
    case $1 in incite) echo enlight ;; indite) echo incite ;; inflight) echo indite ;; alight) echo inflight ;;
        invite) echo alight ;; enlight) echo invite ;; esac
}

dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { # host command (remote shell text)
    local h=$1
    shift
    if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi
    ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"
}
now() { python3 -c 'import time; print(f"{time.time():.3f}")'; }

# gpuq leases stay alive only with calls: one status per ticket every two minutes. However the script ends, no
# server and no stream stays behind.
cleaned=0
cleanup() {
    [ "$cleaned" = 1 ] && return
    cleaned=1
    local h
    for h in $MACS; do
        ssh -o ConnectTimeout=10 "$h" "pkill -CONT -f '[l]ink-test serve'; pkill -f 'trashcan-strea[m] link-test'; ~/$OPS down" >> "$OUT/cleanup.txt" 2>&1
        # The stream module as found: unloaded again if it was not loaded before the window.
        if grep -q '^thunderbolt ' "$OUT/state-before-$h.txt" 2> /dev/null && ! grep -q '^thunderbolt_stream ' "$OUT/state-before-$h.txt"; then
            ssh -o ConnectTimeout=10 "$h" "sudo modprobe -r thunderbolt_stream && echo '$h: thunderbolt_stream unloaded as found'" >> "$OUT/cleanup.txt" 2>&1
        fi
    done
}
if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under chain6.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    hb=$!
    trap 'kill $hb 2>/dev/null; cleanup' EXIT
    trap 'exit 130' INT TERM
fi

servers() { # start|stop hosts...
    local what=$1 h
    shift
    for h in "$@"; do
        if [ "$what" = start ]; then
            on "$h" "pgrep -f '^[^ ]*/trashcan-stream link-test serve --listen 0.0.0.0:$PORT' > /dev/null || { setsid nohup ~/$BIN link-test serve --listen 0.0.0.0:$PORT >> ~/$W/stream-serve.log 2>&1 < /dev/null & }"
        else
            on "$h" "pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; sleep 0.5; true"
        fi
    done
}

client() { # host tag args...: the host's client to the next Mac; one JSON line per test in TAG-HOST.jsonl. A client
    # that hangs (a peer alive but stuck) ends after 3 minutes (12 for the soak), so the window goes on.
    local h=$1 tag=$2 tmo=180
    shift 2
    case " $* " in *" soak "*) tmo=720 ;; esac
    on "$h" "timeout -k 10 $tmo ~/$BIN link-test run --connect $(next_addr "$h"):$PORT --label $tag-$h $*" >> "$OUT/$tag-$h.jsonl" 2>> "$OUT/$tag-$h.err"
}

all_clients() { # tag args...: every Mac's client at once
    local pids="" h
    for h in $MACS; do client "$h" "$@" & pids="$pids $!"; done
    wait $pids
}

suite() { # tag
    all_clients "$1" --mode pingpong --size 64 --count 5000
    all_clients "$1" --mode pingpong --size 20480 --count 2000
    all_clients "$1" --mode exchange --size 40960 --count 2000
    all_clients "$1" --mode bulk --size 5242880 --count 20
    all_clients "$1" --mode chain --count 1000
}

streams() { # up|down hosts...: `up` repeats the pass (up to 30 s) until every working link of the hosts has one,
    # since a neighbour sees a reloaded stream module's service a moment later
    local what=$1 h t missing
    shift
    for t in 0 3 6 9 12 15 18 21 24 27 30; do
        missing=0
        for h in "$@"; do
            echo "== $h $(date +%T)" >> "$OUT/streams-$what.txt"
            on "$h" "~/$OPS $what" > "$OUT/streams-last-$h.txt" 2>&1
            cat "$OUT/streams-last-$h.txt" >> "$OUT/streams-$what.txt"
            [ "$what" = up ] && awk -F'\t' '$2 == "up" && $3 == "-"' "$OUT/streams-last-$h.txt" | grep -q . && missing=1
        done
        [ "$missing" = 0 ] && return 0
        dry && return 0
        sleep 3
    done
    log "  streams still missing on $*: $(cat "$OUT"/streams-last-*.txt | tr '\n' ' ')"
    return 1
}

state() { # tag
    local h
    for h in $MACS; do on "$h" "~/$OPS state" > "$OUT/state-$1-$h.txt" 2>&1; done
}

links_ok() { # every Mac's links up on two lanes
    local h bad=0
    for h in $MACS; do
        on "$h" /usr/local/sbin/tblink links > "$OUT/links-$h.txt" 2>&1
        dry && continue
        awk -F'\t' 'NR > 1 && ($6 != "up" || $5 != 2) { exit 1 }' "$OUT/links-$h.txt" || { log "$h: a link is not up on two lanes"; bad=1; }
        [ "$(awk -F'\t' 'NR > 1 && $6 == "up"' "$OUT/links-$h.txt" | wc -l)" -ge 2 ] || { log "$h: fewer than two links up"; bad=1; }
    done
    return $bad
}

wait_links() { # seconds: until every link is up on two lanes
    local t=0
    while [ "$t" -lt "$1" ]; do
        links_ok && return 0
        dry && return 0
        sleep 5
        t=$((t + 5))
    done
    return 1
}

log "output $OUT"
log "preflight: files and sudo on every Mac"
for h in $MACS; do
    ssh -o ConnectTimeout=10 "$h" "test -x ~/$BIN && test -x ~/$OPS && test -x ~/$W/stream-tblink && test -f ~/$W/stream-tbcheck.py && sudo -n true && ~/$BIN link-test crc --iters 2000" > "$OUT/preflight-$h.txt" 2>&1 \
        || { log "$h: preflight failed: $(cat "$OUT/preflight-$h.txt")"; exit 1; }
done
log "preflight ok: $(cat "$OUT"/preflight-*.txt | python3 -c 'import sys,json; print(sorted(round(json.loads(l)["gb_s"],1) for l in sys.stdin))') GB/s CRC32C"
state before
links_ok || { log "links not all up before the window: stop"; exit 1; }
for h in $MACS; do
    if ! dry && grep -qE '^/sys/kernel/config/thunderbolt/stream/[^/]+/trashcan$' "$OUT/state-before-$h.txt"; then
        log "$h: a stream exists already: stop"; exit 1
    fi
done

log "1. link-test over TCP, every link at once (no streams yet)"
for h in $MACS; do on "$h" ": > ~/$W/stream-serve.log"; done
servers start $MACS
sleep 1
suite tcp

log "2. streams: one pass of the new tblink on every Mac"
streams up $MACS
cat "$OUT/streams-up.txt" | tee -a "$OUT/steps.log"

log "3. driver checks on incite-indite"
A=incite B=indite
DA=$(on "$A" "~/$OPS dev $B") DB=$(on "$B" "~/$OPS dev $A")
log "  devices: $A $DA, $B $DB"
on "$A" "python3 ~/$W/stream-tbcheck.py opentime $DA" >> "$OUT/driver.jsonl" 2>&1
# Frames written while the other end is closed; a CLOSE sent to a closed end.
on "$A" "python3 ~/$W/stream-tbcheck.py writer $DA 300 4" >> "$OUT/driver.jsonl" 2>&1 &
p=$!
on "$B" "python3 ~/$W/stream-tbcheck.py reader $DB 1.5 1.5; python3 ~/$W/stream-tbcheck.py reader $DB 1.5 1" >> "$OUT/driver.jsonl" 2>&1
wait $p
# The receiving end does not read: 1000 frames into a 256-frame ring.
on "$B" "python3 ~/$W/stream-tbcheck.py lazy $DB 3 1" >> "$OUT/driver.jsonl" 2>&1 &
p=$!
sleep 1
on "$A" "python3 ~/$W/stream-tbcheck.py blast $DA 1000" >> "$OUT/driver.jsonl" 2>&1
wait $p
for h in $A $B; do on "$h" "sudo dmesg | grep -i tbstream | tail -20" > "$OUT/dmesg-driver-$h.txt" 2>&1; done
cat "$OUT/driver.jsonl" >> "$OUT/steps.log"

log "4. link-test over the streams, every link at once"
all_clients streamcheck --mode pingpong --size 64 --count 200
if ! dry; then
    onstream=$(cat "$OUT"/streamcheck-*.jsonl 2>/dev/null | python3 -c 'import sys, json
n = 0
for l in sys.stdin:
    d = json.loads(l)
    n += all(x["path"] == "stream" for x in d.get("result", {}).get("lanes", [{"path": "-"}]))
print(n)')
    log "  $onstream of 6 links on the stream"
    if [ "$onstream" != 6 ]; then
        log "not every link uses its stream: no soak or faults this window"
        cat "$OUT"/streamcheck-*.err >> "$OUT/steps.log"
        servers stop $MACS
        streams down $MACS
        state after
        exit 1
    fi
fi
suite stream

log "5. soak: every link, 10 minutes"
all_clients soak --mode soak --seconds 600

log "6. faults"
servers start $MACS
log "6a peer killed mid-session (incite -> indite), then a new session"
client incite f-kill --mode chain --count 200000 &
p=$!
sleep 3
on indite "date +%s.%N; pkill -9 -f '[l]ink-test serve --listen 0.0.0.0:$PORT'" > "$OUT/f-kill-inject.txt" 2>&1
wait $p
servers start indite
sleep 1
client incite f-kill-after --mode chain --count 2000

log "6b the server gives its stream up 2 s into a session (indite -> inflight)"
client indite f-giveup --mode chain --count 20000 --server-drops-stream 2

log "6c the server stopped for 2 s mid-session (inflight -> alight)"
client inflight f-stop --mode chain --count 30000 &
p=$!
sleep 3
on alight "date +%s.%N; pkill -STOP -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; sleep 2; pkill -CONT -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; date +%s.%N" > "$OUT/f-stop-inject.txt" 2>&1
wait $p

log "6d damaged and lost frames (alight -> invite): every 97th frame lost, every 101st damaged"
client alight f-chaos --mode chain --count 5000 --drop 97 --corrupt 101
client alight f-chaos --mode bulk --size 5242880 --count 20 --drop 97 --corrupt 101

log "6e ring overrun: 300 frames past the credits (invite -> enlight)"
client invite f-overdrive --mode bulk --size 5242880 --count 20 --overdrive 300

log "6f two processes on one cable (enlight -> incite)"
client enlight f-two --mode chain --count 3000 &
p=$!
client enlight f-two --mode chain --count 3000
wait $p

log "6g delayed start: the client 8 s before the server (incite -> indite)"
servers stop indite
client incite f-delayed --mode chain --count 1000 &
p=$!
sleep 8
servers start indite
wait $p

# No module reloads, XDomain reconnects (tblink reload) or controller resets on the shared Macs: in the window of
# 2026-10-13 a tblink reload on inflight left its links down, two resets of enlight and incite did not bring them back,
# and the end-of-window reset of all six ended in a kernel Oops on inflight (README).

log "7. cleanup"
servers stop $MACS
for h in $MACS; do on "$h" "cat ~/$W/stream-serve.log" > "$OUT/serve-$h.log" 2>&1; done
streams down $MACS
for h in $MACS; do on "$h" "sudo dmesg | grep -iE 'tbstream|thunderbolt' | tail -60" > "$OUT/dmesg-end-$h.txt" 2>&1; done
state after
wait_links 30 || log "LINKS NOT ALL UP: tell the coordinator (no reset from here)"
log "done; $(grep -l '"error"' "$OUT"/*.jsonl 2>/dev/null | wc -l | tr -d ' ') result files with errors"
echo "stream-w1 done: $OUT"
