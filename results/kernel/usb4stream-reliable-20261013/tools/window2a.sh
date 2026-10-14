#!/bin/bash
# Window 2 part A on all six Mac Pros (about 25 min under chain6.sh; CPU only, no module reloads or resets): the six
# ring links from the live `tblink links` (ring6.py); link-test over TCP; the streams from one pass of this branch's
# tblink; the raw ping-pong probe on every link with the streams' interrupt throttling at 8192 ns (the driver's
# default) and at 0; then the fix the probe picks: throttling 0 with the build that writes nothing extra if that
# releases lone frames on every link, else throttling 8192 and the build whose stream writes end with one more frame;
# with it, link-test over the streams, a 10-minute soak and the faults without reloads (peer killed, stream given up,
# peer stopped, damaged and lost frames, ring overrun, two processes on a cable, delayed start). Leaves no stream, no
# server and the throttling as found; links that are not up are reported, never reset.
# DRY=1 prints the steps and checks the files on every Mac, touching nothing else.
# usage: chain6.sh stream-w2a 25 OWNER -- results/kernel/usb4stream-reliable-20261013/tools/window2a.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/w2a-$(date +%Y%m%d-%H%M%S)}
W=trashcan-exp/flashnext
BINP=trashcan-stream-push
BINN=trashcan-stream-nopush
BIN=$BINP
OPS=$W/stream-ops.sh
PORT=7075
MACS="incite indite inflight alight invite enlight"
CFG=/sys/kernel/config/thunderbolt/stream
mkdir -p "$OUT"

dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { # host command (remote shell text)
    local h=$1
    shift
    if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi
    ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"
}

# The ring from the live links (ring.txt: MAC NEXT ADDR, ADDR the next Mac's name as MAC resolves it); R is the order.
python3 "$here/ring6.py" > "$OUT/ring.txt" || { echo "the six links do not make a ring: $(cat "$OUT/ring.txt")" | tee -a "$OUT/steps.log"; exit 1; }
R=($(awk '{print $1}' "$OUT/ring.txt"))
next_of() { awk -v h="$1" '$1 == h {print $2}' "$OUT/ring.txt"; }
addr_of() { awk -v h="$1" '$1 == h {print $3}' "$OUT/ring.txt"; }
dev_of() { awk -v h="$1" -v p="$2" '$1 == h && $2 == p {print $3}' "$OUT/devs.txt"; }   # MAC's device to PEER

cleaned=0
cleanup() {
    [ "$cleaned" = 1 ] && return
    cleaned=1
    local h
    for h in $MACS; do
        ssh -o ConnectTimeout=10 "$h" "pkill -CONT -f '[l]ink-test serve'; pkill -f 'trashcan-strea[m][-a-z]* link-test'; pkill -f '[s]tream-tbcheck.py'; \
            for d in $CFG/*/trashcan; do [ -d \$d ] && sudo -n rmdir \$d && sudo -n rmdir \$(dirname \$d); done; ls $CFG" >> "$OUT/cleanup.txt" 2>&1
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
            on "$h" "pgrep -f '^[^ ]*/trashcan-stream[-a-z]* link-test serve --listen 0.0.0.0:$PORT' > /dev/null || { setsid nohup ~/$W/$BIN link-test serve --listen 0.0.0.0:$PORT >> ~/$W/stream-serve.log 2>&1 < /dev/null & }"
        else
            on "$h" "pkill -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; sleep 0.5; true"
        fi
    done
}

client() { # host tag args...: the host's client to the next Mac of the ring; one JSON line per test in TAG-HOST.jsonl
    local h=$1 tag=$2 tmo=180
    shift 2
    case " $* " in *" soak "*) tmo=720 ;; esac
    on "$h" "timeout -k 10 $tmo ~/$W/$BIN link-test run --connect $(addr_of "$h"):$PORT --label $tag-$h $*" >> "$OUT/$tag-$h.jsonl" 2>> "$OUT/$tag-$h.err"
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

streams_up() { # one pass of this branch's tblink on every Mac, repeated (up to 30 s) until every up link has a stream
    local h t missing
    for t in 0 3 6 9 12 15 18 21 24 27 30; do
        missing=0
        for h in $MACS; do
            echo "== $h $(date +%T)" >> "$OUT/streams-up.txt"
            on "$h" "~/$OPS up" > "$OUT/streams-last-$h.txt" 2>&1
            cat "$OUT/streams-last-$h.txt" >> "$OUT/streams-up.txt"
            awk -F'\t' '$2 == "up" && $3 == "-"' "$OUT/streams-last-$h.txt" | grep -q . && missing=1
        done
        [ "$missing" = 0 ] && return 0
        dry && return 0
        sleep 3
    done
    log "  streams still missing: $(cat "$OUT"/streams-last-*.txt | tr '\n' ' ')"
    return 1
}

state() { # tag
    local h
    for h in $MACS; do on "$h" "~/$OPS state; grep . $CFG/*/trashcan/throttling 2>/dev/null" > "$OUT/state-$1-$h.txt" 2>&1; done
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

throttle() { # ns: the interrupt throttling of every stream end (devices closed), read back
    local h
    for h in $MACS; do
        on "$h" "for d in $CFG/*/trashcan; do echo $1 | sudo -n tee \$d/throttling > /dev/null; done; grep . $CFG/*/trashcan/throttling" >> "$OUT/throttle-$1.txt" 2>&1
    done
    dry || log "  throttling: $(awk -F: '{print $NF}' "$OUT/throttle-$1.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
}

run=100
probe() { # tag follow rounds: single 112-byte frames there and back on every link at once (tbcheck.py pp)
    local tag=$1 follow=$2 n=$3 h nx pids=""
    run=$((run + 1))
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        on "$nx" "timeout -k 5 60 python3 ~/$W/stream-tbcheck.py pp $(dev_of "$nx" "$h") server $n 112 $follow $run" >> "$OUT/probe-$tag.jsonl" 2>> "$OUT/probe-$tag.err" &
        pids="$pids $!"
    done
    sleep 0.5
    for h in "${R[@]}"; do
        nx=$(next_of "$h")
        on "$h" "timeout -k 5 60 python3 ~/$W/stream-tbcheck.py pp $(dev_of "$h" "$nx") client $n 112 $follow $run" >> "$OUT/probe-$tag.jsonl" 2>> "$OUT/probe-$tag.err" &
        pids="$pids $!"
    done
    wait $pids
}
probe_table() { # tag: one line per link and variant (client side)
    python3 -c '
import json, sys
rows = {}
for l in open(sys.argv[1]):
    try:
        d = json.loads(l)
    except ValueError:
        continue
    if d.get("side") == "client":
        rows[(d["host"], d["follow"])] = d
for (h, f), d in sorted(rows.items()):
    u = d.get("us", {})
    print(f"  {h:9s} follow {f}: {d[\"rounds\"]} rounds, round trip p50 {u.get(\"p50\", 0):9.0f} us p90 {u.get(\"p90\", 0):9.0f} us, fillers p50 {d[\"fillers\"].get(\"p50\")}")
' "$OUT/probe-$1.jsonl"
}
released() { # tag: every link released lone frames by itself (follow 0, round trip p50 under 1 ms on all six)
    python3 -c '
import json, sys
n = ok = 0
for l in open(sys.argv[1]):
    try:
        d = json.loads(l)
    except ValueError:
        continue
    if d.get("side") == "client" and d["follow"] == 0:
        n += 1
        ok += d["rounds"] > 0 and d["us"].get("p50", 1e9) < 1000
sys.exit(0 if n == 6 and ok == 6 else 1)
' "$OUT/probe-$1.jsonl"
}

log "output $OUT"
log "ring: $(awk '{print $1 "->" $3}' "$OUT/ring.txt" | tr '\n' ' ')"
log "preflight: files and sudo on every Mac"
for h in $MACS; do
    ssh -o ConnectTimeout=10 "$h" "test -x ~/$W/$BINP && test -x ~/$W/$BINN && test -x ~/$OPS && test -x ~/$W/stream-tblink && test -f ~/$W/stream-tbcheck.py && sudo -n true && ~/$W/$BINP link-test crc --iters 2000" > "$OUT/preflight-$h.txt" 2>&1 \
        || { log "$h: preflight failed: $(cat "$OUT/preflight-$h.txt")"; exit 1; }
done
log "preflight ok: $(cat "$OUT"/preflight-*.txt | python3 -c 'import sys,json; print(sorted(round(json.loads(l)["gb_s"],1) for l in sys.stdin))') GB/s CRC32C"
state before
links_ok || { log "links not all up before the window: stop"; exit 1; }
for h in $MACS; do
    if ! dry && grep -qE "^$CFG/[^/]+/trashcan$" "$OUT/state-before-$h.txt"; then log "$h: a stream exists already: stop"; exit 1; fi
done

log "1. link-test over TCP, every link at once (no streams yet)"
for h in $MACS; do on "$h" ": > ~/$W/stream-serve.log"; done
servers start $MACS
sleep 1
suite tcp

log "2. streams: one pass of this branch's tblink on every Mac"
servers stop $MACS
streams_up || exit 1
cat "$OUT/streams-up.txt" >> "$OUT/steps.log"
: > "$OUT/devs.txt"
for h in "${R[@]}"; do
    nx=$(next_of "$h")
    echo "$h $nx $(on "$h" "~/$OPS dev $nx")" >> "$OUT/devs.txt"
    echo "$nx $h $(on "$nx" "~/$OPS dev $h")" >> "$OUT/devs.txt"
done
log "  devices: $(tr '\n' ';' < "$OUT/devs.txt")"

log "3. raw ping-pong on every link, throttling 8192 (as created)"
throttle 8192
probe t8192 0 25
probe t8192 1 200
dry || probe_table t8192 | tee -a "$OUT/steps.log"
log "4. throttling 0, raw ping-pong again"
throttle 0
probe t0 0 25
probe t0 1 200
dry || probe_table t0 | tee -a "$OUT/steps.log"
if ! dry && released t0; then
    BIN=$BINN
    log "  throttling 0 releases lone frames on every link: $BIN at throttling 0"
else
    throttle 8192
    BIN=$BINP
    log "  throttling 0 does not release them on every link: $BIN (one more frame after each write) at throttling 8192"
fi
echo "$BIN" > "$OUT/fix.txt"

log "5. link-test over the streams, every link at once ($BIN)"
servers start $MACS
sleep 1
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
        exit 1
    fi
fi
suite stream

log "6. soak: every link, 10 minutes"
all_clients soak --mode soak --seconds 600

log "7. faults (no reloads or resets)"
servers start $MACS
a=${R[0]} b=${R[1]} c=${R[2]} d=${R[3]} e=${R[4]} f=${R[5]}
log "7a peer killed mid-session ($a -> $b), then a new session"
client "$a" f-kill --mode chain --count 200000 &
p=$!
sleep 3
on "$b" "date +%s.%N; pkill -9 -f '[l]ink-test serve --listen 0.0.0.0:$PORT'" > "$OUT/f-kill-inject.txt" 2>&1
wait $p
servers start "$b"
sleep 1
client "$a" f-kill-after --mode chain --count 2000

log "7b the server gives its stream up 2 s into a session ($b -> $c)"
client "$b" f-giveup --mode chain --count 20000 --server-drops-stream 2

log "7c the server stopped for 2 s mid-session ($c -> $d)"
client "$c" f-stop --mode chain --count 30000 &
p=$!
sleep 3
on "$d" "date +%s.%N; pkill -STOP -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; sleep 2; pkill -CONT -f '[l]ink-test serve --listen 0.0.0.0:$PORT'; date +%s.%N" > "$OUT/f-stop-inject.txt" 2>&1
wait $p

log "7d damaged and lost frames ($d -> $e): every 97th frame lost, every 101st damaged"
client "$d" f-chaos --mode chain --count 5000 --drop 97 --corrupt 101
client "$d" f-chaos --mode bulk --size 5242880 --count 20 --drop 97 --corrupt 101

log "7e ring overrun: 300 frames past the credits ($e -> $f)"
client "$e" f-overdrive --mode bulk --size 5242880 --count 20 --overdrive 300

log "7f two processes on one cable ($f -> $a)"
client "$f" f-two --mode chain --count 3000 &
p=$!
client "$f" f-two --mode chain --count 3000
wait $p

log "7g delayed start: the client 8 s before the server ($a -> $b)"
servers stop "$b"
client "$a" f-delayed --mode chain --count 1000 &
p=$!
sleep 8
servers start "$b"
wait $p

log "8. cleanup"
servers stop $MACS
for h in $MACS; do on "$h" "cat ~/$W/stream-serve.log" > "$OUT/serve-$h.log" 2>&1; done
for h in $MACS; do on "$h" "sudo dmesg | grep -iE 'tbstream|thunderbolt' | tail -60" > "$OUT/dmesg-end-$h.txt" 2>&1; done
cleanup
state after
links_ok || log "LINKS NOT ALL UP: tell the coordinator (no reset from here)"
log "done; $(grep -l '"error"' "$OUT"/*.jsonl 2>/dev/null | wc -l | tr -d ' ') result files with errors"
echo "stream-w2a done: $OUT"
