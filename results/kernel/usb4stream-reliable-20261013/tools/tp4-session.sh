#!/bin/bash
# Qwen3.8-27B TP4 baselines on incite and NODE over their one cable, under pair2.sh leases on both (about 20 min): the
# ring-position probe on the cable (CPU only); then alternated rounds of main over TCP (--tp-connect / --listen), this
# branch (which finds the cable's stream itself) and main on the stream by device (--tp-stream / --stream) after both
# ends have read their devices dry: decode twice, MTP once (cards cooled to 60 C before each run); then this branch's
# qwen-validate. Every process under scripts/with-clocks.sh and the 0 us latency request (tprun.py), in
# ~/trashcan-exp/stream-tp on both Macs. No module reloads or resets. DRY=1 prints the runs.
# usage: pair2.sh stream-tp4b 20 OWNER incite enlight -- results/kernel/usb4stream-reliable-20261013/tools/tp4-session.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
ROOT=/home/fairfieldt/trashcan-exp/stream-tp
F=trashcan-exp/flashnext
TPRUN=${TPRUN:-$repo/results/qwen/tp8-opt-20261008/tools/tprun.py}
COOL=${COOL:-$repo/results/qwen/tp4-wire-20261009/tools/cool.sh}
MAIN=${MAIN:-trashcan-15e5457}
MINE=trashcan-stream
NODE=${NODE:-enlight}
MTP=/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/tp4b-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
on() { local h=$1; shift; if dry; then echo "    ssh $h: $*" >> "$OUT/steps.log"; return 0; fi; ssh -o ConnectTimeout=10 -o ServerAliveInterval=5 "$h" "$@"; }

if ! dry; then
    [ -n "${TICKETS:-}" ] || { echo "run under pair2.sh (TICKETS)" >&2; exit 1; }
    ( while :; do for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" status "${t#*=}" > /dev/null 2>&1; done; sleep 120; done ) > /dev/null 2>&1 &
    trap 'for j in $(jobs -p); do pkill -P $j 2>/dev/null; kill $j 2>/dev/null; done
        for h in incite $NODE; do ssh "$h" "pkill -f \"[s]tream-tbcheck.py\"" 2>/dev/null; done' EXIT
    trap 'exit 130' INT TERM
fi

for h in incite "$NODE"; do
    on "$h" "test -x $ROOT/$MAIN && test -x $ROOT/$MINE && test -x $ROOT/scripts/with-clocks.sh && test -f $ROOT/scripts/latency-run.py && test -f ~/$F/stream-tbcheck.py" \
        || { log "$h: files missing"; exit 1; }
done
ADDR=$(on incite /usr/local/sbin/tblink links | awk -F'\t' -v n="$NODE" '$2 == n {print n ".tb" $3}')
DX=$(on incite /usr/local/sbin/tblink links | awk -F'\t' -v n="$NODE" '$2 == n {print $10}')
DY=$(on "$NODE" /usr/local/sbin/tblink links | awk -F'\t' '$2 == "incite" {print $10}')
log "pair incite + $NODE ($ADDR); streams: incite $DX, $NODE $DY; this branch $(on incite "md5sum $ROOT/$MINE" | cut -c1-12)"
case "$DX$DY" in */dev/tbstream*/dev/tbstream*) ;; *) dry || { log "no stream on incite-$NODE"; exit 1; } ;; esac

log "1. the hold against the receiving ring's position (CPU only), both directions"
for dir in "incite $DX $NODE $DY" "$NODE $DY incite $DX"; do
    set -- $dir
    on "$3" "timeout -k 5 60 python3 ~/$F/stream-tbcheck.py pos $4 server 40 301" >> "$OUT/probe-pos.jsonl" 2>> "$OUT/probe-pos.err" &
    p=$!
    sleep 0.5
    on "$1" "timeout -k 5 60 python3 ~/$F/stream-tbcheck.py pos $2 client 40 301" >> "$OUT/probe-pos.jsonl" 2>> "$OUT/probe-pos.err"
    wait $p
done
dry || python3 -c '
import json, sys
for l in open(sys.argv[1]):
    try:
        d = json.loads(l)
    except ValueError:
        continue
    if d.get("side") == "client":
        print("  client %s: %d rounds, fillers needed by the client frame count mod 4: %s" % (d["host"], d["rounds"], d["fillers_by_position_mod_4"]))
' "$OUT/probe-pos.jsonl" | tee -a "$OUT/steps.log"

drain() { # both ends read their devices dry at once
    local p
    on "$NODE" "python3 ~/$F/stream-tbcheck.py drain $DY 0.5" >> "$OUT/drain.jsonl" 2>&1 &
    p=$!
    on incite "python3 ~/$F/stream-tbcheck.py drain $DX 0.5" >> "$OUT/drain.jsonl" 2>&1
    wait $p
    dry || log "  drained: $(tail -2 "$OUT/drain.jsonl" | tr '\n' ' ' | cut -c1-300)"
}
run() { # NAME main-tcp|main-stream|mine DRIVER
    local name=$1 side=$2 driver=$3 bin node extra
    case $side in
        main-tcp) bin=$MAIN node="$NODE:--listen 0.0.0.0:7072" extra="--tp-connect $ADDR:7072" ;;
        main-stream) bin=$MAIN node="$NODE:--stream $DY" extra="--tp-stream $DX"; drain ;;
        mine) bin=$MINE node="$NODE:--listen 0.0.0.0:7072" extra="--tp-connect $ADDR:7072" ;;
    esac
    if dry; then log "  $name ($bin): $driver $extra; node $node"; return 0; fi
    TP8_ROOT=$ROOT python3 "$TPRUN" --name "$name" --binary "$bin" --tickets "$TICKETS" --node "$node" \
        --driver "$driver $extra" > "$OUT/$name.out" 2>&1
    log "  $name rc=$?: $(grep -oE '"(pp|tg)_tok_s":[0-9.]+' "$OUT/$name.out" | tr '\n' ' ')$(grep -m1 -oE 'Error: .{0,120}' "$OUT/$name.out")"
}
cool() { dry || "$COOL" 60 incite "$NODE" >> "$OUT/cool.txt" 2>&1; }

DEC="qwen-bench --pp 512 --tg 128 --reps 2"
log "2. decode and prefill 512: main over TCP, this branch, main on the stream (drained first), twice"
for i in 1 2; do
    run stream-tp4b-dec-tcp-$i main-tcp "$DEC"
    run stream-tp4b-dec-mine-$i mine "$DEC"
    run stream-tp4b-dec-dev-$i main-stream "$DEC"
done
log "3. MTP once each, cards cooled to 60 C before each"
cool; run stream-tp4b-mtp-tcp-1 main-tcp "$DEC --mtp $MTP"
cool; run stream-tp4b-mtp-mine-1 mine "$DEC --mtp $MTP"
cool; run stream-tp4b-mtp-dev-1 main-stream "$DEC --mtp $MTP"
log "4. this branch's validation"
run stream-tp4b-val-mine mine "qwen-validate results/stream-tp4b-val-mine"
log "done"
echo "stream-tp4b done: $OUT"
