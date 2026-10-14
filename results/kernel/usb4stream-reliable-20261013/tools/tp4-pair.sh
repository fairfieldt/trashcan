#!/bin/bash
# Qwen3.8-27B TP4 on incite (driver, machine 0) and NODE (machine 1, a ring neighbour of incite) over their one cable,
# under gpuq leases on both: main's binary on the cable's stream by device (--tp-stream / --stream, as main runs it)
# against this branch, which finds the stream itself (--tp-connect NODE's tblink name), alternated, main first; then
# this branch's gates. Every process under scripts/with-clocks.sh and the 0 us latency request (tprun.py of
# results/qwen/tp8-opt-20261008), in ~/trashcan-exp/stream-tp on both Macs (tp4-workspace.sh). Needs the automatic
# stream setup (tblink) on both. DRY=1 prints the runs.
# usage: TICKETS=incite=ID,NODE=ID NODE=enlight tp4-pair.sh
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../../.." && pwd)
ROOT=/home/fairfieldt/trashcan-exp/stream-tp
TPRUN=${TPRUN:-$repo/results/qwen/tp8-opt-20261008/tools/tprun.py}
COOL=${COOL:-$repo/results/qwen/tp4-wire-20261009/tools/cool.sh}
MAIN=${MAIN:-trashcan-15e5457}
MINE=trashcan-stream
NODE=${NODE:-enlight}
MTP=/home/fairfieldt/models/qwen3.8-27b/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/tp4-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
dry() { [ "${DRY:-0}" = 1 ]; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
[ -n "${TICKETS:-}" ] || dry || { echo "set TICKETS=incite=ID,$NODE=ID" >&2; exit 1; }

for h in incite "$NODE"; do
    ssh "$h" "test -x $ROOT/$MAIN && test -x $ROOT/$MINE && test -x $ROOT/scripts/with-clocks.sh && test -f $ROOT/scripts/latency-run.py" \
        || { log "$h: $ROOT incomplete"; exit 1; }
done
# incite's tblink name for NODE (its bus to NODE), and both ends' stream devices on that cable.
ADDR=$(ssh incite /usr/local/sbin/tblink links | awk -F'\t' -v n="$NODE" '$2 == n {print n ".tb" $3}')
DX=$(ssh incite /usr/local/sbin/tblink links | awk -F'\t' -v n="$NODE" '$2 == n {print $10}')
DY=$(ssh "$NODE" /usr/local/sbin/tblink links | awk -F'\t' '$2 == "incite" {print $10}')
log "pair incite + $NODE ($ADDR); streams: incite $DX, $NODE $DY"
case "$DX$DY" in */dev/tbstream*/dev/tbstream*) ;; *) dry || { log "no stream on incite-$NODE"; exit 1; } ;; esac

run() { # NAME main|mine DRIVER
    local name=$1 side=$2 driver=$3 bin node extra
    if [ "$side" = main ]; then bin=$MAIN node="$NODE:--stream $DY" extra="--tp-stream $DX"
    else bin=$MINE node="$NODE:--listen 0.0.0.0:7072" extra="--tp-connect $ADDR:7072"; fi
    if dry; then log "  $name ($bin): $driver $extra; node $node"; return 0; fi
    TP8_ROOT=$ROOT python3 "$TPRUN" --name "$name" --binary "$bin" --tickets "$TICKETS" --node "$node" \
        --driver "$driver $extra" > "$OUT/$name.out" 2>&1
    log "  $name rc=$?: $(grep -oE '"(pp|tg)_tok_s":[0-9.]+' "$OUT/$name.out" | tr '\n' ' ')"
}
cool() { dry || "$COOL" 55 incite "$NODE" >> "$OUT/cool.txt" 2>&1; }

DEC="qwen-bench --pp 512 --tg 128 --reps 2"
log "warm-up (page caches; not counted)"
run stream-tp4-warm main "qwen-bench --pp 512 --tg 16 --reps 1"
log "decode and prefill 512, alternated"
for i in 1 2; do
    run stream-tp4-dec-main-$i main "$DEC"
    run stream-tp4-dec-mine-$i mine "$DEC"
done
log "MTP, one alternated pair, cards cooled to 55 C before each"
cool; run stream-tp4-mtp-main-1 main "$DEC --mtp $MTP"
cool; run stream-tp4-mtp-mine-1 mine "$DEC --mtp $MTP"
log "gates with this branch"
run stream-tp4-val-mine mine "qwen-validate results/stream-tp4-val-mine"
run stream-tp4-check-mtp-mine mine "qwen-check --mtp $MTP"
run stream-tp4-check-v3-mine mine "qwen-check --verify 3"
log "done"
echo "stream-tp4 done: $OUT"
