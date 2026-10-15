#!/bin/bash
# A short six-Mac engine check of the links under chain6.sh (TICKETS set): RUNS chain runs with the stages loaded
# (results/qwen4/chain-20261012/tools/q4run.py, as the main suite starts them), the 14-token validation first and then
# `bench --pp 512 --tg 32`; after each run, the driver's `links:` line must show every link on its stream from its
# first epoch. Every run goes on regardless; the node logs' link lines are kept. DRY=1 prints the runs.
# usage: chain6.sh stream-benchcheck 15 OWNER -- bash results/kernel/usb4stream-busypoll-20261014/tools/benchcheck-session.sh
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../../qwen4/chain-20261012" || exit 1
BIN=${BIN:-trashcan-41da714}
TAG=${TAG:-bc-${BIN#trashcan-}}
RUNS=${RUNS:-5}
OUT=${OUT:-$HOME/dev/claude-work/flashnext-20261012/stream/benchcheck-$(date +%Y%m%d-%H%M%S)}
W=/home/fairfieldt/trashcan-exp/flashnext
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/steps.log"; }
eval "$(python3 tools/ring.py)" || exit 1
C="--gpus 0,1 --nodes $ADDRS --head-split 12 --context 32768 --batch 128 --ngram disk"
child=
trap '[ -n "$child" ] && kill -TERM $child 2> /dev/null && wait $child; exit 130' INT TERM
log "output $OUT; $BIN, $RUNS runs, chain $NODES"
bad=0
for i in $(seq "$RUNS"); do
    if [ "$i" = 1 ]; then
        name=val14 driver="validate results/$TAG-val14 $C --greedy 10"
    else
        name=pp512-$i driver="bench $C --pp 512 --tg 32 --tokens-file $W/long.ids"
    fi
    if [ "${DRY:-0}" = 1 ]; then
        log "run $TAG-$name: $driver"
        continue
    fi
    s=$(date +%s)
    # In the background, so that a TERM reaches the runner (which stops the nodes and the driver).
    python3 "${Q4RUN:-tools/q4run.py}" --tickets "$TICKETS" --binary "$BIN" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" --driver "$driver" > "$OUT/q4run-$name.txt" 2>&1 &
    child=$!
    wait $child
    rc=$?
    child=
    line=$(ssh -o ConnectTimeout=10 incite "grep -h '^links: ' $W/results/$TAG-$name.err 2>/dev/null | tail -1")
    ssh -o ConnectTimeout=10 incite "grep -h '^link \|^links: \|^validate\|^tg\|^pp\|tok/s' $W/results/$TAG-$name.err 2>/dev/null" > "$OUT/driver-$name.txt"
    for h in ${NODES//,/ }; do
        ssh -o ConnectTimeout=10 "$h" "grep -h '^link ' $W/results/node-$TAG-$name-$h.log 2>/dev/null" > "$OUT/node-$name-$h.txt"
    done
    case "$line" in
        "") verdict="no links line" ;;
        *retried* | *tcp*) verdict="NOT all on the stream at first try"; bad=$((bad + 1)) ;;
        *) verdict="all on the stream at first try" ;;
    esac
    log "$TAG-$name rc $rc $(( $(date +%s) - s )) s: $verdict; $line"
    for h in ${NODES//,/ }; do
        grep -h 'not opened\|stream down\|took' "$OUT/node-$name-$h.txt" | sed "s/^/    $h: /" | tee -a "$OUT/steps.log"
    done
    grep -h 'not opened\|stream down\|took' "$OUT/driver-$name.txt" | sed 's/^/    incite: /' | tee -a "$OUT/steps.log"
done
log "done: $bad of $RUNS runs had a link off its stream at first try"
echo "stream-benchcheck done: $bad of $RUNS runs had a link off its stream at first try"
