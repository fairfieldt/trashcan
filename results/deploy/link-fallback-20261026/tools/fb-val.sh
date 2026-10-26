#!/bin/bash
# Link fallback fix, six Macs (under chain6.sh, TICKETS set): greedy validation of the candidate and the base on the
# 14-token reference prompt and the 100-token prompt, then one decode A/B pair (base, candidate) after a 512-token prompt.
cd $HOME/dev/claude-work/trashcan-release-20261026/fallback/lab/results/qwen4/chain-20261012 || exit 1
CAND=${CAND:-trashcan-fb2}
BASE=${BASE:-trashcan-a16base}
TAG=${TAG:-fb2}
STEPS=${STEPS:-val14 val14base val100 val100base pp512-base pp512-cand}
OUT=$HOME/dev/claude-work/trashcan-release-20261026/fallback/results/val/$TAG
W=/home/fairfieldt/trashcan-exp/flashnext
eval "$(python3 tools/ring.py)" || exit 1
C="--gpus 0,1 --nodes $ADDRS --head-split 12 --context 32768 --ngram disk"
run() { # name binary driver-args
    local name=$1 bin=$2 driver=$3; shift 3
    local s=$(date +%s)
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" "$@" --driver "$driver" > /dev/null 2>&1
    echo "$TAG-$name rc $? $(( $(date +%s) - s )) s"
}
echo "chain: incite,$NODES"
for st in $STEPS; do
    case $st in
        val14) run val14 $CAND "validate results/$TAG-val14 $C --batch 128 --greedy 10" ;;
        val14base) run val14-base $BASE "validate results/$TAG-val14-base $C --batch 128 --greedy 10" ;;
        val100) run val100 $CAND "validate results/$TAG-val100 $C --batch 128 --greedy 10 --tokens-file $W/p100.ids" ;;
        val100base) run val100-base $BASE "validate results/$TAG-val100-base $C --batch 128 --greedy 10 --tokens-file $W/p100.ids" ;;
        pp512-base) run pp512-base $BASE "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $W/long.ids" ;;
        pp512-cand) run pp512-cand $CAND "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $W/long.ids" ;;
    esac
done
mkdir -p $OUT
rsync -a "incite:$W/results/$TAG-*" "incite:$W/results/clocks/$TAG-*" $OUT/ 2> /dev/null
for h in incite ${NODES//,/ }; do
    mkdir -p $OUT/$h
    rsync -a "$h:$W/results/node-$TAG-*" "$h:$W/results/clocks/node-$TAG-*" $OUT/$h/ 2> /dev/null
done
echo "fb-val $TAG done"
