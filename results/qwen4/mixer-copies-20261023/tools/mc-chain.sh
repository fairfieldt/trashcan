#!/bin/bash
# Mixer copies, six Macs (under chain6.sh, TICKETS set): validation of the candidate on the 14-token reference prompt
# and p100 at 128-token chunks, one alternated decode pair after a 512-token prompt, then the candidate at 256-token
# chunks (the change's point: 32K context with 256-token chunks, which main's allocation does not fit): validation
# and prefill at 2048 and 8192 tokens against the candidate at 128. DRY=1 prints the runs.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
CAND=${CAND:-trashcan-mixc1}
BASE=${BASE:-trashcan-b7f4979}
TAG=${TAG:-mc1}
OUT=/Users/fairfieldt/dev/trashcan/results/qwen4/mixer-copies-20261023/chain/$TAG
W=/home/fairfieldt/trashcan-exp/flashnext
eval "$(python3 tools/ring.py)" || exit 1
N=$ADDRS
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
run() { # name binary driver-args
    local name=$1 bin=$2 driver=$3; shift 3
    if [ "${DRY:-0}" = 1 ]; then echo "run $TAG-$name ($bin): $driver" | cut -c1-170; return; fi
    local s=$(date +%s)
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" "$@" --driver "$driver" > /dev/null 2>&1
    local rc=$?
    echo "$TAG-$name rc $rc $(( $(date +%s) - s )) s"
    if [ $rc -ne 0 ]; then fetch; echo "mc chain $TAG stopped: $TAG-$name failed (rc $rc)"; exit 1; fi
}
fetch() {
    mkdir -p $OUT
    rsync -a "incite:$W/results/$TAG-*" "incite:$W/results/clocks/$TAG-*" $OUT/ 2> /dev/null
    for h in incite ${NODES//,/ }; do
        mkdir -p $OUT/$h
        rsync -a "$h:$W/results/node-$TAG-*" "$h:$W/results/clocks/node-$TAG-*" $OUT/$h/ 2> /dev/null
    done
}
echo "chain: incite,$NODES"
run val14 $CAND "validate results/$TAG-val14 $C --batch 128 --greedy 10"
run val100 $CAND "validate results/$TAG-val100 $C --batch 128 --greedy 10 --tokens-file $W/p100.ids"
run pp512-base $BASE "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-cand $CAND "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $W/long.ids"
run val100-b256 $CAND "validate results/$TAG-val100-b256 $C --batch 256 --greedy 10 --tokens-file $W/p100.ids"
run pp2048-b128 $CAND "bench $C --batch 128 --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp2048-b256 $CAND "bench $C --batch 256 --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp8192-b128 $CAND "bench $C --batch 128 --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
run pp8192-b256 $CAND "bench $C --batch 256 --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
[ "${DRY:-0}" = 1 ] && exit 0
fetch
echo "mc chain $TAG done"
