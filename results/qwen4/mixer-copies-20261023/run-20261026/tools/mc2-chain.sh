#!/bin/bash
# Mixer copies, six Macs, rerun 2026-10-26 (under chain6.sh, TICKETS set). As mc-chain.sh, plus baseline val14/val100
# for token-sequence and dump comparison, pp8192 in reversed order, and a final soft check of the baseline at batch
# 256 (it fit on one Mac with 0 MiB margin). Runs from a lab worktree; output under the A1 work directory.
L=/Users/fairfieldt/dev/claude-work/trashcan-release-20261026/a1
cd $L/a1-lab/results/qwen4/chain-20261012 || exit 1
CAND=${CAND:-trashcan-mixc2}
BASE=${BASE:-trashcan-91c3b82}
TAG=${TAG:-mc2}
OUT=$L/chain/$TAG
W=/home/fairfieldt/trashcan-exp/flashnext
eval "$(python3 tools/ring.py)" || exit 1
N=$ADDRS
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
run() { # [soft] name binary driver-args
    local soft=0; [ "$1" = soft ] && { soft=1; shift; }
    local name=$1 bin=$2 driver=$3; shift 3
    if [ "${DRY:-0}" = 1 ]; then echo "run $TAG-$name ($bin): $driver" | cut -c1-170; return; fi
    local s=$(date +%s)
    mkdir -p $OUT
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" "$@" --driver "$driver" > $OUT/q4run-$name.txt 2>&1
    local rc=$?
    echo "$TAG-$name rc $rc $(( $(date +%s) - s )) s $(grep -E 'tok/s' $OUT/q4run-$name.txt | tail -2 | paste -sd' ' - | cut -c1-200)"
    if [ $rc -ne 0 ] && [ $soft = 0 ]; then fetch; echo "mc chain $TAG stopped: $TAG-$name failed (rc $rc)"; exit 1; fi
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
run val14-base $BASE "validate results/$TAG-val14-base $C --batch 128 --greedy 10"
run val14 $CAND "validate results/$TAG-val14 $C --batch 128 --greedy 10"
run val100-base $BASE "validate results/$TAG-val100-base $C --batch 128 --greedy 10 --tokens-file $W/p100.ids"
run val100 $CAND "validate results/$TAG-val100 $C --batch 128 --greedy 10 --tokens-file $W/p100.ids"
run pp512-base $BASE "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-cand $CAND "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $W/long.ids"
run val100-b256 $CAND "validate results/$TAG-val100-b256 $C --batch 256 --greedy 10 --tokens-file $W/p100.ids"
run pp2048-b128 $CAND "bench $C --batch 128 --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp2048-b256 $CAND "bench $C --batch 256 --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp8192-b256 $CAND "bench $C --batch 256 --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
run pp8192-b128 $CAND "bench $C --batch 128 --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
run soft val100-b256-base $BASE "validate results/$TAG-val100-b256-base $C --batch 256 --greedy 10 --tokens-file $W/p100.ids"
[ "${DRY:-0}" = 1 ] && exit 0
fetch
echo "mc chain $TAG done"
