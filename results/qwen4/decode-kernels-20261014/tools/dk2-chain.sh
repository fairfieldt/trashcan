#!/bin/bash
# dk2 six-Mac session (under chain6.sh, TICKETS set): the candidate (CAND, default trashcan-dk2-c2) against main
# (BASE, default trashcan-9baedca): validation of the candidate on the 14-token reference prompt and p100, then
# alternated runs: decode after a 512-token prompt (two pairs), MTP on the three chat prompts (one pass each), prefill
# at 2048 and 8192 tokens (one pair each). Driven like results/qwen4/main-suite-20261013/tools/main-suite.sh (q4run.py,
# the chain order from the live Thunderbolt links). Stops at the first failed run (a GPU that hangs is not retried).
# DRY=1 prints the runs.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a70283bc249c19df0/results/qwen4/chain-20261012 || exit 1
CAND=${CAND:-trashcan-dk2-c2}
BASE=${BASE:-trashcan-9baedca}
TAG=${TAG:-dk2c2}
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
eval "$(python3 tools/ring.py)" || exit 1
N=$ADDRS
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
P=$W/mtp-prompts
run() { # name binary driver-args [extra q4run args]
    local name=$1 bin=$2 driver=$3; shift 3
    if [ "${DRY:-0}" = 1 ]; then echo "run $TAG-$name ($bin): $driver" | cut -c1-160; return; fi
    local s=$(date +%s)
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$TAG-$name" --driver-host incite \
        --nodes "$NODES" "$@" --driver "$driver" > /dev/null 2>&1
    local rc=$?
    echo "$TAG-$name rc $rc $(( $(date +%s) - s )) s"
    if [ $rc -ne 0 ]; then fetch; echo "dk2 chain $TAG stopped: $TAG-$name failed (rc $rc)"; exit 1; fi
}
fetch() { # the driver's outputs and clock logs, and every node's log and clock log
    mkdir -p /Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2/chain/$TAG
    rsync -a "incite:$W/results/$TAG-*" "incite:$W/results/clocks/$TAG-*" \
        /Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2/chain/$TAG/ 2> /dev/null
    for h in incite ${NODES//,/ }; do
        mkdir -p /Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2/chain/$TAG/$h
        rsync -a "$h:$W/results/node-$TAG-*" "$h:$W/results/clocks/node-$TAG-*" \
            /Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2/chain/$TAG/$h/ 2> /dev/null
    done
}
echo "chain: incite,$NODES"
run val14 $CAND "validate results/$TAG-val14 $C --greedy 10"
run val100 $CAND "validate results/$TAG-val100 $C --greedy 10 --tokens-file $W/p100.ids"
run pp512-base-1 $BASE "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-cand-1 $CAND "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-base-2 $BASE "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-cand-2 $CAND "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run mtp-base $BASE "./$BASE qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids" --raw
run mtp-cand $CAND "./$CAND qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids" --raw
run pp2048-base $BASE "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp2048-cand $CAND "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp8192-base $BASE "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
run pp8192-cand $CAND "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
[ "${DRY:-0}" = 1 ] && exit 0
fetch
echo "dk2 chain $TAG done"
