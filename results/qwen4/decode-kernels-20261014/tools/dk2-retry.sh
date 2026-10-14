#!/bin/bash
# dk2 six-Mac session for valid pairs (under chain6.sh, TICKETS set): the decode and prefill pairs of dk2-chain.sh
# again (decode after a 512-token prompt twice, prefill at 2048 and 8192), keeping only runs that ran every link on
# its stream (tools/transport.py --check; on 7.3.0-rc7 the stream driver sometimes leaves a link on TCP, and a pair
# whose arms differ in transport is not a valid comparison): each run is repeated, up to 4 times, until it does.
# Attempts keep their names (RUN-aK); the accepted one is copied to RUN. Stops at the first failed run.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a70283bc249c19df0/results/qwen4/chain-20261012 || exit 1
CAND=${CAND:-trashcan-dk2-c2}
BASE=${BASE:-trashcan-9baedca}
TAG=${TAG:-dk2c2r}
DK=/Users/fairfieldt/dev/claude-work/flashnext-20261012/dk2
L=$DK/chain/$TAG
W=/home/fairfieldt/trashcan-exp/flashnext
eval "$(python3 tools/ring.py)" || exit 1
N=$ADDRS
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
fetch() { # one run's driver outputs and clock log, and every node's log and clock log
    local name=$1
    mkdir -p "$L"
    rsync -a "incite:$W/results/$TAG-$name.*" "incite:$W/results/clocks/$TAG-$name.txt" "$L/" 2> /dev/null
    for h in ${NODES//,/ }; do
        mkdir -p "$L/$h"
        rsync -a "$h:$W/results/node-$TAG-$name-$h.log" "$h:$W/results/clocks/node-$TAG-$name-$h.txt" "$L/$h/" 2> /dev/null
    done
}
run() { # name binary driver-args
    local name=$1 bin=$2 driver=$3 k
    if [ "${DRY:-0}" = 1 ]; then echo "run $TAG-$name ($bin): $driver" | cut -c1-160; return; fi
    for k in 1 2 3 4; do
        local s=$(date +%s)
        python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$TAG-$name-a$k" --driver-host incite \
            --nodes "$NODES" --driver "$driver" > /dev/null 2>&1
        local rc=$?
        fetch "$name-a$k"
        echo "$TAG-$name-a$k rc $rc $(( $(date +%s) - s )) s; $(python3 $DK/tools/transport.py "$L" $TAG --check "$name-a$k")"
        if [ $rc -ne 0 ]; then echo "dk2 retry $TAG stopped: $TAG-$name-a$k failed (rc $rc)"; exit 1; fi
        if python3 $DK/tools/transport.py "$L" $TAG --check "$name-a$k" > /dev/null; then
            for f in "$L/$TAG-$name-a$k".*; do cp "$f" "$L/$TAG-$name.${f##*.}"; done
            for h in ${NODES//,/ }; do
                cp "$L/$h/node-$TAG-$name-a$k-$h.log" "$L/$h/node-$TAG-$name-$h.log" 2> /dev/null
                cp "$L/$h/node-$TAG-$name-a$k-$h.txt" "$L/$h/node-$TAG-$name-$h.txt" 2> /dev/null
            done
            return 0
        fi
    done
    echo "$TAG-$name: no attempt ran every link on its stream"
}
echo "chain: incite,$NODES"
run pp512-base-1 $BASE "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-cand-1 $CAND "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-base-2 $BASE "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp512-cand-2 $CAND "bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run pp2048-base $BASE "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp2048-cand $CAND "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
run pp8192-base $BASE "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
run pp8192-cand $CAND "bench $C --pp 8192 --tg 32 --tokens-file $W/long-32k.ids"
echo "dk2 retry $TAG done"
