#!/bin/bash
# Session 1 (through chain6.sh): the decode timeline. main (4c1346d) as the baseline, the trace build without and with
# the trace (TRASHCAN_Q4_TRACE on every process; each Mac writes results/host-trace1-HOST.json).
cd "$(dirname "$0")/../../chain-20261012" || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
B="bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run() {
    local name=$1 bin=$2; shift 2
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$name" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1
    echo "$name rc $? $(ssh incite "grep -o '\"tg_tok_s\":[0-9.]*' trashcan-exp/flashnext/results/$name.json")"
}
run host-s1-base trashcan-4c1346d --driver "$B"
run host-s1-notrace trashcan-host-trace --driver "$B"
run host-s1-trace trashcan-host-trace --env "TRASHCAN_Q4_TRACE=results/host-trace1" --driver "$B"
run host-s1-base2 trashcan-4c1346d --driver "$B"
echo "session1 done"
