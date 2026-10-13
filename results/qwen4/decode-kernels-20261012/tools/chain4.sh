#!/bin/bash
# Short six-Mac confirmation of this branch rebased on b5b79c7 (trashcan-decb) against trashcan-b5b79c7 (run under
# chain6.sh, TICKETS set): validation on the 14-token reference and p100, one decode bench each.
# usage: chain4.sh TAG
tag=$1
B=trashcan-b5b79c7
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a90072f31ff42427f/results/qwen4/chain-20261012
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
P100=/home/fairfieldt/trashcan-exp/flashnext/p100.ids
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
run() { python3 tools/q4run.py --tickets $TICKETS --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
    > /dev/null 2>&1; echo "rc=$? $*" | cut -c1-100; }
run --name dec-$tag-val14 --binary trashcan-decb --driver "validate results/dec-$tag-val14 $C --batch 128 --greedy 10"
run --name dec-$tag-val100 --binary trashcan-decb --driver "validate results/dec-$tag-val100 $C --batch 128 --greedy 10 --tokens-file $P100"
run --name dec-$tag-val100-base --binary $B --driver "validate results/dec-$tag-val100-base $C --batch 128 --greedy 10 --tokens-file $P100"
run --name dec-$tag-tg-$B-1 --binary $B --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $L"
run --name dec-$tag-tg-trashcan-decb-1 --binary trashcan-decb --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $L"
echo "dec-$tag chain session done"
