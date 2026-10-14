#!/bin/bash
# Session 1 (under chain6.sh): long-prompt baseline on main 4c1346d (8K, 16K, 30K real-text tokens; decode after them)
# and timeline runs of the instrumented binary (trashcan-long: 4c1346d plus TRASHCAN_Q4_TIMELINE events) at 8K and 30K.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
W=/home/fairfieldt/trashcan-exp/flashnext
run() { # name binary pp [env]
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight --env "${4:-}" --driver "bench --gpus 0,1 --nodes $N \
        --head-split 12 --context 32768 --batch 128 --ngram disk --pp $3 --tg 32 --tokens-file $W/long-32k.ids" \
        > /dev/null 2>&1
    echo "$1 rc=$? $(ssh incite "head -c 300 $W/results/$1.json")"
}
TL='TRASHCAN_Q4_TIMELINE=1 TRASHCAN_TP_PROFILE=1'
run long-s1-base-pp8192 trashcan-4c1346d 8192
run long-s1-tl-pp8192 trashcan-long 8192 "$TL"
run long-s1-base-pp16384 trashcan-4c1346d 16384
run long-s1-base-pp30000 trashcan-4c1346d 30000
run long-s1-tl-pp30000 trashcan-long 30000 "$TL"
echo "s1 done"
