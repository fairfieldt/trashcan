#!/bin/bash
# Final six-Mac session (run under chain6.sh, TICKETS set): this branch rebased on 9e0a420 (trashcan-dec) against
# trashcan-9e0a420. First on incite alone: dumps (14 tokens in chunks of 1 and 3, the 100-token prompt in chunks of
# 16), decode stage-bench, chunks of 2, 3, 4, 16, 32, 64; then validation, decode pairs and prefill pairs on six Macs.
# usage: chain3.sh TAG
tag=$1
B=trashcan-9e0a420
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a90072f31ff42427f/results/qwen4/chain-20261012
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
P100=/home/fairfieldt/trashcan-exp/flashnext/p100.ids
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
run() { python3 tools/q4run.py --tickets $TICKETS --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
    > /dev/null 2>&1; echo "rc=$? $*" | cut -c1-100; }
inc=$(echo "$TICKETS" | tr , '\n' | sed -n 's/^incite=//p')
on1() { ssh incite "cd ~/trashcan-exp/flashnext && gpuq run $inc -- $*"; }
ssh incite "cd ~/trashcan-exp/flashnext && echo '248045 846 198 9419 1017 248046 198 248045 74455 198 248068 271 248069 271' > dec-ref14.ids"
for spec in 1:dec-ref14.ids 3:dec-ref14.ids 16:p100.ids; do
    b=${spec%%:*} ids=${spec#*:}
    for v in base:$B new:trashcan-dec; do
        n=dec-$tag-dump-${v%%:*}-b$b
        on1 "bash -c 'rm -rf results/$n; ./${v#*:} qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b --tokens-file $ids --out results/$n > results/$n.log 2>&1'"
        echo "dump ${v%%:*} b$b rc=$?"
    done
done
on1 ./dec-sb.sh 64 32 dec-$tag-sb-new=trashcan-dec dec-$tag-sb-base=$B dec-$tag-sb-newp=trashcan-dec:TRASHCAN_Q4_PROFILE=1
for b in 2 3 4 16 32 64; do
    for v in base:$B new:trashcan-dec; do
        n=dec-$tag-pf$b-${v%%:*}
        on1 "bash -c 'scripts/with-clocks.sh results/clocks/$n.txt ./${v#*:} qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b --head-rows 0:21504,21504:43008 --decode 2 --start 64 --prefill 5 > results/$n.json 2> results/$n.err'"
        echo "pf$b ${v%%:*} rc=$?"
    done
done
run --name dec-$tag-val14-b1 --binary trashcan-dec --driver "validate results/dec-$tag-val14-b1 $C --batch 1 --greedy 10"
run --name dec-$tag-val14 --binary trashcan-dec --driver "validate results/dec-$tag-val14 $C --batch 128 --greedy 10"
run --name dec-$tag-val100 --binary trashcan-dec --driver "validate results/dec-$tag-val100 $C --batch 128 --greedy 10 --tokens-file $P100"
run --name dec-$tag-val100-base --binary $B --driver "validate results/dec-$tag-val100-base $C --batch 128 --greedy 10 --tokens-file $P100"
for r in 1 2 3; do
    for b in $B trashcan-dec; do
        run --name dec-$tag-tg-$b-$r --binary $b --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $L"
    done
done
for r in 1 2; do
    for b in $B trashcan-dec; do
        run --name dec-$tag-pp2048-$b-$r --binary $b --driver "bench $C --batch 128 --pp 2048 --tg 32 --tokens-file $L"
    done
done
echo "dec-$tag chain session done"
