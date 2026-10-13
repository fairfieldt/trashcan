#!/bin/bash
# Six-Mac session for the final binary (run under chain6.sh, TICKETS set): single-Mac stage-bench on incite (decode,
# 2-4 token chunks, 128-token chunks, 32K decode) against trashcan-main, then validation and prefill/decode pairs.
# usage: chain2.sh TAG
tag=$1
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a90072f31ff42427f/results/qwen4/chain-20261012
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
P100=/home/fairfieldt/trashcan-exp/flashnext/p100.ids
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
run() { python3 tools/q4run.py --tickets $TICKETS --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
    > /dev/null 2>&1; echo "rc=$? $*" | cut -c1-100; }
inc=$(echo "$TICKETS" | tr , '\n' | sed -n 's/^incite=//p')
ssh incite "cd ~/trashcan-exp/flashnext && gpuq run $inc -- ./dec-sb.sh 64 32 dec-$tag-sb-new=trashcan-dec dec-$tag-sb-base=trashcan-main dec-$tag-sb-newp=trashcan-dec:TRASHCAN_Q4_PROFILE=1"
for b in 2 3 4 128; do
    for v in base:trashcan-main new:trashcan-dec; do
        ssh incite "cd ~/trashcan-exp/flashnext && gpuq run $inc -- scripts/with-clocks.sh results/clocks/dec-$tag-pf$b-${v%%:*}.txt ./${v#*:} qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch $b --head-rows 0:21504,21504:43008 --decode 2 --start 64 --prefill 5 > results/dec-$tag-pf$b-${v%%:*}.json 2> results/dec-$tag-pf$b-${v%%:*}.err"
        echo "pf$b ${v%%:*} rc=$?"
    done
done
ssh incite "cd ~/trashcan-exp/flashnext && gpuq run $inc -- ./dec-sb.sh 32512 16 dec-$tag-sb32k-new=trashcan-dec dec-$tag-sb32k-base=trashcan-main"
run --name dec-$tag-val14 --binary trashcan-dec --driver "validate results/dec-$tag-val14 $C --batch 128 --greedy 10"
run --name dec-$tag-val100 --binary trashcan-dec --driver "validate results/dec-$tag-val100 $C --batch 128 --greedy 10 --tokens-file $P100"
for r in 1 2; do
    for b in trashcan-main trashcan-dec; do
        run --name dec-$tag-pp2048-$b-$r --binary $b --driver "bench $C --batch 128 --pp 2048 --tg 32 --tokens-file $L"
    done
done
for b in trashcan-main trashcan-dec; do
    run --name dec-$tag-tg-$b-1 --binary $b --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $L"
done
echo "dec-$tag chain session done"
