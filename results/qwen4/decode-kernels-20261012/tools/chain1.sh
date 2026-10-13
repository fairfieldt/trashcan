#!/bin/bash
# Six-Mac session (run under chain6.sh, which exports TICKETS): validation of trashcan-dec, then the decode A/B
# against trashcan-main (alternating), then one pp2048 pair.
# usage: chain1.sh TAG [REPS]
tag=$1 reps=${2:-3}
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a90072f31ff42427f/results/qwen4/chain-20261012
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
P100=/home/fairfieldt/trashcan-exp/flashnext/p100.ids
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
run() { python3 tools/q4run.py --tickets $TICKETS --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
    > /dev/null 2>&1; echo "rc=$? $*" | cut -c1-100; }
# Single-Mac check of the next candidate (trashcan-dec2) on incite first, if present.
inc=$(echo "$TICKETS" | tr , '\n' | sed -n 's/^incite=//p')
ssh incite "cd ~/trashcan-exp/flashnext && test -x trashcan-dec2 && gpuq run $inc -- ./dec-sb.sh 64 32 dec-$tag-sb-dec=trashcan-dec:TRASHCAN_Q4_PROFILE=1 dec-$tag-sb-dec2=trashcan-dec2:TRASHCAN_Q4_PROFILE=1 dec-$tag-sb-dec2upc=trashcan-dec2:TRASHCAN_Q4_PROFILE=1,TRASHCAN_Q4_UPC=1 dec-$tag-sb-dec2old=trashcan-dec2:TRASHCAN_Q4_PROFILE=1,TRASHCAN_Q4_GUOLD=1"
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
for v in dec:trashcan-dec dec2upc:trashcan-dec2; do
    ssh incite "cd ~/trashcan-exp/flashnext && test -x trashcan-dec2 && gpuq run $inc -- env TRASHCAN_Q4_PROFILE=1 TRASHCAN_Q4_UPC=1 ./${v#*:} qwen4 stage-bench $M --ranges 0:4,4:8 --gpus 0,1 --context 32768 --batch 3 --head-rows 0:21504,21504:43008 --decode 2 --start 64 --prefill 5 > results/dec-$tag-pf3-${v%%:*}.json 2> results/dec-$tag-pf3-${v%%:*}.err"
done
run --name dec-$tag-val14-b1 --binary trashcan-dec --driver "validate results/dec-$tag-val14-b1 $C --batch 1 --greedy 10"
run --name dec-$tag-val14 --binary trashcan-dec --driver "validate results/dec-$tag-val14 $C --batch 128 --greedy 10"
run --name dec-$tag-val100 --binary trashcan-dec --driver "validate results/dec-$tag-val100 $C --batch 128 --greedy 10 --tokens-file $P100"
run --name dec-$tag-val100-base --binary trashcan-main --driver "validate results/dec-$tag-val100-base $C --batch 128 --greedy 10 --tokens-file $P100"
run --name dec-$tag-val14-b1-base --binary trashcan-main --driver "validate results/dec-$tag-val14-b1-base $C --batch 1 --greedy 10"
for r in $(seq 1 $reps); do
    for b in trashcan-main trashcan-dec; do
        run --name dec-$tag-tg-$b-$r --binary $b --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $L"
    done
done
for b in trashcan-main trashcan-dec; do
    run --name dec-$tag-pp2048-$b --binary $b --driver "bench $C --batch 128 --pp 2048 --tg 32 --tokens-file $L"
done
echo "dec-$tag chain session done"
