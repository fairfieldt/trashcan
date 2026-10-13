#!/bin/bash
# Six-Mac session for the grouped expert kernels (run through chain6.sh, which exports TICKETS):
#   exp-chain.sh TAG STEPS...      STEPS: val (p100 and p300, exp and main), ab2048, ab512, ab8k
# The baseline binary is $BASE (default trashcan-88def0d, main at 88def0d). Results land in ~/trashcan-exp/flashnext/results on each Mac (driver: incite), names exp-*-TAG.
set -uo pipefail
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-ace0bc9b91e2664e3/results/qwen4/chain-20261012
tag=$1; shift
BASE=${BASE:-trashcan-88def0d}
EXPB=${EXPB:-trashcan-exp}
bin() { [ $1 = main ] && echo $BASE || echo $EXPB; }
W=/home/fairfieldt/trashcan-exp/flashnext
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
run() { python3 tools/q4run.py --tickets $TICKETS --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
    > /dev/null 2>&1; echo "rc $? $*" | cut -c1-120; }
for step in "$@"; do
    case $step in
    val)
        for b in exp main; do
            run --name exp-val100-$b-$tag --binary $(bin $b) --driver "validate results/exp-val100-$b-$tag $C --greedy 10 --tokens-file $W/p100.ids"
            run --name exp-val300-$b-$tag --binary $(bin $b) --driver "validate results/exp-val300-$b-$tag $C --greedy 10 --tokens-file $W/exp-p300.ids"
        done ;;
    ab2048)
        for r in 1 2; do for b in main exp; do
            run --name exp-pp2048-$b-$r-$tag --binary $(bin $b) --driver "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
        done; done ;;
    ab512)
        for r in 1 2; do for b in main exp; do
            run --name exp-pp512-$b-$r-$tag --binary $(bin $b) --driver "bench $C --pp 512 --tg 64 --tokens-file $W/long.ids"
        done; done ;;
    confirm)
        for b in exp main; do
            run --name exp-val100-$b-$tag --binary $(bin $b) --driver "validate results/exp-val100-$b-$tag $C --greedy 10 --tokens-file $W/p100.ids"
        done
        for b in main exp; do
            run --name exp-pp2048-$b-1-$tag --binary $(bin $b) --driver "bench $C --pp 2048 --tg 32 --tokens-file $W/long.ids"
        done
        for b in main exp; do
            run --name exp-pp512-$b-1-$tag --binary $(bin $b) --driver "bench $C --pp 512 --tg 64 --tokens-file $W/long.ids"
        done ;;
    ab8k)
        for b in main exp; do
            run --name exp-pp8k-$b-$tag --binary $(bin $b) --driver "bench $C --pp 8192 --tg 32 --tokens-file $W/exp-long8k.ids"
        done ;;
    esac
done
# With sb: the final stage-bench (layers 0-7 on alight's GPUs) under alight's ticket, all tickets extended first.
case " $* " in *" sb "*)
    for t in ${TICKETS//,/ }; do gpuq --on ${t%%=*} extend ${t#*=} 25 -r "final stage-bench on alight" > /dev/null; done
    aid=$(echo $TICKETS | tr ',' '\n' | sed -n 's/^alight=//p')
    ssh alight "bash ~/trashcan-exp/flashnext/exp-lease-final.sh $aid" 2>&1 | grep -v '^\[clocks\]' ;;
esac
ssh incite "cd $W/results && python3 -c \"
import json,glob
for f in sorted(glob.glob('exp-pp*-$tag.json')):
    try: d=json.load(open(f)); print(f, round(d['pp_tok_s'],1), round(d['tg_tok_s'],2))
    except Exception as e: print(f, 'failed')
\""
echo "exp-chain $tag done"
