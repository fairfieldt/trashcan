#!/bin/bash
# Real-text prefill A/B: baseline (HEAD 0802e42 + bench --tokens-file) against fused mixers to 64 + per-slot experts.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012
T=incite=145,indite=13,inflight=5,alight=5,invite=126,enlight=16
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
run() { python3 tools/q4run.py --tickets $T --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1; echo "done $*" | cut -c1-80; }
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
run --name real512-base-1 --binary trashcan-base --driver "bench $C --pp 512 --tg 64 --tokens-file $L"
run --name real512-trashcan-new-3 --binary trashcan-new --driver "bench $C --pp 512 --tg 64 --tokens-file $L"
run --name real512-base-2 --binary trashcan-base --driver "bench $C --pp 512 --tg 64 --tokens-file $L"
run --name real2048-base --binary trashcan-base --driver "bench $C --pp 2048 --tg 32 --tokens-file $L"
echo ALL DONE
