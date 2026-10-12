#!/bin/bash
# Validation and real-text A/B for fused mixers to 64 tokens + per-slot experts (trashcan-new) against c21ec4e (trashcan).
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012
T=incite=145,indite=13,inflight=5,alight=5,invite=126,enlight=16
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
run() { python3 tools/q4run.py --tickets $T --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1; echo "done $*" | cut -c1-80; }
for p in incite:145 indite:13 inflight:5 alight:5 invite:126 enlight:16; do gpuq --on ${p%%:*} wait ${p##*:} > /dev/null; done
run --name val6-new --binary trashcan-new --driver "validate results/val6-new $C --greedy 10"
run --name val100-new --binary trashcan-new --driver "validate results/val100-new $C --greedy 10 --tokens-file /home/fairfieldt/trashcan-exp/flashnext/p100.ids"
run --name val100-old --binary trashcan --driver "validate results/val100-old $C --greedy 10 --tokens-file /home/fairfieldt/trashcan-exp/flashnext/p100.ids"
for r in 1 2; do
  for b in trashcan trashcan-new; do
    run --name real512-$b-$r --binary $b --driver "bench $C --pp 512 --tg 64 --tokens-file /home/fairfieldt/trashcan-exp/flashnext/long.ids"
  done
done
for b in trashcan trashcan-new; do
  run --name real2048-$b --binary $b --driver "bench $C --pp 2048 --tg 32 --tokens-file /home/fairfieldt/trashcan-exp/flashnext/long.ids"
done
echo ALL DONE
