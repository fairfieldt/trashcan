#!/bin/bash
# Measured chunk bands (trashcan-bands) against main ef4ce87 (the sqrt(16 n / 11) rule) where they choose differently:
# 256 tokens (16 -> 32), 1024 (32 -> 64), 6144 (128 -> 64); real text, alternating; then a validation. Run under chain6.sh.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
run() { python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight --driver "$3" > /dev/null 2>&1; echo "$1 done"; }
for r in 1 2; do
  for b in ef4ce87 bands; do
    run bd-pp256-$b-$r trashcan-$b "bench $C --pp 256 --tg 16 --tokens-file $W/long.ids"
    run bd-pp1024-$b-$r trashcan-$b "bench $C --pp 1024 --tg 16 --tokens-file $W/long.ids"
  done
done
for b in ef4ce87 bands; do run bd-pp6144-$b trashcan-$b "bench $C --pp 6144 --tg 16 --tokens-file $W/long8k.ids"; done
run bd-val100 trashcan-bands "validate results/bd-val100 $C --greedy 10 --tokens-file $W/p100.ids"
echo "bands A/B done"
