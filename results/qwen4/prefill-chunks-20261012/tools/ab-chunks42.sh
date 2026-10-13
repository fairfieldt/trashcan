#!/bin/bash
# The re-fitted chunk rule (42 tokens per chunk, trashcan-chk42) against main ef4ce87 (16, trashcan-ef4ce87) where the
# choice differs: 1024 tokens (32 -> 64) and 4096 (64 -> 128), real text, alternating. Run under chain6.sh.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
run() { python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight --driver "$3" > /dev/null 2>&1; echo "$1 done"; }
for r in 1 2; do
  for b in ef4ce87 chk42; do
    run a42-pp1024-$b-$r trashcan-$b "bench $C --pp 1024 --tg 16 --tokens-file $W/long.ids"
    run a42-pp4096-$b-$r trashcan-$b "bench $C --pp 4096 --tg 16 --tokens-file $W/long4k.ids"
  done
done
run a42-val100 trashcan-chk42 "validate results/a42-val100 $C --greedy 10 --tokens-file $W/p100.ids"
echo "chunk42 A/B done"
