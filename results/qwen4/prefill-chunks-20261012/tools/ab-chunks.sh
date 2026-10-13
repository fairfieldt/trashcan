#!/bin/bash
# Adaptive prefill chunk size (trashcan-chk) against main's fixed 128-token chunks (trashcan-main, 90cb371), real text,
# alternating; then validation of the 100-token prompt (now 16-token chunks). Run under chain6.sh (TICKETS set).
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
run() { # NAME BINARY DRIVER-ARGS
  python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight --driver "$3" > /dev/null 2>&1
  echo "$1 done"
}
for r in 1 2; do
  for b in main chk; do
    run ac-pp512-$b-$r trashcan-$b "bench $C --pp 512 --tg 16 --tokens-file $W/long.ids"
    run ac-pp2048-$b-$r trashcan-$b "bench $C --pp 2048 --tg 16 --tokens-file $W/long.ids"
  done
done
for b in main chk; do
  run ac-pp3000-$b trashcan-$b "bench $C --pp 3000 --tg 16 --tokens-file $W/long.ids"
  run ac-pp8192-$b trashcan-$b "bench $C --pp 8192 --tg 16 --tokens-file $W/long8k.ids"
done
run ac-val100-chk trashcan-chk "validate results/ac-val100-chk $C --greedy 10 --tokens-file $W/p100.ids"
run ac-val14-chk trashcan-chk "validate results/ac-val14-chk $C --greedy 10"
echo "adaptive chunk A/B done"
