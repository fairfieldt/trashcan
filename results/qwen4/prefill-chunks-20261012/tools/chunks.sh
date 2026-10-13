#!/bin/bash
# Prefill chunk size against prompt length on main (90cb371): --batch 128, 64, 32 at 512 and 2048 real-text tokens,
# alternating, run under chain6.sh (TICKETS set). Outputs: ~/trashcan-exp/flashnext/results/chk-*.json on incite.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
run() { # NAME BATCH PP
  python3 tools/q4run.py --tickets "$TICKETS" --binary trashcan-main --name "$1" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight \
    --driver "bench --gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch $2 --ngram disk --pp $3 --tg 16 --tokens-file $L" > /dev/null 2>&1
  echo "$1 done"
}
for r in 1 2; do
  for b in 128 64 32; do run chk-pp512-b$b-$r $b 512; done
done
for b in 128 64 32; do run chk-pp2048-b$b $b 2048; done
echo "chunk sweep done"
