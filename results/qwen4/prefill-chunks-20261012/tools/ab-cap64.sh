#!/bin/bash
# Does 128 ever beat 64-token chunks on the chain? 8192 real-text tokens with trashcan-chk42 at --batch 64 (chunks of 64)
# and --batch 128 (chunks of 128), alternating. Run under chain6.sh.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
W=/home/fairfieldt/trashcan-exp/flashnext
run() { python3 tools/q4run.py --tickets "$TICKETS" --binary trashcan-chk42 --name "$1" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight --driver "bench --gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch $2 --ngram disk --pp 8192 --tg 16 --tokens-file $W/long8k.ids" > /dev/null 2>&1; echo "$1 done"; }
for r in 1 2; do run cap-pp8192-b64-$r 64; run cap-pp8192-b128-$r 128; done
echo "cap64 A/B done"
