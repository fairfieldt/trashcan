#!/bin/bash
# GPU validation of the adaptive chunk rule (88def0d): the 100-token prompt (16-token chunks on 12 stages) and the
# 14-token reference prompt, through `qwen4 validate`. Run under chain6.sh (TICKETS set).
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
for v in "vc-val100:--tokens-file $W/p100.ids" "vc-val14:"; do
  python3 tools/q4run.py --tickets "$TICKETS" --binary trashcan-88def0d --name "${v%%:*}" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight --driver "validate results/${v%%:*} $C --greedy 10 ${v#*:}" > /dev/null 2>&1
  echo "${v%%:*} done"
done
echo "chunk validation done"
