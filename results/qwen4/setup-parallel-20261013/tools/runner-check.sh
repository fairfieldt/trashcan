#!/bin/bash
# The connect retry (trashcan-retry: main 6f6635a + Conn::connect retrying refused connections) with the runner that
# starts the driver once the nodes listen and stops them all at once: validate against main, bit-identical expected;
# the runner prints its phase times. Then main's suite on 4307169 (results/qwen4/main-suite-20261013). Under chain6.sh.
cd /Users/fairfieldt/dev/trashcan/results/qwen4/chain-20261012 || exit 1
for t in ${TICKETS//,/ }; do gpuq --on "${t%%=*}" extend "${t#*=}" 15 -r "main suite on 4307169 added to this session" > /dev/null 2>&1; done
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
run() { s=$(date +%s); python3 tools/q4run.py --tickets "$TICKETS" --binary "$2" --name "$1" --driver-host incite \
    --nodes indite,inflight,alight,invite,enlight --driver "$3" 2>&1 | grep -E "nodes up|nodes stopped|not listening"; echo "$1 $(( $(date +%s) - s )) s"; }
run rc-val14-6f6635a trashcan-6f6635a "validate results/rc-val14-6f6635a $C --greedy 10"
run rc-val14-retry trashcan-retry "validate results/rc-val14-retry $C --greedy 10"
ssh incite 'cd ~/trashcan-exp/flashnext/results; cmp rc-val14-retry/logits.f32 rc-val14-6f6635a/logits.f32 && echo "logits byte-identical"; cat rc-val14-retry.json'
/Users/fairfieldt/dev/trashcan/results/qwen4/main-suite-20261013/tools/main-suite.sh
echo "runner check and main suite done"
