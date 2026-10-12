#!/bin/bash
# Six-Mac session (run through chain6.sh, which exports TICKETS): validation, then the end-to-end A/B, alternating
# the baseline trashcan-88def0d (main) and trashcan-mixk (rebased on it).
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-afc5b5ba2c9d5d118/results/qwen4/chain-20261012
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
P=/home/fairfieldt/trashcan-exp/flashnext/p100.ids
run() { python3 tools/q4run.py --tickets $TICKETS --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1; echo "rc $? $*" | cut -c1-100; }
run --name mixk-val100-new --binary trashcan-mixk --driver "validate results/mixk-val100-new $C --greedy 10 --tokens-file $P"
run --name mixk-val100-old --binary trashcan-88def0d --driver "validate results/mixk-val100-old $C --greedy 10 --tokens-file $P"
run --name mixk-val14-new --binary trashcan-mixk --driver "validate results/mixk-val14-new $C --greedy 10"
for r in 1 2; do
  for b in trashcan-88def0d trashcan-mixk; do
    run --name mixk-pp2048-$b-$r --binary $b --driver "bench $C --pp 2048 --tg 32 --tokens-file $L"
    run --name mixk-pp512-$b-$r --binary $b --driver "bench $C --pp 512 --tg 64 --tokens-file $L"
  done
done
ssh incite 'cd ~/trashcan-exp/flashnext/results && for f in mixk-pp*.json; do python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1], round(d[\"pp_tok_s\"],1), round(d[\"tg_tok_s\"],2))" $f; done'
echo "mixk chain session done"
run --name mixk-val100-new2 --binary trashcan-mixk --driver "validate results/mixk-val100-new2 $C --greedy 10 --tokens-file $P"
echo "mixk chain session done (with val100-new2)"
