#!/bin/bash
# Six-Mac MTP session 2 (through chain6.sh, which exports TICKETS): the final drafter (4-bit dense, Q8 head, rows
# 0-43007 on the driver's GPU head slices). Gate on the reference prompt; greedy decode with and without drafts on
# real text, alternating, k = 3 and 2; the long prompt; the Responses server with --mtp (test-server.py).
# usage: chain6.sh mtp-chain2 EST "Flash-Next MTP agent" -- mtp-chain2.sh THREADS
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-ab5cf8f185c84b7fe/results/qwen4/chain-20261012 || exit 1
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
TH=24  # all logical CPUs: 8.2 ms per draft step against 11.1 with 12 (mtp-cpu6)
TEXT="--tg 192 --runs 3 $P/water.ids $P/csv.ids $P/transformer.ids"
run() {
    python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight \
        --binary trashcan-mtp --raw "$@" > /dev/null 2>&1
    echo "done $2 rc $?"
}
spec() {
    name=$1; shift
    run --name $name --driver "./trashcan-mtp qwen4 mtp spec $MODEL $C --mtp $MTP --draft-threads $TH $*"
}
spec mtp2-gate --draft-tokens 3 --tg 32
spec mtp2-k3 --draft-tokens 3 $TEXT
spec mtp2-k2 --draft-tokens 2 $TEXT
spec mtp2-long --draft-tokens 3 --tg 96 $P/long.ids
run --name mtp2-serve --env "BIN=./trashcan-mtp" --driver "bash scripts/q4serve-test.sh $MODEL results/mtp2-serve.log \
    --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk --mtp $MTP --draft-tokens 3"
echo "mtp-chain2 done"
