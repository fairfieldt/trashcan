#!/bin/bash
# Six-Mac MTP rounds session 1 (through chain6.sh): main 4c1346d against this branch, greedy (the reference prompt,
# chat prompts, the 3K-token document), the confidence gate against no gate (TRASHCAN_MTP_AB=no-gate), sampling at
# temperatures 1.0, 0.6 and 0.8 (top-k 64, top-p 0.95) with this branch's drafting against main's (greedy drafts, no
# gate: TRASHCAN_MTP_AB=greedy-drafts,no-gate), 30 seeds at temperature 1 on the reference prompt, and the server.
# Plain and speculative decoding alternate in every run; identical output is required.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7/results/qwen4/chain-20261012 || exit 1
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
CHAT="$P/water.ids $P/csv.ids $P/transformer.ids"
run() {
    local bin=$1; shift
    python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight \
        --binary $bin --raw "$@" > /dev/null 2>&1
    echo "done $2 rc $?"
}
spec() {
    local bin=$1 name=$2 env=$3; shift 3
    run $bin --name $name --env "$env" --driver "./$bin qwen4 mtp spec $MODEL $C --mtp $MTP --draft-tokens 3 $*"
}
B=trashcan-4c1346d
T=trashcan-mtp2
S="--top-k 64 --top-p 0.95 --seed 1"
spec $B mtp2-base-gate "" --tg 64 --runs 3
spec $T mtp2-gate "" --tg 64 --runs 3
spec $B mtp2-base-chat "" --tg 192 --runs 3 $CHAT
spec $T mtp2-chat "" --tg 192 --runs 3 $CHAT
spec $T mtp2-chat-nogate "TRASHCAN_MTP_AB=no-gate" --tg 192 --runs 3 $CHAT
for t in 1.0 0.6 0.8; do
    spec $T mtp2-t$t "" --tg 192 --runs 2 --temperature $t $S $CHAT
    spec $T mtp2-t$t-main "TRASHCAN_MTP_AB=greedy-drafts,no-gate" --tg 192 --runs 2 --temperature $t $S $CHAT
done
spec $T mtp2-seeds "" --tg 24 --runs 30 --temperature 1 --top-k 64 --top-p 0.95 --seed 100
spec $B mtp2-base-long "" --tg 96 --runs 2 $P/long.ids
spec $T mtp2-long "" --tg 96 --runs 2 $P/long.ids
run $T --name mtp2-serve --env "BIN=./$T" --driver "bash scripts/q4serve-test.sh $MODEL results/mtp2-serve.log \
    --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk --mtp $MTP --draft-tokens 3"
echo "mtp2-chain1 done"
