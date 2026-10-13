#!/bin/bash
# Six-Mac MTP rounds session 2 (through chain6.sh, binary 78e8848 rebased on main 6f6635a): this branch (coupled
# drafts, unsure drafts dropped, the two-rate policy) against TRASHCAN_MTP_AB=no-gate (every draft counted sure: main's
# pooled policy, every chosen draft verified) and against main's drafting (TRASHCAN_MTP_AB=greedy-drafts,no-gate) at
# temperatures 1.0, 0.8 and 0.6; main 6f6635a alongside for greedy decoding: chat prompts, the reference prompt, the
# 3K-token document; the server. Plain and speculative decoding alternate in every run; identical output is required.
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
B=trashcan-6f6635a
T=trashcan-mtp2
MAIN="TRASHCAN_MTP_AB=greedy-drafts,no-gate"
spec $T mtp2s2-chat "" --tg 192 --runs 3 $CHAT
spec $T mtp2s2-chat-nogate "TRASHCAN_MTP_AB=no-gate" --tg 192 --runs 3 $CHAT
spec $B mtp2s2-base-chat "" --tg 192 --runs 3 $CHAT
spec $T mtp2s2-t1.0 "" --tg 192 --runs 3 --temperature 1 --top-k 64 --top-p 0.95 --seed 11 $CHAT
spec $T mtp2s2-t1.0-main "$MAIN" --tg 192 --runs 3 --temperature 1 --top-k 64 --top-p 0.95 --seed 11 $CHAT
for t in 0.8 0.6; do
    spec $T mtp2s2-t$t "" --tg 192 --runs 2 --temperature $t --top-k 64 --top-p 0.95 --seed 11 $CHAT
    spec $T mtp2s2-t$t-main "$MAIN" --tg 192 --runs 2 --temperature $t --top-k 64 --top-p 0.95 --seed 11 $CHAT
done
spec $T mtp2s2-gate "" --tg 64 --runs 3
spec $B mtp2s2-base-gate "" --tg 64 --runs 3
spec $T mtp2s2-long "" --tg 96 --runs 2 $P/long.ids
spec $B mtp2s2-base-long "" --tg 96 --runs 2 $P/long.ids
run $T --name mtp2s2-serve --env "BIN=./$T" --driver "bash scripts/q4serve-test.sh $MODEL results/mtp2s2-serve.log \
    --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk --mtp $MTP --draft-tokens 3"
echo "mtp2-chain2 done"
