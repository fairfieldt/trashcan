#!/bin/bash
# Six-Mac MTP session 4 (through chain6.sh), on main with the decode kernels: adaptive rounds (as many
# drafts as the recent acceptance pays for, shadow drafts during plain steps) against plain decoding and against
# fixed 3-draft rounds. Gate prompt (low acceptance); real text; the long prompt; the server.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-ab5cf8f185c84b7fe/results/qwen4/chain-20261012 || exit 1
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
run() {
    python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight \
        --binary trashcan-mtp --raw "$@" > /dev/null 2>&1
    echo "done $2 rc $?"
}
spec() {
    name=$1; shift
    run --name $name --env "TRASHCAN_MTP_FIXED=1" --driver "./trashcan-mtp qwen4 mtp spec $MODEL $C --mtp $MTP $*"
}
spec mtp4-gate --draft-tokens 3 --tg 64 --runs 3
spec mtp4-k3 --draft-tokens 3 --tg 192 --runs 3 $P/water.ids $P/csv.ids $P/transformer.ids
spec mtp4-long --draft-tokens 3 --tg 96 --runs 2 $P/long.ids
run --name mtp4-serve --env "BIN=./trashcan-mtp" --driver "bash scripts/q4serve-test.sh $MODEL results/mtp4-serve.log \
    --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk --mtp $MTP --draft-tokens 3"
echo "mtp-chain4 done"
