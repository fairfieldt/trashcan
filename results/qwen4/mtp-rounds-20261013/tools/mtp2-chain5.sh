#!/bin/bash
# Six-Mac MTP rounds session 5 (through chain6.sh): session 4's MTP pair again (main eb4dbeb, then this branch rebased
# on it; chat prompts, greedy, --runs 1) after invite's GPU0 was reset (it ran at 450 MHz through session 4).
# DRY=1 prints the commands instead of running them.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7/results/qwen4/chain-20261012 || exit 1
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
F=/home/fairfieldt/trashcan-exp/flashnext
CHAT="$F/mtp-prompts/water.ids $F/mtp-prompts/csv.ids $F/mtp-prompts/transformer.ids"
run() {
    if [ -n "${DRY:-}" ]; then
        echo "q4run $*"
        return
    fi
    python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight \
        "$@" > /dev/null 2>&1
    echo "done $2 rc $?"
}
B=trashcan-eb4dbeb
T=trashcan-mtp2
run --name mtp2s5-base-chat --binary $B --raw --driver "./$B qwen4 mtp spec $MODEL $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $CHAT"
run --name mtp2s5-chat --binary $T --raw --driver "./$T qwen4 mtp spec $MODEL $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $CHAT"
echo "mtp2-chain5 done"
