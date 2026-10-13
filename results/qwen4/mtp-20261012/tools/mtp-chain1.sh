#!/bin/bash
# Six-Mac MTP session 1 (run through chain6.sh, which exports TICKETS): correctness gate on the reference prompt,
# then greedy decode with and without drafts on real text, alternating, for drafter variants.
# usage: chain6.sh mtp-chain1 EST "Flash-Next MTP agent" -- mtp-chain1.sh
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-ab5cf8f185c84b7fe/results/qwen4/chain-20261012 || exit 1
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
TEXT="--tg 192 --runs 3 $P/water.ids $P/csv.ids $P/transformer.ids"
run() {
    python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight \
        --binary trashcan-mtp --raw "$@" > /dev/null 2>&1
    echo "done $2 rc $?"
}
spec() {
    name=$1 env=$2; shift 2
    run --name $name --env "$env" --driver "./trashcan-mtp qwen4 mtp spec $MODEL $C --mtp $MTP --draft-threads 12 $*"
}
spec mtp-gate "TRASHCAN_MTP_REDO=1" --draft-tokens 3 --tg 32
spec mtp-k3-q8-redo "TRASHCAN_MTP_REDO=1" --draft-tokens 3 $TEXT
spec mtp-k3-q4-noredo "TRASHCAN_MTP_REDO=0 TRASHCAN_DRAFT_Q4=head,dense" --draft-tokens 3 $TEXT
spec mtp-k2-q4-noredo "TRASHCAN_MTP_REDO=0 TRASHCAN_DRAFT_Q4=head,dense" --draft-tokens 2 $TEXT
spec mtp-k3-h2-noredo "TRASHCAN_MTP_REDO=0 TRASHCAN_DRAFT_Q4=head2,dense" --draft-tokens 3 --draft-vocab 98304 $TEXT
echo "mtp-chain1 done"
