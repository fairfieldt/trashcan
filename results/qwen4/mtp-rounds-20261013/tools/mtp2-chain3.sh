#!/bin/bash
# Six-Mac MTP rounds session 3 (through chain6.sh): the final check after rebasing on main 2293550 (small-batch
# experts, connect retries, load time, prefill), per the coordinator's testing policy: validation on the reference prompt and p100 (this branch, and main's
# p100 for a bit-for-bit comparison of the plain path), then one alternated MTP pair on the chat prompts (greedy,
# deterministic, --runs 1): main 2293550, this branch. DRY=1 prints the commands instead of running them.
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
B=trashcan-2293550
T=trashcan-mtp2
run --name mtp2s3-val14 --binary $T --driver "validate results/mtp2s3-val14 $C --greedy 10"
run --name mtp2s3-val100 --binary $T --driver "validate results/mtp2s3-val100 $C --greedy 10 --tokens-file $F/p100.ids"
run --name mtp2s3-val100-base --binary $B --driver "validate results/mtp2s3-val100-base $C --greedy 10 --tokens-file $F/p100.ids"
run --name mtp2s3-base-chat --binary $B --raw --driver "./$B qwen4 mtp spec $MODEL $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $CHAT"
run --name mtp2s3-chat --binary $T --raw --driver "./$T qwen4 mtp spec $MODEL $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $CHAT"
echo "mtp2-chain3 done"
