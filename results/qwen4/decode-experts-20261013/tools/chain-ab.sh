#!/bin/bash
# Six-Mac session (run under chain6.sh, TICKETS set): validation of trashcan-moe3, then decode A/B against main's
# binary alternated, then MTP on the chat prompts with both binaries.
# usage: chain-ab.sh TAG [val] [tg] [mtp] [pp2048]
tag=$1; shift
B=trashcan-7da6f93 N=trashcan-moe3
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a7f9db84020875d15/results/qwen4/chain-20261012 || exit 1
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
NODES=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $NODES --head-split 12 --context 32768 --ngram disk"
F=/home/fairfieldt/trashcan-exp/flashnext
run() { python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
    > /dev/null 2>&1; echo "rc=$? $(echo "$@" | cut -c1-110)"; }
for what in "$@"; do case $what in
val)
    run --name moe-$tag-val14b1 --binary $N --driver "validate results/moe-$tag-val14b1 $C --batch 1 --greedy 10"
    run --name moe-$tag-val14 --binary $N --driver "validate results/moe-$tag-val14 $C --batch 128 --greedy 10"
    run --name moe-$tag-val100 --binary $N --driver "validate results/moe-$tag-val100 $C --batch 128 --greedy 10 --tokens-file $F/p100.ids"
    run --name moe-$tag-val100-base --binary $B --driver "validate results/moe-$tag-val100-base $C --batch 128 --greedy 10 --tokens-file $F/p100.ids"
    ;;
tg)
    for i in 1 2 3; do
        run --name moe-$tag-tg-base-$i --binary $B --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $F/long.ids"
        run --name moe-$tag-tg-new-$i --binary $N --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $F/long.ids"
    done ;;
pp2048)
    run --name moe-$tag-pp2048-base --binary $B --driver "bench $C --batch 128 --pp 2048 --tg 32 --tokens-file $F/long.ids"
    run --name moe-$tag-pp2048-new --binary $N --driver "bench $C --batch 128 --pp 2048 --tg 32 --tokens-file $F/long.ids"
    ;;
mtp)
    for b in $B $N; do
        run --name moe-$tag-mtp-$b --binary $b --raw --driver "./$b qwen4 mtp spec $MODEL $C --batch 128 --mtp $MTP --draft-tokens 3 --tg 192 --runs 3 $F/mtp-prompts/water.ids $F/mtp-prompts/csv.ids $F/mtp-prompts/transformer.ids"
    done ;;
esac; done
echo "moe chain $tag done"
