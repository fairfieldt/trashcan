#!/bin/bash
# Six-Mac session 2 of the verification-batch work (run under chain6.sh, TICKETS set; chain runs only), on main
# 4307169: trashcan-vfy10 (this branch) against trashcan-4307169: validation on the 14-token reference prompt in chunks
# of 4 and on p100 (both binaries), plain decode (one alternated pair, a no-regression check), MTP on the chat prompts
# (one --runs 1 pass each, alternated). DRY=1 prints the runs instead.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/results/qwen4/chain-20261012 || exit 1
B=trashcan-4307169 V=trashcan-vfy10
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
P100=/home/fairfieldt/trashcan-exp/flashnext/p100.ids
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
run() {
    if [ "${DRY:-0}" = 1 ]; then echo "run $*" | cut -c1-160; return; fi
    python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
        > /dev/null 2>&1; echo "rc=$? $*" | cut -c1-110
}
run --name vfy-c2-val14b4 --binary $V --driver "validate results/vfy-c2-val14b4 $C --batch 4 --greedy 10"
run --name vfy-c2-val14b4-base --binary $B --driver "validate results/vfy-c2-val14b4-base $C --batch 4 --greedy 10"
run --name vfy-c2-val100 --binary $V --driver "validate results/vfy-c2-val100 $C --batch 128 --greedy 10 --tokens-file $P100"
run --name vfy-c2-val100-base --binary $B --driver "validate results/vfy-c2-val100-base $C --batch 128 --greedy 10 --tokens-file $P100"
for bin in $B $V; do
    run --name vfy-c2-tg-$bin-1 --binary $bin --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $L"
done
for bin in $B $V; do
    run --name vfy-c2-mtp-$bin-1 --binary $bin --raw \
        --driver "./$bin qwen4 mtp spec $MODEL $C --batch 128 --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids"
done
echo "vfy chain session 2 done"
