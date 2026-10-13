#!/bin/bash
# Six-Mac session 1 of the verification-batch work (run under chain6.sh, TICKETS set): trashcan-vfy5 (exact-column
# Q8_0 kernels, qwen4_mix_down_skw for every batch size, qwen4_mix_rows_up) against main trashcan-4c1346d. First on
# invite alone: stage-bench and 96-token dumps in chunks of 1-4; then validation (the 14-token reference prompt as one
# chunk and in chunks of 3, p100), plain decode (3 alternated runs each), MTP on the chat prompts (one 3-run pass each).
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/results/qwen4/chain-20261012 || exit 1
B=trashcan-4c1346d V=trashcan-vfy5 V4=trashcan-vfy4
inv=$(echo "$TICKETS" | tr , '\n' | sed -n 's/^invite=//p')
ssh invite "cd ~/trashcan-exp/flashnext && gpuq run $inv -- ./vfy-sb.sh 64 16 8 vfy-s5-new-b3=$V:3 vfy-s5-base-b3=$B:3 vfy-s5-v4-b3=$V4:3 vfy-s5-new-b2=$V:2 vfy-s5-new-b4=$V:4" | cut -c1-80
ssh invite "cd ~/trashcan-exp/flashnext && gpuq run $inv -- ./vfy-dump.sh vfy-p96.ids vfy-s5-dnew-b1=$V:1 vfy-s5-dnew-b2=$V:2 vfy-s5-dnew-b3=$V:3 vfy-s5-dnew-b4=$V:4"
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
P100=/home/fairfieldt/trashcan-exp/flashnext/p100.ids
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
run() { python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
    > /dev/null 2>&1; echo "rc=$? $*" | cut -c1-110; }
run --name vfy-c1-val14 --binary $V --driver "validate results/vfy-c1-val14 $C --batch 128 --greedy 10"
run --name vfy-c1-val14b3 --binary $V --driver "validate results/vfy-c1-val14b3 $C --batch 3 --greedy 10"
run --name vfy-c1-val14b3-base --binary $B --driver "validate results/vfy-c1-val14b3-base $C --batch 3 --greedy 10"
run --name vfy-c1-val100 --binary $V --driver "validate results/vfy-c1-val100 $C --batch 128 --greedy 10 --tokens-file $P100"
run --name vfy-c1-val100-base --binary $B --driver "validate results/vfy-c1-val100-base $C --batch 128 --greedy 10 --tokens-file $P100"
for i in 1 2 3; do
    for bin in $B $V; do
        run --name vfy-c1-tg-$bin-$i --binary $bin --driver "bench $C --batch 128 --pp 512 --tg 128 --tokens-file $L"
    done
done
for bin in $B $V; do
    run --name vfy-c1-mtp-$bin --binary $bin --raw \
        --driver "./$bin qwen4 mtp spec $MODEL $C --batch 128 --mtp $MTP --draft-tokens 3 --tg 192 --runs 3 $P/water.ids $P/csv.ids $P/transformer.ids"
done
echo "vfy chain session 1 done"
