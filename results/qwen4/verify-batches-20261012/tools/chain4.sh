#!/bin/bash
# Six-Mac session 4 of the verification-batch work (run under chain6.sh, TICKETS set; chain runs only): session 3
# again (its vfy13 runs had connected to stray trashcan-2293550 nodes on four Macs, its base runs did not start).
# trashcan-vfy13 (this branch on main 2293550) and trashcan-2293550 validated on the 14-token reference prompt in
# chunks of 4 and on p100, then MTP on the chat prompts, one --runs 1 pass each, alternated. Every run first checks
# that no engine process of any binary is running on the six Macs, and afterwards that every node bound its port.
# DRY=1 prints the runs instead.
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/results/qwen4/chain-20261012 || exit 1
B=trashcan-2293550 V=trashcan-vfy13
MODEL=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
P100=/home/fairfieldt/trashcan-exp/flashnext/p100.ids
P=/home/fairfieldt/trashcan-exp/flashnext/mtp-prompts
NODES="indite inflight alight invite enlight"
clean() {
    local h busy=""
    for h in incite $NODES; do
        ssh -- "$h" "pgrep -f 'trashcan-exp/flashnext/trashcan-[^ ]* qwen4|^\./trashcan-[^ ]* qwen4' > /dev/null" && busy="$busy $h"
    done
    [ -z "$busy" ] || { echo "engine processes still running on:$busy; session stopped"; exit 1; }
}
run() {
    local name=$2
    if [ "${DRY:-0}" = 1 ]; then echo "run $*" | cut -c1-160; return; fi
    clean
    python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" \
        > /dev/null 2>&1
    local rc=$? h bad=""
    for h in $NODES; do
        ssh -- "$h" "grep -q 'Address already in use' ~/trashcan-exp/flashnext/results/node-$name-$h.log" && bad="$bad $h"
    done
    echo "rc=$rc${bad:+ BIND FAILED on$bad} $*" | cut -c1-130
}
run --name vfy-c4-val14b4 --binary $V --driver "validate results/vfy-c4-val14b4 $C --batch 4 --greedy 10"
run --name vfy-c4-val14b4-base --binary $B --driver "validate results/vfy-c4-val14b4-base $C --batch 4 --greedy 10"
run --name vfy-c4-val100 --binary $V --driver "validate results/vfy-c4-val100 $C --batch 128 --greedy 10 --tokens-file $P100"
run --name vfy-c4-val100-base --binary $B --driver "validate results/vfy-c4-val100-base $C --batch 128 --greedy 10 --tokens-file $P100"
for bin in $B $V; do
    run --name vfy-c4-mtp-$bin-1 --binary $bin --raw \
        --driver "./$bin qwen4 mtp spec $MODEL $C --batch 128 --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids"
done
echo "vfy chain session 4 done"
