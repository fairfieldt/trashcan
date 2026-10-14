#!/bin/bash
# Session 7 (through chain6.sh; DRY=1 prints the runs and checks binaries and inputs on all six Macs instead): the
# candidate rebased onto main eb4dbeb (g3, with the trace flush) against main (trashcan-eb4dbeb): validation
# (reference prompt and p100, bit for bit), two untraced alternated decode pairs, then one traced g3 run (the timeline;
# not part of the A/B), MTP one pass each (ids identical to plain greedy decoding).
cd "$(dirname "$0")/../../chain-20261012" || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
P=$W/mtp-prompts
B="bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
BASE=trashcan-eb4dbeb G=trashcan-host-g3
run() {
    local name=$1 bin=$2; shift 2
    if [ -n "${DRY:-}" ]; then echo "run $name $bin $*"; return; fi
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$name" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1
    echo "$name rc $? $(ssh incite "grep -o '\"tg_tok_s\":[0-9.]*' trashcan-exp/flashnext/results/$name.json")"
}
gpuerr() {
    for h in incite indite inflight alight invite enlight; do
        echo "$h amdgpu timeouts/resets in dmesg: $(ssh -- "$h" "sudo -n dmesg | grep -c -i -E 'amdgpu.*(timeout|reset)|gpu reset'")"
    done
}
same() {
    [ -n "${DRY:-}" ] && return
    ssh incite "cd trashcan-exp/flashnext/results && cat $2.json && cmp $1/greedy.txt $2/greedy.txt && cmp $1/logits.f32 $2/logits.f32 && echo '$2: greedy and logits identical to $1'"
}
if [ -n "${DRY:-}" ]; then
    for h in incite indite inflight alight invite enlight; do
        ssh -- "$h" "cd $W && ls $BASE $G $M ${M%00001-of-00006.gguf}0000{2,3,4,5,6}-of-00006.gguf > /dev/null && md5sum $BASE $G | cut -c1-12 | tr '\n' ' '" && echo " $h ok"
    done
    ssh incite "ls $W/long.ids $W/p100.ids $P/water.ids $P/csv.ids $P/transformer.ids $MTP > /dev/null" && echo "incite inputs ok"
else
    gpuerr
fi
run host-s7-val-main $BASE --driver "validate results/host-s7-val-main $C --greedy 10"
run host-s7-val-g3 $G --driver "validate results/host-s7-val-g3 $C --greedy 10"
same host-s7-val-main host-s7-val-g3
run host-s7-val100-main $BASE --driver "validate results/host-s7-val100-main $C --greedy 10 --tokens-file $W/p100.ids"
run host-s7-val100-g3 $G --driver "validate results/host-s7-val100-g3 $C --greedy 10 --tokens-file $W/p100.ids"
same host-s7-val100-main host-s7-val100-g3
for r in 1 2; do
    run host-s7-main-$r $BASE --driver "$B"
    run host-s7-g3-$r $G --driver "$B"
done
run host-s7-g3-trace $G --env "TRASHCAN_Q4_TRACE=results/host-trace7g3" --driver "$B"
run host-s7-mtp-main $BASE --raw --driver "./$BASE qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids"
run host-s7-mtp-g3 $G --raw --driver "./$G qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids"
[ -z "${DRY:-}" ] && ssh incite 'cd trashcan-exp/flashnext/results && grep -c "\"identical\":true" host-s7-mtp-main.json host-s7-mtp-g3.json; ls -la host-trace7g3-*.json' && gpuerr
echo "session7 done"
