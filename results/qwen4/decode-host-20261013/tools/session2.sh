#!/bin/bash
# Session 2 (through chain6.sh): variant a (stage and head graphs submitted ahead, gated on host events; done events
# polled) against main (4c1346d): validation, alternating benches, one traced run, the server test with an idle gap.
cd "$(dirname "$0")/../../chain-20261012" || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
B="bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run() {
    local name=$1 bin=$2; shift 2
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$name" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1
    echo "$name rc $? $(ssh incite "grep -o '\"tg_tok_s\":[0-9.]*' trashcan-exp/flashnext/results/$name.json")"
}
gpuerr() {
    for h in incite indite inflight alight invite enlight; do
        echo "$h amdgpu timeouts/resets in dmesg: $(ssh $h "sudo -n dmesg | grep -c -i -E 'amdgpu.*(timeout|reset)|gpu reset'")"
    done
}
gpuerr
run host-val-base trashcan-4c1346d --driver "validate results/host-val-base $C --greedy 10"
run host-val-a trashcan-host-a --driver "validate results/host-val-a $C --greedy 10"
ssh incite 'cd trashcan-exp/flashnext/results && cat host-val-a.json && cmp host-val-base/greedy.txt host-val-a/greedy.txt && cmp host-val-base/logits.f32 host-val-a/logits.f32 && echo "a: greedy and logits identical to base"'
for r in 1 2 3; do
    run host-s2-base-$r trashcan-4c1346d --driver "$B"
    run host-s2-a-$r trashcan-host-a --driver "$B"
done
run host-s2-a-trace trashcan-host-a --env "TRASHCAN_Q4_TRACE=results/host-trace2" --driver "$B"
run host-serve-a trashcan-host-a --raw --driver "env BIN=./trashcan-host-a ./host-serve-idle.sh $M results/host-serve-a-server.log --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk"
ssh incite 'cd trashcan-exp/flashnext/results && tail -4 host-serve-a.json; grep -c PASS host-serve-a.json'
gpuerr
echo "session2 done"
