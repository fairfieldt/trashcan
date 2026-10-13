#!/bin/bash
# Session 4 (through chain6.sh): the final tree rebased on main 7da6f93 (BIN, default trashcan-host-f) against main
# (trashcan-7da6f93): validation (reference prompt and p100, bit for bit), alternating benches, a traced run, MTP on the
# chat prompts, the server test with an idle gap, idle power windows.
cd "$(dirname "$0")/../../chain-20261012" || exit 1
F=${BIN:-trashcan-host-f}
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
B="bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
run() {
    local name=$1 bin=$2; shift 2
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$name" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1
    echo "$name rc $? $(ssh incite "grep -o '\"tg_tok_s\":[0-9.]*' trashcan-exp/flashnext/results/$name.json")"
}
bin() { [ "$1" = base ] && echo trashcan-7da6f93 || echo "$F"; }
gpuerr() {
    for h in incite indite inflight alight invite enlight; do
        echo "$h amdgpu timeouts/resets in dmesg: $(ssh $h "sudo -n dmesg | grep -c -i -E 'amdgpu.*(timeout|reset)|gpu reset'"); governors: $(ssh $h 'cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort | uniq -c | tr -s " " | tr "\n" " "')"
    done
}
window() { echo "window $1 $(date +%T)"; }
same() {
    ssh incite "cd trashcan-exp/flashnext/results && cat $2.json && cmp $1/greedy.txt $2/greedy.txt && cmp $1/logits.f32 $2/logits.f32 && echo '$2: greedy and logits identical to $1'"
}
gpuerr
window "idle-start start"; sleep 30; window "idle-start end"
run host-s4-val-base trashcan-7da6f93 --driver "validate results/host-s4-val-base $C --greedy 10"
run host-s4-val-f $F --driver "validate results/host-s4-val-f $C --greedy 10"
same host-s4-val-base host-s4-val-f
run host-s4-val100-base trashcan-7da6f93 --driver "validate results/host-s4-val100-base $C --greedy 10 --tokens-file $W/p100.ids"
run host-s4-val100-f $F --driver "validate results/host-s4-val100-f $C --greedy 10 --tokens-file $W/p100.ids"
same host-s4-val100-base host-s4-val100-f
for r in 1 2 3 4; do
    for v in base f; do
        run host-s4-$v-$r $(bin $v) --driver "$B"
    done
done
run host-s4-f-trace $F --env "TRASHCAN_Q4_TRACE=results/host-trace4f" --driver "$B"
P=$W/mtp-prompts
for r in 1 2; do
    for v in base f; do
        run host-s4-mtp-$v-$r $(bin $v) --raw --driver "./$(bin $v) qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids"
    done
done
run host-serve-f $F --raw --driver "env BIN=./$F ./host-serve-idle.sh $M results/host-serve-f-server.log --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk"
ssh incite 'cd trashcan-exp/flashnext/results && grep -E "turn|PASS|FAIL" host-serve-f.json | head; grep -c PASS host-serve-f.json; grep -i -E "error|fail" host-serve-f-server.log | head -5; grep window host-serve-f.err'
window "idle-end start"; sleep 30; window "idle-end end"
gpuerr
echo "session4 done"
