#!/bin/bash
# Session 3 (through chain6.sh; DRY=1 prints the runs and checks binaries and inputs on all six Macs instead): the final
# tree f4 on main 4307169 against main (trashcan-4307169), and d4 (f4 without the CPU governor hold):
# validation (reference prompt and p100, bit for bit), base/f4 as two alternated pairs with d4 alongside, one traced
# run of f4, MTP one pair (--runs 1), the server test with an idle gap, the watchdog check (stall4: f4 with
# stall-test.patch), idle power windows.
cd "$(dirname "$0")/../../chain-20261012" || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
MTP=/home/fairfieldt/models/qwen3.8-flash-next/mtp/mtp-G-iq4.gguf
P=$W/mtp-prompts
B="bench $C --pp 512 --tg 128 --tokens-file $W/long.ids"
BASE=trashcan-4307169 F=trashcan-host-f4 D=trashcan-host-d4 S=trashcan-host-stall4
run() {
    local name=$1 bin=$2; shift 2
    if [ -n "${DRY:-}" ]; then echo "run $name $bin $*"; return; fi
    python3 tools/q4run.py --tickets "$TICKETS" --binary "$bin" --name "$name" --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1
    echo "$name rc $? $(ssh incite "grep -o '\"tg_tok_s\":[0-9.]*' trashcan-exp/flashnext/results/$name.json")"
}
gpuerr() {
    for h in incite indite inflight alight invite enlight; do
        echo "$h amdgpu timeouts/resets in dmesg: $(ssh $h "sudo -n dmesg | grep -c -i -E 'amdgpu.*(timeout|reset)|gpu reset'"); governors: $(ssh $h 'cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | sort | uniq -c | tr -s " " | tr "\n" " "')"
    done
}
same() {
    [ -n "${DRY:-}" ] && return
    ssh incite "cd trashcan-exp/flashnext/results && cat $2.json && cmp $1/greedy.txt $2/greedy.txt && cmp $1/logits.f32 $2/logits.f32 && echo '$2: greedy and logits identical to $1'"
}
window() { [ -n "${DRY:-}" ] && { echo "window $1 (${2:-30} s)"; return; }; echo "window $1 start $(date +%T)"; sleep "${2:-30}"; echo "window $1 end $(date +%T)"; }
if [ -n "${DRY:-}" ]; then
    for h in incite indite inflight alight invite enlight; do
        ssh "$h" "cd $W && ls $BASE $F $D $S $M ${M%00001-of-00006.gguf}0000{2,3,4,5,6}-of-00006.gguf > /dev/null && md5sum $BASE $F $D $S | cut -c1-12 | tr '\n' ' '" && echo " $h ok"
    done
    ssh incite "ls $W/long.ids $W/p100.ids $P/water.ids $P/csv.ids $P/transformer.ids $MTP $W/host-serve-idle.sh $W/scripts/q4-turns.py $W/scripts/test-server.py > /dev/null" && echo "incite inputs ok"
else
    gpuerr
fi
window idle-start
run host-s3-val-base $BASE --driver "validate results/host-s3-val-base $C --greedy 10"
run host-s3-val-f4 $F --driver "validate results/host-s3-val-f4 $C --greedy 10"
same host-s3-val-base host-s3-val-f4
run host-s3-val100-base $BASE --driver "validate results/host-s3-val100-base $C --greedy 10 --tokens-file $W/p100.ids"
run host-s3-val100-f4 $F --driver "validate results/host-s3-val100-f4 $C --greedy 10 --tokens-file $W/p100.ids"
same host-s3-val100-base host-s3-val100-f4
for r in 1 2; do
    run host-s3-base-$r $BASE --driver "$B"
    run host-s3-f4-$r $F --driver "$B"
    run host-s3-d4-$r $D --driver "$B"
done
run host-s3-f4-trace $F --env "TRASHCAN_Q4_TRACE=results/host-trace3f4" --driver "$B"
run host-s3-mtp-base $BASE --raw --driver "./$BASE qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids"
run host-s3-mtp-f4 $F --raw --driver "./$F qwen4 mtp spec $M $C --mtp $MTP --draft-tokens 3 --tg 192 --runs 1 $P/water.ids $P/csv.ids $P/transformer.ids"
run host-serve-f4 $F --raw --driver "env BIN=./$F ./host-serve-idle.sh $M results/host-serve-f4-server.log --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk"
[ -z "${DRY:-}" ] && ssh incite 'cd trashcan-exp/flashnext/results && grep -E "turn|PASS|FAIL|passed" host-serve-f4.json; grep -i -E "error|fail" host-serve-f4-server.log | head -5; grep window host-serve-f4.err'
# Watchdog: the driver stalls 3 s between the notice of a decode step and its forward; every node releases its gated
# work after 2 s and fails the session; no GPU may reset.
run host-stall $S --env "TRASHCAN_Q4_TEST_STALL=520:3000" --driver "bench $C --pp 512 --tg 16 --tokens-file $W/long.ids"
if [ -z "${DRY:-}" ]; then
    ssh incite 'tail -3 trashcan-exp/flashnext/results/host-stall.err'
    for h in indite inflight alight invite enlight; do
        echo "$h: $(ssh $h "grep -i -E 'watchdog|released|session ended' trashcan-exp/flashnext/results/node-host-stall-$h.log | head -2 | tr '\n' ' '")"
    done
fi
window idle-end
[ -z "${DRY:-}" ] && gpuerr
echo "session3 done"
