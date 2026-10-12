#!/bin/bash
# Prefill overlap (trashcan-ovl, rebased on main 6d48a71) against trashcan-main (6d48a71): validation first (stop if
# an ovl run fails or its logits/greedy tokens differ from main's), then the real-text prefill A/B alternating, then
# the server test on trashcan-ovl. Run under chain6.sh (TICKETS exported).
REPO=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-adf3537cc78b259ba
S=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ovl
cd $REPO/results/qwen4/chain-20261012 || exit 1
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk"
W=/home/fairfieldt/trashcan-exp/flashnext
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
cleanup() {
    for h in incite indite inflight alight invite enlight; do
        ssh "$h" "pkill -KILL -f '^$W/trashcan-(ovl|main) (qwen4|serve)'"
    done
}
run() {
    timeout 900 python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" > /dev/null 2>&1
    rc=$?
    echo "rc $rc: $*" | cut -c1-170
    [ $rc -ne 0 ] && cleanup
    return $rc
}
# Same logits and greedy tokens as main, or stop.
same() {
    rsync -a "incite:trashcan-exp/flashnext/results/$1" "incite:trashcan-exp/flashnext/results/$2" $S/
    python3 $REPO/tools/compare_activations.py $S/$1 $S/$2 --all > $S/compare-$1-vs-$2.txt
    tail -2 $S/compare-$1-vs-$2.txt
    grep -q "^logits: rel 0.00e+00" $S/compare-$1-vs-$2.txt && cmp -s $S/$1/greedy.txt $S/$2/greedy.txt
}
run --name ovl-val14 --binary trashcan-ovl --driver "validate results/ovl-val14 $C --batch 128 --greedy 10" || exit 1
for b in 128 64; do
    run --name ovl-val300-b$b-main --binary trashcan-main --driver "validate results/ovl-val300-b$b-main $C --batch $b --greedy 10 --tokens-file $W/ovl-p300.ids"
    run --name ovl-val300-b$b --binary trashcan-ovl --driver "validate results/ovl-val300-b$b $C --batch $b --greedy 10 --tokens-file $W/ovl-p300.ids" || exit 1
    same ovl-val300-b$b ovl-val300-b$b-main || { echo "MISMATCH at batch $b"; exit 1; }
done
C="$C --batch 128"
for r in 1 2; do
    for pp in 2048 512; do
        tg=$([ $pp = 512 ] && echo 64 || echo 32)
        run --name ovl-pp$pp-main-$r --binary trashcan-main --driver "bench $C --pp $pp --tg $tg --tokens-file $W/long.ids"
        run --name ovl-pp$pp-ovl-$r --binary trashcan-ovl --driver "bench $C --pp $pp --tg $tg --tokens-file $W/long.ids"
    done
done
run --name ovl-pp3000-main --binary trashcan-main --driver "bench $C --pp 3000 --tg 32 --tokens-file $W/long.ids"
run --name ovl-pp3000-ovl --binary trashcan-ovl --driver "bench $C --pp 3000 --tg 32 --tokens-file $W/long.ids"
run --name ovl-serve --binary trashcan-ovl --raw --driver "env BIN=./trashcan-ovl scripts/q4serve-test.sh $M results/ovl-serve-server.log --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk"
echo "session done"
