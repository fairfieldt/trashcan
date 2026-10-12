#!/usr/bin/env bash
# Six-Mac A/B for the parallel head slices (run under chain6.sh, which exports TICKETS):
# base = trashcan-c55128f, a = trashcan-head-a (h first on the reply path), b = trashcan-head (a + the ending stage
# scores its rows after h is out). Validation of all three, alternating benches, then the server test.
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a966aea2cf0b152d3
cd $W/results/qwen4/chain-20261012 || exit 1
LOG=$HOME/dev/claude-work/flashnext-20261012/head/ab-head.log
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
L=/home/fairfieldt/trashcan-exp/flashnext/long.ids
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
bin() { case $1 in base) echo trashcan-c55128f ;; a) echo trashcan-head-a ;; b) echo trashcan-head ;; esac; }
run() {
    echo "$(date +%T) start $*" >> $LOG
    python3 tools/q4run.py --tickets $TICKETS --driver-host incite --nodes indite,inflight,alight,invite,enlight "$@" >> $LOG 2>&1
    echo "$(date +%T) rc $? $*" | cut -c1-120 >> $LOG
}
for v in base a b; do
    run --name head-val-$v --binary $(bin $v) --driver "validate results/head-val-$v $C --greedy 10"
done
ok_b=1
ssh incite 'cd trashcan-exp/flashnext/results && cmp head-val-base/greedy.txt head-val-b/greedy.txt' >> $LOG 2>&1 || ok_b=0
ok_a=1
ssh incite 'cd trashcan-exp/flashnext/results && cmp head-val-base/greedy.txt head-val-a/greedy.txt' >> $LOG 2>&1 || ok_a=0
echo "greedy identical to base: a=$ok_a b=$ok_b" >> $LOG
ssh incite 'cd trashcan-exp/flashnext/results && for v in a b; do cmp head-val-base/logits.f32 head-val-$v/logits.f32 && echo "logits $v identical"; done' >> $LOG 2>&1
vs="base"
[ $ok_a = 1 ] && vs="$vs a"
[ $ok_b = 1 ] && vs="$vs b"
for r in 1 2 3; do
    for v in $vs; do
        run --name head-tg128-$v-$r --binary $(bin $v) --driver "bench $C --pp 512 --tg 128 --tokens-file $L"
    done
done
s=b
[ $ok_b = 1 ] || s=a
run --name head-serve-$s --binary $(bin $s) --raw --driver "env BIN=./$(bin $s) scripts/q4serve-test.sh $M results/head-serve-$s-server.log --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk"
echo "ab-head done: a=$ok_a b=$ok_b serve=$s"
