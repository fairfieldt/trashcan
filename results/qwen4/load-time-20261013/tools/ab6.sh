#!/bin/bash
# Six-Mac A/B of the load path (under chain6.sh, which exports TICKETS): whole `qwen4 validate` runs on the 14-token
# reference prompt, then `trashcan serve` start-up (tools/q4serve-test.sh with TURNS_ONLY=1: up, then the two-turn
# chat check), main 5e48d7e against trashcan-load (both with the q4run.py of 5e48d7e), alternating, 2 runs each. Then the outputs are compared.
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-adae52fc399671416
cd $W/results/qwen4/chain-20261012 || exit 1
LOG=$HOME/dev/claude-work/flashnext-20261012/load/ab6.log
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --batch 128 --ngram disk"
M=/home/fairfieldt/models/qwen3.8-flash-next/G-iq4/G-iq4-00001-of-00006.gguf
bin() { case $1 in main) echo trashcan-5e48d7e ;; new) echo trashcan-load ;; esac; }
summary=""
run() {
    local name=$1
    shift
    local s=$(date +%s.%N)
    echo "$(date +%T) start $name" >> $LOG
    python3 tools/q4run.py --tickets "$TICKETS" --driver-host incite --nodes indite,inflight,alight,invite,enlight \
        --name $name "$@" >> $LOG 2>&1
    local rc=$?
    local wall=$(printf %.1f $(echo "$(date +%s.%N) - $s" | bc))
    local driver=$(grep -o 'nodes up after [0-9.]* s; driver exited [0-9-]* after [0-9]* s' $LOG | tail -1)
    echo "$(date +%T) $name rc $rc wall $wall s ($driver)" | tee -a $LOG
    summary="$summary $name=${wall}s(rc$rc)"
}
for r in 1 2; do
    for v in main new; do
        run load-val14-$v-$r --binary $(bin $v) --driver "validate results/load-val14-$v-$r $C --greedy 10"
    done
done
for r in 1 2; do
    for v in main new; do
        run load-serve-$v-$r --binary $(bin $v) --raw --driver "env BIN=./$(bin $v) TURNS_ONLY=1 scripts/q4serve-test.sh $M results/load-serve-$v-$r-server.log --nodes $N --head-split 12 --context 16384 --batch 128 --ngram disk"
    done
done
check=$(ssh incite 'cd ~/trashcan-exp/flashnext/results
for r in 1 2; do
    n=0; d=0
    for f in load-val14-main-1/*.f32; do
        n=$((n + 1)); cmp -s $f load-val14-new-$r/$(basename $f) || d=$((d + 1))
    done
    echo "new-$r: $n tensors $d differ, greedy $(cmp -s load-val14-main-1/greedy.txt load-val14-new-$r/greedy.txt && echo same || echo DIFFERENT), logits $(cmp -s load-val14-main-1/logits.f32 load-val14-new-$r/logits.f32 && echo identical || echo DIFFERENT);"
done
echo "main-2 logits $(cmp -s load-val14-main-1/logits.f32 load-val14-main-2/logits.f32 && echo identical || echo DIFFERENT);"
for f in load-serve-*.err; do echo "$f: $(grep -o "server up after [0-9]* s" $f)"; done')
echo "$check" | tee -a $LOG
up=$(echo "$check" | grep -o 'load-serve-[a-z]*-[0-9].err: server up after [0-9]* s' | sed 's/load-serve-\([a-z]*-[0-9]\).err: server up after \([0-9]*\) s/\1=\2s/' | tr '\n' ' ')
echo "ab6 done:$summary; server up: $up; $(echo "$check" | grep -c 'logits identical') logits identical"
