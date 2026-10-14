#!/bin/bash
# Session "sv" (under chain6.sh, watched): the six-Mac 3061-token validation that hung invite in long-s4, first with
# r21s (r21 with main's order in Local::step: one chunk per GPU at a time), then with r21 (chunks queued two deep), then
# main 5e48d7e for the comparison. Stops at the first failed run. DRY=1: check binaries and inputs, run nothing.
set -u
cd /Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/results/qwen4/chain-20261012 || exit 1
L=$HOME/dev/claude-work/flashnext-20261012/long
CMP=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a43eb2477e990f76f/tools/compare_activations.py
N=indite.tb0:7071,inflight.tb1:7071,alight.tb0:7071,invite.tb1:7071,enlight.tb0:7071
W=/home/fairfieldt/trashcan-exp/flashnext
C="--gpus 0,1 --nodes $N --head-split 12 --context 32768 --ngram disk --batch 128"
if [ "${DRY:-0}" = 1 ]; then
    for h in incite indite inflight alight invite enlight; do
        echo "== $h: $(ssh "$h" "cd $W && md5sum trashcan-long-r21s trashcan-long-r21 trashcan-5e48d7e 2>&1 | cut -c1-8 | tr '\n' ' '")"
    done
    ssh incite "ls -la $W/long.ids" | cut -c30-
    exit 0
fi
for b in r21s r21 base; do
    bin=trashcan-long-$b; [ $b = base ] && bin=trashcan-5e48d7e
    name=long-sv-val3061-$b
    python3 tools/q4run.py --tickets "$TICKETS" --binary $bin --name $name --driver-host incite \
        --nodes indite,inflight,alight,invite,enlight --driver "validate results/$name $C --greedy 10 --tokens-file $W/long.ids" \
        > /dev/null 2>&1
    rc=$?
    echo "$name rc=$rc $(date +%H:%M:%S) $(ssh incite "head -c 160 $W/results/$name.json")"
    [ $rc = 0 ] || { echo "stopping after a failed run"; exit 1; }
done
mkdir -p "$L/val"
rsync -a "incite:$W/results/long-sv-val3061-*" "$L/val/"
for b in r21s r21; do
    echo "== $b vs main"; python3 "$CMP" "$L/val/long-sv-val3061-$b" "$L/val/long-sv-val3061-base" 2>&1 | grep -i "logits"
    cmp -s "$L/val/long-sv-val3061-$b/greedy.txt" "$L/val/long-sv-val3061-base/greedy.txt" \
        && echo "$b: greedy tokens identical to main" || echo "$b: greedy tokens differ from main"
done
cmp -s "$L/val/long-sv-val3061-r21s/logits.f32" "$L/val/long-sv-val3061-r21/logits.f32" && echo "r21s and r21 logits identical"
echo "sv done"
