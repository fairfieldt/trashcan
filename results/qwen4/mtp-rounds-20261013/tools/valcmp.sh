#!/bin/bash
# Validation comparisons of a six-Mac session (validate runs PREFIX-val14, PREFIX-val100 of this branch and
# PREFIX-val100-base of main): logits and worst tensor against the CPU references, and the branch's p100 dump against
# main's (the plain path; expect byte-identical files). The dumps are copied from incite into the scratch directory.
# usage: valcmp.sh PREFIX   (mtp2s3: session 3, main 2293550; mtp2s4: session 4, main eb4dbeb, no val14 run)
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7
C=$W/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
REF100=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100
mkdir -p /Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/val3
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/val3 || exit 1
S=${1:-mtp2s3}
for d in $S-val14 $S-val100 $S-val100-base; do
    [ -d "$d" ] || ! ssh incite "test -d trashcan-exp/flashnext/results/$d" || rsync -a "incite:trashcan-exp/flashnext/results/$d/" "./$d/"
done
du -sh $S-*
cmp1() { echo "== $1 vs $3"; python3 "$C" "$1" "$2" 2>&1 | grep -i "logits\|worst\|result_norm"; }
[ -d $S-val14 ] && cmp1 $S-val14 $REF14 cpu-reference-14
cmp1 $S-val100 $REF100 cpu-reference-p100
cmp1 $S-val100-base $REF100 cpu-reference-p100
cmp1 $S-val100 $S-val100-base main-p100
n=0
s=0
for f in $S-val100/*; do
    b=$S-val100-base/$(basename "$f")
    n=$((n + 1))
    cmp -s "$f" "$b" && s=$((s + 1))
done
echo "byte-identical dump files, branch vs main p100: $s of $n (main has $(ls $S-val100-base | wc -l))"
