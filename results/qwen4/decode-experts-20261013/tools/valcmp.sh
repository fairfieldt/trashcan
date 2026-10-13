#!/bin/bash
# The six-Mac validation runs of a session (gather.sh first) against the CPU references and main's binary.
# usage: valcmp.sh TAG
tag=$1
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/moe/$tag || exit 1
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a7f9db84020875d15/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
REF100=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100
cmp1() { echo "== $1 vs $3"; python3 "$C" "$1" "$2" 2>&1 | grep -i "logits\|worst\|result_norm"; }
cmp1 moe-$tag-val14b1 $REF14 cpu-reference
cmp1 moe-$tag-val14 $REF14 cpu-reference
cmp1 moe-$tag-val100 $REF100 cpu-reference-p100
cmp1 moe-$tag-val100-base $REF100 cpu-reference-p100
cmp1 moe-$tag-val100 moe-$tag-val100-base baseline-p100
for d in moe-$tag-val14b1 moe-$tag-val14 moe-$tag-val100 moe-$tag-val100-base; do
    echo "$d greedy: $(tr '\n' ' ' < $d/greedy.txt)"
done
