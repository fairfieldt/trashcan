#!/bin/bash
# Confirmation comparisons on b5b79c7 (round r5).
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/dec/r5 || exit 1
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a90072f31ff42427f/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
REF100=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100
cmp1() { echo "== $1 vs $3"; python3 "$C" "$1" "$2" 2>&1 | grep -i "logits\|worst\|result_norm"; }
cmp1 dec-r5-val14 $REF14 cpu-reference
cmp1 dec-r5-val100 $REF100 cpu-reference-p100
cmp1 dec-r5-val100-base $REF100 cpu-reference-p100
cmp1 dec-r5-val100 dec-r5-val100-base baseline-p100
for d in dec-r5-val14 dec-r5-val100 dec-r5-val100-base; do echo "$d greedy: $(tr '\n' ' ' < $d/greedy.txt)"; done
