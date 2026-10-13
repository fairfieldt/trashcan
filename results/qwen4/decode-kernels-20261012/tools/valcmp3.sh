#!/bin/bash
# Final-session comparisons: val14 against the CPU reference, val100 against main's and the r2 binary's dumps.
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/dec/r3 || exit 1
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a90072f31ff42427f/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
echo "== dec-r3-val14 vs cpu-reference"; python3 "$C" dec-r3-val14 "$REF14" 2>&1 | grep -i "logits\|result_norm\|worst"
echo "== dec-r3-val100 vs main (dec-r2-val100-base)"; python3 "$C" dec-r3-val100 ../r2/dec-r2-val100-base 2>&1 | grep -i "logits\|worst"
echo "== dec-r3-val100 vs dec-r2-val100"; python3 "$C" dec-r3-val100 ../r2/dec-r2-val100 2>&1 | grep -i "logits\|worst"
echo "val14 greedy: $(tr '\n' ' ' < dec-r3-val14/greedy.txt)"
echo "val100 greedy: $(tr '\n' ' ' < dec-r3-val100/greedy.txt)"
