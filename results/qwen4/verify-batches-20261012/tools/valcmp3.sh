#!/bin/bash
# Validation comparisons of six-Mac session 3 (the final tree on main 5e48d7e): logits and worst tensor against the CPU
# references, main 4307169 in the same chunks (session 2's base runs; 5e48d7e changed only the host connection code),
# and trashcan-vfy10 (session 2).
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
REF100=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy || exit 1
cmp1() { echo "== $1 vs $3"; python3 "$C" "$1" "$2" 2>&1 | grep -i "logits\|worst\|result_norm"; }
cmp1 c3/vfy-c3-val14b4 $REF14 cpu-reference-14
cmp1 c3/vfy-c3-val14b4 c2/vfy-c2-val14b4-base main-4307169-14-chunks-of-4
cmp1 c3/vfy-c3-val14b4 c2/vfy-c2-val14b4 vfy10-14-chunks-of-4
cmp1 c3/vfy-c3-val100 $REF100 cpu-reference-p100
cmp1 c3/vfy-c3-val100 c2/vfy-c2-val100-base main-4307169-p100
cmp1 c3/vfy-c3-val100 c2/vfy-c2-val100 vfy10-p100
for d in c3/vfy-c3-val14b4 c3/vfy-c3-val100; do echo "$d greedy: $(cat $d.json)"; done
