#!/bin/bash
# Validation comparisons of six-Mac session 2: logits and worst tensor against the CPU references and main.
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
REF100=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy/c2 || exit 1
cmp1() { echo "== $1 vs $3"; python3 "$C" "$1" "$2" 2>&1 | grep -i "logits\|worst\|result_norm"; }
cmp1 vfy-c2-val14b4 $REF14 cpu-reference-14
cmp1 vfy-c2-val14b4-base $REF14 cpu-reference-14
cmp1 vfy-c2-val14b4 vfy-c2-val14b4-base main-14-chunks-of-4
cmp1 vfy-c2-val100 $REF100 cpu-reference-p100
cmp1 vfy-c2-val100-base $REF100 cpu-reference-p100
cmp1 vfy-c2-val100 vfy-c2-val100-base main-p100
for d in vfy-c2-val14b4 vfy-c2-val14b4-base vfy-c2-val100 vfy-c2-val100-base; do echo "$d greedy: $(cat $d.json)"; done
