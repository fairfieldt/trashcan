#!/bin/bash
# Validation comparisons of a six-Mac session: logits and worst tensor against the CPU references and the baseline.
# usage: valcmp.sh DIR PREFIX (e.g. c1 vfy-c1)
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
REF100=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100
cd "$1" || exit 1
p=$2
cmp1() { echo "== $1 vs $3"; python3 "$C" "$1" "$2" 2>&1 | grep -i "logits\|worst\|result_norm\|top"; }
cmp1 $p-val14 $REF14 cpu-reference-14
cmp1 $p-val14b3 $REF14 cpu-reference-14
cmp1 $p-val14b3-base $REF14 cpu-reference-14
cmp1 $p-val14b3 $p-val14b3-base baseline-14-chunks-of-3
cmp1 $p-val100 $REF100 cpu-reference-p100
cmp1 $p-val100-base $REF100 cpu-reference-p100
cmp1 $p-val100 $p-val100-base baseline-p100
