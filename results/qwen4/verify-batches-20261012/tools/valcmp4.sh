#!/bin/bash
# Validation comparisons of six-Mac session 4 (the final tree on main 2293550): logits and worst tensor against the
# CPU references and main 2293550 in the same chunks, and the 14-token prompt against trashcan-vfy10 (session 2).
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/tools/compare_activations.py
REF14=/Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu
REF100=/Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy || exit 1
cmp1() { echo "== $1 vs $3"; python3 "$C" "$1" "$2" 2>&1 | grep -i "logits\|worst\|result_norm"; }
cmp1 c4/vfy-c4-val14b4 $REF14 cpu-reference-14
cmp1 c4/vfy-c4-val14b4-base $REF14 cpu-reference-14
cmp1 c4/vfy-c4-val14b4 c4/vfy-c4-val14b4-base main-2293550-14-chunks-of-4
cmp1 c4/vfy-c4-val14b4 c2/vfy-c2-val14b4 vfy10-14-chunks-of-4
cmp1 c4/vfy-c4-val100 $REF100 cpu-reference-p100
cmp1 c4/vfy-c4-val100-base $REF100 cpu-reference-p100
cmp1 c4/vfy-c4-val100 c4/vfy-c4-val100-base main-2293550-p100
for d in c4/vfy-c4-val14b4 c4/vfy-c4-val14b4-base c4/vfy-c4-val100 c4/vfy-c4-val100-base; do
    echo "$d greedy: $(cat $d.json)"
done
