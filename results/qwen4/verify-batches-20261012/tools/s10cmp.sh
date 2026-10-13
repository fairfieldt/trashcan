#!/bin/bash
# Session 10 tables: walls and phases per run, the mixers of layers 5 and 7, and the dumps of trashcan-vfy11b/11a
# against session 9's trashcan-vfy10 dumps.
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy/s10 || exit 1
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/tools/compare_activations.py
R="vfy-s10-x-b3.json vfy-s10-n-b3.json vfy-s10-a-b3.json vfy-s10-x2-b3.json vfy-s10-n2-b3.json vfy-s10-a2-b3.json"
python3 ../tools/prof.py --nodisp $R | grep -v "^   layer"
for L in 5 7; do
    python3 ../tools/cmp.py --layer $L $R | python3 ../tools/compact.py | grep -E "layer|mix|mv|sum"
    python3 ../tools/cmp.py --chunk --layer $L $R | python3 ../tools/compact.py | grep -E "layer|mix|mv|sum"
done
echo "dn-b1 vs s9 dnew-b1: $(python3 "$C" vfy-s10-dn-b1 ../s9/vfy-s9-dnew-b1 --all 2>/dev/null | tail -1)"
echo "dn-b4 vs s9 dnew-b4: $(python3 "$C" vfy-s10-dn-b4 ../s9/vfy-s9-dnew-b4 --all 2>/dev/null | tail -1)"
echo "da-b1 vs s9 dnew-b1: $(python3 "$C" vfy-s10-da-b1 ../s9/vfy-s9-dnew-b1 --all 2>/dev/null | tail -1)"
