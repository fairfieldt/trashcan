#!/bin/bash
# Write the comparison tables of the final kernels into the results directory: one-Mac dumps of the 96-token prompt
# (session 9 on main 4307169, session 8 for chunks of 3 on main 7da6f93, session 7 for chunks of 2 with the K split
# behind its switch), main's own chunks against its single tokens (session 4, main 4c1346d), and the six-Mac
# validations of chain sessions 1, 2 and 4 (session 3's runs were invalid).
set -uo pipefail
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/results/qwen4/verify-batches-20261012
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/tools/compare_activations.py
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy || exit 1
rm -f "$R"/compare-p96-*.txt
python3 "$C" s9/vfy-s9-dnew-b1 s9/vfy-s9-dbase-b1 --all > "$R/compare-p96-chunks-b1-vs-main.txt" 2>/dev/null
python3 "$C" s9/vfy-s9-dnew-b4 s9/vfy-s9-dbase-b1 --all > "$R/compare-p96-chunks-b4-vs-main.txt" 2>/dev/null
python3 "$C" s9/vfy-s9-dnew-b4 s9/vfy-s9-dnew-b1 --all > "$R/compare-p96-chunks-b4-vs-chunks-b1.txt" 2>/dev/null
python3 "$C" s7/vfy-s7-dk2-b2 s7/vfy-s7-dk2-b1 --all > "$R/compare-p96-s7-chunks-b2-vs-chunks-b1.txt" 2>/dev/null
python3 "$C" s8/vfy-s8-dnew-b3 s8/vfy-s8-dbase-b1 --all > "$R/compare-p96-s8-chunks-b3-vs-main.txt" 2>/dev/null
python3 "$C" s8/vfy-s8-dnew-b3 s8/vfy-s8-dnew-b1 --all > "$R/compare-p96-s8-chunks-b3-vs-chunks-b1.txt" 2>/dev/null
for b in b2 b3 b4; do
    python3 "$C" s4/vfy-s4-dbase-$b s4/vfy-s4-dbase-b1 --all > "$R/compare-p96-main-chunks-$b-vs-main-chunks-b1.txt" 2>/dev/null
done
grep -H worst "$R"/compare-p96-*.txt | sed -e "s|$R/||"
tools/valcmp.sh /Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy/c1 vfy-c1 > "$R/compare-chain1-validate.txt"
tools/valcmp2.sh > "$R/compare-chain2-validate.txt"
tools/valcmp4.sh > "$R/compare-chain4-validate.txt"
grep -c "" "$R"/compare-chain*-validate.txt
