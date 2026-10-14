#!/bin/bash
# A six-Mac session's validation runs against the CPU references (tools/compare_activations.py: logits, top-10, greedy)
# and bit for bit against main's dumps of the same runs (results/qwen4/main-suite-20261013, ms-15e5457).
# usage: validation.sh SESSION_DIR TAG > SESSION_DIR/validation.txt
d=$1 tag=$2
here=$(dirname "$0")
repo=/Users/fairfieldt/dev/trashcan
main=$repo/results/qwen4/main-suite-20261013/ms-15e5457
echo "# $tag-val14 against the CPU reference (g-iq4-cpu)"
python3 $repo/tools/compare_activations.py "$d/$tag-val14" /Users/fairfieldt/dev/claude-work/trashcan-20261006/qwen-flash/ref/g-iq4-cpu 2>&1 | tail -3
echo "# $tag-val100 against the CPU reference (ref-p100)"
python3 $repo/tools/compare_activations.py "$d/$tag-val100" /Users/fairfieldt/dev/claude-work/flashnext-20261012/ref-p100 2>&1 | tail -3
echo "# bit for bit against main's dumps"
"$here/valcmp.sh" "$d/$tag-val14" "$main/ms-15e5457-val14" | tail -1
"$here/valcmp.sh" "$d/$tag-val100" "$main/ms-15e5457-val100" | tail -1
