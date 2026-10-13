#!/bin/bash
# Pairwise dump comparisons (worst relative error line, or with -f the full table into OUT/compare-A-vs-B.txt).
# usage: cmpdumps.sh DIR [-o OUTDIR] A:B ...
set -uo pipefail
C=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/tools/compare_activations.py
cd "$1"; shift
out=""
if [ "${1:-}" = -o ]; then out=$2; shift 2; mkdir -p "$out"; fi
for pair in "$@"; do
    a=${pair%%:*} b=${pair#*:}
    if [ -n "$out" ]; then
        python3 "$C" "$a" "$b" --all > "$out/compare-$a-vs-$b.txt" 2>&1
        echo "$a vs $b: $(grep worst "$out/compare-$a-vs-$b.txt")"
    else
        echo "$a vs $b: $(python3 "$C" "$a" "$b" --all 2>/dev/null | tail -1)"
    fi
done
