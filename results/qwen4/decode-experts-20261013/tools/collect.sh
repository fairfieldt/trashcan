#!/bin/bash
# Copy the single-Mac sessions' profiles (JSON, stderr, clock logs; not the dumps) into the results directory.
# usage: collect.sh SESSION...
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a7f9db84020875d15/results/qwen4/decode-experts-20261013
M=/Users/fairfieldt/dev/claude-work/flashnext-20261012/moe
for s in "$@"; do
    mkdir -p "$R/enlight/$s/clocks"
    for f in "$M/$s"/moe-*.json "$M/$s"/moe-*.err; do
        case $(basename "$f") in *dump*) continue ;; esac
        cp "$f" "$R/enlight/$s/"
    done
    cp "$M/$s"/clocks/*.txt "$R/enlight/$s/clocks/" 2>/dev/null
done
du -sh "$R/enlight"
