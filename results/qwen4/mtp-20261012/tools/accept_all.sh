#!/bin/bash
# Acceptance of greedy MTP drafts on the CPU reference (M3 Ultra): G-iq4 trunk, MTP block at the six-Mac recipe.
# usage: accept_all.sh MTP_GGUF OUTDIR [STEPS]
T=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-ab5cf8f185c84b7fe/target/release/trashcan
G=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
P=/Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp/prompts
M=$1
O=$2
S=${3:-256}
mkdir -p "$O"
for p in water csv transformer long; do
    "$T" qwen4 mtp accept "$G" "$M" "$O/$p.json" --tokens-file "$P/$p.ids" --steps "$S" --depth 3 > "$O/$p.log" 2>&1
    echo "$p: $(tail -1 "$O/$p.log")"
done
