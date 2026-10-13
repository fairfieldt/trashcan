#!/bin/bash
# Copy a six-Mac session's results: the driver's JSON and stderr, node logs and clock logs from every Mac into the
# results directory, the validation dumps (large) into the scratch directory.
# usage: gather.sh TAG        (e.g. c1)
tag=$1
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a7f9db84020875d15/results/qwen4/decode-experts-20261013
S=/Users/fairfieldt/dev/claude-work/flashnext-20261012/moe/$tag
mkdir -p "$S" "$R/incite"
rsync -a "incite:trashcan-exp/flashnext/results/moe-$tag-*.json" "incite:trashcan-exp/flashnext/results/moe-$tag-*.err" "$R/incite/"
for d in val14b1 val14 val100 val100-base; do
    rsync -a "incite:trashcan-exp/flashnext/results/moe-$tag-$d" "$S/" 2>/dev/null
done
for h in incite indite inflight alight invite enlight; do
    mkdir -p "$R/$h/clocks"
    rsync -a "$h:trashcan-exp/flashnext/results/node-moe-$tag-*" "$R/$h/" 2>/dev/null
    rsync -a "$h:trashcan-exp/flashnext/results/clocks/moe-$tag-*" "$h:trashcan-exp/flashnext/results/clocks/node-moe-$tag-*" \
        "$R/$h/clocks/" 2>/dev/null
done
rsync -a "incite:trashcan-exp/flashnext/results/commands.log" "$S/commands-all.log"
grep " moe-$tag-" "$S/commands-all.log" > "$R/incite/commands-$tag.log"
du -sh "$R" "$S"
