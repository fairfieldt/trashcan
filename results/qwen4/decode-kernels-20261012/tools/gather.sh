#!/bin/bash
# Copy the decode-kernel rounds' node logs and clock logs from every Mac into the results directory.
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a90072f31ff42427f/results/qwen4/decode-kernels-20261012
for h in incite indite inflight alight invite enlight; do
    mkdir -p "$R/$h/clocks"
    rsync -a "$h:trashcan-exp/flashnext/results/node-dec-r*" "$R/$h/" 2>/dev/null
    rsync -a "$h:trashcan-exp/flashnext/results/clocks/dec-*" "$h:trashcan-exp/flashnext/results/clocks/node-dec-*" "$R/$h/clocks/" 2>/dev/null
done
rsync -a "incite:trashcan-exp/flashnext/results/commands.log" "$R/incite/commands-all.log"
grep " dec-r" "$R/incite/commands-all.log" > "$R/incite/commands.log"; rm -f "$R/incite/commands-all.log"
du -sh "$R"
