#!/bin/bash
# Copy a six-Mac session's results (driver JSON/err, clocks, node logs, commands) from incite into results.
# usage: fetch_session.sh PREFIX DIR   (e.g. mtp3 chain3)
P=$1
D=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-ab5cf8f185c84b7fe/results/qwen4/mtp-20261012/$2
mkdir -p "$D"
scp -q "incite:trashcan-exp/flashnext/results/$P-*" "$D/"
scp -q "incite:trashcan-exp/flashnext/results/clocks/$P-*" "$D/"
ssh incite "cd ~/trashcan-exp/flashnext && grep ' $P-' results/commands.log" > "$D/commands.log"
for h in indite inflight alight invite enlight; do
    scp -q "$h:trashcan-exp/flashnext/results/clocks/node-$P-*" "$D/" 2>/dev/null
done
ls "$D"
