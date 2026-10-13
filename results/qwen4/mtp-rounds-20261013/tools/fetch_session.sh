#!/bin/bash
# Copy a six-Mac session's results (driver JSON/err, clocks, node clock logs, commands) from the Macs into results.
# usage: fetch_session.sh DIR NAME...   (e.g. chain1 mtp2-gate mtp2-chat)
D=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7/results/qwen4/mtp-rounds-20261013/$1
shift
mkdir -p "$D"
for n in "$@"; do
    scp -q "incite:trashcan-exp/flashnext/results/$n.json" "incite:trashcan-exp/flashnext/results/$n.err" "$D/"
    scp -q "incite:trashcan-exp/flashnext/results/clocks/$n.txt" "$D/clocks-$n.txt" 2>/dev/null
    ssh incite "cd ~/trashcan-exp/flashnext && grep ' $n ' results/commands.log" >> "$D/commands.log"
    for h in indite inflight alight invite enlight; do
        scp -q "$h:trashcan-exp/flashnext/results/clocks/node-$n-$h.txt" "$D/clocks-node-$n-$h.txt" 2>/dev/null
    done
done
ls "$D" | head -50
