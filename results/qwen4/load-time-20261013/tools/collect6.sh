#!/bin/bash
# After tools/ab6.sh: copy the driver outputs (incite) and node logs (the other Macs) of the load-* runs into the
# results folder, and print each GPU's load time from the logs of the new binary.
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-adae52fc399671416/results/qwen4/load-time-20261013
mkdir -p $R/incite $R/nodes
rsync -a --include 'load-val14-*.json' --include 'load-val14-*.err' --include 'load-serve-*.err' \
    --include 'load-serve-*.json' --include 'load-serve-*-server.log' --exclude '*' \
    incite:trashcan-exp/flashnext/results/ $R/incite/
rsync -a --include 'load-*.txt' --exclude '*' incite:trashcan-exp/flashnext/results/clocks/ $R/incite/clocks/
for h in indite inflight alight invite enlight; do
    mkdir -p $R/nodes/$h/clocks
    rsync -a --include 'node-load-*.log' --exclude '*' "$h:trashcan-exp/flashnext/results/" $R/nodes/$h/
    rsync -a --include 'node-load-*.txt' --exclude '*' "$h:trashcan-exp/flashnext/results/clocks/" $R/nodes/$h/clocks/
done
echo "load times (new binary):"
grep -h 'allocated in' $R/incite/load-*-new-*.err $R/incite/load-serve-new-*-server.log 2>/dev/null | sed 's/^/incite /'
for h in indite inflight alight invite enlight; do
    for f in $R/nodes/$h/node-load-*-new-*.log; do
        echo "$h $(basename $f): $(grep -o 'allocated in [0-9.]* s' $f | tr '\n' ' ')"
    done
done
