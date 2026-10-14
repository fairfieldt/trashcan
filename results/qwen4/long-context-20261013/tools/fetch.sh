#!/bin/bash
# Fetch a chain run's driver output (incite) and node logs into ~/dev/claude-work/flashnext-20261012/long/runs/NAME/.
# usage: fetch.sh NAME...
for name in "$@"; do
    d=$HOME/dev/claude-work/flashnext-20261012/long/runs/$name
    mkdir -p "$d"
    rsync -a "incite:trashcan-exp/flashnext/results/${name}.json" "incite:trashcan-exp/flashnext/results/${name}.err" "$d/" 2>/dev/null
    rsync -a "incite:trashcan-exp/flashnext/results/clocks/${name}.txt" "$d/clocks-incite.txt" 2>/dev/null
    for h in indite inflight alight invite enlight; do
        rsync -a "${h}:trashcan-exp/flashnext/results/node-${name}-${h}.log" "$d/" 2>/dev/null
        rsync -a "${h}:trashcan-exp/flashnext/results/clocks/node-${name}-${h}.txt" "$d/clocks-${h}.txt" 2>/dev/null
    done
    echo "$name: $(ls "$d" | wc -l) files"
done
