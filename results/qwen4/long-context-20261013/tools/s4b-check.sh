#!/bin/bash
# Fetch and check s4b-1mac.sh's results (no GPU use): r20 against r19 dumps (every tensor identical expected) and the
# stage-bench profiles of r20 and r19.
L=$HOME/dev/claude-work/flashnext-20261012/long
W=trashcan-exp/flashnext
cd "$L" || exit 1
rsync -a "alight:$W/results/long-equiv-s4t-*" "alight:$W/results/long-equiv-s4tq8-*" dumps/ 2>/dev/null
rsync -a "inflight:$W/results/long-sb-s4t*" "inflight:$W/results/clocks/long-sb-s4t*" sb/ 2>/dev/null
for t in s4t:8193 s4t:8196 s4t:30081 s4tq8:8196; do
    tag=${t%%:*} n=${t#*:} a=dumps/long-equiv-$tag-a-$n b=dumps/long-equiv-$tag-b-$n
    [ -e $a/index.tsv ] && [ -e $b/index.tsv ] || { echo "MISSING $a or $b"; continue; }
    out=$(python3 dumpcmp.py $b $a)
    echo "== r20 vs r19 $tag $n: $(echo "$out" | grep -v identical | grep -vc ' 0 rows differ (0 as sets)') of $(echo "$out" | wc -l | tr -d ' ') tensors differ"
    echo "$out" | grep -v identical | grep -v ' 0 rows differ (0 as sets)' | head -5
done
ls sb/long-sb-s4t20-*.json sb/long-sb-s4t19-*.json 2>/dev/null
[ -e sb/long-sb-s4t19-128.json ] && [ -e sb/long-sb-s4t20-128.json ] && \
    python3 sbtable.py 128=sb/long-sb-s4t19-128.json,sb/long-sb-s4t20-128.json 8320=sb/long-sb-s4t19-8320.json,sb/long-sb-s4t20-8320.json
