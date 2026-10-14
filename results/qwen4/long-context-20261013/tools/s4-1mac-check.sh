#!/bin/bash
# Fetch and check the results of s4-1mac.sh (no GPU use): dumps and stage-bench JSON into dumps/ and sb/ here, then
#  - r17 against r18 (f16 at 8196 and 30000, Q8 K/V at 8196): every tensor identical expected;
#  - main 4307169 against r19 at 8196 and 30000: layers 0-2 identical, the QSA layers' outputs within 1e-3;
#  - the fused-fma score leaves (r15f) against the replaced kernel: block scores bit for bit, selections;
#  - stage-bench summaries.
L=$HOME/dev/claude-work/flashnext-20261012/long
W=trashcan-exp/flashnext
cd "$L" || exit 1
mkdir -p dumps sb
rsync -a "inflight:$W/results/long-equiv-s4on-*" "inflight:$W/results/long-dump-s4-fma-*" dumps/ 2>/dev/null
rsync -a "alight:$W/results/long-equiv-s4bn-*" "alight:$W/results/long-equiv-s4oq8-*" dumps/ 2>/dev/null
for h in incite indite invite enlight; do rsync -a "$h:$W/results/long-sb-s4*" "$h:$W/results/clocks/long-sb-s4*" sb/ 2>/dev/null; done
have() { [ -e "$1/index.tsv" ] || { echo "MISSING $1"; return 1; }; }
for t in s4on:8196 s4on:30000 s4oq8:8196; do
    tag=${t%%:*} n=${t#*:} a=dumps/long-equiv-$tag-a-$n b=dumps/long-equiv-$tag-b-$n
    have $a && have $b || continue
    out=$(python3 dumpcmp.py $b $a)
    echo "== r18 vs r17 $tag $n: $(echo "$out" | grep -v identical | grep -vc ' 0 rows differ (0 as sets)') of $(echo "$out" | wc -l | tr -d ' ') tensors differ"
    echo "$out" | grep -v identical | head -8
done
for n in 8196 30000; do
    a=dumps/long-equiv-s4bn-a-$n b=dumps/long-equiv-s4bn-b-$n
    have $a && have $b || continue
    out=$(python3 dumpcmp.py $b $a)
    echo "== r19 vs main $n"; echo "$out" | grep -E 'attn_pregate|l_last-7|missing|length'
    echo "$out" | grep -E '^[^ ]*-[012](#2)? ' | grep -v identical | head -3 | sed 's/^/CHECK layers 0-2: /'
    echo "$out" | awk '/^(attn_pregate|l_last)/ && $3+0 > 1e-3 {print "CHECK output off: " $0}'
done
for n in 8196 30000; do
    a=dumps/long-dump-s4-fma-$n-new b=dumps/long-dump-s4-fma-$n-old
    have $a && have $b || continue
    echo "== fused fma leaves $n: $(python3 scorecmp.py $a $b $n)"
    python3 dumpcmp.py $a $b | grep indexer_sel
done
for f in sb/long-sb-s4*.json; do [ -e "$f" ] && python3 sbsum.py "$f" --disp 8; done
