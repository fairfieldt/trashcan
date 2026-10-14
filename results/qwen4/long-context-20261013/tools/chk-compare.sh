#!/bin/bash
# Fetch long-chk3061.sh's dumps from MACHINE and compare (no GPU use): b (queued) against a2 (one chunk at a time, the
# same capacity) and a (capacity 64), every tensor identical expected; a against main 5e48d7e: layers 32-34 identical,
# the QSA layers (35, 39) and the stage output within 1e-3.
# usage: chk-compare.sh MACHINE TAG
L=$HOME/dev/claude-work/flashnext-20261012/long
m=$1 tag=$2
mkdir -p "$L/chk"
rsync -a "$m:trashcan-exp/flashnext/results/long-chk-$tag-*" "$L/chk/" || exit 1
d=$L/chk/long-chk-$tag
for x in a a2 b main; do [ -e "$d-$x/index.tsv" ] || echo "MISSING $d-$x"; done
cmpd() {
    [ -e "$1/index.tsv" ] && [ -e "$2/index.tsv" ] || return
    out=$(python3 "$L/dumpcmp.py" "$1" "$2")
    echo "== $3: $(echo "$out" | grep -v identical | grep -vc ' 0 rows differ (0 as sets)') of $(echo "$out" | wc -l | tr -d ' ') tensors differ"
    echo "$out" | grep -v identical | grep -v ' 0 rows differ (0 as sets)' | grep -E "${4:-.}" | head -12
}
cmpd "$d-b" "$d-a2" "queued (b) vs one at a time (a2)"
cmpd "$d-a2" "$d-a" "capacity 128 (a2) vs 64 (a)"
cmpd "$d-a" "$d-main" "r21 (a) vs main 5e48d7e" 'attn_pregate|l_last|indexer_sel|-3[234] '
grep -h 'clocks\]' "$L"/chk/*.log 2>/dev/null | head -0
for x in a a2 b main; do echo "$x: $(grep -h 'rror\|panick' "$d-$x.log" 2>/dev/null | head -2)"; done
