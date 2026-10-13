#!/bin/bash
# The [clocks] summary of every node log of a chain session (TAG prefix) on the five node Macs, and the driver's.
# usage: collect-clocks.sh TAG OUT
tag=$1 out=$2
: > "$out"
for h in indite inflight alight invite enlight; do
    ssh "$h" "cd ~/trashcan-exp/flashnext/results && for f in node-$tag-*-$h.log; do echo \"$h \$f \$(grep 'clocks\]' \$f | cut -c1-220)\"; done" >> "$out"
done
ssh incite "cd ~/trashcan-exp/flashnext/results && for f in $tag-*.err; do echo \"incite \$f \$(grep 'clocks\]' \$f | cut -c1-220)\"; done" >> "$out"
wc -l "$out"
