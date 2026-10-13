#!/bin/bash
# List processes of this work's binaries (trashcan-vfy*) on all six Macs.
for h in incite indite inflight alight invite enlight; do
    echo "$h: $(ssh -- "$h" 'pgrep -fa "trashcan-exp/flashnext/trashcan-vfy|\./trashcan-vfy" | grep -v pgrep | cut -c1-100' | tr '\n' ' ')"
done
