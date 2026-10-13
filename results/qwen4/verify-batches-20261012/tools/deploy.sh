#!/bin/bash
# Copy a binary to ~/trashcan-exp/flashnext/NAME on all six Macs and print the hashes.
# usage: deploy.sh LOCAL_BINARY NAME
set -euo pipefail
src=$1 name=$2
for h in incite indite inflight alight invite enlight; do
    rsync -a "$src" "$h:trashcan-exp/flashnext/$name"
    echo "$h $(ssh "$h" "sha256sum ~/trashcan-exp/flashnext/$name" | cut -c1-16)"
done
