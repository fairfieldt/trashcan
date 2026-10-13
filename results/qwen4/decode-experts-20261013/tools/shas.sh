#!/bin/bash
# Print the sha256 prefix of a binary in ~/trashcan-exp/flashnext on every Mac.
# usage: shas.sh [NAME]   (default trashcan-moe)
name=${1:-trashcan-moe}
for h in incite indite inflight alight invite enlight; do
    echo "$h $(ssh -- "$h" "sha256sum ~/trashcan-exp/flashnext/$name" | cut -c1-16)"
done
