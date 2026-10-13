#!/bin/bash
# Copy a six-Mac session's outputs (driver JSON/err on incite, node logs and clock logs from every Mac) into the results
# directory, without the activation dumps.
# usage: gather.sh PREFIX DEST   (e.g. vfy-c1 .../results/qwen4/verify-batches-20261012/chain1)
set -uo pipefail
pre=$1 dest=$2
mkdir -p "$dest"
rsync -a --include="$pre-*.json" --include="$pre-*.err" --exclude='*' incite:trashcan-exp/flashnext/results/ "$dest/"
for h in incite indite inflight alight invite enlight; do
    mkdir -p "$dest/$h/clocks"
    rsync -a --include="node-$pre-*.log" --exclude='*' "$h:trashcan-exp/flashnext/results/" "$dest/$h/" 2>/dev/null
    rsync -a --include="*$pre-*.txt" --exclude='*' "$h:trashcan-exp/flashnext/results/clocks/" "$dest/$h/clocks/" 2>/dev/null
done
ssh incite "grep ' $pre-' ~/trashcan-exp/flashnext/results/commands.log" > "$dest/commands.log"
find "$dest" -type f | wc -l
