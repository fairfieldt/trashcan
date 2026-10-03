#!/bin/bash
# Push the workspace to incite (~/trashcan) and pull results back into results/.
# usage: scripts/sync.sh [push|pull|both]   (default: push)
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-push}
if [ "$mode" = push ] || [ "$mode" = both ]; then
    rsync -a --delete --exclude target Cargo.toml Cargo.lock engine scripts bench incite:trashcan/
    rsync -a kernel/build.sh kernel/patches incite:trashcan/kernel/
fi
if [ "$mode" = pull ] || [ "$mode" = both ]; then
    rsync -a incite:trashcan/results/ results/
fi
