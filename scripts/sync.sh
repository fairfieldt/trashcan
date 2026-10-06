#!/bin/bash
# Push the workspace to incite (~/trashcan) and pull results back into results/.
# usage: scripts/sync.sh [push|pull|both]   (default: push)
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-push}
if [ "$mode" = push ] || [ "$mode" = both ]; then
    rsync -a --delete --exclude target --exclude /tools/dump-activations/dump --exclude /bench/xgpu/xgpu --exclude '/bench/xgpu/*.spv' Cargo.toml Cargo.lock README.md engine scripts bench tools tests docs incite:trashcan/
    rsync -a kernel/build.sh kernel/build-7.2.sh kernel/configs kernel/patches kernel/patches-7.2 kernel/upstream incite:trashcan/kernel/
fi
if [ "$mode" = pull ] || [ "$mode" = both ]; then
    rsync -a incite:trashcan/results/ results/
fi
