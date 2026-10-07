#!/bin/bash
# Push the workspace to a Mac Pro node (~/trashcan; default incite) and pull results back into results/.
# usage: [HOST=name] scripts/sync.sh [push|pull|both]   (default: push)
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-push}
host=${HOST:-incite}
if [ "$mode" = push ] || [ "$mode" = both ]; then
    ssh "$host" 'mkdir -p ~/trashcan/kernel'
    rsync -a --delete --exclude target --exclude /tools/dump-activations/dump --exclude /bench/xgpu/xgpu --exclude '/bench/xgpu/*.spv' Cargo.toml Cargo.lock README.md engine scripts bench tools tests docs "$host":trashcan/
    rsync -a kernel/build.sh kernel/build-7.2.sh kernel/configs kernel/patches kernel/patches-7.2 kernel/upstream "$host":trashcan/kernel/
    # The correctness target scripts/test-engine.sh compares against.
    ssh "$host" 'mkdir -p ~/trashcan/results'
    rsync -a results/reference-vulkan "$host":trashcan/results/
fi
if [ "$mode" = pull ] || [ "$mode" = both ]; then
    rsync -a "$host":trashcan/results/ results/
fi
