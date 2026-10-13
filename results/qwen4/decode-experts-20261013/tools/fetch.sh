#!/bin/bash
# Copy a session's results (JSON, stderr, dumps, clock logs) from a Mac into moe/SESSION.
# usage: fetch.sh HOST SESSION      (files named moe-SESSION-*)
set -euo pipefail
host=$1 s=$2
out=/Users/fairfieldt/dev/claude-work/flashnext-20261012/moe/$s
mkdir -p "$out/clocks"
rsync -a "${host}:trashcan-exp/flashnext/results/moe/moe-${s}-*" "$out/"
rsync -a "${host}:trashcan-exp/flashnext/results/clocks/moe-${s}-*" "$out/clocks/" 2>/dev/null || true
ls "$out" | head -50
