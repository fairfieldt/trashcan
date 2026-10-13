#!/bin/bash
# After six-Mac session 4: copy its outputs into the results directory and print the node checks, validation, MTP and
# clock tables.
set -uo pipefail
V=/Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/results/qwen4/verify-batches-20261012
mkdir -p "$V/c4"
rsync -a 'incite:trashcan-exp/flashnext/results/vfy-c4-*' "$V/c4/"
bash "$V/tools/gather.sh" vfy-c4 "$R/chain4"
echo "node logs with a bind error: $(grep -l 'Address already in use' "$R"/chain4/*/node-*.log | wc -l)"
echo "node logs that loaded their layers: $(grep -l 'allocated in' "$R"/chain4/*/node-*.log | wc -l) of $(ls "$R"/chain4/*/node-*.log | wc -l)"
bash "$V/tools/valcmp4.sh" > "$R/compare-chain4-validate.txt"
cat "$R/compare-chain4-validate.txt"
python3 "$R/../mtp-20261012/summary.py" "$R"/chain4/vfy-c4-mtp-*.json
python3 "$V/tools/clocksum.py" "$R"/chain4/*/clocks/* | cut -c1-150
