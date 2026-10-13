#!/bin/bash
# After six-Mac session 3: copy its outputs into the results directory and print the validation, MTP and clock tables.
set -uo pipefail
V=/Users/fairfieldt/dev/claude-work/flashnext-20261012/vfy
R=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a39bc1d57bef46fec/results/qwen4/verify-batches-20261012
mkdir -p "$V/c3"
rsync -a 'incite:trashcan-exp/flashnext/results/vfy-c3-*' "$V/c3/"
bash "$V/tools/gather.sh" vfy-c3 "$R/chain3"
bash "$V/tools/valcmp3.sh" > "$R/compare-chain3-validate.txt"
cat "$R/compare-chain3-validate.txt"
python3 "$R/../mtp-20261012/summary.py" "$R"/chain3/vfy-c3-mtp-*.json
python3 "$V/tools/clocksum.py" "$R"/chain3/*/clocks/* | cut -c1-170
