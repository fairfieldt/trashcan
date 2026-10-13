#!/bin/bash
# The reference prompt's greedy reply (it degenerates into a repeated token) with the drafter replayed along it.
set -uo pipefail
cd "$(dirname "$0")"
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7
M=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
X=/Volumes/Green/Qwen/flash-next/mtp/mtp-G-iq4.gguf
T=./trashcan-pasteos; export PAST_EOS=1
for t in 0 1.0; do
    $T qwen4 mtp accept $M $X run-reference-t$t.json --tokens-file reference.ids --steps 128 --depth 0 \
        --temperature $t --top-k 64 --top-p 0.95 --seed 1 > run-reference-t$t.log 2>&1
    $W/target/release/trashcan qwen4 mtp replay $M $X run-reference-t$t.json rep-reference-t$t.json --depth 3 --window 64 \
        --teacher >> run-reference-t$t.log 2>&1
    echo "done reference t=$t"
done
