#!/bin/bash
# Acceptance of drafts above temperature 0 on the CPU reference (Mac): sample 256 tokens of each chat prompt from the
# trunk at each temperature (top-k 64, top-p 0.95, seed 1), then replay the engine's drafter along each run with the
# run's own tokens, saving the trunk's and the drafter's top 64 candidates. sampling.py computes the acceptance.
set -uo pipefail
cd "$(dirname "$0")"
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7
M=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
X=/Volumes/Green/Qwen/flash-next/mtp/mtp-G-iq4.gguf
cp $W/target/release/trashcan ./trashcan-study
T=./trashcan-study
P=$W/results/qwen4/mtp-20261012/acceptance
one() {
    local t=$1
    for p in water csv transformer; do
        $T qwen4 mtp accept $M $X run-$p-t$t.json --tokens-file $P/$p.ids --steps 256 --depth 0 \
            --temperature $t --top-k 64 --top-p 0.95 --seed 1 > run-$p-t$t.log 2>&1
        $T qwen4 mtp replay $M $X run-$p-t$t.json rep-$p-t$t.json --depth 3 --window 64 --teacher >> run-$p-t$t.log 2>&1
        echo "done $p t=$t"
    done
}
for t in 0 0.6 0.8 1.0; do
    one $t &
done
wait
echo "study done"
