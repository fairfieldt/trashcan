#!/bin/bash
# Teacher-forced replays of the greedy chat runs with chains of 7 drafts: the acceptance of drafts 4-7 of a chain,
# which drafting the next round during a verification would verify (rounds of 3 drafts, the 4th the predicted bonus).
set -uo pipefail
M=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
X=/Volumes/Green/Qwen/flash-next/mtp/mtp-G-iq4.gguf
D=/Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/pipe
cd /Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/samp || exit 1
for p in water csv transformer; do
    $D/trashcan-pipe qwen4 mtp replay $M $X run-$p-t0.json $D/rep7-$p-t0.json --depth 7 --window 64 --teacher > $D/rep7-$p.log 2>&1
    echo "done $p rc $?"
done
