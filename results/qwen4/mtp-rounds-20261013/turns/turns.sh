#!/bin/bash
# The block's context in a conversation's second turn: the turn-2 prompt (turn 1's prompt and greedy reply, then a
# follow-up question) decoded greedily on the CPU reference, then the engine's drafter in 3-draft rounds with the last
# W prompt tokens as its context (W = 2: what a continuation's 1-token last chunk leaves it; 16 and 64: a last chunk of
# that size; 128: most of the conversation).
set -uo pipefail
cd "$(dirname "$0")"
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7
M=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
X=/Volumes/Green/Qwen/flash-next/mtp/mtp-G-iq4.gguf
T=$W/target/release/trashcan
$T qwen4 mtp accept $M $X turn2.json --tokens-file turn2.ids --steps 128 --depth 0 > turn2.log 2>&1
for w in 2 8 16 32 64 128; do
    $T qwen4 mtp replay $M $X turn2.json rounds-w$w.json --depth 3 --window $w --rounds 2>> turn2.log | tail -1
done
