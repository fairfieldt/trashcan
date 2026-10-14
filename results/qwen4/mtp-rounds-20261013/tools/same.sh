#!/bin/bash
# The restructured drafter must give the same draft candidates as before: replay one sampled run (teacher-forced
# chains of 3, the prompt window of 64 rows) with the study binary (main's drafter) and the new one.
set -uo pipefail
cd "$(dirname "$0")"
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7
M=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
X=/Volumes/Green/Qwen/flash-next/mtp/mtp-G-iq4.gguf
S=/Users/fairfieldt/dev/claude-work/flashnext-20261012/mtp2/samp
cp $W/target/release/trashcan ./trashcan-new
for run in water-t1.0 transformer-t0.6; do
    ./trashcan-new qwen4 mtp replay $M $X $S/run-$run.json same-new-$run.json --depth 3 --window 64 --teacher > /dev/null 2>&1
    python3 -c "
import json
a = json.load(open('$S/rep-$run.json')); b = json.load(open('same-new-$run.json'))
print('$run', 'identical draft candidates:', a['draft_tops'] == b['draft_tops'], 'old %.1f s new %.1f s' % (a['draft_s'], b['draft_s']))
"
done
