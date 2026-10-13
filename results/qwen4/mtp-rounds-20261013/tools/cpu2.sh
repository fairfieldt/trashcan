#!/bin/bash
# Two-process CPU chain on the Mac (layers 0-23 on the driver, 24-47 and the head on a node over TCP): speculative
# decoding must return plain decoding's tokens, greedy and sampled.
set -uo pipefail
cd "$(dirname "$0")"
W=/Users/fairfieldt/dev/trashcan/.claude/worktrees/agent-a173c86cfe38f66c7
M=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
X=/Volumes/Green/Qwen/flash-next/mtp/mtp-G-iq4.gguf
cp $W/target/release/trashcan ./trashcan-cpu
T=./trashcan-cpu
$T qwen4 node $M --listen 127.0.0.1:7181 --cpu > node.log 2>&1 &
node=$!
trap 'kill $node 2>/dev/null; wait $node 2>/dev/null' EXIT
sleep 2
P=$W/results/qwen4/mtp-20261012/acceptance
for t in "${@:-0 1}"; do
    $T qwen4 mtp spec $M --cpu --nodes 127.0.0.1:7181 --cuts 24 --context 512 --batch 64 --mtp $X --draft-tokens 3 \
        --tg 24 --runs 2 --temperature $t --seed 11 $P/water.ids > spec-t$t.json 2> spec-t$t.err
    echo "t=$t rc $?"
    python3 -c "
import json
for l in open('spec-t$t.json'):
    r = json.loads(l); m = r['mtp']
    print(r['prompt'], r['sampling'], 'identical', m['identical'], 'accepted/draft %.2f' % m['accepted_per_draft'], 'by_k', m['stats']['by_k'], m['ids'][:12])
"
done
