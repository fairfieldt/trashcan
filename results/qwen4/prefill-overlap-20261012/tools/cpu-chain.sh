#!/bin/bash
# Local protocol test of trashcan-ovl: driver + two CPU nodes (G-iq4), head split over the three stages, prompt in
# chunks of BATCH (every chunk but the last streams on without a reply), dump + greedy.
# usage: cpu-chain.sh BIN NAME BATCH
set -uo pipefail
bin=$1 name=$2 batch=$3
d=$HOME/dev/claude-work/flashnext-20261012/ovl
m=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
"$bin" qwen4 node $m --listen 127.0.0.1:7192 --cpu > $d/$name-node2.log 2>&1 &
n2=$!
"$bin" qwen4 node $m --listen 127.0.0.1:7191 --cpu > $d/$name-node1.log 2>&1 &
n1=$!
trap 'kill $n1 $n2 2>/dev/null' EXIT
sleep 2
"$bin" qwen4 validate $m $d/$name --cpu --nodes 127.0.0.1:7191,127.0.0.1:7192 --cuts 16,32 --head-split 3 --context 256 --batch $batch --greedy 4 > $d/$name.json 2> $d/$name.err
echo "rc $?"
