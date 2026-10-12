#!/bin/bash
# Local protocol test: driver + two CPU nodes (G-iq4), head split over the three stages, dump + greedy.
# usage: cpu-chain.sh BIN NAME
set -uo pipefail
bin=$1 name=$2
d=$HOME/dev/claude-work/flashnext-20261012/head
m=/Volumes/Green/Qwen/flash-next/cand/G-iq4/G-iq4-00001-of-00006.gguf
"$bin" qwen4 node $m --listen 127.0.0.1:7182 --cpu > $d/$name-node2.log 2>&1 &
n2=$!
"$bin" qwen4 node $m --listen 127.0.0.1:7181 --cpu > $d/$name-node1.log 2>&1 &
n1=$!
trap 'kill $n1 $n2 2>/dev/null' EXIT
sleep 2
time "$bin" qwen4 validate $m $d/$name --cpu --nodes 127.0.0.1:7181,127.0.0.1:7182 --cuts 16,32 --head-split 3 --context 256 --batch 16 --greedy 4 > $d/$name.json 2> $d/$name.err
echo "rc $?"
