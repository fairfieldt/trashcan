#!/bin/bash
# The other Mac's ranks over the three xmac streams, in the background (start it inside a gpuq lease on invite),
# with a clock log and the 0 us CPU latency request. Stop it with: pkill -f "^[.]/trashcan[-a-z]* qwen-tp-node"
# usage: [ENV="VAR=1 ..."] [BIN=./trashcan] node.sh NAME MODEL [ARGS...]   (ARGS e.g. --mtp FILE)
set -uo pipefail
cd ~/trashcan-exp/merged
name=$1; shift
bin=${BIN:-./trashcan}
mkdir -p results/clocks
echo "$(date -Is) $(hostname) node $name: ${ENV:-} $bin $*" >> results/commands.log
nohup scripts/with-clocks.sh results/clocks/$name.txt sudo -n env ${ENV:-} python3 scripts/latency-run.py \
    $bin qwen-tp-node "$@" --stream /dev/tbstream0,/dev/tbstream1,/dev/tbstream2 > results/$name.log 2>&1 < /dev/null &
sleep 1
echo "started node $name"
