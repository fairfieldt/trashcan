#!/bin/bash
# usage: node.sh NAME BINARY ARGS...  — start a qwen-tp-node in the background (clock log, 0 us CPU latency
# request, env passed through); stop it with: pkill -f "^[.]/trashcan[-a-z]* qwen-tp-node"
name=$1; bin=$2; shift 2
cd ~/trashcan-exp/decode
mkdir -p results/clocks
nohup scripts/with-clocks.sh results/clocks/$name.txt sudo -n --preserve-env=TRASHCAN_TP_PROFILE,TRASHCAN_TP_PROFILE_DISPATCH,TRASHCAN_TP_EVENTS,TRASHCAN_TP_FREE,TRASHCAN_TP_SUBMIT,TRASHCAN_QWEN_ORDER,TRASHCAN_Q4_DECODE,TRASHCAN_QWEN_ATTN,TRASHCAN_QWEN_NORM,TRASHCAN_TP_ARENA,TRASHCAN_TP_PARTS,TRASHCAN_QWEN_MERGE,TRASHCAN_Q4_MULTI,TRASHCAN_TP_HOSTNORM,TRASHCAN_QWEN_TOPK,TRASHCAN_TP_DRAFT_HEAD python3 scripts/latency-run.py $bin qwen-tp-node "$@" > results/$name.log 2>&1 < /dev/null &
sleep 1
echo started $name
