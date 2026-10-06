#!/bin/bash
# Qwen3.8-27B on incite alone: a GPU node on localhost holds layers CUT..64 (both D700s, the head and the
# MTP layer) and the driver runs layers 0..CUT on the CPU. Run inside ~/trashcan-exp/locked-run.sh.
# usage: scripts/qwen-hybrid.sh CUT,MID COMMAND [ARGS...]   (COMMAND: qwen-check, qwen-chat, ...)
set -euo pipefail
cuts=$1 cmd=$2
shift 2
m=$HOME/models/qwen3.8-27b
bin/trashcan qwen-node $m/Qwen3.8-27B-trashcan-Q4_0.gguf --listen 127.0.0.1:7070 \
    --mtp $m/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf 2> results/hybrid-node.log &
node=$!
trap 'kill $node 2>/dev/null; wait $node 2>/dev/null' EXIT
until grep -q "on 127.0.0.1:7070" results/hybrid-node.log 2>/dev/null; do sleep 0.2; done
bin/trashcan "$cmd" $m/Qwen3.8-27B-trashcan-Q4_0.gguf --cpu --nodes 127.0.0.1:7070 --cuts "$cuts" \
    --mtp $m/mtp-Qwen3.8-27B-trashcan-Q4_0.gguf "$@"
