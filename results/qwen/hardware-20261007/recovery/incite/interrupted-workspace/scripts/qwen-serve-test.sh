#!/bin/bash
# Start the Responses server on a Qwen model, run scripts/test-server.py against it, stop it.
# Run under a gpuq lease (`gpuq run ID -- ...`) from a workspace holding bin/trashcan and scripts/.
# usage: scripts/qwen-serve-test.sh MODEL LOG [EXTRA serve ARGS...]
set -euo pipefail
model=$1 log=$2
shift 2
bin/trashcan serve "$model" --port 8011 "$@" 2> "$log" &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:8011/health > /dev/null; do
    kill -0 $server 2>/dev/null || { cat "$log"; exit 1; }
    sleep 0.5
done
python3 scripts/test-server.py --url http://127.0.0.1:8011
