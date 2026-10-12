#!/bin/bash
# Start the Responses server as the chain's driver, run scripts/test-server.py and tools/q4-turns.py against it, stop it.
# Run from the workspace (trashcan, scripts/) with the nodes already listening.
# usage: q4serve-test.sh MODEL LOG [serve ARGS...]
set -uo pipefail
model=$1 log=$2
shift 2
"${BIN:-./trashcan}" serve "$model" --port 8011 "$@" 2> "$log" &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:8011/health > /dev/null; do
    kill -0 $server 2>/dev/null || { tail "$log"; exit 1; }
    sleep 1
done
echo "server up after $SECONDS s" >&2
python3 scripts/q4-turns.py --url http://127.0.0.1:8011 || exit 1
[ -n "${TURNS_ONLY:-}" ] && exit 0
python3 scripts/test-server.py --url http://127.0.0.1:8011
