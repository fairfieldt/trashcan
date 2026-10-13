#!/bin/bash
# The server test with an idle gap longer than amdgpu's job timeout: start the Responses server as the chain's driver,
# run q4-turns.py, wait 15 s without requests, run q4-turns.py again, then scripts/test-server.py; stop the server.
# Run from the workspace (~/trashcan-exp/flashnext, which holds scripts/) with the nodes already listening.
# usage: BIN=./trashcan-host-a host-serve-idle.sh MODEL LOG [serve ARGS...]
set -uo pipefail
model=$1 log=$2
shift 2
"$BIN" serve "$model" --port 8011 "$@" 2> "$log" &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:8011/health > /dev/null; do
    kill -0 $server 2>/dev/null || { tail "$log"; exit 1; }
    sleep 1
done
echo "server up after $SECONDS s" >&2
python3 scripts/q4-turns.py --url http://127.0.0.1:8011 || exit 1
# The clocks drop back once the chain has been idle for a second (engine/src/clocks.rs).
echo "window serve-idle start $(date -d @$((EPOCHSECONDS + 2)) +%T)" >&2
sleep 15
echo "window serve-idle end $(date +%T)" >&2
python3 scripts/q4-turns.py --url http://127.0.0.1:8011 || exit 1
python3 scripts/test-server.py --url http://127.0.0.1:8011
