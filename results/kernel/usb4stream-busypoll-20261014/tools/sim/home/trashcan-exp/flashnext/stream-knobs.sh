#!/bin/bash
# Stub of stream-knobs.sh on a simulated Mac: remembers the switches in the home and prints them.
peer=$1
shift
f="$HOME/knobs-$peer"
for kv in "$@"; do echo "$kv" >> "$f"; done
tail -5 "$f" 2> /dev/null | tr '\n' ' '
echo
