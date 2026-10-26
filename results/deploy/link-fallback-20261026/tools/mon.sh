#!/bin/bash
# Per-node monitor during a run: vmstat every second, PSI (io, memory, cpu) every 0.2 s, into DIR.
d=$1; mkdir -p "$d"; cd "$d" || exit 1
vmstat -t 1 > vmstat.log 2>&1 &
v=$!
trap 'kill $v; exit 0' TERM INT
while :; do
    echo "$(date +%T.%N | cut -c1-12) io $(head -1 /proc/pressure/io) mem $(head -1 /proc/pressure/memory) cpu $(head -1 /proc/pressure/cpu)"
    sleep 0.2
done > psi.log
