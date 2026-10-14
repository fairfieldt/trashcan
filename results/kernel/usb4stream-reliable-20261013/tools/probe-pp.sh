#!/bin/bash
# Raw-device ping-pong on incite-indite (tbcheck.py pp, CPU only) inside window 1's lease, while that link is idle:
# starts when the window's steps.log reaches fault 6b and stops before 6g (which uses the link again). Single frames
# there and back; a side that has waited 20 ms since its last write sends a filler frame, so the fillers per round
# count the extra frames it takes until the other side sees a lone frame; `follow` sends that many fillers at once.
# usage: probe-pp.sh WINDOW_DIR
set -uo pipefail
w=$1
F=trashcan-exp/flashnext
until grep -q '^\[[0-9:]*\] 6b ' "$w/steps.log"; do
    grep -q '^\[[0-9:]*\] 7\. \|STOP\|not every link' "$w/steps.log" && { echo "window past the faults: no probe"; exit 1; }
    sleep 1
done
DA=$(ssh incite "~/$F/stream-ops.sh dev indite")
DB=$(ssh indite "~/$F/stream-ops.sh dev incite")
echo "[$(date +%T)] probe on incite $DA, indite $DB" | tee -a "$w/probe.log"
run=10
for v in "incite 112 0 40" "indite 112 0 40" "incite 112 1 200" "incite 112 3 200" "incite 4096 0 40" "indite 112 1 200"; do
    set -- $v
    c=$1 size=$2 follow=$3 n=$4
    if [ "$c" = incite ]; then s=indite dc=$DA ds=$DB; else s=incite dc=$DB ds=$DA; fi
    run=$((run + 1))
    if grep -q '^\[[0-9:]*\] 6g ' "$w/steps.log"; then echo "[$(date +%T)] 6g started: probe stopped" | tee -a "$w/probe.log"; break; fi
    ssh "$s" "timeout -k 5 40 python3 ~/$F/stream-tbcheck.py pp $ds server $n $size $follow $run" >> "$w/probe.jsonl" 2>> "$w/probe.err" &
    p=$!
    sleep 0.3
    ssh "$c" "timeout -k 5 40 python3 ~/$F/stream-tbcheck.py pp $dc client $n $size $follow $run" >> "$w/probe.jsonl" 2>> "$w/probe.err"
    wait $p
    echo "[$(date +%T)] run $run: client $c, $size B, follow $follow, $n rounds" | tee -a "$w/probe.log"
done
echo "[$(date +%T)] probe done" | tee -a "$w/probe.log"
