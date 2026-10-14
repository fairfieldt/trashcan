#!/bin/bash
# r20 (8ea4003: the decode attention's and the decode block scores' loops kept, small code) against r19, single-Mac
# jobs before long-s4: alight, dumps of the last chunk (1-token chunks at 8193 and 30081, 4 tokens at 8196; Q8 K/V at
# 8196), every tensor identical expected; inflight, stage-bench profiles of r20 at 128, 1024, 8320 and 30080 and of
# r19 at 128 and 8320. Logs to s4b-1mac-MACHINE.log here.
L=$HOME/dev/claude-work/flashnext-20261012/long
A=trashcan-long-r19 B=trashcan-long-r20
j() { local m=$1 est=$2; shift 2; "$L/long-1mac.sh" "$m" "long-s4-r20-$m" "$est" -- "$*" > "$L/s4b-1mac-$m.log" 2>&1 & }
j alight 7 "./long-equiv.sh $A $B s4t 8193 8196 30081; KV=q8 ./long-equiv.sh $A $B s4tq8 8196"
j inflight 8 "./long-sb.sh $B s4t20 128 1024 8320 30080; ./long-sb.sh $A s4t19 128 8320"
wait
echo "s4b-1mac done"
