#!/bin/bash
# r20 on the Macs that profiled main 4307169 (same-Mac pairs for the README): invite at 128, 1024, 2176 and 16512;
# enlight, one QSA layer at 204,800 positions with a Q8 K/V cache. Logs to s4c-1mac-MACHINE.log here.
L=$HOME/dev/claude-work/flashnext-20261012/long
j() { local m=$1 est=$2; shift 2; "$L/long-1mac.sh" "$m" "long-s4-r20b-$m" "$est" -- "$*" > "$L/s4c-1mac-$m.log" 2>&1 & }
j invite 5 "./long-sb.sh trashcan-long-r20 s4u20 128 1024 2176 16512"
j enlight 4 "./long-200k.sh trashcan-long-r20 long-sb-s4u20-200k"
wait
echo "s4c-1mac done"
